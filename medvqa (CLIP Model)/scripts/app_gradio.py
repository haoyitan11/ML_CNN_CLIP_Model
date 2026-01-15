import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import gradio as gr

# Path setup: allow importing from ../src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medvqa_clip.utils.runtime import get_device, prepare_test_loader
from medvqa_clip.utils.io import read_json, write_json
from medvqa_clip.models.multitask_clip import MultiTaskCLIP
from medvqa_clip.models.hybrid_answerer import HybridAnswerer


# Helpers
def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str, device: str) -> bool:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        print("[WARN] The app will still run, but predictions may be random.")
        return False

    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {ckpt_path}")
        print(f"Reason: {e}")
        print("[WARN] The app will still run, but predictions may be random.")
        return False


def build_model_and_answerer(args, device: str):
    if not os.path.exists(args.vocab_bundle):
        raise FileNotFoundError(
            f"Missing vocab bundle: {args.vocab_bundle}\n"
            f"Run: python scripts/build_vocab.py --out_path outputs/vocabs.json"
        )

    vocabs = read_json(args.vocab_bundle)

    model = MultiTaskCLIP(
        vocabs=vocabs,
        clip_name=args.clip_name,
        hidden_dim=512,
        freeze_vision=False,
        freeze_text=False,
    ).to(device)

    load_checkpoint_safely(model, args.ckpt_path, device)
    model.eval()

    answerer = HybridAnswerer(
        model=model,
        vocabs=vocabs,
        clip_name=args.clip_name,
        device=device,
        use_xrv=args.use_xrv,
    )

    return model, vocabs, answerer


# Metrics (EM + macro-F1 per task; weighted overall F1)
def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _macro_f1_from_counts(tp, pred_cnt, true_cnt) -> float:
    # tp, pred_cnt, true_cnt are torch.LongTensor
    tp = tp.float()
    fp = pred_cnt.float() - tp
    fn = true_cnt.float() - tp
    denom = (2 * tp + fp + fn)
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))

    active = (pred_cnt + true_cnt) > 0
    return float(f1[active].mean().item()) if active.any() else 0.0


@torch.no_grad()
def evaluate_clip_emf1(model, loader, device: str, vocabs: dict):
    model.eval()

    total = 0
    correct = 0

    per_task = {}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        tasks = batch["tasks"]
        y = batch["labels"].to(device)

        feats = model.forward_features(pixel_values, input_ids, attn)

        # group indices by task
        task_to_idx = defaultdict(list)
        for i, t in enumerate(tasks):
            task_to_idx[t].append(i)

        pred = torch.empty_like(y)

        for t, idxs in task_to_idx.items():
            idx = torch.tensor(idxs, device=device)
            logits = model.forward_task(t, feats.index_select(0, idx))
            pred.index_copy_(0, idx, logits.argmax(dim=-1))

            if t not in per_task:
                num_classes = len(vocabs[t]["itos"])
                per_task[t] = {
                    "n": 0,
                    "correct": 0,
                    "tp": torch.zeros(num_classes, dtype=torch.long),
                    "pred_cnt": torch.zeros(num_classes, dtype=torch.long),
                    "true_cnt": torch.zeros(num_classes, dtype=torch.long),
                }

            yt = y.index_select(0, idx).detach()
            pt = pred.index_select(0, idx).detach()

            match = (yt == pt)
            per_task[t]["n"] += int(yt.numel())
            per_task[t]["correct"] += int(match.sum().item())

            num_classes = per_task[t]["tp"].numel()
            tp_add = torch.bincount(yt[match], minlength=num_classes).cpu()
            pred_add = torch.bincount(pt, minlength=num_classes).cpu()
            true_add = torch.bincount(yt, minlength=num_classes).cpu()

            per_task[t]["tp"] += tp_add
            per_task[t]["pred_cnt"] += pred_add
            per_task[t]["true_cnt"] += true_add

        ok = (pred == y)
        total += int(y.numel())
        correct += int(ok.sum().item())

    # build per-task report + weighted overall f1
    per_task_out = {}
    weighted_f1_sum = 0.0

    for t, st in per_task.items():
        em = _safe_div(st["correct"], st["n"])
        f1m = _macro_f1_from_counts(st["tp"], st["pred_cnt"], st["true_cnt"])
        per_task_out[t] = {"em": em, "f1_macro": f1m, "n": st["n"]}
        weighted_f1_sum += f1m * st["n"]

    overall_em = _safe_div(correct, total)
    overall_f1 = _safe_div(weighted_f1_sum, total)

    return {
        "overall": {"em": overall_em, "f1": overall_f1, "n": total},
        "per_task": per_task_out,
    }


# Mode: Gradio App
def run_app(args, device: str):
    _, _, answerer = build_model_and_answerer(args, device)

    def predict(image, question):
        if image is None or not question or not question.strip():
            return "Please upload an image and type a question.", ""

        res = answerer.predict(image, question, topk=args.topk)

        topk_str = "\n".join([f"{a}: {p:.4f}" for a, p in res.topk])
        meta = f"Used head/task: {res.used}"
        return f"{res.answer}\n\n{meta}", topk_str

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Medical Image"),
            gr.Textbox(label="Question", placeholder="e.g., Is there pneumothorax?")
        ],
        outputs=[
            gr.Textbox(label="Predicted Answer"),
            gr.Textbox(label=f"Top-{args.topk} Answers (probability)", lines=8, max_lines=20)
        ],
        title="Med-VQA CLIP",
        description="CLIP multi-task Med-VQA model."
    )

    demo.launch(share=args.share, server_port=args.server_port)


# Mode: Evaluation (EM/F1)
def run_eval(args, device: str):
    model, vocabs, _ = build_model_and_answerer(args, device)

    loader = prepare_test_loader(
        dataset_name=args.dataset_name,
        vocabs=vocabs,
        clip_name=args.clip_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,  # if your loader supports it
    )

    metrics = evaluate_clip_emf1(model, loader, device=device, vocabs=vocabs)

    os.makedirs(args.exp_dir, exist_ok=True)
    out_path = os.path.join(args.exp_dir, args.out_metrics_name)

    if args.save_json:
        write_json(metrics, out_path)
        print(f"[OK] Saved metrics to: {out_path}")

    print(json.dumps(metrics, indent=2))


# Main
def main():
    ap = argparse.ArgumentParser(description="CLIP multitask Med-VQA: Gradio demo + EM/F1 eval")

    ap.add_argument("--mode", type=str, choices=["app", "eval"], default="app",
                    help="app=Gradio demo; eval=compute EM+F1 on a labeled split")

    # shared
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json")
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_clip/checkpoints/best.pt")
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--use_xrv", action="store_true")

    ap.add_argument("--cpu", action="store_true")

    # app-only
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server_port", type=int, default=7860)

    # eval-only
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--split", type=str, default="test", help="dataset split to evaluate (if supported by loader)")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--exp_dir", type=str, default="outputs/exp_clip")
    ap.add_argument("--out_metrics_name", type=str, default="metrics_test_emf1.json")
    ap.add_argument("--save_json", action="store_true", help="save metrics JSON into exp_dir")

    args = ap.parse_args()

    device = "cpu" if args.cpu else get_device()
    print(f"[INFO] Using device: {device}")

    if args.mode == "app":
        run_app(args, device)
    else:
        run_eval(args, device)


if __name__ == "__main__":
    main()
