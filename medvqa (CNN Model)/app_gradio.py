# --- auto-path: allow running without installing the package ---
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


import argparse
import json
from collections import Counter

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from medvqa.utils.io import read_json, write_json
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.utils.question_router import route_question
from medvqa.utils.runtime import get_device, prepare_test_loader
from medvqa.utils.text import normalize_answer


# Answer-space constraints (same as your current app)
MODALITY_ANS = {"x-ray", "chest x-ray", "ct", "mri", "ultrasound", "pet", "angiography", "fluoroscopy", "flair", "t2", "adc", "dwi"}
PLANE_ANS    = {"axial", "coronal", "sagittal"}
VIEW_ANS     = {"pa", "ap", "lateral"}
ORGAN_ANS    = {"lungs", "lung", "chest", "brain", "abdomen", "kidney", "kidneys", "liver", "heart"}
LATERAL_ANS  = {"right", "left", "bilateral"}


def mask_logits(logits: torch.Tensor, itos: list, allowed_set: set) -> torch.Tensor:
    keep = [i for i, a in enumerate(itos) if a in allowed_set]
    if not keep:
        return logits
    masked = torch.full_like(logits, float("-inf"))
    masked[:, keep] = logits[:, keep]
    return masked


# Per-sample EM + token F1 (needs GT string)
def exact_match(pred: str, gt: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gt))

def token_f1(pred: str, gt: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gt).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0

    pc = Counter(p)
    gc = Counter(g)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(p)
    recall = num_same / len(g)
    return (2 * precision * recall) / (precision + recall)


# Dataset-level macro-F1 for classification (open/closed)
def _macro_f1_from_counts(tp: torch.Tensor, pred_cnt: torch.Tensor, true_cnt: torch.Tensor) -> float:
    tp = tp.float()
    fp = pred_cnt.float() - tp
    fn = true_cnt.float() - tp
    denom = (2 * tp + fp + fn)
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))

    active = (pred_cnt + true_cnt) > 0
    return float(f1[active].mean().item()) if active.any() else 0.0


@torch.no_grad()
def evaluate_cnn_em_f1(model, loader, device: torch.device, num_open_classes: int):
    model.eval()

    # Open-head counts
    open_n = 0
    open_correct = 0
    open_tp = torch.zeros(num_open_classes, dtype=torch.long)
    open_pred_cnt = torch.zeros(num_open_classes, dtype=torch.long)
    open_true_cnt = torch.zeros(num_open_classes, dtype=torch.long)

    # Closed-head counts (2 classes: 0=no, 1=yes)
    closed_n = 0
    closed_correct = 0
    closed_tp = torch.zeros(2, dtype=torch.long)
    closed_pred_cnt = torch.zeros(2, dtype=torch.long)
    closed_true_cnt = torch.zeros(2, dtype=torch.long)

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        open_y = batch["open_label"].to(device)
        closed_y = batch["closed_label"].to(device)
        is_closed_q = batch["is_closed_q"].to(device).bool()

        logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

        # OPEN questions (defined by your dataset field; matches your current training)
        mask_o = ~is_closed_q
        if mask_o.any():
            lo = logits_open[mask_o]
            yo = open_y[mask_o]
            pred_o = lo.argmax(dim=-1)

            open_n += int(yo.numel())
            open_correct += int((pred_o == yo).sum().item())

            # update macro-f1 counts
            open_pred_cnt += torch.bincount(pred_o.cpu(), minlength=num_open_classes)
            open_true_cnt += torch.bincount(yo.cpu(), minlength=num_open_classes)
            match = (pred_o == yo)
            if match.any():
                open_tp += torch.bincount(yo[match].cpu(), minlength=num_open_classes)

        # CLOSED yes/no questions (only where closed_y is valid)
        mask_c = is_closed_q & (closed_y != -1)
        if mask_c.any():
            lc = logits_closed[mask_c]
            yc = closed_y[mask_c]
            pred_c = lc.argmax(dim=-1)

            closed_n += int(yc.numel())
            closed_correct += int((pred_c == yc).sum().item())

            closed_pred_cnt += torch.bincount(pred_c.cpu(), minlength=2)
            closed_true_cnt += torch.bincount(yc.cpu(), minlength=2)
            match = (pred_c == yc)
            if match.any():
                closed_tp += torch.bincount(yc[match].cpu(), minlength=2)

    open_em = open_correct / max(1, open_n)
    closed_em = closed_correct / max(1, closed_n)

    open_f1 = _macro_f1_from_counts(open_tp, open_pred_cnt, open_true_cnt)
    closed_f1 = _macro_f1_from_counts(closed_tp, closed_pred_cnt, closed_true_cnt)

    # overall weighted
    total_n = open_n + closed_n
    overall_em = (open_correct + closed_correct) / max(1, total_n)
    overall_f1 = (open_f1 * open_n + closed_f1 * closed_n) / max(1, total_n)

    return {
        "overall": {"em": overall_em, "f1_macro": overall_f1, "n": total_n},
        "open": {"em": open_em, "f1_macro": open_f1, "n": open_n},
        "closed": {"em": closed_em, "f1_macro": closed_f1, "n": closed_n},
    }


# Build model once
def build_model(args, device: torch.device):
    vocab = read_json(args.vocab_path)
    itos = vocab["itos"]

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    model = CNNVQAModel(
        num_answers=len(itos),
        text_model_name=args.text_model_name,
        freeze_resnet=args.freeze_resnet,
        freeze_text=args.freeze_text,
        attn_dim=args.attn_dim,
    ).to(device)

    if not os.path.exists(args.ckpt_path):
        print(f"[WARN] Checkpoint not found: {args.ckpt_path} (predictions may be random)")
    else:
        state = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model, vocab, itos, tokenizer, tfm


# APP mode 
def run_app(args):
    device = get_device()
    model, vocab, itos, tokenizer, tfm = build_model(args, device)

    @torch.no_grad()
    def predict(image: Image.Image, question: str, ground_truth: str):
        if image is None or not question or not question.strip():
            return "Please upload an image and type a question.", "", "", "", ""

        q = question.strip()
        img = image.convert("RGB")
        x_img = tfm(img).unsqueeze(0).to(device)

        tok = tokenizer([q], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        logits_open, logits_closed = model(image=x_img, input_ids=input_ids, attention_mask=attention_mask)

        r = route_question(q)

        #CLOSED yes/no
        if r.qtype == "closed_yesno":
            probs = torch.softmax(logits_closed, dim=-1).squeeze(0)
            p_no = float(probs[0].item())
            p_yes = float(probs[1].item())
            pred = "yes" if p_yes >= p_no else "no"
            top_str = "\n".join([f"no:  {p_no:.6f}", f"yes: {p_yes:.6f}"])
            debug = f"Detected: {r.qtype} | subtype={r.subtype} | {r.debug}"

        #OPEN vocab
        else:
            logits = logits_open
            if r.subtype == "modality":
                logits = mask_logits(logits, itos, MODALITY_ANS)
            elif r.subtype == "plane":
                logits = mask_logits(logits, itos, PLANE_ANS)
            elif r.subtype == "view":
                logits = mask_logits(logits, itos, VIEW_ANS)
            elif r.subtype == "organ":
                logits = mask_logits(logits, itos, ORGAN_ANS)
            elif r.subtype == "laterality":
                logits = mask_logits(logits, itos, LATERAL_ANS)

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            k = min(args.topk, probs.numel())
            vals, idxs = probs.topk(k)

            pred = itos[idxs[0].item()]
            top_str = "\n".join([f"{itos[i.item()]}: {v.item():.6f}" for v, i in zip(vals, idxs)])
            debug = f"Detected: {r.qtype} | subtype={r.subtype} | {r.debug}"

        # Optional per-sample EM/F1 if user provides ground truth
        gt = (ground_truth or "").strip()
        if gt:
            em = exact_match(pred, gt)
            f1 = token_f1(pred, gt)
            em_str = str(em)
            f1_str = f"{f1:.4f}"
        else:
            em_str = ""
            f1_str = ""

        return pred, top_str, debug, em_str, f1_str

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Medical Image"),
            gr.Textbox(label="Question"),
            gr.Textbox(label="Ground Truth (optional, to compute EM/F1)", placeholder="Type the correct answer if you have it"),
        ],
        outputs=[
            gr.Textbox(label="Predicted Answer"),
            gr.Textbox(label=f"Top-{args.topk} Answers (probability)"),
            gr.Textbox(label="Debug"),
            gr.Textbox(label="EM (per-sample)"),
            gr.Textbox(label="Token F1 (per-sample)"),
        ],
        title="Med-VQA CNN Baseline (Inference + optional EM/F1)",
    )

    demo.launch(share=args.share, server_port=args.server_port)



# EVAL mode (dataset-level EM + macro-F1)
def run_eval(args):
    device = get_device()
    model, vocab, itos, tokenizer, tfm = build_model(args, device)

    loader = prepare_test_loader(
        dataset_name=args.dataset_name,
        vocab=vocab,
        text_model_name=args.text_model_name,
        img_size=args.img_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    metrics = evaluate_cnn_em_f1(
        model=model,
        loader=loader,
        device=device,
        num_open_classes=len(itos),
    )

    os.makedirs(args.exp_dir, exist_ok=True)
    out_path = os.path.join(args.exp_dir, args.out_metrics_name)
    write_json(metrics, out_path)

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {out_path}")


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["app", "eval"], default="app")

    # paths
    ap.add_argument("--vocab_path", default=os.environ.get("MEDVQA_VOCAB", "outputs/answer_vocab.json"))
    ap.add_argument("--ckpt_path", default=os.environ.get("MEDVQA_CKPT", "outputs/exp_cnn/checkpoints/best.pt"))
    ap.add_argument("--exp_dir", default="outputs/exp_cnn")
    ap.add_argument("--out_metrics_name", default="metrics_test_emf1.json")

    # model/config
    ap.add_argument("--text_model_name", default=os.environ.get("MEDVQA_TEXT_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"))
    ap.add_argument("--img_size", type=int, default=int(os.environ.get("MEDVQA_IMG_SIZE", "224")))
    ap.add_argument("--max_len", type=int, default=int(os.environ.get("MEDVQA_MAX_LEN", "64")))
    ap.add_argument("--attn_dim", type=int, default=int(os.environ.get("MEDVQA_ATTN_DIM", "512")))
    ap.add_argument("--freeze_resnet", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    # app
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server_port", type=int, default=7860)

    # eval
    ap.add_argument("--dataset_name", default="flaviagiammarino/vqa-rad")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    if args.mode == "app":
        run_app(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
