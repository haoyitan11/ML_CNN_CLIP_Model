import os
import sys
import json
import argparse
from typing import Tuple, Optional

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


# 0) JSON helper (local, avoids mixing CNN/CLIP read_json)
def read_json_local(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1) PATH SETUP: add CNN/src to sys.path
ROOT_CNN = os.path.abspath(os.path.dirname(__file__))
SRC_CNN = os.path.join(ROOT_CNN, "src")
if SRC_CNN not in sys.path:
    sys.path.insert(0, SRC_CNN)


# 2) CNN imports (from CNN project)
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.utils.io import read_json as read_json_cnn
from medvqa.utils.question_router import route_question
from medvqa.utils.runtime import get_device


# 3) CNN builder
def build_cnn(args, device: torch.device):
    vocab = read_json_cnn(args.cnn_vocab_path)
    itos = vocab["itos"]

    tokenizer = AutoTokenizer.from_pretrained(args.cnn_text_model_name)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    model = CNNVQAModel(
        num_answers=len(itos),
        text_model_name=args.cnn_text_model_name,
        freeze_resnet=args.freeze_resnet,
        freeze_text=args.freeze_text,
        attn_dim=args.attn_dim,
    ).to(device)

    if os.path.exists(args.cnn_ckpt_path):
        state = torch.load(args.cnn_ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[OK] Loaded CNN checkpoint: {args.cnn_ckpt_path}")
    else:
        print(f"[WARN] CNN checkpoint not found: {args.cnn_ckpt_path}")

    model.eval()
    return model, itos, tokenizer, tfm


# 4) CLIP helpers: resolve paths + build
def resolve_path(*parts) -> str:
    return os.path.normpath(os.path.abspath(os.path.join(*parts)))


def resolve_clip_vocabs_path(clip_root: str, user_path: str) -> str:

    candidates = [
        user_path,
        resolve_path(clip_root, "outputs", "vocabs.json"),
        resolve_path(clip_root, "outputs", "exp_clip", "vocabs.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]  # for error message


def build_clip(args, device: torch.device):
    clip_root = resolve_path(args.clip_project_root)
    clip_src = resolve_path(clip_root, "src")
    if clip_src not in sys.path:
        sys.path.insert(0, clip_src)

    # imports AFTER sys.path update
    from medvqa_clip.models.multitask_clip import MultiTaskCLIP
    from medvqa_clip.models.hybrid_answerer import HybridAnswerer

    vocabs_path = resolve_clip_vocabs_path(clip_root, resolve_path(args.clip_vocabs_path))
    if not os.path.exists(vocabs_path):
        raise FileNotFoundError(
            "CLIP vocabs.json not found.\n"
            f"Tried: {vocabs_path}\n"
            f"Also expected one of:\n"
            f"  - {resolve_path(clip_root, 'outputs', 'vocabs.json')}\n"
            f"  - {resolve_path(clip_root, 'outputs', 'exp_clip', 'vocabs.json')}\n"
        )

    vocabs = read_json_local(vocabs_path)

    model = MultiTaskCLIP(
        vocabs=vocabs,
        clip_name=args.clip_backbone,
        hidden_dim=args.clip_hidden_dim,
        freeze_vision=args.freeze_clip_vision,
        freeze_text=args.freeze_clip_text,
    ).to(device)

    ckpt_path = resolve_path(args.clip_ckpt_path)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[OK] Loaded CLIP checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] CLIP checkpoint not found: {ckpt_path}")

    model.eval()

    answerer = HybridAnswerer(
        model=model,
        vocabs=vocabs,
        clip_name=args.clip_backbone,
        device=str(device),  # HybridAnswerer expects a string
        use_xrv=args.use_xrv,
        max_len=args.max_len,
        xrv_threshold=args.xrv_threshold,
    )

    return answerer


# 5) Inference: CNN / CLIP (answer + confidence)
@torch.no_grad()
def infer_cnn(cnn_bundle, device: torch.device, image: Image.Image, question: str) -> Tuple[str, float]:
    model, itos, tokenizer, tfm = cnn_bundle

    img = image.convert("RGB")
    x_img = tfm(img).unsqueeze(0).to(device)

    tok = tokenizer([question], padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    logits_open, logits_closed = model(image=x_img, input_ids=input_ids, attention_mask=attention_mask)
    r = route_question(question)

    # closed yes/no head
    if r.qtype == "closed_yesno":
        probs = torch.softmax(logits_closed, dim=-1).squeeze(0)
        p_no = float(probs[0].item())
        p_yes = float(probs[1].item())
        pred = "yes" if p_yes >= p_no else "no"
        conf = max(p_yes, p_no)
        return pred, conf

    # open head
    probs = torch.softmax(logits_open, dim=-1).squeeze(0)
    idx = int(probs.argmax().item())
    pred = itos[idx]
    conf = float(probs[idx].item())
    return pred, conf


@torch.no_grad()
def infer_clip(clip_answerer, image: Image.Image, question: str) -> Tuple[str, float]:
    res = clip_answerer.predict(image=image, question=question, topk=1)
    pred = res.answer
    conf = float(res.topk[0][1]) if (res.topk and len(res.topk) > 0) else 0.0
    return pred, conf


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"



# 6) Gradio App
def run_app(args):
    device = get_device()
    print(f"[INFO] Device: {device}")

    cnn_bundle = None
    clip_answerer = None

    # Load at startup based on startup_model
    if args.startup_model in ("cnn", "both"):
        cnn_bundle = build_cnn(args, device)

    if args.startup_model in ("clip", "both"):
        clip_answerer = build_clip(args, device)

    def run(image, question, run_mode):
        if image is None or not question or not question.strip():
            return "", "", "", ""

        q = question.strip()

        cnn_ans, cnn_conf = "", ""
        clip_ans, clip_conf = "", ""

        if run_mode in ("cnn", "both"):
            if cnn_bundle is None:
                cnn_ans, cnn_conf = "[CNN not loaded]", ""
            else:
                ans, conf = infer_cnn(cnn_bundle, device, image, q)
                cnn_ans, cnn_conf = ans, fmt_pct(conf)

        if run_mode in ("clip", "both"):
            if clip_answerer is None:
                clip_ans, clip_conf = "[CLIP not loaded]", ""
            else:
                ans, conf = infer_clip(clip_answerer, image, q)
                clip_ans, clip_conf = ans, fmt_pct(conf)

        return cnn_ans, cnn_conf, clip_ans, clip_conf

    with gr.Blocks() as demo:
        gr.Markdown("# Med-VQA Demo (CNN / CLIP / BOTH)")

        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Medical Image")
            with gr.Column():
                q_in = gr.Textbox(label="Question")
                mode_in = gr.Radio(
                    choices=["cnn", "clip", "both"],
                    value=args.startup_model,
                    label="Run Mode"
                )
                run_btn = gr.Button("Run")

        gr.Markdown("## CNN Result")
        cnn_ans_out = gr.Textbox(label="CNN Predicted Answer")
        cnn_conf_out = gr.Textbox(label="CNN Confidence (%)")

        gr.Markdown("## CLIP Result")
        clip_ans_out = gr.Textbox(label="CLIP Predicted Answer")
        clip_conf_out = gr.Textbox(label="CLIP Confidence (%)")

        run_btn.click(
            run,
            inputs=[img_in, q_in, mode_in],
            outputs=[cnn_ans_out, cnn_conf_out, clip_ans_out, clip_conf_out]
        )

    demo.launch(share=args.share, server_port=args.server_port)


# 7) CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--startup_model", choices=["cnn", "clip", "both"], default="both")

    #CNN
    ap.add_argument("--cnn_vocab_path", default="outputs/answer_vocab.json")
    ap.add_argument("--cnn_ckpt_path", default=r"outputs\exp_cnn\checkpoints\best.pt")
    ap.add_argument("--cnn_text_model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--attn_dim", type=int, default=512)
    ap.add_argument("--freeze_resnet", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    #CLIP
    ap.add_argument("--clip_project_root", default=r"..\medvqa (CLIP Model)")
    ap.add_argument("--clip_ckpt_path", default=r"..\medvqa (CLIP Model)\outputs\exp_clip\checkpoints\best.pt")
    ap.add_argument("--clip_vocabs_path", default=r"..\medvqa (CLIP Model)\outputs\vocabs.json")
    ap.add_argument("--clip_backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_hidden_dim", type=int, default=512)
    ap.add_argument("--freeze_clip_vision", action="store_true")
    ap.add_argument("--freeze_clip_text", action="store_true")

    #HybridAnswerer extras
    ap.add_argument("--use_xrv", action="store_true")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--xrv_threshold", type=float, default=0.5)

    #App
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server_port", type=int, default=7860)

    args = ap.parse_args()
    run_app(args)


if __name__ == "__main__":
    main()
