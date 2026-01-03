import os
import sys
import argparse
import torch
import gradio as gr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medvqa_clip.utils.runtime import get_device
from medvqa_clip.utils.io import read_json
from medvqa_clip.models.multitask_clip import MultiTaskCLIP
from medvqa_clip.models.hybrid_answerer import HybridAnswerer


def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str, device: str) -> bool:
    #Try to load checkpoint weights safely
    #Return true if success
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


def main():
    #Parse command-line args
    ap = argparse.ArgumentParser()
    
    #Files 
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json",
                    help="Path to vocabs.json built by scripts/build_vocab.py")
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_clip/checkpoints/best.pt",
                    help="Path to trained checkpoint")
    
    #Model backbone
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32",
                    help="CLIP backbone name (HuggingFace)")
    
    #Inference controls
    ap.add_argument("--topk", type=int, default=5, help="Top-k answers to display")
    ap.add_argument("--use_xrv", action="store_true",
                    help="Use torchxrayvision (if installed) for chest findings")
    
    #RunTime/Gradio setting
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available")
    ap.add_argument("--share", action="store_true",
                    help="Gradio share link")
    ap.add_argument("--server_port", type=int, default=7860,
                    help="Gradio server port")
    
    args = ap.parse_args()

    #choose device
    device = "cpu" if args.cpu else get_device()
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(args.vocab_bundle):
        raise FileNotFoundError(
            f"Missing vocab bundle: {args.vocab_bundle}\n"
            f"Run: python scripts/build_vocab.py --out_path outputs/vocabs.json"
        )

    vocabs = read_json(args.vocab_bundle)

    #Build MultitaskCLIP model
    model = MultiTaskCLIP(
        vocabs=vocabs,
        clip_name=args.clip_name,
        hidden_dim=512,
        freeze_vision=False,
        freeze_text=False,
    ).to(device)


    #load checkpoint
    _loaded = load_checkpoint_safely(model, args.ckpt_path, device)
    model.eval()

    #Build HybridAnswerer
    answerer = HybridAnswerer(
        model=model,
        vocabs=vocabs,
        clip_name=args.clip_name,
        device=device,
        use_xrv=args.use_xrv,
    )

    #Gradio inference function
    def answer(image, question):
        #Validate inputs
        if image is None or not question or not question.strip():
            return "Please upload an image and type a question.", ""

        #Predict using hybrid router
        res = answerer.predict(image, question, topk=args.topk)

        #Format top-k prediction
        topk_str = "\n".join([f"{a}: {p:.4f}" for a, p in res.topk])

        #Show metadata so you can debug routing
        meta = f"Used: {res.used}"

        return f"{res.answer}\n\n{meta}", topk_str

    #Gradio UI
    demo = gr.Interface(
        fn=answer,
        inputs=[
            gr.Image(type="pil", label="Upload Medical Image"),
            gr.Textbox(label="Question", placeholder="e.g., Is there pneumonia?")
        ],
        outputs=[
            gr.Textbox(label="Predicted Answer"),
            gr.Textbox(label=f"Top-{args.topk} Answers (probability)")
        ],
        title="Med-VQA CLIP Hybrid"
    )

    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
