import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
import torch
from PIL import Image

from src.medvqa_clip.utils.io import read_json
from src.medvqa_clip.utils.runtime import get_device
from src.medvqa_clip.models.multitask_clip import MultiTaskCLIP
from src.medvqa_clip.models.hybrid_answerer import HybridAnswerer

def main():
    #command-line arguments
    ap = argparse.ArgumentParser()
    
    #required: image path and question text
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    
    #vocab file + checkpoint file and question text
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json")
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_clip/checkpoints/best.pt")
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--topk", type=int, default=5)
    
    #enable touchxrayvision if available
    ap.add_argument("--use_xrv", action="store_true")
    args = ap.parse_args()

    #select device
    device = get_device()
    
    #load vocab bundle
    vocabs = read_json(args.vocab_bundle)

    #build CLIP multitask model
    model = MultiTaskCLIP(vocabs=vocabs, clip_name=args.clip_name, hidden_dim=512, freeze_vision=False, freeze_text=False).to(device)
    
    #load checkpoint weights
    try:
        state = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        pass
    
    #set model to evaluation mode
    model.eval()

    #create hybrid answerer (route question, normally touchxrayvision)
    answerer = HybridAnswerer(model=model, vocabs=vocabs, clip_name=args.clip_name, device=device, use_xrv=args.use_xrv)

    #load image from disk
    img = Image.open(args.image)
    
    #run prediction
    res = answerer.predict(img, args.question, topk=args.topk)
    
    #display result
    print("Answer:", res.answer)
    print("Used:", res.used)
    for a, p in res.topk:
        print(f"  {a}: {p:.4f}")

if __name__ == "__main__":
    main()
