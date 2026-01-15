import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse, os, json
import torch

from medvqa_clip.utils.io import read_json, write_json
from medvqa_clip.utils.runtime import get_device, prepare_test_loader
from medvqa_clip.models.multitask_clip import MultiTaskCLIP
from medvqa_clip.engine.evaluate import evaluate

def main():
    #parse command-line argument
    ap = argparse.ArgumentParser()
    
    #dataset + paths
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json")
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_clip/checkpoints/best.pt")
    ap.add_argument("--exp_dir", type=str, default="outputs/exp_clip")

    #CLIP preprocessing / dataloader settings
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    
    args = ap.parse_args()

    #select device
    device = get_device()
    
    #load vocab bundle
    vocabs = read_json(args.vocab_bundle)

    #prepare test dataloader
    loader = prepare_test_loader(
        dataset_name=args.dataset_name,
        vocabs=vocabs,
        clip_name=args.clip_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    #build model
    model = MultiTaskCLIP(vocabs=vocabs, clip_name=args.clip_name, hidden_dim=512, freeze_vision=False, freeze_text=False).to(device)
    
    #load trained checkpoint weights
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    #evaluate model on test set
    metrics = evaluate(model, loader, device, show_progress=True)
    
    #save metrics JSON 
    os.makedirs(args.exp_dir, exist_ok=True)
    write_json(metrics, os.path.join(args.exp_dir, "metrics_test.json"))
    
    #display result
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.exp_dir}/metrics_test.json")

if __name__ == "__main__":
    main()
