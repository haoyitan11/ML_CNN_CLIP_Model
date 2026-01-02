import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse, os
import torch

from src.medvqa_clip.utils.seed import set_seed
from src.medvqa_clip.utils.runtime import get_device, prepare_loaders
from src.medvqa_clip.utils.io import read_json
from src.medvqa_clip.utils.logging import log_line
from src.medvqa_clip.engine.train import train
from src.medvqa_clip.models.multitask_clip import MultiTaskCLIP

def main():
    #Parse command-line arguments
    ap = argparse.ArgumentParser()
    
    #dataset + paths
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json")
    ap.add_argument("--exp_dir", type=str, default="outputs/exp_clip")

    #CLIP backbone + text token max length
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--max_len", type=int, default=64)

    #Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    #Optional: freeze CLIP vision encoder/text encoder
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    
    args = ap.parse_args()

    #set random seed
    set_seed(args.seed)
    
    #select device 
    device = get_device()

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, "checkpoints"), exist_ok=True)

    #load vocab bundle JSON
    vocabs = read_json(args.vocab_bundle)

    #prepare train and validation loaders
    train_loader, val_loader = prepare_loaders(
        dataset_name=args.dataset_name,
        vocabs=vocabs,
        clip_name=args.clip_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    #build multitaskCLIP model
    model = MultiTaskCLIP(
        vocabs=vocabs,
        clip_name=args.clip_name,
        hidden_dim=512,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
    ).to(device)

    #optimizer: adamW over trainable parameters only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    #log summary
    log_line(args.exp_dir, f"Device={device} | Tasks={len(vocabs)} | TrainN={len(train_loader.dataset)} | ValN={len(val_loader.dataset)}")

    #train loop
    best_path = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        exp_dir=args.exp_dir,
        logger=log_line,
        config=vars(args),
    )
    
    #log message
    log_line(args.exp_dir, f"Training done. Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
