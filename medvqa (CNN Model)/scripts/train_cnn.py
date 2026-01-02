from _setup_path import *  # noqa: F401,F403

import os
import argparse
import torch

from medvqa.utils.seed import set_seed
from medvqa.utils.io import read_json
from medvqa.utils.logging import log_line
from medvqa.utils.runtime import get_device, prepare_loaders
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.engine.train import train

def main():
    #parse args
    ap = argparse.ArgumentParser()

    #build loaders, build model, train, save checkpoint
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--vocab_path", type=str, default="outputs/answer_vocab.json")
    ap.add_argument("--exp_dir", type=str, default="outputs/exp_cnn")

    ap.add_argument("--text_model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--freeze_resnet", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--attn_dim", type=int, default=512)

    ap.add_argument("--alpha_closed", type=float, default=1.0)
    ap.add_argument("--balanced_closed_sampler", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    args = ap.parse_args()

    #set seeds 
    set_seed(args.seed)
    device = get_device()

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, "checkpoints"), exist_ok=True)

    #Load open-answer vocabulary
    vocab = read_json(args.vocab_path)

    #Build the training and validation DataLoaders
    train_loader, val_loader, sampler_info = prepare_loaders(
        dataset_name=args.dataset_name,
        vocab=vocab,
        text_model_name=args.text_model_name,
        img_size=args.img_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        balanced_closed_sampler=args.balanced_closed_sampler,
    )

    #Instantiate the CNN VQA model
    model = CNNVQAModel(
        num_answers=len(vocab["itos"]),
        text_model_name=args.text_model_name,
        freeze_resnet=args.freeze_resnet,
        freeze_text=args.freeze_text,
        attn_dim=args.attn_dim,
    ).to(device)

    #Create adamW optimizer over trainable parameters only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    #config setting
    config = {
        "dataset_name": args.dataset_name,
        "vocab_path": args.vocab_path,
        "exp_dir": args.exp_dir,
        "text_model_name": args.text_model_name,
        "img_size": args.img_size,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_resnet": args.freeze_resnet,
        "freeze_text": args.freeze_text,
        "attn_dim": args.attn_dim,
        "alpha_closed": args.alpha_closed,
        "balanced_closed_sampler": args.balanced_closed_sampler,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "val_ratio": args.val_ratio,
        "num_answers": len(vocab["itos"]),
        "device": str(device),
        "sampler_closed_counts": sampler_info,
    }

    log_line(args.exp_dir, f"Device={device} | Open Answers={len(vocab['itos'])} | Two-head + spatial attention + FiLM")
    if sampler_info is not None:
        log_line(args.exp_dir, f"Balanced closed sampler enabled: counts={sampler_info}")

    #run training
    best_path = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        exp_dir=args.exp_dir,
        logger=log_line,
        config=config,
        alpha_closed=args.alpha_closed,
    )
    log_line(args.exp_dir, f"Training done. Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
