from _setup_path import * 

import os
import json
import csv
import argparse
import torch

from medvqa.utils.io import read_json, write_json
from medvqa.utils.runtime import get_device, prepare_test_loader
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.engine.evaluate import evaluate_two_head_emf1

@torch.no_grad()
def save_predictions(model, loader, device, out_csv: str):
    #Save per-example predictions 
    
    #Ensure output directory
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    #Put the model into evaluation mode
    model.eval()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        #csv writer
        w = csv.writer(f)
        w.writerow(["is_closed_q", "pred_open_id", "gt_open_id", "pred_closed", "gt_closed"])

        #Iterate over the test dataloader in batches
        for batch in loader:
            #Image tensor batch
            image = batch["image"].to(device)
            #Tokenized question
            input_ids = batch["input_ids"].to(device)
            #Attention mask for the text model 
            attention_mask = batch["attention_mask"].to(device)

            #Ground-truth labels for open-answer classification
            open_y = batch["open_label"].to(device)
            #Ground-truth labels for closed-answer classification
            closed_y = batch["closed_label"].to(device)
            #Indicator of whether question is closed-ended
            is_closed_q = batch["is_closed_q"].to(device)

            #Forward pass: return logits for open-head and close-head
            logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

            #Predicted open class
            pred_open = logits_open.argmax(dim=-1)
            #Predicted closed yes/no class
            pred_closed = logits_closed.argmax(dim=-1)

            #Move tensors to CPU and convert to Python lists
            for ic, po, go, pc, gc in zip(
                is_closed_q.cpu().tolist(),
                pred_open.cpu().tolist(),
                open_y.cpu().tolist(),
                pred_closed.cpu().tolist(),
                closed_y.cpu().tolist(),
            ):
                
                #write a single row
                w.writerow([ic, po, go, pc, gc])

def main():
    ap = argparse.ArgumentParser()

    #HuggingFace dataset, save answer vocabulary, Trained CNN model checkpoint, experiment output directory
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--vocab_path", type=str, default="outputs/answer_vocab.json")
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_cnn/checkpoints/best.pt")
    ap.add_argument("--exp_dir", type=str, default="outputs/exp_cnn")

    #Text encoder backbone used inside CNNVQAModel
    ap.add_argument("--text_model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--attn_dim", type=int, default=512)

    args = ap.parse_args()

    device = get_device()
    
    #Load the answer vocabulary
    vocab = read_json(args.vocab_path)

    #Build the test dataloader
    test_loader = prepare_test_loader(
        dataset_name=args.dataset_name,
        vocab=vocab,
        text_model_name=args.text_model_name,
        img_size=args.img_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    #Instantiate the CNN VQA model
    model = CNNVQAModel(
        num_answers=len(vocab["itos"]),
        text_model_name=args.text_model_name,
        freeze_resnet=False,
        freeze_text=False,
        attn_dim=args.attn_dim,
    ).to(device)

    #Load checkpoint weights
    state_dict = torch.load(args.ckpt_path, map_location=device)
    
    #Apply the loaded weights
    model.load_state_dict(state_dict)

    #Evalaute on the test set
    other_open_id = vocab["stoi"].get("OTHER", None)

    metrics = evaluate_two_head_emf1(
    model, test_loader, device,
    show_progress=True,
    ignore_open_other=True,
    other_open_id=other_open_id,
    )

    #ensure directory exists
    os.makedirs(args.exp_dir, exist_ok=True)
    
    #save as JSON file
    write_json(metrics, os.path.join(args.exp_dir, "metrics_test.json"))
    
    #save prediction
    save_predictions(model, test_loader, device, out_csv=os.path.join(args.exp_dir, "predictions.csv"))

    #display message
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.exp_dir}/metrics_test.json and predictions.csv")

if __name__ == "__main__":
    main()
