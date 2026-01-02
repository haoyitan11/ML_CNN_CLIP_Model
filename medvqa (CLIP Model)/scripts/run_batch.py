import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse, os, json
from src.medvqa_clip.utils.io import read_json, write_json
from src.medvqa_clip.utils.runtime import get_device
from src.medvqa_clip.models.multitask_clip import MultiTaskCLIP
from src.medvqa_clip.models.hybrid_answerer import HybridAnswerer
import torch
from PIL import Image

def main():
    #Parse command-line arguments
    ap = argparse.ArgumentParser()
    
    #Path to questions_template.json
    ap.add_argument("--questions", type=str, required=True, help="Path to questions_template.json")
    
    #Output path for batch result JSON
    ap.add_argument("--out", type=str, default="outputs/batch_results.json")
    
    #Path to vocab bundle JSON
    ap.add_argument("--vocab_bundle", type=str, default="outputs/vocabs.json")
    
    #Path to trained checkpoint
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_clip/checkpoints/best.pt")
    
    #CLIP backbone
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    
    #How many top-k answers to store per question
    ap.add_argument("--topk", type=int, default=5)
    
    #optional: enable touchxrayvision
    ap.add_argument("--use_xrv", action="store_true")
    args = ap.parse_args()

    device = get_device()
    
    #load vocab.json
    vocabs = read_json(args.vocab_bundle)

    #build multitaskCLIP model
    model = MultiTaskCLIP(vocabs=vocabs, clip_name=args.clip_name, hidden_dim=512, freeze_vision=False, freeze_text=False).to(device)
    
    #load checkpoint weights if checkpoint exists
    if os.path.exists(args.ckpt_path):
        state = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state)
        
    #set model to evaluation mode
    model.eval()

    #Build hybridAnswerer
    answerer = HybridAnswerer(model=model, vocabs=vocabs, clip_name=args.clip_name, device=device, use_xrv=args.use_xrv)

    #Load question specification JSON
    spec = read_json(args.questions)
    items = spec.get("items", [])
    
    #results will store output for all images
    results = []

    #loop through each item
    for item in items:
        base_dir = item.get("base_dir", "")
        img_name = item.get("image", "")
        path = os.path.join(base_dir, img_name) if base_dir else img_name
        path = path.replace("/", os.sep)

        #questions list for this image
        qs = item.get("questions", [])
        
        img = Image.open(path)
        
        #create result block for this image
        block = {"image_path": path, "qa": []}
        
        #ask all questions for this image
        for q in qs:
            #predict using HybridAnswerer
            res = answerer.predict(img, q, topk=args.topk)
            
            #store the answer + metadata + topic distribution
            block["qa"].append({
                "question": q,
                "answer": res.answer,
                "used": res.used,
                "topk": [{"answer": a, "prob": p} for a, p in res.topk]
            })
            
        #add image block to results
        results.append(block)

    #save results JSON
    write_json({"results": results}, args.out)
    
    #display message
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
