from _setup_path import *  # noqa: F401,F403

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from medvqa.utils.io import read_json
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.utils.question_router import route_question

#Allowed answer sets used to restrict the open-vocabulary classifer for certain routed question
#Rule-based safety/constraint to reduce nonsensical predictions
MODALITY_ANS = {"x-ray", "chest x-ray", "ct", "mri", "ultrasound", "pet", "angiography", "fluoroscopy", "flair", "t2", "adc", "dwi"}
PLANE_ANS    = {"axial", "coronal", "sagittal"}
VIEW_ANS     = {"pa", "ap", "lateral"}
ORGAN_ANS    = {"lungs", "lung", "chest", "brain", "abdomen", "kidney", "kidneys", "liver", "heart"}
LATERAL_ANS  = {"right", "left", "bilateral"}

#Mask logits so subset of answers are allowed
def mask_logits(logits: torch.Tensor, itos: list, allowed_set: set) -> torch.Tensor:
    
    #Find indices in the vocabulary whose answer string is allowed
    keep = [i for i, a in enumerate(itos) if a in allowed_set]
    
    #Logits unchanged if cant find any allowed labels in vocab
    if not keep:
        return logits
    
    #Create a tensor filled 
    masked = torch.full_like(logits, float("-inf"))
    
    #Copy original logits only for allowed indices
    masked[:, keep] = logits[:, keep]
    return masked

#Disable gradient tracking for inference
@torch.no_grad()
def predict_one(model, device, itos, tokenizer, tfm, img: Image.Image, q: str, max_len: int = 64, topk: int = 5):
    #Ensure consistent 3-channel color format
    if hasattr(img, "convert"):
        img = img.convert("RGB")
        
    #Apply image preprocessing transform
    x_img = tfm(img).unsqueeze(0).to(device)

    #Tokenize the question text into input + attention for the text encoder
    tok = tokenizer([q], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    #forward pass through the model
    logits_open, logits_closed = model(image=x_img, input_ids=input_ids, attention_mask=attention_mask)

    #Route the question using heuristic rules
    r = route_question(q)

    #If its yes/no question, choose close head
    if r.qtype == "closed_yesno":
        probs = torch.softmax(logits_closed, dim=-1).squeeze(0)
        p_no = float(probs[0].item())
        p_yes = float(probs[1].item())
        pred = "yes" if p_yes >= p_no else "no"
        top_str = f"no:  {p_no:.6f}\nyes: {p_yes:.6f}"
        return pred, top_str, f"{r.qtype} | {r.subtype} | {r.debug}"

    #Otherwise treat as an open-ended question: start from open Logits
    logits = logits_open
    
    #Restrict answer space based on routed subtype
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

    #Convert logits > probabilities
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    
    #choose k = min
    k = min(topk, probs.numel())
    
    #Extract top-k probabilities and indices
    vals, idxs = probs.topk(k)

    #Predicted answer is the highest-probability vocab entry
    pred = itos[idxs[0].item()]
    #Build a multi-line string showing the top-K answers with probabilities
    top_str = "\n".join([f"{itos[i.item()]}: {v.item():.6f}" for v, i in zip(vals, idxs)])
    #return prediction, top-k string, and router debug info
    return pred, top_str, f"{r.qtype} | {r.subtype} | {r.debug}"

def main():
    #CLI argument parser
    ap = argparse.ArgumentParser()
    #exported/deduplicated images
    ap.add_argument("--images_dir", type=str, required=True)
    #JSON file describing which images + questions to run
    ap.add_argument("--questions_json", type=str, default="tests/questions.json")
    #Answer vocabulary JSON
    ap.add_argument("--vocab_path", type=str, default="outputs/answer_vocab.json")
    #Path to trained CNN checkpoint
    ap.add_argument("--ckpt_path", type=str, default="outputs/exp_cnn/checkpoints/best.pt")
    #HuggingFace text model name used by CNNVQAModel
    ap.add_argument("--text_model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    #image resize size
    ap.add_argument("--img_size", type=int, default=224)
    #max token
    ap.add_argument("--max_len", type=int, default=64)
    #How many top answers to show
    ap.add_argument("--topk", type=int, default=5)
    #output folder
    ap.add_argument("--out_dir", type=str, default="outputs/batch_tests")
    #Internal attention project dimension for the model
    ap.add_argument("--attn_dim", type=int, default=512)
    #Parse CLI args
    args = ap.parse_args()

    #Compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load vocabulary JSON
    vocab = read_json(args.vocab_path)
    itos = vocab["itos"]

    #Load tokenizer for the specified text model
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    
    #image preprocessing pipeline
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    #Build CNN VQA model
    model = CNNVQAModel(
        num_answers=len(itos),
        text_model_name=args.text_model_name,
        freeze_resnet=False,
        freeze_text=False,
        attn_dim=args.attn_dim,
    ).to(device)
    
    #Load trained weights from checkpoint
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    
    #model evaluation mode
    model.eval()

    #read questions specification JSON file
    qdata = json.loads(Path(args.questions_json).read_text(encoding="utf-8"))

    #Resolve input and output directory
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #Store per-question rows for later CSV export
    rows = []
    
    #Loop through JSON "Images" List
    for item in qdata["images"]:
        name = item["name"]
        filename = item["file"]
        img_path = images_dir / filename
        
        #If image file is missing
        if not img_path.exists():
            print(f"[WARN] Missing: {img_path} (skip {name})")
            continue

        #open the image once and reuse it for multiple questions
        img = Image.open(img_path)
        
        #output text file per image (contains all questions/answers)
        out_txt = out_dir / f"{name}.txt"
        #Header line for per-image report
        lines = [f"== {name} ({filename}) =="]

        #Run all questions for this image
        for q in item["questions"]:
            #Predict answer + top-K + debug info
            pred, top5, dbg = predict_one(model, device, itos, tokenizer, tfm, img, q, max_len=args.max_len, topk=args.topk)
            
            #Append human-readable log
            lines.append("")
            lines.append(f"Q: {q}")
            lines.append(f"A: {pred}")
            lines.append("Top-5:")
            lines.append(top5)
            lines.append(f"Debug: {dbg}")
            
            #Also collect a compact row for CSV output
            rows.append({"image": name, "question": q, "pred": pred, "top5": top5.replace("\n"," | "), "debug": dbg})

        #Write the per-image text report to disk
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved: {out_txt}")

    #csv only
    import csv
    
    #Write a summary CSV file for all predictions across all images/questions
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image","question","pred","top5","debug"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    #display message
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
