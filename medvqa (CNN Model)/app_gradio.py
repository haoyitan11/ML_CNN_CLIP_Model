# --- auto-path: allow running without installing the package ---
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# ----------------------------------------------------------------

import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from medvqa.utils.io import read_json
from medvqa.models.cnn_vqa import CNNVQAModel
from medvqa.utils.question_router import route_question

TOPK = 5

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

VOCAB_PATH = os.environ.get("MEDVQA_VOCAB", "outputs/answer_vocab.json")
CKPT_PATH  = os.environ.get("MEDVQA_CKPT",  "outputs/exp_cnn/checkpoints/best.pt")
TEXT_MODEL = os.environ.get("MEDVQA_TEXT_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

IMG_SIZE = int(os.environ.get("MEDVQA_IMG_SIZE", "224"))
MAX_LEN  = int(os.environ.get("MEDVQA_MAX_LEN", "64"))
ATTN_DIM = int(os.environ.get("MEDVQA_ATTN_DIM", "512"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = read_json(VOCAB_PATH)
itos = vocab["itos"]

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model = CNNVQAModel(num_answers=len(itos), text_model_name=TEXT_MODEL, freeze_resnet=False, freeze_text=False, attn_dim=ATTN_DIM).to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

@torch.no_grad()
def predict(image: Image.Image, question: str):
    if image is None or not question or not question.strip():
        return "Please upload an image and type a question.", "", ""

    q = question.strip()
    img = image.convert("RGB")
    x_img = tfm(img).unsqueeze(0).to(device)

    tok = tokenizer([q], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    logits_open, logits_closed = model(image=x_img, input_ids=input_ids, attention_mask=attention_mask)

    r = route_question(q)

    if r.qtype == "closed_yesno":
        probs = torch.softmax(logits_closed, dim=-1).squeeze(0)
        p_no = float(probs[0].item())
        p_yes = float(probs[1].item())
        pred = "yes" if p_yes >= p_no else "no"
        top_str = "\n".join([f"no:  {p_no:.6f}", f"yes: {p_yes:.6f}"])
        debug = f"Detected: {r.qtype} | subtype={r.subtype} | {r.debug}"
        return pred, top_str, debug

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
    k = min(TOPK, probs.numel())
    vals, idxs = probs.topk(k)

    pred = itos[idxs[0].item()]
    top_str = "\n".join([f"{itos[i.item()]}: {v.item():.6f}" for v, i in zip(vals, idxs)])
    debug = f"Detected: {r.qtype} | subtype={r.subtype} | {r.debug}"
    return pred, top_str, debug

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Medical Image"),
        gr.Textbox(label="Question"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Answer"),
        gr.Textbox(label="Top-5 Answers (probability)"),
        gr.Textbox(label="Debug"),
    ],
    title="Med-VQA CNN Baseline (Two-Head + Spatial Attention + FiLM)",
)

if __name__ == "__main__":
    demo.launch()
