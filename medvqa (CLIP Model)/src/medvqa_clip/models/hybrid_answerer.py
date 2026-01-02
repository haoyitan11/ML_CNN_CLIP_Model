from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from transformers import CLIPProcessor

from medvqa_clip.data.router import route_question


#Datacontainer for results
@dataclass
class AnswerResult:
    #Final predicted answer
    answer: str
    
    #Top-k answers with probabilities
    topk: List[Tuple[str, float]]
    
    #Which subsystem produced the output
    used: str


# HybridAnswerer
class HybridAnswerer:
    def __init__(
        self,
        model,
        vocabs: dict,
        clip_name: str,
        device: str,
        use_xrv: bool = True,
        max_len: int = 64,
        xrv_threshold: float = 0.5,
    ):
        #save references (MultitaskCLIP model)
        self.model = model
        self.vocabs = vocabs
        self.device = device
        self.max_len = max_len
        self.xrv_threshold = xrv_threshold

        #CLIP processor for image+text encoding
        self.proc = CLIPProcessor.from_pretrained(clip_name)

        #Optional torchxrayvision setup
        self.use_xrv = use_xrv
        self.xrv = None
        self.xrv_model = None
        self.xrv_labels = None

        #if enabled, try to load the specialist CXR model
        if use_xrv:
            try:
                import torchxrayvision as xrv

                #Load pretrained XRV DenseNet model (trained on multiple CXR datasets)
                self.xrv = xrv
                self.xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
                self.xrv_model = self.xrv_model.to(device)
                self.xrv_model.eval()
                
                #Labels/pathology
                self.xrv_labels = list(self.xrv_model.pathologies)

                print("[OK] torchxrayvision loaded. Using XRV for chest findings.")
            except Exception as e:
                #if torchxrayvision isnt installed or fail to load
                self.xrv = None
                self.xrv_model = None
                self.xrv_labels = None
                print(f"[WARN] torchxrayvision not available. Reason: {e}")

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str, topk: int = 5) -> AnswerResult:
        # Ensure RGB image for CLIP
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        #Route question > task
        spec = route_question(question)

        #Choose the actual task name
        if spec.name in self.vocabs:
            task = spec.name
        else:
            task = "yesno" if spec.kind == "closed" else "modality"

        #Chest pathology
        if (
            self.xrv_model is not None
            and task in {"pneumonia", "effusion", "pneumothorax", "cardiomegaly", "mass"}
        ):
            ans, tk = self._predict_xrv(image, task)
            if ans is not None:
                return AnswerResult(answer=ans, topk=tk, used=f"xrv:{task}")

        #MultitaskCILP inferences
        enc = self.proc(
            text=[question],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        #Move encoded tensors to device
        pixel_values = enc["pixel_values"].to(self.device)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        #Forward pass through shared feature extractor
        feats = self.model.forward_features(pixel_values, input_ids, attention_mask)
        
        #Forward pass through the selected task head
        logits = self.model.forward_task(task, feats)

        #Convert logits to probabilities 
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        #label list
        itos = self.vocabs[task]["itos"]

        #Compute top-k predictions
        k = min(topk, probs.numel())
        vals, idxs = probs.topk(k)

        #Best prediction is top-1 index
        pred = itos[idxs[0].item()]
        
        #Build top-k output list
        tk = [(itos[i.item()], float(v.item())) for v, i in zip(vals, idxs)]

        return AnswerResult(answer=pred, topk=tk, used=f"clip:{task}")

    #XRV pathology predictor
    def _predict_xrv(self, image: Image.Image, task: str):
        #Map task name to XRV label name
        mapping = {
            "pneumonia": ["Pneumonia"],
            "effusion": ["Effusion", "Pleural Effusion"],
            "pneumothorax": ["Pneumothorax"],
            "cardiomegaly": ["Cardiomegaly"],
            "mass": ["Mass"],
        }
        
        #candidate label names
        want = mapping.get(task, [])
        if not want or self.xrv_labels is None:
            return None, []

        #find first matching label name
        idx = None
        for w in want:
            if w in self.xrv_labels:
                idx = self.xrv_labels.index(w)
                break
        if idx is None:
            #task label not supported by this XRV model
            return None, []

        #Preprocess for XRV
        import torchvision.transforms as T

        xrv_mean = 0.5
        xrv_std = 0.25

        #Convert to grayscale for client chest x-ray classifier
        img = image.convert("L")
        
        #Transform pipeline > tensor > normalized tensor
        tfm = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[xrv_mean], std=[xrv_std]),
        ])

        #Add batch dimension and move to device
        x = tfm(img).unsqueeze(0).to(self.device)

        # Forward -> sigmoid probabilities
        y = self.xrv_model(x)
        y = torch.sigmoid(y).squeeze(0)

        #Probability for the selected pathology index
        score = float(y[idx].item())

        #Convert probability into yes/no using threshold
        ans = "yes" if score >= self.xrv_threshold else "no"
        tk = [("yes", score), ("no", 1.0 - score)]
        return ans, tk
