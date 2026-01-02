import torch
from transformers import CLIPProcessor

#collate function for dataloader
class CLIPMultiTaskCollate:
    def __init__(self, clip_name: str, max_len: int = 64):
        #load HuggingFace CLIPprocessor
        self.proc = CLIPProcessor.from_pretrained(clip_name)
        
        #maximum token length
        self.max_len = max_len

    def __call__(self, batch):
        #batch: list of dataset items
        images = []
        questions = []
        tasks = []
        labels = []
        
        #collect fields from each sample into lists
        for b in batch:
            img = b["image"]
            
            #ensure image is RGB
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                
            images.append(img)
            questions.append(b["question"])
            
            #keep tasks as python list of strings
            tasks.append(b["task"])
            
            #labels will later be converted into a tensor
            labels.append(b["label"])

        #Run CLIP preprocessing
        enc = self.proc(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        
        #return batch dictionary used by training loop/model
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", None),
            "tasks": tasks,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
