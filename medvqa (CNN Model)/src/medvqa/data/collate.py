import torch
from torchvision import transforms
from transformers import AutoTokenizer

#Collate function class for a dataloader
class CNNVQACollate:
    def __init__(self, text_model_name: str, img_size: int = 224, max_len: int = 64):
        #Load the tokenizer matching the text backbone
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        #Max question length 
        self.max_len = max_len
        
        #Compose several image transforms into one pipeline
        self.tfm = transforms.Compose([
            #resize
            transforms.Resize((img_size, img_size)),
            #convert PIL image to tensor
            transforms.ToTensor(),
            #normalize using ImageNet 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __call__(self, batch):
        #batch: a list of dataset items
        
        #will store transformed image tensors for each sample in the batch
        imgs = []
        
        #Loop over each sample
        for b in batch:
            #Fetch the image field
            img = b["image"]
            #if look liks PIL image, convert to RGB
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                
            #Apply preprocessing transforms and add resulting tensor
            imgs.append(self.tfm(img))
            
        #Stack list of tensors into single batch tensor
        images = torch.stack(imgs)

        #Collect all question strings for this batch
        questions = [b["question"] for b in batch]
        
        #Batch tokenization of questions strings
        tok = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        #create tensors of open-ended answer
        open_labels = torch.tensor([b["open_label"] for b in batch], dtype=torch.long)
        #create tensors of closed-ended answer
        closed_labels = torch.tensor([b["closed_label"] for b in batch], dtype=torch.long)
        
        #indicate whether question is closed (yes/no) or not.
        is_closed_q = torch.tensor([b["is_closed_q"] for b in batch], dtype=torch.long)

        #return the exact structure expected by training evaluation flag
        return {
            "image": images,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "open_label": open_labels,
            "closed_label": closed_labels,
            "is_closed_q": is_closed_q,
        }
