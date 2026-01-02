import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class MultiTaskCLIP(nn.Module):
    def __init__(self, vocabs: dict, clip_name: str, hidden_dim: int = 512,
                 freeze_vision: bool = False, freeze_text: bool = False):
        super().__init__()
        
        #Load pretrained CLIP model from HuggingFace (Vision encoder + Text encoder)
        self.clip = CLIPModel.from_pretrained(clip_name)

        #Freeze encoder to train only small heads
        if freeze_vision:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False

        #CLIP produces embeddings of size projection_dim, For VIT-B/32, projection_dim is typically 512
        proj_dim = self.clip.config.projection_dim 
        in_dim = proj_dim * 4

        #Small fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        #One classification head per task
        self.heads = nn.ModuleDict()
        for task, v in vocabs.items():
            n = len(v["itos"])
            self.heads[task] = nn.Linear(hidden_dim, n)

    def forward_features(self, pixel_values, input_ids, attention_mask=None):
        #encode image > embedding
        v = self.clip.get_image_features(pixel_values=pixel_values)
        
        #encode text > embedding
        t = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        #normalize embeddings to unit length
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        #create interaction features and concatenate into one vector
        x = torch.cat([v, t, v*t, (v-t).abs()], dim=-1)
        
        #Fuse into a single shared feature representation
        return self.fuse(x)

    def forward_task(self, task: str, feats: torch.Tensor):
        return self.heads[task](feats)
