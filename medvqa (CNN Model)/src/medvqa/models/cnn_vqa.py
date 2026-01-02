import torch
import torch.nn as nn
import torchvision.models as tv
from transformers import AutoModel

class CNNVQAModel(nn.Module):
    def __init__(self, num_answers: int, text_model_name: str, freeze_resnet: bool = True, freeze_text: bool = False, attn_dim: int = 512):
        super().__init__()

        #Load an ImageNet-pretrained ResNet-50 as the visual backbone
        resnet = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2)
        
        #Keep convolutional layers only (remove avgpool + classifier) > feature map (B,2048,7,7)
        self.vision = nn.Sequential(*list(resnet.children())[:-2])  
        self.vision_dim = 2048

        #Optionally freeze ResNet weights
        if freeze_resnet:
            for p in self.vision.parameters():
                p.requires_grad = False

        #Load a transformer encoder for text (PubMedBERT) for question understanding
        self.text = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text.config.hidden_size

        #Optionally freeze text encoder weights
        if freeze_text:
            for p in self.text.parameters():
                p.requires_grad = False

        #FiLM layer: generates per-channel modulation parameters from text embeddings
        self.film = nn.Linear(self.text_dim, self.vision_dim * 2)

        #Project visual feature map into attention dimension
        self.v_proj = nn.Conv2d(self.vision_dim, attn_dim, kernel_size=1, bias=False)
        
        #Project text embedding into attention dimenstion
        self.t_proj = nn.Linear(self.text_dim, attn_dim, bias=False)

        #MLP trunk after fusing attended visual vector + text vector
        fused_dim = self.vision_dim + self.text_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        #open-answer head: predicts among a vocabulary of possible answers
        self.open_head = nn.Linear(1024, num_answers)
        self.closed_head = nn.Linear(1024, 2)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        #Extract spatial visual features from the CNN
        V = self.vision(image) 
        B, C, H, W = V.shape

        #Encode question text
        T = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  

        #Compute FiLM parameters from text
        gb = self.film(T)  # (B,2C)
        gamma, beta = gb[:, :C], gb[:, C:]
        
        #Reshape to broadcast across spatial dims
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        #Apply FiLM modulation to visual feature map
        V_film = V * (1.0 + torch.tanh(gamma)) + beta

        #Project visual map to attention space
        Vh = self.v_proj(V_film)              
        
        #Project text to attention query      
        q = self.t_proj(T).unsqueeze(2)          

        #Flatten spatial positions 
        Vh_flat = Vh.flatten(2).transpose(1, 2)     
        
        #Compiute attention logits for each spatial location using dot-product
        attn_logits = torch.bmm(Vh_flat, q).squeeze(2)  
        
        #Normalize attention weights across spatial positions
        attn = torch.softmax(attn_logits, dim=-1)      

        #Apply attention to ORIGINAL FiLM Visual Features
        V_flat = V_film.flatten(2).transpose(1, 2)  
        
        #Weighted sum to get a single attended visual vector
        v_att = torch.bmm(attn.unsqueeze(1), V_flat).squeeze(1)  

        #Fuse attended visual vector with text vector
        x = torch.cat([v_att, T], dim=-1)
        
        #Shared trunk representation
        h = self.trunk(x)

        #Return logits for open-answer and yes/no heads
        return self.open_head(h), self.closed_head(h)
