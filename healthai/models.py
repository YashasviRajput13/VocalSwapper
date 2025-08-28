# models.py
import torch, torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from config import Config

class CNN2D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        m = efficientnet_v2_s(weights=None if not hasattr(efficientnet_v2_s, 'Weights')
                              else efficientnet_v2_s.Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        self.backbone = m
        self.fc       = nn.Linear(1280, out_dim)
    def forward(self, x): return self.fc(self.backbone(x))

class CNN3D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, out_dim)
    def forward(self, x): return self.fc(self.conv(x).view(x.size(0), -1))

# class FusionTransformer(nn.Module):
#     def __init__(self, mods, dim, heads=1, layers=2):
#         super().__init__()
#         assert dim % heads == 0
#         enc = TransformerEncoderLayer(d_model=dim, nhead=heads)
#         self.tr  = TransformerEncoder(enc, num_layers=layers)
#         self.out = nn.Linear(dim, len(Config.DISEASES))
# Update FusionTransformer initialization
class FusionTransformer(nn.Module):
    def __init__(self, mods, dim, heads=1, layers=2):
        super().__init__()
        enc = TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True  # Add this line
        )
        self.tr = TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(dim, len(Config.DISEASES))
    def forward(self, feats):
        x = self.tr(feats)        # [B,mods,dim]
        return self.out(x.mean(1))

class MultiModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleDict({
            m:(CNN2D(len(Config.DISEASES)) if m not in ['ct','mri']
               else CNN3D(len(Config.DISEASES)))
            for m in Config.MODALITIES
        })
        self.fusion = FusionTransformer(len(Config.MODALITIES),
                                        len(Config.DISEASES))
    def forward(self, inputs):
        feats = []
        for m, net in self.nets.items():
            if m in inputs:
                out = net(inputs[m]).unsqueeze(1)
                feats.append(out)
        feats = torch.cat(feats, dim=1)
        return self.fusion(feats)




