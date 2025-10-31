import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Backbone
class PretrainedViT_B16(nn.Module):
    def __init__(self):
        super(PretrainedViT_B16, self).__init__()

        # Load pretrained ViT-B/16
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)

        # Freeze all pretrained parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # Remove original classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        # Custom FC layers (和 PretrainedGoogLeNet 风格一致)
        self.extra_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            Swish(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            Swish(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)   # binary output
        )

    def forward(self, x):
        x = self.base(x)        # ViT backbone features
        x = self.extra_fc(x)    # custom classifier
        return x
