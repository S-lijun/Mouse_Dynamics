import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_b_32
from torchvision.models import ViT_B_32_Weights

'''
class PretrainedViT_B32_Multilabel(nn.Module):
    def __init__(self, num_users, image_size=224):
        super().__init__()

        # Standard ImageNet pretrained ViT-B/32

        self.backbone = vit_b_32(
            weights=ViT_B_32_Weights.IMAGENET1K_V1,
            image_size=448
        )

        # CLS feature dimension = 768
        feature_dim = self.backbone.heads.head.in_features

        # Remove ImageNet classifier
        self.backbone.heads = nn.Identity()

        # Fine-tune entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.num_users = num_users

        # Chong-style user-specific heads
        self.user_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1)
            for _ in range(num_users)
        ])

    def forward(self, x):

        # Standard ViT forward
        # returns CLS token feature after encoder
        features = self.backbone(x)   # [B, 768]

        logits = [
            head(features)
            for head in self.user_heads
        ]

        return torch.cat(logits, dim=1)
'''

import timm
import torch
import torch.nn as nn


class PretrainedViT_B32_Multilabel(nn.Module):

    def __init__(self, num_users):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch32_224",
            pretrained=True,
            img_size=448,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.user_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1)
            for _ in range(num_users)
        ])

    def forward(self, x):

        features = self.backbone(x)

        logits = [
            head(features)
            for head in self.user_heads
        ]

        return torch.cat(logits, dim=1)