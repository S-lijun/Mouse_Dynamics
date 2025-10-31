import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PretrainedViT_B16_Multilabel(nn.Module):
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained ViT-B/16 backbone
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)

        # Freeze backbone
        for param in self.base.parameters():
            param.requires_grad = False

        # Remove classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        self.num_users = num_users

        # 每个用户独立的 FC head
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 512),
                Swish(),
                nn.Dropout(0.5),

                nn.Linear(512, 128),
                Swish(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)   # binary output
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        # ===== ViT forward 展开 =====
        # Patch embedding
        x = self.base._process_input(x)   # [B, num_patches, hidden_dim]

        # 加上 CLS token
        batch_size = x.shape[0]
        cls_token = self.base.class_token.expand(batch_size, -1, -1)   # [B, 1, hidden_dim]

        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, hidden_dim]

        # 加上位置编码
        x = x + self.base.encoder.pos_embedding[:, :(x.size(1)), :]

        # Dropout
        x = self.base.encoder.dropout(x)

        # Transformer Encoder (frozen, no gradient)
        x = self.base.encoder.layers(x)   # [B, num_patches+1, hidden_dim]

        # LayerNorm
        x = self.base.encoder.ln(x)

        # 取 CLS token
        x = x[:, 0]   # [B, hidden_dim]

        # ===== 每个用户独立 head =====
        logits = [head(x) for head in self.user_heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)               # [B, num_users]

        return logits  # 不做 sigmoid
