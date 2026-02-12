import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

'''
# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PretrainedViT_B16_Multilabel_NoCLS_NoPos(nn.Module):
    """
    - 去掉 CLS token
    - 去掉 positional embeddings
    - 使用所有 patch features 做 max aggregation (论文风格)
    - 不修改你的 multi-label user heads
    """
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained ViT-B/16 backbone
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)

        # unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        # Remove classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        self.num_users = num_users

        # === Multi-user independent heads (unchanged) ===
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 512),
                Swish(),
                nn.Dropout(0.5),

                nn.Linear(512, 128),
                Swish(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)  # binary output per user
            )
            for _ in range(num_users)
        ])

    # -------------------------------------------------------------------------
    #  🔥 前向传播 = 论文风格：
    #    - 无 CLS token
    #    - 无 positional embedding
    #    - transformer patches → max pooling → heads
    # -------------------------------------------------------------------------
    def forward(self, x):

        # x: [B, C, H, W]  (your images)
        # Patch embedding, but WITHOUT adding CLS token
        x = self.base._process_input(x)  # [B, num_patches, hidden_dim]

        B, N, D = x.shape

        # ====== 🔥 不加 class token ======
        # ====== 🔥 不加 positional embedding ======
        # 所以直接送进 encoder

        # the ViT encoder expects shape [B, N, D]
        # but torchvision uses: encoder(x) where x is [B, N, D]

        # Go through transformer layers without CLS token
        x = self.base.encoder.dropout(x)  # keep consistency
        x = self.base.encoder.layers(x)   # transformer blocks
        x = self.base.encoder.ln(x)

        # ===== 🔥 MAX-AGGR-2 (论文推荐) =====
        # 取每个 feature 维度在所有 patches 的最大值
        # x: [B, N, D] → pooled: [B, D]
        pooled = torch.max(x, dim=1)[0]

        # ===== Multi-head classifier (your unchanged part) =====
        logits = [head(pooled) for head in self.user_heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)                    # [B, num_users]

        return logits
'''

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PretrainedViT_B16_Multilabel_NoCLS_NoPos(nn.Module):
    """
    - 支持自定义 image_size (如 448)
    - 去掉 CLS token
    - 去掉 positional embeddings
    - 使用所有 patch features 做 max aggregation (论文风格)
    - 不修改你的 multi-label user heads
    """
    def __init__(self, num_users, image_size=224): # 1. 增加 image_size 参数，默认224以保持兼容
        super().__init__()

        # Load pretrained ViT-B/16 backbone
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        
        # 2. 在这里显式传入 image_size，解决 torchvision 的尺寸校验报错
        self.base = vit_b_16(weights=weights, image_size=image_size)

        # unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        # Remove classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        self.num_users = num_users

        # === Multi-user independent heads (unchanged) ===
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 512),
                Swish(),
                nn.Dropout(0.5),

                nn.Linear(512, 128),
                Swish(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)  # binary output per user
            )
            for _ in range(num_users)
        ])

    # -------------------------------------------------------------------------
    #  🔥 前向传播 = 论文风格：
    #    - 无 CLS token
    #    - 无 positional embedding
    #    - transformer patches → max pooling → heads
    # -------------------------------------------------------------------------
    def forward(self, x):

        # x: [B, C, H, W]  (your images)
        # Patch embedding. 此时内部 self.image_size 已更新，不会再报错
        x = self.base._process_input(x)  # [B, num_patches, hidden_dim]

        B, N, D = x.shape

        # ====== 🔥 不加 class token ======
        # ====== 🔥 不加 positional embedding ======
        # 所以直接送进 encoder

        # the ViT encoder expects shape [B, N, D]
        # but torchvision uses: encoder(x) where x is [B, N, D]

        # Go through transformer layers without CLS token
        x = self.base.encoder.dropout(x)  # keep consistency
        x = self.base.encoder.layers(x)   # transformer blocks
        x = self.base.encoder.ln(x)

        # ===== 🔥 MAX-AGGR-2 (论文推荐) =====
        # 取每个 feature 维度在所有 patches 的最大值
        # x: [B, N, D] → pooled: [B, D]
        pooled = torch.max(x, dim=1)[0]

        # ===== Multi-head classifier (your unchanged part) =====
        logits = [head(pooled) for head in self.user_heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)                    # [B, num_users]

        return logits
