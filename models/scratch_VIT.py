import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----- Patch Embedding -----
class PatchEmbedding(nn.Module):
    def __init__(self, dim, img_size, patch_size, n_channels):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = n_channels * (patch_size ** 2)
        self.linear_project = nn.Linear(self.patch_dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Split into patches
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H/P, W/P, C, P, P]
        x = x.reshape(B, -1, self.patch_dim)  # [B, N, patch_dim]
        x = self.linear_project(x)  # [B, N, dim]
        return x

# ----- Multi-head Self Attention -----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "Embedding dim must be divisible by number of heads"
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

# ----- Transformer Block -----
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ----- Vision Transformer -----
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

        self.patch_embed = PatchEmbedding(dim, image_size, patch_size, in_channels)
        num_patches = (image_size // patch_size) ** 2

        # CLS token & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # MLP Head (2-layer, better than single linear)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, dim]

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, dim]

        # add positional embedding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # take CLS token
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# ----- InsiderThreatViT -----
class InsiderThreatViT(ViT):
    def __init__(self):
        super().__init__(
            image_size=224,
            patch_size=8,
            in_channels=3,
            num_classes=1,
            dim=384,
            depth=6,
            heads=6,
            mlp_dim=512
        )

