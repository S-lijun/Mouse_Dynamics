import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== Patch Embedding ===============
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # [B,3,H,W] -> [B,D,H/P,W/P]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B,N,D]
        return x


# =============== Standard Transformer Block ===============
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# =============== Scratch ViT (Mean Pool, Multi-Label) ===============
class ScratchViT_MultiLabel(nn.Module):
    def __init__(
        self,
        num_users,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        drop=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Multi-label heads
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B,N,D]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Mean pooling (no CLS)
        x = x.mean(dim=1)  # [B,D]

        # Multi-head outputs
        logits = [head(x) for head in self.user_heads]
        logits = torch.cat(logits, dim=1)

        return logits