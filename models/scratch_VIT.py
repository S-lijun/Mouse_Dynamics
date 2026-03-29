import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Linear Patch Embedding (论文公式)
# ============================================================

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=15, in_chans=3, embed_dim=225):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        patch_dim = patch_size * patch_size * in_chans

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):

        # x: [B,C,H,W]

        patches = self.unfold(x)
        patches = patches.transpose(1,2)

        embeddings = self.proj(patches)

        return embeddings



# ============================================================
# Efficient Attention (论文 Section III-E)
# ============================================================

class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads=3, qkv_bias = True):

        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias = qkv_bias)
        self.k = nn.Linear(dim, dim, bias = qkv_bias)
        self.v = nn.Linear(dim, dim, bias = qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):

        B,N,C = x.shape

        q = self.q(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.k(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        v = self.v(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)

        kv = torch.einsum("bhnd,bhne->bhde",k,v)
        kv = kv / math.sqrt(self.head_dim)

        out = torch.einsum("bhnd,bhde->bhne",q,kv)

        out = out.permute(0,2,1,3).reshape(B,N,C)

        return self.proj(out)



# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, dim=225, num_heads=3, mlp_ratio=512/225):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim,num_heads = num_heads)

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim,hidden),
            nn.GELU(),
            nn.Linear(hidden,dim)
        )

    def forward(self,x):

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x



# ============================================================
# Binary ViT
# ============================================================

class BinaryViT(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=15,
                 in_chans=3,
                 embed_dim=225,
                 depth=3,
                 num_heads=3,
                 mlp_ratio=512/225):

        super().__init__()

        self.patch_embed = PatchEmbed(img_size,patch_size,in_chans,embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim,num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decision Layer
        self.head = nn.Linear(embed_dim,1)


    def forward(self,x):

        x = self.patch_embed(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # MAX-AGGR-2
        x = x.max(dim=1)[0]

        logits = self.head(x)

        return logits