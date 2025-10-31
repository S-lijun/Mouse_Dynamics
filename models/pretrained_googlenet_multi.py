import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
'''
class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users):
        super(PretrainedGoogLeNet_Multilabel, self).__init__()

        # Load pretrained GoogLeNet with weights
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)

        # Remove auxiliary classifiers and output layers
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()  # Disable final 1000-dim output

        # Freeze most layers except inception5 and avgpool
        for name, param in self.base.named_parameters():
            if "inception5" in name or "avgpool" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Final per-user binary classifiers
        self.user_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )for _ in range(num_users)
        ])

    def forward(self, x):
        # Forward through backbone up to feature map
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x)
        x = self.base.inception5b(x)
        x = self.base.avgpool(x)    # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)     # → [B, 1024]

        # Pass through per-user branches
        outputs = [branch(x) for branch in self.user_branches]  # list of [B,1]
        out = torch.cat(outputs, dim=1)  # [B, num_users]

        return out  # No sigmoid here; handled in loss
'''

'''
class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users, embedding_dim=128):
        super().__init__()

        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()

        for name, param in self.base.named_parameters():
            if "inception5" in name or "avgpool" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.num_users = num_users
        self.embedding_dim = embedding_dim

        # Learnable user embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Shared attention branch for all users
        self.attention_layer = nn.Sequential(
            nn.Linear(1024 + embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x) # unfreezed
        x = self.base.inception5b(x) # unfreezed
        x = self.base.avgpool(x)      # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)       # [B, 1024]

        # Compute scores for each user using attention
        batch_size = x.shape[0]
        user_ids = torch.arange(self.num_users, device=x.device)  # [num_users]
        user_embed = self.user_embeddings(user_ids)               # [num_users, D]
        user_embed = user_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_users, D]

        x_expanded = x.unsqueeze(1).expand(-1, self.num_users, -1)       # [B, num_users, 1024]

        concat = torch.cat([x_expanded, user_embed], dim=-1)            # [B, num_users, 1024+D]

        logits = self.attention_layer(concat).squeeze(-1)               # [B, num_users]
        return logits

'''

# Model: All the branches share one FC layers. Only difference across users are the embedding layer
'''
class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users, embedding_dim=128):
        super().__init__()

        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()

        # Unfreeze all layers in the backbone
        for param in self.base.parameters():
            param.requires_grad = True

        self.num_users = num_users
        self.embedding_dim = embedding_dim

        # Learnable user embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Shared attention branch for all users
        self.attention_layer = nn.Sequential(
            nn.Linear(1024 + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),

            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x)
        x = self.base.inception5b(x)
        x = self.base.avgpool(x)      # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)       # [B, 1024]

        # Compute scores for each user using attention
        batch_size = x.shape[0]
        user_ids = torch.arange(self.num_users, device=x.device)   # [num_users]
        user_embed = self.user_embeddings(user_ids)                # [num_users, D]
        user_embed = user_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_users, D]

        x_expanded = x.unsqueeze(1).expand(-1, self.num_users, -1)       # [B, num_users, 1024]
        concat = torch.cat([x_expanded, user_embed], dim=-1)             # [B, num_users, 1024+D]

        logits = self.attention_layer(concat).squeeze(-1)                # [B, num_users]
        return logits
'''
# Model: Every branches have independnet FC layers

import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights


class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained GoogLeNet backbone
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()

        # Unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        self.num_users = num_users

        # 每个用户一套独立的FC head
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        # Backbone feature extraction
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x)
        x = self.base.inception5b(x)
        x = self.base.avgpool(x)     # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)      # [B, 1024]

        # 每个用户独立head计算logit
        logits = [head(x) for head in self.user_heads]   # list of [B,1]
        logits = torch.cat(logits, dim=1)                # [B, num_users]

        return logits
