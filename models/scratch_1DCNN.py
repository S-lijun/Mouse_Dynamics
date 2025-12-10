'''Not yet implement'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Single1DCNN(nn.Module):
    """
    Simple 1D-CNN for single-label classification.
      - Global Average Pooling
      - LayerNorm
      - Dropout (0.2)
      - GELU activation
    """

    def __init__(self, input_dim=3, seq_len=86, num_classes=1):
        super(Single1DCNN, self).__init__()

        # Save configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes

        # === CNN Backbone ===
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.2)
        self.feature_dim = 128

        # === Shared classification head ===
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)   # single-head softmax output
        )

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        return: logits (B, num_classes)
        """
        # Channel-last → Channel-first
        x = x.permute(0, 2, 1)  # [B, input_dim, seq_len]

        # Backbone
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # [B, 128, seq_len/2]

        # Global Average Pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, 128]

        # Normalization + Dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Classification head
        logits = self.classifier(x)
        return logits
