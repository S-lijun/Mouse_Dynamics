import torch
import torch.nn as nn
import torch.nn.functional as F


class Multi1DCNN_MultiLabel(nn.Module):
    """
    Improved 1D-CNN multi-label model:
      - Global Average Pooling
      - LayerNorm
      - Dropout (0.2)
      - GELU activation
    """

    def __init__(self, input_dim=3, seq_len=86, num_users=10):
        super(Multi1DCNN_MultiLabel, self).__init__()

        # Save configuration for trainer reload
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_users = num_users

        # === Shared CNN Backbone ===
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        # === Global Average Pooling + Normalization ===
        # 最后输出 [B, 128]
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.2)
        self.feature_dim = 128

        # === Per-user independent heads ===
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.GELU(),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        return: logits (B, num_users)
        """
        # === Channel-last -> Channel-first ===
        x = x.permute(0, 2, 1)  # [B, 3, 86]

        # === Backbone ===
        x = F.gelu(self.bn1(self.conv1(x)))   # [B, 64, 86]
        x = F.gelu(self.bn2(self.conv2(x)))   # [B, 128, 86]
        x = self.pool(x)                      # [B, 128, 43]

        # === Global Average Pooling ===
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, 128]

        # === LayerNorm + Dropout ===
        x = self.norm(x)
        x = self.dropout(x)

        # === Per-user classification heads ===
        outputs = [head(x) for head in self.user_heads]
        logits = torch.cat(outputs, dim=1)  # [B, num_users]
        return logits
