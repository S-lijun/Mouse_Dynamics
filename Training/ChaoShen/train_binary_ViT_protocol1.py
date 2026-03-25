# -*- coding: utf-8 -*-

import sys, os, datetime, gc
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ======================================================
# ROOT（唯一可信路径）
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Mouse_Dynamics
print("[PROJECT ROOT]:", PROJECT_ROOT)

# ======================================================
# Logging
# ======================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = PROJECT_ROOT / "output_logs" / "train_binary_vit_tensor"
log_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / f"BinaryViT_tensor_{timestamp}.out"

class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(log_path)

# ======================================================
# IMPORT（强制 root import）
# ======================================================

sys.path.append(str(PROJECT_ROOT))

from models.scratch_VIT import BinaryViT
from Training.Trainers.fast_binary_class_trainer_ViT import BinaryClassTrainer
from Training.Score_Fusion.Score_Fusion_Binary import binary_score_fusion

# ======================================================
# Dataset（自动补全路径）
# ======================================================

class BinaryTensorDataset(Dataset):

    def __init__(self, root):

        # 🔥 自动拼到 PROJECT_ROOT
        root = PROJECT_ROOT / root

        if not root.exists():
            raise FileNotFoundError(f"\n[ERROR] Path not found:\n{root}\n")

        print("[Loading dataset]:", root)

        img_path = root / "images.npy"
        lab_path = root / "labels.npy"
        ses_path = root / "sessions.npy"

        if not img_path.exists():
            raise FileNotFoundError(f"Missing: {img_path}")
        if not lab_path.exists():
            raise FileNotFoundError(f"Missing: {lab_path}")
        if not ses_path.exists():
            raise FileNotFoundError(f"Missing: {ses_path}")

        H, W = 224, 224

        self.labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
        N = len(self.labels)

        self.images = np.memmap(
            img_path,
            dtype=np.uint8,
            mode="r",
            shape=(N, 3, H, W)
        )

        self.sessions = np.load(ses_path, allow_pickle=True)

        print(f"[Dataset Loaded] N = {N}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = torch.from_numpy(self.images[idx]).float() / 255.0
        label = torch.tensor(float(self.labels[idx]))

        return img, label, self.sessions[idx]

# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("="*80)
    print("Binary ViT Training (Path-Safe Version)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================================================
    # INPUT（只输入相对路径）
    # ======================================================

    train_root = input("\nTrain tensor path (relative to project root): ").strip()
    test_root  = input("Test tensor path (relative to project root): ").strip()

    # ======================================================
    # Dataset
    # ======================================================

    train_dataset = BinaryTensorDataset(train_root)
    test_dataset  = BinaryTensorDataset(test_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # ======================================================
    # Model
    # ======================================================

    model = BinaryViT(img_size=224).to(device)

    trainer = BinaryClassTrainer(model, train_loader, test_loader)

    trainer.train(
        optim_name="adamw",
        num_epochs=20,
        learning_rate=0.001,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True
    )

    gc.collect()
    torch.cuda.empty_cache()