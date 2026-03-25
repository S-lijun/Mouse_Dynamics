# -*- coding: utf-8 -*-

import sys, os, datetime, gc, json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ======================================================
# Env
# ======================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================

log_dir = Path(project_root) / "output_logs" / "train_binary_vit_tensor"
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
# Imports
# ======================================================

from models.scratch_VIT import BinaryViT
from Training.Trainers.fast_binary_class_trainer_ViT import BinaryClassTrainer
from Training.Score_Fusion.Score_Fusion_Binary import binary_score_fusion

# ======================================================
# Dataset
# ======================================================

class BinaryTensorDataset(Dataset):

    def __init__(self, root):

        img_path = os.path.join(root, "images.npy")
        lab_path = os.path.join(root, "labels.npy")
        ses_path = os.path.join(root, "sessions.npy")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = input("Train tensor path: ")
    test_root = input("Test tensor path: ")

    train_dataset = BinaryTensorDataset(train_root)
    test_dataset = BinaryTensorDataset(test_root)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

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