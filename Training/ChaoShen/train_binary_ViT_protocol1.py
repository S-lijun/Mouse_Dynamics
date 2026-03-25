# -*- coding: utf-8 -*-

import sys, os, datetime, gc, json
from pathlib import Path
from collections import defaultdict
import re

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
# Utils
# ======================================================

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', string)]

# ======================================================
# Dataset
# ======================================================

class TensorBinaryMouseDataset(Dataset):

    def __init__(self, tensor_root, target_user, num_users):

        print("[Dataset] Loading:", tensor_root)

        img_path = os.path.join(tensor_root, "images.npy")
        lab_path = os.path.join(tensor_root, "labels.npy")
        sess_path = os.path.join(tensor_root, "sessions.npy")

        H, W = 224, 224

        raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
        assert raw_labels.size % num_users == 0

        N = raw_labels.size // num_users

        raw_images = np.memmap(
            img_path,
            dtype=np.uint8,
            mode="r",
            shape=(N, 3, H, W)
        )

        self.images = raw_images
        self.labels = raw_labels.reshape(N, num_users)
        self.sessions = np.load(sess_path, allow_pickle=True)

        # ======================================================
        # 🔥 正确 mapping（关键修复）
        # ======================================================

        parts = Path(tensor_root).parts
        idx = parts.index("ImagesTensors")

        dataset_name = parts[idx + 1]
        representation = parts[idx + 2]

        image_root = Path(project_root) / "Images" / dataset_name / representation

        users = sorted(os.listdir(image_root), key=natural_key)

        self.user_to_idx = {u: i for i, u in enumerate(users)}

        if target_user not in self.user_to_idx:
            raise ValueError(f"{target_user} not found in dataset")

        self.target_user = self.user_to_idx[target_user]

        print("[Dataset] Users order:", users[:10])
        print("[Dataset] Target user:", target_user)
        print("[Dataset] Mapped index:", self.target_user)
        print("[Dataset] Samples:", N)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = torch.from_numpy(self.images[idx]).float().div_(255)

        label = float(self.labels[idx][self.target_user])
        session_id = self.sessions[idx]

        return img, torch.tensor(label).float(), session_id

# ======================================================
# Score Collection
# ======================================================

def collect_scores(model, loader, device):

    model.eval()

    scores, labels, sessions = [], [], []

    with torch.no_grad():

        for X, y, s in loader:

            X = X.to(device, non_blocking=True)
            logits = model(X)

            scores.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y.numpy())
            sessions.extend(s)

    return np.array(scores), np.array(labels), np.array(sessions)

# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("="*80)
    print("Binary ViT Tensor Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_tensor_folder = input("Enter training tensor folder: ").strip()
    test_tensor_folder = input("Enter testing tensor folder: ").strip()

    train_root = Path(project_root) / "ImagesTensors" / train_tensor_folder
    test_root  = Path(project_root) / "ImagesTensors" / test_tensor_folder

    # ======================================================
    # 🔥 自动获取真实 user_list（关键修复）
    # ======================================================

    parts = Path(train_root).parts
    idx = parts.index("ImagesTensors")

    dataset_name = parts[idx + 1]
    representation = parts[idx + 2]

    image_root = Path(project_root) / "Images" / dataset_name / representation

    user_list = sorted(os.listdir(image_root), key=natural_key)

    print("\n[User List]")
    print(user_list)

    num_users = len(user_list)

    user_scores = {}
    user_labels = {}
    user_sessions = {}

    # ======================================================
    # Train per user
    # ======================================================

    for user in user_list:

        print("\n==============================")
        print("Training model for user:", user)
        print("==============================")

        train_dataset = TensorBinaryMouseDataset(train_root, user, num_users)
        test_dataset  = TensorBinaryMouseDataset(test_root, user, num_users)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
        test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

        net = BinaryViT(img_size=224).to(device)

        trainer = BinaryClassTrainer(net, train_loader, test_loader)

        _, best_model, *_ = trainer.train(
            optim_name="adamw",
            num_epochs=20,
            learning_rate=0.0001,
            step_size=5,
            learning_rate_decay=0.1,
            verbose=True
        )

        scores, labels, sessions = collect_scores(best_model, test_loader, device)

        print(f"\n===== Score Fusion Curve for {user} =====")

        for n in range(1, 31):

            metrics = binary_score_fusion(scores, labels, sessions, n)

            print(f"[n={n:02d}] EER: {metrics['EER']:.4f} | AUC: {metrics['AUC']:.4f}")

        user_scores[user] = scores
        user_labels[user] = labels
        user_sessions[user] = sessions

    # ======================================================
    # Global Curve
    # ======================================================

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1, 31):

        valid_eers = []
        valid_aucs = []

        for user in user_list:

            scores = user_scores[user]
            labels = user_labels[user]
            sessions = user_sessions[user]

            metrics = binary_score_fusion(scores, labels, sessions, n)

            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        print(f"[n={n:02d}] Avg EER: {np.mean(valid_eers):.4f} | Avg AUC: {np.mean(valid_aucs):.4f}")

    gc.collect()
    torch.cuda.empty_cache()