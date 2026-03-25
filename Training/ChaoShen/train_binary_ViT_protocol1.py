import sys, os, datetime, gc, json, re
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
# Natural sort
# ======================================================

def natural_key(s):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]

# ======================================================
# Tensor Dataset
# ======================================================

class TensorBinaryMouseDataset(Dataset):

    def __init__(self, tensor_root, target_user, user_list):

        print("[Dataset] Loading:", tensor_root)

        img_path = os.path.join(tensor_root, "images.npy")
        lab_path = os.path.join(tensor_root, "labels.npy")
        sess_path = os.path.join(tensor_root, "sessions.npy")

        H = 224
        W = 224

        num_users = len(user_list)

        raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
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

        # 🔥 正确 mapping（基于 folder 顺序）
        self.target_user = user_list.index(target_user)

        print("[Dataset] Samples:", N)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = torch.from_numpy(self.images[idx]).float().div_(255)

        multi_label = self.labels[idx]
        label = float(multi_label[self.target_user])

        session_id = self.sessions[idx]

        return img, torch.tensor(label).float(), session_id

# ======================================================
# Score Collection
# ======================================================

def collect_scores(model, loader, device):

    model.eval()

    scores = []
    labels = []
    sessions = []

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

    # 控制分批训练
    start_idx = int(input("Start user index (e.g. 0): "))
    end_idx = int(input("End user index (e.g. 9): "))

    train_root = Path(project_root) / "ImagesTensors" / train_tensor_folder
    test_root  = Path(project_root) / "ImagesTensors" / test_tensor_folder

    # ======================================================
    # Build user list from folder
    # ======================================================

    user_list = sorted(
        [d for d in os.listdir(train_root.parent / train_tensor_folder.replace("_protocol1","")) if d.startswith("user")],
        key=natural_key
    )

    print("\n[User List]")
    print(user_list)

    selected_users = user_list[start_idx:end_idx]

    print("\n[Selected Users]")
    print(selected_users)

    user_scores = {}
    user_labels = {}
    user_sessions = {}

    # ======================================================
    # train per user
    # ======================================================

    for user in selected_users:

        print("\n==============================")
        print("Training model for user:", user)
        print("==============================")

        train_dataset = TensorBinaryMouseDataset(train_root, user, user_list)
        test_dataset  = TensorBinaryMouseDataset(test_root, user, user_list)

        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        net = BinaryViT(img_size=224).to(device)

        trainer = BinaryClassTrainer(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader
        )

        _, best_model, *_ = trainer.train(
            optim_name="adamw",
            num_epochs=17,
            learning_rate=0.0001,
            step_size=5,
            learning_rate_decay=0.1,
            verbose=True
        )

        scores, labels, sessions = collect_scores(best_model, test_loader, device)

        user_scores[user] = scores
        user_labels[user] = labels
        user_sessions[user] = sessions

    # ======================================================
    # Score Fusion
    # ======================================================

    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "BinaryProtocol1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1,31):

        valid_eers = []
        valid_aucs = []

        for user in selected_users:

            scores = user_scores[user]
            labels = user_labels[user]
            sessions = user_sessions[user]

            metrics = binary_score_fusion(scores, labels, sessions, n)

            semantic_user_curve[user][str(n)] = {
                "User": user,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"])
            }

            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = np.mean(valid_eers)
        avg_auc = np.mean(valid_aucs)

        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    with open(out_dir / "P1_fusion_summary.json","w") as f:
        json.dump(result,f,indent=2)

    with open(out_dir / "P1_per_user_results.json","w") as f:
        json.dump(semantic_user_curve,f,indent=2)

    print("\nResults saved:", out_dir)

    gc.collect()
    torch.cuda.empty_cache()