import sys, os, datetime, re, gc, json
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

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

log_dir = Path(project_root) / "output_logs" / "train_binary_vit"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Binary_training_{timestamp}.out"

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
#from models.scratch_ViT_ import BinaryViT
from Training.Trainers.binary_class_trainer_ViT import BinaryClassTrainer
from Training.Score_Fusion.Score_Fusion_Binary import (
    binary_score_fusion
)

# ======================================================
# Utils
# ======================================================

def parse_session_and_index(filename: str):
    m = re.match(r"(session_\d+)-(\d+)\.npy", filename) # tensors
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))

# ======================================================
# Dataset
# ======================================================

class BinaryMouseDataset(Dataset):

    def __init__(self, root, target_user, all_users, transform=None):

        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform

        for user in all_users:

            user_path = os.path.join(root, user)
            if not os.path.exists(user_path):
                continue

            files = []

            for f in os.listdir(user_path):

                if f.endswith(".npy"): # tensors files

                    try:
                        sess, idx = parse_session_and_index(f)
                        files.append((sess, idx, f))
                    except:
                        continue

            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:

                self.samples.append(os.path.join(user_path, f))

                label = 1.0 if user == target_user else 0.0
                self.labels.append(label)

                self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)
    '''
    def __getitem__(self, idx):

        img = Image.open(self.samples[idx]).convert("RGB") # 3 channels
        #img = Image.open(self.samples[idx]).convert("L") # 1 channel

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx]).float(), self.session_ids[idx]
    '''
    def __getitem__(self, idx):

        # load float32
        rp = np.load(self.samples[idx])   # [H, W]

        # 转 tensor
        img = torch.from_numpy(rp).float()

        # 加 channel 维度
        img = img.unsqueeze(0)   # [1, H, W]

        return img, torch.tensor(self.labels[idx]).float(), self.session_ids[idx]

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

            X = X.to(device)

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
    print("Binary ViT Authentication Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_folder = input("Enter training folder: ").strip()
    testing_folder = input("Enter testing folder: ").strip()

    img_size = 300

    #train_root = Path(project_root) / "Images" / training_folder
    #test_root  = Path(project_root) / "Images" / testing_folder

    train_root = Path(project_root) / "ImagesTensors" / training_folder
    test_root  = Path(project_root) / "ImagesTensors" / testing_folder

    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])

    print("Detected users:", len(user_list))

    '''
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    '''


    user_scores = {}
    user_labels = {}
    user_sessions = {}

    # ======================================================
    # Train one model per user
    # ======================================================

    for user in user_list:

        print("\n==============================")
        print("Training model for user:", user)
        print("==============================")

        #train_dataset = BinaryMouseDataset(train_root, user, user_list, transform)
        #test_dataset  = BinaryMouseDataset(test_root, user, user_list, transform)

        train_dataset = BinaryMouseDataset(train_root, user, user_list)
        test_dataset  = BinaryMouseDataset(test_root, user, user_list)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

        net = BinaryViT(img_size=img_size, patch_size=15, in_chans=1).to(device)

        trainer = BinaryClassTrainer(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader
        )

        _, best_model, *_ = trainer.train(
            optim_name="adam",
            num_epochs=100,
            learning_rate=0.001,
            step_size=60,
            learning_rate_decay=0.1,
            verbose=True
        )

        scores, labels, sessions = collect_scores(best_model, test_loader, device)

        print(f"\n===== Score Fusion Curve for {user} =====")

        for n in range(1, 2):

            metrics = binary_score_fusion(scores, labels, sessions, n)

            print(f"[n={n:02d}] EER: {metrics['EER']:.4f} | AUC: {metrics['AUC']:.4f}")

        user_scores[user] = scores
        user_labels[user] = labels
        user_sessions[user] = sessions

    # ======================================================
    # Score Fusion Curve
    # ======================================================

    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "BinaryProtocol1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1, 2):

        valid_eers = []
        valid_aucs = []

        for user in user_list:

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

    # ======================================================
    # Save JSON
    # ======================================================

    with open(out_dir / "P1_fusion_summary.json","w") as f:
        json.dump(result,f,indent=2)

    with open(out_dir / "P1_per_user_results.json","w") as f:
        json.dump(semantic_user_curve,f,indent=2)

    print("\nResults saved:", out_dir)

    gc.collect()
    torch.cuda.empty_cache()