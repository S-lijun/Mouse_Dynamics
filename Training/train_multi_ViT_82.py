# train_multi_ViT.py
import sys, os, datetime, random, re, gc, json
from collections import defaultdict
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

# ======================================================
# Env / Path
# ======================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================
log_dir = Path(project_root) / "output_logs" / "train_multi_label"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"ViT_training_{timestamp}.out"

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
# Imports (Model / Trainer / Score Fusion)
# ======================================================
from models.scratch_ViT_multi import ScratchMiniViT_MultiLabel as insiderThreatViT
from Training.Trainers.multi_class_trainer_ViT_test import MultiLabelTrainerViT as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
)

# ======================================================
# Utils
# ======================================================
def parse_session_id(img_path: str) -> str:
    name = os.path.basename(img_path)
    m = re.match(r"(session_\d+)-\d+\.png", name)
    if m is None:
        raise RuntimeError(f"Bad filename: {name}")
    return m.group(1)

# ======================================================
# Dataset (image, label, session_id)
# ======================================================
class RawMouseDataset(Dataset):
    def __init__(self, root_dir, all_users, transform=None):
        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform

        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        for user in all_users:
            user_path = os.path.join(root_dir, user)
            for f in sorted(os.listdir(user_path)):
                if not f.endswith(".png"):
                    continue

                path = os.path.join(user_path, f)
                self.samples.append(path)

                y = torch.zeros(self.num_users)
                y[self.user2index[user]] = 1.0
                self.labels.append(y)

                self.session_ids.append(parse_session_id(path))

        print(f"[INFO] Loaded {len(self.samples)} samples from {len(all_users)} users.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Session-aware K-Fold (LOSO when K=1)
# ======================================================
def build_kfold_session_splits(dataset, user_list, k=1, seed=42):
    random.seed(seed)
    user_session_indices = {u: defaultdict(list) for u in user_list}

    for idx, path in enumerate(dataset.samples):
        user = os.path.basename(os.path.dirname(path))
        sess = parse_session_id(path)
        user_session_indices[user][sess].append(idx)

    folds = []
    for _ in range(k):
        train_idx, test_idx = [], []
        for u, sess_dict in user_session_indices.items():
            test_sess = sorted(sess_dict.keys())[0]  # LOSO
            for s, idxs in sess_dict.items():
                if s == test_sess:
                    test_idx.extend(idxs)
                else:
                    train_idx.extend(idxs)
        folds.append({"train": train_idx, "test": test_idx})

    return folds

# ======================================================
# Collect scores
# ======================================================
def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device)
            logits = model(X)
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)

    thresholds = []
    for u in range(labels.shape[1]):
        _, _, thr = calculate_eer(labels[:, u], scores[:, u])
        thresholds.append(thr)

    return scores, labels, session_ids, np.asarray(thresholds)

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    print("=" * 80)
    print("[INFO] Training Start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    print("[INFO] Scratch ViT (single C=60)")
    print("=" * 80)

    Images = [
        "Chunk/Balabit_chunks_XY_black_white/event300"
    ]

    C_pos = 60
    C_neg = 60
    K_FOLD = 1

    cv_root = (
        Path(project_root)
        / "Training"
        / "Results"
        / f"CV_{K_FOLD}_fold"
        / timestamp
    )

    for images in Images:
        print("\n" + "=" * 80)
        print(f"[DATASET] {images}")
        print("=" * 80)

        image_dir = os.path.join(project_root, "Images", images)

        user_list = sorted([
            u for u in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, u))
        ])
        num_users = len(user_list)
        print(f"[INFO] Users ({num_users}): {user_list}")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RawMouseDataset(image_dir, user_list, transform)
        folds = build_kfold_session_splits(dataset, user_list, k=K_FOLD)

        out_dir = cv_root / images.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fold_id, fold in enumerate(folds):

            print(f"\n========== Fold {fold_id+1}/1 ==========")

            train_ds = Subset(dataset, fold["train"])
            test_ds  = Subset(dataset, fold["test"])

            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
            test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

            net = insiderThreatViT(num_users=num_users).to(device)

            trainer = MultiLabelTrainer(
                net=net,
                train_loader=train_loader,
                val_loader=test_loader,
                neg_weight_value=1.0,
                C_pos=C_pos,
                C_neg=C_neg
            )

            _, best_model, *_ = trainer.train(
                optim_name="adamw",
                num_epochs=17,
                learning_rate=1e-4,
                step_size=5,
                learning_rate_decay=0.1,
                verbose=True
            )

            # ================= Save Best Model =================
            model_dir = Path(project_root) / "saved_models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"multilabel_ViT_{timestamp}.pth"
            torch.save(best_model.state_dict(), model_path)
            print(f"[INFO] Best model saved to: {model_path}")

            # ================= Score Fusion =================
            scores, labels, session_ids, thresholds = collect_val_scores(
                best_model, test_loader, device
            )

            user_ids = list(range(num_users))
            result = {"n": [], "avg_eer": [], "avg_auc": []}

            print("\n===== Score Fusion Curve =====")
            for n in range(1, 50):
                res = multilabel_score_fusion(
                    scores,
                    labels,
                    session_ids,
                    thresholds,
                    user_ids,
                    n
                )
                avg_eer = np.mean([v["EER"] for v in res.values()])
                avg_auc = np.mean([v["AUC"] for v in res.values()])

                print(f"[n={n:02d}]  Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

                result["n"].append(n)
                result["avg_eer"].append(avg_eer)
                result["avg_auc"].append(avg_auc)

            result_path = out_dir / f"score_fusion_{timestamp}_fold_{fold_id}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"[INFO] Score fusion results saved to: {result_path}")

            del net, trainer, train_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()

            print("[INFO] CUDA memory allocated:",
                  torch.cuda.memory_allocated())

    print("\n[INFO] All folds finished.")
