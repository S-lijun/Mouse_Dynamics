# train_multi_ViT.py
import sys, os, datetime, random, re, gc, json
from collections import defaultdict
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# ======================================================
# Logging
# ======================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
# Model / Trainer / Score Fusion
# ======================================================
from models.scratch_ViT_multi import ScratchMiniViT_MultiLabel as insiderThreatViT
from Training.Trainers.multi_class_trainer_ViT import MultiLabelTrainerViT as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi import (
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
# Dataset (returns image, label, session_id)
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
            # ⭐ 保证 session 内 chunk 时间顺序
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
# Session-aware K-Fold Split
# ======================================================
def build_kfold_session_splits(dataset, user_list, k=5, seed=42):
    random.seed(seed)

    user_session_indices = {u: defaultdict(list) for u in user_list}

    for idx, path in enumerate(dataset.samples):
        user = os.path.basename(os.path.dirname(path))
        sess = parse_session_id(path)
        user_session_indices[user][sess].append(idx)

    user_sessions = {}
    for u, d in user_session_indices.items():
        sess = list(d.keys())
        if len(sess) < k:
            raise RuntimeError(f"{u} has only {len(sess)} sessions (<{k})")
        random.shuffle(sess)
        user_sessions[u] = sess[:k]

    folds = []
    for f in range(k):
        train_idx, test_idx = [], []
        for u in user_list:
            test_sess = user_sessions[u][f]
            for s, idxs in user_session_indices[u].items():
                if s == test_sess:
                    test_idx.extend(idxs)
                else:
                    train_idx.extend(idxs)
        folds.append({"train": train_idx, "test": test_idx})

    return folds

# ======================================================
# Collect scores + session ids
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

    scores = torch.cat(outs).numpy()   # [N, U]
    labels = torch.cat(labs).numpy()   # [N, U]
    session_ids = np.asarray(sess)     # [N]

    thresholds = []
    for u in range(labels.shape[1]):
        _, _, thr = calculate_eer(labels[:, u], scores[:, u])
        thresholds.append(thr)

    return scores, labels, session_ids, np.asarray(thresholds)

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    K_FOLDS = 1

    Images = [
        "Chunk/Balabit_chunks_XY_black_white/event300"
    ]

    cv_root = (
        Path(project_root)
        / "Training"
        / "Results"
        / f"CV_{K_FOLDS}_fold"
        / timestamp
    )

    for images in Images:
        image_dir = os.path.join(project_root, "Images", images)
        print(f"\n[DATASET] {images}")

        user_list = sorted([
            u for u in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, u))
        ])
        num_users = len(user_list)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RawMouseDataset(image_dir, user_list, transform)
        folds = build_kfold_session_splits(dataset, user_list, k=K_FOLDS, seed=42)

        image_tag = images.replace("/", "_")
        out_dir = cv_root / image_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for fold_id, fold in enumerate(folds):
            print(f"\n========== Fold {fold_id+1}/{K_FOLDS} ==========")

            train_ds = Subset(dataset, fold["train"])
            test_ds  = Subset(dataset, fold["test"])

            train_loader = DataLoader(
                train_ds, batch_size=128, shuffle=True, num_workers=2
            )
            test_loader = DataLoader(
                test_ds, batch_size=128, shuffle=False, num_workers=2
            )

            net = insiderThreatViT(num_users=num_users).to(device)

            trainer = MultiLabelTrainer(
                net=net,
                train_loader=train_loader,
                val_loader=test_loader,
                neg_weight_value=1.0,
                C_pos=60,
                C_neg=60
            )

            _, best_model, *_ = trainer.train(
                optim_name="adamw",
                num_epochs=17,
                learning_rate=1e-4,
                step_size=5,
                learning_rate_decay=0.1,
                verbose=True
            )

            scores, labels, session_ids, thresholds = collect_val_scores(
                best_model, test_loader, device
            )

            user_ids = list(range(num_users))
            result = {"n": [], "avg_eer": [], "avg_auc": []}

            for n in range(1, 50):
                res = multilabel_score_fusion(
                    scores,
                    labels,
                    session_ids,
                    thresholds,
                    user_ids,
                    n
                )
                result["n"].append(n)
                result["avg_eer"].append(np.mean([v["EER"] for v in res.values()]))
                result["avg_auc"].append(np.mean([v["AUC"] for v in res.values()]))

            with open(out_dir / f"fold_{fold_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            del net, trainer, train_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()

    print("[INFO] All folds finished.")
