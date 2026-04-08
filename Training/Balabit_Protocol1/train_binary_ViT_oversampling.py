import sys, os, datetime, re, gc, json
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np

# ======================================================
# Env
# ======================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================

log_dir = Path(project_root) / "output_logs" / "train_binary_vit_oversampling"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Binary_training_oversampling_{timestamp}.out"


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
from Training.Trainers.binary_class_trainer_ViT import BinaryClassTrainer
from Training.Score_Fusion.Score_Fusion_Binary import (
    binary_score_fusion,
)

# ======================================================
# Utils
# ======================================================


def parse_session_and_index(filename: str):
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
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

                if f.endswith(".png"):

                    try:
                        sess, idx = parse_session_and_index(f)
                        files.append((sess, idx, f))
                    except Exception:
                        continue

            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:

                self.samples.append(os.path.join(user_path, f))

                label = 1.0 if user == target_user else 0.0
                self.labels.append(label)

                self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img = Image.open(self.samples[idx]).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx]).float(), self.session_ids[idx]


class IndexedSubsetDataset(Dataset):
    """Indexes into an existing BinaryMouseDataset (for oversampled indices)."""

    def __init__(self, base: BinaryMouseDataset, indices):
        self.base = base
        self.indices = list(indices)
        self.labels = [base.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def build_oversample_indices_even(labels):
    """
    Repeat positive indices in round-robin until there are as many positives
    as negatives; keep each negative once. Returns a list of dataset indices
    of length 2 * n_neg (balanced). Order is deterministic (all negs, then
    oversampled pos); shuffle only in the train DataLoader.

    If n_pos == 0, raises. If n_pos >= n_neg, returns None (caller uses full
    dataset without oversampling).
    """
    pos_idx = [i for i, lb in enumerate(labels) if lb > 0.5]
    neg_idx = [i for i, lb in enumerate(labels) if lb <= 0.5]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0:
        raise RuntimeError("No positive samples for target user.")
    if n_pos >= n_neg:
        return None
    # Each original positive appears floor(n_neg/n_pos) or ceil(...) times.
    pos_oversampled = [pos_idx[k % n_pos] for k in range(n_neg)]
    return neg_idx + pos_oversampled


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

            logits = model(X).squeeze(-1)

            scores.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
            labels.extend(y.numpy())
            sessions.extend(s)

    return np.array(scores), np.array(labels), np.array(sessions)


# ======================================================
# Training sampling
# ======================================================

USE_BALANCED_TRAIN_SAMPLER = False

VIT_LOSS_TYPE = "bce"

VIT_DROPOUT = 0.1


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("=" * 80)
    print("Binary ViT Authentication Training (positive oversampling to match negatives)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_folder = input("Enter training folder: ").strip()
    testing_folder = input("Enter testing folder: ").strip()

    img_size = 300

    train_root = Path(project_root) / "Images" / training_folder
    test_root = Path(project_root) / "Images" / testing_folder

    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])

    print("Detected users:", len(user_list))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

        # Train: may use IndexedSubsetDataset (oversampling). Test: always full
        # protocol set, no oversampling, fixed order (see test_loader shuffle=False).
        train_dataset = BinaryMouseDataset(train_root, user, user_list, transform=transform)
        test_dataset = BinaryMouseDataset(test_root, user, user_list, transform=transform)

        n_pos = sum(1 for lb in train_dataset.labels if lb > 0.5)
        n_neg = len(train_dataset) - n_pos

        oversample_idx = build_oversample_indices_even(train_dataset.labels)
        if oversample_idx is not None:
            train_dataset_eff = IndexedSubsetDataset(train_dataset, oversample_idx)
            n_pos_eff = sum(1 for lb in train_dataset_eff.labels if lb > 0.5)
            n_neg_eff = len(train_dataset_eff) - n_pos_eff
            print(
                f"[{user}] oversampling: raw pos={n_pos}, neg={n_neg} -> "
                f"train len={len(train_dataset_eff)} (pos={n_pos_eff}, neg={n_neg_eff}), "
                f"round-robin repeats per positive ~{n_neg / n_pos:.3f}x"
            )
            pos_weight = 1.0
        else:
            train_dataset_eff = train_dataset
            n_pos_eff, n_neg_eff = n_pos, n_neg
            print(
                f"[{user}] no oversampling (pos >= neg): pos={n_pos}, neg={n_neg}"
            )
            pos_weight = float(n_neg) / max(n_pos, 1)

        print(f"[{user}] BCE pos_weight={pos_weight:.2f}")

        if USE_BALANCED_TRAIN_SAMPLER:
            w_pos = 1.0 / max(n_pos_eff, 1)
            w_neg = 1.0 / max(n_neg_eff, 1)
            sample_weights = [
                w_pos if train_dataset_eff.labels[i] > 0.5 else w_neg
                for i in range(len(train_dataset_eff))
            ]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset_eff),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset_eff,
                batch_size=64,
                sampler=sampler,
                shuffle=False,
                num_workers=16,
            )
            print(f"[{user}] train: WeightedRandomSampler (balanced-ish batches)")
        else:
            # Single place training order is randomized (each epoch).
            train_loader = DataLoader(
                train_dataset_eff, batch_size=64, shuffle=True, num_workers=16
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=16,
        )

        net = BinaryViT(
            img_size=img_size, patch_size=15, in_chans=1, dropout=VIT_DROPOUT
        ).to(device)

        trainer = BinaryClassTrainer(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader,
            pos_weight=pos_weight,
        )

        _, best_model, *_ = trainer.train(
            optim_name="adam",
            num_epochs=100,
            learning_rate=0.001,
            lr_milestones=[60, 80],
            learning_rate_decay=0.1,
            loss_type=VIT_LOSS_TYPE,
            verbose=True,
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

    out_dir = Path(project_root) / "Training" / "Results" / "BinaryProtocol1" / f"{timestamp}_oversampling"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1, 6):

        valid_eers = []
        valid_aucs = []

        for u in user_list:

            scores = user_scores[u]
            labels = user_labels[u]
            sessions = user_sessions[u]

            metrics = binary_score_fusion(scores, labels, sessions, n)

            semantic_user_curve[u][str(n)] = {
                "User": u,
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

    with open(out_dir / "P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    print("\nResults saved:", out_dir)

    gc.collect()
    torch.cuda.empty_cache()
