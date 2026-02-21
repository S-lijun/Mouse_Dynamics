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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================
log_dir = Path(project_root) / "output_logs" / "train_multi_label"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"CNN_training_{timestamp}.out"

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

#from models.scratch_CNN_multi import ScratchMultiCNN as insiderThreatViT
from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatViT
from Training.Trainers.multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
)

# ======================================================
# Utils
# ======================================================
def parse_session_and_index(filename: str):
    """
    session_2144641057-32.png
    -> ("session_2144641057", 32)
    """
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))

# ======================================================
# Dataset (image, label, session_id)  FIXED ORDERING
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

            files = []
            for f in os.listdir(user_path):
                if f.endswith(".png"):
                    sess, idx = parse_session_and_index(f)
                    files.append((sess, idx, f))

            # sorted by (session_id, time_index)
            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:
                path = os.path.join(user_path, f)
                self.samples.append(path)

                y = torch.zeros(self.num_users)
                y[self.user2index[user]] = 1.0
                self.labels.append(y)

                self.session_ids.append(sess)

        print(f"[INFO] Loaded {len(self.samples)} samples from {len(all_users)} users.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Session-aware K-Fold (per-user, without replacement)
# ======================================================
def build_kfold_session_splits(dataset, user_list, k, seed=42):
    random.seed(seed)

    user_session_indices = {u: defaultdict(list) for u in user_list}

    for idx, path in enumerate(dataset.samples):
        user = os.path.basename(os.path.dirname(path))
        sess, _ = parse_session_and_index(os.path.basename(path))
        user_session_indices[user][sess].append(idx)

    min_sessions = min(len(sess_dict) for sess_dict in user_session_indices.values())
    if k > min_sessions:
        raise ValueError(
            f"K={k} is larger than the minimum number of sessions per user ({min_sessions})."
        )

    user_shuffled_sessions = {}
    for u, sess_dict in user_session_indices.items():
        sess_list = list(sess_dict.keys())
        random.shuffle(sess_list)
        user_shuffled_sessions[u] = sess_list

    folds = []

    for fold_id in range(k):
        train_idx, test_idx = [], []
        print(f"\n[Fold {fold_id}] Test sessions:")

        for u in user_list:
            test_sess = user_shuffled_sessions[u][fold_id]
            print(f"  {u}: {test_sess}")

            for sess, idxs in user_session_indices[u].items():
                if sess == test_sess:
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
    print("=" * 80)

    Images = [
        "pixel_vs_chunk/event300","pixel_vs_chunk/event120","pixel_vs_chunk/event60","pixel_vs_chunk/event30"
        ]
    
    Images = [
    "Chunk/Balabit_chunks_XY_black_white/event60","Chunk/Balabit_chunks_XY_black_white_cdf/training/event60"
    ]
    
    #ImagesSize = [1002,634,448,317]
    ImagesSize = [224,224]

    C_pos = 60
    C_neg = 60
    K_FOLD = 1

    cv_root = Path(project_root) / "Training" / "Results" / f"CV_{K_FOLD}_fold" / timestamp
    ImagesSizeIndex = 0
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

        transform = transforms.Compose([
            transforms.Resize((ImagesSize[ImagesSizeIndex], ImagesSize[ImagesSizeIndex])), # -> 448 sizes
            transforms.ToTensor()
        ])

        dataset = RawMouseDataset(image_dir, user_list, transform)
        folds = build_kfold_session_splits(dataset, user_list, k=K_FOLD)

        out_dir = cv_root / images.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fold_id, fold in enumerate(folds):
            print(f"\n========== Fold {fold_id+1}/{K_FOLD} ==========")

            train_ds = Subset(dataset, fold["train"])
            test_ds  = Subset(dataset, fold["test"])

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
            test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

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
                learning_rate=0.0001,
                step_size=5,
                learning_rate_decay=0.1,
                verbose=True
            )

            # ================= Save Best Model =================
            model_dir = Path(project_root) / "saved_models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"multilabel_CNN_{timestamp}.pth"
            torch.save(best_model.state_dict(), model_path)
            print(f"[INFO] Best model saved to: {model_path}")

            SEMANTIC_USER_LIST = [
                "user12", "user15", "user16", "user20", "user21",
                "user23", "user29", "user35", "user7", "user9"
            ]

            scores, labels, session_ids, _ = collect_val_scores(
                best_model, test_loader, device
            )

            user_ids = list(range(num_users))
            result = {"n": [], "avg_eer": [], "avg_auc": []}

            semantic_user_curve = defaultdict(dict)

            print("\n===== Score Fusion Curve =====")
            for n in range(1, 101):
                res = multilabel_score_fusion(
                    scores, labels, session_ids, user_ids, n
                )

                    # -------- per-user (semantic) --------
                for col_key, metrics in res.items():
                    col = int(col_key.replace("user", ""))     # 0 ~ 9
                    real_user = SEMANTIC_USER_LIST[col]        # "user12" ...

                    semantic_user_curve[real_user][str(n)] = {
                        "User": real_user,
                        "n": n,
                        "EER": float(metrics["EER"]),
                        "AUC": float(metrics["AUC"])
                    }
                
                avg_eer = np.mean([v["EER"] for v in res.values()])
                avg_auc = np.mean([v["AUC"] for v in res.values()])

                print(f"[n={n:02d}]  Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

                result["n"].append(n)
                result["avg_eer"].append(avg_eer)
                result["avg_auc"].append(avg_auc)

            result_path = out_dir / f"MultiViT_score_fusion_{timestamp}_fold_{fold_id}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            per_user_path = out_dir / f"MultiViT_score_fusion_{timestamp}_fold_{fold_id}_per_user.json"
            with open(per_user_path, "w") as f:
                json.dump(semantic_user_curve, f, indent=2)

            print(f"[INFO] Per-user score fusion saved to: {per_user_path}")
            print(f"[INFO] Score fusion results saved to: {result_path}")
        

            gc.collect()
            torch.cuda.empty_cache()

            print("[INFO] CUDA memory allocated:",
                torch.cuda.memory_allocated())
        ImagesSizeIndex += 1

    print("\n[INFO] All folds finished.")
