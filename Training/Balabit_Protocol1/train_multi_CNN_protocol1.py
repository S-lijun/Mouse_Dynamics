# train_multi_CNN_protocol1.py
import sys, os, datetime, re, gc, json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# ======================================================
# Path / Env
# ======================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================
log_dir = Path(project_root) / "output_logs" / "protocol1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"CNN_protocol1_{timestamp}.out"

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
from models.scratch_CNN_multi import ScratchMultiCNN
from Training.Trainers.multi_class_trainer_82 import MultiLabelTrainerCNN
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion
)

# ======================================================
# Utils
# ======================================================
def parse_session_and_index(filename):
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))

# ======================================================
# Dataset (fixed ordering!)
# ======================================================
class RawMouseDataset(Dataset):
    def __init__(self, root_dir, user_list, transform=None):
        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform

        self.user2idx = {u: i for i, u in enumerate(user_list)}
        self.num_users = len(user_list)

        for user in user_list:
            user_dir = os.path.join(root_dir, user)

            files = []
            for f in os.listdir(user_dir):
                if f.endswith(".png"):
                    sess, idx = parse_session_and_index(f)
                    files.append((sess, idx, f))

            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:
                self.samples.append(os.path.join(user_dir, f))

                y = torch.zeros(self.num_users)
                y[self.user2idx[user]] = 1.0
                self.labels.append(y)

                self.session_ids.append(sess)

        print(f"[INFO] Loaded {len(self.samples)} samples from {len(user_list)} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Collect scores on testing_files
# ======================================================
def collect_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device)
            logits = model(X)
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    return (
        torch.cat(outs).numpy(),
        torch.cat(labs).numpy(),
        np.asarray(sess)
    )

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    event_folders = [
        "event30",
        "event60",
        "event120"
    ]

    for event in event_folders:
        print("\n" + "=" * 80)
        print(f"[EVENT] {event}")
        print("=" * 80)

        train_root = Path(project_root) / "Images" / "Balabit_chunks_XY_black_white" / "training" / event
        test_root  = Path(project_root) / "Images" / "Balabit_chunks_XY_black_white" / "testing" / event

        user_list = sorted(os.listdir(train_root))
        num_users = len(user_list)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_ds = RawMouseDataset(train_root, user_list, transform)
        test_ds  = RawMouseDataset(test_root,  user_list, transform)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

        net = ScratchMultiCNN(num_users=num_users).to(device)

        trainer = MultiLabelTrainerCNN(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader,
            neg_weight_value=1.0,
            C_pos=60,
            C_neg=60
        )

        _, best_model, *_ = trainer.train(
            optim_name="sgd",
            num_epochs=20,
            learning_rate=0.01,
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

        scores, labels, session_ids = collect_scores(
            best_model, test_loader, device
        )

        # ======================================================
        # Score Fusion (Protocol 1, session-aware)
        # ======================================================
        user_ids = list(range(num_users))

        result = {
            "n": [],
            "avg_eer": [],
            "avg_auc": []
        }

        print("\n===== Score Fusion Curve (Protocol 1) =====")
        for n in range(1, 101):
            res = multilabel_score_fusion(
                scores=scores,
                labels=labels,
                session_ids=session_ids,
                user_ids=user_ids,
                n=n
            )

            avg_eer = np.mean([v["EER"] for v in res.values()])
            avg_auc = np.mean([v["AUC"] for v in res.values()])

            print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

            result["n"].append(n)
            result["avg_eer"].append(avg_eer)
            result["avg_auc"].append(avg_auc)

        # ======================================================
        # Save results
        # ======================================================
        out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / event
        out_dir.mkdir(parents=True, exist_ok=True)

        result_path = out_dir / f"score_fusion_protocol1_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[INFO] Score fusion results saved to: {result_path}")

        gc.collect()
        torch.cuda.empty_cache()
