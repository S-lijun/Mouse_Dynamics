# train_multi_CNN_protocol2.py

import sys, os, datetime, re, gc, json
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
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
# Imports
# ======================================================
from models.scratch_CNN_multi import ScratchMultiCNN as insiderThreatCNN
from models.scratch_ViT_multi import ScratchMiniViT_MultiLabel as insiderThreatCNN
from Training.Trainers.multi_class_trainer_protocol2 import MultiLabelTrainerProtocol2 as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion_one

# ======================================================
# Dataset
# ======================================================
class Protocol2MouseDataset(Dataset):
    def __init__(self, split_root, all_users, is_test=False,
                 transform=None, event=60, return_user=False):

        self.samples = []
        self.labels = []
        self.session_ids = []
        self.sample_users = []
        self.transform = transform
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)
        self.return_user = return_user

        if is_test:
            sub_folders = [f"genuine/event{event}", f"imposter/event{event}"]
        else:
            sub_folders = [""]

        for sub in sub_folders:
            base_path = os.path.join(split_root, sub)
            if not os.path.exists(base_path):
                continue

            for user in all_users:
                user_path = os.path.join(base_path, user)
                if not os.path.exists(user_path):
                    continue

                u_idx = self.user2index[user]

                files = []
                for f in os.listdir(user_path):
                    if f.endswith(".png"):
                        m = re.match(r"(session_\d+)-(\d+)\.png", f)
                        if m:
                            files.append((m.group(1), int(m.group(2)), f))

                files.sort(key=lambda x: (x[0], x[1]))

                for sess, idx, f in files:
                    self.samples.append(os.path.join(user_path, f))
                    self.sample_users.append(user)

                    y = torch.zeros(self.num_users)

                    if is_test:
                        if "genuine" in sub:
                            y[u_idx] = 1.0
                    else:
                        y[u_idx] = 1.0

                    self.labels.append(y)
                    self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.return_user:
            return img, self.labels[idx], self.session_ids[idx], self.sample_users[idx]
        else:
            return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Collect Scores
# ======================================================
def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess, users = [], [], [], []

    with torch.no_grad():
        for X, y, s, u in loader:
            logits = model(X.to(device))
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)
            users.extend(u)

    return (
        torch.cat(outs).numpy(),
        torch.cat(labs).numpy(),
        np.asarray(sess),
        np.asarray(users),
    )

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_folder = "XYPlot/event60"
    testing_folder  = "XYPlot_protocol2"
    img_size = 224

    train_root = Path(project_root) / "Images" / training_folder
    test_root  = Path(project_root) / "Images" / testing_folder

    user_list = sorted([u for u in os.listdir(train_root)
                        if os.path.isdir(train_root / u)])
    num_users = len(user_list)

    print(f"[INFO] Detected {num_users} users.")
    print("[INFO] Device:", device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # =========================
    # Dataset
    # =========================

    train_ds = Protocol2MouseDataset(
        train_root, user_list,
        is_test=False,
        transform=transform,
        return_user=False
    )

    # 只用一个 test_loader
    test_ds = Protocol2MouseDataset(
        test_root, user_list,
        is_test=True,
        transform=transform,
        return_user=True   # trainer + evaluation 
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Test samples:  {len(test_ds)}")

    # =========================
    # Training
    # =========================

    net = insiderThreatCNN(num_users=num_users, img_size=img_size).to(device)

    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=test_loader,
        C_pos=60,
        C_neg=60
    )

    print("\n========== Training ==========")
    _, best_model, *_ = trainer.train(
        optim_name="adamw",
        num_epochs=17,
        learning_rate=0.0001,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True
    )

    # =========================
    # Evaluation
    # =========================

    scores, labels, session_ids, users = collect_val_scores(
        best_model, test_loader, device
    )

    print("\n===== Protocol2 Per-User Score Fusion =====")

    result_summary = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol2" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, 31):

        valid_eers = []
        valid_aucs = []

        for u_idx in range(num_users):

            user_name = user_list[u_idx]
            mask = users == user_name

            user_scores   = scores[mask, u_idx]
            user_labels   = labels[mask, u_idx]
            user_sessions = session_ids[mask]

            metrics = multilabel_score_fusion_one(
                user_scores,
                user_labels,
                user_sessions,
                n=n
            )

            semantic_user_curve[user_name][str(n)] = {
                "User": user_name,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"])
            }

            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = np.mean(valid_eers)
        avg_auc = np.mean(valid_aucs)

        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

        result_summary["n"].append(n)
        result_summary["avg_eer"].append(float(avg_eer))
        result_summary["avg_auc"].append(float(avg_auc))

    with open(out_dir / "P2_fusion_summary.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    with open(out_dir / "P2_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    print(f"\n[INFO] Results saved to: {out_dir}")

    gc.collect()
    torch.cuda.empty_cache()