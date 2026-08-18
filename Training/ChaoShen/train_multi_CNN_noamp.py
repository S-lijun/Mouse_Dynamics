"""ChaoShen Protocol1 CNN training with AMP forced off (compute1 workaround).

Same as train_multi_CNN.py, but disables torch.cuda.amp in-process so the
shared fast trainer runs FP32. Use this on compute1 if AMP yields NaN /
misaligned address; leave train_multi_CNN.py untouched.
"""
import sys, os, datetime, random, re, gc, json, contextlib
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

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

log_dir = Path(project_root) / "output_logs" / "train_multi_label_p1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Protocol1_training_noamp_{timestamp}.out"


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

from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatCNN
from Training.Trainers.fast_multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Trainers.checkpoint_utils import setup_training_checkpoint
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
)


def _disable_cuda_amp_in_process():
    @contextlib.contextmanager
    def _noop_autocast(*args, **kwargs):
        yield

    _OrigScaler = torch.cuda.amp.GradScaler

    def _DisabledScaler(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["enabled"] = False
        return _OrigScaler(*args, **kwargs)

    torch.cuda.amp.autocast = _noop_autocast
    torch.cuda.amp.GradScaler = _DisabledScaler
    print("[INFO] CUDA AMP disabled (train_multi_CNN_noamp.py)")


# ======================================================
# Tensor Dataset
# ======================================================


class TensorMouseDataset(Dataset):

    def __init__(self, tensor_root):

        print("[Dataset] Loading tensor dataset from:", tensor_root)

        img_path = os.path.join(tensor_root, "images.npy")
        lab_path = os.path.join(tensor_root, "labels.npy")

        num_users = 28
        H = 448
        W = 448

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

        self.sessions = np.load(
            os.path.join(tensor_root, "sessions.npy"),
            allow_pickle=True
        )

        self.num_users = num_users

        print("[Dataset] Samples:", N)
        print("[Dataset] Users:", num_users)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Explicit copy: memmap view + pin_memory can misalign on compute1.
        img = torch.from_numpy(
            np.array(self.images[idx], dtype=np.uint8, copy=True)
        ).to(torch.float32).div_(255).contiguous()

        label = torch.from_numpy(
            np.array(self.labels[idx], dtype=np.float32, copy=True)
        )
        session_id = self.sessions[idx]

        return img, label, session_id


def collect_val_scores(model, loader, device):

    model.eval()

    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")

    with torch.no_grad():

        for X, y, s in loader:

            X = X.to(device, non_blocking=True)
            logits = model(X)
            outs.append(torch.sigmoid(logits.float()).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)

    return scores, labels, session_ids


if __name__ == "__main__":

    print("=" * 80)
    print(f"[INFO] Training Protocol 1 (no AMP) - Started at {timestamp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    _disable_cuda_amp_in_process()
    print("[INFO] Using device:", device)

    train_tensor_folder = input("Enter training tensor folder (relative to ImagesTensors/): ").strip()
    test_tensor_folder = input("Enter testing tensor folder (relative to ImagesTensors/): ").strip()

    ckpt_dir, resume_path = setup_training_checkpoint(
        project_root, timestamp, run_prefix="ChaoShen_CNN_P1_noamp"
    )

    train_root = Path(project_root) / "ImagesTensors" / train_tensor_folder
    test_root = Path(project_root) / "ImagesTensors" / test_tensor_folder

    train_dataset = TensorMouseDataset(train_root)
    test_dataset = TensorMouseDataset(test_root)

    num_users = train_dataset.num_users

    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=20,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    net = insiderThreatCNN(num_users=num_users).to(device)

    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=test_loader,
        neg_weight_value=1.0,
        C_pos=60,
        C_neg=60
    )

    print("\n========== Training Execution ==========")

    _, best_model, *_ = trainer.train(
        optim_name="adamw",
        num_epochs=17,
        learning_rate=0.0001,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=1,
        resume_path=resume_path,
    )

    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"multilabel_P1_best_noamp_{timestamp}.pth"

    torch.save(best_model.state_dict(), model_path)

    print(f"[INFO] Model saved: {model_path}")

    scores, labels, session_ids = collect_val_scores(best_model, test_loader, device)

    user_ids = list(range(num_users))

    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / f"{timestamp}_noamp"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1, 16):

        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)

        valid_eers = []
        valid_aucs = []

        for col_key, metrics in res.items():

            col = int(col_key.replace("user", ""))

            semantic_user_curve[col][str(n)] = {
                "User": col,
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

    with open(out_dir / "P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    print("\n[INFO] Results saved to:", out_dir)

    gc.collect()
    torch.cuda.empty_cache()

    print("[INFO] Protocol 1 (no AMP) Finished.")
