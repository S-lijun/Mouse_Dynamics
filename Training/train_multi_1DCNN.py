# train_multi_1DCNN.py
import sys, os, datetime, random, gc, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

# --- Logging setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = Path(project_root) / "output_logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "train_multi_label" / f"1DCNN_training_{timestamp}.out"

class TeeLogger(object):
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
print(f"[INFO] Logging training output from: {os.path.basename(__file__)}")
print(f"[INFO] Logging training output to:   {log_path}")

# --- Model and Trainer ---
from models.scratch_1DCNN_multi import Multi1DCNN_MultiLabel as insiderThreatCNN
from Training.Trainers.multi_class_trainer_1DCNN import MultiLabelTrainer1DCNN as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi import multilabel_score_fusion, calculate_eer


# ------------------ collect_val_scores ------------------ #
def collect_val_scores(model, val_loader, device):
    """Return sigmoid scores [N,U], labels [N,U] and per-user EER thresholds."""
    model.eval()
    out_list, y_list = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits = model(X)
            out_list.append(torch.sigmoid(logits).cpu())
            y_list.append(y)
    scores = torch.cat(out_list).numpy()   # [N, U]
    labels = torch.cat(y_list).numpy()     # [N, U]

    thresholds = []
    for u in range(labels.shape[1]):
        _, _, thr = calculate_eer(labels[:, u], scores[:, u])
        thresholds.append(thr)
    return scores, labels, np.array(thresholds, dtype=np.float32)


# ------------------ RawMouseDataset ------------------ #
class RawMouseDataset(Dataset):
    """
    Multi-user dataset for .npy files under Data/.
    Each user's .npy has shape (N_segments, seq_len, input_dim).
    """
    def __init__(self, root_dir, all_users):
        self.samples = []
        self.labels = []
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        seq_lens, feat_dims = [], []

        for user in all_users:
            path = os.path.join(root_dir, f"{user}.npy")
            data = np.load(path, allow_pickle=True)
            seq_lens.append(data.shape[1])
            feat_dims.append(data.shape[2])

            for seg in data:
                self.samples.append(torch.tensor(seg, dtype=torch.float32))
                label = torch.zeros(self.num_users)
                label[self.user2index[user]] = 1.0
                self.labels.append(label)

        # 自动检测 seq_len 和 input_dim
        self.seq_len = seq_lens[0]
        self.input_dim = feat_dims[0]

        print(f"[INFO] Auto-detected input_dim={self.input_dim}, seq_len={self.seq_len}")
        print(f"[INFO] Loaded {len(self.samples)} samples from {len(all_users)} users.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]         

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, y


# ------------------ Main ------------------ #
if __name__ == "__main__":
    print("[INFO] Training Start:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # ==== Dataset path ====
    dataset = "Balabit_PeakClick/training"

    data_root = os.path.join(project_root, "Data", dataset)
    print(f"[INFO] Dataset root: {data_root}")

    user_files = sorted([f for f in os.listdir(data_root) if f.endswith(".npy")])
    user_list = [f.replace(".npy", "") for f in user_files]
    num_users = len(user_list)
    print(f"[INFO] Users ({num_users}): {user_list}")

    # ==== Dataset ====
    dataset = RawMouseDataset(root_dir=data_root, all_users=user_list)
    seq_len, input_dim = dataset.seq_len, dataset.input_dim

    # ==== Split train / val ====
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # ==== Model ====
    net = insiderThreatCNN(input_dim=input_dim, seq_len=seq_len, num_users=num_users).to(device)

    # ==== Trainer ====
    C_pos, C_neg = 40, 40
    print(f"[INFO] C_pos: {C_pos}")
    print(f"[INFO] C_neg: {C_neg}")

    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        neg_weight_value=1.0,
        C_pos=C_pos,
        C_neg=C_neg
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ==== Train ====
    model, best_model, *_ = trainer.train(
        optim_name='sgd',
        num_epochs=100,
        learning_rate=0.01,
        step_size=20,
        learning_rate_decay=0.1,
        acc_frequency=1,
        verbose=True
    )

    # ==== Save best model ====
    saved_model_dir = Path("saved_models")
    saved_model_dir.mkdir(exist_ok=True)
    model_filename = f"multi1DCNN_best_{timestamp}.pth"
    model_path = saved_model_dir / model_filename
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Best model saved to: {model_path}")

    # ==== Validation evaluation ====
    scores, labels, thresholds = collect_val_scores(best_model, val_loader, device)
    num_users = scores.shape[1]
    user_ids = list(range(num_users))

    # ==== Score Fusion ====
    user_result_dict = {"n": [], "avg_eer": [], "avg_auc": []}
    for n in range(1, 50):
        res = multilabel_score_fusion(scores, labels, thresholds, user_ids=user_ids, n=n)
        avg_eer = np.mean([v["EER"] for v in res.values()])
        avg_auc = np.mean([v["AUC"] for v in res.values()])
        print(f"[n={n}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")
        user_result_dict["n"].append(n)
        user_result_dict["avg_eer"].append(avg_eer)
        user_result_dict["avg_auc"].append(avg_auc)

    # ==== Save Results ====
    results_dir = os.path.join(project_root, "Training", "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"train_multi_1DCNN/score_fusion_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(user_result_dict, f, indent=2)
    print(f"[INFO] Score fusion results saved to: {results_path}")

    # ==== Cleanup ====
    del model, trainer, net, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Training finished. CUDA memory allocated:", torch.cuda.memory_allocated())
