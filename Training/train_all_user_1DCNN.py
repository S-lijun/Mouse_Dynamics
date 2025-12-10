import sys, os, datetime, random, gc, time, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# ======================================================
#                   Logging Setup
# ======================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

run_subdir = Path(project_root) / "saved_models" / f"run_{timestamp}"
run_subdir.mkdir(parents=True, exist_ok=True)
log_dir = Path(project_root) / "output_logs" / "train_all_users"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"1DCNN_all_users_{timestamp}.out"

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
print(f"[INFO] Logging training output to: {log_path}")

# ======================================================
#               Import model and trainer
# ======================================================
from models.scratch_1DCNN import Single1DCNN
from Training.Trainers.binary_class_trainer_1DCNN import BinaryClassTrainer1DCNN
from Training.Score_Fusion.Score_Fusion import get_scores, binary_test


# ======================================================
#               Dataset Definition
# ======================================================
class RawMouseDataset(Dataset):
    """
    用于单个 target_user 的数据集:
      - 正类: target_user 的所有 segment
      - 负类: 其他用户的所有 segment
    """
    def __init__(self, root_dir, target_user):
        self.samples = []
        self.labels = []
        self.positive_indices = []
        self.negative_indices = []

        all_users = [f.replace(".npy", "") for f in os.listdir(root_dir) if f.endswith(".npy")]

        # 正类
        pos_path = os.path.join(root_dir, f"{target_user}.npy")
        pos_data = np.load(pos_path, allow_pickle=True)
        start_idx = len(self.samples)
        self.samples += [torch.tensor(seg, dtype=torch.float32) for seg in pos_data]
        self.labels += [1] * len(pos_data)
        self.positive_indices = list(range(start_idx, start_idx + len(pos_data)))

        # 负类
        for u in all_users:
            if u == target_user:
                continue
            neg_path = os.path.join(root_dir, f"{u}.npy")
            neg_data = np.load(neg_path, allow_pickle=True)
            start_idx = len(self.samples)
            self.samples += [torch.tensor(seg, dtype=torch.float32) for seg in neg_data]
            self.labels += [0] * len(neg_data)
            self.negative_indices += list(range(start_idx, start_idx + len(neg_data)))

        # 自动检测 shape
        self.seq_len = self.samples[0].shape[0]
        self.input_dim = self.samples[0].shape[1]
        print(f"[INFO] User {target_user}: seq_len={self.seq_len}, input_dim={self.input_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, torch.tensor(y, dtype=torch.float32)


class OversampledTrainDataset(Dataset):
    """
    Oversample positive samples to balance with negatives.
    """
    def __init__(self, dataset, pos_indices, neg_indices):
        pos_samples = [dataset.samples[i] for i in pos_indices]
        neg_samples = [dataset.samples[i] for i in neg_indices]

        oversample_factor = len(neg_samples) // len(pos_samples)
        remainder = len(neg_samples) % len(pos_samples)
        oversampled_pos = pos_samples * oversample_factor + random.sample(pos_samples, remainder)

        self.samples = oversampled_pos + neg_samples
        self.labels = [1] * len(oversampled_pos) + [0] * len(neg_samples)

        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, torch.tensor(y, dtype=torch.float32)


# ======================================================
#                     Main Loop
# ======================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset_folder = input("Dataset folder (e.g., Balabit_PeakClick/training): ").strip()
    data_root = os.path.join(project_root, "Data", dataset_folder)
    print(f"[INFO] Dataset root: {data_root}")

    user_list = sorted([f.replace(".npy", "") for f in os.listdir(data_root) if f.endswith(".npy")])
    print(f"[INFO] Users: {user_list}")

    global_start = time.perf_counter()
    final_results = {}

    for target_user in user_list:
        print(f"\n========== [TRAIN USER: {target_user}] ==========")
        full_dataset = RawMouseDataset(data_root, target_user)

        # --- 拆分正负索引 ---
        pos_indices = full_dataset.positive_indices
        neg_indices = full_dataset.negative_indices
        pos_train, pos_val = train_test_split(pos_indices, test_size=0.2, random_state=42)
        neg_train, neg_val = train_test_split(neg_indices, test_size=0.2, random_state=42)

        # --- Oversampled train set ---
        train_dataset = OversampledTrainDataset(full_dataset, pos_train, neg_train)
        val_indices = pos_val + neg_val
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        raw_pos_count = len(pos_indices)
        raw_neg_count = len(neg_indices)
        print(f"[INFO] Raw count (pos={raw_pos_count}, neg={raw_neg_count})")

        # --- Model ---
        net = Single1DCNN(input_dim=full_dataset.input_dim, seq_len=full_dataset.seq_len, num_classes=1).to(device)

        # --- Trainer ---
        trainer = BinaryClassTrainer1DCNN(
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_count=raw_pos_count,
            neg_count=raw_neg_count,
            neg_weight_value=raw_neg_count / max(raw_pos_count, 1)
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model, best_model, train_losses, val_losses, train_accs, val_accs, val_eer, val_auc = trainer.train(
            optim_name='sgd',
            num_epochs=100,
            learning_rate=0.01,
            step_size=15,
            learning_rate_decay=0.1,
            acc_frequency=1,
            verbose=True
        )

        # --- Save model ---
        model_filename = f"{target_user}_1DCNN_best_{timestamp}.pth"
        model_path = run_subdir / model_filename
        torch.save(best_model.state_dict(), model_path)
        print(f"[INFO] Best model saved to: {model_path}")

        # --- Score Fusion ---
        val_loader_fusion = DataLoader(val_dataset, batch_size=256, shuffle=False)
        all_outs, all_ys = get_scores(val_loader_fusion, model_path=model_path, device=device, model_class=Single1DCNN)

        user_result_dict = {}
        for n in range(1, 50):
            print(f"\n=== Score Fusion (n={n}) ===")
            results = binary_test(all_outs, all_ys, n=n, user_id=target_user)
            user_result_dict[n] = results

        final_results[target_user] = user_result_dict

        del net, model, best_model, trainer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[INFO] Finished training {target_user} | CUDA mem: {torch.cuda.memory_allocated()}")

    # --- Save all results ---
    results_dir = Path(project_root) / "Training" / "Results" / "train_all_user_1DCNN"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"score_fusion_results_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"[INFO] Score fusion results saved to: {results_path}")

    total_elapsed = time.perf_counter() - global_start
    print(f"\n[INFO] All users done, total time: {total_elapsed/60:.2f} min")
