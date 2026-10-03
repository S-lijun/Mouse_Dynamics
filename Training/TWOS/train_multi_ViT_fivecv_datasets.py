import sys, os, datetime, gc, json, re
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
_ORIG_STDOUT = sys.__stdout__

N_FOLDS = 5
NUM_USERS = 24
IMG_H = 448
IMG_W = 448

# None = folds 0-4. Or pick, e.g. FOLDS = [0, 2, 4].
# A dataset entry can override with "folds": [1, 3].
# TRAIN_FOLDS=0,2 overrides this global list, but not a per-dataset "folds" key.
FOLDS = None

# One dataset finishes its selected folds before the next starts.
# Paths are relative to ImagesTensors/ and must contain folds.npy.
# Comment out any entry you do not want in this run.
DATASETS = [
    {
        "name": "TWOS_XYPlot_chunk_per_user_5fold",
        "tensor": "TWOS/XYPlot_chunk_per_user_5fold/Chong_chunk_per_user",
    },
    {
        "name": "TWOS_XYPlot_chunk_per_user_vxvy_5fold",
        "tensor": "TWOS/XYPlot_chunk_per_user_vxvy_5fold/Chong_chunk_per_user_vxvy",
    },
    {
        "name": "TWOS_SRP_uc_r_gxy_b_vxvy_diag_5fold",
        "tensor": "TWOS/SRP_uc_r_gxy_b_vxvy_diag_5fold/event125",
    },
]


# ======================================================
# Logging
# ======================================================


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = _ORIG_STDOUT
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.flush()
        self.log.close()


# ======================================================
# Imports
# ======================================================

from models.pretrained_VIT_B16_multi import PretrainedViT_B16_Multilabel as insiderThreatViT
from Training.Trainers.fast_multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion

# ======================================================
# Tensor Dataset
# ======================================================


def load_tensor_store(tensor_root):
    print("[Dataset] Loading tensor dataset from:", tensor_root)

    img_path = os.path.join(tensor_root, "images.npy")
    lab_path = os.path.join(tensor_root, "labels.npy")
    fold_path = os.path.join(tensor_root, "folds.npy")
    if not os.path.isfile(fold_path):
        raise FileNotFoundError(
            "Missing folds.npy under %s. Generate with --five-fold."
            % tensor_root
        )

    raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
    n_samples = raw_labels.size // NUM_USERS
    images = np.memmap(
        img_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_samples, 3, IMG_H, IMG_W),
    )
    labels = raw_labels.reshape(n_samples, NUM_USERS)
    sessions = np.load(os.path.join(tensor_root, "sessions.npy"), allow_pickle=True)
    folds = np.asarray(np.memmap(fold_path, dtype=np.uint8, mode="r", shape=(n_samples,)))

    if len(sessions) != n_samples:
        raise RuntimeError(
            "sessions.npy length %d != sample count %d" % (len(sessions), n_samples)
        )

    print("[Dataset] Samples:", n_samples)
    print("[Dataset] Users:", NUM_USERS)
    for fold in range(N_FOLDS):
        print("[Dataset] Fold %d samples: %d" % (fold, int((folds == fold).sum())))

    return images, labels, sessions, folds


class FoldTensorDataset(Dataset):
    """One fold split. .labels is only this split, which the trainer uses for class weights."""

    def __init__(self, images, labels, sessions, indices):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.array(labels[self.indices], dtype=np.uint8, copy=True)
        self.sessions = np.asarray(sessions[self.indices])
        self.num_users = labels.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        src = int(self.indices[idx])
        img = torch.from_numpy(np.array(self.images[src], copy=True)).to(torch.float32).div_(255)
        label = torch.from_numpy(self.labels[idx]).float()
        user = int(self.labels[idx].argmax())
        session_id = "%d_%s" % (user, self.sessions[idx])
        return img, label, session_id


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=256,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


# ======================================================
# Score Collection
# ======================================================


def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)
    return scores, labels, session_ids


def run_score_fusion(scores, labels, session_ids, num_users):
    user_ids = list(range(num_users))
    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

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
                "AUC": float(metrics["AUC"]),
            }
            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = float(np.mean(valid_eers))
        avg_auc = float(np.mean(valid_aucs))
        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")
        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    return result, semantic_user_curve


# ======================================================
# Paths / resume
# ======================================================


def checkpoint_dir_for(name, fold, run_timestamp):
    return (
        Path(project_root)
        / "saved_models"
        / "checkpoints"
        / ("TWOS_ViT_5fold_%s_f%d_%s" % (name, fold, run_timestamp))
    )


def model_path_for(model_dir, name, fold, run_timestamp):
    return model_dir / (
        "multilabel_P1_ViT_5fold_%s_f%d_best_%s.pth" % (name, fold, run_timestamp)
    )


def fold_result_paths(out_dir, fold):
    fold_dir = out_dir / ("fold%d" % fold)
    return fold_dir / "P1_fusion_summary.json", fold_dir / "P1_per_user_results.json"


def fold_is_done(out_dir, fold):
    summary_path, per_user_path = fold_result_paths(out_dir, fold)
    return summary_path.is_file() and per_user_path.is_file()


def load_torch(path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def parse_fold_list(raw, source):
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = list(raw)
    if not parts:
        raise RuntimeError("%s did not select any fold." % source)

    folds = []
    for part in parts:
        fold = int(part)
        if fold < 0 or fold >= N_FOLDS:
            raise RuntimeError(
                "%s has fold %d, valid folds are 0-%d." % (source, fold, N_FOLDS - 1)
            )
        if fold in folds:
            raise RuntimeError("%s repeats fold %d." % (source, fold))
        folds.append(fold)
    return folds


def resolve_folds(dataset_cfg):
    """Per-dataset folds, then TRAIN_FOLDS, then FOLDS, otherwise all 5."""
    if dataset_cfg.get("folds") is not None:
        return parse_fold_list(dataset_cfg["folds"], dataset_cfg["name"])
    env_folds = os.environ.get("TRAIN_FOLDS", "").strip()
    if env_folds:
        return parse_fold_list(env_folds, "TRAIN_FOLDS")
    if FOLDS is not None:
        return parse_fold_list(FOLDS, "FOLDS")
    return list(range(N_FOLDS))


def resolve_batch_timestamp():
    """TRAIN_RESUME=<YYYYMMDD_HHMMSS> continues that batch. Otherwise start a new one."""
    resume_text = os.environ.get("TRAIN_RESUME", "").strip()
    if not resume_text:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", resume_text):
        raise RuntimeError(
            "TRAIN_RESUME must be a batch timestamp YYYYMMDD_HHMMSS, got %s" % resume_text
        )
    print("[CKPT] TRAIN_RESUME=%s" % resume_text)
    return resume_text


def write_mean_summary(out_dir, fold_summaries, selected_folds):
    n_values = fold_summaries[0]["n"]
    mean_summary = {
        "folds": selected_folds,
        "n": n_values,
        "avg_eer": [
            float(np.mean([summary["avg_eer"][i] for summary in fold_summaries]))
            for i in range(len(n_values))
        ],
        "avg_auc": [
            float(np.mean([summary["avg_auc"][i] for summary in fold_summaries]))
            for i in range(len(n_values))
        ],
    }

    print("\n===== Mean Score Fusion over folds %s =====" % selected_folds)
    for n, eer, auc in zip(mean_summary["n"], mean_summary["avg_eer"], mean_summary["avg_auc"]):
        print(f"[n={n:02d}] Mean EER: {eer:.4f} | Mean AUC: {auc:.4f}")

    with open(out_dir / "P1_5fold_mean_summary.json", "w") as f:
        json.dump(mean_summary, f, indent=2)
    return mean_summary


# ======================================================
# One dataset = all 5 folds
# ======================================================


def run_one_dataset(dataset_cfg, run_timestamp, device):
    name = dataset_cfg["name"]
    tensor_folder = dataset_cfg["tensor"]
    selected_folds = resolve_folds(dataset_cfg)

    log_dir = Path(project_root) / "output_logs" / "train_multi_label_p1"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("Protocol1_5fold_ViT_%s_%s.out" % (name, run_timestamp))

    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print("=" * 80)
        print("[INFO] Training Protocol 1 ViT 5-fold - %s - batch %s" % (name, run_timestamp))
        print("[INFO] Tensor folder:", tensor_folder)
        print("[INFO] Selected folds:", selected_folds)
        print("[INFO] Using device:", device)

        tensor_root = Path(project_root) / "ImagesTensors" / tensor_folder
        out_dir = (
            Path(project_root)
            / "Training"
            / "Results"
            / "Protocol1_5fold_ViT"
            / ("%s_%s" % (name, run_timestamp))
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(project_root) / "saved_models"
        model_dir.mkdir(exist_ok=True)

        run_meta_path = out_dir / "run.json"
        meta = {}
        if run_meta_path.is_file():
            with open(run_meta_path) as f:
                meta = json.load(f)
            saved_folder = meta.get("tensor_folder")
            if saved_folder != tensor_folder:
                raise RuntimeError(
                    "This run used tensor folder %s, got %s."
                    % (saved_folder, tensor_folder)
                )
        meta.update({
            "name": name,
            "tensor_folder": tensor_folder,
            "folds": selected_folds,
        })
        with open(run_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        done_flags = {fold: fold_is_done(out_dir, fold) for fold in selected_folds}
        for index, fold in enumerate(selected_folds):
            earlier = selected_folds[:index]
            if done_flags[fold] and not all(done_flags[prev] for prev in earlier):
                raise RuntimeError(
                    "Fold %d is finished but an earlier selected fold is not. Refusing to resume."
                    % fold
                )

        images = labels = sessions = folds = None
        if not all(done_flags.values()):
            images, labels, sessions, folds = load_tensor_store(tensor_root)

        fold_summaries = []

        for fold in selected_folds:
            print("\n" + "=" * 80)
            print("[INFO] %s fold %d: test = fold %d, train = the other folds" % (name, fold, fold))
            print("=" * 80)

            summary_path, per_user_path = fold_result_paths(out_dir, fold)
            if done_flags[fold]:
                with open(summary_path) as f:
                    fold_summaries.append(json.load(f))
                print("[INFO] Fold %d already finished: %s" % (fold, summary_path))
                continue

            test_idx = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            if len(test_idx) == 0 or len(train_idx) == 0:
                raise RuntimeError(
                    "Fold %d has train=%d test=%d" % (fold, len(train_idx), len(test_idx))
                )
            print("[INFO] Train samples: %d | Test samples: %d" % (len(train_idx), len(test_idx)))

            test_dataset = FoldTensorDataset(images, labels, sessions, test_idx)
            test_loader = make_loader(test_dataset, shuffle=False)
            model_path = model_path_for(model_dir, name, fold, run_timestamp)
            resume_file = checkpoint_dir_for(name, fold, run_timestamp) / "latest.pt"
            if not resume_file.is_file():
                resume_file = None

            if model_path.is_file():
                print("[INFO] Fold %d weights exist, running score fusion only: %s" % (fold, model_path))
                best_model = insiderThreatViT(num_users=NUM_USERS).to(device)
                best_model.load_state_dict(load_torch(model_path, device))
                train_loader = trainer = net = None
            else:
                train_dataset = FoldTensorDataset(images, labels, sessions, train_idx)
                train_loader = make_loader(train_dataset, shuffle=True)
                ckpt_dir = checkpoint_dir_for(name, fold, run_timestamp)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if resume_file is None:
                    resume_path = None
                    print("[CKPT] New run checkpoint dir: %s" % ckpt_dir)
                else:
                    resume_path = str(resume_file)
                    print("[CKPT] Resume from: %s" % resume_path)

                net = insiderThreatViT(num_users=NUM_USERS).to(device)
                trainer = MultiLabelTrainer(
                    net=net,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    neg_weight_value=1.0,
                    C_pos=60,
                    C_neg=60,
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

                torch.save(best_model.state_dict(), model_path)
                print("[INFO] Model saved: %s" % model_path)

            scores, score_labels, session_ids = collect_val_scores(best_model, test_loader, device)
            result, semantic_user_curve = run_score_fusion(
                scores, score_labels, session_ids, NUM_USERS
            )

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            with open(per_user_path, "w") as f:
                json.dump(semantic_user_curve, f, indent=2)

            fold_summaries.append(result)
            print("\n[INFO] Fold %d results saved to: %s" % (fold, summary_path.parent))

            del test_loader, best_model
            if train_loader is not None:
                del train_loader, trainer, net
            gc.collect()
            torch.cuda.empty_cache()

        write_mean_summary(out_dir, fold_summaries, selected_folds)
        print("\n[INFO] Results saved to:", out_dir)
        print("[INFO] Protocol 1 ViT 5-fold finished:", name)

        del images, labels, sessions, folds
        gc.collect()
        torch.cuda.empty_cache()

    finally:
        sys.stdout = _ORIG_STDOUT
        logger.close()


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    run_timestamp = resolve_batch_timestamp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("[INFO] TWOS batch ViT 5-fold started")
    print("[INFO] Batch timestamp:", run_timestamp)
    print("[INFO] To resume later: TRAIN_RESUME=%s" % run_timestamp)
    print("[INFO] Datasets:", len(DATASETS))
    print("[INFO] Default folds:", list(range(N_FOLDS)) if FOLDS is None else FOLDS)
    print("[INFO] TRAIN_FOLDS overrides the default. A dataset \"folds\" key overrides both.")
    print("=" * 80)

    for index, dataset_cfg in enumerate(DATASETS, start=1):
        print(
            "\n[INFO] Dataset %d/%d: %s"
            % (index, len(DATASETS), dataset_cfg["name"])
        )
        run_one_dataset(dataset_cfg, run_timestamp, device)
        print("[INFO] Dataset finished: %s" % dataset_cfg["name"])

    print("\n[INFO] ALL DATASETS FINISHED")
