# train_multi_ViT_balabit_p1.py
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
log_dir = Path(project_root) / "output_logs" / "train_multi_label_p1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Protocol1_training_{timestamp}.out"

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
from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatViT
from Training.Trainers.multi_class_trainer_82 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
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
# Dataset (image, label, session_id)
# ======================================================
class Protocol1MouseDataset(Dataset):
    """
    Dataset that loads from a specific split_path (training or test)
    """
    def __init__(self, split_root, all_users, transform=None):
        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        for user in all_users:
            user_path = os.path.join(split_root, user)
            if not os.path.exists(user_path):
                continue

            files = []
            for f in os.listdir(user_path):
                if f.endswith(".png"):
                    try:
                        sess, idx = parse_session_and_index(f)
                        files.append((sess, idx, f))
                    except: continue

            # Sorted for consistency
            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:
                self.samples.append(os.path.join(user_path, f))
                y = torch.zeros(self.num_users)
                y[self.user2index[user]] = 1.0
                self.labels.append(y)
                self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Score Collection
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
    return scores, labels, session_ids

# ======================================================
# Main Logic
# ======================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"[INFO] Training Protocol 1 - Started at {timestamp}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    
    # 定义训练和测试的具体子文件夹路径
    # 按照您的要求：一个文件夹作为训练，另一个文件夹作为测试
    training_folder = "pixel_vs_chunk/event60"
    testing_folder  = "protocol1/event60" 
    
    img_size = 448
    C_pos, C_neg = 60, 60
    
    # 确定 User List (必须保证 Train/Test 的 Label 索引一致)
    train_root = Path(project_root) / "Images" / training_folder
    test_root  = Path(project_root) / "Images" / testing_folder
    
    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])
    num_users = len(user_list)
    print(f"[INFO] Detected {num_users} users: {user_list}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 1. 加载数据集
    train_dataset = Protocol1MouseDataset(train_root, user_list, transform)
    test_dataset  = Protocol1MouseDataset(test_root, user_list, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # 2. 初始化模型
    net = insiderThreatViT(num_users=num_users).to(device)

    # 3. 训练
    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=test_loader,
        neg_weight_value=1.0,
        C_pos=C_pos,
        C_neg=C_neg
    )

    print("\n========== Training Execution ==========")
    _, best_model, *_ = trainer.train(
        optim_name="adamw",
        num_epochs=17,
        learning_rate=0.0001,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True
    )

    # 4. 保存模型
    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"multilabel_P1_best_{timestamp}.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")

    # 5. Score Fusion (Protocol 1 评估)
    SEMANTIC_USER_LIST = user_list # 动态使用当前检测到的用户列表
    scores, labels, session_ids = collect_val_scores(best_model, test_loader, device)

    user_ids = list(range(num_users))
    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")
    for n in range(1, 31):
        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)
        
        for col_key, metrics in res.items():
            col = int(col_key.replace("user", "")) 
            real_user = SEMANTIC_USER_LIST[col]

            semantic_user_curve[real_user][str(n)] = {
                "User": real_user,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"])
            }
        
        avg_eer = np.mean([v["EER"] for v in res.values()])
        avg_auc = np.mean([v["AUC"] for v in res.values()])

    
        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    # 6. 保存结果
    per_user_path = out_dir / f"P1_per_user_results.json"
    result_path =  out_dir / f"P1_fusion_summary.json"
    
    with open(out_dir / f"P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / f"P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)
    
    print(f"[INFO] Per-user score fusion saved to: {per_user_path}")
    print(f"[INFO] Score fusion results saved to: {result_path}")

    gc.collect()
    torch.cuda.empty_cache()
    print("\n[INFO] Protocol 1 Training and Evaluation Finished.")