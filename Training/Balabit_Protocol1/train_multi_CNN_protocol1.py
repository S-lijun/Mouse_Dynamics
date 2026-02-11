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
    # 适配文件名格式：session_0041905381-0.png
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))

# ======================================================
# Dataset (已包含时序自然排序)
# ======================================================
class Protocol1MouseDataset(Dataset):
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

            # --- 关键：自然排序 (先按 session 字符串排，再按 chunk 序号数字排) ---
            # 这保证了 Session 内图片的连续性，规避了 glob 乱序问题
            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:
                self.samples.append(os.path.join(user_path, f))
                y = torch.zeros(self.num_users)
                y[self.user2index[user]] = 1.0
                self.labels.append(y)
                # 记录 Session ID，Fusion 阶段将基于此进行物理隔离
                self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 返回 session_id (s) 供评估使用
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Score Collection
# ======================================================
def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device)
            logits = model(X)
            # 使用 Sigmoid 获取 0-1 之间的分数值
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
    
    # 路径配置
    training_folder = "SRP_angle/event60"
    testing_folder  = "SRP_angle_protocol1/event60" 
    
    img_size = 448
    C_pos, C_neg = 60, 60
    
    train_root = Path(project_root) / "Images" / training_folder
    test_root  = Path(project_root) / "Images" / testing_folder
    
    # 获取用户列表
    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])
    num_users = len(user_list)
    print(f"[INFO] Detected {num_users} users.")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 1. 加载数据集 (内部已完成自然排序)
    train_dataset = Protocol1MouseDataset(train_root, user_list, transform)
    test_dataset  = Protocol1MouseDataset(test_root, user_list, transform)

    # shuffle=True 只用于训练集；测试集必须 shuffle=False 以保持 Dataset 里的时序
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

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

    # 4. 保存最佳模型
    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"multilabel_P1_best_{timestamp}.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")

    # 5. Score Fusion (时序相邻融合 + Session 物理隔离)
    SEMANTIC_USER_LIST = user_list 
    scores, labels, session_ids = collect_val_scores(best_model, test_loader, device)

    user_ids = list(range(num_users))
    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")
    for n in range(1, 31):
        # 这里的 session_ids 确保了不同 session 的分数永远不会被平均
        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)
        
        valid_eers = []
        valid_aucs = []

        for col_key, metrics in res.items():
            col = int(col_key.replace("user", "")) 
            real_user = SEMANTIC_USER_LIST[col]
            
            # 记录分用户结果
            semantic_user_curve[real_user][str(n)] = {
                "User": real_user,
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

    # 6. 保存评估 JSON
    with open(out_dir / f"P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / f"P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {out_dir}")
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Protocol 1 Finished.")