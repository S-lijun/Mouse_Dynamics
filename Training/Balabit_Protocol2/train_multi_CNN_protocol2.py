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
# Logging
# ======================================================
log_dir = Path(project_root) / "output_logs" / "train_multi_label_p2"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Protocol2_training_{timestamp}.out"

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
from models.scratch_CNN_multi import ScratchMultiCNN as insiderThreatCNN
from Training.Trainers.multi_class_trainer_82 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion

# ======================================================
# Dataset
# ======================================================
class Protocol2MouseDataset(Dataset):
    def __init__(self, split_root, all_users, is_test=False, transform=None,event = 60):
        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        # 测试集多了一层 event60
        sub_folders = [f"genuine/{event}", f"imposter/{event}"] if is_test else [""]

        for sub in sub_folders:
            base_path = os.path.join(split_root, sub)
            if not os.path.exists(base_path): continue

            for user in all_users:
                user_path = os.path.join(base_path, user)
                if not os.path.exists(user_path): continue

                u_idx = self.user2index[user]
                files = []
                for f in os.listdir(user_path):
                    if f.endswith(".png"):
                        # 解析文件名：session_001-0.png
                        m = re.match(r"(session_\d+)-(\d+)\.png", f)
                        if m:
                            files.append((m.group(1), int(m.group(2)), f))
                
                # 排序保证 session 连续且索引递增，确保相邻图像融合
                files.sort(key=lambda x: (x[0], x[1]))

                for sess, idx, f in files:
                    self.samples.append(os.path.join(user_path, f))
                    y = torch.zeros(self.num_users)
                    # 只有在 genuine 路径下，该样本对该用户才是 True (1.0)
                    if "genuine" in sub:
                        y[u_idx] = 1.0
                    self.labels.append(y)
                    self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.session_ids[idx]

def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []
    with torch.no_grad():
        for X, y, s in loader:
            logits = model(X.to(device))
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)
    return torch.cat(outs).numpy(), torch.cat(labs).numpy(), np.asarray(sess)

# ======================================================
# Main Logic
# ======================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path 配置
    training_folder = "SRP_224/event60"
    testing_folder  = "SRP_224_protocol2" # genuine/imposter 会在 Dataset 里拼接

    img_size = 224
    train_root = Path(project_root) / "Images" / training_folder
    test_root  = Path(project_root) / "Images" / testing_folder
    
    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])
    num_users = len(user_list)
    print(f"[INFO] Detected {num_users} users.")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor()])

    train_ds = Protocol2MouseDataset(train_root, user_list, is_test=False, transform=transform)
    test_ds  = Protocol2MouseDataset(test_root, user_list, is_test=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # 1. 训练
    net = insiderThreatCNN(num_users=num_users, image_size=img_size).to(device)
    trainer = MultiLabelTrainer(net=net, train_loader=train_loader, val_loader=test_loader, C_pos=60, C_neg=60)
    
    print("\n========== Training Execution ==========")
    _, best_model, *_ = trainer.train(num_epochs=17, learning_rate=0.0001)

    # 2. 评估
    scores, labels, session_ids = collect_val_scores(best_model, test_loader, device)

    result_summary = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)
    
    out_dir = Path(project_root) / "Training" / "Results" / "Protocol2" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 2 Score Fusion Evaluation =====")
    user_ids = list(range(num_users))

    for n in range(1, 31):
        # 按 Session 分组融合，逻辑与 P1 保持一致
        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)
        
        valid_eers, valid_aucs = [], []
        for col_key, metrics in res.items():
            u_idx = int(col_key.replace("user", ""))
            real_user_name = user_list[u_idx]
            
            semantic_user_curve[real_user_name][str(n)] = {
                "User": real_user_name,
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
        result_summary["avg_eer"].append(avg_eer)
        result_summary["avg_auc"].append(avg_auc)

    # 3. 保存双 JSON 结果
    with open(out_dir / "P2_fusion_summary.json", "w") as f:
        json.dump(result_summary, f, indent=2)
    with open(out_dir / "P2_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {out_dir}")
    gc.collect()
    torch.cuda.empty_cache()