import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import sys
import copy
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
'''
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ------------------ Model comparison ------------------ #
def compare_models(model1, model2):
    device = next(model1.parameters()).device
    model2 = model2.to(device)
    return all(torch.equal(p1, p2) for p1, p2 in zip(model1.state_dict().values(), model2.state_dict().values()))

# ------------------ EER calculation ------------------ #
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    except (ValueError, RuntimeError):
        eer = 0.5
        eer_threshold = 0.5

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)
    return eer, auc, eer_threshold

# ------------------ GHM Loss ------------------ #
class GHMBCE(nn.Module):
    def __init__(self, bins=10, epsilon=1e-12):
        super().__init__()
        self.bins = bins
        self.epsilon = epsilon
        self.delta = 1.0 / bins

    def forward(self, logits, targets):
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            g = torch.abs(pred - targets)
            n = logits.numel()

            GD = torch.zeros_like(g)
            for i in range(n):
                gi = g[i]
                close = (torch.abs(g - gi) <= self.delta)
                GD[i] = close.sum() / self.delta

            beta = n / (GD + self.epsilon)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        return (beta * loss).mean()

# ------------------ Trainer ------------------ #
class BinaryClassTrainer:
    def __init__(self, net=None, train_loader=None, val_loader=None,
                 pos_count=None, neg_count=None, neg_weight_value=1.0):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_state = None
        self.best_val_eer = float('inf')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)

        if pos_count is None or neg_count is None:
            raise ValueError("Must provide pos_count and neg_count explicitly!")

        pos_weight_value = neg_count / max(pos_count, 1)
        self.pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(self.device)
        self.neg_weight = torch.tensor([neg_weight_value], dtype=torch.float).to(self.device)
        print(f"[INFO] Using neg_weight = {self.neg_weight.item():.2f} (to control FP)")
        print(f"[INFO] Using pos_weight = {self.pos_weight.item():.2f} (pos={pos_count}, neg={neg_count})")


    def train(self, optim_name='adam', num_epochs=10, learning_rate=1e-3,
              reg=0.0, step_size=1, learning_rate_decay=0.95,
              acc_frequency=1, verbose=False, loss_type="custom"):

        # ------------------ Loss selection ------------------ #
        def custom_loss(logits, labels):
            probs = torch.sigmoid(logits)
            loss = - (labels * torch.log(probs + 1e-6) +
                      self.neg_weight * (1 - labels) * torch.log(1 - probs + 1e-6))
            return loss.mean()

        if loss_type == "bce_logits":
            loss_function = nn.BCEWithLogitsLoss()
        elif loss_type == "ghm":
            loss_function = GHMBCE()
        elif loss_type == "custom":
            loss_function = custom_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        # ------------------ History ------------------ #
        train_losses, val_losses = [], []
        train_acc_history, val_acc_history = [], []
        val_eer_history, val_auc_history = [], []

        # Early stopping
        patience = 15
        patience_counter = 0
        self.best_val_eer = float('inf')

        # Optimizer
        if optim_name.lower() == 'adam':
            optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=0.01)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=learning_rate_decay)

        # ------------------ Epoch Loop ------------------ #
        for epoch in range(num_epochs):
            self.net.train()
            epoch_train_loss = 0
            correct_train, total_train = 0, 0

            for X, y in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                outputs = self.net(X).squeeze(dim=1)
                loss = loss_function(outputs, y)

                if reg > 0:
                    decay_penalty = sum(p.pow(2).sum() for p in self.net.parameters())
                    loss += reg * decay_penalty

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_train += (preds == y).sum().item()
                total_train += y.size(0)

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_acc = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_acc_history.append(train_acc)

            # ------------------ Validation ------------------ #
            self.net.eval()
            epoch_val_loss = 0
            total_val = 0
            val_scores = []
            val_labels = []

            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.net(X).squeeze(dim=1)
                    loss = loss_function(outputs, y)
                    epoch_val_loss += loss.item()
                    total_val += y.size(0)

                    val_scores.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            # ------------------ Metrics ------------------ #
            val_scores = np.array(val_scores)
            val_labels = np.array(val_labels)
            eer, auc, eer_threshold = calculate_eer(val_labels, val_scores)

            preds = (torch.tensor(val_scores) >= eer_threshold).float()
            val_labels_tensor = torch.tensor(val_labels)

            tp = ((val_labels_tensor == 1) & (preds == 1)).sum().item()
            fp = ((val_labels_tensor == 0) & (preds == 1)).sum().item()
            tn = ((val_labels_tensor == 0) & (preds == 0)).sum().item()
            fn = ((val_labels_tensor == 1) & (preds == 0)).sum().item()

            val_acc = (tp + tn) / total_val
            val_acc_history.append(val_acc)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            val_eer_history.append(eer)
            val_auc_history.append(auc)

            # ------------------ Save best ------------------ #
            if eer < self.best_val_eer:
                self.best_val_eer = eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                print(f"New best model saved at epoch {epoch + 1} with EER: {eer:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

            if verbose or (epoch + 1) % acc_frequency == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f} | "
                      f"EER: {eer:.4f}, AUC: {auc:.4f}")

        # ------------------ Best Model ------------------ #
        # ------------------ Best Model ------------------ #
        best_model = self.net.__class__()   
        best_model.load_state_dict(self.best_model_state)
        best_model.to(self.device)


        return self.net, best_model, train_losses, val_losses, train_acc_history, val_acc_history, val_eer_history, val_auc_history
'''
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ============================================================
# EER
# ============================================================

def calculate_eer(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    try:
        eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = thresholds[np.nanargmin(np.abs((1 - tpr) - fpr))]
    except:
        eer = np.nan
        eer_threshold = np.nan

    return eer, auc, eer_threshold


# ============================================================
# GHM BCE (fast: g_i = |σ(z_i) − y_i|, one forward per step)
# ============================================================


class GHMBCE(nn.Module):

    def __init__(self, delta=0.1, pos_weight=1.0):
        super().__init__()
        self.delta = delta
        pw = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(())
        self.register_buffer("pos_weight", pw)

    def forward(self, logits, targets):
        y = targets.view(-1).float()
        logits = logits.view(-1)

        pred = torch.sigmoid(logits)
        g = torch.abs(pred.detach() - y)

        n = g.numel()
        g_flat = g.reshape(-1)

        with torch.no_grad():
            diff = torch.abs(g_flat.unsqueeze(0) - g_flat.unsqueeze(1))
            mask = (diff <= self.delta).to(g_flat.dtype)
            GD = mask.sum(dim=1) / self.delta
            beta = n / (GD + 1e-12)

        pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=pw
        )
        weighted = (beta * per_elem).mean()
        pure_mean = per_elem.mean()
        return weighted, pure_mean

# ============================================================
# Trainer
# ============================================================

class BinaryClassTrainer:

    def __init__(self, net, train_loader, val_loader, pos_weight = 10):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_model_state = None
        self.best_val_eer = float("inf")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(self.device)
        print(f"[BinaryClassTrainer] GHMBCE BCE pos_weight={pos_weight} (positive class loss scale)")

    def train(
        self,
        optim_name="adamw",
        num_epochs=17,
        learning_rate=1e-4,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True,
    ):

        loss_function = GHMBCE(pos_weight=self.pos_weight.item())

        # ====================================================
        # Optimizer
        # ====================================================

        if optim_name == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=0.0001)

        elif optim_name == "adamw":
            optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=0.01)

        elif optim_name == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)

        else:
            raise ValueError("Unsupported optimizer")


        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=learning_rate_decay
        )

        # ====================================================
        # Early stopping
        # ====================================================

        patience = 100
        patience_counter = 0

        train_losses = []
        val_losses = []

        val_eer_history = []
        val_auc_history = []


        # ====================================================
        # Epoch Loop
        # ====================================================

        for epoch in range(num_epochs):

            # ---------------- Train ---------------- #

            self.net.train()

            epoch_train_loss = 0
            correct = 0
            total = 0

            for X, y, _ in tqdm.tqdm(self.train_loader,
                                     desc=f"Epoch {epoch+1}/{num_epochs}",
                                     leave=False):

                X = X.to(self.device)
                y = y.to(self.device).float()

                optimizer.zero_grad()

                logits = self.net(X).squeeze(dim=1)
                loss, _ = loss_function(logits, y)

                loss.backward()

                optimizer.step()

                epoch_train_loss += loss.item()

                preds = (torch.sigmoid(logits) >= 0.5).float()

                correct += (preds == y).sum().item()
                total += y.size(0)


            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)


            # ---------------- Validation ---------------- #

            self.net.eval()

            scores = []
            labels = []

            epoch_val_loss = 0

            with torch.no_grad():

                for X, y, _ in self.val_loader:

                    X = X.to(self.device)
                    y = y.to(self.device).float()

                    logits = self.net(X).squeeze(dim=1)
                    vloss, _ = loss_function(logits, y)
                    epoch_val_loss += vloss.item()

                    scores.extend(torch.sigmoid(logits).cpu().numpy())
                    labels.extend(y.cpu().numpy())


            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            scores = np.array(scores)
            labels = np.array(labels)

            eer, auc, eer_threshold = calculate_eer(labels, scores)

            pos = scores[labels == 1]
            neg = scores[labels == 0]

            print("\n[Score Stats]")
            print(f"Pos mean: {pos.mean():.4f}, std: {pos.std():.4f}")
            print(f"Neg mean: {neg.mean():.4f}, std: {neg.std():.4f}")

            print(f"Pos min/max: {pos.min():.4f} / {pos.max():.4f}")
            print(f"Neg min/max: {neg.min():.4f} / {neg.max():.4f}")

            # overlap indicator
            print(f"Overlap approx: pos_mean - neg_mean = {pos.mean() - neg.mean():.4f}")

            val_eer_history.append(eer)
            val_auc_history.append(auc)


            # ---------------- Precision Recall F1 ---------------- #

            preds = (scores >= eer_threshold).astype(int)

            tp = ((labels == 1) & (preds == 1)).sum()
            fp = ((labels == 0) & (preds == 1)).sum()
            tn = ((labels == 0) & (preds == 0)).sum()
            fn = ((labels == 1) & (preds == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)


            # ====================================================
            # PRINT (截图格式)
            # ====================================================

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val   Loss: {avg_val_loss:.4f}")
            print(f"EER: {eer:.4f} | AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


            # ---------------- Early stopping ---------------- #

            if eer < self.best_val_eer:

                self.best_val_eer = eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                patience_counter = 0

            else:

                patience_counter += 1

                print(f"[EarlyStop] No EER improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:

                    print("Early stopping.")
                    break


            scheduler.step()


        # ====================================================
        # Load Best Model
        # ====================================================

        best_model = copy.deepcopy(self.net)
        best_model.load_state_dict(self.best_model_state)

        return (
            self.net,
            best_model,
            train_losses,
            val_losses,
            val_eer_history,
            val_auc_history
        )