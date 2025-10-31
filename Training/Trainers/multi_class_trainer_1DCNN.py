import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import sys
import copy
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ------------------ Utility Functions ------------------ #
def compare_models(model1, model2):
    device = next(model1.parameters()).device
    model2 = model2.to(device)
    return all(torch.equal(p1, p2) for p1, p2 in zip(model1.state_dict().values(), model2.state_dict().values()))


def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = thresholds[np.nanargmin(np.abs((1 - tpr) - fpr))]
    except (ValueError, RuntimeError):
        eer, eer_threshold = 0.5, 0.5
    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)
    return eer, auc, eer_threshold


# ------------------ Trainer for 1D-CNN ------------------ #
class MultiLabelTrainer1DCNN:
    def __init__(self, net, train_loader, val_loader, neg_weight_value=1.0, C_pos=10.0, C_neg=10.0):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.best_model_state = None
        self.best_val_eer = float('inf')

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)
        self.neg_weight = torch.tensor([neg_weight_value], dtype=torch.float).to(self.device)
        self.num_users = train_loader.dataset[0][1].shape[0]
        print(f"[INFO] Using neg_weight = {self.neg_weight.item():.2f} (applied to all users)")

        # === Precompute dataset-level positive/negative ratios ===
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        total_count = all_labels.shape[0]
        self.F1_all = (all_labels.sum(dim=0) / total_count).clamp(min=1e-6)
        self.F2_all = ((1.0 - all_labels).sum(dim=0) / total_count).clamp(min=1e-6)
        print("[INFO] Using dataset-level static F1 and F2 for weighted loss.")

    # ------------------ Custom Weighted BCE Loss ------------------ #
    def _loss_fn(self, logits, labels):
        probs = torch.sigmoid(logits)
        w1 = (self.C_pos / self.F1_all).to(self.device)
        w2 = (self.C_neg / self.F2_all).to(self.device)
        loss = - (w1 * labels * torch.log(probs + 1e-6) +
                  w2 * (1 - labels) * torch.log(1 - probs + 1e-6))
        return loss.mean()

    # ------------------ Training Loop ------------------ #
    def train(self, optim_name='adam', num_epochs=30, learning_rate=1e-3,
              reg=0.0, step_size=5, learning_rate_decay=0.9,
              acc_frequency=1, verbose=True):

        if optim_name.lower() == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=learning_rate_decay)

        patience, min_delta = 10, 0.001
        patience_counter = 0

        val_eer_history, val_auc_history = [], []
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.net.train()
            epoch_train_loss = 0

            for X, y in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits = self.net(X)
                loss = self._loss_fn(logits, y)

                if reg > 0:
                    loss += reg * sum(p.pow(2).sum() for p in self.net.parameters())

                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # -------- Validation --------
            self.net.eval()
            epoch_val_loss = 0
            val_scores, val_labels = [], []

            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.net(X)
                    loss = self._loss_fn(logits, y)
                    epoch_val_loss += loss.item()
                    val_scores.append(torch.sigmoid(logits).cpu())
                    val_labels.append(y.cpu())

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)
            val_scores = torch.cat(val_scores).numpy()
            val_labels = torch.cat(val_labels).numpy()

            # Per-user EER, AUC
            user_eers, user_aucs = [], []
            for u in range(val_labels.shape[1]):
                eer, auc, _ = calculate_eer(val_labels[:, u], val_scores[:, u])
                user_eers.append(eer)
                user_aucs.append(auc)
            avg_eer, avg_auc = np.mean(user_eers), np.mean(user_aucs)

            val_eer_history.append(avg_eer)
            val_auc_history.append(avg_auc)

            # Early stopping & checkpoint
            if self.best_val_eer - avg_eer > min_delta:
                self.best_val_eer = avg_eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                print(f"[INFO] New best model saved at epoch {epoch + 1} with Avg EER: {avg_eer:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[EarlyStop] No improvement in EER > {min_delta:.4f} for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

            if verbose or (epoch + 1) % acc_frequency == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss:   {avg_val_loss:.4f}")
                print(f"  EER: {avg_eer:.4f}, AUC: {avg_auc:.4f}")

        # ------------------ Load Best Model ------------------ #
        if self.best_model_state is not None:
            best_model = self.net.__class__(
                input_dim=self.net.input_dim,
                seq_len=self.net.seq_len,
                num_users=self.num_users
            )
            best_model.load_state_dict(self.best_model_state)
            best_model.to(self.device)
        else:
            best_model = self.net

        return self.net, best_model, train_losses, val_losses, val_eer_history, val_auc_history
