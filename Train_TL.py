"""
train_v3_tl_best.py

V3-TL BEST: Optimized Transfer Learning Ensemble
=================================================
Combines the winning elements from all tested strategies:

✅ From Strategy B: Gradual unfreezing (best at 91.7%)
✅ From Strategy C: Discriminative learning rates
✅ NEW: Knowledge distillation (prevent catastrophic forgetting)
✅ NEW: Cosine annealing with warm restarts
✅ NEW: Longer training phases with better scheduling
✅ NEW: Mixup data augmentation

Key insight from experiments:
  - Reinitializing head = BAD (Strategy A proved this)
  - Gradual unfreezing = BEST (Strategy B: 91.7%)
  - The main enemy is CATASTROPHIC FORGETTING
  - Solution: Knowledge distillation keeps model close to original

Expected: Should match or beat V3-Final (93.5%)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import json
import copy
import math

from Src.data_loader import load_subject
from Src.preprocessing import preprocess_emg, window_data
from Src.feature_extraction import build_feature_matrix

# ============================================================
# CONFIGURATION
# ============================================================
SUBJECTS      = list(range(1, 11))
EXERCISES     = [1, 2, 3]
KEEP_GESTURES = [0, 1, 2, 3, 4, 5]
GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open",
                 "Pinch", "Point", "Wave"]

TEST_SIZE = 0.20
CV_FOLDS  = 8

# SVM params
SVM_C      = 100
SVM_GAMMA  = 0.01
SVM_KERNEL = 'rbf'

# CNN
CNN_MODEL_PATH = "models/cnn_lstm_model.pth"
BATCH_SIZE     = 64
DROPOUT        = 0.5

# Ensemble
SVM_WEIGHT = 0.4
CNN_WEIGHT = 0.6

# ── Transfer Learning Phases ──
# Phase 1: Head only — warm up FC layers
PHASE1_LR       = 3e-4
PHASE1_EPOCHS   = 15
PHASE1_PATIENCE = 8

# Phase 2: Head + LSTM — adapt temporal understanding
PHASE2_LR_LSTM  = 5e-5
PHASE2_LR_FC    = 1e-4
PHASE2_EPOCHS   = 20
PHASE2_PATIENCE = 10

# Phase 3: All layers — global fine-tune with tiny LR
PHASE3_LR_CNN   = 1e-6
PHASE3_LR_LSTM  = 5e-6
PHASE3_LR_FC    = 5e-5
PHASE3_EPOCHS   = 20
PHASE3_PATIENCE = 10

# ── Knowledge Distillation ──
# Teacher = original frozen model
# Student = model being fine-tuned
# Loss = α * CE_loss + (1-α) * KD_loss
KD_ALPHA       = 0.7    # Weight for hard label loss (CE)
KD_TEMPERATURE = 3.0    # Softens teacher probabilities

# ── Mixup Augmentation ──
MIXUP_ALPHA = 0.2       # Beta distribution parameter (0 = no mixup)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results/v3_tl_best"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# CNN-LSTM Architecture
# ============================================================
class CNN_LSTM_V1(nn.Module):
    def __init__(self, n_channels=16, window_size=200, n_classes=6, dropout=0.5):
        super().__init__()
        self.conv1     = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1       = nn.BatchNorm1d(64)
        self.pool1     = nn.MaxPool1d(2)
        self.conv2     = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2       = nn.BatchNorm1d(128)
        self.pool2     = nn.MaxPool1d(2)
        self.conv3     = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3       = nn.BatchNorm1d(64)
        self.pool3     = nn.MaxPool1d(2)
        self.drop_cnn  = nn.Dropout(dropout * 0.4)
        self.lstm      = nn.LSTM(input_size=64, hidden_size=64, num_layers=2,
                                 batch_first=True, dropout=dropout * 0.5,
                                 bidirectional=True)
        self.drop_lstm = nn.Dropout(dropout)
        self.fc1       = nn.Linear(128, 64)
        self.bn_fc     = nn.BatchNorm1d(64)
        self.drop_fc   = nn.Dropout(dropout)
        self.fc2       = nn.Linear(64, 6)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop_cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        x = self.drop_lstm(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data():
    print("\n" + "=" * 65)
    print("  Loading EMG data...")
    print("=" * 65)

    emg_all, labels_all = [], []
    for subj in SUBJECTS:
        print(f"  Subject {subj}...", end=" ")
        count = 0
        for ex in EXERCISES:
            file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
            try:
                emg, labels, _ = load_subject(file)
                emg = preprocess_emg(emg)
                windows, win_labels = window_data(emg, labels)
                emg_all.append(windows)
                labels_all.append(win_labels)
                count += len(win_labels)
            except:
                pass
        print(f"{count} windows")

    X = np.vstack(emg_all)
    y = np.hstack(labels_all)
    print(f"\n  Total: {X.shape[0]}")

    mask = np.isin(y, KEEP_GESTURES)
    X, y = X[mask], y[mask]
    label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
    y = np.array([label_map[l] for l in y])

    classes   = np.unique(y)
    min_count = min(np.sum(y == c) for c in classes)
    rng       = np.random.default_rng(42)

    X_bal, y_bal = [], []
    for c in classes:
        idx    = np.where(y == c)[0]
        chosen = rng.choice(idx, min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])
        print(f"    {GESTURE_NAMES[c]}: {min_count}")

    X = np.vstack(X_bal).astype(np.float32)
    y = np.hstack(y_bal)
    print(f"  Balanced total: {len(y)}")
    return X, y


# ============================================================
# SVM TRAINING
# ============================================================
def train_svm(F_train, y_train, F_test, y_test, F_all, y_all):
    print("\n" + "=" * 65)
    print("  COMPONENT 1: SVM")
    print(f"  Params: C={SVM_C}, gamma={SVM_GAMMA}, kernel={SVM_KERNEL}")
    print("=" * 65)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=SVM_C, gamma=SVM_GAMMA, kernel=SVM_KERNEL,
                     class_weight="balanced", probability=True, random_state=42))
    ])

    # Cross validation
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_acc = cross_val_score(svm, F_all, y_all, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_acc.mean()*100:.1f}% ± {cv_acc.std()*100:.1f}%")

    t0 = time.time()
    svm.fit(F_train, y_train)
    t1 = time.time()

    train_acc = accuracy_score(y_train, svm.predict(F_train)) * 100
    test_acc  = accuracy_score(y_test,  svm.predict(F_test))  * 100
    test_f1   = f1_score(y_test, svm.predict(F_test), average='macro') * 100

    print(f"\n  Train: {train_acc:.1f}%  Test: {test_acc:.1f}%  F1: {test_f1:.1f}%")
    print(f"  CV-Test Gap: {abs(cv_acc.mean()*100 - test_acc):.1f}% ✅")
    print(f"  Time: {t1-t0:.1f}s")

    joblib.dump(svm, "models/v3_tl_best_svm.pkl")
    print(f"  ✅ Saved: models/v3_tl_best_svm.pkl")

    return svm, {
        'train_acc': train_acc, 'test_acc': test_acc, 'test_f1': test_f1,
        'cv_acc': cv_acc.mean() * 100, 'cv_std': cv_acc.std() * 100
    }


# ============================================================
# KNOWLEDGE DISTILLATION LOSS
# ============================================================
class DistillationLoss(nn.Module):
    """
    Combined loss: Hard labels (CE) + Soft labels (KD from teacher)
    This prevents catastrophic forgetting by keeping student
    predictions close to the teacher (original model).
    """
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, true_labels, class_weights=None):
        # Hard label loss (standard cross entropy)
        if class_weights is not None:
            ce_loss = F.cross_entropy(student_logits, true_labels, weight=class_weights)
        else:
            ce_loss = F.cross_entropy(student_logits, true_labels)

        # Soft label loss (knowledge distillation)
        # Soften both student and teacher predictions with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence between soft predictions
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)  # Scale by T^2

        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        return total_loss, ce_loss.item(), kd_loss.item()


# ============================================================
# MIXUP DATA AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    Mixup: Creates virtual training examples by blending pairs.
    Helps prevent overfitting and improves generalization.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam    = np.random.beta(alpha, alpha)
    lam    = max(lam, 1 - lam)  # Ensure lam >= 0.5
    idx    = torch.randperm(x.size(0)).to(x.device)
    mixed  = lam * x + (1 - lam) * x[idx]
    y_a    = y
    y_b    = y[idx]
    return mixed, y_a, y_b, lam


def mixup_criterion(criterion_fn, pred, y_a, y_b, lam, class_weights=None):
    """Mixup-compatible loss."""
    if class_weights is not None:
        loss_a = F.cross_entropy(pred, y_a, weight=class_weights)
        loss_b = F.cross_entropy(pred, y_b, weight=class_weights)
    else:
        loss_a = F.cross_entropy(pred, y_a)
        loss_b = F.cross_entropy(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


# ============================================================
# COSINE ANNEALING SCHEDULER
# ============================================================
def get_cosine_schedule(optimizer, num_epochs, warmup_epochs=3):
    """
    Cosine annealing with linear warmup.
    Better than ReduceLROnPlateau for fine-tuning.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# CNN EVALUATION
# ============================================================
def evaluate_cnn(model, X, y):
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch  = torch.FloatTensor(X[i:i+BATCH_SIZE]).to(DEVICE)
            logits = model(batch)
            probs  = F.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.vstack(all_probs)
    acc   = accuracy_score(y, preds) * 100
    f1    = f1_score(y, preds, average='macro') * 100
    return preds, probs, acc, f1


# ============================================================
# PHASE TRAINING WITH KNOWLEDGE DISTILLATION
# ============================================================
def train_phase(student, teacher, X_train, y_train, X_val, y_val,
                param_groups, epochs, patience, phase_name,
                use_mixup=False, use_kd=True):
    """
    Train one phase with:
    - Knowledge distillation from teacher
    - Optional mixup augmentation
    - Cosine annealing with warmup
    - Gradient clipping
    """
    print(f"\n  ── {phase_name} ──")

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Class weights
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    # Optimizer with parameter groups
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    scheduler = get_cosine_schedule(optimizer, epochs, warmup_epochs=2)

    # Knowledge distillation loss
    kd_criterion = DistillationLoss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())

    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"  Epochs: {epochs}  |  Patience: {patience}")
    print(f"  Knowledge Distillation: {'ON' if use_kd else 'OFF'} (α={KD_ALPHA}, T={KD_TEMPERATURE})")
    print(f"  Mixup: {'ON' if use_mixup else 'OFF'} (α={MIXUP_ALPHA})")

    # Show learning rates
    for i, pg in enumerate(param_groups):
        print(f"  Param group {i}: LR={pg['lr']}")

    best_val_loss = float('inf')
    best_epoch    = 0
    best_state    = None
    patience_ctr  = 0
    history       = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
                     'ce_loss': [], 'kd_loss': []}

    teacher.eval()  # Teacher always in eval mode

    for epoch in range(1, epochs + 1):
        student.train()
        e_loss = e_correct = e_total = 0
        e_ce = e_kd = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # Optional mixup
            if use_mixup:
                xb_mixed, ya, yb_mix, lam = mixup_data(xb, yb)
            else:
                xb_mixed, ya, yb_mix, lam = xb, yb, yb, 1.0

            optimizer.zero_grad()
            student_logits = student(xb_mixed)

            if use_kd:
                # Get teacher predictions (no grad needed)
                with torch.no_grad():
                    teacher_logits = teacher(xb_mixed)

                # Knowledge distillation loss
                if use_mixup and lam < 1.0:
                    # For mixup: blend the KD loss too
                    loss_a, ce_a, kd_a = kd_criterion(student_logits, teacher_logits, ya, class_weights)
                    loss_b, ce_b, kd_b = kd_criterion(student_logits, teacher_logits, yb_mix, class_weights)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    ce_val = lam * ce_a + (1 - lam) * ce_b
                    kd_val = lam * kd_a + (1 - lam) * kd_b
                else:
                    loss, ce_val, kd_val = kd_criterion(student_logits, teacher_logits, ya, class_weights)
                e_ce += ce_val
                e_kd += kd_val
            else:
                # Standard CE loss
                if use_mixup and lam < 1.0:
                    loss = mixup_criterion(None, student_logits, ya, yb_mix, lam, class_weights)
                else:
                    loss = F.cross_entropy(student_logits, ya, weight=class_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss    += loss.item() * len(yb)
            e_correct += (torch.argmax(student_logits, 1) == yb).sum().item()
            e_total   += len(yb)

        scheduler.step()
        train_loss = e_loss / e_total
        train_acc  = e_correct / e_total

        # Validate
        student.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for i in range(0, len(X_val), BATCH_SIZE):
                xb = torch.FloatTensor(X_val[i:i+BATCH_SIZE]).to(DEVICE)
                yb = torch.LongTensor(y_val[i:i+BATCH_SIZE]).to(DEVICE)
                s_logits = student(xb)

                if use_kd:
                    t_logits = teacher(xb)
                    loss, _, _ = kd_criterion(s_logits, t_logits, yb, class_weights)
                else:
                    loss = F.cross_entropy(s_logits, yb, weight=class_weights)

                v_loss    += loss.item() * len(yb)
                v_correct += (torch.argmax(s_logits, 1) == yb).sum().item()
                v_total   += len(yb)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if use_kd:
            n_batches = len(train_dl)
            history['ce_loss'].append(e_ce / n_batches)
            history['kd_loss'].append(e_kd / n_batches)

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            gap  = train_acc - val_acc
            icon = "✅" if abs(gap) < 0.10 else "⚠️"
            lr   = optimizer.param_groups[-1]['lr']
            kd_str = f" CE:{e_ce/len(train_dl):.3f} KD:{e_kd/len(train_dl):.3f}" if use_kd else ""
            print(f"  Epoch {epoch:3d}/{epochs} │ "
                  f"Train: {train_acc*100:.1f}% │ "
                  f"Val: {val_acc*100:.1f}% │ "
                  f"Gap: {gap*100:+.1f}% {icon} │ "
                  f"LR: {lr:.1e}{kd_str}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    student.load_state_dict(best_state)
    student = student.to(DEVICE)

    print(f"  ✅ Phase complete (best epoch: {best_epoch})")
    return student, history


# ============================================================
# MAIN TRANSFER LEARNING PIPELINE
# ============================================================
def train_cnn_best_tl(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 65)
    print("  COMPONENT 2: CNN — Best Transfer Learning")
    print("  Strategy: Gradual Unfreeze + Knowledge Distillation + Mixup")
    print("=" * 65)

    # ── Load pre-trained model as TEACHER (frozen, never changes) ──
    teacher = CNN_LSTM_V1().to(DEVICE)
    ckpt    = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    original_acc = ckpt.get('accuracy', 'N/A')
    print(f"\n  🎓 Teacher loaded: {CNN_MODEL_PATH}")
    print(f"     Original accuracy: {original_acc}")
    print(f"     Teacher is FROZEN — never changes")

    # ── Load pre-trained model as STUDENT (will be fine-tuned) ──
    student = CNN_LSTM_V1().to(DEVICE)
    student.load_state_dict(ckpt['model_state_dict'])  # Same starting weights

    print(f"\n  🔥 Student loaded: Same starting weights as teacher")
    print(f"     Will be gradually fine-tuned with KD guidance")

    # ── Verify starting accuracy ──
    _, _, start_acc, start_f1 = evaluate_cnn(student, X_test, y_test)
    print(f"\n  📊 Starting accuracy (before any training): {start_acc:.1f}%")

    all_history = {'phase': [], 'train_acc': [], 'val_acc': []}

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Fine-tune HEAD only (FC layers)
    # Backbone stays frozen, head gently adapts
    # ═══════════════════════════════════════════════════════════

    # Freeze everything except FC
    for name, param in student.named_parameters():
        if any(k in name for k in ['conv', 'bn1', 'bn2', 'bn3', 'lstm', 'drop_cnn']):
            param.requires_grad = False
        else:
            param.requires_grad = True

    param_groups_p1 = [
        {'params': [p for n, p in student.named_parameters() if p.requires_grad],
         'lr': PHASE1_LR}
    ]

    student, hist1 = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=param_groups_p1,
        epochs=PHASE1_EPOCHS,
        patience=PHASE1_PATIENCE,
        phase_name="PHASE 1: Head Only ❄️❄️❄️🔥",
        use_mixup=False,     # No mixup for head warmup
        use_kd=True          # KD from teacher
    )

    _, _, p1_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"  📊 Phase 1 test accuracy: {p1_acc:.1f}%")

    all_history['phase'].append('Phase 1')
    all_history['train_acc'].extend(hist1['train_acc'])
    all_history['val_acc'].extend(hist1['val_acc'])

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze LSTM + Head
    # LSTM adapts temporal patterns, CNN backbone still frozen
    # ═══════════════════════════════════════════════════════════

    for name, param in student.named_parameters():
        if 'lstm' in name or 'drop_lstm' in name:
            param.requires_grad = True

    # Discriminative LRs
    lstm_params = [p for n, p in student.named_parameters()
                   if ('lstm' in n or 'drop_lstm' in n) and p.requires_grad]
    fc_params   = [p for n, p in student.named_parameters()
                   if ('fc' in n or 'bn_fc' in n or 'drop_fc' in n) and p.requires_grad]

    param_groups_p2 = [
        {'params': lstm_params, 'lr': PHASE2_LR_LSTM},
        {'params': fc_params,   'lr': PHASE2_LR_FC}
    ]

    student, hist2 = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=param_groups_p2,
        epochs=PHASE2_EPOCHS,
        patience=PHASE2_PATIENCE,
        phase_name="PHASE 2: LSTM + Head ❄️❄️🔥🔥",
        use_mixup=True,      # Mixup helps prevent overfitting
        use_kd=True           # KD keeps model close to teacher
    )

    _, _, p2_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"  📊 Phase 2 test accuracy: {p2_acc:.1f}%")

    all_history['phase'].append('Phase 2')
    all_history['train_acc'].extend(hist2['train_acc'])
    all_history['val_acc'].extend(hist2['val_acc'])

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Unfreeze ALL layers
    # Very small LR for CNN, small for LSTM, moderate for FC
    # ═══════════════════════════════════════════════════════════

    for param in student.parameters():
        param.requires_grad = True

    cnn_params  = [p for n, p in student.named_parameters()
                   if any(k in n for k in ['conv', 'bn1', 'bn2', 'bn3', 'drop_cnn'])]
    lstm_params = [p for n, p in student.named_parameters()
                   if ('lstm' in n or 'drop_lstm' in n)]
    fc_params   = [p for n, p in student.named_parameters()
                   if ('fc' in n or 'bn_fc' in n or 'drop_fc' in n)]

    param_groups_p3 = [
        {'params': cnn_params,  'lr': PHASE3_LR_CNN},
        {'params': lstm_params, 'lr': PHASE3_LR_LSTM},
        {'params': fc_params,   'lr': PHASE3_LR_FC}
    ]

    student, hist3 = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=param_groups_p3,
        epochs=PHASE3_EPOCHS,
        patience=PHASE3_PATIENCE,
        phase_name="PHASE 3: All Layers 🔥🔥🔥🔥 (tiny LR)",
        use_mixup=True,
        use_kd=True
    )

    _, _, p3_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"  📊 Phase 3 test accuracy: {p3_acc:.1f}%")

    all_history['phase'].append('Phase 3')
    all_history['train_acc'].extend(hist3['train_acc'])
    all_history['val_acc'].extend(hist3['val_acc'])

    # ── Final evaluation ──
    _, _, train_acc, train_f1 = evaluate_cnn(student, X_train, y_train)
    _, _, val_acc,   val_f1   = evaluate_cnn(student, X_val,   y_val)
    _, _, test_acc,  test_f1  = evaluate_cnn(student, X_test,  y_test)
    gap = train_acc - test_acc

    print(f"\n  {'='*55}")
    print(f"  CNN Transfer Learning — Final Results:")
    print(f"  {'='*55}")
    print(f"  {'Split':<10} {'Accuracy':<12} {'F1':<12}")
    print(f"  {'-'*35}")
    print(f"  {'Train':<10} {train_acc:<12.1f} {train_f1:<12.1f}")
    print(f"  {'Val':<10} {val_acc:<12.1f} {val_f1:<12.1f}")
    print(f"  {'Test':<10} {test_acc:<12.1f} {test_f1:<12.1f}")
    print(f"  {'Gap':<10} {gap:+.1f}% {'✅' if abs(gap) < 5 else '⚠️'}")

    print(f"\n  Progress: {start_acc:.1f}% → {p1_acc:.1f}% → {p2_acc:.1f}% → {p3_acc:.1f}%")

    # Save
    torch.save({
        'model_state_dict': student.state_dict(),
        'accuracy':         test_acc / 100,
        'test_f1':          test_f1,
        'gap':              gap,
        'transfer_learning': True,
        'knowledge_distillation': True,
        'phases': {
            'p1_acc': p1_acc, 'p2_acc': p2_acc, 'p3_acc': p3_acc
        }
    }, "models/v3_tl_best_cnn.pth")
    print(f"\n  ✅ Saved: models/v3_tl_best_cnn.pth")

    return student, {
        'train_acc': train_acc, 'test_acc': test_acc, 'test_f1': test_f1,
        'gap': gap, 'start_acc': start_acc,
        'p1_acc': p1_acc, 'p2_acc': p2_acc, 'p3_acc': p3_acc
    }, all_history


# ============================================================
# ENSEMBLE EVALUATION
# ============================================================
def evaluate_ensemble(svm, cnn, X_test, y_test):
    print("\n" + "=" * 65)
    print("  ENSEMBLE EVALUATION")
    print("=" * 65)

    # SVM
    F_test    = build_feature_matrix(X_test)
    F_test    = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)
    svm_probs = svm.predict_proba(F_test)
    svm_pred  = np.argmax(svm_probs, axis=1)

    # CNN
    _, cnn_probs, cnn_acc, _ = evaluate_cnn(cnn, X_test, y_test)
    cnn_pred = np.argmax(cnn_probs, axis=1)

    # Ensemble
    ens_probs = SVM_WEIGHT * svm_probs + CNN_WEIGHT * cnn_probs
    ens_pred  = np.argmax(ens_probs, axis=1)

    svm_acc = accuracy_score(y_test, svm_pred) * 100
    ens_acc = accuracy_score(y_test, ens_pred) * 100
    ens_f1  = f1_score(y_test, ens_pred, average='macro') * 100

    print(f"\n  {'Model':<30} {'Acc':<10} {'F1':<10}")
    print(f"  {'-'*50}")
    print(f"  {'SVM (hand-crafted)':<30} {svm_acc:<10.1f}")
    print(f"  {'CNN (TL Best)':<30} {cnn_acc:<10.1f}")
    print(f"  {'ENSEMBLE':<30} {ens_acc:<10.1f} {ens_f1:.1f}%")

    # Per gesture
    print(f"\n  {'Gesture':<25} {'Ens':<8} {'SVM':<8} {'CNN':<8} {'Status'}")
    print(f"  {'-'*55}")
    all_pass = True
    per_gesture = {}

    for g in range(len(GESTURE_NAMES)):
        mask = y_test == g
        if np.sum(mask) == 0:
            continue
        g_ens = accuracy_score(y_test[mask], ens_pred[mask]) * 100
        g_svm = accuracy_score(y_test[mask], svm_pred[mask]) * 100
        g_cnn = accuracy_score(y_test[mask], cnn_pred[mask]) * 100
        status = "✅" if g_ens >= 80 else "⚠️" if g_ens >= 75 else "❌"
        if g_ens < 80:
            all_pass = False
        per_gesture[GESTURE_NAMES[g]] = {'ens': g_ens, 'svm': g_svm, 'cnn': g_cnn}
        print(f"  {GESTURE_NAMES[g]:<25} {g_ens:<8.1f} {g_svm:<8.1f} {g_cnn:<8.1f} {status}")

    print(f"\n  {'✅ ALL gestures ≥80%!' if all_pass else '⚠️ Some below 80%'}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, ens_pred, target_names=GESTURE_NAMES))

    return {
        'ensemble_acc': ens_acc, 'ensemble_f1': ens_f1,
        'svm_acc': svm_acc, 'cnn_acc': cnn_acc,
        'all_pass': all_pass, 'per_gesture': per_gesture
    }, ens_pred


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================
def plot_diagnostics(cnn_results, ens_results, history, y_test, ens_pred):
    print("\n" + "=" * 65)
    print("  GENERATING DIAGNOSTIC PLOTS")
    print("=" * 65)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Plot 1: Phase Progress ──
    phases     = ['Start\n(Original)', 'Phase 1\n(Head)', 'Phase 2\n(+LSTM)', 'Phase 3\n(All)']
    phase_accs = [cnn_results['start_acc'], cnn_results['p1_acc'],
                  cnn_results['p2_acc'], cnn_results['p3_acc']]

    colors = ['lightblue' if a < 85 else 'gold' if a < 93 else 'lightgreen' for a in phase_accs]
    bars = axes[0, 0].bar(phases, phase_accs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=93.5, color='blue', linestyle='--', alpha=0.7, label='V3-Final (93.5%)')
    axes[0, 0].axhline(y=91.7, color='orange', linestyle='--', alpha=0.7, label='Prev Best TL (91.7%)')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Transfer Learning Phase Progress\n(KD + Gradual Unfreeze)',
                          fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_ylim(70, 100)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, phase_accs):
        axes[0, 0].annotate(f'{acc:.1f}%',
                            (bar.get_x() + bar.get_width()/2, acc + 0.5),
                            ha='center', fontsize=12, fontweight='bold')

    # ── Plot 2: Training Curves ──
    epochs = range(1, len(history['train_acc']) + 1)
    axes[0, 1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val', linewidth=2)

    # Mark phase boundaries
    p1_end = PHASE1_EPOCHS
    p2_end = p1_end + PHASE2_EPOCHS
    if len(history['train_acc']) > p1_end:
        axes[0, 1].axvline(x=p1_end, color='green', linestyle='--', alpha=0.5, label='Phase 1→2')
    if len(history['train_acc']) > p2_end:
        axes[0, 1].axvline(x=p2_end, color='purple', linestyle='--', alpha=0.5, label='Phase 2→3')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Curves (All Phases)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # ── Plot 3: Per-Gesture ──
    gestures = list(ens_results['per_gesture'].keys())
    g_ens = [ens_results['per_gesture'][g]['ens'] for g in gestures]
    g_svm = [ens_results['per_gesture'][g]['svm'] for g in gestures]
    g_cnn = [ens_results['per_gesture'][g]['cnn'] for g in gestures]

    x = np.arange(len(gestures))
    w = 0.25
    axes[1, 0].bar(x - w, g_svm, w, label='SVM', color='skyblue')
    axes[1, 0].bar(x,     g_cnn, w, label='CNN (TL)', color='lightgreen')
    axes[1, 0].bar(x + w, g_ens, w, label='Ensemble', color='gold', edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([g[:8] for g in gestures], rotation=45, ha='right')
    axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target (80%)')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Per-Gesture Performance', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_ylim(50, 105)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # ── Plot 4: Confusion Matrix ──
    cm = confusion_matrix(y_test, ens_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[1, 1],
                xticklabels=[n[:8] for n in GESTURE_NAMES],
                yticklabels=[n[:8] for n in GESTURE_NAMES],
                vmin=0, vmax=np.max(cm))
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].set_title(f'Confusion Matrix\nAcc: {ens_results["ensemble_acc"]:.1f}%',
                          fontsize=12, fontweight='bold')

    plt.suptitle('V3-TL Best — Knowledge Distillation + Gradual Unfreeze',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/tl_best_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {SAVE_DIR}/tl_best_dashboard.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   V3-TL BEST: Optimized Transfer Learning Ensemble")
    print("   Gradual Unfreeze + Knowledge Distillation + Mixup")
    print("=" * 70)
    print(f"\n  Device:  {DEVICE}")
    print(f"  CNN:     {CNN_MODEL_PATH}")
    print(f"  Method:  3-Phase Gradual Unfreeze")
    print(f"  KD:      α={KD_ALPHA}, T={KD_TEMPERATURE}")
    print(f"  Mixup:   α={MIXUP_ALPHA}")

    # ── Load data ──
    X, y = load_all_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"\n  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ── SVM features ──
    print("\n  Extracting hand-crafted features...")
    F_train = build_feature_matrix(X_train)
    F_test  = build_feature_matrix(X_test)
    F_all   = build_feature_matrix(X)
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_test  = np.nan_to_num(F_test,  nan=0.0, posinf=0.0, neginf=0.0)
    F_all   = np.nan_to_num(F_all,   nan=0.0, posinf=0.0, neginf=0.0)

    # ── SVM ──
    svm, svm_results = train_svm(F_train, y_train, F_test, y_test, F_all, y)

    # ── CNN with Best TL ──
    cnn, cnn_results, cnn_history = train_cnn_best_tl(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── Ensemble ──
    ens_results, ens_pred = evaluate_ensemble(svm, cnn, X_test, y_test)

    # ── Plots ──
    plot_diagnostics(cnn_results, ens_results, cnn_history, y_test, ens_pred)

    # ── Save config ──
    config = {
        'version': 'V3-TL Best',
        'strategy': 'Gradual Unfreeze + Knowledge Distillation + Mixup',
        'kd_alpha': KD_ALPHA,
        'kd_temperature': KD_TEMPERATURE,
        'mixup_alpha': MIXUP_ALPHA,
        'phases': {
            'phase1': {'lr': PHASE1_LR, 'epochs': PHASE1_EPOCHS, 'layers': 'Head only'},
            'phase2': {'lr_lstm': PHASE2_LR_LSTM, 'lr_fc': PHASE2_LR_FC,
                       'epochs': PHASE2_EPOCHS, 'layers': 'LSTM + Head'},
            'phase3': {'lr_cnn': PHASE3_LR_CNN, 'lr_lstm': PHASE3_LR_LSTM,
                       'lr_fc': PHASE3_LR_FC, 'epochs': PHASE3_EPOCHS, 'layers': 'All'}
        },
        'results': {
            'svm': svm_results,
            'cnn': cnn_results,
            'ensemble': {
                'acc': ens_results['ensemble_acc'],
                'f1': ens_results['ensemble_f1'],
                'all_pass': ens_results['all_pass']
            }
        }
    }

    with open(f"models/v3_tl_best_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # ── Final Comparison ──
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)

    v3_final = 93.5
    prev_tl  = 91.7

    print(f"""
  ╔═════════════════════════════════════════════════════════════════════╗
  ║  Version                        CNN       Ensemble   All ≥80%?    ║
  ╠═════════════════════════════════════════════════════════════════════╣
  ║  V3-Final (no TL)              93.2%      93.5%      ✅           ║
  ║  Previous Best TL (Strat B)    90.8%      91.7%      ✅           ║
  ║  V3-TL BEST (this run)         {cnn_results['test_acc']:<6.1f}%    {ens_results['ensemble_acc']:<6.1f}%    {'✅' if ens_results['all_pass'] else '❌'}           ║
  ╠═════════════════════════════════════════════════════════════════════╣
  ║  vs V3-Final:                  {cnn_results['test_acc']-93.2:+.1f}%      {ens_results['ensemble_acc']-v3_final:+.1f}%                     ║
  ║  vs Prev TL:                   {cnn_results['test_acc']-90.8:+.1f}%      {ens_results['ensemble_acc']-prev_tl:+.1f}%                     ║
  ╚═════════════════════════════════════════════════════════════════════╝
    """)

    if ens_results['ensemble_acc'] >= v3_final:
        print("  🎉 TL BEST matched or beat V3-Final!")
    elif ens_results['ensemble_acc'] >= prev_tl:
        print("  ✅ TL BEST improved over previous TL strategies!")
    else:
        print("  ⚠️ TL still below V3-Final — the original CNN is hard to beat!")

    print("\n✅ V3-TL Best complete!\n")


if __name__ == "__main__":
    main()