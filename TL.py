"""
train_v3_tl_unanimous.py

V3-TL UNANIMOUS: Transfer Learning with Agreement-Based Classification
======================================================================
Both models must AGREE before acting — no weighted blending.

✅ From Strategy B: Gradual unfreezing (best at 91.7%)
✅ From Strategy C: Discriminative learning rates
✅ Knowledge distillation (prevent catastrophic forgetting)
✅ Cosine annealing with warm restarts
✅ Mixup data augmentation
✅ NEW: Unanimous agreement (SVM == CNN required to act)

Key Philosophy:
  - SVM and CNN are trained INDEPENDENTLY
  - At inference: both must agree to act
  - Disagreement → abstain / hold last state
  - Agreement Rate = TL quality metric

For Prosthetics:
  - Wrong action (accidental grasp) is WORSE than no action
  - Unanimous agreement = natural confidence gate
  - Two different "viewpoints" agreeing = high reliability
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

# ── NO ENSEMBLE WEIGHTS ──
# Models must AGREE, not blend

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
KD_ALPHA       = 0.7    # Weight for hard label loss (CE)
KD_TEMPERATURE = 3.0    # Softens teacher probabilities

# ── Mixup Augmentation ──
MIXUP_ALPHA = 0.2       # Beta distribution parameter (0 = no mixup)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results/v3_tl_unanimous"
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
    print("  COMPONENT 1: SVM (Independent)")
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

    joblib.dump(svm, "models/v3_tl_unanimous_svm.pkl")
    print(f"  ✅ Saved: models/v3_tl_unanimous_svm.pkl")

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
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence between soft predictions
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        return total_loss, ce_loss.item(), kd_loss.item()


# ============================================================
# MIXUP DATA AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    Mixup: Creates virtual training examples by blending pairs.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam    = np.random.beta(alpha, alpha)
    lam    = max(lam, 1 - lam)
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
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
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

    for i, pg in enumerate(param_groups):
        print(f"  Param group {i}: LR={pg['lr']}")

    best_val_loss = float('inf')
    best_epoch    = 0
    best_state    = None
    patience_ctr  = 0
    history       = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
                     'ce_loss': [], 'kd_loss': []}

    teacher.eval()

    for epoch in range(1, epochs + 1):
        student.train()
        e_loss = e_correct = e_total = 0
        e_ce = e_kd = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if use_mixup:
                xb_mixed, ya, yb_mix, lam = mixup_data(xb, yb)
            else:
                xb_mixed, ya, yb_mix, lam = xb, yb, yb, 1.0

            optimizer.zero_grad()
            student_logits = student(xb_mixed)

            if use_kd:
                with torch.no_grad():
                    teacher_logits = teacher(xb_mixed)

                if use_mixup and lam < 1.0:
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
def train_cnn_tl(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 65)
    print("  COMPONENT 2: CNN — Transfer Learning (Independent)")
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
    student.load_state_dict(ckpt['model_state_dict'])

    print(f"\n  🔥 Student loaded: Same starting weights as teacher")
    print(f"     Will be gradually fine-tuned with KD guidance")

    # ── Verify starting accuracy ──
    _, _, start_acc, start_f1 = evaluate_cnn(student, X_test, y_test)
    print(f"\n  📊 Starting accuracy (before any training): {start_acc:.1f}%")

    all_history = {'phase': [], 'train_acc': [], 'val_acc': []}

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Fine-tune HEAD only (FC layers)
    # ═══════════════════════════════════════════════════════════

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
        use_mixup=False,
        use_kd=True
    )

    _, _, p1_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"  📊 Phase 1 test accuracy: {p1_acc:.1f}%")

    all_history['phase'].append('Phase 1')
    all_history['train_acc'].extend(hist1['train_acc'])
    all_history['val_acc'].extend(hist1['val_acc'])

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze LSTM + Head
    # ═══════════════════════════════════════════════════════════

    for name, param in student.named_parameters():
        if 'lstm' in name or 'drop_lstm' in name:
            param.requires_grad = True

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
        use_mixup=True,
        use_kd=True
    )

    _, _, p2_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"  📊 Phase 2 test accuracy: {p2_acc:.1f}%")

    all_history['phase'].append('Phase 2')
    all_history['train_acc'].extend(hist2['train_acc'])
    all_history['val_acc'].extend(hist2['val_acc'])

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Unfreeze ALL layers
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
    }, "models/v3_tl_unanimous_cnn.pth")
    print(f"\n  ✅ Saved: models/v3_tl_unanimous_cnn.pth")

    return student, {
        'train_acc': train_acc, 'test_acc': test_acc, 'test_f1': test_f1,
        'gap': gap, 'start_acc': start_acc,
        'p1_acc': p1_acc, 'p2_acc': p2_acc, 'p3_acc': p3_acc
    }, all_history


# ============================================================
# UNANIMOUS AGREEMENT EVALUATION (REPLACES ENSEMBLE)
# ============================================================
def evaluate_unanimous(svm, cnn, X_test, y_test):
    """
    Evaluate using UNANIMOUS AGREEMENT:
    - Both SVM and CNN must predict the SAME class to act
    - If they disagree → abstain (hold last state)
    
    This is the core philosophy: two independent models agreeing
    provides higher confidence than any single model or blend.
    """
    print("\n" + "=" * 65)
    print("  UNANIMOUS AGREEMENT EVALUATION")
    print("  Philosophy: Both models must AGREE to act")
    print("=" * 65)

    # ── Get SVM predictions ──
    F_test    = build_feature_matrix(X_test)
    F_test    = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)
    svm_probs = svm.predict_proba(F_test)
    svm_pred  = np.argmax(svm_probs, axis=1)
    svm_conf  = np.max(svm_probs, axis=1)

    # ── Get CNN predictions ──
    cnn_pred, cnn_probs, cnn_acc, _ = evaluate_cnn(cnn, X_test, y_test)
    cnn_conf = np.max(cnn_probs, axis=1)

    # ── Unanimous agreement check ──
    agreement_mask = (svm_pred == cnn_pred)
    n_total        = len(y_test)
    n_agree        = np.sum(agreement_mask)
    n_disagree     = n_total - n_agree
    agreement_rate = n_agree / n_total * 100
    abstention_rate = n_disagree / n_total * 100

    # ── Build final predictions ──
    # Where they agree: use their unanimous prediction
    # Where they disagree: mark as -1 (abstain)
    unanimous_pred = np.where(agreement_mask, svm_pred, -1)

    # ── Accuracy metrics ──
    # Individual model accuracies (on all samples)
    svm_acc = accuracy_score(y_test, svm_pred) * 100
    cnn_acc_full = accuracy_score(y_test, cnn_pred) * 100

    # Unanimous accuracy (only on samples where they agreed)
    if n_agree > 0:
        unanimous_acc = accuracy_score(y_test[agreement_mask], 
                                       unanimous_pred[agreement_mask]) * 100
        unanimous_f1 = f1_score(y_test[agreement_mask], 
                                unanimous_pred[agreement_mask], 
                                average='macro') * 100
    else:
        unanimous_acc = 0.0
        unanimous_f1 = 0.0

    # Coverage-adjusted accuracy (treating abstentions as errors)
    # This is harsh but realistic for prosthetics
    coverage_acc = (np.sum((unanimous_pred == y_test) & agreement_mask) / n_total) * 100

    # ── Disagreement analysis ──
    # When they disagree, which one was right?
    disagree_idx = np.where(~agreement_mask)[0]
    if len(disagree_idx) > 0:
        svm_right_when_disagree = np.sum(svm_pred[disagree_idx] == y_test[disagree_idx])
        cnn_right_when_disagree = np.sum(cnn_pred[disagree_idx] == y_test[disagree_idx])
        neither_right = len(disagree_idx) - svm_right_when_disagree - cnn_right_when_disagree + \
                        np.sum((svm_pred[disagree_idx] == y_test[disagree_idx]) & 
                               (cnn_pred[disagree_idx] == y_test[disagree_idx]))
    else:
        svm_right_when_disagree = 0
        cnn_right_when_disagree = 0
        neither_right = 0

    print(f"\n  {'='*55}")
    print(f"  AGREEMENT STATISTICS")
    print(f"  {'='*55}")
    print(f"  Total samples:     {n_total}")
    print(f"  Agreed (act):      {n_agree} ({agreement_rate:.1f}%)")
    print(f"  Disagreed (hold):  {n_disagree} ({abstention_rate:.1f}%)")
    
    print(f"\n  {'='*55}")
    print(f"  ACCURACY METRICS")
    print(f"  {'='*55}")
    print(f"  {'Metric':<30} {'Value':<15} {'Notes'}")
    print(f"  {'-'*60}")
    print(f"  {'SVM (standalone)':<30} {svm_acc:<15.1f} All samples")
    print(f"  {'CNN (standalone)':<30} {cnn_acc_full:<15.1f} All samples")
    print(f"  {'Unanimous (when agreed)':<30} {unanimous_acc:<15.1f} Only agreed samples")
    print(f"  {'Coverage-Adjusted':<30} {coverage_acc:<15.1f} Abstain = wrong")

    # Quality indicator
    if unanimous_acc >= 95:
        quality = "🌟 EXCELLENT"
    elif unanimous_acc >= 90:
        quality = "✅ GREAT"
    elif unanimous_acc >= 85:
        quality = "👍 GOOD"
    else:
        quality = "⚠️ NEEDS WORK"
    
    print(f"\n  Unanimous Quality: {quality}")
    print(f"  (When both models agree, they're right {unanimous_acc:.1f}% of the time)")

    if len(disagree_idx) > 0:
        print(f"\n  {'='*55}")
        print(f"  DISAGREEMENT ANALYSIS (n={len(disagree_idx)})")
        print(f"  {'='*55}")
        print(f"  SVM was right:     {svm_right_when_disagree} ({svm_right_when_disagree/len(disagree_idx)*100:.1f}%)")
        print(f"  CNN was right:     {cnn_right_when_disagree} ({cnn_right_when_disagree/len(disagree_idx)*100:.1f}%)")
        print(f"  Both wrong:        {len(disagree_idx) - svm_right_when_disagree - cnn_right_when_disagree + np.sum((svm_pred[disagree_idx] == y_test[disagree_idx]) & (cnn_pred[disagree_idx] == y_test[disagree_idx]))}")

    # ── Per-gesture analysis ──
    print(f"\n  {'='*55}")
    print(f"  PER-GESTURE BREAKDOWN")
    print(f"  {'='*55}")
    print(f"  {'Gesture':<20} {'Agree%':<10} {'Unan.Acc':<10} {'SVM':<8} {'CNN':<8} {'Status'}")
    print(f"  {'-'*70}")
    
    per_gesture = {}
    all_pass = True

    for g in range(len(GESTURE_NAMES)):
        mask_g = y_test == g
        if np.sum(mask_g) == 0:
            continue
        
        # Agreement rate for this gesture
        agree_g = np.sum(agreement_mask[mask_g])
        total_g = np.sum(mask_g)
        agree_rate_g = agree_g / total_g * 100
        
        # Accuracy when agreed for this gesture
        agree_and_g = agreement_mask & mask_g
        if np.sum(agree_and_g) > 0:
            unan_acc_g = accuracy_score(y_test[agree_and_g], 
                                        unanimous_pred[agree_and_g]) * 100
        else:
            unan_acc_g = 0.0
        
        # Individual model accuracy for this gesture
        svm_acc_g = accuracy_score(y_test[mask_g], svm_pred[mask_g]) * 100
        cnn_acc_g = accuracy_score(y_test[mask_g], cnn_pred[mask_g]) * 100
        
        # Status based on unanimous accuracy
        if unan_acc_g >= 80:
            status = "✅"
        elif unan_acc_g >= 75:
            status = "⚠️"
            all_pass = False
        else:
            status = "❌"
            all_pass = False
        
        per_gesture[GESTURE_NAMES[g]] = {
            'agreement_rate': agree_rate_g,
            'unanimous_acc': unan_acc_g,
            'svm_acc': svm_acc_g,
            'cnn_acc': cnn_acc_g
        }
        
        print(f"  {GESTURE_NAMES[g]:<20} {agree_rate_g:<10.1f} {unan_acc_g:<10.1f} {svm_acc_g:<8.1f} {cnn_acc_g:<8.1f} {status}")

    print(f"\n  {'✅ ALL gestures ≥80% unanimous accuracy!' if all_pass else '⚠️ Some gestures below 80%'}")

    # ── Confidence analysis ──
    print(f"\n  {'='*55}")
    print(f"  CONFIDENCE ANALYSIS")
    print(f"  {'='*55}")
    
    # Average confidence when agreed vs disagreed
    if n_agree > 0:
        avg_svm_conf_agree = np.mean(svm_conf[agreement_mask]) * 100
        avg_cnn_conf_agree = np.mean(cnn_conf[agreement_mask]) * 100
    else:
        avg_svm_conf_agree = 0
        avg_cnn_conf_agree = 0
    
    if n_disagree > 0:
        avg_svm_conf_disagree = np.mean(svm_conf[~agreement_mask]) * 100
        avg_cnn_conf_disagree = np.mean(cnn_conf[~agreement_mask]) * 100
    else:
        avg_svm_conf_disagree = 0
        avg_cnn_conf_disagree = 0
    
    print(f"  When AGREED:")
    print(f"    SVM avg confidence: {avg_svm_conf_agree:.1f}%")
    print(f"    CNN avg confidence: {avg_cnn_conf_agree:.1f}%")
    print(f"  When DISAGREED:")
    print(f"    SVM avg confidence: {avg_svm_conf_disagree:.1f}%")
    print(f"    CNN avg confidence: {avg_cnn_conf_disagree:.1f}%")

    # ── Classification report (only on agreed samples) ──
    if n_agree > 0:
        print(f"\n  Classification Report (Unanimous predictions only):")
        agreed_true = y_test[agreement_mask]
        agreed_pred = unanimous_pred[agreement_mask]
        print(classification_report(agreed_true, agreed_pred, 
                                    target_names=GESTURE_NAMES))

    return {
        'agreement_rate': agreement_rate,
        'abstention_rate': abstention_rate,
        'unanimous_acc': unanimous_acc,
        'unanimous_f1': unanimous_f1,
        'coverage_acc': coverage_acc,
        'svm_acc': svm_acc,
        'cnn_acc': cnn_acc_full,
        'all_pass': all_pass,
        'per_gesture': per_gesture,
        'n_agree': n_agree,
        'n_disagree': n_disagree,
        'svm_right_when_disagree': svm_right_when_disagree,
        'cnn_right_when_disagree': cnn_right_when_disagree
    }, unanimous_pred, agreement_mask, svm_pred, cnn_pred


# ============================================================
# DIAGNOSTIC PLOTS (UPDATED FOR UNANIMOUS)
# ============================================================
def plot_diagnostics(cnn_results, unan_results, history, y_test, 
                     unanimous_pred, agreement_mask, svm_pred, cnn_pred):
    print("\n" + "=" * 65)
    print("  GENERATING DIAGNOSTIC PLOTS")
    print("=" * 65)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── Plot 1: Phase Progress ──
    phases     = ['Start\n(Original)', 'Phase 1\n(Head)', 'Phase 2\n(+LSTM)', 'Phase 3\n(All)']
    phase_accs = [cnn_results['start_acc'], cnn_results['p1_acc'],
                  cnn_results['p2_acc'], cnn_results['p3_acc']]

    colors = ['lightblue' if a < 85 else 'gold' if a < 93 else 'lightgreen' for a in phase_accs]
    bars = axes[0, 0].bar(phases, phase_accs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=93.5, color='blue', linestyle='--', alpha=0.7, label='V3-Final (93.5%)')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('CNN Transfer Learning Phases', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_ylim(70, 100)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, phase_accs):
        axes[0, 0].annotate(f'{acc:.1f}%',
                            (bar.get_x() + bar.get_width()/2, acc + 0.5),
                            ha='center', fontsize=11, fontweight='bold')

    # ── Plot 2: Training Curves ──
    epochs = range(1, len(history['train_acc']) + 1)
    axes[0, 1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Curves (All Phases)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # ── Plot 3: Agreement vs Disagreement Pie ──
    agree_correct = np.sum((unanimous_pred == y_test) & agreement_mask)
    agree_wrong   = np.sum((unanimous_pred != y_test) & agreement_mask & (unanimous_pred != -1))
    disagree      = np.sum(~agreement_mask)
    
    pie_data   = [agree_correct, agree_wrong, disagree]
    pie_labels = [f'Agree & Correct\n({agree_correct})', 
                  f'Agree & Wrong\n({agree_wrong})', 
                  f'Disagree (Abstain)\n({disagree})']
    pie_colors = ['lightgreen', 'lightcoral', 'lightyellow']
    
    axes[0, 2].pie(pie_data, labels=pie_labels, colors=pie_colors, 
                   autopct='%1.1f%%', startangle=90, explode=(0.02, 0.02, 0.05))
    axes[0, 2].set_title(f'Agreement Breakdown\n(Agreement Rate: {unan_results["agreement_rate"]:.1f}%)',
                          fontsize=12, fontweight='bold')

    # ── Plot 4: Per-Gesture Agreement & Accuracy ──
    gestures   = list(unan_results['per_gesture'].keys())
    agree_pct  = [unan_results['per_gesture'][g]['agreement_rate'] for g in gestures]
    unan_acc   = [unan_results['per_gesture'][g]['unanimous_acc'] for g in gestures]

    x = np.arange(len(gestures))
    w = 0.35
    bars1 = axes[1, 0].bar(x - w/2, agree_pct, w, label='Agreement %', color='skyblue')
    bars2 = axes[1, 0].bar(x + w/2, unan_acc, w, label='Unanimous Acc %', color='lightgreen')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([g[:8] for g in gestures], rotation=45, ha='right')
    axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target (80%)')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].set_title('Per-Gesture: Agreement & Unanimous Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # ── Plot 5: SVM vs CNN comparison per gesture ──
    svm_accs = [unan_results['per_gesture'][g]['svm_acc'] for g in gestures]
    cnn_accs = [unan_results['per_gesture'][g]['cnn_acc'] for g in gestures]

    axes[1, 1].bar(x - w/2, svm_accs, w, label='SVM', color='coral')
    axes[1, 1].bar(x + w/2, cnn_accs, w, label='CNN', color='mediumseagreen')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([g[:8] for g in gestures], rotation=45, ha='right')
    axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Individual Model Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylim(50, 105)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # ── Plot 6: Confusion Matrix (Unanimous only) ──
    # Only show confusion for samples where models agreed
    agreed_true = y_test[agreement_mask]
    agreed_pred = unanimous_pred[agreement_mask]
    
    if len(agreed_true) > 0:
        cm = confusion_matrix(agreed_true, agreed_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[1, 2],
                    xticklabels=[n[:8] for n in GESTURE_NAMES],
                    yticklabels=[n[:8] for n in GESTURE_NAMES],
                    vmin=0, vmax=np.max(cm))
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('True')
        axes[1, 2].set_title(f'Confusion Matrix (Unanimous Only)\nAcc: {unan_results["unanimous_acc"]:.1f}%',
                              fontsize=12, fontweight='bold')

    plt.suptitle('V3-TL Unanimous — Both Models Must Agree',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/tl_unanimous_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {SAVE_DIR}/tl_unanimous_dashboard.png")
    plt.close()

    # ── Additional Plot: Disagreement Heatmap ──
    # Shows which gesture pairs cause disagreement
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    disagree_matrix = np.zeros((len(GESTURE_NAMES), len(GESTURE_NAMES)), dtype=int)
    disagree_idx = np.where(~agreement_mask)[0]
    
    for idx in disagree_idx:
        s = svm_pred[idx]
        c = cnn_pred[idx]
        disagree_matrix[s, c] += 1
    
    sns.heatmap(disagree_matrix, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=[n[:8] for n in GESTURE_NAMES],
                yticklabels=[n[:8] for n in GESTURE_NAMES])
    ax2.set_xlabel('CNN Prediction')
    ax2.set_ylabel('SVM Prediction')
    ax2.set_title(f'Disagreement Heatmap\n(SVM says X, CNN says Y)\nTotal disagreements: {len(disagree_idx)}',
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/disagreement_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {SAVE_DIR}/disagreement_heatmap.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   V3-TL UNANIMOUS: Transfer Learning with Agreement-Based Output")
    print("   Both SVM and CNN must AGREE to act — no blending")
    print("=" * 70)
    print(f"\n  Device:  {DEVICE}")
    print(f"  CNN:     {CNN_MODEL_PATH}")
    print(f"  Method:  3-Phase Gradual Unfreeze + KD")
    print(f"  Output:  UNANIMOUS (SVM == CNN required)")

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

    # ── SVM (Independent) ──
    svm, svm_results = train_svm(F_train, y_train, F_test, y_test, F_all, y)

    # ── CNN with TL (Independent) ──
    cnn, cnn_results, cnn_history = train_cnn_tl(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── Unanimous Evaluation ──
    unan_results, unanimous_pred, agreement_mask, svm_pred, cnn_pred = \
        evaluate_unanimous(svm, cnn, X_test, y_test)

    # ── Plots ──
    plot_diagnostics(cnn_results, unan_results, cnn_history, y_test,
                     unanimous_pred, agreement_mask, svm_pred, cnn_pred)

    # ── Save config ──
    config = {
        'version': 'V3-TL Unanimous',
        'strategy': 'Gradual Unfreeze + KD + Unanimous Agreement',
        'output_method': 'UNANIMOUS (both models must agree)',
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
            'unanimous': {
                'agreement_rate': unan_results['agreement_rate'],
                'unanimous_acc': unan_results['unanimous_acc'],
                'unanimous_f1': unan_results['unanimous_f1'],
                'coverage_acc': unan_results['coverage_acc'],
                'abstention_rate': unan_results['abstention_rate'],
                'all_pass': unan_results['all_pass']
            }
        }
    }

    with open(f"models/v3_tl_unanimous_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: UNANIMOUS AGREEMENT APPROACH")
    print("=" * 70)

    print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  INDIVIDUAL MODELS                                                   ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  SVM Accuracy:           {svm_results['test_acc']:<6.1f}%                                  ║
  ║  CNN Accuracy:           {cnn_results['test_acc']:<6.1f}%                                  ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  UNANIMOUS AGREEMENT                                                 ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  Agreement Rate:         {unan_results['agreement_rate']:<6.1f}%  (both models same pred)       ║
  ║  Abstention Rate:        {unan_results['abstention_rate']:<6.1f}%  (held / skipped)              ║
  ║  Unanimous Accuracy:     {unan_results['unanimous_acc']:<6.1f}%  (when they agreed)            ║
  ║  Coverage-Adjusted Acc:  {unan_results['coverage_acc']:<6.1f}%  (abstain = wrong)             ║
  ║  All Gestures ≥80%:      {'✅ YES' if unan_results['all_pass'] else '❌ NO':<6}                                     ║
  ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Interpretation
    print("  📊 INTERPRETATION:")
    print(f"  • When SVM and CNN agree ({unan_results['agreement_rate']:.1f}% of the time),")
    print(f"    they are correct {unan_results['unanimous_acc']:.1f}% of the time.")
    print(f"  • The {unan_results['abstention_rate']:.1f}% abstention rate means the prosthetic")
    print(f"    would 'hold last state' for those uncertain moments.")
    
    if unan_results['unanimous_acc'] >= 95:
        print(f"\n  🌟 EXCELLENT: Unanimous agreement is highly reliable!")
    elif unan_results['unanimous_acc'] >= 90:
        print(f"\n  ✅ GREAT: Safe for prosthetic deployment with hold-state logic.")
    elif unan_results['unanimous_acc'] >= 85:
        print(f"\n  👍 GOOD: Acceptable, but consider improving agreement rate.")
    else:
        print(f"\n  ⚠️ NEEDS WORK: Models are not aligning well after TL.")

    print("\n✅ V3-TL Unanimous complete!\n")


if __name__ == "__main__":
    main()