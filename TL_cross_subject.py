"""
TL_cross_subject.py

V3-TL UNANIMOUS: Cross-Subject Transfer Learning
=================================================
Leave-One-Subject-Out (LOSO) cross-validation.

Each fold:
  - Train on 9 subjects
  - Test on the 1 held-out subject
  - Both SVM and CNN trained independently
  - Unanimous agreement at inference

This answers: "Does the model generalize to NEW users?"

Key differences from TL.py:
  - LOSO split instead of random 80/20
  - Subject IDs tracked throughout
  - Per-subject results reported
  - Average metrics across all 10 folds
  - Best fold model saved
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import json
import math
import copy

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
GESTURE_SHORT = ["Rest", "Grasp", "Open", "Pinch", "Point", "Wave"]

CV_FOLDS = 5   # Inner CV for SVM (reduced — outer loop is LOSO)

# SVM params
SVM_C      = 100
SVM_GAMMA  = 0.01
SVM_KERNEL = 'rbf'

# CNN
CNN_MODEL_PATH = "models/cnn_lstm_model.pth"
BATCH_SIZE     = 64
DROPOUT        = 0.5

# ── Transfer Learning Phases ──
PHASE1_LR       = 3e-4
PHASE1_EPOCHS   = 15
PHASE1_PATIENCE = 8

PHASE2_LR_LSTM  = 5e-5
PHASE2_LR_FC    = 1e-4
PHASE2_EPOCHS   = 20
PHASE2_PATIENCE = 10

PHASE3_LR_CNN   = 1e-6
PHASE3_LR_LSTM  = 5e-6
PHASE3_LR_FC    = 5e-5
PHASE3_EPOCHS   = 20
PHASE3_PATIENCE = 10

# ── Knowledge Distillation ──
KD_ALPHA       = 0.7
KD_TEMPERATURE = 3.0

# ── Mixup ──
MIXUP_ALPHA = 0.2

# ── Validation split from training subjects ──
VAL_SUBJECT_COUNT = 1   # Hold out 1 training subject for validation

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results/v3_tl_cross_subject"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models/cross_subject", exist_ok=True)


# ============================================================
# CNN-LSTM ARCHITECTURE
# ============================================================
class CNN_LSTM_V1(nn.Module):
    def __init__(self, n_channels=16, window_size=200,
                 n_classes=6, dropout=0.5):
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
        self.lstm      = nn.LSTM(input_size=64, hidden_size=64,
                                 num_layers=2, batch_first=True,
                                 dropout=dropout * 0.5, bidirectional=True)
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
# DATA LOADING — Per Subject
# ============================================================
def load_all_subjects():
    """
    Load data for ALL subjects separately.
    Returns dict: {subject_id: (X, y)}

    This keeps subject boundaries intact for LOSO.
    """
    print("\n" + "=" * 65)
    print("  Loading EMG data (Per Subject)...")
    print("=" * 65)

    subject_data = {}

    for subj in SUBJECTS:
        print(f"  Subject {subj:02d}...", end=" ")
        emg_all, labels_all = [], []

        for ex in EXERCISES:
            file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
            try:
                emg, labels, _ = load_subject(file)
                emg = preprocess_emg(emg)
                windows, win_labels = window_data(emg, labels)
                emg_all.append(windows)
                labels_all.append(win_labels)
            except Exception as e:
                pass

        if len(emg_all) == 0:
            print("⚠️ No data found!")
            continue

        X_subj = np.vstack(emg_all)
        y_subj = np.hstack(labels_all)

        # Filter gestures
        mask      = np.isin(y_subj, KEEP_GESTURES)
        X_subj    = X_subj[mask]
        y_subj    = y_subj[mask]
        label_map = {old: new for new, old in
                     enumerate(sorted(KEEP_GESTURES))}
        y_subj    = np.array([label_map[l] for l in y_subj])

        subject_data[subj] = (X_subj.astype(np.float32), y_subj)
        print(f"{len(y_subj)} windows "
              f"({', '.join([f'{np.sum(y_subj==g)}' for g in range(6)])})")

    print(f"\n  ✅ Loaded {len(subject_data)} subjects")
    return subject_data


def balance_dataset(X, y, rng=None):
    """Balance classes by downsampling to smallest class."""
    if rng is None:
        rng = np.random.default_rng(42)

    classes   = np.unique(y)
    min_count = min(np.sum(y == c) for c in classes)

    X_bal, y_bal = [], []
    for c in classes:
        idx    = np.where(y == c)[0]
        chosen = rng.choice(idx, min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    return np.vstack(X_bal), np.hstack(y_bal)


def build_loso_split(subject_data, test_subject):
    """
    Build one LOSO fold:
      - Test:  held-out subject (raw, unbalanced)
      - Train: all other subjects (combined + balanced)
      - Val:   one random training subject held out

    Returns X_train, y_train, X_val, y_val, X_test, y_test
    """
    rng = np.random.default_rng(42 + test_subject)

    train_subjects = [s for s in SUBJECTS if s != test_subject
                      and s in subject_data]

    # ── Val subject: randomly pick 1 from training subjects ──
    val_subject = rng.choice(train_subjects)
    pure_train_subjects = [s for s in train_subjects
                           if s != val_subject]

    # ── Assemble train ──
    X_train_list, y_train_list = [], []
    for s in pure_train_subjects:
        X_s, y_s = subject_data[s]
        X_train_list.append(X_s)
        y_train_list.append(y_s)

    X_train_raw = np.vstack(X_train_list)
    y_train_raw = np.hstack(y_train_list)
    X_train, y_train = balance_dataset(X_train_raw, y_train_raw, rng)

    # ── Assemble val ──
    X_val_raw, y_val_raw = subject_data[val_subject]
    X_val, y_val = balance_dataset(X_val_raw, y_val_raw, rng)

    # ── Test: held-out subject (balance for fair evaluation) ──
    X_test_raw, y_test_raw = subject_data[test_subject]
    X_test, y_test = balance_dataset(X_test_raw, y_test_raw, rng)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# KNOWLEDGE DISTILLATION LOSS
# ============================================================
class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE):
        super().__init__()
        self.alpha       = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits,
                true_labels, class_weights=None):
        if class_weights is not None:
            ce_loss = F.cross_entropy(student_logits, true_labels,
                                      weight=class_weights)
        else:
            ce_loss = F.cross_entropy(student_logits, true_labels)

        student_soft = F.log_softmax(
            student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(
            teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft,
                           reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return total_loss, ce_loss.item(), kd_loss.item()


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam   = np.random.beta(alpha, alpha)
    lam   = max(lam, 1 - lam)
    idx   = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(pred, y_a, y_b, lam, class_weights=None):
    if class_weights is not None:
        loss_a = F.cross_entropy(pred, y_a, weight=class_weights)
        loss_b = F.cross_entropy(pred, y_b, weight=class_weights)
    else:
        loss_a = F.cross_entropy(pred, y_a)
        loss_b = F.cross_entropy(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


# ============================================================
# COSINE LR SCHEDULE
# ============================================================
def get_cosine_schedule(optimizer, num_epochs, warmup_epochs=2):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / \
                   max(num_epochs - warmup_epochs, 1)
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
            all_preds.append(torch.argmax(logits, 1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.vstack(all_probs)
    acc   = accuracy_score(y, preds) * 100
    f1    = f1_score(y, preds, average='macro') * 100
    return preds, probs, acc, f1


# ============================================================
# PHASE TRAINING
# ============================================================
def train_phase(student, teacher, X_train, y_train, X_val, y_val,
                param_groups, epochs, patience, phase_name,
                use_mixup=False, use_kd=True):

    print(f"\n    ── {phase_name} ──")

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train),
                      torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    optimizer    = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    scheduler    = get_cosine_schedule(optimizer, epochs)
    kd_criterion = DistillationLoss(KD_ALPHA, KD_TEMPERATURE)

    trainable = sum(p.numel() for p in student.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")

    best_val_loss = float('inf')
    best_epoch    = 0
    best_state    = None
    patience_ctr  = 0
    history       = {'train_acc': [], 'val_acc': []}

    teacher.eval()

    for epoch in range(1, epochs + 1):
        student.train()
        e_loss = e_correct = e_total = e_ce = e_kd = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if use_mixup:
                xb_mixed, ya, yb_mix, lam = mixup_data(xb, yb)
            else:
                xb_mixed, ya, yb_mix, lam = xb, yb, yb, 1.0

            optimizer.zero_grad()
            s_logits = student(xb_mixed)

            if use_kd:
                with torch.no_grad():
                    t_logits = teacher(xb_mixed)
                if use_mixup and lam < 1.0:
                    la, ca, ka = kd_criterion(
                        s_logits, t_logits, ya, class_weights)
                    lb, cb, kb = kd_criterion(
                        s_logits, t_logits, yb_mix, class_weights)
                    loss   = lam * la   + (1-lam) * lb
                    ce_val = lam * ca   + (1-lam) * cb
                    kd_val = lam * ka   + (1-lam) * kb
                else:
                    loss, ce_val, kd_val = kd_criterion(
                        s_logits, t_logits, ya, class_weights)
                e_ce += ce_val
                e_kd += kd_val
            else:
                if use_mixup and lam < 1.0:
                    loss = mixup_criterion(
                        s_logits, ya, yb_mix, lam, class_weights)
                else:
                    loss = F.cross_entropy(
                        s_logits, ya, weight=class_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss    += loss.item() * len(yb)
            e_correct += (torch.argmax(s_logits, 1) == yb).sum().item()
            e_total   += len(yb)

        scheduler.step()
        train_acc = e_correct / e_total

        # Validate
        student.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for i in range(0, len(X_val), BATCH_SIZE):
                xb = torch.FloatTensor(
                    X_val[i:i+BATCH_SIZE]).to(DEVICE)
                yb = torch.LongTensor(
                    y_val[i:i+BATCH_SIZE]).to(DEVICE)
                s_logits = student(xb)
                if use_kd:
                    t_logits = teacher(xb)
                    loss, _, _ = kd_criterion(
                        s_logits, t_logits, yb, class_weights)
                else:
                    loss = F.cross_entropy(
                        s_logits, yb, weight=class_weights)
                v_loss    += loss.item() * len(yb)
                v_correct += (torch.argmax(s_logits, 1) == yb
                              ).sum().item()
                v_total   += len(yb)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            gap  = train_acc - val_acc
            icon = "✅" if abs(gap) < 0.10 else "⚠️"
            kd_str = (f" CE:{e_ce/len(train_dl):.3f} "
                      f"KD:{e_kd/len(train_dl):.3f}") if use_kd else ""
            print(f"    Ep {epoch:3d}/{epochs} │ "
                  f"Tr: {train_acc*100:.1f}% │ "
                  f"Val: {val_acc*100:.1f}% │ "
                  f"Gap: {gap*100:+.1f}% {icon}{kd_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone()
                             for k, v in student.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    ⏹ Early stop epoch {epoch} "
                      f"(best: {best_epoch})")
                break

    student.load_state_dict(best_state)
    student = student.to(DEVICE)
    print(f"    ✅ Done (best epoch: {best_epoch})")
    return student, history


# ============================================================
# SVM TRAINING — ONE FOLD
# ============================================================
def train_svm_fold(F_train, y_train, F_val, y_val,
                   F_test, y_test, fold):
    print(f"\n    SVM training (fold {fold})...")

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=SVM_C, gamma=SVM_GAMMA, kernel=SVM_KERNEL,
                    class_weight="balanced", probability=True,
                    random_state=42))
    ])

    t0 = time.time()
    svm.fit(F_train, y_train)
    t1 = time.time()

    train_acc = accuracy_score(y_train, svm.predict(F_train)) * 100
    val_acc   = accuracy_score(y_val,   svm.predict(F_val))   * 100
    test_acc  = accuracy_score(y_test,  svm.predict(F_test))  * 100
    test_f1   = f1_score(y_test, svm.predict(F_test),
                         average='macro') * 100

    print(f"    Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}% | "
          f"Test: {test_acc:.1f}% | "
          f"F1: {test_f1:.1f}% | "
          f"Time: {t1-t0:.1f}s")

    return svm, {
        'train_acc': train_acc,
        'val_acc':   val_acc,
        'test_acc':  test_acc,
        'test_f1':   test_f1
    }


# ============================================================
# CNN TL TRAINING — ONE FOLD
# ============================================================
def train_cnn_fold(X_train, y_train, X_val, y_val,
                   X_test, y_test, fold):
    print(f"\n    CNN TL training (fold {fold})...")

    # Teacher — frozen original
    teacher = CNN_LSTM_V1().to(DEVICE)
    ckpt    = torch.load(CNN_MODEL_PATH, map_location=DEVICE,
                         weights_only=False)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Student — starts from same weights
    student = CNN_LSTM_V1().to(DEVICE)
    student.load_state_dict(ckpt['model_state_dict'])

    _, _, start_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"    Starting accuracy: {start_acc:.1f}%")

    # ── Phase 1: Head only ──
    for name, param in student.named_parameters():
        if any(k in name for k in
               ['conv', 'bn1', 'bn2', 'bn3', 'lstm', 'drop_cnn']):
            param.requires_grad = False
        else:
            param.requires_grad = True

    student, _ = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=[{'params': [p for n, p in
                       student.named_parameters()
                       if p.requires_grad], 'lr': PHASE1_LR}],
        epochs=PHASE1_EPOCHS, patience=PHASE1_PATIENCE,
        phase_name="P1: Head ❄️❄️❄️🔥",
        use_mixup=False, use_kd=True
    )
    _, _, p1_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"    Phase 1 test acc: {p1_acc:.1f}%")

    # ── Phase 2: LSTM + Head ──
    for name, param in student.named_parameters():
        if 'lstm' in name or 'drop_lstm' in name:
            param.requires_grad = True

    lstm_p = [p for n, p in student.named_parameters()
              if ('lstm' in n or 'drop_lstm' in n)
              and p.requires_grad]
    fc_p   = [p for n, p in student.named_parameters()
              if ('fc' in n or 'bn_fc' in n or 'drop_fc' in n)
              and p.requires_grad]

    student, _ = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=[{'params': lstm_p, 'lr': PHASE2_LR_LSTM},
                      {'params': fc_p,   'lr': PHASE2_LR_FC}],
        epochs=PHASE2_EPOCHS, patience=PHASE2_PATIENCE,
        phase_name="P2: LSTM+Head ❄️❄️🔥🔥",
        use_mixup=True, use_kd=True
    )
    _, _, p2_acc, _ = evaluate_cnn(student, X_test, y_test)
    print(f"    Phase 2 test acc: {p2_acc:.1f}%")

    # ── Phase 3: All layers ──
    for param in student.parameters():
        param.requires_grad = True

    cnn_p  = [p for n, p in student.named_parameters()
              if any(k in n for k in
                     ['conv', 'bn1', 'bn2', 'bn3', 'drop_cnn'])]
    lstm_p = [p for n, p in student.named_parameters()
              if 'lstm' in n or 'drop_lstm' in n]
    fc_p   = [p for n, p in student.named_parameters()
              if 'fc' in n or 'bn_fc' in n or 'drop_fc' in n]

    student, _ = train_phase(
        student, teacher, X_train, y_train, X_val, y_val,
        param_groups=[{'params': cnn_p,  'lr': PHASE3_LR_CNN},
                      {'params': lstm_p, 'lr': PHASE3_LR_LSTM},
                      {'params': fc_p,   'lr': PHASE3_LR_FC}],
        epochs=PHASE3_EPOCHS, patience=PHASE3_PATIENCE,
        phase_name="P3: All 🔥🔥🔥🔥",
        use_mixup=True, use_kd=True
    )

    _, _, test_acc, test_f1 = evaluate_cnn(student, X_test, y_test)
    _, _, train_acc, _      = evaluate_cnn(student, X_train, y_train)
    _, _, val_acc,   _      = evaluate_cnn(student, X_val,   y_val)
    gap = train_acc - test_acc

    print(f"    Phase 3 test acc: {test_acc:.1f}%")
    print(f"    Progress: {start_acc:.1f}% → "
          f"{p1_acc:.1f}% → {p2_acc:.1f}% → {test_acc:.1f}%")
    print(f"    Gap: {gap:+.1f}% "
          f"{'✅' if abs(gap) < 5 else '⚠️'}")

    return student, {
        'start_acc': start_acc,
        'p1_acc':    p1_acc,
        'p2_acc':    p2_acc,
        'train_acc': train_acc,
        'val_acc':   val_acc,
        'test_acc':  test_acc,
        'test_f1':   test_f1,
        'gap':       gap
    }


# ============================================================
# UNANIMOUS EVALUATION — ONE FOLD
# ============================================================
def evaluate_unanimous_fold(svm, cnn, X_test, y_test, fold):
    """Unanimous evaluation for one LOSO fold."""

    # SVM predictions
    F_test    = build_feature_matrix(X_test)
    F_test    = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)
    svm_probs = svm.predict_proba(F_test)
    svm_pred  = np.argmax(svm_probs, axis=1)
    svm_conf  = np.max(svm_probs, axis=1)

    # CNN predictions
    cnn_pred, cnn_probs, cnn_acc, _ = evaluate_cnn(cnn, X_test, y_test)
    cnn_conf = np.max(cnn_probs, axis=1)

    # Unanimous check
    agree_mask    = svm_pred == cnn_pred
    n_total       = len(y_test)
    n_agree       = np.sum(agree_mask)
    agree_rate    = n_agree / n_total * 100
    unan_pred     = np.where(agree_mask, svm_pred, -1)

    svm_acc = accuracy_score(y_test, svm_pred) * 100
    cnn_acc = accuracy_score(y_test, cnn_pred) * 100

    if n_agree > 0:
        unan_acc = accuracy_score(
            y_test[agree_mask], unan_pred[agree_mask]) * 100
        unan_f1  = f1_score(
            y_test[agree_mask], unan_pred[agree_mask],
            average='macro') * 100
    else:
        unan_acc = unan_f1 = 0.0

    coverage_acc = (np.sum(
        (unan_pred == y_test) & agree_mask) / n_total) * 100

    # Per-gesture
    per_gesture = {}
    for g in range(len(GESTURE_NAMES)):
        mask_g  = y_test == g
        agree_g = agree_mask & mask_g
        if mask_g.sum() == 0:
            continue
        per_gesture[GESTURE_NAMES[g]] = {
            'agreement_rate': agree_g.mean() * 100,
            'unanimous_acc':  accuracy_score(
                y_test[agree_g], unan_pred[agree_g]) * 100
                              if agree_g.sum() > 0 else 0.0,
            'svm_acc': accuracy_score(
                y_test[mask_g], svm_pred[mask_g]) * 100,
            'cnn_acc': accuracy_score(
                y_test[mask_g], cnn_pred[mask_g]) * 100
        }

    return {
        'fold':          fold,
        'svm_acc':       svm_acc,
        'cnn_acc':       cnn_acc,
        'agree_rate':    agree_rate,
        'abstain_rate':  100 - agree_rate,
        'unanimous_acc': unan_acc,
        'unanimous_f1':  unan_f1,
        'coverage_acc':  coverage_acc,
        'n_total':       n_total,
        'n_agree':       n_agree,
        'per_gesture':   per_gesture,
        'unan_pred':     unan_pred,
        'agree_mask':    agree_mask,
        'svm_pred':      svm_pred,
        'cnn_pred':      cnn_pred,
        'y_test':        y_test
    }


# ============================================================
# PRINT FOLD SUMMARY
# ============================================================
def print_fold_summary(fold_result, test_subject):
    r = fold_result
    all_pass = all(
        r['per_gesture'][g]['unanimous_acc'] >= 80
        for g in r['per_gesture']
    )
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  FOLD {fold_result['fold']:2d} — Test Subject: S{test_subject:02d}           │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  SVM Acc:        {r['svm_acc']:>6.1f}%                        │")
    print(f"  │  CNN Acc:        {r['cnn_acc']:>6.1f}%                        │")
    print(f"  │  Agreement:      {r['agree_rate']:>6.1f}%                        │")
    print(f"  │  Unanimous Acc:  {r['unanimous_acc']:>6.1f}%  (when agreed)      │")
    print(f"  │  Coverage Adj:   {r['coverage_acc']:>6.1f}%  (abstain=wrong)    │")
    print(f"  │  All ≥80%:       {'✅ YES' if all_pass else '❌ NO':<6}                         │")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  Per-Gesture:")
    print(f"  {'Gesture':<22} {'Agree%':<10} "
          f"{'Unan.Acc':<12} {'SVM':<8} {'CNN':<8} {'Status'}")
    print(f"  {'-'*68}")
    for g, vals in r['per_gesture'].items():
        status = ("✅" if vals['unanimous_acc'] >= 80 else
                  "⚠️" if vals['unanimous_acc'] >= 75 else "❌")
        print(f"  {g:<22} {vals['agreement_rate']:<10.1f} "
              f"{vals['unanimous_acc']:<12.1f} "
              f"{vals['svm_acc']:<8.1f} "
              f"{vals['cnn_acc']:<8.1f} {status}")


# ============================================================
# FINAL PLOTS — ALL FOLDS
# ============================================================
def plot_cross_subject_results(all_fold_results, best_fold_result):
    print("\n  Generating cross-subject plots...")

    subjects     = [r['fold'] for r in all_fold_results]
    svm_accs     = [r['svm_acc']       for r in all_fold_results]
    cnn_accs     = [r['cnn_acc']       for r in all_fold_results]
    unan_accs    = [r['unanimous_acc'] for r in all_fold_results]
    agree_rates  = [r['agree_rate']    for r in all_fold_results]
    coverage_acc = [r['coverage_acc']  for r in all_fold_results]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    x = np.arange(len(subjects))
    w = 0.25
    subj_labels = [f'S{s:02d}' for s in subjects]

    # ── Plot 1: Per-subject accuracy comparison ──
    axes[0, 0].bar(x - w, svm_accs,  w, label='SVM',
                   color='#FF9800', edgecolor='black', alpha=0.8)
    axes[0, 0].bar(x,     cnn_accs,  w, label='CNN',
                   color='#2196F3', edgecolor='black', alpha=0.8)
    axes[0, 0].bar(x + w, unan_accs, w, label='Unanimous',
                   color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[0, 0].axhline(80, color='red', linestyle='--',
                        alpha=0.5, label='Target 80%')
    axes[0, 0].axhline(np.mean(unan_accs), color='green',
                        linestyle='-', alpha=0.5, linewidth=2,
                        label=f'Unan. Mean: {np.mean(unan_accs):.1f}%')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(subj_labels, rotation=45)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Per-Subject: SVM vs CNN vs Unanimous',
                          fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # ── Plot 2: Agreement rate per subject ──
    bar_colors = ['#4CAF50' if a >= 70 else
                  '#FF9800' if a >= 50 else
                  '#f44336' for a in agree_rates]
    bars = axes[0, 1].bar(subj_labels, agree_rates,
                           color=bar_colors, edgecolor='black')
    for bar, val in zip(bars, agree_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                         val + 0.5, f'{val:.1f}%',
                         ha='center', fontsize=8,
                         fontweight='bold')
    axes[0, 1].axhline(np.mean(agree_rates), color='blue',
                        linestyle='--',
                        label=f'Mean: {np.mean(agree_rates):.1f}%')
    axes[0, 1].set_ylabel('Agreement Rate (%)')
    axes[0, 1].set_title('Per-Subject Agreement Rate',
                          fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # ── Plot 3: Coverage-adjusted vs Unanimous ──
    axes[0, 2].plot(subj_labels, unan_accs, 'go-',
                     linewidth=2, markersize=8,
                     label='Unanimous (when agreed)')
    axes[0, 2].plot(subj_labels, coverage_acc, 'rs--',
                     linewidth=2, markersize=8,
                     label='Coverage-Adjusted')
    axes[0, 2].fill_between(range(len(subjects)),
                              unan_accs, coverage_acc,
                              alpha=0.1, color='gray',
                              label='Abstention cost')
    axes[0, 2].axhline(80, color='red', linestyle='--',
                        alpha=0.5, label='Target')
    axes[0, 2].set_xticks(range(len(subjects)))
    axes[0, 2].set_xticklabels(subj_labels, rotation=45)
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Unanimous vs Coverage-Adjusted\n'
                          '(Gap = abstention cost)',
                          fontweight='bold')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_ylim(0, 105)
    axes[0, 2].grid(alpha=0.3)

    # ── Plot 4: Per-gesture unanimous accuracy (heatmap) ──
    gesture_matrix = np.zeros(
        (len(GESTURE_NAMES), len(subjects)))
    for col, r in enumerate(all_fold_results):
        for row, g in enumerate(GESTURE_NAMES):
            if g in r['per_gesture']:
                gesture_matrix[row, col] = \
                    r['per_gesture'][g]['unanimous_acc']

    sns.heatmap(gesture_matrix, annot=True, fmt='.0f',
                cmap='RdYlGn', ax=axes[1, 0],
                xticklabels=subj_labels,
                yticklabels=[n[:10] for n in GESTURE_NAMES],
                vmin=0, vmax=100,
                linewidths=0.5)
    axes[1, 0].set_title('Per-Gesture Unanimous Accuracy\n'
                          '(rows=gesture, cols=subject)',
                          fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # ── Plot 5: Summary box plots ──
    box_data  = [svm_accs, cnn_accs, unan_accs]
    bp = axes[1, 1].boxplot(
        box_data, patch_artist=True,
        labels=['SVM', 'CNN', 'Unanimous'])
    colors_bp = ['#FF9800', '#2196F3', '#4CAF50']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1, 1].axhline(80, color='red', linestyle='--',
                        alpha=0.5, label='Target 80%')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy Distribution\nAcross All Subjects',
                          fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # ── Plot 6: Best fold confusion matrix ──
    best_r    = best_fold_result
    agree_idx = best_r['agree_mask']
    if agree_idx.sum() > 0:
        cm = confusion_matrix(
            best_r['y_test'][agree_idx],
            best_r['unan_pred'][agree_idx])
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='RdYlGn', ax=axes[1, 2],
                    xticklabels=[n[:8] for n in GESTURE_NAMES],
                    yticklabels=[n[:8] for n in GESTURE_NAMES],
                    vmin=0, vmax=np.max(cm))
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('True')
        axes[1, 2].set_title(
            f'Best Fold (S{best_r["fold"]:02d}) '
            f'Confusion Matrix\n'
            f'Unanimous Acc: {best_r["unanimous_acc"]:.1f}%',
            fontweight='bold')

    plt.suptitle('V3-TL Unanimous — Cross-Subject LOSO Results',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = f"{SAVE_DIR}/cross_subject_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================
# MAIN — LOSO CROSS-SUBJECT LOOP
# ============================================================
def main():
    print("=" * 70)
    print("  V3-TL UNANIMOUS — CROSS-SUBJECT (LOSO)")
    print("  Leave-One-Subject-Out Validation")
    print("  Tests: Can model generalize to a NEW user?")
    print("=" * 70)
    print(f"\n  Device:  {DEVICE}")
    print(f"  CNN:     {CNN_MODEL_PATH}")
    print(f"  Method:  LOSO × {len(SUBJECTS)} folds")
    print(f"  Output:  UNANIMOUS (SVM == CNN required)")

    # ── Load all subjects ──
    subject_data = load_all_subjects()

    all_fold_results = []
    best_fold        = None
    best_unan_acc    = 0.0

    # ── LOSO Loop ──
    for fold, test_subj in enumerate(SUBJECTS, 1):

        if test_subj not in subject_data:
            print(f"\n  ⚠️ Subject {test_subj} not found, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  FOLD {fold:2d}/{len(SUBJECTS)} — "
              f"Test Subject: S{test_subj:02d} "
              f"(Train: S{[s for s in SUBJECTS if s != test_subj]})")
        print(f"{'='*70}")

        # ── Build split ──
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test) = build_loso_split(subject_data, test_subj)

        print(f"\n  Split:")
        print(f"    Train: {len(y_train)} samples "
              f"(subjects: "
              f"{[s for s in SUBJECTS if s != test_subj]})")
        print(f"    Val:   {len(y_val)} samples")
        print(f"    Test:  {len(y_test)} samples "
              f"(subject S{test_subj:02d})")

        # ── SVM Features ──
        print(f"\n  Extracting features...")
        F_train = build_feature_matrix(X_train)
        F_val   = build_feature_matrix(X_val)
        F_test  = build_feature_matrix(X_test)
        for F in [F_train, F_val, F_test]:
            np.nan_to_num(F, copy=False,
                          nan=0.0, posinf=0.0, neginf=0.0)

        # ── Train SVM ──
        svm, svm_results = train_svm_fold(
            F_train, y_train, F_val, y_val,
            F_test, y_test, fold)

        # ── Train CNN ──
        cnn, cnn_results = train_cnn_fold(
            X_train, y_train, X_val, y_val,
            X_test, y_test, fold)

        # ── Evaluate Unanimous ──
        fold_result = evaluate_unanimous_fold(
            svm, cnn, X_test, y_test, fold)

        print_fold_summary(fold_result, test_subj)

        # ── Save fold models ──
        joblib.dump(svm,
            f"models/cross_subject/fold{fold:02d}_svm.pkl")
        torch.save({
            'model_state_dict': cnn.state_dict(),
            'fold':             fold,
            'test_subject':     test_subj,
            'unanimous_acc':    fold_result['unanimous_acc']
        }, f"models/cross_subject/fold{fold:02d}_cnn.pth")

        all_fold_results.append(fold_result)

        # Track best fold
        if fold_result['unanimous_acc'] > best_unan_acc:
            best_unan_acc  = fold_result['unanimous_acc']
            best_fold      = fold_result
            # Save as best model
            joblib.dump(svm,  "models/v3_tl_unanimous_svm.pkl")
            torch.save({
                'model_state_dict': cnn.state_dict(),
                'accuracy': fold_result['unanimous_acc'] / 100,
                'fold':     fold,
                'test_subject': test_subj,
                'cross_subject': True
            }, "models/v3_tl_unanimous_cnn.pth")
            print(f"\n  🌟 New best! "
                  f"Fold {fold} unanimous: {best_unan_acc:.1f}%")

    # ============================================================
    # FINAL SUMMARY ACROSS ALL FOLDS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT FINAL SUMMARY")
    print(f"{'='*70}")

    svm_accs  = [r['svm_acc']       for r in all_fold_results]
    cnn_accs  = [r['cnn_acc']       for r in all_fold_results]
    unan_accs = [r['unanimous_acc'] for r in all_fold_results]
    agr_rates = [r['agree_rate']    for r in all_fold_results]
    cov_accs  = [r['coverage_acc']  for r in all_fold_results]

    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  CROSS-SUBJECT RESULTS (LOSO, n={len(all_fold_results)} subjects)         ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Metric               Mean ± Std      Min      Max          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  SVM Accuracy    {np.mean(svm_accs):>6.1f}% ± {np.std(svm_accs):>4.1f}%  {np.min(svm_accs):>6.1f}%  {np.max(svm_accs):>6.1f}%  ║
  ║  CNN Accuracy    {np.mean(cnn_accs):>6.1f}% ± {np.std(cnn_accs):>4.1f}%  {np.min(cnn_accs):>6.1f}%  {np.max(cnn_accs):>6.1f}%  ║
  ║  Agreement Rate  {np.mean(agr_rates):>6.1f}% ± {np.std(agr_rates):>4.1f}%  {np.min(agr_rates):>6.1f}%  {np.max(agr_rates):>6.1f}%  ║
  ║  Unanimous Acc   {np.mean(unan_accs):>6.1f}% ± {np.std(unan_accs):>4.1f}%  {np.min(unan_accs):>6.1f}%  {np.max(unan_accs):>6.1f}%  ║
  ║  Coverage Adj    {np.mean(cov_accs):>6.1f}% ± {np.std(cov_accs):>4.1f}%  {np.min(cov_accs):>6.1f}%  {np.max(cov_accs):>6.1f}%  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Best Fold:  S{best_fold['fold']:02d}  Unanimous: {best_unan_acc:.1f}%               ║
  ╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"  Per-Fold Breakdown:")
    print(f"  {'Fold':<8} {'Test Subj':<12} {'SVM':<8} "
          f"{'CNN':<8} {'Agree%':<10} {'Unan.Acc':<12} {'Status'}")
    print(f"  {'-'*65}")

    for r in all_fold_results:
        all_pass = all(
            r['per_gesture'][g]['unanimous_acc'] >= 80
            for g in r['per_gesture'])
        status = "✅" if all_pass else "⚠️"
        print(f"  {r['fold']:<8} S{r['fold']:02d}         "
              f"{r['svm_acc']:<8.1f} {r['cnn_acc']:<8.1f} "
              f"{r['agree_rate']:<10.1f} {r['unanimous_acc']:<12.1f} "
              f"{status}")

    # ── Save final config ──
    config = {
        'version':   'V3-TL Unanimous Cross-Subject',
        'strategy':  'LOSO + Gradual Unfreeze + KD + Unanimous',
        'n_folds':   len(all_fold_results),
        'summary': {
            'svm_mean':  float(np.mean(svm_accs)),
            'cnn_mean':  float(np.mean(cnn_accs)),
            'unan_mean': float(np.mean(unan_accs)),
            'unan_std':  float(np.std(unan_accs)),
            'agree_mean': float(np.mean(agr_rates)),
            'best_fold': int(best_fold['fold']),
            'best_unan': float(best_unan_acc)
        },
        'per_fold': [{
            'fold':       r['fold'],
            'svm_acc':    r['svm_acc'],
            'cnn_acc':    r['cnn_acc'],
            'agree_rate': r['agree_rate'],
            'unan_acc':   r['unanimous_acc'],
            'coverage':   r['coverage_acc']
        } for r in all_fold_results]
    }

    with open(f"models/v3_tl_unanimous_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n  ✅ Config saved: models/v3_tl_unanimous_config.json")

    # ── Plots ──
    plot_cross_subject_results(all_fold_results, best_fold)

    # ── Interpretation ──
    mean_unan = np.mean(unan_accs)
    print(f"\n  📊 WHAT THIS MEANS:")
    print(f"  The model was tested on each subject as if they were")
    print(f"  a completely NEW user — never seen during training.")
    print(f"  Average unanimous accuracy: {mean_unan:.1f}%")

    if mean_unan >= 90:
        print(f"\n  🌟 EXCELLENT cross-subject generalization!")
    elif mean_unan >= 80:
        print(f"\n  ✅ GOOD cross-subject generalization.")
    elif mean_unan >= 70:
        print(f"\n  👍 MODERATE — consider per-user fine-tuning.")
    else:
        print(f"\n  ⚠️ LOW — significant subject variability.")
        print(f"  Consider: data augmentation, domain adaptation,")
        print(f"  or per-subject calibration steps.")

    print(f"\n✅ Cross-Subject TL complete!\n")


if __name__ == "__main__":
    main()