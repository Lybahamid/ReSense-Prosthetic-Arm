"""
test_menu.py

Menu-Driven Testing Interface for V3-TL Unanimous
==================================================
Interactive testing of the unanimous agreement system.

Menu Options:
  1. Quick Test       — Overall accuracy snapshot
  2. Per-Gesture      — Detailed per-gesture breakdown
  3. Model Comparison — SVM vs CNN vs Unanimous
  4. Confidence       — Confidence analysis
  5. Disagreement     — What causes disagreements
  6. Per-Subject      — Performance per subject
  7. Stress Test      — Edge cases & hard samples
  8. Full Report      — Everything at once
  0. Exit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import time

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

TEST_SIZE  = 0.20
BATCH_SIZE = 64

CNN_PATH = "models/v3_tl_unanimous_cnn.pth"
SVM_PATH = "models/v3_tl_unanimous_svm.pkl"

SAVE_DIR = "results/test_menu"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CNN ARCHITECTURE
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
# DATA & MODEL LOADING
# ============================================================
def load_data():
    print("\n  ⏳ Loading data...")
    emg_all, labels_all, subject_ids = [], [], []

    for subj in SUBJECTS:
        for ex in EXERCISES:
            file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
            try:
                emg, labels, _ = load_subject(file)
                emg = preprocess_emg(emg)
                windows, win_labels = window_data(emg, labels)
                emg_all.append(windows)
                labels_all.append(win_labels)
                subject_ids.extend([subj] * len(win_labels))
            except:
                pass

    X = np.vstack(emg_all)
    y = np.hstack(labels_all)
    s = np.array(subject_ids)

    mask      = np.isin(y, KEEP_GESTURES)
    X, y, s   = X[mask], y[mask], s[mask]
    label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
    y         = np.array([label_map[l] for l in y])

    classes   = np.unique(y)
    min_count = min(np.sum(y == c) for c in classes)
    rng       = np.random.default_rng(42)

    X_bal, y_bal, s_bal = [], [], []
    for c in classes:
        idx    = np.where(y == c)[0]
        chosen = rng.choice(idx, min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])
        s_bal.append(s[chosen])

    X = np.vstack(X_bal).astype(np.float32)
    y = np.hstack(y_bal)
    s = np.hstack(s_bal)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=TEST_SIZE, stratify=y, random_state=42
    )

    print(f"  ✅ Data loaded — Test set: {len(y_test)} samples")
    return X_train, X_test, y_train, y_test, s_train, s_test


def load_models():
    print("\n  ⏳ Loading models...")
    svm = joblib.load(SVM_PATH)

    cnn  = CNN_LSTM_V1().to(DEVICE)
    ckpt = torch.load(CNN_PATH, map_location=DEVICE, weights_only=False)
    cnn.load_state_dict(ckpt['model_state_dict'])
    cnn.eval()

    print(f"  ✅ Models loaded (Device: {DEVICE})")
    return svm, cnn


def get_predictions(svm, cnn, X, y):
    """Get all predictions from both models."""
    # CNN
    cnn.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch  = torch.FloatTensor(X[i:i+BATCH_SIZE]).to(DEVICE)
            logits = cnn(batch)
            probs  = F.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits, 1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    cnn_pred  = np.concatenate(all_preds)
    cnn_probs = np.vstack(all_probs)

    # SVM
    F_feat    = build_feature_matrix(X)
    F_feat    = np.nan_to_num(F_feat, nan=0.0, posinf=0.0, neginf=0.0)
    svm_probs = svm.predict_proba(F_feat)
    svm_pred  = np.argmax(svm_probs, axis=1)

    # Unanimous
    agree_mask    = svm_pred == cnn_pred
    unan_pred     = np.where(agree_mask, cnn_pred, -1)
    svm_conf      = np.max(svm_probs, axis=1)
    cnn_conf      = np.max(cnn_probs, axis=1)

    return {
        'svm_pred':   svm_pred,   'svm_probs': svm_probs,
        'cnn_pred':   cnn_pred,   'cnn_probs': cnn_probs,
        'unan_pred':  unan_pred,  'agree_mask': agree_mask,
        'svm_conf':   svm_conf,   'cnn_conf':   cnn_conf,
        'y_true':     y
    }


# ============================================================
# MENU OPTIONS
# ============================================================

# ── Option 1: Quick Test ──
def quick_test(preds):
    print("\n" + "=" * 55)
    print("  QUICK TEST — Overall Accuracy Snapshot")
    print("=" * 55)

    y        = preds['y_true']
    agree    = preds['agree_mask']
    unan     = preds['unan_pred']

    svm_acc  = accuracy_score(y, preds['svm_pred'])  * 100
    cnn_acc  = accuracy_score(y, preds['cnn_pred'])  * 100
    agree_rate = agree.mean() * 100

    if agree.sum() > 0:
        unan_acc = accuracy_score(y[agree], unan[agree]) * 100
    else:
        unan_acc = 0.0

    coverage_acc = (np.sum((unan == y) & agree) / len(y)) * 100

    print(f"""
  ┌─────────────────────────────────────────────┐
  │  MODEL              ACCURACY                │
  ├─────────────────────────────────────────────┤
  │  SVM (standalone)   {svm_acc:<6.1f}%               │
  │  CNN (standalone)   {cnn_acc:<6.1f}%               │
  ├─────────────────────────────────────────────┤
  │  Agreement Rate     {agree_rate:<6.1f}%               │
  │  Unanimous Acc      {unan_acc:<6.1f}%  (when agreed) │
  │  Coverage-Adjusted  {coverage_acc:<6.1f}%  (abstain=wrong)│
  │  Abstention Rate    {100-agree_rate:<6.1f}%  (hold state)  │
  └─────────────────────────────────────────────┘
    """)

    quality = ("🌟 EXCELLENT" if unan_acc >= 95 else
               "✅ GREAT"    if unan_acc >= 90 else
               "👍 GOOD"     if unan_acc >= 85 else
               "⚠️ NEEDS WORK")
    print(f"  Unanimous Quality: {quality}")
    print(f"  All gestures ≥80%: checking...")

    all_pass = True
    for g in range(len(GESTURE_NAMES)):
        mask_g = y == g
        agree_g = agree & mask_g
        if agree_g.sum() > 0:
            acc_g = accuracy_score(y[agree_g], unan[agree_g]) * 100
            icon  = "✅" if acc_g >= 80 else "❌"
            if acc_g < 80:
                all_pass = False
            print(f"    {GESTURE_NAMES[g]:<20} {acc_g:.1f}% {icon}")

    print(f"\n  {'✅ ALL gestures ≥80%!' if all_pass else '❌ Some below 80%'}")
    input("\n  Press Enter to return to menu...")


# ── Option 2: Per-Gesture ──
def per_gesture(preds):
    print("\n" + "=" * 70)
    print("  PER-GESTURE DETAILED BREAKDOWN")
    print("=" * 70)

    y      = preds['y_true']
    agree  = preds['agree_mask']
    unan   = preds['unan_pred']

    print(f"\n  {'Gesture':<22} {'Agree%':<10} {'Unan.Acc':<12}"
          f"{'SVM':<10} {'CNN':<10} {'Support':<8} {'Status'}")
    print(f"  {'-'*75}")

    for g in range(len(GESTURE_NAMES)):
        mask_g  = y == g
        total_g = mask_g.sum()
        if total_g == 0:
            continue

        agree_g    = agree & mask_g
        agree_rate = agree_g.mean() * 100 if mask_g.sum() > 0 else 0

        unan_acc = (accuracy_score(y[agree_g], unan[agree_g]) * 100
                    if agree_g.sum() > 0 else 0.0)
        svm_acc  = accuracy_score(y[mask_g], preds['svm_pred'][mask_g]) * 100
        cnn_acc  = accuracy_score(y[mask_g], preds['cnn_pred'][mask_g]) * 100

        status = ("✅" if unan_acc >= 80 else
                  "⚠️" if unan_acc >= 75 else "❌")

        print(f"  {GESTURE_NAMES[g]:<22} {agree_rate:<10.1f} "
              f"{unan_acc:<12.1f}{svm_acc:<10.1f} {cnn_acc:<10.1f} "
              f"{total_g:<8} {status}")

    print(f"\n  Full Classification Report (Unanimous only):")
    print(f"  {'-'*55}")
    agreed_mask_bool = agree
    if agreed_mask_bool.sum() > 0:
        print(classification_report(
            y[agreed_mask_bool],
            unan[agreed_mask_bool],
            target_names=GESTURE_NAMES
        ))

    input("\n  Press Enter to return to menu...")


# ── Option 3: Model Comparison ──
def model_comparison(preds):
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON: SVM vs CNN vs Unanimous")
    print("=" * 55)

    y     = preds['y_true']
    agree = preds['agree_mask']
    unan  = preds['unan_pred']

    svm_acc  = accuracy_score(y, preds['svm_pred'])  * 100
    cnn_acc  = accuracy_score(y, preds['cnn_pred'])  * 100
    svm_f1   = f1_score(y, preds['svm_pred'], average='macro') * 100
    cnn_f1   = f1_score(y, preds['cnn_pred'], average='macro') * 100

    if agree.sum() > 0:
        unan_acc = accuracy_score(y[agree], unan[agree]) * 100
        unan_f1  = f1_score(y[agree], unan[agree], average='macro') * 100
    else:
        unan_acc = unan_f1 = 0.0

    print(f"\n  {'Model':<25} {'Accuracy':<12} {'F1':<12} {'Coverage'}")
    print(f"  {'-'*60}")
    print(f"  {'SVM (standalone)':<25} {svm_acc:<12.1f} {svm_f1:<12.1f} 100%")
    print(f"  {'CNN (standalone)':<25} {cnn_acc:<12.1f} {cnn_f1:<12.1f} 100%")
    print(f"  {'Unanimous':<25} {unan_acc:<12.1f} {unan_f1:<12.1f} "
          f"{agree.mean()*100:.1f}%")

    print(f"\n  WHO WINS PER GESTURE:")
    print(f"  {'Gesture':<22} {'SVM':<10} {'CNN':<10} {'Winner'}")
    print(f"  {'-'*50}")

    for g in range(len(GESTURE_NAMES)):
        mask_g  = y == g
        if mask_g.sum() == 0:
            continue
        svm_g = accuracy_score(y[mask_g], preds['svm_pred'][mask_g]) * 100
        cnn_g = accuracy_score(y[mask_g], preds['cnn_pred'][mask_g]) * 100
        winner = "CNN 🧠" if cnn_g > svm_g else "SVM 📊" if svm_g > cnn_g else "TIE 🤝"
        print(f"  {GESTURE_NAMES[g]:<22} {svm_g:<10.1f} {cnn_g:<10.1f} {winner}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = ['SVM', 'CNN', 'Unanimous']
    accs   = [svm_acc, cnn_acc, unan_acc]
    f1s    = [svm_f1,  cnn_f1,  unan_f1]
    colors = ['#FF9800', '#2196F3', '#4CAF50']

    for ax, values, title in zip(axes,
        [accs, f1s], ['Accuracy (%)', 'Macro F1 (%)']):
        bars = ax.bar(models, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.3, f'{val:.1f}%',
                    ha='center', fontweight='bold')
        ax.set_ylim(50, 105)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{SAVE_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Plot saved: {path}")
    input("\n  Press Enter to return to menu...")


# ── Option 4: Confidence Analysis ──
def confidence_analysis(preds):
    print("\n" + "=" * 55)
    print("  CONFIDENCE ANALYSIS")
    print("=" * 55)

    agree = preds['agree_mask']
    svm_c = preds['svm_conf']
    cnn_c = preds['cnn_conf']

    print(f"\n  {'Situation':<25} {'SVM Conf':<15} {'CNN Conf'}")
    print(f"  {'-'*50}")
    print(f"  {'When AGREED':<25} "
          f"{svm_c[agree].mean()*100:<15.1f} "
          f"{cnn_c[agree].mean()*100:.1f}%")
    print(f"  {'When DISAGREED':<25} "
          f"{svm_c[~agree].mean()*100:<15.1f} "
          f"{cnn_c[~agree].mean()*100:.1f}%")
    print(f"  {'Overall':<25} "
          f"{svm_c.mean()*100:<15.1f} "
          f"{cnn_c.mean()*100:.1f}%")

    print(f"\n  CONFIDENCE THRESHOLD ANALYSIS:")
    print(f"  What if we only act when BOTH models exceed threshold?")
    print(f"\n  {'Threshold':<15} {'Samples Acted':<18} "
          f"{'Coverage':<12} {'Est. Accuracy'}")
    print(f"  {'-'*60}")

    y     = preds['y_true']
    unan  = preds['unan_pred']

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        high_conf = (svm_c >= thresh) & (cnn_c >= thresh) & agree
        if high_conf.sum() > 0:
            acc      = accuracy_score(y[high_conf], unan[high_conf]) * 100
            coverage = high_conf.mean() * 100
            print(f"  {thresh:<15.1f} {high_conf.sum():<18} "
                  f"{coverage:<12.1f} {acc:.1f}%")
        else:
            print(f"  {thresh:<15.1f} {'0':<18} {'0.0':<12} N/A")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (model, conf) in zip(axes, [
        ('SVM', svm_c), ('CNN', cnn_c)
    ]):
        ax.hist(conf[agree],  bins=30, alpha=0.7,
                color='#4CAF50', label='Agreed',   density=True)
        ax.hist(conf[~agree], bins=30, alpha=0.7,
                color='#f44336', label='Disagreed', density=True)
        ax.axvline(conf[agree].mean(),  color='green',
                   linestyle='--', linewidth=2,
                   label=f'Agreed avg: {conf[agree].mean():.2f}')
        ax.axvline(conf[~agree].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Disagreed avg: {conf[~agree].mean():.2f}')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Density')
        ax.set_title(f'{model} Confidence Distribution', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{SAVE_DIR}/confidence_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Plot saved: {path}")
    input("\n  Press Enter to return to menu...")


# ── Option 5: Disagreement Analysis ──
def disagreement_analysis(preds):
    print("\n" + "=" * 55)
    print("  DISAGREEMENT ANALYSIS")
    print("=" * 55)

    y      = preds['y_true']
    agree  = preds['agree_mask']
    s_pred = preds['svm_pred']
    c_pred = preds['cnn_pred']

    disagree_idx = np.where(~agree)[0]
    n_disagree   = len(disagree_idx)

    svm_right = np.sum(s_pred[disagree_idx] == y[disagree_idx])
    cnn_right = np.sum(c_pred[disagree_idx] == y[disagree_idx])
    both_wrong = n_disagree - svm_right - cnn_right + np.sum(
        (s_pred[disagree_idx] == y[disagree_idx]) &
        (c_pred[disagree_idx] == y[disagree_idx])
    )

    print(f"\n  Total disagreements: {n_disagree} "
          f"({n_disagree/len(y)*100:.1f}% of test set)")
    print(f"\n  When they disagreed:")
    print(f"    CNN was right:  {cnn_right} ({cnn_right/n_disagree*100:.1f}%)")
    print(f"    SVM was right:  {svm_right} ({svm_right/n_disagree*100:.1f}%)")
    print(f"    Both wrong:     {both_wrong} ({both_wrong/n_disagree*100:.1f}%)")

    print(f"\n  DISAGREEMENT HEATMAP (SVM says row, CNN says col):")
    disagree_matrix = np.zeros(
        (len(GESTURE_NAMES), len(GESTURE_NAMES)), dtype=int)
    for idx in disagree_idx:
        disagree_matrix[s_pred[idx], c_pred[idx]] += 1

    print(f"\n  {'':>12}", end="")
    for name in GESTURE_SHORT:
        print(f"  {name:>8}", end="")
    print()
    for i, name in enumerate(GESTURE_SHORT):
        print(f"  {name:>12}", end="")
        for j in range(len(GESTURE_NAMES)):
            val = disagree_matrix[i, j]
            print(f"  {val:>8}", end="")
        print()

    print(f"\n  Most common disagreement pairs:")
    pairs = []
    for i in range(len(GESTURE_NAMES)):
        for j in range(len(GESTURE_NAMES)):
            if i != j and disagree_matrix[i, j] > 0:
                pairs.append((disagree_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    for count, i, j in pairs[:5]:
        print(f"    SVM→{GESTURE_SHORT[i]:<8} CNN→{GESTURE_SHORT[j]:<8} "
              f"{count} times")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(disagree_matrix, annot=True, fmt='d',
                cmap='Reds', ax=ax,
                xticklabels=[n[:8] for n in GESTURE_NAMES],
                yticklabels=[n[:8] for n in GESTURE_NAMES])
    ax.set_xlabel('CNN Prediction')
    ax.set_ylabel('SVM Prediction')
    ax.set_title(f'Disagreement Heatmap\n'
                 f'Total: {n_disagree} disagreements',
                 fontweight='bold')
    plt.tight_layout()
    path = f"{SAVE_DIR}/disagreement_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Plot saved: {path}")
    input("\n  Press Enter to return to menu...")


# ── Option 6: Per-Subject ──
def per_subject(svm, cnn, X_test, y_test, s_test):
    print("\n" + "=" * 65)
    print("  PER-SUBJECT PERFORMANCE")
    print("=" * 65)

    print(f"\n  {'Subject':<12} {'Samples':<10} {'CNN':<10} "
          f"{'SVM':<10} {'Agree%':<10} {'Unan.Acc':<12} {'Status'}")
    print(f"  {'-'*70}")

    subject_results = {}

    for subj in sorted(np.unique(s_test)):
        mask = s_test == subj
        if mask.sum() == 0:
            continue

        Xs = X_test[mask]
        ys = y_test[mask]

        # CNN
        cnn.eval()
        c_preds, c_probs = [], []
        with torch.no_grad():
            for i in range(0, len(Xs), BATCH_SIZE):
                batch  = torch.FloatTensor(Xs[i:i+BATCH_SIZE]).to(DEVICE)
                logits = cnn(batch)
                c_preds.append(torch.argmax(logits, 1).cpu().numpy())
                c_probs.append(F.softmax(logits, 1).cpu().numpy())
        cnn_pred  = np.concatenate(c_preds)

        # SVM
        Fs       = build_feature_matrix(Xs)
        Fs       = np.nan_to_num(Fs, nan=0.0, posinf=0.0, neginf=0.0)
        svm_pred = np.argmax(svm.predict_proba(Fs), axis=1)

        # Unanimous
        agree_s  = svm_pred == cnn_pred
        unan_s   = np.where(agree_s, cnn_pred, -1)

        cnn_acc  = accuracy_score(ys, cnn_pred)  * 100
        svm_acc  = accuracy_score(ys, svm_pred)  * 100
        agree_rt = agree_s.mean() * 100
        unan_acc = (accuracy_score(ys[agree_s], unan_s[agree_s]) * 100
                    if agree_s.sum() > 0 else 0.0)
        status   = "✅" if unan_acc >= 80 else "⚠️" if unan_acc >= 70 else "❌"

        subject_results[subj] = {
            'cnn': cnn_acc, 'svm': svm_acc,
            'agree': agree_rt, 'unan': unan_acc
        }

        print(f"  Subject {subj:<5} {mask.sum():<10} {cnn_acc:<10.1f} "
              f"{svm_acc:<10.1f} {agree_rt:<10.1f} {unan_acc:<12.1f} {status}")

    # Plot
    subjects   = sorted(subject_results.keys())
    unan_accs  = [subject_results[s]['unan']  for s in subjects]
    cnn_accs   = [subject_results[s]['cnn']   for s in subjects]
    agree_rts  = [subject_results[s]['agree'] for s in subjects]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(subjects))
    w = 0.35

    axes[0].bar(x - w/2, cnn_accs,  w, label='CNN',
                color='#2196F3', edgecolor='black')
    axes[0].bar(x + w/2, unan_accs, w, label='Unanimous',
                color='#4CAF50', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'S{s}' for s in subjects])
    axes[0].axhline(80, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylim(50, 105)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Per-Subject Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, agree_rts, color='#FF9800', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'S{s}' for s in subjects])
    axes[1].set_ylabel('Agreement Rate (%)')
    axes[1].set_title('Per-Subject Agreement Rate', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Subject Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{SAVE_DIR}/per_subject.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Plot saved: {path}")
    input("\n  Press Enter to return to menu...")


# ── Option 7: Stress Test ──
def stress_test(preds):
    print("\n" + "=" * 55)
    print("  STRESS TEST — Edge Cases & Hard Samples")
    print("=" * 55)

    y      = preds['y_true']
    agree  = preds['agree_mask']
    unan   = preds['unan_pred']
    svm_c  = preds['svm_conf']
    cnn_c  = preds['cnn_conf']

    # ── Test 1: Low confidence predictions ──
    print(f"\n  TEST 1: Low Confidence Samples")
    print(f"  (Samples where CNN confidence < 0.7)")
    low_conf = cnn_c < 0.7
    if low_conf.sum() > 0:
        agree_lc = agree[low_conf].mean() * 100
        print(f"    Low confidence samples: {low_conf.sum()}")
        print(f"    Agreement rate here:    {agree_lc:.1f}%")
        print(f"    → System abstains on most of these ✅" 
              if agree_lc < 60 else
              f"    → Still agreeing on some low confidence ⚠️")
    else:
        print(f"    No low confidence samples found ✅")

    # ── Test 2: Hard gestures ──
    print(f"\n  TEST 2: Hardest Gestures (lowest CNN accuracy)")
    accs = []
    for g in range(len(GESTURE_NAMES)):
        mask_g = y == g
        if mask_g.sum() > 0:
            acc = accuracy_score(y[mask_g], preds['cnn_pred'][mask_g]) * 100
            accs.append((acc, g))
    accs.sort()
    for acc, g in accs[:3]:
        print(f"    {GESTURE_NAMES[g]:<20}: {acc:.1f}% CNN accuracy")

    # ── Test 3: Consecutive disagreements ──
    print(f"\n  TEST 3: Disagreement Streaks")
    print(f"  (In real-time, consecutive disagreements = hold state)")
    disagree = (~agree).astype(int)
    max_streak, curr = 0, 0
    for d in disagree:
        curr = curr + 1 if d else 0
        max_streak = max(max_streak, curr)
    avg_streak = disagree.sum() / max(1, np.sum(np.diff(
        np.concatenate([[0], disagree, [0]])) == 1))
    print(f"    Max consecutive disagreements: {max_streak}")
    print(f"    Avg disagreement streak:       {avg_streak:.1f}")

    # ── Test 4: False confidence ──
    print(f"\n  TEST 4: False High Confidence")
    print(f"  (High CNN confidence but WRONG prediction)")
    cnn_wrong_highconf = (cnn_c > 0.9) & \
                         (preds['cnn_pred'] != y)
    print(f"    CNN confident (>90%) but wrong: {cnn_wrong_highconf.sum()}")
    print(f"    → These would be caught by SVM disagreement: "
          f"{(cnn_wrong_highconf & ~agree).sum()}")

    input("\n  Press Enter to return to menu...")


# ── Option 8: Full Report ──
def full_report(preds, svm, cnn, X_test, y_test, s_test):
    print("\n  ⏳ Generating full report...")

    quick_test(preds)
    per_gesture(preds)
    model_comparison(preds)
    confidence_analysis(preds)
    disagreement_analysis(preds)
    per_subject(svm, cnn, X_test, y_test, s_test)

    # Save JSON report
    y     = preds['y_true']
    agree = preds['agree_mask']
    unan  = preds['unan_pred']

    report = {
        'svm_acc':     float(accuracy_score(y, preds['svm_pred']) * 100),
        'cnn_acc':     float(accuracy_score(y, preds['cnn_pred']) * 100),
        'agree_rate':  float(agree.mean() * 100),
        'unan_acc':    float(accuracy_score(y[agree], unan[agree]) * 100)
                       if agree.sum() > 0 else 0.0,
        'abstention':  float((~agree).mean() * 100),
        'per_gesture': {}
    }

    for g in range(len(GESTURE_NAMES)):
        mask_g  = y == g
        agree_g = agree & mask_g
        report['per_gesture'][GESTURE_NAMES[g]] = {
            'agreement_rate': float(agree_g.mean() * 100)
                              if mask_g.sum() > 0 else 0.0,
            'unanimous_acc':  float(accuracy_score(
                              y[agree_g], unan[agree_g]) * 100)
                              if agree_g.sum() > 0 else 0.0
        }

    path = f"{SAVE_DIR}/full_report.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ Full report saved: {path}")
    input("\n  Press Enter to return to menu...")


# ============================================================
# MAIN MENU
# ============================================================
def print_menu():
    print("\n" + "=" * 55)
    print("  V3-TL UNANIMOUS — TEST MENU")
    print("  EMG Gesture Classification — ReSense Project")
    print("=" * 55)
    print("""
  1. Quick Test         — Overall accuracy snapshot
  2. Per-Gesture        — Detailed gesture breakdown
  3. Model Comparison   — SVM vs CNN vs Unanimous
  4. Confidence         — Confidence analysis
  5. Disagreement       — What causes disagreements
  6. Per-Subject        — Performance per subject
  7. Stress Test        — Edge cases & hard samples
  8. Full Report        — Everything at once
  ─────────────────────────────────────────────
  0. Exit
    """)
    print("=" * 55)


def main():
    print("\n" + "=" * 55)
    print("  V3-TL UNANIMOUS — TESTING INTERFACE")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    # Load once
    X_train, X_test, y_train, y_test, s_train, s_test = load_data()
    svm, cnn = load_models()

    print("\n  ⏳ Computing predictions (this takes ~30 seconds)...")
    preds = get_predictions(svm, cnn, X_test, y_test)
    print("  ✅ Predictions ready!\n")

    while True:
        print_menu()
        choice = input("  Enter choice (0-8): ").strip()

        if   choice == '1': quick_test(preds)
        elif choice == '2': per_gesture(preds)
        elif choice == '3': model_comparison(preds)
        elif choice == '4': confidence_analysis(preds)
        elif choice == '5': disagreement_analysis(preds)
        elif choice == '6': per_subject(svm, cnn, X_test, y_test, s_test)
        elif choice == '7': stress_test(preds)
        elif choice == '8': full_report(preds, svm, cnn,
                                        X_test, y_test, s_test)
        elif choice == '0':
            print("\n  👋 Exiting...\n")
            break
        else:
            print("\n  ⚠️ Invalid choice. Enter 0-8.")


if __name__ == "__main__":
    main()