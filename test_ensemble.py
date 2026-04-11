"""
test_ensemble_final.py

V3-FINAL Ensemble Test Suite
==============================
Menu-driven testing matching SVM test script style.
Works with: v3_final_svm.pkl + cnn_lstm_model.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import time
import os
import json
from collections import Counter

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

SVM_MODEL_PATH = "models/v3_final_svm.pkl"
CNN_MODEL_PATH = "models/cnn_lstm_model.pth"
CONFIG_PATH    = "models/v3_final_config.json"

CONFIDENCE_THRESHOLD = 0.68
SLIDING_WINDOW_SIZE  = 5
CONSENSUS_THRESHOLD  = 0.60

SVM_WEIGHT = 0.4
CNN_WEIGHT = 0.6

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
SAVE_DIR   = "results/test_ensemble_final"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# CNN-LSTM (matching V1 checkpoint)
# ============================================================
class CNN_LSTM_V1(nn.Module):
    def __init__(self, n_channels=16, window_size=200, n_classes=6, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.drop_cnn = nn.Dropout(dropout * 0.4)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2,
                           batch_first=True, dropout=dropout * 0.5, bidirectional=True)
        self.drop_lstm = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 6)

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
# ENSEMBLE CLASS
# ============================================================
class EnsembleHybrid:
    def __init__(self, svm_model, cnn_model):
        self.svm = svm_model
        self.cnn = cnn_model
        self.cnn.eval()

    def predict_proba(self, X_windows):
        F_svm = build_feature_matrix(X_windows)
        F_svm = np.nan_to_num(F_svm, nan=0.0, posinf=0.0, neginf=0.0)
        svm_probs = self.svm.predict_proba(F_svm)

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_windows), BATCH_SIZE):
                batch = torch.FloatTensor(X_windows[i:i+BATCH_SIZE]).to(DEVICE)
                logits = self.cnn(batch)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        cnn_probs = np.vstack(all_probs)

        ensemble_probs = (SVM_WEIGHT * svm_probs + CNN_WEIGHT * cnn_probs)
        return ensemble_probs, svm_probs, cnn_probs

    def predict(self, X_windows):
        ens_probs, _, _ = self.predict_proba(X_windows)
        return np.argmax(ens_probs, axis=1)

    def predict_single(self, X_single):
        """Single window prediction with timing."""
        t0 = time.perf_counter()
        F_svm = build_feature_matrix(X_single)
        F_svm = np.nan_to_num(F_svm, nan=0.0, posinf=0.0, neginf=0.0)
        svm_probs = self.svm.predict_proba(F_svm)

        with torch.no_grad():
            batch = torch.FloatTensor(X_single).to(DEVICE)
            logits = self.cnn(batch)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            cnn_probs = F.softmax(logits, dim=1).cpu().numpy()

        ens_probs = (SVM_WEIGHT * svm_probs + CNN_WEIGHT * cnn_probs)
        t1 = time.perf_counter()

        return ens_probs[0], (t1 - t0) * 1000


# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    print("=" * 65)
    print("   LOADING V3-FINAL ENSEMBLE")
    print("=" * 65)

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"✓ Config: {CONFIG_PATH}")

    if not os.path.exists(SVM_MODEL_PATH):
        print(f"❌ SVM not found: {SVM_MODEL_PATH}")
        return None, None
    svm = joblib.load(SVM_MODEL_PATH)
    print(f"✓ SVM: {SVM_MODEL_PATH}")

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"❌ CNN not found: {CNN_MODEL_PATH}")
        return None, None
    ckpt = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cnn = CNN_LSTM_V1(16, 200, 6, 0.5)
    cnn.load_state_dict(ckpt['model_state_dict'])
    cnn = cnn.to(DEVICE)
    cnn.eval()
    print(f"✓ CNN: {CNN_MODEL_PATH} (acc={ckpt.get('accuracy', 'N/A'):.4f})")

    ensemble = EnsembleHybrid(svm, cnn)
    print(f"✓ Ensemble ready (SVM={SVM_WEIGHT}, CNN={CNN_WEIGHT})")

    return ensemble, config


# ============================================================
# DATA LOADING
# ============================================================
def load_test_data(subject_id, exercise_id=1):
    path = f"Data/sub{subject_id:02d}/S{subject_id}_E{exercise_id}_A1.mat"
    print(f"\nLoading: {path}")
    try:
        emg, labels, _ = load_subject(path)
        emg = preprocess_emg(emg)
        windows, wlabs = window_data(emg, labels)
        mask = np.isin(wlabs, KEEP_GESTURES)
        windows, wlabs = windows[mask], wlabs[mask]
        label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
        wlabs = np.array([label_map[l] for l in wlabs])
        print(f"✓ {len(wlabs)} windows")
        for g in range(len(GESTURE_NAMES)):
            c = np.sum(wlabs == g)
            if c > 0:
                print(f"  {GESTURE_NAMES[g]}: {c} ({c/len(wlabs)*100:.1f}%)")
        return windows.astype(np.float32), wlabs
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def load_all_exercises(subject_id):
    all_w, all_l = [], []
    for ex in EXERCISES:
        w, l = load_test_data(subject_id, ex)
        if w is not None:
            all_w.append(w)
            all_l.append(l)
    if not all_w:
        return None, None
    return np.vstack(all_w), np.hstack(all_l)


# ============================================================
# TEST 1: Random Sample Predictions
# ============================================================
def test_random_samples(ens, X, y, n=20):
    print("\n" + "=" * 65)
    print("TEST 1: Random Sample Predictions")
    print("=" * 65)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), min(n, len(y)), replace=False)
    ep, sp, cp = ens.predict_proba(X[idx])
    ep_pred = np.argmax(ep, axis=1)
    sp_pred = np.argmax(sp, axis=1)
    cp_pred = np.argmax(cp, axis=1)
    confs = ep.max(axis=1) * 100

    print(f"\n{'#':<4} {'True':<18} {'Ensemble':<18} {'SVM':<12} {'CNN':<12} {'Conf':<8} {'Result'}")
    print("-" * 85)

    correct = 0
    for i, (t, e, s, c, conf) in enumerate(zip(y[idx], ep_pred, sp_pred, cp_pred, confs)):
        m = "✓" if t == e else "✗"
        correct += (t == e)
        print(f"{i+1:<4} {GESTURE_NAMES[t]:<18} {GESTURE_NAMES[e]:<18} "
              f"{GESTURE_NAMES[s]:<12} {GESTURE_NAMES[c]:<12} {conf:>5.1f}%  {m}")

    print("-" * 85)
    print(f"Accuracy: {correct}/{len(idx)} = {correct/len(idx)*100:.1f}%")


# ============================================================
# TEST 2: Component Comparison
# ============================================================
def test_component_comparison(ens, X, y):
    print("\n" + "=" * 65)
    print("TEST 2: Component Comparison (SVM vs CNN vs Ensemble)")
    print("=" * 65)

    ep, sp, cp = ens.predict_proba(X)
    e_pred = np.argmax(ep, axis=1)
    s_pred = np.argmax(sp, axis=1)
    c_pred = np.argmax(cp, axis=1)

    print(f"\n{'Model':<20} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 45)
    for name, pred in [('SVM', s_pred), ('CNN', c_pred), ('ENSEMBLE', e_pred)]:
        acc = accuracy_score(y, pred) * 100
        f1 = f1_score(y, pred, average='macro') * 100
        icon = "🏆" if name == 'ENSEMBLE' else ""
        print(f"{name:<20} {acc:<12.1f} {f1:<12.1f} {icon}")

    print(f"\n{'Gesture':<25} {'Ensemble':<12} {'SVM':<12} {'CNN':<12} {'Best'}")
    print("-" * 65)
    for g in range(len(GESTURE_NAMES)):
        mask = y == g
        if np.sum(mask) == 0:
            continue
        ge = accuracy_score(y[mask], e_pred[mask]) * 100
        gs = accuracy_score(y[mask], s_pred[mask]) * 100
        gc = accuracy_score(y[mask], c_pred[mask]) * 100
        best = "ENS" if ge >= max(gs, gc) else ("SVM" if gs > gc else "CNN")
        print(f"{GESTURE_NAMES[g]:<25} {ge:<12.1f} {gs:<12.1f} {gc:<12.1f} {best}")


# ============================================================
# TEST 3: Per-Gesture (80% Check)
# ============================================================
def test_per_gesture(ens, X, y):
    print("\n" + "=" * 65)
    print("TEST 3: Per-Gesture Accuracy (80% Threshold)")
    print("=" * 65)

    preds = ens.predict(X)

    print(f"\n{'Gesture':<25} {'Samples':<10} {'Correct':<10} {'Acc':<10} {'F1':<10} {'Status'}")
    print("-" * 70)

    all_pass = True
    for g in range(len(GESTURE_NAMES)):
        mask = y == g
        if np.sum(mask) == 0:
            continue
        gc = np.sum(preds[mask] == y[mask])
        gt = np.sum(mask)
        ga = gc / gt * 100
        gf = f1_score(y == g, preds == g, average='binary', zero_division=0) * 100
        ok = ga >= 80 and gf >= 80
        if not ok:
            all_pass = False
        st = "✅" if ok else "⚠️" if ga >= 75 else "❌"
        print(f"{GESTURE_NAMES[g]:<25} {gt:<10} {gc:<10} {ga:<10.1f} {gf:<10.1f} {st}")

    print("-" * 70)
    oa = accuracy_score(y, preds) * 100
    of1 = f1_score(y, preds, average='macro') * 100
    print(f"{'OVERALL':<25} {len(y):<10} {'':<10} {oa:<10.1f} {of1:<10.1f}")
    print(f"\n{'✅ ALL gestures ≥80%!' if all_pass else '⚠️ Some below 80%'}")


# ============================================================
# TEST 4: Real-Time + Latency
# ============================================================
def test_realtime_latency(ens, X, y, n=50):
    print("\n" + "=" * 65)
    print("TEST 4: Real-Time Simulation + Latency")
    print("=" * 65)

    latencies = []
    correct = 0
    start = np.random.randint(0, max(1, len(y) - n))

    print(f"\n{'#':<5} {'True':<18} {'Pred':<18} {'Conf':<8} {'Latency':<12} {'Result'}")
    print("-" * 70)

    for i in range(n):
        idx = start + i
        if idx >= len(y):
            break

        probs, lat = ens.predict_single(X[idx:idx+1])
        latencies.append(lat)

        pred = np.argmax(probs)
        conf = probs.max() * 100
        m = "✓" if y[idx] == pred else "✗"
        correct += (y[idx] == pred)
        li = "✅" if lat < 250 else "⚠️"

        print(f"{i+1:<5} {GESTURE_NAMES[y[idx]]:<18} {GESTURE_NAMES[pred]:<18} "
              f"{conf:>5.1f}%  {lat:>7.2f}ms {li} {m}")

    print("-" * 70)
    print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.1f}%")
    print(f"\nLatency:")
    print(f"  Mean:   {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")

    over = np.sum(np.array(latencies) > 250)
    print(f"\n  {'✅ All within 250ms!' if over == 0 else f'❌ {over} exceeded!'}")


# ============================================================
# TEST 5: Sliding-Window Confidence
# ============================================================
def test_sliding_window(ens, X, y, n=80):
    print("\n" + "=" * 65)
    print("TEST 5: Sliding-Window Confidence")
    print(f"  Window={SLIDING_WINDOW_SIZE} | Conf≥{CONFIDENCE_THRESHOLD*100:.0f}% | Consensus≥{CONSENSUS_THRESHOLD*100:.0f}%")
    print("=" * 65)

    buf = []
    last_stable = None
    raw_ok = gated_ok = gated_n = holds = 0
    start = np.random.randint(0, max(1, len(y) - n))

    print(f"\n{'#':<5} {'True':<16} {'Raw':<16} {'Conf':<8} {'Output':<16} {'Status'}")
    print("-" * 75)

    for i in range(n):
        idx = start + i
        if idx >= len(y):
            break

        probs, _ = ens.predict_single(X[idx:idx+1])
        raw = np.argmax(probs)
        conf = probs.max()

        gated = raw if conf >= CONFIDENCE_THRESHOLD else -1
        buf.append(gated)
        if len(buf) > SLIDING_WINDOW_SIZE:
            buf.pop(0)

        valid = [p for p in buf if p >= 0]

        if len(valid) >= SLIDING_WINDOW_SIZE * CONSENSUS_THRESHOLD:
            counts = Counter(valid)
            dom, dom_c = counts.most_common(1)[0]
            if dom_c / SLIDING_WINDOW_SIZE >= CONSENSUS_THRESHOLD:
                output = dom
                last_stable = dom
            else:
                output = last_stable
                holds += 1
        else:
            output = last_stable
            holds += 1

        raw_ok += (raw == y[idx])
        if output is not None:
            gated_n += 1
            gated_ok += (output == y[idx])

        out_name = GESTURE_NAMES[output] if output is not None else "---"
        st = "✓" if output == y[idx] else "✗"

        if i < 25 or i >= n - 5:
            print(f"{i+1:<5} {GESTURE_NAMES[y[idx]]:<16} {GESTURE_NAMES[raw]:<16} "
                  f"{conf*100:>5.1f}%  {out_name:<16} {st}")
        if i == 25 and n > 30:
            print(f"  ... ({n - 30} more) ...")

    print(f"\n{'─'*65}")
    ra = raw_ok / n * 100
    ga = gated_ok / gated_n * 100 if gated_n > 0 else 0
    imp = ga - ra
    print(f"  Raw accuracy:      {ra:.1f}%")
    print(f"  Gated accuracy:    {ga:.1f}%")
    print(f"  Holds:             {holds}/{n} ({holds/n*100:.1f}%)")
    print(f"  Committed:         {gated_n}/{n}")
    print(f"\n  {'✅' if imp >= 0 else '⚠️'} Gating {'improved' if imp >= 0 else 'reduced'} by {abs(imp):.1f}%")


# ============================================================
# TEST 6: Confusion Matrix
# ============================================================
def test_confusion_matrix(ens, X, y):
    print("\n" + "=" * 65)
    print("TEST 6: Confusion Matrix")
    print("=" * 65)

    preds = ens.predict(X)
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=GESTURE_NAMES))

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('V3-Final Ensemble — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", dpi=150)
    print(f"✓ Saved: {SAVE_DIR}/confusion_matrix.png")
    plt.show()


# ============================================================
# TEST 7: Overfitting Analysis (with plots)
# ============================================================
def test_overfitting(config):
    print("\n" + "=" * 65)
    print("TEST 7: Overfitting Analysis")
    print("=" * 65)

    if not config or 'results' not in config:
        print("  ⚠ No config found. Run train_hybrid_ensemble_final.py first.")
        return

    r = config['results']
    svm = r.get('svm', {})
    cnn = r.get('cnn', {})

    print(f"\n  {'Model':<15} {'Train':<10} {'Test':<10} {'CV':<12} {'CV-Test':<10} {'Status'}")
    print(f"  {'-'*60}")

    if svm:
        cv_test = abs(svm.get('cv_acc', svm['test_acc']) - svm['test_acc'])
        print(f"  {'SVM':<15} {svm['train_acc']:<10.1f} {svm['test_acc']:<10.1f} "
              f"{svm.get('cv_acc', 'N/A'):<12} {cv_test:<10.1f} "
              f"{'✅' if cv_test < 2 else '⚠️'}")
    if cnn:
        gap = abs(cnn['gap'])
        print(f"  {'CNN':<15} {cnn['train_acc']:<10.1f} {cnn['test_acc']:<10.1f} "
              f"{'—':<12} {gap:<10.1f} {'✅' if gap < 5 else '⚠️'}")

    # Plot
    if svm and cnn:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Train vs Test vs CV
        models = ['SVM', 'CNN']
        trains = [svm['train_acc'], cnn['train_acc']]
        tests = [svm['test_acc'], cnn['test_acc']]

        x = np.arange(2)
        w = 0.25
        axes[0].bar(x - w, trains, w, label='Train', color='skyblue', edgecolor='black')
        axes[0].bar(x, tests, w, label='Test', color='lightcoral', edgecolor='black')
        if 'cv_acc' in svm:
            axes[0].bar(0 + w, svm['cv_acc'], w, label='CV', color='lightgreen', edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Train vs Test vs CV\n(CV = honest metric)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(50, 105)

        # SVM explanation
        if 'cv_acc' in svm:
            cats = ['Train\n(Misleading)', 'CV\n(Honest)', 'Test\n(Final)']
            vals = [svm['train_acc'], svm['cv_acc'], svm['test_acc']]
            cols = ['red', 'green', 'green']
            axes[1].bar(cats, vals, color=cols, alpha=0.7, edgecolor='black')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('SVM: Why 100% Train ≠ Overfitting\n(CV ≈ Test → No overfitting)')
            axes[1].set_ylim(50, 105)
            axes[1].grid(axis='y', alpha=0.3)
            for i, v in enumerate(vals):
                axes[1].annotate(f'{v:.1f}%', (i, v + 1), ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/overfitting_analysis.png", dpi=150)
        print(f"\n  ✓ Saved: {SAVE_DIR}/overfitting_analysis.png")
        plt.show()


# ============================================================
# TEST 8: Balanced Samples
# ============================================================
def test_balanced(ens, X, y, n_per=15):
    print("\n" + "=" * 65)
    print(f"TEST 8: Balanced Sample Testing ({n_per}/gesture)")
    print("=" * 65)

    rng = np.random.default_rng(42)
    idx = []
    for g in range(len(GESTURE_NAMES)):
        gi = np.where(y == g)[0]
        if len(gi) > 0:
            idx.extend(rng.choice(gi, min(n_per, len(gi)), replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)

    ep, _, _ = ens.predict_proba(X[idx])
    preds = np.argmax(ep, axis=1)
    confs = ep.max(axis=1) * 100

    print(f"\n{'#':<4} {'True':<20} {'Predicted':<20} {'Conf':<8} {'Result'}")
    print("-" * 60)

    stats = {g: {'c': 0, 't': 0} for g in range(len(GESTURE_NAMES))}
    for i, (t, p, c) in enumerate(zip(y[idx], preds, confs)):
        m = "✓" if t == p else "✗"
        stats[t]['t'] += 1
        if t == p:
            stats[t]['c'] += 1
        print(f"{i+1:<4} {GESTURE_NAMES[t]:<20} {GESTURE_NAMES[p]:<20} {c:>5.1f}%  {m}")

    print(f"\n{'Gesture':<25} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 55)
    tc = tt = 0
    for g in range(len(GESTURE_NAMES)):
        s = stats[g]
        if s['t'] > 0:
            print(f"{GESTURE_NAMES[g]:<25} {s['c']:<10} {s['t']:<10} {s['c']/s['t']*100:.1f}%")
            tc += s['c']
            tt += s['t']
    print("-" * 55)
    print(f"{'OVERALL (BALANCED)':<25} {tc:<10} {tt:<10} {tc/tt*100:.1f}%")


# ============================================================
# TEST 9: Gesture Deep Dive
# ============================================================
def test_deep_dive(ens, X, y):
    print("\n" + "=" * 65)
    print("TEST 9: Gesture Deep Dive")
    print("=" * 65)

    print("\nSelect gesture:")
    for i, n in enumerate(GESTURE_NAMES):
        print(f"  {i}: {n} ({np.sum(y == i)} samples)")

    try:
        g = int(input("\nGesture (0-5): ").strip())
    except ValueError:
        print("Invalid.")
        return

    mask = y == g
    ep, _, _ = ens.predict_proba(X[mask])
    preds = np.argmax(ep, axis=1)
    confs = ep.max(axis=1)

    correct = np.sum(preds == y[mask])
    total = np.sum(mask)

    print(f"\n{'═'*55}")
    print(f"  {GESTURE_NAMES[g]}")
    print(f"{'═'*55}")
    print(f"  Total: {total} | Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"  Avg confidence: {confs.mean()*100:.1f}%")

    if total - correct > 0:
        print(f"\n  Misclassified as:")
        wrong = preds[preds != y[mask]]
        for o in range(len(GESTURE_NAMES)):
            if o == g:
                continue
            cnt = np.sum(wrong == o)
            if cnt > 0:
                print(f"    → {GESTURE_NAMES[o]}: {cnt} ({cnt/(total-correct)*100:.1f}%)")


# ============================================================
# TEST 10: All Subjects
# ============================================================
def test_all_subjects(ens):
    print("\n" + "=" * 65)
    print("TEST 10: Compare All Subjects")
    print("=" * 65)

    results = []
    for subj in SUBJECTS:
        X, y = load_all_exercises(subj)
        if X is None:
            continue
        preds = ens.predict(X)
        acc = accuracy_score(y, preds) * 100
        f1 = f1_score(y, preds, average='macro') * 100
        results.append((subj, acc, f1, len(y)))
        icon = "🌟" if acc >= 95 else "✅" if acc >= 90 else "⚠️" if acc >= 80 else "❌"
        print(f"  Subject {subj:2d}: Acc={acc:.1f}%, F1={f1:.1f}% ({len(y)}) {icon}")

    if results:
        print("-" * 55)
        accs = [r[1] for r in results]
        f1s = [r[2] for r in results]
        print(f"  Average: Acc={np.mean(accs):.1f}%, F1={np.mean(f1s):.1f}%")
        print(f"  Best:    Sub {max(results, key=lambda x:x[1])[0]} ({max(accs):.1f}%)")
        print(f"  Worst:   Sub {min(results, key=lambda x:x[1])[0]} ({min(accs):.1f}%)")


# ============================================================
# TEST 11: Confidence Sweep
# ============================================================
def test_confidence_sweep(ens, X, y, n=200):
    print("\n" + "=" * 65)
    print("TEST 11: Confidence Threshold Sweep")
    print("=" * 65)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), min(n, len(y)), replace=False)
    ep, _, _ = ens.predict_proba(X[idx])
    preds = np.argmax(ep, axis=1)
    confs = ep.max(axis=1)
    correct = preds == y[idx]

    thresholds = [0.50, 0.60, 0.68, 0.75, 0.80, 0.85, 0.90, 0.95]

    print(f"\n{'Threshold':<12} {'Kept':<10} {'Accuracy':<12} {'Coverage'}")
    print("-" * 50)

    for t in thresholds:
        mask = confs >= t
        kept = np.sum(mask)
        acc = np.mean(correct[mask]) * 100 if kept > 0 else 0
        cov = kept / len(confs) * 100
        marker = " ◄── current" if t == CONFIDENCE_THRESHOLD else ""
        print(f"  ≥{t*100:.0f}%     {kept:<10} {acc:<12.1f} {cov:.1f}%{marker}")

    print(f"\n  💡 Higher = more accurate but fewer predictions")


# ============================================================
# TEST 12: Quick Suite
# ============================================================
def test_quick(ens, X, y, config):
    print("\n" + "=" * 65)
    print("TEST 12: Quick Test Suite")
    print("=" * 65)

    test_component_comparison(ens, X, y)
    test_per_gesture(ens, X, y)
    test_overfitting(config)

    print("\n✅ Quick suite complete!")


# ============================================================
# MAIN MENU
# ============================================================
def main():
    print("=" * 65)
    print("   V3-FINAL ENSEMBLE TEST SUITE")
    print("   SVM + CNN-LSTM Hybrid")
    print("=" * 65)

    ens, config = load_models()
    if ens is None:
        return

    print("\nLoading default test data (Subject 1, Exercise 1)...")
    X, y = load_test_data(1, 1)
    if X is None:
        return

    while True:
        print(f"\n{'='*65}")
        print("MENU")
        print(f"{'='*65}")
        print("1.  Random Sample Predictions")
        print("2.  Component Comparison (SVM vs CNN vs Ensemble) ★")
        print("3.  Per-Gesture Accuracy (80% Check)")
        print("4.  Real-Time Simulation + Latency ★")
        print("5.  Sliding-Window Confidence ★")
        print("6.  Confusion Matrix")
        print("7.  Overfitting Analysis (with plots) ★★")
        print("8.  Balanced Sample Testing")
        print("9.  Gesture Deep Dive")
        print("10. Compare All Subjects")
        print("11. Confidence Threshold Sweep")
        print("12. Quick Test Suite (recommended)")
        print("13. Load Different Data")
        print("0.  Exit")
        print("-" * 65)

        c = input("Choice (0-13): ").strip()

        if   c == '1':
            n = input("Samples? (20): ").strip()
            test_random_samples(ens, X, y, int(n) if n else 20)
        elif c == '2':  test_component_comparison(ens, X, y)
        elif c == '3':  test_per_gesture(ens, X, y)
        elif c == '4':
            n = input("Windows? (50): ").strip()
            test_realtime_latency(ens, X, y, int(n) if n else 50)
        elif c == '5':
            n = input("Windows? (80): ").strip()
            test_sliding_window(ens, X, y, int(n) if n else 80)
        elif c == '6':  test_confusion_matrix(ens, X, y)
        elif c == '7':  test_overfitting(config)
        elif c == '8':
            n = input("Per gesture? (15): ").strip()
            test_balanced(ens, X, y, int(n) if n else 15)
        elif c == '9':  test_deep_dive(ens, X, y)
        elif c == '10': test_all_subjects(ens)
        elif c == '11': test_confidence_sweep(ens, X, y)
        elif c == '12': test_quick(ens, X, y, config)
        elif c == '13':
            s = input("Subject (1-10): ").strip()
            e = input("Exercise (1-3): ").strip()
            try:
                Xn, yn = load_test_data(int(s), int(e) if e else 1)
                if Xn is not None:
                    X, y = Xn, yn
            except ValueError:
                print("Invalid.")
        elif c == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid (0-13).")


if __name__ == "__main__":
    main()