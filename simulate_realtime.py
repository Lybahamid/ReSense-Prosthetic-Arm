"""
simulate_realtime.py

Real-Time EMG Prosthetic Simulation
=====================================
Simulates continuous EMG stream with:
  - Per-prediction latency timing
  - Confidence threshold gating
  - Sliding-window consensus
  - Hold-last-state logic
  - Live dashboard
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
import joblib

from Src.data_loader import load_subject
from Src.preprocessing import preprocess_emg, window_data
from Src.feature_extraction import build_feature_matrix
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
KEEP_GESTURES = [0, 1, 2, 3, 4, 5]
GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open",
                 "Pinch", "Point", "Wave"]
GESTURE_ICONS = ["✋", "👊", "🖐️", "🤏", "👆", "👋"]

SVM_MODEL_PATH = "models/v3_final_svm.pkl"
CNN_MODEL_PATH = "models/cnn_lstm_model.pth"

SVM_WEIGHT = 0.4
CNN_WEIGHT = 0.6

CONFIDENCE_THRESHOLD = 0.68
SLIDING_WINDOW_SIZE  = 5
CONSENSUS_THRESHOLD  = 0.60
STREAM_DELAY         = 0.05   # 50ms between predictions (simulates real-time)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CNN Architecture
# ============================================================
class CNN_LSTM_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(16, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.drop_cnn = nn.Dropout(0.2)
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True, dropout=0.25, bidirectional=True)
        self.drop_lstm = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop_cnn(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        x = torch.cat([h[-2], h[-1]], dim=1)
        x = self.drop_lstm(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)


# ============================================================
# LOAD MODELS
# ============================================================
def load_ensemble():
    print("Loading models...")
    svm = joblib.load(SVM_MODEL_PATH)
    print(f"  ✓ SVM: {SVM_MODEL_PATH}")

    ckpt = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cnn = CNN_LSTM_V1()
    cnn.load_state_dict(ckpt['model_state_dict'])
    cnn = cnn.to(DEVICE)
    cnn.eval()
    print(f"  ✓ CNN: {CNN_MODEL_PATH}")

    return svm, cnn


def predict_ensemble(svm, cnn, window):
    """Single window prediction with latency."""
    t0 = time.perf_counter()

    feat = build_feature_matrix(window)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    svm_probs = svm.predict_proba(feat)

    with torch.no_grad():
        batch = torch.FloatTensor(window).to(DEVICE)
        logits = cnn(batch)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        cnn_probs = F.softmax(logits, dim=1).cpu().numpy()

    ens_probs = SVM_WEIGHT * svm_probs + CNN_WEIGHT * cnn_probs

    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000

    return ens_probs[0], latency


# ============================================================
# LOAD EMG STREAM DATA
# ============================================================
def load_stream_data(subject=1, exercise=1):
    path = f"Data/sub{subject:02d}/S{subject}_E{exercise}_A1.mat"
    print(f"\nLoading stream: {path}")
    emg, labels, _ = load_subject(path)
    emg = preprocess_emg(emg)
    windows, wlabs = window_data(emg, labels)
    mask = np.isin(wlabs, KEEP_GESTURES)
    windows, wlabs = windows[mask], wlabs[mask]
    label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
    wlabs = np.array([label_map[l] for l in wlabs])
    print(f"  ✓ {len(wlabs)} windows ready for streaming")
    return windows.astype(np.float32), wlabs


# ============================================================
# SIMULATION
# ============================================================
def run_simulation(svm, cnn, X, y, n_windows=None, delay=STREAM_DELAY):
    if n_windows is None:
        n_windows = len(y)
    n_windows = min(n_windows, len(y))

    print("\n" + "=" * 75)
    print("   REAL-TIME EMG PROSTHETIC SIMULATION")
    print("=" * 75)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Sliding window:      {SLIDING_WINDOW_SIZE} frames")
    print(f"  Consensus:           {CONSENSUS_THRESHOLD*100:.0f}%")
    print(f"  Stream delay:        {delay*1000:.0f}ms")
    print(f"  Total windows:       {n_windows}")
    print()

    # State
    buf = []
    last_stable = None
    last_gesture_name = "---"

    # Stats
    raw_correct = 0
    gated_correct = 0
    gated_total = 0
    holds = 0
    latencies = []
    gesture_stats = {g: {'raw_ok': 0, 'gated_ok': 0, 'total': 0}
                     for g in range(len(GESTURE_NAMES))}

    print(f"{'Time':<8} {'True':<18} {'Raw':<18} {'Conf':<8} {'Action':<18} "
          f"{'Latency':<10} {'State'}")
    print("─" * 95)

    try:
        for i in range(n_windows):
            true_label = y[i]
            window = X[i:i+1]

            # Predict
            probs, lat = predict_ensemble(svm, cnn, window)
            latencies.append(lat)

            raw_pred = np.argmax(probs)
            conf = probs.max()

            # Stage 1: Confidence gate
            if conf >= CONFIDENCE_THRESHOLD:
                gated = raw_pred
            else:
                gated = -1

            # Stage 2: Sliding window
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
                    action = f"→ {GESTURE_NAMES[dom]} {GESTURE_ICONS[dom]}"
                    state = "COMMIT"
                else:
                    output = last_stable
                    holds += 1
                    action = f"HOLD {GESTURE_ICONS[last_stable] if last_stable is not None else '?'}"
                    state = "HOLD"
            else:
                output = last_stable
                holds += 1
                action = "WAITING..."
                state = "BUFFER"

            # Stats
            raw_correct += (raw_pred == true_label)
            gesture_stats[true_label]['total'] += 1
            gesture_stats[true_label]['raw_ok'] += (raw_pred == true_label)

            if output is not None:
                gated_total += 1
                gated_correct += (output == true_label)
                gesture_stats[true_label]['gated_ok'] += (output == true_label)

            # Display
            lat_icon = "✅" if lat < 250 else "⚠️"
            match = "✓" if output == true_label else "✗"
            timestamp = f"{i * delay:.2f}s"

            if i < 40 or i >= n_windows - 5 or i % 50 == 0:
                print(f"{timestamp:<8} {GESTURE_NAMES[true_label]:<18} "
                      f"{GESTURE_NAMES[raw_pred]:<18} {conf*100:>5.1f}%  "
                      f"{action:<18} {lat:>6.1f}ms {lat_icon}  {match}")

            if i == 40 and n_windows > 50:
                print(f"  ... streaming ({n_windows - 45} more windows) ...")

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n\n  ⏹ Simulation stopped by user")
        n_windows = max(i, 1)

    # ── Summary ──
    print("\n" + "═" * 75)
    print("   SIMULATION RESULTS")
    print("═" * 75)

    raw_acc = raw_correct / n_windows * 100
    gated_acc = gated_correct / gated_total * 100 if gated_total > 0 else 0

    print(f"\n  {'Metric':<30} {'Value'}")
    print(f"  {'-'*45}")
    print(f"  {'Raw accuracy':<30} {raw_acc:.1f}%")
    print(f"  {'Gated accuracy':<30} {gated_acc:.1f}%")
    print(f"  {'Improvement':<30} {gated_acc - raw_acc:+.1f}%")
    print(f"  {'Hold events':<30} {holds}/{n_windows} ({holds/n_windows*100:.1f}%)")
    print(f"  {'Committed predictions':<30} {gated_total}/{n_windows}")

    print(f"\n  Latency:")
    print(f"    Mean:   {np.mean(latencies):.2f} ms")
    print(f"    Median: {np.median(latencies):.2f} ms")
    print(f"    P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"    Max:    {np.max(latencies):.2f} ms")

    over = np.sum(np.array(latencies) > 250)
    print(f"    Over budget: {over}/{len(latencies)} "
          f"{'✅' if over == 0 else '❌'}")

    print(f"\n  Per-Gesture:")
    print(f"  {'Gesture':<25} {'Raw Acc':<12} {'Gated Acc':<12} {'Samples'}")
    print(f"  {'-'*55}")
    for g in range(len(GESTURE_NAMES)):
        s = gesture_stats[g]
        if s['total'] > 0:
            ra = s['raw_ok'] / s['total'] * 100
            ga = s['gated_ok'] / s['total'] * 100
            print(f"  {GESTURE_NAMES[g]:<25} {ra:<12.1f} {ga:<12.1f} {s['total']}")

    # Save results
    results = {
        'raw_accuracy': raw_acc,
        'gated_accuracy': gated_acc,
        'holds': holds,
        'total_windows': n_windows,
        'mean_latency_ms': float(np.mean(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'max_latency_ms': float(np.max(latencies)),
    }

    os.makedirs("results/simulation", exist_ok=True)
    with open("results/simulation/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved: results/simulation/results.json")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 75)
    print("   EMG PROSTHETIC REAL-TIME SIMULATION")
    print("   V3-Final Ensemble (SVM + CNN-LSTM)")
    print("=" * 75)

    svm, cnn = load_ensemble()

    while True:
        print(f"\n{'='*50}")
        print("SIMULATION MENU")
        print(f"{'='*50}")
        print("1. Quick simulation (100 windows)")
        print("2. Full simulation (all windows)")
        print("3. Custom simulation")
        print("4. Change subject/exercise")
        print("5. Change parameters")
        print("0. Exit")
        print("-" * 50)

        choice = input("Choice (0-5): ").strip()

        if choice == '0':
            print("\n👋 Goodbye!")
            break

        if choice in ['1', '2', '3']:
            sub = input("Subject (1-10, default 1): ").strip()
            ex = input("Exercise (1-3, default 1): ").strip()
            sub = int(sub) if sub else 1
            ex = int(ex) if ex else 1

            X, y = load_stream_data(sub, ex)

            if choice == '1':
                run_simulation(svm, cnn, X, y, n_windows=100)
            elif choice == '2':
                run_simulation(svm, cnn, X, y)
            elif choice == '3':
                n = input("Windows (default 200): ").strip()
                d = input("Delay ms (default 50): ").strip()
                run_simulation(svm, cnn, X, y,
                              n_windows=int(n) if n else 200,
                              delay=float(d)/1000 if d else STREAM_DELAY)

        elif choice == '5':
            global CONFIDENCE_THRESHOLD, SLIDING_WINDOW_SIZE, CONSENSUS_THRESHOLD
            ct = input(f"Confidence threshold ({CONFIDENCE_THRESHOLD}): ").strip()
            sw = input(f"Sliding window size ({SLIDING_WINDOW_SIZE}): ").strip()
            cs = input(f"Consensus threshold ({CONSENSUS_THRESHOLD}): ").strip()
            if ct:
                CONFIDENCE_THRESHOLD = float(ct)
            if sw:
                SLIDING_WINDOW_SIZE = int(sw)
            if cs:
                CONSENSUS_THRESHOLD = float(cs)
            print(f"  Updated: conf={CONFIDENCE_THRESHOLD}, window={SLIDING_WINDOW_SIZE}, consensus={CONSENSUS_THRESHOLD}")


if __name__ == "__main__":
    main()