import numpy as np
import scipy.io as sio

# -----------------------------
# Load a single subject/session
# -----------------------------
def load_subject(filepath):
    """Load EMG data + labels from a .mat file"""
    mat = sio.loadmat(filepath)
    emg = mat["emg"]      # shape: (samples, channels)
    labels = mat["restimulus"].ravel()  # stimulus labels
    return emg, labels, mat

# -----------------------------
# Balance classes (includes Rest)
# -----------------------------
def balance_gestures(X, y):
    """
    Keep Rest + 5 gestures (Grasp, Open, Pinch, Point, Wave).
    Downsample so all classes are balanced.
    """
    # Adjust if your dataset uses different label numbers
    KEEP = [0, 1, 2, 3, 4, 5]   # 0 = Rest
    mask = np.isin(y, KEEP)
    X, y = X[mask], y[mask]

    # Remap labels to 0..5
    label_map = {old: new for new, old in enumerate(sorted(KEEP))}
    y = np.array([label_map[label] for label in y])

    # Balance
    classes = np.unique(y)
    min_size = min([np.sum(y == c) for c in classes])

    X_bal, y_bal = [], []
    rng = np.random.default_rng(42)

    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, min_size, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    return np.vstack(X_bal), np.hstack(y_bal), classes
