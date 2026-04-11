import numpy as np
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # ← ADD THIS IMPORT
import os      # ← ADD THIS IMPORT

from Src.data_loader import load_subject
from Src.preprocessing import preprocess_emg, window_data
from Src.feature_extraction import build_feature_matrix

# ============================================================
# CONFIG — CHANGE THESE VALUES AS NEEDED
# ============================================================
SUBJECTS        = list(range(1, 11))       # subjects 1 through 10
EXERCISES       = [1, 2, 3]                # exercise sessions per subject
TEST_SIZE       = 0.20                     # 20% held-out test
SAMPLES_PER_CLS = None                    # None = use smallest class size
CV_FOLDS        = 8                        # number of cross-validation folds
KEEP_GESTURES   = [0, 1, 2, 3, 4, 5]      # 0=Rest, 1=Grasp, 2=Open, 3=Pinch, 4=Point, 5=Wave
GESTURE_NAMES   = ["Rest", "Grasp (Hand Close)", "Hand Open",
                   "Pinch", "Point", "Wave"]

# ============================================================
# Helper: balance classes with configurable sample count
# ============================================================
def balance_classes(X, y, samples_per_class=None):
    """
    Downsample classes.
    """
    classes = np.unique(y)
    class_counts = {c: int(np.sum(y == c)) for c in classes}

    print("\n  Original class distribution:")
    for c in classes:
        name = GESTURE_NAMES[c] if c < len(GESTURE_NAMES) else f"Class {c}"
        print(f"    Class {c} ({name}): {class_counts[c]} samples")

    if samples_per_class is None:
        target = min(class_counts.values())
    else:
        target = samples_per_class

    X_bal, y_bal = [], []
    rng = np.random.default_rng(42)

    for c in classes:
        idx = np.where(y == c)[0]
        n = min(target, len(idx))
        chosen = rng.choice(idx, n, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    X_out, y_out = np.vstack(X_bal), np.hstack(y_bal)
    print(f"  Balanced to {target} samples/class (capped if needed)")
    print(f"  Total samples after balancing: {len(y_out)}")
    return X_out, y_out

# ============================================================
# Load & preprocess data from ALL subjects
# ============================================================
emg_all, labels_all = [], []

for subj in SUBJECTS:
    print(f"\nLoading Subject {subj}...")
    for ex in EXERCISES:
        file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
        try:
            emg, labels, _ = load_subject(file)
            emg = preprocess_emg(emg)
            windows, win_labels = window_data(emg, labels)
            emg_all.append(windows)
            labels_all.append(win_labels)
            print(f"  ✓ {file}  →  {len(win_labels)} windows")
        except FileNotFoundError:
            print(f"  ✗ {file} not found, skipping")
        except Exception as e:
            print(f"  ✗ {file} error: {e}, skipping")

X = np.vstack(emg_all)
y = np.hstack(labels_all)
print(f"\nTotal raw windows loaded: {X.shape[0]}")

# ============================================================
# Keep only the gestures we care about + remap labels
# ============================================================
mask = np.isin(y, KEEP_GESTURES)
X, y = X[mask], y[mask]
print(f"Windows after gesture filter: {X.shape[0]}")

label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
y = np.array([label_map[l] for l in y])

# ============================================================
# Balance classes
# ============================================================
X, y = balance_classes(X, y, samples_per_class=SAMPLES_PER_CLS)

# ============================================================
# Feature extraction
# ============================================================
F = build_feature_matrix(X)
F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)  # ← ADD THIS LINE
print(f"\nFeature matrix shape: {F.shape}")
print(f"Features per window:  {F.shape[1]}")

# ============================================================
# Pipeline + GridSearchCV
# ============================================================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        class_weight="balanced", 
        random_state=42,
        probability=True  # ← ADD THIS for confidence analysis
    ))
])

param_grid = {
    "svm__C":      [10, 100, 500, 1000],
    "svm__gamma":  ["scale", 0.01, 0.05, 0.1],
    "svm__kernel": ["rbf"]
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

print("\nRunning GridSearchCV (this may take a while with 10 subjects)...")
grid = GridSearchCV(
    pipe, param_grid, cv=cv,
    scoring="f1_macro", n_jobs=-1, verbose=1
)
grid.fit(F, y)

best_model = grid.best_estimator_
print(f"\n✓ Best parameters found: {grid.best_params_}")
print(f"  Best CV Macro F1:      {grid.best_score_:.4f}")

# ============================================================
# Cross-validation evaluation
# ============================================================
acc_scores = cross_val_score(best_model, F, y, cv=cv, scoring="accuracy")
f1_scores  = cross_val_score(best_model, F, y, cv=cv, scoring="f1_macro")

print(f"\nCross-validation Accuracy: {acc_scores.mean():.2f} ± {acc_scores.std():.2f}")
print(f"Cross-validation Macro F1: {f1_scores.mean():.2f} ± {f1_scores.std():.2f}")

# ============================================================
# Final held-out test evaluation
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    F, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"\n{'='*55}")
print(f" HELD-OUT TEST RESULTS ({int(TEST_SIZE*100)}% test, "
      f"{int((1-TEST_SIZE)*100)}% train)")
print(f"{'='*55}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1:  {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=GESTURE_NAMES))

# ============================================================
# Confusion matrix plot
# ============================================================
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(GESTURE_NAMES)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — {len(SUBJECTS)} Subjects, "
          f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} Split")
plt.tight_layout()

# ← ADD: Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png", dpi=150)
print("\n✓ Saved: results/confusion_matrix.png")

plt.show()

# ============================================================
# ← ADD: SAVE THE TRAINED MODEL
# ============================================================
os.makedirs("models", exist_ok=True)
model_path = "models/svm_model_final.pkl"
joblib.dump(best_model, model_path)
print(f"✓ Saved model: {model_path}")

print("\n" + "=" * 55)
print("✅ TRAINING COMPLETE!")
print("=" * 55)
print(f"Model saved to: {model_path}")
print("You can now run: python test_gesture_prediction.py")