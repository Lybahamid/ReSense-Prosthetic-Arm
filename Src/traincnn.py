# """
# train_cnn_lstm.py

# Train CNN-LSTM model for EMG Gesture Classification
# ====================================================
# Features:
#     - Early stopping
#     - Learning rate scheduling
#     - Model checkpointing
#     - Training visualization
#     - Overfitting prevention
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# import os

# from Src.data_loader import load_subject
# from Src.preprocessing import preprocess_emg, window_data
# from Src.cnn_lstm import CNN_LSTM, CNN_LSTM_Simple

# # ============================================================
# # CONFIGURATION
# # ============================================================
# SUBJECTS = list(range(1, 11))
# EXERCISES = [1, 2, 3]
# KEEP_GESTURES = [0, 1, 2, 3, 4, 5]
# GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open",
#                  "Pinch", "Point", "Wave"]

# # Training parameters
# BATCH_SIZE = 64
# EPOCHS = 100
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 1e-4
# EARLY_STOPPING_PATIENCE = 15
# TEST_SIZE = 0.20
# VAL_SIZE = 0.15

# # Model parameters
# USE_SIMPLE_MODEL = False  # Set True for faster training
# DROPOUT = 0.5

# # Device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ============================================================
# # Data Loading
# # ============================================================
# def load_all_data():
#     """Load and preprocess all EMG data."""
#     print("=" * 60)
#     print("Loading EMG data...")
#     print("=" * 60)
    
#     windows_all, labels_all = [], []
    
#     for subj in SUBJECTS:
#         print(f"  Subject {subj}...", end=" ")
#         subj_windows = 0
        
#         for ex in EXERCISES:
#             file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
#             try:
#                 emg, labels, _ = load_subject(file)
#                 emg = preprocess_emg(emg)
#                 windows, win_labels = window_data(emg, labels)
                
#                 windows_all.append(windows)
#                 labels_all.append(win_labels)
#                 subj_windows += len(win_labels)
#             except Exception as e:
#                 pass
        
#         print(f"{subj_windows} windows")
    
#     X = np.vstack(windows_all)
#     y = np.hstack(labels_all)
    
#     print(f"\nTotal windows: {X.shape[0]}")
#     print(f"Window shape: {X.shape[1:]} (samples × channels)")
    
#     # Filter gestures
#     mask = np.isin(y, KEEP_GESTURES)
#     X, y = X[mask], y[mask]
    
#     # Remap labels
#     label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
#     y = np.array([label_map[l] for l in y])
    
#     print(f"After filtering: {X.shape[0]} windows")
    
#     return X, y

# def balance_classes(X, y):
#     """Balance classes by downsampling."""
#     classes = np.unique(y)
#     min_count = min([np.sum(y == c) for c in classes])
    
#     X_bal, y_bal = [], []
#     rng = np.random.default_rng(42)
    
#     for c in classes:
#         idx = np.where(y == c)[0]
#         chosen = rng.choice(idx, min_count, replace=False)
#         X_bal.append(X[chosen])
#         y_bal.append(y[chosen])
    
#     return np.vstack(X_bal), np.hstack(y_bal)

# def create_dataloaders(X, y):
#     """Create train, validation, and test dataloaders."""
#     # First split: train+val vs test
#     X_trainval, X_test, y_trainval, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, stratify=y, random_state=42
#     )
    
#     # Second split: train vs val
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval, random_state=42
#     )
    
#     print(f"\nDataset splits:")
#     print(f"  Train:      {len(y_train)}")
#     print(f"  Validation: {len(y_val)}")
#     print(f"  Test:       {len(y_test)}")
    
#     # Convert to tensors
#     X_train = torch.FloatTensor(X_train)
#     X_val = torch.FloatTensor(X_val)
#     X_test = torch.FloatTensor(X_test)
    
#     y_train = torch.LongTensor(y_train)
#     y_val = torch.LongTensor(y_val)
#     y_test = torch.LongTensor(y_test)
    
#     # Create datasets
#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)
#     test_dataset = TensorDataset(X_test, y_test)
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     return train_loader, val_loader, test_loader, (X_test, y_test)

# # ============================================================
# # Training Functions
# # ============================================================
# def train_epoch(model, train_loader, criterion, optimizer, device):
#     """Train for one epoch."""
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
    
#     avg_loss = running_loss / len(train_loader)
#     accuracy = correct / total
    
#     return avg_loss, accuracy

# def validate(model, val_loader, criterion, device):
#     """Validate the model."""
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
    
#     avg_loss = running_loss / len(val_loader)
#     accuracy = correct / total
    
#     return avg_loss, accuracy

# def evaluate(model, test_loader, device):
#     """Evaluate on test set."""
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
            
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.numpy())
    
#     return np.array(all_preds), np.array(all_labels)

# # ============================================================
# # Main Training Loop
# # ============================================================
# def main():
#     print("=" * 60)
#     print("   CNN-LSTM EMG GESTURE CLASSIFICATION")
#     print("=" * 60)
#     print(f"\nDevice: {DEVICE}")
    
#     # Load data
#     X, y = load_all_data()
    
#     # Balance classes
#     print("\nBalancing classes...")
#     X, y = balance_classes(X, y)
#     print(f"After balancing: {len(y)} samples")
    
#     # Get data dimensions
#     window_size = X.shape[1]
#     n_channels = X.shape[2]
#     n_classes = len(np.unique(y))
    
#     print(f"\nData dimensions:")
#     print(f"  Window size:  {window_size}")
#     print(f"  Channels:     {n_channels}")
#     print(f"  Classes:      {n_classes}")
    
#     # Create dataloaders
#     train_loader, val_loader, test_loader, (X_test, y_test) = create_dataloaders(X, y)
    
#     # Create model
#     print("\n" + "=" * 60)
#     print("Creating model...")
#     print("=" * 60)
    
#     if USE_SIMPLE_MODEL:
#         model = CNN_LSTM_Simple(
#             n_channels=n_channels,
#             window_size=window_size,
#             n_classes=n_classes,
#             dropout=DROPOUT
#         )
#         print("Using: CNN_LSTM_Simple")
#     else:
#         model = CNN_LSTM(
#             n_channels=n_channels,
#             window_size=window_size,
#             n_classes=n_classes,
#             dropout=DROPOUT
#         )
#         print("Using: CNN_LSTM (Full)")
    
#     model = model.to(DEVICE)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters:     {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#     # Training history
#     history = {
#         'train_loss': [], 'train_acc': [],
#         'val_loss': [], 'val_acc': []
#     }
    
#     # Early stopping
#     best_val_loss = float('inf')
#     best_val_acc = 0
#     patience_counter = 0
#     best_model_state = None
    
#     # Training loop
#     print("\n" + "=" * 60)
#     print("Training...")
#     print("=" * 60)
#     print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10}")
#     print("-" * 70)
    
#     start_time = time.time()
    
#     for epoch in range(1, EPOCHS + 1):
#         # Train
#         train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
#         # Validate
#         val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
#         # Update scheduler
#         scheduler.step(val_loss)
        
#         # Save history
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
        
#         # Get current learning rate
#         current_lr = optimizer.param_groups[0]['lr']
        
#         # Print progress
#         print(f"{epoch:<8} {train_loss:<12.4f} {train_acc*100:<12.1f} {val_loss:<12.4f} {val_acc*100:<12.1f} {current_lr:<10.6f}")
        
#         # Early stopping check
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_val_acc = val_acc
#             best_model_state = model.state_dict().copy()
#             patience_counter = 0
#         else:
#             patience_counter += 1
        
#         if patience_counter >= EARLY_STOPPING_PATIENCE:
#             print(f"\nEarly stopping at epoch {epoch}!")
#             break
    
#     training_time = time.time() - start_time
#     print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
#     # Load best model
#     model.load_state_dict(best_model_state)
    
#     # ============================================================
#     # Evaluation
#     # ============================================================
#     print("\n" + "=" * 60)
#     print("Evaluating on test set...")
#     print("=" * 60)
    
#     y_pred, y_true = evaluate(model, test_loader, DEVICE)
    
#     test_acc = accuracy_score(y_true, y_pred)
#     test_f1 = f1_score(y_true, y_pred, average='macro')
    
#     print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
#     print(f"  Test Macro F1: {test_f1*100:.2f}%")
    
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=GESTURE_NAMES))
    
#     # ============================================================
#     # Save Model
#     # ============================================================
#     os.makedirs("models", exist_ok=True)
#     model_path = "models/cnn_lstm_model.pth"
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'model_config': {
#             'n_channels': n_channels,
#             'window_size': window_size,
#             'n_classes': n_classes,
#             'dropout': DROPOUT,
#             'use_simple': USE_SIMPLE_MODEL
#         },
#         'accuracy': test_acc,
#         'f1_score': test_f1
#     }, model_path)
#     print(f"\n✓ Model saved: {model_path}")
    
#     # ============================================================
#     # Visualization
#     # ============================================================
#     os.makedirs("results", exist_ok=True)
    
#     # Training curves
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Loss
#     axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
#     axes[0].plot(history['val_loss'], label='Val Loss', color='red')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Training and Validation Loss')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
    
#     # Accuracy
#     axes[1].plot([a*100 for a in history['train_acc']], label='Train Acc', color='blue')
#     axes[1].plot([a*100 for a in history['val_acc']], label='Val Acc', color='red')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy (%)')
#     axes[1].set_title('Training and Validation Accuracy')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig("results/cnn_lstm_training_curves.png", dpi=150)
#     print("✓ Saved: results/cnn_lstm_training_curves.png")
    
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'CNN-LSTM Confusion Matrix\nAccuracy: {test_acc*100:.1f}% | F1: {test_f1*100:.1f}%')
#     plt.tight_layout()
#     plt.savefig("results/cnn_lstm_confusion_matrix.png", dpi=150)
#     print("✓ Saved: results/cnn_lstm_confusion_matrix.png")
    
#     plt.show()
    
#     # ============================================================
#     # Summary
#     # ============================================================
#     print("\n" + "=" * 60)
#     print("SUMMARY")
#     print("=" * 60)
    
#     # Calculate overfitting gap
#     final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
#     gap = final_train_acc - test_acc
    
#     print(f"""
#   Model:              {'CNN_LSTM_Simple' if USE_SIMPLE_MODEL else 'CNN_LSTM'}
#   Parameters:         {total_params:,}
#   Epochs trained:     {len(history['train_loss'])}
#   Training time:      {training_time/60:.1f} minutes
  
#   Best Validation:
#     Loss:             {best_val_loss:.4f}
#     Accuracy:         {best_val_acc*100:.2f}%
  
#   Test Results:
#     Accuracy:         {test_acc*100:.2f}%
#     Macro F1:         {test_f1*100:.2f}%
#     Overfit Gap:      {gap*100:.2f}%
    
#   Files saved:
#     - models/cnn_lstm_model.pth
#     - results/cnn_lstm_training_curves.png
#     - results/cnn_lstm_confusion_matrix.png
#     """)
    
#     # Compare with SVM
#     print("=" * 60)
#     print("COMPARISON WITH SVM")
#     print("=" * 60)
#     print(f"""
#   SVM Model:          78% cross-subject accuracy
#   CNN-LSTM Model:     {test_acc*100:.1f}% accuracy
  
#   Difference:         {'+' if test_acc > 0.78 else ''}{(test_acc - 0.78)*100:.1f}%
#     """)
    
#     if test_acc > 0.80:
#         print("  🎉 CNN-LSTM achieves 80%+ accuracy!")
    
#     print("=" * 60)

# if __name__ == "__main__":
#     main()"""
"""evaluate_cnn_lstm_balanced.py

Fair evaluation of CNN-LSTM model with balanced test data.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from Src.data_loader import load_subject
from Src.preprocessing import preprocess_emg, window_data

# ============================================================
# CONFIG
# ============================================================
SUBJECTS = list(range(1, 11))
EXERCISES = [1, 2, 3]
KEEP_GESTURES = [0, 1, 2, 3, 4, 5]
GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open",
                 "Pinch", "Point", "Wave"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model (same architecture as training)
# ============================================================
class CNN_LSTM_Full(torch.nn.Module):
    def __init__(self, n_channels=16, window_size=200, n_classes=6, dropout=0.5):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.pool1 = torch.nn.MaxPool1d(2)
        
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.pool2 = torch.nn.MaxPool1d(2)
        
        self.conv3 = torch.nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.pool3 = torch.nn.MaxPool1d(2)
        
        self.drop_cnn = torch.nn.Dropout(dropout * 0.4)
        
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout * 0.5,
            bidirectional=True
        )
        self.drop_lstm = torch.nn.Dropout(dropout)
        
        self.fc1 = torch.nn.Linear(128, 64)
        self.bn_fc = torch.nn.BatchNorm1d(64)
        self.drop_fc = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(64, n_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop_cnn(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.drop_lstm(x)
        
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x

# ============================================================
# Load and Balance Data for Fair Testing
# ============================================================
def load_balanced_test_data():
    """Load ALL data and create BALANCED test set."""
    print("=" * 60)
    print("Loading data for balanced evaluation...")
    print("=" * 60)
    
    windows_all, labels_all = [], []
    
    for subj in SUBJECTS:
        for ex in EXERCISES:
            file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
            try:
                emg, labels, _ = load_subject(file)
                emg = preprocess_emg(emg)
                windows, win_labels = window_data(emg, labels)
                windows_all.append(windows)
                labels_all.append(win_labels)
            except:
                pass
    
    X = np.vstack(windows_all).astype(np.float32)
    y = np.hstack(labels_all)
    
    # Filter gestures
    mask = np.isin(y, KEEP_GESTURES)
    X, y = X[mask], y[mask]
    
    # Remap labels
    label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
    y = np.array([label_map[l] for l in y])
    
    print(f"\nTotal samples: {len(y)}")
    
    # BALANCE the dataset
    classes = np.unique(y)
    min_count = min([np.sum(y == c) for c in classes])
    
    print(f"\nBalancing to {min_count} samples per class:")
    
    X_bal, y_bal = [], []
    rng = np.random.default_rng(42)
    
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])
        print(f"  {GESTURE_NAMES[c]}: {min_count} samples")
    
    X_bal = np.vstack(X_bal)
    y_bal = np.hstack(y_bal)
    
    print(f"\nTotal balanced samples: {len(y_bal)}")
    
    return X_bal, y_bal

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("   CNN-LSTM BALANCED EVALUATION")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load("models/cnn_lstm_model.pth", map_location=DEVICE)
    
    config = checkpoint.get('model_config', {})
    n_channels = config.get('n_channels', 16)
    window_size = config.get('window_size', 200)
    n_classes = config.get('n_classes', 6)
    
    model = CNN_LSTM_Full(
        n_channels=n_channels,
        window_size=window_size,
        n_classes=n_classes,
        dropout=0.5
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded!")
    
    # Load BALANCED data
    X, y = load_balanced_test_data()
    
    # Split (use larger test set for reliable evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    print(f"\nBalanced test set: {len(y_test)} samples")
    print(f"Per class: ~{len(y_test)//6} samples each")
    
    # Predict
    print("\nPredicting...")
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    all_preds = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            outputs = model(batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
    
    y_pred = np.array(all_preds)
    
    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n" + "=" * 60)
    print("BALANCED EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n  Test Accuracy (Balanced): {test_acc*100:.2f}%")
    print(f"  Test Macro F1 (Balanced): {test_f1*100:.2f}%")
    
    print("\nClassification Report (BALANCED):")
    print(classification_report(y_test, y_pred, target_names=GESTURE_NAMES))
    
    # Per-subject evaluation
    print("\n" + "=" * 60)
    print("PER-SUBJECT ACCURACY")
    print("=" * 60)
    
    print(f"\n{'Subject':<10} {'Accuracy':<12} {'F1 Score':<12} {'Status'}")
    print("-" * 50)
    
    subject_accs = []
    subject_f1s = []
    
    for subj in SUBJECTS:
        subj_features, subj_labels = [], []
        
        for ex in EXERCISES:
            try:
                file = f"Data/sub{subj:02d}/S{subj}_E{ex}_A1.mat"
                emg, labels, _ = load_subject(file)
                emg = preprocess_emg(emg)
                windows, win_labels = window_data(emg, labels)
                
                mask = np.isin(win_labels, KEEP_GESTURES)
                windows = windows[mask]
                win_labels = win_labels[mask]
                
                label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
                win_labels = np.array([label_map.get(l, l) for l in win_labels])
                
                subj_features.append(windows.astype(np.float32))
                subj_labels.append(win_labels)
            except:
                pass
        
        if subj_features:
            features = np.vstack(subj_features)
            labels = np.hstack(subj_labels)
            
            # Predict
            features_tensor = torch.FloatTensor(features).to(DEVICE)
            preds = []
            
            with torch.no_grad():
                for i in range(0, len(features_tensor), batch_size):
                    batch = features_tensor[i:i+batch_size]
                    outputs = model(batch)
                    _, predicted = outputs.max(1)
                    preds.extend(predicted.cpu().numpy())
            
            preds = np.array(preds)
            
            acc = accuracy_score(labels, preds) * 100
            f1 = f1_score(labels, preds, average='macro') * 100
            
            subject_accs.append(acc)
            subject_f1s.append(f1)
            
            status = "🌟" if acc >= 95 else "✅" if acc >= 90 else "⚠️" if acc >= 85 else "—"
            print(f"Sub {subj:<5} {acc:<12.1f} {f1:<12.1f} {status}")
    
    print("-" * 50)
    print(f"{'Average':<10} {np.mean(subject_accs):<12.1f} {np.mean(subject_f1s):<12.1f}")
    print(f"{'Min':<10} {np.min(subject_accs):<12.1f} {np.min(subject_f1s):<12.1f}")
    print(f"{'Max':<10} {np.max(subject_accs):<12.1f} {np.max(subject_f1s):<12.1f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'CNN-LSTM Balanced Evaluation\n'
              f'Accuracy: {test_acc*100:.1f}% | Macro F1: {test_f1*100:.1f}%')
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/cnn_lstm_balanced_confusion.png", dpi=150)
    print("\n✓ Saved: results/cnn_lstm_balanced_confusion.png")
    plt.show()
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: ALL MODELS")
    print("=" * 60)
    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  Model               Accuracy    Macro F1    Gap          ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  SVM (cross-subj)    78.0%       78.0%       8.1%         ║
  ║  CNN-LSTM Option A   62.6%       61.7%       13.8%        ║
  ║  CNN-LSTM Option B                                        ║
  ║    Unbalanced test   94.4%       76.2%       5.6%         ║
  ║    Balanced test     {test_acc*100:.1f}%       {test_f1*100:.1f}%       —            ║
  ║  Per-Subject Avg     {np.mean(subject_accs):.1f}%       {np.mean(subject_f1s):.1f}%       —            ║
  ╚═══════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()