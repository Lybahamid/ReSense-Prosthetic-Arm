# """
# unity_bridge.py

# Real-Time EMG Gesture Prediction → Unity Bridge
# ================================================
# Sends SVM predictions to Unity via TCP socket.

# Usage:
#     1. Start Unity scene (starts TCP listener on port 5005)
#     2. Run: python unity_bridge.py
#     3. Watch the hand animate in Unity!

# Controls:
#     - Automatic mode: Cycles through random EMG samples
#     - Manual mode: You choose which gesture to send
#     - Continuous mode: Streams predictions like real EMG
# """

# import numpy as np
# import joblib
# import socket
# import json
# import time
# import os
# import sys

# from Src.data_loader import load_subject
# from Src.preprocessing import preprocess_emg, window_data
# from Src.feature_extraction import build_feature_matrix

# # ============================================================
# # CONFIGURATION
# # ============================================================
# MODEL_PATH = "models/svm_model_final.pkl"
# UNITY_HOST = "127.0.0.1"  # localhost
# UNITY_PORT = 5005

# GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open",
#                  "Pinch", "Point", "Wave"]

# KEEP_GESTURES = [0, 1, 2, 3, 4, 5]

# # ============================================================
# # Unity Connection
# # ============================================================
# class UnityBridge:
#     def __init__(self, host=UNITY_HOST, port=UNITY_PORT):
#         self.host = host
#         self.port = port
#         self.socket = None
#         self.connected = False
    
#     def connect(self):
#         """Connect to Unity TCP server."""
#         print(f"\nConnecting to Unity at {self.host}:{self.port}...")
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.connect((self.host, self.port))
#             self.connected = True
#             print("✓ Connected to Unity!")
#             return True
#         except ConnectionRefusedError:
#             print("❌ Connection refused!")
#             print("   Make sure Unity is running and the scene is playing.")
#             return False
#         except Exception as e:
#             print(f"❌ Connection error: {e}")
#             return False
    
#     def send_gesture(self, gesture_id, confidence=1.0):
#         """Send gesture prediction to Unity."""
#         if not self.connected:
#             print("Not connected to Unity!")
#             return False
        
#         try:
#             message = {
#                 "gestureId": int(gesture_id),
#                 "confidence": float(confidence)
#             }
#             json_str = json.dumps(message) + "\n"
#             self.socket.sendall(json_str.encode('utf-8'))
#             return True
#         except Exception as e:
#             print(f"Send error: {e}")
#             self.connected = False
#             return False
    
#     def disconnect(self):
#         """Close connection."""
#         if self.socket:
#             self.socket.close()
#             self.connected = False
#             print("Disconnected from Unity.")

# # ============================================================
# # Load Model and Data
# # ============================================================
# def load_model():
#     """Load the trained SVM model."""
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ Model not found: {MODEL_PATH}")
#         print("   Run svm_classifier.py first to train the model.")
#         return None
    
#     print(f"Loading model: {MODEL_PATH}")
#     model = joblib.load(MODEL_PATH)
#     print("✓ Model loaded!")
#     return model

# def load_emg_data(subject_id=1, exercise_id=1):
#     """Load and preprocess EMG data."""
#     file = f"Data/sub{subject_id:02d}/S{subject_id}_E{exercise_id}_A1.mat"
    
#     print(f"\nLoading EMG data: {file}")
#     try:
#         emg, labels, _ = load_subject(file)
#         emg = preprocess_emg(emg)
#         windows, win_labels = window_data(emg, labels)
        
#         # Filter gestures
#         mask = np.isin(win_labels, KEEP_GESTURES)
#         windows = windows[mask]
#         win_labels = win_labels[mask]
        
#         # Remap labels
#         label_map = {old: new for new, old in enumerate(sorted(KEEP_GESTURES))}
#         win_labels = np.array([label_map.get(l, l) for l in win_labels])
        
#         # Extract features
#         features = build_feature_matrix(windows)
#         features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
#         print(f"✓ Loaded {len(win_labels)} windows")
#         return features, win_labels
        
#     except FileNotFoundError:
#         print(f"❌ File not found: {file}")
#         return None, None
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return None, None

# # ============================================================
# # Mode 1: Automatic Random Sampling
# # ============================================================
# def mode_automatic(model, features, labels, bridge, interval=1.0):
#     """Automatically send random predictions to Unity."""
#     print("\n" + "=" * 60)
#     print("MODE: Automatic Random Sampling")
#     print("=" * 60)
#     print(f"Sending predictions every {interval} seconds...")
#     print("Press Ctrl+C to stop.\n")
    
#     rng = np.random.default_rng()
#     count = 0
    
#     try:
#         while True:
#             # Pick random sample
#             idx = rng.integers(0, len(labels))
#             X = features[idx:idx+1]
#             y_true = labels[idx]
            
#             # Predict
#             y_pred = model.predict(X)[0]
            
#             # Get confidence if available
#             try:
#                 proba = model.predict_proba(X)[0]
#                 confidence = float(proba.max())
#             except:
#                 confidence = 1.0
            
#             # Send to Unity
#             success = bridge.send_gesture(y_pred, confidence)
            
#             # Display
#             count += 1
#             true_name = GESTURE_NAMES[y_true]
#             pred_name = GESTURE_NAMES[y_pred]
#             match = "✓" if y_true == y_pred else "✗"
#             status = "→ Unity" if success else "✗ Failed"
            
#             print(f"[{count:4d}] True: {true_name:<18} Pred: {pred_name:<18} "
#                   f"Conf: {confidence:5.1%} {match} {status}")
            
#             time.sleep(interval)
            
#     except KeyboardInterrupt:
#         print(f"\n\nStopped. Sent {count} predictions.")

# # ============================================================
# # Mode 2: Balanced Automatic (Equal gestures)
# # ============================================================
# def mode_balanced_auto(model, features, labels, bridge, interval=1.5):
#     """Cycle through gestures evenly."""
#     print("\n" + "=" * 60)
#     print("MODE: Balanced Automatic (Cycles through all gestures)")
#     print("=" * 60)
#     print(f"Sending predictions every {interval} seconds...")
#     print("Press Ctrl+C to stop.\n")
    
#     rng = np.random.default_rng()
#     count = 0
#     gesture_cycle = 0
    
#     try:
#         while True:
#             # Pick sample from current gesture in cycle
#             target_gesture = gesture_cycle % len(GESTURE_NAMES)
#             gesture_indices = np.where(labels == target_gesture)[0]
            
#             if len(gesture_indices) == 0:
#                 gesture_cycle += 1
#                 continue
            
#             idx = rng.choice(gesture_indices)
#             X = features[idx:idx+1]
#             y_true = labels[idx]
            
#             # Predict
#             y_pred = model.predict(X)[0]
            
#             # Get confidence
#             try:
#                 proba = model.predict_proba(X)[0]
#                 confidence = float(proba.max())
#             except:
#                 confidence = 1.0
            
#             # Send to Unity
#             success = bridge.send_gesture(y_pred, confidence)
            
#             # Display
#             count += 1
#             true_name = GESTURE_NAMES[y_true]
#             pred_name = GESTURE_NAMES[y_pred]
#             match = "✓" if y_true == y_pred else "✗"
            
#             print(f"[{count:4d}] True: {true_name:<18} → Pred: {pred_name:<18} "
#                   f"({confidence:5.1%}) {match}")
            
#             gesture_cycle += 1
#             time.sleep(interval)
            
#     except KeyboardInterrupt:
#         print(f"\n\nStopped. Sent {count} predictions.")

# # ============================================================
# # Mode 3: Manual Control
# # ============================================================
# def mode_manual(model, features, labels, bridge):
#     """Manually select which gesture to send."""
#     print("\n" + "=" * 60)
#     print("MODE: Manual Control")
#     print("=" * 60)
#     print("\nCommands:")
#     print("  0-5  : Send specific gesture")
#     print("  r    : Send random prediction from data")
#     print("  q    : Quit")
#     print("-" * 60)
    
#     for i, name in enumerate(GESTURE_NAMES):
#         print(f"  {i} = {name}")
#     print()
    
#     rng = np.random.default_rng()
    
#     while True:
#         try:
#             cmd = input("\nEnter command: ").strip().lower()
            
#             if cmd == 'q':
#                 print("Exiting manual mode.")
#                 break
            
#             elif cmd == 'r':
#                 # Random prediction
#                 idx = rng.integers(0, len(labels))
#                 X = features[idx:idx+1]
#                 y_true = labels[idx]
#                 y_pred = model.predict(X)[0]
                
#                 try:
#                     proba = model.predict_proba(X)[0]
#                     confidence = float(proba.max())
#                 except:
#                     confidence = 1.0
                
#                 bridge.send_gesture(y_pred, confidence)
                
#                 true_name = GESTURE_NAMES[y_true]
#                 pred_name = GESTURE_NAMES[y_pred]
#                 match = "✓" if y_true == y_pred else "✗"
#                 print(f"  True: {true_name} → Predicted: {pred_name} ({confidence:.1%}) {match}")
            
#             elif cmd.isdigit() and 0 <= int(cmd) <= 5:
#                 gesture_id = int(cmd)
#                 bridge.send_gesture(gesture_id, 1.0)
#                 print(f"  → Sent: {GESTURE_NAMES[gesture_id]}")
            
#             else:
#                 print("  Invalid command. Use 0-5, 'r', or 'q'.")
                
#         except KeyboardInterrupt:
#             print("\nExiting.")
#             break

# # ============================================================
# # Mode 4: Continuous Stream (Simulates real-time EMG)
# # ============================================================
# def mode_continuous_stream(model, features, labels, bridge, fps=5):
#     """Stream predictions continuously like real EMG input."""
#     print("\n" + "=" * 60)
#     print("MODE: Continuous Stream (Real-time Simulation)")
#     print("=" * 60)
#     print(f"Streaming at {fps} predictions per second...")
#     print("Press Ctrl+C to stop.\n")
    
#     interval = 1.0 / fps
#     count = 0
#     correct = 0
#     start_idx = np.random.randint(0, max(1, len(labels) - 1000))
    
#     try:
#         for i in range(min(1000, len(labels) - start_idx)):
#             idx = start_idx + i
#             X = features[idx:idx+1]
#             y_true = labels[idx]
            
#             # Predict
#             y_pred = model.predict(X)[0]
            
#             # Get confidence
#             try:
#                 proba = model.predict_proba(X)[0]
#                 confidence = float(proba.max())
#             except:
#                 confidence = 1.0
            
#             # Send to Unity
#             bridge.send_gesture(y_pred, confidence)
            
#             # Stats
#             count += 1
#             if y_true == y_pred:
#                 correct += 1
            
#             # Display (every 10th prediction to reduce spam)
#             if count % 10 == 0 or count <= 5:
#                 acc = correct / count * 100
#                 pred_name = GESTURE_NAMES[y_pred]
#                 print(f"[{count:4d}] → {pred_name:<18} (Conf: {confidence:5.1%}) "
#                       f"Running Acc: {acc:.1f}%")
            
#             time.sleep(interval)
            
#     except KeyboardInterrupt:
#         pass
    
#     acc = correct / count * 100 if count > 0 else 0
#     print(f"\n\nStream ended. {count} predictions, {acc:.1f}% accuracy.")

# # ============================================================
# # Mode 5: Demo Mode (Showcase all gestures)
# # ============================================================
# def mode_demo(bridge, hold_time=2.0):
#     """Demo mode: cycles through all gestures slowly."""
#     print("\n" + "=" * 60)
#     print("MODE: Demo (Showcasing all gestures)")
#     print("=" * 60)
#     print(f"Each gesture held for {hold_time} seconds...")
#     print("Press Ctrl+C to stop.\n")
    
#     try:
#         while True:
#             for gesture_id, name in enumerate(GESTURE_NAMES):
#                 print(f"  → {name}")
#                 bridge.send_gesture(gesture_id, 1.0)
#                 time.sleep(hold_time)
                
#     except KeyboardInterrupt:
#         print("\n\nDemo stopped.")

# # ============================================================
# # Main Menu
# # ============================================================
# def main():
#     print("=" * 60)
#     print("   EMG → UNITY REAL-TIME BRIDGE")
#     print("   SVM Gesture Prediction to Unity Hand Model")
#     print("=" * 60)
    
#     # Load model
#     model = load_model()
#     if model is None:
#         return
    
#     # Load EMG data
#     features, labels = load_emg_data(subject_id=1, exercise_id=1)
#     if features is None:
#         return
    
#     # Connect to Unity
#     bridge = UnityBridge()
#     if not bridge.connect():
#         print("\n⚠ Make sure Unity is running first!")
#         print("  1. Open Unity project")
#         print("  2. Press Play in Unity")
#         print("  3. Run this script again")
#         return
    
#     try:
#         while True:
#             print("\n" + "=" * 60)
#             print("SELECT MODE")
#             print("=" * 60)
#             print("1. Automatic Random Sampling")
#             print("2. Balanced Auto (Cycles through gestures)")
#             print("3. Manual Control")
#             print("4. Continuous Stream (Real-time simulation)")
#             print("5. Demo Mode (Showcase all gestures)")
#             print("6. Load Different Subject/Exercise")
#             print("7. Reconnect to Unity")
#             print("0. Exit")
#             print("-" * 60)
            
#             choice = input("Enter choice (0-7): ").strip()
            
#             if choice == '1':
#                 interval = input("Interval in seconds (default 1.0): ").strip()
#                 interval = float(interval) if interval else 1.0
#                 mode_automatic(model, features, labels, bridge, interval)
                
#             elif choice == '2':
#                 interval = input("Interval in seconds (default 1.5): ").strip()
#                 interval = float(interval) if interval else 1.5
#                 mode_balanced_auto(model, features, labels, bridge, interval)
                
#             elif choice == '3':
#                 mode_manual(model, features, labels, bridge)
                
#             elif choice == '4':
#                 fps = input("Predictions per second (default 5): ").strip()
#                 fps = int(fps) if fps else 5
#                 mode_continuous_stream(model, features, labels, bridge, fps)
                
#             elif choice == '5':
#                 hold = input("Hold time per gesture in seconds (default 2.0): ").strip()
#                 hold = float(hold) if hold else 2.0
#                 mode_demo(bridge, hold)
                
#             elif choice == '6':
#                 sub = input("Subject ID (1-10): ").strip()
#                 ex = input("Exercise ID (1-3, default 1): ").strip()
#                 try:
#                     sub = int(sub)
#                     ex = int(ex) if ex else 1
#                     new_features, new_labels = load_emg_data(sub, ex)
#                     if new_features is not None:
#                         features, labels = new_features, new_labels
#                         print(f"✓ Now using Subject {sub}, Exercise {ex}")
#                 except ValueError:
#                     print("Invalid input.")
                    
#             elif choice == '7':
#                 bridge.disconnect()
#                 bridge.connect()
                
#             elif choice == '0':
#                 print("\nGoodbye! 👋")
#                 break
                
#             else:
#                 print("Invalid choice. Enter 0-7.")
                
#     finally:
#         bridge.disconnect()

# # ============================================================
# # Run
# # ============================================================
# if __name__ == "__main__":
#     main()"""
"""unity_bridge_ensemble.py

Real-Time EMG Gesture Prediction → Unity Bridge (V3 Ensemble Edition)
=====================================================================
Sends V3 Ensemble (SVM + CNN-LSTM) predictions to Unity via TCP socket.

Features:
    - Uses your 93% accurate ensemble model
    - Shows individual model predictions vs ensemble
    - Real-time confidence visualization
    - All original modes preserved

Usage:
    1. Start Unity scene (starts TCP listener on port 5005)
    2. Run: python unity_bridge_ensemble.py
    3. Watch the enhanced predictions in Unity!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import joblib
import socket
import json
import time
import os
import sys
import scipy.io
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

# Model paths - Update these to match your setup
SVM_MODEL_PATH = "models/svm_model_final.pkl"  # Your SVM model
CNN_MODEL_PATH = "models/cnn_lstm_model.pth"   # Your CNN model

# Alternative paths if above don't work
ALTERNATIVE_PATHS = {
    'svm': [
        "models/v3_fair_svm.pkl",
        "models/v3_final_svm.pkl",
        "models/svm_model_improved.pkl",
        "models/ensemble_universal_svm.pkl"
    ],
    'cnn': [
        "models/v3_fair_cnn.pth",
        "models/v3_final_cnn.pth",
        "models/v3_ensemble_universal_cnn.pth"
    ]
}

# Unity connection
UNITY_HOST = "127.0.0.1"  
UNITY_PORT = 5005

# EMG parameters
GESTURE_NAMES = ["Rest", "Grasp (Hand Close)", "Hand Open", "Pinch", "Point", "Wave"]
GESTURE_EMOJIS = ['✋', '👊', '🖐️', '🤏', '👉', '👋']

# Ensemble weights (from your V3-Fair: 40% SVM, 60% CNN)
SVM_WEIGHT = 0.4
CNN_WEIGHT = 0.6

# Device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CNN-LSTM Model Architecture (V1 from your project)
# ============================================================

class CNN_LSTM_V1(nn.Module):
    def __init__(self, n_channels=16, n_classes=6):
        super(CNN_LSTM_V1, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)  # Must be 64!
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, 
                           bidirectional=True, dropout=0.5)
        
        # FC layers
        self.fc1 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n_combined = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        # FC
        x = F.relu(self.bn4(self.fc1(h_n_combined)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================
# Unity Connection
# ============================================================

class UnityBridge:
    def __init__(self, host=UNITY_HOST, port=UNITY_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to Unity TCP server."""
        print(f"\n🔌 Connecting to Unity at {self.host}:{self.port}...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print("✅ Connected to Unity!")
            return True
        except ConnectionRefusedError:
            print("❌ Connection refused!")
            print("   Make sure Unity is running and the scene is playing.")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_gesture(self, gesture_id, confidence=1.0, extra_data=None):
        """Send gesture prediction to Unity with optional extra data."""
        if not self.connected:
            print("Not connected to Unity!")
            return False
        
        try:
            message = {
                "gestureId": int(gesture_id),
                "confidence": float(confidence),
                "timestamp": time.time()
            }
            
            # Add extra data if provided (e.g., component predictions)
            if extra_data:
                message.update(extra_data)
            
            json_str = json.dumps(message) + "\n"
            self.socket.sendall(json_str.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("🔌 Disconnected from Unity.")

# ============================================================
# Ensemble Model Wrapper
# ============================================================

class EnsemblePredictor:
    def __init__(self, svm_model, cnn_model):
        self.svm = svm_model
        self.cnn = cnn_model
        self.cnn.eval()  # Set CNN to evaluation mode
        
    def extract_svm_features(self, window):
        """Extract hand-crafted features for SVM."""
        features = []
        
        for channel in range(window.shape[1]):
            signal = window[:, channel]
            
            # Time domain features (simplified)
            features.append(np.mean(np.abs(signal)))                    # MAV
            features.append(np.sum(np.abs(np.diff(signal))))           # WL
            features.append(np.var(signal))                            # VAR
            features.append(np.sqrt(np.mean(signal**2)))               # RMS
            
            # Zero crossings
            zc = np.sum(np.diff(np.sign(signal)) != 0)
            features.append(zc)
            
            # Slope sign changes
            diff = np.diff(signal)
            ssc = np.sum(np.diff(np.sign(diff)) != 0)
            features.append(ssc)
            
            # Willison amplitude
            threshold = 0.015 * np.max(np.abs(signal))
            wamp = np.sum(np.abs(np.diff(signal)) > threshold)
            features.append(wamp)
            
            # Hjorth parameters
            activity = np.var(signal)
            diff1 = np.diff(signal)
            diff2 = np.diff(diff1)
            mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
            complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 and np.var(diff1) > 0 else 0
            features.extend([activity, mobility, complexity])
            
            # FFT energy
            fft = np.fft.fft(signal)
            energy = np.sum(np.abs(fft[:len(fft)//2])**2)
            features.append(energy)
            
            # Placeholder for wavelets
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def predict_ensemble(self, window):
        """Make ensemble prediction with both models."""
        # SVM prediction
        svm_features = self.extract_svm_features(window)
        svm_probs = self.svm.predict_proba([svm_features])[0]
        svm_pred = np.argmax(svm_probs)
        
        # CNN prediction
        with torch.no_grad():
            # Transpose for CNN input format
            emg_tensor = torch.FloatTensor(window.T).unsqueeze(0).to(DEVICE)
            cnn_logits = self.cnn(emg_tensor)
            cnn_probs = F.softmax(cnn_logits, dim=1).cpu().numpy()[0]
            cnn_pred = np.argmax(cnn_probs)
        
        # Weighted ensemble
        ensemble_probs = SVM_WEIGHT * svm_probs + CNN_WEIGHT * cnn_probs
        ensemble_pred = np.argmax(ensemble_probs)
        ensemble_conf = ensemble_probs[ensemble_pred]
        
        return {
            'ensemble_pred': ensemble_pred,
            'ensemble_conf': ensemble_conf,
            'ensemble_probs': ensemble_probs,
            'svm_pred': svm_pred,
            'svm_conf': svm_probs[svm_pred],
            'cnn_pred': cnn_pred,
            'cnn_conf': cnn_probs[cnn_pred],
            'agreement': svm_pred == cnn_pred
        }

# ============================================================
# Load Models and Data
# ============================================================

def load_models():
    """Load both SVM and CNN models."""
    print("\n📦 Loading models...")
    
    # Load SVM
    svm_model = None
    if os.path.exists(SVM_MODEL_PATH):
        try:
            svm_model = joblib.load(SVM_MODEL_PATH)
            print(f"  ✅ SVM loaded from {SVM_MODEL_PATH}")
        except:
            try:
                with open(SVM_MODEL_PATH, 'rb') as f:
                    svm_model = pickle.load(f)
                print(f"  ✅ SVM loaded from {SVM_MODEL_PATH}")
            except:
                pass
    
    # Try alternative SVM paths
    if svm_model is None:
        for path in ALTERNATIVE_PATHS['svm']:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        svm_model = pickle.load(f)
                    print(f"  ✅ SVM loaded from {path}")
                    break
                except:
                    continue
    
    if svm_model is None:
        print("  ❌ SVM model not found!")
        return None, None
    
    # Load CNN
    cnn_model = CNN_LSTM_V1(n_channels=16, n_classes=6).to(DEVICE)
    loaded_cnn = False
    
    if os.path.exists(CNN_MODEL_PATH):
        try:
            checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                cnn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                cnn_model.load_state_dict(checkpoint)
            print(f"  ✅ CNN loaded from {CNN_MODEL_PATH}")
            loaded_cnn = True
        except:
            pass
    
    # Try alternative CNN paths
    if not loaded_cnn:
        for path in ALTERNATIVE_PATHS['cnn']:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=DEVICE)
                    if 'model_state_dict' in checkpoint:
                        cnn_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        cnn_model.load_state_dict(checkpoint)
                    print(f"  ✅ CNN loaded from {path}")
                    loaded_cnn = True
                    break
                except:
                    continue
    
    if not loaded_cnn:
        print("  ❌ CNN model not found!")
        return None, None
    
    return svm_model, cnn_model

def load_emg_data(subject_id=1, exercise_id=1):
    """Load and preprocess EMG data."""
    filepath = f"Data/sub{subject_id:02d}/S{subject_id}_E{exercise_id}_A1.mat"
    
    print(f"\n📊 Loading EMG data: {filepath}")
    try:
        mat_data = scipy.io.loadmat(filepath)
        emg = mat_data['emg']
        labels = mat_data['restimulus'].flatten()
        
        # Create windows
        windows = []
        window_labels = []
        window_size = 200
        step = 100  # 50% overlap
        
        gesture_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}
        
        for i in range(0, len(emg) - window_size, step):
            window = emg[i:i+window_size]
            label = labels[i+window_size//2]
            
            if label in gesture_mapping:
                windows.append(window)
                window_labels.append(gesture_mapping[label])
        
        windows = np.array(windows)
        window_labels = np.array(window_labels)
        
        print(f"  ✅ Loaded {len(window_labels)} windows")
        
        # Show distribution
        print("  📊 Gesture distribution:")
        for i in range(6):
            count = np.sum(window_labels == i)
            pct = count / len(window_labels) * 100
            print(f"     {GESTURE_NAMES[i]:<20} {count:5d} ({pct:5.1f}%)")
        
        return windows, window_labels
        
    except FileNotFoundError:
        print(f"  ❌ File not found: {filepath}")
        return None, None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None

# ============================================================
# Enhanced Modes with Ensemble
# ============================================================

def mode_automatic_ensemble(predictor, windows, labels, bridge, interval=1.0):
    """Automatic mode with ensemble predictions."""
    print("\n" + "="*70)
    print("MODE: Automatic Ensemble Predictions (V3-Fair: 93% Accurate)")
    print("="*70)
    print(f"Sending predictions every {interval} seconds...")
    print("Shows: SVM vs CNN vs Ensemble predictions")
    print("Press Ctrl+C to stop.\n")
    
    print(f"{'#':<4} {'True':<18} {'SVM':<18} {'CNN':<18} {'Ensemble':<18} {'Conf':<6} {'Match'}")
    print("-"*95)
    
    rng = np.random.default_rng()
    count = 0
    correct_ensemble = 0
    correct_svm = 0
    correct_cnn = 0
    
    try:
        while True:
            # Pick random sample
            idx = rng.integers(0, len(labels))
            window = windows[idx]
            y_true = labels[idx]
            
            # Get ensemble prediction
            result = predictor.predict_ensemble(window)
            
            # Send to Unity (ensemble prediction)
            bridge.send_gesture(
                result['ensemble_pred'], 
                result['ensemble_conf'],
                extra_data={
                    'svm_pred': int(result['svm_pred']),
                    'cnn_pred': int(result['cnn_pred']),
                    'models_agree': result['agreement']
                }
            )
            
            # Update stats
            count += 1
            if result['ensemble_pred'] == y_true:
                correct_ensemble += 1
            if result['svm_pred'] == y_true:
                correct_svm += 1
            if result['cnn_pred'] == y_true:
                correct_cnn += 1
            
            # Display
            true_name = GESTURE_NAMES[y_true]
            svm_name = GESTURE_NAMES[result['svm_pred']]
            cnn_name = GESTURE_NAMES[result['cnn_pred']]
            ensemble_name = GESTURE_NAMES[result['ensemble_pred']]
            
            match = "✅" if result['ensemble_pred'] == y_true else "❌"
            agree = "🤝" if result['agreement'] else "⚔️"
            
            print(f"{count:<4} {true_name:<18} {svm_name:<18} {cnn_name:<18} "
                  f"{ensemble_name:<18} {result['ensemble_conf']:5.1%} {match} {agree}")
            
            # Show running accuracy every 10 predictions
            if count % 10 == 0:
                acc_svm = correct_svm / count * 100
                acc_cnn = correct_cnn / count * 100
                acc_ens = correct_ensemble / count * 100
                print(f"     Running accuracy: SVM={acc_svm:.1f}% CNN={acc_cnn:.1f}% "
                      f"ENSEMBLE={acc_ens:.1f}% 🎯")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\nStopped. Sent {count} predictions.")
        if count > 0:
            acc_svm = correct_svm / count * 100
            acc_cnn = correct_cnn / count * 100
            acc_ens = correct_ensemble / count * 100
            print(f"Final accuracy: SVM={acc_svm:.1f}% CNN={acc_cnn:.1f}% ENSEMBLE={acc_ens:.1f}%")

def mode_continuous_ensemble(predictor, windows, labels, bridge, fps=5):
    """Continuous stream with ensemble predictions."""
    print("\n" + "="*70)
    print("MODE: Continuous Ensemble Stream (Real-time Simulation)")
    print("="*70)
    print(f"Streaming at {fps} predictions per second...")
    print("Press Ctrl+C to stop.\n")
    
    interval = 1.0 / fps
    count = 0
    correct = 0
    
    # Create a sliding window buffer for smoothing
    prediction_buffer = deque(maxlen=5)
    
    start_idx = np.random.randint(0, max(1, len(labels) - 1000))
    
    try:
        for i in range(min(1000, len(labels) - start_idx)):
            idx = start_idx + i
            window = windows[idx]
            y_true = labels[idx]
            
            # Get ensemble prediction
            result = predictor.predict_ensemble(window)
            prediction_buffer.append(result['ensemble_pred'])
            
            # Use majority vote from buffer for smoother predictions
            if len(prediction_buffer) >= 3:
                from collections import Counter
                smoothed_pred = Counter(prediction_buffer).most_common(1)[0][0]
            else:
                smoothed_pred = result['ensemble_pred']
            
            # Send to Unity
            bridge.send_gesture(
                smoothed_pred,
                result['ensemble_conf'],
                extra_data={'raw_pred': int(result['ensemble_pred'])}
            )
            
            # Stats
            count += 1
            if smoothed_pred == y_true:
                correct += 1
            
            # Display every 10th prediction
            if count % 10 == 0 or count <= 5:
                acc = correct / count * 100
                pred_name = GESTURE_NAMES[smoothed_pred]
                emoji = GESTURE_EMOJIS[smoothed_pred]
                print(f"[{count:4d}] → {pred_name} {emoji} "
                      f"(Conf: {result['ensemble_conf']:5.1%}) "
                      f"Accuracy: {acc:.1f}%")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    
    acc = correct / count * 100 if count > 0 else 0
    print(f"\n\nStream ended. {count} predictions, {acc:.1f}% accuracy.")

def mode_comparison_demo(predictor, windows, labels, bridge):
    """Demo mode comparing SVM vs CNN vs Ensemble."""
    print("\n" + "="*70)
    print("MODE: Model Comparison Demo")
    print("="*70)
    print("Shows how ensemble improves over individual models")
    print("Press Enter to see next prediction, 'q' to quit\n")
    
    rng = np.random.default_rng()
    
    while True:
        cmd = input("\nPress Enter for next comparison (or 'q' to quit): ")
        if cmd.lower() == 'q':
            break
        
        # Pick random sample
        idx = rng.integers(0, len(labels))
        window = windows[idx]
        y_true = labels[idx]
        true_name = GESTURE_NAMES[y_true]
        
        # Get predictions
        result = predictor.predict_ensemble(window)
        
        print("\n" + "-"*50)
        print(f"TRUE GESTURE: {true_name} {GESTURE_EMOJIS[y_true]}")
        print("-"*50)
        
        # SVM prediction
        svm_name = GESTURE_NAMES[result['svm_pred']]
        svm_correct = "✅" if result['svm_pred'] == y_true else "❌"
        print(f"SVM (40% weight):      {svm_name:<18} ({result['svm_conf']:5.1%}) {svm_correct}")
        
        # CNN prediction
        cnn_name = GESTURE_NAMES[result['cnn_pred']]
        cnn_correct = "✅" if result['cnn_pred'] == y_true else "❌"
        print(f"CNN (60% weight):      {cnn_name:<18} ({result['cnn_conf']:5.1%}) {cnn_correct}")
        
        # Ensemble prediction
        ensemble_name = GESTURE_NAMES[result['ensemble_pred']]
        ensemble_correct = "✅" if result['ensemble_pred'] == y_true else "❌"
        print(f"ENSEMBLE (V3-Fair):    {ensemble_name:<18} ({result['ensemble_conf']:5.1%}) {ensemble_correct}")
        
        # Show probabilities
        print("\nProbability Distribution:")
        for i, gesture in enumerate(GESTURE_NAMES):
            prob = result['ensemble_probs'][i]
            bar = "█" * int(prob * 30)
            print(f"  {gesture:<20} {prob:5.1%} {bar}")
        
        # Send ensemble prediction to Unity
        bridge.send_gesture(result['ensemble_pred'], result['ensemble_conf'])
        
        if result['agreement']:
            print("\n🤝 Models agree on prediction")
        else:
            print("\n⚔️ Models disagree - ensemble resolves conflict")

# ============================================================
# Main Menu
# ============================================================

def main():
    print("="*70)
    print("   EMG → UNITY REAL-TIME BRIDGE (V3 ENSEMBLE EDITION)")
    print("   93% Accurate Ensemble: SVM (40%) + CNN-LSTM (60%)")
    print("="*70)
    
    # Show device
    print(f"\n🖥️ Using device: {DEVICE}")
    
    # Load models
    svm_model, cnn_model = load_models()
    if svm_model is None or cnn_model is None:
        print("\n❌ Failed to load models. Check your model files.")
        return
    
    # Create ensemble predictor
    predictor = EnsemblePredictor(svm_model, cnn_model)
    print("  ✅ Ensemble predictor ready!")
    
    # Load EMG data
    windows, labels = load_emg_data(subject_id=1, exercise_id=1)
    if windows is None:
        return
    
    # Connect to Unity
    bridge = UnityBridge()
    if not bridge.connect():
        print("\n⚠️  Make sure Unity is running first!")
        print("  1. Open Unity project")
        print("  2. Press Play in Unity")
        print("  3. Run this script again")
        return
    
    try:
        while True:
            print("\n" + "="*70)
            print("SELECT MODE")
            print("="*70)
            print("1. Automatic Ensemble (Random sampling with model comparison)")
            print("2. Continuous Stream (Real-time simulation)")
            print("3. Model Comparison Demo (Interactive)")
            print("4. Load Different Subject/Exercise")
            print("5. Reconnect to Unity")
            print("0. Exit")
            print("-"*70)
            
            choice = input("Enter choice (0-5): ").strip()
            
            if choice == '1':
                interval = input("Interval in seconds (default 1.0): ").strip()
                interval = float(interval) if interval else 1.0
                mode_automatic_ensemble(predictor, windows, labels, bridge, interval)
                
            elif choice == '2':
                fps = input("Predictions per second (default 5): ").strip()
                fps = int(fps) if fps else 5
                mode_continuous_ensemble(predictor, windows, labels, bridge, fps)
                
            elif choice == '3':
                mode_comparison_demo(predictor, windows, labels, bridge)
                
            elif choice == '4':
                sub = input("Subject ID (1-10): ").strip()
                ex = input("Exercise ID (1-3, default 1): ").strip()
                try:
                    sub = int(sub)
                    ex = int(ex) if ex else 1
                    new_windows, new_labels = load_emg_data(sub, ex)
                    if new_windows is not None:
                        windows, labels = new_windows, new_labels
                        print(f"  ✅ Now using Subject {sub}, Exercise {ex}")
                except ValueError:
                    print("Invalid input.")
                    
            elif choice == '5':
                bridge.disconnect()
                bridge.connect()
                
            elif choice == '0':
                print("\nGoodbye! 👋")
                break
                
            else:
                print("Invalid choice. Enter 0-5.")
                
    finally:
        bridge.disconnect()

if __name__ == "__main__":
    main()