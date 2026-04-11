"""
Src/cnn_lstm.py

CNN-LSTM Deep Learning Model for EMG Gesture Classification
============================================================
Architecture:
    Input → CNN (spatial features) → LSTM (temporal features) → FC → Output

Features:
    - Automatic feature learning (no manual feature extraction)
    - Handles raw EMG windows directly
    - Batch normalization for stability
    - Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM model for EMG gesture classification.
    
    Architecture:
        Conv1D layers → BatchNorm → ReLU → Dropout →
        LSTM layers → FC layers → Softmax
    """
    
    def __init__(self, 
                 n_channels=12,          # Number of EMG channels
                 window_size=400,        # Samples per window
                 n_classes=6,            # Number of gestures
                 cnn_filters=[64, 128, 256],  # CNN filter sizes
                 lstm_hidden=128,        # LSTM hidden size
                 lstm_layers=2,          # Number of LSTM layers
                 dropout=0.5):           # Dropout rate
        
        super(CNN_LSTM, self).__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_classes = n_classes
        
        # ============================================
        # CNN Layers (Spatial Feature Extraction)
        # ============================================
        self.conv1 = nn.Conv1d(n_channels, cnn_filters[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_filters[2])
        self.pool3 = nn.MaxPool1d(2)
        
        self.dropout_cnn = nn.Dropout(dropout * 0.5)  # Lower dropout for CNN
        
        # Calculate CNN output size
        cnn_output_size = window_size // 8  # After 3 pooling layers
        
        # ============================================
        # LSTM Layers (Temporal Feature Extraction)
        # ============================================
        self.lstm = nn.LSTM(
            input_size=cnn_filters[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # ============================================
        # Fully Connected Layers (Classification)
        # ============================================
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Input shape: (batch, window_size, n_channels)
        Output shape: (batch, n_classes)
        """
        # Reshape for Conv1d: (batch, channels, sequence)
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM: (batch, sequence, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden)
        h_forward = h_n[-2, :, :]  # Last forward
        h_backward = h_n[-1, :, :]  # Last backward
        x = torch.cat([h_forward, h_backward], dim=1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x):
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


class CNN_LSTM_Simple(nn.Module):
    """
    Simpler CNN-LSTM variant (faster training, good baseline).
    """
    
    def __init__(self, 
                 n_channels=12,
                 window_size=400,
                 n_classes=6,
                 dropout=0.5):
        
        super(CNN_LSTM_Simple, self).__init__()
        
        # CNN
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # FC
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # (batch, window, channels) → (batch, channels, window)
        x = x.permute(0, 2, 1)
        
        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # (batch, channels, seq) → (batch, seq, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        
        # FC
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Quick test
if __name__ == "__main__":
    # Test the model
    batch_size = 32
    window_size = 400
    n_channels = 12
    n_classes = 6
    
    # Create random input
    x = torch.randn(batch_size, window_size, n_channels)
    
    # Test full model
    model = CNN_LSTM(n_channels=n_channels, window_size=window_size, n_classes=n_classes)
    output = model(x)
    print(f"CNN_LSTM output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test simple model
    model_simple = CNN_LSTM_Simple(n_channels=n_channels, window_size=window_size, n_classes=n_classes)
    output_simple = model_simple(x)
    print(f"\nCNN_LSTM_Simple output shape: {output_simple.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_simple.parameters()):,}")