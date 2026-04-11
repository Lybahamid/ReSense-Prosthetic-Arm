"""
Src/preprocessing.py

Preprocessing utilities for EMG signals:
- Bandpass filter (20–450 Hz)
- Notch filter at 50 Hz (to remove powerline noise)
- Normalization
- Windowing
"""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch


def butter_bandpass(lowcut=20, highcut=450, fs=2000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal, lowcut=20, highcut=450, fs=2000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, signal, axis=0)


def notch_filter(signal, notch_freq=50.0, fs=2000.0, quality=30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return lfilter(b, a, signal, axis=0)


def preprocess_emg(emg, fs=2000):
    """
    Full preprocessing pipeline for EMG data.
    """
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = notch_filter(emg, 50, fs)
    # Normalize per channel
    emg = (emg - np.mean(emg, axis=0)) / (np.std(emg, axis=0) + 1e-8)
    return emg


def window_data(emg, labels, window_size=200, step_size=100):
    """
    Segment EMG into overlapping windows.
    Returns:
        X_windows: list of windows (window_size, n_channels)
        y_labels: list of window labels (majority vote)
    """
    n_samples = emg.shape[0]
    n_channels = emg.shape[1]
    X_windows = []
    y_labels = []

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        window = emg[start:end, :]
        label = np.bincount(labels[start:end]).argmax()  # majority vote
        X_windows.append(window)
        y_labels.append(label)

    return np.array(X_windows), np.array(y_labels)
