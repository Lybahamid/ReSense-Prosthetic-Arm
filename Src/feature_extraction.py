"""
Src/feature_extraction.py

Extracts features from EMG windows.
"""

import numpy as np
import pywt


def extract_features(window):
    feats = []

    for ch in range(window.shape[1]):
        sig = window[:, ch]

        # === Time-domain features ===
        mav = np.mean(np.abs(sig))  # Mean Absolute Value
        wl = np.sum(np.abs(np.diff(sig)))  # Waveform Length
        var = np.var(sig)  # Variance
        rms = np.sqrt(np.mean(sig ** 2))  # RMS
        zc = np.sum(np.diff(np.sign(sig)) != 0)  # Zero Crossing
        ssc = np.sum(np.diff(np.sign(np.diff(sig))) != 0)  # Slope Sign Change

        # Willison Amplitude (threshold = 0.05 * std)
        thresh = 0.05 * np.std(sig)
        wamp = np.sum(np.abs(np.diff(sig)) > thresh)

        # Hjorth parameters
        first_deriv = np.diff(sig)
        second_deriv = np.diff(first_deriv)
        activity = np.var(sig)
        mobility = np.sqrt(np.var(first_deriv) / (activity + 1e-6))
        complexity = np.sqrt(
            (np.var(second_deriv) / (np.var(first_deriv) + 1e-6)) /
            (mobility + 1e-6)
        )

        feats.extend([mav, wl, var, rms, zc, ssc, wamp,
                      activity, mobility, complexity])

        # === Frequency-domain features ===
        fft_vals = np.abs(np.fft.rfft(sig))
        fft_energy = np.sum(fft_vals ** 2)
        feats.append(fft_energy)

        # === Wavelet packet energy ===
        coeffs = pywt.wavedec(sig, "db4", level=3)
        for c in coeffs:
            feats.append(np.sum(np.square(c)))

    return np.array(feats)


def build_feature_matrix(X):
    return np.vstack([extract_features(win) for win in X])
