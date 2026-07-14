"""Acoustic distress classifier (the second Sentinel head).

Separating hunger vs. pain vs. respiratory (cough) vocalisations from a raw
call is a learned audio task no threshold can express. This module implements
the log-mel front end in pure numpy (no librosa dependency, so it runs in CI
and ports cleanly to firmware) and a compact classifier over pooled features.

The deployable edge version replaces the classifier with a small int8 audio
CNN on the same log-mel features; the front end is identical, which is the
point of keeping it here.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

CLASSES = ("normal", "hunger", "pain", "respiratory")


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Triangular mel filterbank, shape [n_mels, n_fft//2 + 1]."""
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sr / 2)),
                          n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            if centre > left:
                fb[m - 1, k] = (k - left) / (centre - left)
        for k in range(centre, right):
            if right > centre:
                fb[m - 1, k] = (right - k) / (right - centre)
    return fb


def logmel(signal: np.ndarray, sr: int = 8000, n_fft: int = 256,
           hop: int = 128, n_mels: int = 16) -> np.ndarray:
    """Compute a log-mel spectrogram, shape [n_frames, n_mels]."""
    signal = np.asarray(signal, dtype=float)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(signal) - n_fft + 1, hop):
        seg = signal[start:start + n_fft] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        frames.append(spec)
    if not frames:
        frames = [np.abs(np.fft.rfft(signal[:n_fft] * window)) ** 2]
    power = np.array(frames)                       # [frames, bins]
    fb = mel_filterbank(sr, n_fft, n_mels)         # [n_mels, bins]
    mel = power @ fb.T                             # [frames, n_mels]
    return np.log(mel + 1e-8)


def pooled_features(signal: np.ndarray, **kw) -> np.ndarray:
    """Mean+std pooling of the log-mel frames -> a fixed-length vector."""
    lm = logmel(signal, **kw)
    return np.concatenate([lm.mean(axis=0), lm.std(axis=0)])


class AcousticClassifier:
    """Classifies a vocalisation into one of CLASSES from pooled log-mel feats."""

    def __init__(self, random_state: int = 0):
        self.clf = RandomForestClassifier(n_estimators=120, random_state=random_state)
        self.classes_ = CLASSES

    def fit(self, signals: list[np.ndarray], labels: list[str]) -> "AcousticClassifier":
        X = np.array([pooled_features(s) for s in signals])
        self.clf.fit(X, labels)
        return self

    def predict(self, signals: list[np.ndarray]) -> list[str]:
        X = np.array([pooled_features(s) for s in signals])
        return list(self.clf.predict(X))

    def predict_proba(self, signals: list[np.ndarray]) -> np.ndarray:
        X = np.array([pooled_features(s) for s in signals])
        return self.clf.predict_proba(X)


# --- synthetic calls -------------------------------------------------------
# Distinct spectral signatures per class let the log-mel front end + classifier
# be exercised end-to-end in CI without any audio files. Replace with real
# recordings for production training; the interface is unchanged.
def synth_call(label: str, sr: int = 8000, dur: float = 0.4,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a synthetic vocalisation waveform for the given class."""
    rng = np.random.default_rng() if rng is None else rng
    n = int(sr * dur)
    t = np.arange(n) / sr
    if label == "normal":                       # quiet, low fundamental
        x = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
        x *= 0.5 + 0.3 * rng.random()
    elif label == "hunger":                     # rhythmic mid-band bursts
        env = np.sin(2 * np.pi * 6 * t) ** 2
        x = env * (0.6 * np.sin(2 * np.pi * 250 * t) + 0.3 * np.sin(2 * np.pi * 500 * t))
    elif label == "pain":                       # loud, high, harmonic-rich
        x = (0.8 * np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)
             + 0.3 * np.sin(2 * np.pi * 1500 * t))
    elif label == "respiratory":                # low broadband + cough bursts
        noise = rng.normal(0, 1, n)
        x = 0.5 * np.convolve(noise, np.ones(20) / 20, mode="same")
        burst = (np.sin(2 * np.pi * 3 * t) > 0.7).astype(float)
        x = x + 0.6 * burst * np.sin(2 * np.pi * 120 * t)
    else:
        raise ValueError(f"unknown class: {label}")
    x = x + 0.02 * rng.normal(0, 1, n)
    return x.astype(np.float32)


def make_acoustic_dataset(n_per_class: int = 40, seed: int = 0
                          ) -> tuple[list[np.ndarray], list[str]]:
    """Balanced synthetic set of (signals, labels) across the four CLASSES."""
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    labels: list[str] = []
    for cls in CLASSES:
        for _ in range(n_per_class):
            signals.append(synth_call(cls, rng=rng))
            labels.append(cls)
    return signals, labels
