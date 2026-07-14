"""Seeded neonatal-cohort simulator for development and CI.

Real labelled decline/mortality events are scarce (that is why the proposal
leans on synthetic augmentation). This module generates a physiologically
plausible, fully deterministic cohort so the model, the baselines and the
benchmark all run end-to-end without the farm dataset. Swap in real arrays of
the same shape and every downstream script is unchanged.

Each animal is a [T, 5] array over the welfare channels; a "case" animal has a
gradual multi-channel decline ramping into an event, which is exactly the
trajectory the AI is meant to catch before any single threshold trips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calfnest.features import feature_vector

CHANNELS = ("intake", "vocal", "movement", "environment", "ammonia")
_BASE = {"intake": 2.0, "vocal": 0.2, "movement": 0.5, "environment": 25.0, "ammonia": 1.0}
_BASELINE_STEPS = 12   # early healthy window used to learn each animal's own normal


@dataclass
class Animal:
    streams: np.ndarray            # [T, 5]
    baselines: dict                # channel -> personal baseline
    is_case: bool
    event_time: int | None         # index in [0, T) or None


def _ar_noise(T: int, rng, sigma: float, rho: float = 0.6) -> np.ndarray:
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(0, sigma)
    return x


def _animal(rng, is_case: bool, T: int) -> Animal:
    s = np.zeros((T, len(CHANNELS)))
    off = {c: rng.normal(0, 0.05 * abs(_BASE[c]) + 1e-3) for c in CHANNELS}
    for i, c in enumerate(CHANNELS):
        sigma = {"intake": 0.06, "vocal": 0.04, "movement": 0.05,
                 "environment": 0.5, "ammonia": 0.06}[c]
        s[:, i] = _BASE[c] + off[c] + _ar_noise(T, rng, sigma)

    event_time = None
    if is_case:
        event_time = int(rng.integers(T - 32, T - 2))
        decline_len = int(rng.integers(14, 22))
        onset = max(_BASELINE_STEPS + 1, event_time - decline_len)
        ramp = np.clip((np.arange(T) - onset) / max(1, event_time - onset), 0, 1)
        ramp[np.arange(T) > event_time] = 1.0
        s[:, 0] -= ramp * 0.6 * _BASE["intake"]     # intake falls
        s[:, 1] += ramp * 0.7                        # vocalisation rises
        s[:, 2] += ramp * 0.9                        # movement/restlessness rises
        s[:, 4] += ramp * 1.4                        # ammonia rises
        # environment (channel 3) left near-normal on purpose

    baselines = {c: float(s[:_BASELINE_STEPS, i].mean()) for i, c in enumerate(CHANNELS)}
    return Animal(streams=s, baselines=baselines, is_case=is_case, event_time=event_time)


def simulate_cohort(n_healthy: int = 120, n_case: int = 80, T: int = 72,
                    seed: int = 7) -> list[Animal]:
    rng = np.random.default_rng(seed)
    animals = [_animal(rng, False, T) for _ in range(n_healthy)]
    animals += [_animal(rng, True, T) for _ in range(n_case)]
    rng.shuffle(animals)
    return animals


def _window_streams(a: Animal, t: int, window: int) -> dict:
    return {c: a.streams[t - window:t, i] for i, c in enumerate(CHANNELS)}


def feature_names(window: int = 6) -> list[str]:
    """Canonical, ordered feature names (so train and inference agree)."""
    a = _animal(np.random.default_rng(0), True, 40)
    feats = feature_vector(_window_streams(a, 20, window), a.baselines)
    return sorted(feats.keys())


def to_vector(feats: dict, names: list[str]) -> np.ndarray:
    return np.array([feats[n] for n in names], dtype=float)


def build_training_matrix(animals: list[Animal], window: int = 6):
    """Pool per-timestep windows into (X, y, names). y=1 inside the decline ramp."""
    names = feature_names(window)
    X, y = [], []
    for a in animals:
        T = a.streams.shape[0]
        for t in range(window, T):
            feats = feature_vector(_window_streams(a, t, window), a.baselines)
            X.append(to_vector(feats, names))
            label = 0
            if a.is_case and a.event_time is not None:
                onset = a.event_time - 21
                label = 1 if (onset <= t <= a.event_time) else 0
            y.append(label)
    return np.array(X), np.array(y), names


def animal_feature_series(a: Animal, names: list[str], window: int = 6):
    """Return (feature_matrix over time, event_idx) in feature-timeline space."""
    T = a.streams.shape[0]
    rows = [to_vector(feature_vector(_window_streams(a, t, window), a.baselines), names)
            for t in range(window, T)]
    event_idx = None if a.event_time is None else max(0, a.event_time - window)
    return np.array(rows), event_idx


def animal_windows(a: Animal, window: int = 6) -> np.ndarray:
    """Flattened raw windows over time — input for the per-animal anomaly head."""
    T = a.streams.shape[0]
    return np.array([a.streams[t - window:t, :].ravel() for t in range(window, T)])
