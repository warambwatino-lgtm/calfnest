"""Temporal feature extraction from the six welfare sensor streams.

The Sentinel risk model consumes engineered features over a sliding window,
plus each animal's rolling personal baseline. Keeping feature logic here
(rather than in the model) makes it unit-testable and portable to the
firmware's C implementation.
"""
from __future__ import annotations
from typing import Sequence
import math


def rolling_mean(x: Sequence[float]) -> float:
    xs = list(x)
    return sum(xs) / len(xs) if xs else 0.0


def rolling_std(x: Sequence[float]) -> float:
    xs = list(x)
    if len(xs) < 2:
        return 0.0
    m = rolling_mean(xs)
    var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
    return math.sqrt(var)


def baseline_deviation(window: Sequence[float], baseline: float) -> float:
    """Signed deviation of the window mean from an animal's personal baseline.

    Positive means the window is above the animal's own normal. This is the
    core 'per-animal' signal a fixed global threshold cannot capture.
    """
    return rolling_mean(window) - float(baseline)


def intake_decline_fraction(window: Sequence[float], baseline: float) -> float:
    """Fraction by which intake has fallen below the animal's baseline, in [0,1].

    Reduced intake is the strongest pre-weaning mortality predictor, so we
    surface it explicitly as a bounded stress fraction usable by the SIS too.
    """
    b = float(baseline)
    if b <= 0:
        return 0.0
    m = rolling_mean(window)
    drop = (b - m) / b
    return 0.0 if drop < 0 else 1.0 if drop > 1 else drop


def feature_vector(streams: dict[str, Sequence[float]],
                   baselines: dict[str, float]) -> dict[str, float]:
    """Assemble the model input for one inference cycle.

    Args:
        streams: name -> recent window of readings (e.g. 'intake', 'vocal', ...)
        baselines: name -> this animal's rolling personal baseline
    """
    feats: dict[str, float] = {}
    for name, window in streams.items():
        feats[f"{name}_mean"] = rolling_mean(window)
        feats[f"{name}_std"] = rolling_std(window)
        feats[f"{name}_dev"] = baseline_deviation(window, baselines.get(name, 0.0))
    if "intake" in streams:
        feats["intake_decline"] = intake_decline_fraction(
            streams["intake"], baselines.get("intake", 0.0)
        )
    return feats
