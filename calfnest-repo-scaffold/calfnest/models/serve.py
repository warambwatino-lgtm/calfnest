"""REST API that serves the trained CALFNEST Sentinel risk model.

This is the connection between the model and any interface (the dashboard, a
mobile app, or the edge gateway): a client POSTs a short window of an animal's
sensor streams to ``/predict`` and gets back the model's risk score, the
traffic-light band, and the rule-based Stress Index for cross-check.

Run it:

    make serve            # or: uvicorn models.serve:app --reload

Then, e.g.:

    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \\
         -d '{"streams": {"intake": [...], "vocal": [...], ...}}'

The scoring function ``score()`` is importable and unit-tested without starting
a server. FastAPI/uvicorn are optional (the ``serve`` extra); the model itself
needs only numpy/scikit-learn.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from calfnest.evaluate import balance_classes
from calfnest.features import feature_vector
from calfnest.rulemap import sis_subscores
from calfnest.sis import stress_index
from models.simulate import (CHANNELS, build_training_matrix, feature_names,
                             simulate_cohort)

_WINDOW = 6


@lru_cache(maxsize=1)
def get_model():
    """Train (once) and cache the balanced risk model the API serves."""
    from calfnest.model import RiskModel
    animals = simulate_cohort(seed=0)
    X, y, _ = build_training_matrix(animals, _WINDOW)
    Xb, yb = balance_classes(X, y, seed=0)
    return RiskModel(feature_names=feature_names(_WINDOW)).fit(Xb, yb)


def _band(risk: float) -> str:
    return "GREEN" if risk <= 30 else "AMBER" if risk <= 60 else "RED"


def score(streams: dict[str, list[float]],
          baselines: dict[str, float] | None = None) -> dict:
    """Risk-score one window. ``streams`` maps each channel to its recent values.

    Channels: intake, vocal, movement, environment, ammonia. If ``baselines``
    is omitted, each channel's baseline is taken from the start of its window.
    """
    streams = {c: list(map(float, streams[c])) for c in CHANNELS if c in streams}
    missing = [c for c in CHANNELS if c not in streams]
    if missing:
        raise ValueError(f"missing channels: {missing}")
    if baselines is None:
        baselines = {c: float(np.mean(v[: max(len(v) // 2, 1)]))
                     for c, v in streams.items()}

    feats = feature_vector(streams, baselines)
    names = feature_names(_WINDOW)
    x = np.array([[feats.get(n, 0.0) for n in names]])
    risk = float(get_model().predict_risk(x)[0] * 100.0)
    sis = float(stress_index(sis_subscores(streams, baselines)).score)
    return {"risk": round(risk, 1), "band": _band(risk),
            "sis": round(sis, 1), "channels": list(CHANNELS)}


# ---- optional HTTP layer (needs the `serve` extra: fastapi + uvicorn) -------
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        streams: dict[str, list[float]]
        baselines: dict[str, float] | None = None

    app = FastAPI(title="CALFNEST Sentinel API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "CALFNEST Sentinel risk head",
                "channels": list(CHANNELS)}

    @app.post("/predict")
    def predict(req: PredictRequest):
        return score(req.streams, req.baselines)

except ImportError:                       # pragma: no cover
    app = None                            # server not available; score() still works
