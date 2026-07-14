"""Map a raw sensor window to SIS sub-scores.

The deterministic SIS (``sis.py``) expects normalised stress fractions in
[0, 1]. To benchmark the rule head-to-head against the AI on identical data,
we need to derive those fractions from the same windows the model sees. This
module is that adapter, and it is the only place the rule's normalisation
constants live.
"""
from __future__ import annotations

from typing import Sequence

from .features import baseline_deviation, intake_decline_fraction, rolling_mean

# Normalisation scales: how big a deviation counts as "fully stressed" (=1.0)
# on each axis. Tunable; documented here so the rule stays auditable.
_SCALE = {
    "vocal": 0.6,
    "movement": 0.7,
    "environment": 5.0,
    "ammonia": 1.5,
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def sis_subscores(streams: dict[str, Sequence[float]],
                  baselines: dict[str, float]) -> dict[str, float]:
    """Return the {intake, vocal, movement, environment, ammonia} stress dict.

    * intake  -> fraction the intake has fallen below the animal's baseline
    * vocal / ammonia -> upward deviation (more == worse), scaled
    * movement / environment -> absolute deviation (either direction), scaled
    """
    subs: dict[str, float] = {}
    subs["intake"] = intake_decline_fraction(
        streams.get("intake", []), baselines.get("intake", 0.0)
    )
    subs["vocal"] = _clamp01(
        baseline_deviation(streams.get("vocal", []), baselines.get("vocal", 0.0))
        / _SCALE["vocal"]
    )
    subs["movement"] = _clamp01(
        abs(baseline_deviation(streams.get("movement", []), baselines.get("movement", 0.0)))
        / _SCALE["movement"]
    )
    subs["environment"] = _clamp01(
        abs(baseline_deviation(streams.get("environment", []),
                               baselines.get("environment", 0.0)))
        / _SCALE["environment"]
    )
    subs["ammonia"] = _clamp01(
        baseline_deviation(streams.get("ammonia", []), baselines.get("ammonia", 0.0))
        / _SCALE["ammonia"]
    )
    return subs
