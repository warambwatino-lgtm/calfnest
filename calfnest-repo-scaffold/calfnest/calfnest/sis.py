"""Deterministic Stress Index Score (SIS).

This is the rule-based FAILSAFE layer, intentionally NOT machine learning.
It is retained so the device never depends solely on a model. The MQ-9
fire signal is a hardware-priority override handled in firmware.
"""
from __future__ import annotations
from dataclasses import dataclass

# Evidence-weighted contributions (sum of non-override weights = 100).
WEIGHTS = {
    "intake": 30,      # milk/colostrum + concentrate intake
    "vocal": 25,       # noise-filtered vocalisation events
    "movement": 20,    # lying / restlessness
    "environment": 15, # temperature + humidity stress
    "ammonia": 10,     # NH3 respiratory irritant
}


@dataclass(frozen=True)
class SISResult:
    score: int          # 0..100
    status: str         # "GREEN" | "AMBER" | "RED"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def stress_index(sub_scores: dict[str, float]) -> SISResult:
    """Compute the SIS from normalised sub-scores in [0, 1].

    Args:
        sub_scores: mapping of parameter name -> stress fraction in [0, 1],
            where 0 is ideal welfare and 1 is maximal stress on that axis.

    Returns:
        SISResult with an integer 0..100 score and a traffic-light status.
    """
    total = 0.0
    for name, weight in WEIGHTS.items():
        total += weight * _clamp01(float(sub_scores.get(name, 0.0)))
    score = int(round(total))
    if score <= 30:
        status = "GREEN"
    elif score <= 60:
        status = "AMBER"
    else:
        status = "RED"
    return SISResult(score=score, status=status)
