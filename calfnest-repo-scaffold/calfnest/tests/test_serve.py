"""The model API scores windows and reacts to decline."""
import pytest
from models.serve import score

HEALTHY = {"intake":[1.0,1.0,0.99,1.01,1.0,1.0], "vocal":[0.15,0.15,0.16,0.14,0.15,0.15],
           "movement":[0.6,0.6,0.61,0.59,0.6,0.6], "environment":[0.3,0.3,0.3,0.31,0.3,0.3],
           "ammonia":[0.2,0.2,0.2,0.21,0.2,0.2]}
DECLINING = {"intake":[1.0,0.9,0.75,0.6,0.45,0.3], "vocal":[0.15,0.25,0.4,0.55,0.7,0.85],
             "movement":[0.6,0.55,0.48,0.4,0.32,0.25], "environment":[0.3,0.32,0.35,0.38,0.4,0.42],
             "ammonia":[0.2,0.3,0.4,0.5,0.6,0.7]}

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def test_declining_scores_higher_and_bands_are_valid():
    h, d = score(HEALTHY), score(DECLINING)
    assert d["risk"] > h["risk"]
    assert h["band"] in {"GREEN", "AMBER", "RED"} and d["band"] == "RED"
    assert 0.0 <= h["risk"] <= 100.0


def test_missing_channel_raises():
    with pytest.raises(ValueError):
        score({"intake": [1, 1, 1, 1, 1, 1]})
