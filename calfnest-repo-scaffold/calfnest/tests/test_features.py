import math
from calfnest.features import (
    rolling_mean, rolling_std, baseline_deviation,
    intake_decline_fraction, feature_vector,
)


def test_rolling_mean_and_std():
    assert rolling_mean([2, 4, 6]) == 4
    assert math.isclose(rolling_std([2, 4, 6]), 2.0)
    assert rolling_std([5]) == 0.0


def test_baseline_deviation_sign():
    assert baseline_deviation([10, 10, 10], 8) == 2.0
    assert baseline_deviation([6, 6, 6], 8) == -2.0


def test_intake_decline_is_bounded():
    assert intake_decline_fraction([4, 4], 8) == 0.5     # halved intake
    assert intake_decline_fraction([9, 9], 8) == 0.0     # above baseline -> 0
    assert intake_decline_fraction([0, 0], 8) == 1.0     # full decline
    assert intake_decline_fraction([1, 1], 0) == 0.0     # guard zero baseline


def test_feature_vector_keys():
    streams = {"intake": [4, 4, 4], "vocal": [1, 2, 3]}
    baselines = {"intake": 8, "vocal": 1}
    f = feature_vector(streams, baselines)
    assert "intake_decline" in f
    assert "intake_mean" in f and "vocal_dev" in f
    assert f["intake_decline"] == 0.5
