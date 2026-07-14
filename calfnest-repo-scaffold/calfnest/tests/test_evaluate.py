"""Lead-time / FAR benchmark maths behave as specified."""
import numpy as np

from calfnest.evaluate import (benchmark, lead_times, sensitivity_specificity,
                               threshold_for_far)


def test_sensitivity_specificity_toy():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    sens, spec = sensitivity_specificity(y_true, y_pred)
    assert sens == 0.5 and spec == 1.0


def test_threshold_hits_target_far():
    peaks = np.linspace(0, 1, 100)
    healthy = [np.array([p]) for p in peaks]
    thr = threshold_for_far(healthy, target_far=0.10)
    realised = np.mean([p >= thr for p in peaks])
    assert abs(realised - 0.10) < 0.03


def test_lead_time_positive_before_event():
    # risk crosses 0.5 at index 3; event at index 6 -> lead 3
    case = [np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])]
    healthy = [np.zeros(7)]
    res = lead_times(case, [6], healthy, threshold=0.5, sample_hours=1.0)
    assert res.detection_rate == 1.0 and res.mean_lead == 3.0


def test_benchmark_returns_result_per_scorer():
    case = [np.array([0.0, 0.4, 0.9])]
    healthy = [np.array([0.0, 0.1, 0.2])]
    out = benchmark({"a": (healthy, case), "b": (healthy, case)},
                    event_times=[2], target_far=0.1)
    assert set(out) == {"a", "b"}
    assert out["a"].detection_rate == 1.0


def test_classification_metrics_toy():
    from calfnest.evaluate import classification_metrics
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]           # tp=2, fn=1, tn=2, fp=0
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.8
    assert m["precision"] == 1.0        # 2/(2+0)
    assert abs(m["recall"] - 2/3) < 1e-9
    assert abs(m["f1"] - 0.8) < 1e-9
