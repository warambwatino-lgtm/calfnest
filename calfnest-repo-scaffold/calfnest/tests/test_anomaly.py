"""The per-animal detector flags windows unlike the animal's own normal."""
import numpy as np

from calfnest.anomaly import PerAnimalAnomaly


def test_flags_deviation_from_own_baseline():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.05, size=(30, 8)).astype(float)
    det = PerAnimalAnomaly(n_components=3, contamination=0.05).fit(normal)
    assert det.threshold_ is not None

    normal_like = rng.normal(0.0, 0.05, size=(5, 8))
    anomalous = normal_like + 1.0                 # shifted well off baseline
    assert det.score(anomalous).mean() > det.score(normal_like).mean()
    assert det.is_anomalous(anomalous).all()
