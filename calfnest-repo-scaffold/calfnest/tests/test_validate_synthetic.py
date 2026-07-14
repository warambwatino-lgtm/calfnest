import numpy as np
from models.validate_synthetic import report, discriminator_auc


def test_identical_data_is_indistinguishable():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (120, 12, 4)).cumsum(axis=1)
    # near-identical copy with tiny noise -> discriminator ~0.5, low distances
    synth = real + rng.normal(0, 1e-3, real.shape)
    r = report(real, synth)
    assert r["ks_mean"] < 0.15
    assert 0.35 <= r["discriminator_auc"] <= 0.65


def test_report_has_all_metrics():
    rng = np.random.default_rng(1)
    real = rng.normal(0, 1, (80, 10, 3)).cumsum(axis=1)
    synth = rng.normal(0, 1, (80, 10, 3)).cumsum(axis=1)
    r = report(real, synth)
    for k in ("ks_mean", "js_mean", "autocorr_err", "cross_corr_err", "discriminator_auc"):
        assert k in r
