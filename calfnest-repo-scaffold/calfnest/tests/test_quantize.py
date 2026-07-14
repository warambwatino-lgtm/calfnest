"""int8 quantisation shrinks the model and preserves its decisions."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from calfnest.model import RiskModel
from models.quantize import (dequantize, quantize_int8, quantize_risk_model,
                             quantized_predict_risk)
from models.simulate import build_training_matrix, feature_names, simulate_cohort

pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning")


def test_quantize_int8_roundtrip():
    w = np.array([[-1.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    q, s = quantize_int8(w)
    assert q.dtype == np.int8
    assert np.max(np.abs(dequantize(q, s) - w)) < s      # within one step


def test_model_quantisation_small_and_faithful():
    animals = simulate_cohort(seed=0)
    tr, te = train_test_split(animals, test_size=0.4, random_state=0,
                              stratify=[a.is_case for a in animals])
    Xtr, ytr, *_ = build_training_matrix(tr, 6)
    Xte, yte, *_ = build_training_matrix(te, 6)
    m = RiskModel(feature_names=feature_names(6)).fit(Xtr, ytr)

    rep = quantize_risk_model(m)
    assert rep["int8_bytes"] < rep["fp32_bytes"]
    assert rep["within_budget"]                          # < 256 KB flash

    r_fp32 = m.predict_risk(Xte)
    r_int8 = quantized_predict_risk(m, Xte, rep)
    assert np.mean(np.abs(r_fp32 - r_int8)) < 0.05
    assert np.mean((r_fp32 >= 0.5) == (r_int8 >= 0.5)) > 0.95
