"""The risk head learns and its outputs are well-formed and persistable."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from calfnest.model import RiskModel
from models.simulate import build_training_matrix, feature_names, simulate_cohort

pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning")


def _data():
    animals = simulate_cohort(seed=0)
    tr, te = train_test_split(animals, test_size=0.4, random_state=0,
                              stratify=[a.is_case for a in animals])
    Xtr, ytr, *_ = build_training_matrix(tr, 6)
    Xte, yte, *_ = build_training_matrix(te, 6)
    return Xtr, ytr, Xte, yte


def test_risk_in_unit_interval_and_beats_chance():
    Xtr, ytr, Xte, yte = _data()
    m = RiskModel(feature_names=feature_names(6)).fit(Xtr, ytr)
    r = m.predict_risk(Xte)
    assert r.shape == (Xte.shape[0],)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
    assert roc_auc_score(yte, r) > 0.8          # clearly better than chance


def test_save_load_roundtrip(tmp_path):
    Xtr, ytr, Xte, _ = _data()
    m = RiskModel(feature_names=feature_names(6)).fit(Xtr, ytr)
    card = m.save(tmp_path / "risk.joblib")
    assert len(card["sha256"]) == 64
    back = RiskModel.load(tmp_path / "risk.joblib")
    assert np.allclose(m.predict_risk(Xte), back.predict_risk(Xte))
