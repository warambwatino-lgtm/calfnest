"""The rule adapter maps windows to valid SIS sub-scores, and reacts to stress."""
from calfnest.rulemap import sis_subscores
from calfnest.sis import stress_index
from models.simulate import _window_streams, simulate_cohort


def _score_at(a, t):
    subs = sis_subscores(_window_streams(a, t, 6), a.baselines)
    for v in subs.values():
        assert 0.0 <= v <= 1.0
    return stress_index(subs).score


def test_declining_animal_scores_higher_late():
    case = next(a for a in simulate_cohort(seed=4) if a.is_case)
    early = _score_at(case, 12)
    late = _score_at(case, case.streams.shape[0] - 1)
    assert late >= early
