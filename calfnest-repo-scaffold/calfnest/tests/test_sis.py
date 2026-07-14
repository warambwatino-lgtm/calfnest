from calfnest.sis import stress_index, WEIGHTS


def test_all_calm_is_green():
    r = stress_index({k: 0.0 for k in WEIGHTS})
    assert r.score == 0
    assert r.status == "GREEN"


def test_all_max_is_red():
    r = stress_index({k: 1.0 for k in WEIGHTS})
    assert r.score == 100
    assert r.status == "RED"


def test_amber_band_boundaries():
    # intake fully stressed = 30 -> GREEN edge
    assert stress_index({"intake": 1.0}).status == "GREEN"
    # intake + vocal = 55 -> AMBER
    assert stress_index({"intake": 1.0, "vocal": 1.0}).status == "AMBER"
    # intake + vocal + movement = 75 -> RED
    assert stress_index({"intake": 1.0, "vocal": 1.0, "movement": 1.0}).status == "RED"


def test_inputs_are_clamped():
    r = stress_index({"intake": 5.0})   # out-of-range clamps to 1.0
    assert r.score == 30


def test_unknown_keys_ignored():
    assert stress_index({"not_a_sensor": 1.0}).score == 0
