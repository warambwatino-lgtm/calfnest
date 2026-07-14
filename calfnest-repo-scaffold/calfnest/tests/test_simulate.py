"""Synthetic cohort has the structure the models rely on."""
import numpy as np
from models.simulate import (Animal, animal_windows, build_training_matrix,
                             feature_names, simulate_cohort)


def test_cohort_shapes_and_labels():
    animals = simulate_cohort(n_healthy=20, n_case=15, T=72, seed=1)
    assert len(animals) == 35
    assert all(isinstance(a, Animal) for a in animals)
    for a in animals:
        assert a.streams.shape[1] == 5
        assert (a.event_time is None) == (not a.is_case)


def test_case_animals_decline_in_intake():
    animals = simulate_cohort(n_healthy=0, n_case=8, T=72, seed=2)
    for a in animals:
        early = a.streams[:12, 0].mean()   # intake channel
        late = a.streams[-6:, 0].mean()
        assert late < early                # intake falls into the event


def test_training_matrix_and_windows():
    animals = simulate_cohort(n_healthy=10, n_case=10, T=64, seed=3)
    X, y, *_ = build_training_matrix(animals, window=6)
    assert X.ndim == 2 and X.shape[0] == y.shape[0]
    assert set(np.unique(y)).issubset({0, 1})
    assert len(feature_names(6)) == X.shape[1]
    W = animal_windows(animals[0], window=6)
    assert W.ndim == 2 and W.shape[0] > 0
