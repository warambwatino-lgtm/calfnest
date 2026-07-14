"""The acoustic head separates the four call classes above chance."""
import numpy as np
from sklearn.model_selection import train_test_split

from calfnest.acoustic import (AcousticClassifier, CLASSES, logmel,
                               make_acoustic_dataset, synth_call)


def test_logmel_and_synth_shapes():
    sig = synth_call("pain", sr=8000, dur=0.4)
    assert sig.ndim == 1 and sig.size == int(8000 * 0.4)
    lm = logmel(sig)
    assert lm.ndim == 2 and np.isfinite(lm).all()


def test_classifier_learns_classes():
    sigs, labs = make_acoustic_dataset(n_per_class=40, seed=0)
    tr, te, ytr, yte = train_test_split(sigs, labs, test_size=0.3,
                                        random_state=0, stratify=labs)
    clf = AcousticClassifier().fit(tr, ytr)
    preds = clf.predict(te)
    acc = np.mean([p == y for p, y in zip(preds, yte)])
    assert acc > 0.7                              # 4-class chance is 0.25
    assert set(preds).issubset(set(CLASSES))
