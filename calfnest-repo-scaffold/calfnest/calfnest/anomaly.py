"""Per-animal anomaly detector (the third Sentinel head).

Fixed thresholds and a single population model both miss the calf that is
subtly declining *relative to its own normal* while every global limit still
reads green. This head learns each animal's baseline and scores how far a new
window departs from it, via PCA reconstruction error — a lightweight, robust
autoencoder that runs on the edge and needs no labels.

Fit one detector per animal on its early, healthy windows; then score live
windows. High reconstruction error == the animal is behaving unlike itself.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PerAnimalAnomaly:
    """PCA reconstruction-error anomaly detector for a single animal."""

    def __init__(self, n_components: int = 3, contamination: float = 0.05):
        self.n_components = n_components
        self.contamination = contamination
        self._pca: PCA | None = None
        self._mean: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, normal_windows: np.ndarray) -> "PerAnimalAnomaly":
        """Fit on windows representing the animal's own healthy behaviour."""
        X = np.asarray(normal_windows, dtype=float)
        k = int(min(self.n_components, X.shape[1], max(1, X.shape[0] - 1)))
        self._mean = X.mean(axis=0)
        self._pca = PCA(n_components=k, random_state=0).fit(X - self._mean)
        errs = self._errors(X)
        # threshold at the (1 - contamination) quantile of self-error
        self.threshold_ = float(np.quantile(errs, 1.0 - self.contamination))
        return self

    def _errors(self, X: np.ndarray) -> np.ndarray:
        assert self._pca is not None and self._mean is not None
        centred = X - self._mean
        recon = self._pca.inverse_transform(self._pca.transform(centred))
        return np.sqrt(((centred - recon) ** 2).sum(axis=1))

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Return reconstruction error per window (higher == more anomalous)."""
        X = np.atleast_2d(np.asarray(windows, dtype=float))
        return self._errors(X)

    def is_anomalous(self, windows: np.ndarray) -> np.ndarray:
        """Boolean flag per window using the fitted per-animal threshold."""
        if self.threshold_ is None:
            raise RuntimeError("call fit() first")
        return self.score(windows) > self.threshold_
