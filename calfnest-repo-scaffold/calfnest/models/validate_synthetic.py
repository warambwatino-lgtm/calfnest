"""Validate GAN/TimeGAN-synthesised sensor sequences against real data.

Implements the statistical correlation tests the AI4I rubric (C3) rewards:
  1. Per-feature Kolmogorov-Smirnov distance (distributional similarity)
  2. Jensen-Shannon divergence (distributional similarity)
  3. Autocorrelation-structure error (temporal fidelity)
  4. Cross-channel correlation-matrix error (inter-sensor structure)
  5. Real-vs-synthetic discriminator AUC (~0.5 == indistinguishable)

Run:  python -m models.validate_synthetic
The __main__ block uses a seeded toy generator so the script runs anywhere;
swap in your real arrays (shape [n_samples, n_timesteps, n_channels]).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def ks_distances(real: np.ndarray, synth: np.ndarray) -> np.ndarray:
    """Per-channel KS statistic between flattened real and synthetic values."""
    c = real.shape[-1]
    return np.array([
        ks_2samp(real[..., i].ravel(), synth[..., i].ravel()).statistic
        for i in range(c)
    ])


def js_divergences(real: np.ndarray, synth: np.ndarray, bins: int = 30) -> np.ndarray:
    """Per-channel Jensen-Shannon divergence on histogram densities."""
    c = real.shape[-1]
    out = []
    for i in range(c):
        r = real[..., i].ravel()
        s = synth[..., i].ravel()
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        edges = np.linspace(lo, hi, bins + 1)
        pr, _ = np.histogram(r, bins=edges, density=True)
        ps, _ = np.histogram(s, bins=edges, density=True)
        pr += 1e-12
        ps += 1e-12
        out.append(float(jensenshannon(pr, ps)))
    return np.array(out)


def autocorr_error(real: np.ndarray, synth: np.ndarray, lag: int = 1) -> float:
    """Mean abs difference in lag-`lag` autocorrelation across channels."""
    def ac(x):  # x: [n, t, c]
        c = x.shape[-1]
        vals = []
        for i in range(c):
            a = x[:, :-lag, i].ravel()
            b = x[:, lag:, i].ravel()
            if a.std() == 0 or b.std() == 0:
                vals.append(0.0)
            else:
                vals.append(np.corrcoef(a, b)[0, 1])
        return np.array(vals)
    return float(np.mean(np.abs(ac(real) - ac(synth))))


def cross_corr_error(real: np.ndarray, synth: np.ndarray) -> float:
    """Frobenius norm of the difference in inter-channel correlation matrices."""
    def cm(x):
        flat = x.reshape(-1, x.shape[-1])
        return np.corrcoef(flat, rowvar=False)
    return float(np.linalg.norm(cm(real) - cm(synth)))


def discriminator_auc(real: np.ndarray, synth: np.ndarray) -> float:
    """Cross-val AUC of a classifier separating real from synthetic.

    ~0.5 means synthetic is statistically indistinguishable from real (good).
    """
    xr = real.reshape(real.shape[0], -1)
    xs = synth.reshape(synth.shape[0], -1)
    X = np.vstack([xr, xs])
    y = np.concatenate([np.ones(len(xr)), np.zeros(len(xs))])
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def report(real: np.ndarray, synth: np.ndarray) -> dict:
    return {
        "ks_mean": float(ks_distances(real, synth).mean()),
        "js_mean": float(js_divergences(real, synth).mean()),
        "autocorr_err": autocorr_error(real, synth),
        "cross_corr_err": cross_corr_error(real, synth),
        "discriminator_auc": discriminator_auc(real, synth),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Toy stand-ins: replace with real [n, t, c] arrays and your generator output.
    n, t, c = 200, 24, 6
    real = rng.normal(0, 1, (n, t, c)).cumsum(axis=1)
    synth = rng.normal(0, 1, (n, t, c)).cumsum(axis=1)  # naive generator
    r = report(real, synth)
    print("Synthetic-data validation report")
    for k, v in r.items():
        print(f"  {k:>18}: {v:.4f}")
    print("\nInterpretation: lower KS/JS/errors are better; discriminator "
          "AUC near 0.50 means real and synthetic are indistinguishable.")
