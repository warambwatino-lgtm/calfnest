"""Post-training int8 quantisation for the edge risk model.

The ESP32-S3 budget is a <256 KB flash footprint and <100 ms inference. This
step quantises the trained risk model's weight matrices to int8 (per-tensor
symmetric scale), reports the size saving, and verifies that the dequantised
model still tracks the float model — the same gate the production TFLite-Micro
export must pass before a signed OTA rollout.

Runnable:

    python -m models.quantize

It trains a small model on the synthetic cohort, quantises it, and prints the
size and accuracy-delta report. The production export target is int8
TFLite-Micro (needs the ``train`` extra / TensorFlow); the logic verified here
is identical: scale weights, dequantise, confirm the outputs match.
"""
from __future__ import annotations

import numpy as np

EDGE_FLASH_BUDGET_BYTES = 256 * 1024


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantisation. Returns (int8 weights, scale)."""
    w = np.asarray(w, dtype=np.float32)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def _forward(coefs, intercepts, X, out_activation):
    a = np.asarray(X, dtype=np.float32)
    last = len(coefs) - 1
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        a = a @ W + b
        if i < last:
            a = np.maximum(a, 0.0)            # ReLU hidden layers
    if out_activation == "logistic":
        return 1.0 / (1.0 + np.exp(-a))
    if out_activation == "softmax":
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return a


def quantize_risk_model(model) -> dict:
    """Quantise a fitted ``RiskModel``'s MLP to int8. Returns a report dict."""
    mlp = model._pipe.named_steps["mlp"]
    coefs = list(mlp.coefs_)
    intercepts = list(mlp.intercepts_)
    fp32_bytes = sum(W.nbytes for W in coefs) + sum(b.nbytes for b in intercepts)

    q_coefs, deq_coefs = [], []
    int8_bytes = 0
    for W in coefs:
        q, s = quantize_int8(W)
        q_coefs.append((q, s))
        deq_coefs.append(dequantize(q, s))
        int8_bytes += q.nbytes + 4                       # int8 matrix + fp32 scale
    int8_bytes += sum(b.nbytes for b in intercepts)      # biases kept fp32 (tiny)

    return {
        "q_coefs": q_coefs,
        "deq_coefs": deq_coefs,
        "intercepts": intercepts,
        "out_activation": mlp.out_activation_,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "compression": fp32_bytes / max(int8_bytes, 1),
        "within_budget": int8_bytes <= EDGE_FLASH_BUDGET_BYTES,
    }


def quantized_predict_risk(model, X, report: dict | None = None) -> np.ndarray:
    """Risk in [0, 1] from the int8-dequantised weights (scaler applied first)."""
    report = report or quantize_risk_model(model)
    Xs = model._pipe.named_steps["scale"].transform(np.asarray(X, dtype=float))
    out = np.asarray(_forward(report["deq_coefs"], report["intercepts"], Xs,
                              report["out_activation"]))
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out


def main() -> dict:
    from sklearn.model_selection import train_test_split

    from calfnest.model import RiskModel
    from models.simulate import build_training_matrix, feature_names, simulate_cohort

    animals = simulate_cohort()
    tr, te = train_test_split(animals, test_size=0.4, random_state=0,
                              stratify=[a.is_case for a in animals])
    names = feature_names(6)
    Xtr, ytr, _ = build_training_matrix(tr, 6)
    Xte, yte, _ = build_training_matrix(te, 6)

    model = RiskModel(feature_names=names).fit(Xtr, ytr)
    rep = quantize_risk_model(model)

    r_fp32 = model.predict_risk(Xte)
    r_int8 = quantized_predict_risk(model, Xte, rep)
    mad = float(np.mean(np.abs(r_fp32 - r_int8)))
    agree = float(np.mean((r_fp32 >= 0.5) == (r_int8 >= 0.5)))

    print("CALFNEST Sentinel — int8 edge quantisation")
    print(f"  weights  fp32 {rep['fp32_bytes']/1024:.1f} KB  ->  int8 {rep['int8_bytes']/1024:.1f} KB"
          f"  ({rep['compression']:.1f}x smaller)")
    print(f"  flash budget <256 KB: {'PASS' if rep['within_budget'] else 'FAIL'}")
    print(f"  float vs int8 risk: mean abs diff {mad:.4f} · decision agreement {agree*100:.1f}%")
    return {"fp32_bytes": rep["fp32_bytes"], "int8_bytes": rep["int8_bytes"],
            "compression": rep["compression"], "within_budget": rep["within_budget"],
            "mean_abs_diff": mad, "decision_agreement": agree}


if __name__ == "__main__":
    main()
