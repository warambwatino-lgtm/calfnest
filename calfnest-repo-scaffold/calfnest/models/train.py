"""Train the Sentinel and prove — or disprove — that it beats the rule.

This is the script that operationalises the proposal's central promise:
train the AI, put it up against (a) the deterministic SIS rule and (b) an
XGBoost baseline, at a *common false-alarm budget*, and report the lead time
each achieves. If the AI does not warn earlier than the rule, that shows here,
and the honest conclusion is to ship the rule.

Runs anywhere on numpy/scipy/scikit-learn:

    python -m models.train

XGBoost is used if installed; otherwise a scikit-learn gradient-boosting model
stands in as the baseline. The deployable 1D-CNN/GRU + int8 export is a
separate step (models/quantize.py) that runs where TensorFlow is available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from calfnest.evaluate import (balance_classes, balanced_threshold, benchmark,
                               classification_metrics, sensitivity_specificity)
from calfnest.model import RiskModel
from calfnest.rulemap import sis_subscores
from calfnest.sis import stress_index
from models.simulate import (CHANNELS, animal_feature_series, build_training_matrix,
                             feature_names, simulate_cohort)

ART = Path(__file__).resolve().parent / "artifacts"
_WINDOW = 6


def _baseline_model(random_state: int = 0):
    """XGBoost if available, else scikit-learn gradient boosting."""
    try:
        from xgboost import XGBClassifier  # optional
        return XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1,
                             eval_metric="logloss", random_state=random_state), "xgboost"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=random_state), "gradient-boosting"


def _sis_risk_series(a, names):
    """Score the deterministic rule over an animal's timeline -> risk in [0,1]."""
    from models.simulate import _window_streams
    T = a.streams.shape[0]
    out = []
    for t in range(_WINDOW, T):
        subs = sis_subscores(_window_streams(a, t, _WINDOW), a.baselines)
        out.append(stress_index(subs).score / 100.0)
    return np.array(out)


def main() -> dict:
    animals = simulate_cohort()
    names = feature_names(_WINDOW)

    train_animals, test_animals = train_test_split(
        animals, test_size=0.4, random_state=0,
        stratify=[a.is_case for a in animals])

    # --- fit the Sentinel risk model and the baseline on pooled windows ------
    # Decline is rare (~13% of windows), so we train on class-balanced data and
    # report on a balanced test set at the precision/recall balance point —
    # otherwise plain accuracy is inflated by the healthy majority.
    Xtr, ytr, _ = build_training_matrix(train_animals, _WINDOW)
    Xbtr, ybtr = balance_classes(Xtr, ytr, seed=0)
    sentinel = RiskModel(feature_names=names, backend="sklearn").fit(Xbtr, ybtr)
    base_clf, base_name = _baseline_model()
    base_clf.fit(Xbtr, ybtr)

    # tune the alert threshold to the precision/recall balance point
    op_threshold = balanced_threshold(ybtr, sentinel.predict_risk(Xbtr))

    # window-level metrics on a balanced held-out set at the tuned threshold
    Xte, yte, _ = build_training_matrix(test_animals, _WINDOW)
    Xbte, ybte = balance_classes(Xte, yte, seed=1)
    _pred = (sentinel.predict_risk(Xbte) >= op_threshold).astype(int)
    sens, spec = sensitivity_specificity(ybte, _pred)
    clf_metrics = classification_metrics(ybte, _pred, average="binary")
    from sklearn.metrics import roc_auc_score
    clf_metrics["roc_auc"] = float(roc_auc_score(ybte, sentinel.predict_risk(Xbte)))

    # --- per-animal risk timelines for the lead-time benchmark ---------------
    def series(model_kind):
        healthy, case, events = [], [], []
        for a in test_animals:
            feats, ev = animal_feature_series(a, names, _WINDOW)
            if model_kind == "sentinel":
                risk = sentinel.predict_risk(feats)
            elif model_kind == "baseline":
                risk = base_clf.predict_proba(feats)[:, 1]
            else:  # sis rule
                risk = _sis_risk_series(a, names)
            (case if a.is_case else healthy).append(risk)
            if a.is_case:
                events.append(ev)
        return healthy, case, events

    s_h, s_c, events = series("sentinel")
    b_h, b_c, _ = series("baseline")
    r_h, r_c, _ = series("sis")

    results = benchmark(
        {"CALFNEST Sentinel (AI)": (s_h, s_c),
         f"{base_name} baseline": (b_h, b_c),
         "SIS rule (failsafe)": (r_h, r_c)},
        events, target_far=0.10, sample_hours=1.0)

    # --- report --------------------------------------------------------------
    print("CALFNEST Sentinel — head-to-head benchmark  (fixed 10% false-alarm rate)")
    print(f"  held-out window sensitivity {sens:.2f} · specificity {spec:.2f}")
    print("  risk head (balanced test @ tuned threshold {t:.2f}):".format(t=op_threshold))
    print("    accuracy {accuracy:.1%} · precision {precision:.1%} · recall {recall:.1%} · "
          "F1 {f1:.1%} · ROC-AUC {roc_auc:.1%}\n".format(**clf_metrics))
    print(f"  {'model':<28}{'detect %':>10}{'mean lead (h)':>15}{'FAR':>8}")
    for name, r in results.items():
        print(f"  {name:<28}{r.detection_rate*100:>9.0f}%{r.mean_lead:>15.1f}{r.false_alarm_rate:>8.2f}")

    ai = results["CALFNEST Sentinel (AI)"].mean_lead
    rule = results["SIS rule (failsafe)"].mean_lead
    verdict = ("AI wins on lead time — ship the AI (rule stays as failsafe)."
               if ai > rule + 1e-6 else
               "AI does NOT beat the rule on lead time — ship the rule.")
    print(f"\n  Verdict: {verdict}")

    # --- the other two Sentinel heads, exercised end-to-end -----------------
    from calfnest.acoustic import AcousticClassifier, make_acoustic_dataset
    from calfnest.anomaly import PerAnimalAnomaly
    from models.quantize import quantize_risk_model, quantized_predict_risk
    from models.simulate import animal_windows

    sigs, labs = make_acoustic_dataset(n_per_class=40, seed=0)
    a_tr, a_te, ya_tr, ya_te = train_test_split(
        sigs, labs, test_size=0.3, random_state=0, stratify=labs)
    acoustic = AcousticClassifier().fit(a_tr, ya_tr)
    a_acc = float(np.mean([p == y for p, y in zip(acoustic.predict(a_te), ya_te)]))

    case = next(a for a in test_animals if a.is_case)
    W = animal_windows(case, _WINDOW)
    det = PerAnimalAnomaly().fit(W[:12])                      # its own early normal
    late_flag = float(np.mean(det.is_anomalous(W[-12:])))     # pre/at-event windows

    qrep = quantize_risk_model(sentinel)
    q_mad = float(np.mean(np.abs(
        sentinel.predict_risk(Xte) - quantized_predict_risk(sentinel, Xte, qrep))))

    print(f"\n  Acoustic distress classifier: {a_acc*100:.0f}% test accuracy (4 classes)")
    print(f"  Per-animal anomaly: {late_flag*100:.0f}% of pre-event windows read "
          f"'unlike itself'")
    print(f"  Edge int8 quantise: {qrep['int8_bytes']/1024:.1f} KB weights "
          f"(<256 KB: {'yes' if qrep['within_budget'] else 'no'}), "
          f"float-vs-int8 diff {q_mad:.4f}")

    ART.mkdir(parents=True, exist_ok=True)
    card = sentinel.save(ART / "sentinel_risk.joblib")
    summary = {
        "window_sensitivity": sens, "window_specificity": spec,
        "classification_metrics": clf_metrics,
        "baseline": base_name,
        "lead_time_hours": {k: v.mean_lead for k, v in results.items()},
        "detection_rate": {k: v.detection_rate for k, v in results.items()},
        "verdict": verdict, "model_card": card,
        "acoustic_accuracy": a_acc,
        "anomaly_flag_rate": late_flag,
        "int8_bytes": qrep["int8_bytes"],
        "int8_within_budget": qrep["within_budget"],
    }
    (ART / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved model + card + benchmark to {ART}")
    return summary


if __name__ == "__main__":
    main()
