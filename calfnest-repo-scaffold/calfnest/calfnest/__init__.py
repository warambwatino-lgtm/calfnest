"""CALFNEST Sentinel — edge welfare intelligence for neonatal livestock.

Rule-based failsafe (not AI):
    sis       — deterministic Stress Index Score
    rulemap   — maps a sensor window onto the SIS sub-scores

The AI (CALFNEST Sentinel), three heads:
    model     — neonatal risk predictor over the welfare trajectory
    acoustic  — vocalisation distress classifier (log-mel front end)
    anomaly   — per-animal anomaly detector (PCA reconstruction)

Supporting:
    features  — engineered temporal features from the six welfare streams
    evaluate  — lead-time-at-fixed-false-alarm-rate benchmark
    consent   — Data Protection Act [Chapter 12:07] consent gate

Training, synthetic data, int8 quantisation and statistical validation live in
the top-level ``models/`` package (models.train, models.quantize,
models.simulate, models.validate_synthetic).
"""
__version__ = "0.1.0"
