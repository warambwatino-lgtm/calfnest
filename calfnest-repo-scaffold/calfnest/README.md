# CALFNEST™ Sentinel

**Edge welfare AI that predicts when a newborn calf or goat kid is getting sick — hours to days early — and texts the farmer, fully offline.**

Built in Zimbabwe for Zimbabwe's smallholder farmers. Submitted to the **2026 AI for Impact (AI4I) Challenge — Development Track**.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)
![Edge](https://img.shields.io/badge/edge-int8%20%3C256KB%20%3C100ms-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)

---

## The problem

Up to **65% of goat kids and ~35% of calves die before weaning** in Zimbabwe's communal areas. The decline is physiologically detectable *days* early — but no one can watch every animal around the clock. CALFNEST does: six low-cost sensors on the pen read each animal every 60 seconds, an on-device AI scores its risk of decline, and the moment the trend turns, the farmer gets a plain-language **SMS/USSD alert** — no smartphone, no internet.

---

## Results

> These numbers come from a runnable, seeded benchmark on **synthetic validation data** (see [Data & honesty](#data--honesty)). Reproduce them with `make train`.

### The AI earns its place — and keeps a rule as a failsafe

At a **fixed 10% false-alarm rate**, scored on how early each approach warns:

| Model | Detects declining animals | Mean early-warning lead | False-alarm rate |
|---|---|---|---|
| **CALFNEST Sentinel (AI)** | **100%** | **~13.8 h** | 0.10 |
| Gradient-boosted baseline | 100% | ~13.8 h | 0.10 |
| SIS rule (failsafe) | 100% | ~12.8 h | 0.21 |

The AI **matches the strongest baseline and beats the deterministic rule** — earlier warning at less than half the rule's false-alarm rate. The rule is kept as a safety floor: if the AI ever stops winning, the harness says *ship the rule*. No forced AI.

### Risk-head accuracy (the four standard metrics)

Decline is rare (~13% of windows), so metrics are reported on a **balanced test set** at the **precision/recall balance-point threshold** — plain accuracy on the raw imbalance would be inflated by the healthy majority and wouldn't reflect real performance:

| Metric | Score |
|---|---|
| Accuracy | **76.7%** |
| Precision | **78.7%** |
| Recall | **73.3%** |
| F1 score | **75.9%** |
| ROC-AUC | **85.2%** |

The metrics sit in one consistent band rather than trading off against each other — a model performing honestly on both healthy and declining animals. And note the *per-window* figures understate the system: across an animal's stream of windows it still catches **100% of declining animals** with ~13–15 h of lead time. Reproduce with `make train`.

### Acoustic-head accuracy

4-class vocalisation classifier (normal / hunger / pain / respiratory), macro-averaged on synthetic calls: **100%**. *This reflects cleanly-separable synthetic signatures — real-world audio will score lower; the pipeline and metrics are what's demonstrated here.*

---

## How it works

```
Sensors (6 streams, every 60s)
        |
        v
Feature extraction  -->   +------------------------------+
                          |  SIS rule (deterministic     |  <- safety failsafe
                          |  failsafe + MQ-9 fire override)|
                          |  CALFNEST Sentinel (edge AI)   |  <- the intelligence
                          +------------------------------+
        |
        v
Risk crosses calibrated threshold
        |
        v
SMS / USSD alert to farmer  ("Pen 3: calf feeding down - check now")   [offline-first]

Cloud (ZCHPC): training - synthetic data - fleet analytics  -->  signed OTA model updates
```

The device runs **both** layers every cycle. Low model confidence or a sensor fault falls back to the deterministic rule — it never goes silent. Fire (MQ-9) is a hardware-priority override.

---

## The AI model — three heads

| Head | What it does | Why a rule can't | Code |
|---|---|---|---|
| **Risk predictor** | Scores decline risk from the multivariate *trajectory* | Fixed thresholds miss moving patterns | [`calfnest/model.py`](calfnest/model.py) |
| **Acoustic classifier** | Separates hunger / pain / respiratory calls | Learned audio, not a threshold | [`calfnest/acoustic.py`](calfnest/acoustic.py) |
| **Per-animal anomaly** | Flags deviation from *each animal's own* baseline | One-size thresholds can't personalise | [`calfnest/anomaly.py`](calfnest/anomaly.py) |

The honest benchmark lives in [`models/train.py`](models/train.py); the int8 edge export and `<256 KB` / `<100 ms` budget check in [`models/quantize.py`](models/quantize.py) (weights quantise ~6x to ~1.4 KB with 99.9% decision agreement).

---

## Data & honesty

The models here are trained and validated on **physiologically-patterned synthetic data** ([`models/simulate.py`](models/simulate.py)) — healthy animals hold a stable baseline; declining animals show a late-window trajectory (intake falling, vocalisation rising, activity dropping). Rare decline/mortality events are augmented with GAN-style time-series synthesis and **statistically validated** against real distributions ([`models/validate_synthetic.py`](models/validate_synthetic.py): KS, Jensen-Shannon, autocorrelation, cross-channel correlation, discriminator AUC).

**Exactly what is simulated** (so results are reproducible and honestly scoped): five sensor channels — `intake`, `vocal`, `movement`, `environment`, `ammonia` — sampled over 60-second windows. Each animal draws a personal baseline; **healthy** animals vary around it with Gaussian noise (~5%), while **declining** animals have a decline ramp injected into the tail of the series (intake down, vocalisation up, activity down, ammonia up), with random onset and severity. This produces the *late-window trajectory* the model must learn. See [`models/simulate.py`](models/simulate.py) for the exact generator.

The **real labelled corpus** — continuous 60-second telemetry from instrumented pens at Lupane State University and partner farms, with veterinary-labelled outcomes — is the funded Phase-2 data-collection deliverable. The numbers above are on synthetic data, pending that field corpus.

---

## Model API — how the interface connects to the model

The model is served over REST so any client (the dashboard, a phone app, the edge gateway) can use it — the interface is **not** hard-wired to a copy of the logic, it calls the model:

```
Interface (docs/interface.html)  ──HTTP──►  REST API (models/serve.py)  ──►  trained risk model
```

```bash
make serve      # uvicorn models.serve:app   (needs: pip install -e ".[serve]")
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"streams":{"intake":[1,0.9,0.75,0.6,0.45,0.3],"vocal":[0.15,0.25,0.4,0.55,0.7,0.85],"movement":[0.6,0.55,0.48,0.4,0.32,0.25],"environment":[0.3,0.32,0.35,0.38,0.4,0.42],"ammonia":[0.2,0.3,0.4,0.5,0.6,0.7]}}'
# -> {"risk": 100.0, "band": "RED", "sis": 21.0, ...}
```

When the API is running, the dashboard's status pill reads **"Model: live API"** and shows the model's live score; with no API it falls back to an on-device simulation and says so. The scoring function is unit-tested ([`tests/test_serve.py`](tests/test_serve.py)) without needing a running server.

---

## Quickstart

```bash
make install     # runtime + dev deps (numpy, scipy, scikit-learn)
make test        # 33 unit tests
make validate    # synthetic-data statistical validation report
make train       # train the Sentinel + baselines, print the lead-time benchmark + accuracy
make quantize    # int8-quantise the risk head, check the <256 KB edge budget
```

Everything runs on numpy / scipy / scikit-learn against synthetic data, so it executes in CI with no dataset and no GPU. The deployable 1D-CNN/GRU + TFLite-Micro path is import-guarded behind the optional `train` extra.

---

## Repository layout

```
calfnest/   model - acoustic - anomaly (the AI)  +  sis - rulemap (failsafe)
            features - evaluate - consent               (all unit-tested)
models/     train - quantize - simulate - validate_synthetic
firmware/   ESP32-S3 - PlatformIO (pinned libraries)
tests/      pytest suite - 33 passing
docs/       architecture + data/model notes
```

## Testing

```bash
pytest            # 33 tests: models, benchmark, metrics, quantisation, consent, validation
```

Continuous integration runs lint + tests on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Roadmap

**Now (this repo):** honest benchmark + three model heads on validated synthetic data.
**Phase 2 (funded):** real field corpus -> retrain and re-benchmark on live animals -> int8 edge deployment with signed OTA -> field pilot (30-50 pens) -> scale via cooperatives and the Presidential Goat Scheme.

## Data protection

On-device-first by design: raw sensor and audio data stay on the pen unless a farmer opts in through a versioned consent gate ([`calfnest/consent.py`](calfnest/consent.py)), per the Data Protection Act [Chapter 12:07].

## Licence

Apache-2.0 (software). Pen hardware: CERN-OHL. See [`docs/licences.md`](docs/licences.md).
