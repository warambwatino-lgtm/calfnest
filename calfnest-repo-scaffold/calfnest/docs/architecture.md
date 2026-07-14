# Architecture

Edge (ESP32-S3) → offline-first sync (MQTT/TLS over LoRaWAN/GSM) → farmer
SMS/USSD + cooperative dashboard → ZCHPC Cloud Compute Environment (training,
synthetic data, fleet analytics) → signed OTA model updates back to the edge.

The edge node runs, every 60 s: sensor read → feature extraction
(`calfnest/features.py`) → **both** the deterministic SIS (`calfnest/sis.py`)
and the Sentinel TinyML model → local traffic-light + buzzer. Inference is
int8, <120 KB RAM, <100 ms. If model confidence is low or a sensor-fault flag
is raised, the device falls back to the SIS — it never goes silent. The MQ-9
fire signal is a hardware-priority override.
