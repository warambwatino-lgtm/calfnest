# CALFNEST edge firmware (ESP32-S3)

Drop the dissertation firmware (Appendix D) into `src/` as `main.cpp`. It already
implements sensor polling, the rule-based SIS, RGB/buzzer alerts, SD logging and
MQTT. The Challenge work adds the TFLite-Micro Sentinel inference call and a
signed-OTA update path. Build with `pio run`; flash with `pio run -t upload`.
