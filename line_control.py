"""Assembly-line motor control bridge.

When the inference backend flags a defect on the live stream, the conveyor motor
must stop. The motor is driven by an Arduino behind an ESP32 Wi-Fi board that
exposes a tiny HTTP API (see firmware/). This module sends the stop signal to
that ESP32.

It is intentionally best-effort and non-blocking:
  * disabled unless the ESP32 base URL is configured (env ``UFV_ESP32_URL``),
    e.g. ``UFV_ESP32_URL=http://192.168.1.50``;
  * the HTTP call runs on a daemon thread so websocket frame handling is never
    blocked by line latency or a missing board;
  * a cooldown prevents every defective frame from re-sending the stop.

Endpoints used (provided by conveyor_esp32.ino):
  GET <base>/defect   -> tells the Arduino to halt (reason = defect)
  GET <base>/start    -> resume
"""
import os
import time
import threading
import urllib.request

# Base URL of the ESP32 conveyor controller, e.g. "http://192.168.1.50".
ESP32_URL = os.environ.get("UFV_ESP32_URL", "").rstrip("/")
# Minimum seconds between two stop signals (a defect usually spans many frames).
STOP_COOLDOWN_S = float(os.environ.get("UFV_ESP32_COOLDOWN", "5"))
# HTTP timeout so a slow/absent board cannot stall the thread for long.
_TIMEOUT_S = 2.0

_last_stop = 0.0
_lock = threading.Lock()


def enabled() -> bool:
    """True if a conveyor controller URL is configured."""
    return bool(ESP32_URL)


def _fire(path: str):
    url = f"{ESP32_URL}/{path.lstrip('/')}"
    try:
        urllib.request.urlopen(url, timeout=_TIMEOUT_S).read()
    except Exception:
        # Best-effort: a missing board or network hiccup must not crash inference.
        pass


def _send_async(path: str):
    threading.Thread(target=_fire, args=(path,), daemon=True).start()


def notify(is_anomalous: bool):
    """Call once per processed live frame with its defect verdict.

    On a defect (subject to the cooldown) this sends STOP to the conveyor. No-op
    when the controller URL is not configured.
    """
    if not is_anomalous or not ESP32_URL:
        return
    global _last_stop
    now = time.time()
    with _lock:
        if now - _last_stop < STOP_COOLDOWN_S:
            return
        _last_stop = now
    _send_async("defect")


def resume():
    """Manually resume the conveyor (e.g. after an operator clears a defect)."""
    if ESP32_URL:
        _send_async("start")
