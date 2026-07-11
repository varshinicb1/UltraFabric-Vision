"""Assembly-line motor control bridge.

The conveyor motor is driven by an Arduino behind an ESP32 Wi-Fi board that
exposes a small HTTP API (see firmware/). This module is the single place the
backend talks to that board, used for two things:

  1. Stop-on-defect: when the live stream flags a defect, send STOP.
  2. Dashboard control: the /api/line/* endpoints proxy calibration, manual
     run/stop and auto-batch commands through here so the browser only ever
     talks to the backend (one origin, no CORS surprises).

Design goals: never block frame handling, never crash inference, and always
give the dashboard a clear connected/not-connected answer.

  * The board URL is set from the dashboard (persisted to line_config.json) and
    falls back to the env var ``UFV_ESP32_URL``.
  * All calls are short-timeout and failures degrade to ``{"connected": false}``.

ESP32 endpoints used: /status /start /stop /defect /auto /config /jog /cal
"""
import os
import json
import time
import threading
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(_HERE, "line_config.json")

_TIMEOUT_S = 2.5
STOP_COOLDOWN_S = float(os.environ.get("UFV_ESP32_COOLDOWN", "5"))

_lock = threading.Lock()
_last_stop = 0.0


def _load_url() -> str:
    try:
        with open(_CFG_PATH) as f:
            return (json.load(f).get("esp32_url") or "").rstrip("/")
    except Exception:
        return os.environ.get("UFV_ESP32_URL", "").rstrip("/")


_url = _load_url()


def get_url() -> str:
    return _url


def set_url(url: str) -> str:
    """Persist the ESP32 base URL (e.g. http://192.168.1.50). Returns normalized."""
    global _url
    url = (url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    _url = url
    try:
        with open(_CFG_PATH, "w") as f:
            json.dump({"esp32_url": _url}, f)
    except Exception:
        pass
    return _url


def enabled() -> bool:
    return bool(_url)


def request(path: str, params: dict = None, timeout: float = _TIMEOUT_S):
    """GET <url>/<path>[?params]. Returns (ok, text). Never raises."""
    if not _url:
        return False, "no board configured"
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    full = f"{_url}/{path.lstrip('/')}{q}"
    try:
        with urllib.request.urlopen(full, timeout=timeout) as r:
            return True, r.read().decode("utf-8", "replace")
    except Exception as e:
        return False, str(e)


def _send_async(path: str, params: dict = None):
    threading.Thread(target=request, args=(path, params), daemon=True).start()


def status() -> dict:
    """Current line status for the dashboard. Always returns a dict."""
    if not _url:
        return {"configured": False, "connected": False, "url": ""}
    ok, body = request("status")
    if not ok:
        return {"configured": True, "connected": False, "url": _url, "error": body}
    try:
        data = json.loads(body)
    except Exception:
        data = {}
    data.update({"configured": True, "connected": True, "url": _url})
    return data


# ---- command helpers used by the /api/line/* proxy endpoints ----
def start():
    return request("start")


def stop():
    return request("stop")


def resume():
    return request("start")


def jog(revs: float):
    """Move an exact number of motor revolutions for calibration measurement."""
    return request("jog", {"revs": revs})


def calibrate(measured_m: float, revs: float):
    """Given the belt travel (metres) measured over ``revs`` revolutions, compute
    and store mm-per-revolution on the Arduino. Returns (ok, mm_per_rev)."""
    revs = float(revs) or 1.0
    mm_per_rev = float(measured_m) * 1000.0 / revs
    ok, _ = request("cal", {"mmrev": round(mm_per_rev, 3)})
    return ok, round(mm_per_rev, 3)


def auto(cloth_m: float, line_m: float, speed_m_min: float):
    """Configure and start an automatic batch run (units: metres, m/min)."""
    ok, _ = request("config", {"cloth": cloth_m, "line": line_m, "speed": speed_m_min})
    if not ok:
        return False, "config failed"
    return request("auto")


# ---- stop-on-defect (called from the live /ws/stream loop) ----
def notify(is_anomalous: bool):
    """Send STOP to the conveyor on a defect, subject to a cooldown. No-op when
    no board is configured. Non-blocking and exception-safe."""
    if not is_anomalous or not _url:
        return
    global _last_stop
    now = time.time()
    with _lock:
        if now - _last_stop < STOP_COOLDOWN_S:
            return
        _last_stop = now
    _send_async("defect")
