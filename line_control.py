"""Assembly-line motor control bridge.

The conveyor motor is driven by an Arduino behind an ESP32 Wi-Fi board that
exposes a small HTTP API (see firmware/). This module is the single place the
backend talks to that board. It handles:

  1. Stop-on-BIG-defect: when the live stream flags a defect whose area exceeds
     a configurable fraction of the frame, send STOP. Small specks are ignored
     so the line does not halt on cosmetic noise.
  2. Auto-resume: optionally restart the belt N seconds after such a stop.
  3. Dashboard control: the /api/line/* endpoints proxy calibration, manual
     run/stop and auto-batch commands through here (browser -> backend -> board).

Design goals: never block frame handling, never crash inference, always give the
dashboard a clear connected/not-connected answer. Settings persist to
line_config.json and fall back to environment variables.

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

# ---- persisted settings ----
_DEFAULTS = {
    "esp32_url": os.environ.get("UFV_ESP32_URL", "").rstrip("/"),
    # A defect must cover at least this fraction of the frame to stop the belt.
    "stop_min_area": float(os.environ.get("UFV_STOP_MIN_AREA", "0.03")),   # 3%
    # Seconds to wait before auto-resuming after a defect stop (0 = manual only).
    "auto_resume_s": float(os.environ.get("UFV_AUTO_RESUME", "0")),
    # Automatically record + inspect each batch as the belt advances.
    "auto_record": False,
}
_cfg = dict(_DEFAULTS)


def _load():
    try:
        with open(_CFG_PATH) as f:
            data = json.load(f)
        for k in _cfg:
            if k in data and data[k] is not None:
                _cfg[k] = data[k]
        _cfg["esp32_url"] = (_cfg.get("esp32_url") or "").rstrip("/")
    except Exception:
        pass


def _save():
    try:
        with open(_CFG_PATH, "w") as f:
            json.dump(_cfg, f)
    except Exception:
        pass


_load()


# ---- URL ----
def get_url() -> str:
    return _cfg["esp32_url"]


def set_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    _cfg["esp32_url"] = url
    _save()
    return url


def enabled() -> bool:
    return bool(_cfg["esp32_url"])


# ---- settings ----
def get_settings() -> dict:
    return {
        "stop_min_area": _cfg["stop_min_area"],
        "auto_resume_s": _cfg["auto_resume_s"],
        "auto_record": _cfg["auto_record"],
    }


def set_settings(stop_min_area=None, auto_resume_s=None, auto_record=None) -> dict:
    if stop_min_area is not None:
        _cfg["stop_min_area"] = max(0.0, min(1.0, float(stop_min_area)))
    if auto_resume_s is not None:
        _cfg["auto_resume_s"] = max(0.0, float(auto_resume_s))
    if auto_record is not None:
        _cfg["auto_record"] = bool(auto_record)
    _save()
    return get_settings()


# ---- HTTP ----
def request(path: str, params: dict = None, timeout: float = _TIMEOUT_S):
    """GET <url>/<path>[?params]. Returns (ok, text). Never raises."""
    url = _cfg["esp32_url"]
    if not url:
        return False, "no board configured"
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    full = f"{url}/{path.lstrip('/')}{q}"
    try:
        with urllib.request.urlopen(full, timeout=timeout) as r:
            return True, r.read().decode("utf-8", "replace")
    except Exception as e:
        return False, str(e)


def _send_async(path: str, params: dict = None):
    threading.Thread(target=request, args=(path, params), daemon=True).start()


def status() -> dict:
    """Current line status + settings for the dashboard. Always returns a dict."""
    base = {"configured": bool(_cfg["esp32_url"]), "url": _cfg["esp32_url"]}
    base.update(get_settings())
    if not _cfg["esp32_url"]:
        base["connected"] = False
        return base
    ok, body = request("status")
    if not ok:
        base.update({"connected": False, "error": body})
        return base
    try:
        data = json.loads(body)
    except Exception:
        data = {}
    data.update(base)
    data["connected"] = True
    return data


# ---- command helpers used by the /api/line/* proxy endpoints ----
def start():
    return request("start")


def stop():
    return request("stop")


def resume():
    return request("start")


def jog(revs: float):
    return request("jog", {"revs": revs})


def calibrate(measured_m: float, revs: float):
    revs = float(revs) or 1.0
    mm_per_rev = float(measured_m) * 1000.0 / revs
    ok, _ = request("cal", {"mmrev": round(mm_per_rev, 3)})
    return ok, round(mm_per_rev, 3)


def auto(cloth_m: float, line_m: float, speed_m_min: float):
    ok, _ = request("config", {"cloth": cloth_m, "line": line_m, "speed": speed_m_min})
    if not ok:
        return False, "config failed"
    return request("auto")


# ---- stop-on-BIG-defect (called from the live /ws/stream loop) ----
def notify(is_anomalous: bool, defect_area_frac: float = 0.0):
    """Stop the conveyor only for a *significant* defect.

    ``defect_area_frac`` is the fraction of the frame covered by defect boxes.
    The belt stops only when it meets/exceeds ``stop_min_area`` (so small specks
    are ignored). Subject to a cooldown; optionally auto-resumes after
    ``auto_resume_s`` seconds. Non-blocking and exception-safe; a no-op when no
    board is configured.
    """
    if not is_anomalous or not _cfg["esp32_url"]:
        return
    if float(defect_area_frac) < _cfg["stop_min_area"]:
        return  # small defect -> keep running
    global _last_stop
    now = time.time()
    with _lock:
        if now - _last_stop < STOP_COOLDOWN_S:
            return
        _last_stop = now
    _send_async("defect")
    resume_s = _cfg["auto_resume_s"]
    if resume_s and resume_s > 0:
        threading.Timer(resume_s, lambda: _send_async("start")).start()
