"""
End-to-end tests for UltraFabric-Vision.
Exercises the full production path: HTTP -> model load -> preprocess -> ensemble
-> calibrated threshold -> response, plus the embeddable InferenceEngine SDK.

Run:  python -m pytest tests/test_e2e.py -v -s
(These load real models and run real inference; allow a minute on CPU.)
"""
import os
import glob
import io

import cv2
import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD = sorted(glob.glob(os.path.join(BASE, 'data', 'test', 'good', '*')))
DEFECT = sorted(glob.glob(os.path.join(BASE, 'data', 'test', 'defect', '*')))

pytestmark = pytest.mark.skipif(
    not GOOD or not DEFECT or not os.path.exists(os.path.join(BASE, 'weights', 'patchcore_memory_bank.pkl')),
    reason="requires data/ and weights/ present",
)


def _jpg_bytes(path):
    img = cv2.imread(path)
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


# --------------------------------------------------------------------------
# Embeddable SDK
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine():
    from fabric_engine import InferenceEngine
    return InferenceEngine(capture_attention=False, warmup=True)


def test_engine_separates_good_from_defect(engine):
    good_scores = [engine.predict_bgr(cv2.imread(f)).score for f in GOOD[:8]]
    defect_scores = [engine.predict_bgr(cv2.imread(f)).score for f in DEFECT[:8]]
    print(f"\n  good  mean={np.mean(good_scores):.2f}  defect mean={np.mean(defect_scores):.2f}")
    # Defects must score clearly higher than good product.
    assert np.mean(defect_scores) > np.mean(good_scores) + 1.0


def test_engine_decisions_correct(engine):
    good_ok = sum(not engine.predict_bgr(cv2.imread(f)).is_defect for f in GOOD[:8])
    defect_hit = sum(engine.predict_bgr(cv2.imread(f)).is_defect for f in DEFECT[:8])
    print(f"\n  good correct={good_ok}/8  defect correct={defect_hit}/8")
    assert good_ok >= 7           # allow at most 1 false alarm on the sample
    assert defect_hit >= 7        # allow at most 1 miss on the sample


def test_engine_result_shape(engine):
    r = engine.predict_bgr(cv2.imread(GOOD[0]))
    assert r.heatmap.shape[:2] == (224, 224)
    assert isinstance(r.is_defect, bool)
    assert r.latency_ms > 0


# --------------------------------------------------------------------------
# HTTP API (full server path via TestClient — runs startup/model load)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import backend_api
    with TestClient(backend_api.app) as c:   # context manager triggers startup
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["engine"] == "loaded"


def test_version(client):
    j = client.get("/api/version").json()
    assert j["engine_loaded"] is True
    assert "threshold" in j and len(j["models"]) == 3


def test_upload_defect_flagged(client):
    r = client.post("/api/upload_image",
                    files={"file": ("d.jpg", _jpg_bytes(DEFECT[0]), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert set(["score", "is_anomalous", "latency_ms", "boxes"]).issubset(j)
    assert j["is_anomalous"] is True


def test_upload_good_passes(client):
    r = client.post("/api/upload_image",
                    files={"file": ("g.jpg", _jpg_bytes(GOOD[0]), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["is_anomalous"] is False


def test_upload_invalid_image_400(client):
    r = client.post("/api/upload_image",
                    files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400
