"""Predictor tests run everywhere. API tests need fastapi + httpx (installed in CI)."""
import pytest

from src import train
from src.ingest import make_synthetic

SAMPLE = {
    "pm2_5_mean": 135.0, "pm10_mean": 240.0, "o3_max": 55.0, "no2_mean": 62.0,
    "temp_mean": 14.0, "temp_min": 8.5, "temp_max": 22.0, "humidity_mean": 72.0,
    "wind_mean": 4.0, "precip_sum": 0.0,
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    data = d / "daily.csv"
    make_synthetic(500).to_csv(data, index=False)
    train.run(data_path=data, model_dir=d / "champ", reports_dir=d / "rep")
    return d / "champ"


def test_predictor_returns_valid_output(model_dir):
    from src.predictor import Predictor
    out = Predictor(model_dir).predict({**SAMPLE, "date": "2025-12-10"})
    assert out["risk"] in {"Low", "Moderate", "High"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert "not a medical diagnosis" in out["disclaimer"]


def test_predictor_clean_air_is_lower_risk_than_smog(model_dir):
    from src.predictor import Predictor
    p = Predictor(model_dir)
    clean = {**SAMPLE, "pm2_5_mean": 12, "pm10_mean": 25, "no2_mean": 15, "temp_min": 22, "temp_mean": 28,
             "temp_max": 34, "date": "2025-07-10"}
    smog = {**SAMPLE, "pm2_5_mean": 220, "pm10_mean": 380, "date": "2025-12-10"}
    assert p.predict(clean)["risk_level"] <= p.predict(smog)["risk_level"]
    assert p.predict(smog)["risk"] == "High"


def test_api_endpoints(model_dir, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src import api

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    api.get_predictor.cache_clear()
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/model-info").status_code == 200
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200 and r.json()["risk"] in {"Low", "Moderate", "High"}
    bad = client.post("/predict", json={**SAMPLE, "humidity_mean": 250})
    assert bad.status_code == 422
    api.get_predictor.cache_clear()


def test_api_without_model_returns_503(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src import api

    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "nothing"))
    api.get_predictor.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json=SAMPLE).status_code == 503
    api.get_predictor.cache_clear()
