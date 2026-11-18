from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

pytest.importorskip("fastapi")

from pathsnap.api import app
from pathsnap.features.candidates import CandidateConfig, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def api_client():
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            yield client


@pytest.fixture
def configured_api(tmp_path, monkeypatch):
    def write_trip(partition, trip_id):
        trip = tmp_path / partition / trip_id
        trip.mkdir(parents=True)
        (trip / f"{trip_id}.track").write_text(
            "8.0001 49.0 0\n8.0002 49.0 1\n8.0003 49.0 20\n8.0004 49.0 21\n",
        )
        (trip / f"{trip_id}.nodes").write_text("8.0 49.0\n8.001 49.0\n")
        (trip / f"{trip_id}.arcs").write_text("0 1\n1 0\n")
        reference = trip / f"{trip_id}.route"
        reference.write_text("0\n")
        return reference

    reference = write_trip("student_test", "00000009")
    write_trip("evaluator_test", "00000042")
    artifact = tmp_path / "model.joblib"
    features = pd.DataFrame(0, index=[0, 1], columns=FEATURE_COLUMNS)
    estimator = DummyClassifier(strategy="uniform").fit(features, [0, 1])
    joblib.dump({
        "estimator": estimator,
        "feature_columns": FEATURE_COLUMNS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_config": asdict(CandidateConfig()),
    }, artifact)
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_MODEL", str(artifact))
    monkeypatch.delenv("PATHSNAP_TRIP_PARTITION", raising=False)
    return reference


async def test_prediction_contract_and_reference_isolation(configured_api):
    async with api_client() as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.get("/trips")).json() == {"trips": ["00000009"]}
        payload = {"trip_id": "00000009"}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert set(result) == {"predicted_route", "metrics", "timings_ms"}
        assert result["predicted_route"] == {
            "type": "MultiLineString",
            "coordinates": [[[8.0, 49.0], [8.001, 49.0]]] * 2,
        }
        assert result["metrics"] == {"accuracy": 1, "matched_points": 4, "total_points": 4}
        assert 0 <= result["timings_ms"]["inference"] <= result["timings_ms"]["total"]

        configured_api.write_text("1\n")
        rescored = (await client.post("/predict", json=payload)).json()
        assert rescored["predicted_route"] == result["predicted_route"]
        assert rescored["metrics"] == {"accuracy": 0, "matched_points": 0, "total_points": 4}


async def test_only_active_partition_can_be_requested(configured_api, monkeypatch):
    monkeypatch.setenv("PATHSNAP_TRIP_PARTITION", "evaluator_test")
    async with api_client() as client:
        assert (await client.get("/trips")).json() == {"trips": ["00000042"]}
        response = await client.post("/predict", json={"trip_id": "00000009"})
        assert response.status_code == 404
        for invalid in [42, "42", "../00042", "0000004a"]:
            response = await client.post("/predict", json={"trip_id": invalid})
            assert response.status_code == 422


async def test_missing_model_makes_service_unavailable(configured_api, monkeypatch):
    monkeypatch.setenv("PATHSNAP_MODEL", str(configured_api.with_name("missing.joblib")))
    async with api_client() as client:
        assert (await client.get("/health")).status_code == 503
        assert (await client.get("/trips")).status_code == 503
        response = await client.post("/predict", json={"trip_id": "00000009"})
        assert response.status_code == 503
