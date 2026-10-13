"""API tests. Run `pytest` from the project root."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import Predictor

HEALTHY = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}

HEAT_DISSIPATION = {
    "type": "L",
    "air_temperature_k": 302.3,
    "process_temperature_k": 310.1,
    "rotational_speed_rpm": 1353,
    "torque_nm": 50.9,
    "tool_wear_min": 36,
}

POWER_FAILURE = {
    "type": "L",
    "air_temperature_k": 298.9,
    "process_temperature_k": 309.1,
    "rotational_speed_rpm": 2861,
    "torque_nm": 4.6,
    "tool_wear_min": 143,
}

OVERSTRAIN = {
    "type": "L",
    "air_temperature_k": 298.0,
    "process_temperature_k": 308.3,
    "rotational_speed_rpm": 1362,
    "torque_nm": 56.8,
    "tool_wear_min": 216,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_every_class(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 6


def test_console_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Abhishek Grover" in response.text


def test_prediction_payload_is_consistent(client):
    body = client.post("/predict", json=HEALTHY).json()
    assert set(body) == {"prediction", "confidence_score", "all_probabilities"}
    assert body["prediction"] in body["all_probabilities"]
    assert body["confidence_score"] == max(body["all_probabilities"].values())
    assert sum(body["all_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (HEALTHY, "No Failure"),
        (HEAT_DISSIPATION, "Heat Dissipation Failure"),
        (POWER_FAILURE, "Power Failure"),
        (OVERSTRAIN, "Overstrain Failure"),
    ],
    ids=["healthy", "heat-dissipation", "power", "overstrain"],
)
def test_known_machine_states(client, payload, expected):
    assert client.post("/predict", json=payload).json()["prediction"] == expected


@pytest.mark.parametrize("machine_type", ["L", "M", "H", "m", " h "])
def test_machine_type_is_normalised(client, machine_type):
    response = client.post("/predict", json={**HEALTHY, "type": machine_type})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"type": "X"},
        {"air_temperature_k": 25},  # degrees Celsius sent instead of kelvin
        {"process_temperature_k": 1000},
        {"rotational_speed_rpm": 0},
        {"torque_nm": -5},
        {"tool_wear_min": -1},
        {"tool_wear_min": "worn"},
    ],
)
def test_out_of_range_input_is_rejected(client, change):
    response = client.post("/predict", json={**HEALTHY, **change})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {key: value for key, value in HEALTHY.items() if key != "torque_nm"}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_artefacts_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.load(tmp_path)
