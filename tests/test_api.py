
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_missing_fields():
    response = client.post(
        "/predict",
        json={"customer": {}}
    )

    assert response.status_code == 400


def test_invalid_endpoint():
    response = client.get("/invalid")

    assert response.status_code == 404
