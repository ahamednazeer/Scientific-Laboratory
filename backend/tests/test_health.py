from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_name" in payload
    assert "detector_ready" in payload
    assert "detector_detail" in payload
    assert "ai_detection_available" in payload
