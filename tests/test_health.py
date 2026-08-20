"""Smoke tests for the foundation: app boots and health check responds."""


def test_health_check(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "version" in body


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "health" in response.json()
