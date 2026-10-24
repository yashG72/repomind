from fastapi.testclient import TestClient

from repomind import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "RepoMind",
        "version": __version__,
        "environment": "test",
    }


def test_response_has_generated_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers["x-request-id"]) == 32


def test_incoming_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/does-not-exist").status_code == 404
