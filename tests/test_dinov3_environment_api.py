from __future__ import annotations

from fastapi.testclient import TestClient

from system.backend import dinov3_environment
from system.backend import main as backend_main


def test_dinov3_environment_status_endpoint(monkeypatch):
    monkeypatch.setattr(
        dinov3_environment,
        "dinov3_component_status",
        lambda: {
            "installed": False,
            "healthy": False,
            "architecture": "DINOv3 ViT-B/16",
            "component_version": 1,
            "source_commit": "6876159a11b4df116f30f667f8c9888617df0751",
            "message": "DINOv3 未安装。",
        },
    )
    with TestClient(backend_main.app) as client:
        response = client.get("/api/environment/dinov3-status")
    assert response.status_code == 200
    assert response.json()["healthy"] is False


def test_dinov3_environment_install_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dinov3_environment,
        "start_dinov3_install",
        lambda env_choice, package_source: calls.append((env_choice, package_source))
        or {
            "operation": "install_dinov3",
            "message": "DINOv3 安装已开始。",
            "progress": 0,
        },
    )
    monkeypatch.setattr(dinov3_environment, "schedule_backend_shutdown", lambda *args: None)
    with TestClient(backend_main.app) as client:
        response = client.post(
            "/api/environment/install-dinov3",
            json={"env_choice": "CPU Only", "package_source": "nju"},
        )
    assert response.status_code == 202
    assert response.json()["operation"] == "install_dinov3"
    assert calls == [("CPU Only", "nju")]


def test_dinov3_environment_remove_endpoint(monkeypatch):
    monkeypatch.setattr(
        dinov3_environment,
        "start_dinov3_remove",
        lambda: {
            "operation": "remove_dinov3",
            "message": "DINOv3 删除已开始。",
            "progress": 0,
        },
    )
    monkeypatch.setattr(dinov3_environment, "schedule_backend_shutdown", lambda *args: None)
    with TestClient(backend_main.app) as client:
        response = client.post("/api/environment/remove-dinov3")
    assert response.status_code == 202
    assert response.json()["operation"] == "remove_dinov3"
