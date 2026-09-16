from __future__ import annotations


def test_environment_router_exposes_only_dinov2_paths():
    from system.backend.dinov2_environment import dinov2_environment_router

    router = dinov2_environment_router()
    paths = {route.path for route in router.routes}

    assert "/api/environment/dinov2-status" in paths
    assert "/api/environment/install-dinov2" in paths
    assert "/api/environment/remove-dinov2" in paths
    assert all("dinov3" not in path.lower() for path in paths)


def test_component_status_payload_uses_dinov2_fields(monkeypatch):
    import system.backend.dinov2_environment as environment

    monkeypatch.setattr(
        environment,
        "dinov2_component_status",
        lambda: {
            "installed": True,
            "healthy": True,
            "architecture": "DINOv2 ViT-B/14",
            "component_version": 1,
            "classifier_filename": "classifier.pt",
            "classifier_fingerprint": "a" * 64,
            "classifier_head_type": "multi_prototype",
            "prototype_count": 75,
            "message": "DINOv2 ViT-B/14 已安装。",
        },
    )

    payload = environment._component_status_payload()

    assert payload["architecture"] == "DINOv2 ViT-B/14"
    assert payload["prototype_count"] == 75
    assert "source_commit" not in payload
