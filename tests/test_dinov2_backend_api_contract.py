from __future__ import annotations


def test_model_info_accepts_dinov2_backend():
    from system.backend.models import ModelInfo

    model = ModelInfo(name="v2", path="x", kind="cls", backend="dinov2")
    assert model.backend == "dinov2"


def test_dinov2_registry_router_has_no_dinov3_paths():
    from system.dinov2.api import dinov2_registry_router

    paths = [route.path for route in dinov2_registry_router().routes]
    assert paths
    assert all("/api/dinov2/" in path for path in paths)
    assert all("dinov3" not in path.lower() for path in paths)


def test_dinov2_feedback_router_has_no_dinov3_paths():
    from system.backend.dinov2_feedback_api import dinov2_feedback_router

    paths = [route.path for route in dinov2_feedback_router().routes]
    assert paths
    assert all("/api/dinov2/" in path for path in paths)
    assert all("dinov3" not in path.lower() for path in paths)
