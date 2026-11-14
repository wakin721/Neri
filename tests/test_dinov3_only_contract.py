"""Contract checks for the DINOv3-only Neri1.1 runtime."""
from __future__ import annotations

from pathlib import Path


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").lower()


def test_dinov2_package_and_backend_modules_are_removed():
    assert not Path("system/dinov2").exists()
    assert not list(Path("system/backend").glob("dinov2_*.py"))
    assert not list(Path("frontend/lib").rglob("*dinov2*"))
    assert not list(Path("frontend/test").glob("*dinov2*"))


def test_active_python_and_flutter_sources_do_not_reference_dinov2():
    for root in (Path("system"), Path("frontend/lib"), Path("frontend/test")):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".dart"}:
                continue
            text = _source(path)
            assert "system.dinov2" not in text, path
            assert "dinov2_" not in text, path
            assert "/api/dinov2/" not in text, path


def test_environment_api_only_exposes_dinov3_component():
    from system.backend.main import app

    # Included routers may be lazy; inspect the public API contract rather
    # than assuming app.routes contains a flat list of APIRoute instances.
    paths = set(app.openapi()["paths"])
    assert "/api/environment/dinov3-status" in paths
    assert "/api/environment/install-dinov3" in paths
    assert "/api/environment/remove-dinov3" in paths
    assert "/api/dino/feedback/box" in paths
    assert not any("dinov2" in path for path in paths)


def test_component_installer_uses_dinov3_root_and_manifest():
    from system.dinov3.component import (
        DINO_COMPONENT_ARCHITECTURE,
        DINO_COMPONENT_DISPLAY_NAME,
    )

    assert DINO_COMPONENT_ARCHITECTURE == "DINOv3 ViT-B/16"
    assert DINO_COMPONENT_DISPLAY_NAME == "Neri1.1"
    backend = _source(Path("system/dinov3/component.py"))
    assert "read_manifest" in backend
    assert "checkpoint_sha256" in backend
    assert "encoder_sha256" in backend
