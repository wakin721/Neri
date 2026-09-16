from pathlib import Path


def _source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_dinov2_registry_dto_has_no_dinov3_dependency():
    assert "dinov3" not in _source("frontend/lib/src/models/dinov2_registry.dart")


def test_dinov2_registry_examples_are_named_for_dinov2():
    assert "dinov3" not in _source("system/dinov2/registry_examples.py")


def test_active_backend_does_not_attach_legacy_dinov3_runtime_alias():
    source = _source("system/backend/services.py")
    assert "dinov3_runtime" not in source
