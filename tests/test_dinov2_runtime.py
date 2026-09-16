from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from system.dinov2.prototype_bank import PrototypeBank
from system.dinov2.runtime import (
    DinoV2ManifestError,
    load_dinov2_model,
    resolve_dinov2_manifest,
)
from tests.dinov2_fixtures import make_dinov2_payload


class FakeStore:
    def __init__(self, fingerprint: str, path: Path):
        self.model_fingerprint = fingerprint
        self.path = path
        self.closed = False

    def prototype_bank(self, center):
        return PrototypeBank(())

    def close(self):
        self.closed = True


class FakeEncoder:
    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs


def _fixture(tmp_path: Path, **manifest_overrides):
    payload = make_dinov2_payload()
    checkpoint = tmp_path / "classifier.pt"
    torch.save(payload, checkpoint)
    manifest = {
        "schema_version": 2,
        "backend": "dinov2",
        "display_name": "DINOv2 test",
        "checkpoint": checkpoint.name,
        "architecture": "dinov2_vitb14",
        "feature_dim": 768,
        "encoder_sha256": payload["encoder_sha256"],
        "preprocessing": payload["preprocessing"],
        "event_aggregation": payload["event_aggregation"],
        "requires_detector": True,
        "supports_full_image": False,
        "supports_video_fast": True,
        "supports_video_all": False,
    }
    manifest.update(manifest_overrides)
    path = tmp_path / "classifier.neri.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, checkpoint


def test_runtime_loads_dinov2_manifest_with_injected_state(tmp_path):
    manifest, _ = _fixture(tmp_path)
    from system.dinov2.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(tmp_path / "classifier.pt")
    registry = FakeStore(checkpoint.fingerprint, tmp_path / "registry.sqlite3")
    feedback = FakeStore(checkpoint.fingerprint, tmp_path / "feedback.sqlite3")
    runtime = load_dinov2_model(
        manifest,
        registry=registry,
        feedback=feedback,
        encoder_factory=FakeEncoder,
        device="cpu",
    )
    try:
        assert runtime.manifest_path == manifest.resolve()
        assert runtime.checkpoint.backbone == "facebook/dinov2-base"
        assert runtime.classifier.backend == "dinov2"
        assert runtime.classifier.encoder.kwargs["device"] == "cpu"
        assert runtime.owns_registry is False
        assert runtime.owns_feedback is False
    finally:
        runtime.close()
    assert registry.closed is False
    assert feedback.closed is False


def test_runtime_rejects_dinov3_manifest_explicitly(tmp_path):
    manifest, _ = _fixture(tmp_path, backend="dinov3")
    with pytest.raises(DinoV2ManifestError, match="DINOv3.*不再支持"):
        resolve_dinov2_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("schema_version", 1, "schema_version"),
        ("architecture", "dinov3_vitb16", "architecture"),
        ("feature_dim", 1024, "feature dimension"),
        ("encoder_sha256", "0" * 64, "encoder_sha256"),
        ("preprocessing", "other", "preprocessing"),
        ("event_aggregation", "other", "event_aggregation"),
    ],
)
def test_manifest_contract_must_match_checkpoint(tmp_path, field, value, pattern):
    manifest, _ = _fixture(tmp_path, **{field: value})
    from system.dinov2.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(tmp_path / "classifier.pt")
    registry = FakeStore(checkpoint.fingerprint, tmp_path / "registry.sqlite3")
    feedback = FakeStore(checkpoint.fingerprint, tmp_path / "feedback.sqlite3")
    with pytest.raises(DinoV2ManifestError, match=pattern):
        load_dinov2_model(
            manifest,
            registry=registry,
            feedback=feedback,
            encoder_factory=FakeEncoder,
        )


def test_checkpoint_path_resolves_its_manifest(tmp_path):
    manifest, checkpoint = _fixture(tmp_path)
    assert resolve_dinov2_manifest(checkpoint) == manifest.resolve()


def test_injected_state_fingerprint_must_match_checkpoint(tmp_path):
    manifest, _ = _fixture(tmp_path)
    registry = FakeStore("f" * 64, tmp_path / "registry.sqlite3")
    feedback = FakeStore("f" * 64, tmp_path / "feedback.sqlite3")
    with pytest.raises(ValueError, match="Registry model fingerprint"):
        load_dinov2_model(
            manifest,
            registry=registry,
            feedback=feedback,
            encoder_factory=FakeEncoder,
        )
