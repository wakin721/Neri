from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.backend import model_services, services
from system.dinov3.encoder import PREPROCESSING, sha256
from system.model_sync.catalog import discover_models, resolve_saved_model_path
from system.model_sync.layout import get_model_layout


def _make_fake_dinov3_assets(target_dir: Path):
    checkpoint_file = target_dir / "checkpoint.pt"
    checkpoint_file.write_bytes(b"dummy_checkpoint_data_for_test")
    checkpoint_sha = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()

    weights_file = target_dir / "encoder_weights.pth"
    weights_file.write_bytes(b"dummy_encoder_weights_for_test")
    weights_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()

    code_dir = target_dir / "encoder_code"
    (code_dir / "dinov3" / "hub").mkdir(parents=True, exist_ok=True)
    (code_dir / "dinov3" / "hub" / "backbones.py").write_text("# mock", encoding="utf-8")

    manifest_payload = {
        "schema_version": 1,
        "backend": "dinov3",
        "display_name": "Neri1.1",
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha,
        "encoder_weights": "encoder_weights.pth",
        "encoder_sha256": weights_sha,
        "encoder_code": "encoder_code",
        "architecture": "dinov3_vitb16",
        "head_type": "ridge_within",
        "feature_dim": 768,
        "preprocessing": PREPROCESSING,
        "event_aggregation": "complete_sequence",
        "requires_detector": True,
        "supports_video_fast": False,
        "supports_video_all": False,
    }
    manifest_file = target_dir / "joint.neri.json"
    manifest_file.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return manifest_file, checkpoint_file, weights_file, manifest_payload


def test_dinov3_catalog_discovery_and_hiding_referenced_weights(tmp_path):
    layout = get_model_layout(tmp_path)
    manifest, checkpoint, weights, _ = _make_fake_dinov3_assets(layout.cls_user)

    models = discover_models(layout, "cls")
    assert len(models) == 1
    model = models[0]
    assert model.name == "Neri1.1"
    assert model.backend == "dinov3"
    assert model.architecture == "dinov3_vitb16"
    assert model.feature_dim == 768
    assert model.requires_detector is True
    assert model.supports_video_fast is True
    assert model.supports_video_all is False
    assert model.checkpoint_path == str(checkpoint.resolve())

    assert resolve_saved_model_path(str(checkpoint.resolve()), models) == str(
        manifest.resolve()
    )
    # Ensure neither checkpoint nor encoder_weights falls through as YOLO
    pt_models = [m for m in models if m.backend == "yolo"]
    assert pt_models == []


def test_old_dinov3_manifest_is_hidden_from_catalog(tmp_path):
    layout = get_model_layout(tmp_path)
    old_pt = layout.cls_user / "old.pt"
    old_pt.write_bytes(b"legacy")
    old_manifest = layout.cls_user / "old.neri.json"
    old_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov3",
                "checkpoint": "old.pt",
            }
        ),
        encoding="utf-8",
    )
    models = discover_models(layout, "cls")
    assert models == []


def test_model_services_exposes_fast_video_for_old_and_new_dinov3_manifests(
    tmp_path, monkeypatch
):
    layout = get_model_layout(tmp_path)
    old_manifest, _, _, old_payload = _make_fake_dinov3_assets(layout.cls_user)
    assert old_payload["supports_video_fast"] is False

    new_dir = layout.cls_sync
    new_manifest, _, _, new_payload = _make_fake_dinov3_assets(new_dir)
    new_payload = dict(new_payload, display_name="Neri1.2", supports_video_fast=True)
    new_manifest.write_text(json.dumps(new_payload), encoding="utf-8")

    monkeypatch.setattr(model_services, "get_model_layout", lambda: layout)
    models = model_services.list_available_classification_models()
    by_name = {model.name: model for model in models}
    assert set(by_name) == {"Neri1.1", "Neri1.2"}
    for model in by_name.values():
        assert model.supports_video_fast is True
        assert model.supports_video_all is False


def test_model_services_list_classes_without_loading_encoder(tmp_path, monkeypatch):
    manifest, checkpoint, _, _ = _make_fake_dinov3_assets(tmp_path)

    class MockDinoV3Model:
        classes = ["species_alpha", "species_beta", "species_gamma"]

        @classmethod
        def load(cls, path):
            assert Path(path).resolve() == checkpoint.resolve()
            return cls()

    import system.dinov3.model as dinov3_model_mod

    monkeypatch.setattr(dinov3_model_mod, "DinoV3RidgeWithinModel", MockDinoV3Model)

    classes = model_services.list_model_classes(str(manifest))
    assert [c.name for c in classes] == ["species_alpha", "species_beta", "species_gamma"]
    assert [c.id for c in classes] == [0, 1, 2]


def test_model_services_rejects_old_dinov3_and_reports_invalid_joint(tmp_path):
    old_manifest = tmp_path / "old.neri.json"
    old_manifest.write_text(
        json.dumps({"backend": "dinov3", "schema_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DINOv3联合模型无效"):
        model_services.list_model_classes(str(old_manifest))

    # New contract but missing asset
    missing_asset_manifest = tmp_path / "missing.neri.json"
    missing_asset_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "architecture": "dinov3_vitb16",
                "head_type": "ridge_within",
                "feature_dim": 768,
                "preprocessing": PREPROCESSING,
                "event_aggregation": "complete_sequence",
                "requires_detector": True,
                "supports_video_fast": False,
                "supports_video_all": False,
                "encoder_sha256": "0" * 64,
                "checkpoint_sha256": "1" * 64,
                "checkpoint": "nonexistent.pt",
                "encoder_weights": "weights.pth",
                "encoder_code": "code",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DINOv3联合模型无效"):
        model_services.list_model_classes(str(missing_asset_manifest))


def test_services_validate_dinov3_job_options(tmp_path):
    manifest, _, _, _ = _make_fake_dinov3_assets(tmp_path)

    # Requires detector
    with pytest.raises(ValueError, match="Neri1.1 分类模型必须同时选择探测模型。"):
        services._validate_dino_job_options(None, str(manifest), "fast")

    # Rejects video_mode == 'all' with message recommending fast or skip
    with pytest.raises(ValueError, match="Neri1.1 暂不支持完整视频追踪，请使用快速识别或跳过视频。"):
        services._validate_dino_job_options("detector.pt", str(manifest), "all")

    # Allows 'fast' and 'skip' during job options validation
    services._validate_dino_job_options("detector.pt", str(manifest), "fast")
    services._validate_dino_job_options("detector.pt", str(manifest), "skip")


def test_services_load_detector_attaches_dinov3_runtime(tmp_path, monkeypatch):
    manifest, _, _, _ = _make_fake_dinov3_assets(tmp_path)

    fake_classifier = SimpleNamespace(backend="dinov3")
    fake_runtime = SimpleNamespace(
        classifier=fake_classifier,
        registry=None,
        feedback=None,
        manifest_path=manifest,
    )

    import system.dinov3.runtime as dinov3_runtime_mod

    monkeypatch.setattr(
        dinov3_runtime_mod,
        "load_dinov3_model",
        lambda path, **kw: fake_runtime,
    )

    detect_file = tmp_path / "detect.pt"
    detect_file.write_bytes(b"weights")

    class MockImageProcessor:
        def __init__(self, model_path):
            self.model_path = model_path
            self.model = object()
            self.dino_classifier = None
            self.dino_registry = None
            self.dino_feedback = None
            self.dino_runtime = None

        def load_dino_classifier(self, classifier):
            self.dino_classifier = classifier

    import system.dino.image_processor as img_proc_mod

    monkeypatch.setattr(img_proc_mod, "ImageProcessor", MockImageProcessor)

    detector = services._load_detector(str(detect_file), str(manifest))
    assert detector.dino_classifier is fake_classifier
    assert detector.dino_registry is None
    assert detector.dino_feedback is None
    assert detector.dino_runtime is fake_runtime


@pytest.mark.parametrize("legacy_fallback", [False, True])
@pytest.mark.parametrize("action", ["correct", "update"])
def test_neri_marking_saves_annotations_without_opening_v2_feedback(tmp_path, monkeypatch, action, legacy_fallback):
    from PIL import Image
    from system.backend.models import ValidationBatchMarkRequest
    from system.backend import validation_fast
    import system.training

    manifest, _, _, _ = _make_fake_dinov3_assets(tmp_path)
    photo = tmp_path / "animal.jpg"
    Image.new("RGB", (32, 24), "green").save(photo)
    services._persist_validation_updates([(photo, {"物种名称": "A", "物种数量": "1", "检测框": []}, False)], tmp_path)
    enqueued = []
    monkeypatch.setattr(system.training, "get_queue", lambda: SimpleNamespace(
        enqueue=lambda path, data, **kw: enqueued.append((path, data, kw))))
    if legacy_fallback:
        monkeypatch.setattr(services, "_learnable_observations_for_file", services._legacy._learnable_observations_for_file)
    request = ValidationBatchMarkRequest(input_path=str(tmp_path), file_paths=[str(photo)],
        action=action, species_name="B" if action == "update" else None,
        classification_model_path=str(manifest))
    updated = services.mark_validation_items(request)
    assert len(updated) == 1 and updated[0].validated is True
    # An explicit empty detection list overrides its stale automatic summary.
    # Only an explicit species update supplies a new human image label.
    expected = "B" if action == "update" else "空"
    assert updated[0].species == ([expected] if action == "update" else [])
    assert services._load_detection_data_for_path(photo, [tmp_path])["物种名称"] == expected
    assert len(enqueued) == 1 and enqueued[0][2]["validated"] is True


def test_neri_direct_feedback_helpers_do_not_open_v2_state(tmp_path, monkeypatch):
    manifest, _, _, _ = _make_fake_dinov3_assets(tmp_path)
    assert services._learnable_observations_for_file(str(manifest), tmp_path / "a.jpg") == []
