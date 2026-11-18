"""Task submission checks dependencies of the selected model combination."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


@pytest.mark.parametrize(
    "detector,classification,installed,expected",
    [
        (".tflite", None, (), []),
        (".litert", "cpu", (), []),
        (".tflite", "cpu", (), []),
        (".tflite", "gpu", ("torch",), []),
        (".tflite", "gpu", (), ["torch"]),
        (".pt", "cpu", (), ["torch", "torchvision", "ultralytics"]),
        (".pt", None, ("ultralytics",), ["torch", "torchvision"]),
        (".pt", "gpu", ("torch", "torchvision", "ultralytics"), []),
        (".tflite", "yolo", (), ["torch", "torchvision", "ultralytics"]),
        (None, "yolo", (), ["torch", "torchvision", "ultralytics"]),
    ],
)
def test_create_job_checks_only_selected_model_dependencies(
    tmp_path, monkeypatch, detector, classification, installed, expected
):
    from system.backend import main_core_legacy as core
    from system.backend.models import CreateJobRequest, ProcessingOptions

    detector_path = tmp_path / f"detector{detector}" if detector else None
    if detector_path:
        detector_path.write_bytes(b"detector")
    classifier_path = None
    if classification == "yolo":
        classifier_path = tmp_path / "classifier.pt"
        classifier_path.write_bytes(b"classifier")
    elif classification:
        classifier_path = tmp_path / "classifier.neri.json"
        encoder = "encoder.tflite" if classification == "cpu" else "encoder.pth"
        (tmp_path / encoder).write_bytes(b"encoder")
        (tmp_path / "head.neri-model").write_bytes(b"head")
        (tmp_path / "code").mkdir()
        classifier_path.write_text(json.dumps(dict(
            schema_version=1, backend="dinov3", architecture="dinov3_vitb16",
            encoder_backend="litert" if classification == "cpu" else "pytorch",
            encoder_weights=encoder, encoder_code="code",
            encoder_sha256="a" * 64, encoder_litert_sha256="b" * 64,
            checkpoint="head.neri-model", checkpoint_sha256="c" * 64,
            head_type="ridge_within_experiment", inference_recipe="experiment_full",
            feature_dim=768, preprocessing="letterbox224_imagenet",
            event_aggregation="complete_sequence", requires_detector=True,
            supports_video_fast=False, supports_video_all=False,
        )), encoding="utf-8")

    original_find_spec = importlib.util.find_spec
    modules = {"torch", "torchvision", "ultralytics"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: (
        object() if name in installed else None
    ) if name in modules else original_find_spec(name))
    submitted = []
    queued = object()
    monkeypatch.setattr(core, "job_manager", SimpleNamespace(
        create_job=lambda request: submitted.append(request) or queued,
    ))
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        model_path=str(detector_path) if detector_path else None,
        classification_model_path=str(classifier_path) if classifier_path else None,
        enable_detection=True, video_mode="skip",
    ))

    if expected:
        with pytest.raises(HTTPException) as error:
            core.create_job(request)
        assert error.value.status_code == 409
        assert error.value.detail["missing"] == expected
        assert submitted == []
    else:
        assert core.create_job(request) is queued
        assert submitted == [request]


def test_indexing_does_not_require_inference_dependencies(tmp_path, monkeypatch):
    from system.backend import main_core_legacy as core
    from system.backend.models import CreateJobRequest

    monkeypatch.setattr(core, "missing_yolo_dependencies", lambda: pytest.fail(
        "Indexing must not check inference dependencies"
    ))
    queued = object()
    monkeypatch.setattr(core, "job_manager", SimpleNamespace(create_job=lambda _: queued))
    assert core.create_job(CreateJobRequest(input_dir=str(tmp_path))) is queued


def test_invalid_joint_manifest_is_not_reported_as_missing_yolo(tmp_path, monkeypatch):
    from system.backend import main_core_legacy as core
    from system.backend.models import CreateJobRequest, ProcessingOptions

    detector = tmp_path / "detector.tflite"
    detector.write_bytes(b"detector")
    manifest = tmp_path / "invalid.neri.json"
    manifest.write_text('{"backend":"dinov3"}', encoding="utf-8")
    monkeypatch.setattr(core, "job_manager", SimpleNamespace(
        create_job=lambda _: pytest.fail("Invalid manifest must not be queued"),
    ))
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        model_path=str(detector), classification_model_path=str(manifest),
        enable_detection=True, video_mode="skip",
    ))
    with pytest.raises(HTTPException) as error:
        core.create_job(request)
    assert error.value.status_code == 400
    assert "DINOv3" in error.value.detail
