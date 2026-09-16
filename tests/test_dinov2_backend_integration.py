from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_dinov2_image_processor_exposes_new_attachment_api():
    from system.dinov2.image_processor import ImageProcessor

    processor = ImageProcessor(None)
    sentinel = object()
    processor.load_dinov2_classifier(sentinel)

    assert processor.dinov2_classifier is sentinel
    assert processor.drain_dinov2_observations() == ()
    assert processor.drain_dinov2_observations() == ()


def test_dinov2_manifest_detection_and_legacy_rejection(tmp_path):
    from system.backend import services

    manifest = tmp_path / "v2.neri.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "backend": "dinov2"}),
        encoding="utf-8",
    )
    assert services._dinov2_manifest_payload(manifest)["backend"] == "dinov2"

    legacy = tmp_path / "v3.neri.json"
    legacy.write_text(
        json.dumps({"schema_version": 1, "backend": "dinov3"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DINOv3.*不再支持"):
        services._dinov2_manifest_payload(legacy)


def test_validate_dinov2_job_requires_detector(tmp_path):
    from system.backend import services

    manifest = tmp_path / "v2.neri.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "backend": "dinov2"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="探测模型"):
        services._validate_dinov2_job_options(None, str(manifest), "fast")
    with pytest.raises(ValueError, match="完整识别"):
        services._validate_dinov2_job_options(
            "detect.pt", str(manifest), "all"
        )


def test_load_detector_attaches_dinov2_runtime(tmp_path, monkeypatch):
    from system.backend import services

    manifest = tmp_path / "v2.neri.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "backend": "dinov2"}),
        encoding="utf-8",
    )
    detector_path = tmp_path / "detect.pt"
    detector_path.write_bytes(b"x")
    fake_runtime = SimpleNamespace(
        classifier=object(),
        registry=object(),
        feedback=object(),
    )
    monkeypatch.setattr(
        services, "_resolve_model_path", lambda _path: detector_path
    )
    monkeypatch.setattr(
        services,
        "_resolve_classification_model_path",
        lambda _path: manifest,
    )
    import system.dinov2.runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "load_dinov2_model",
        lambda _path: fake_runtime,
    )

    detector = services._load_detector(str(detector_path), str(manifest))

    assert detector.dinov2_classifier is fake_runtime.classifier
    assert detector.dinov2_registry is fake_runtime.registry
    assert detector.dinov2_feedback is fake_runtime.feedback
    assert detector.dinov2_runtime is fake_runtime


def test_persistence_drains_dinov2_observations_and_routes_unknown(tmp_path):
    import numpy as np

    from system.backend.dinov2_feedback_service import persist_runtime_observations
    from system.dinov2.classifier import DinoV2Observation

    persisted = []
    unknown = []

    class Feedback:
        def persist_observation(self, observation):
            persisted.append(observation)

    class Registry:
        def record_unknown(self, embedding, **kwargs):
            unknown.append((embedding.copy(), kwargs))

    detector = SimpleNamespace(
        dinov2_feedback=Feedback(),
        dinov2_registry=Registry(),
    )
    embedding = np.zeros(768, dtype=np.float32)
    embedding[0] = 1.0
    observations = [
        DinoV2Observation(
            result_index=0,
            box_index=0,
            embedding=embedding,
            accepted=False,
            species="Unknown",
            source="checkpoint",
            registry_id=None,
            registration_status=None,
            known_score=0.2,
            threshold=0.3,
            detection_confidence=0.9,
            observation_id="obs-1",
            best_known_species="A",
            bbox=(1, 2, 3, 4),
            squared_distance=0.8,
            class_margin=0.1,
            adjusted_distance_score=-0.6,
            score_threshold=-0.5,
            registry_action="new_mode_candidate",
        )
    ]
    detector.drain_dinov2_observations = lambda: tuple(observations)
    media = tmp_path / "cam1" / "x.jpg"
    media.parent.mkdir()
    media.write_bytes(b"x")
    item = SimpleNamespace(date_taken="2026-09-16T12:00:00")

    persist_runtime_observations(detector, [media], [item], tmp_path)

    assert len(persisted) == 1
    assert persisted[0].id == "obs-1"
    assert persisted[0].best_known_species == "A"
    assert len(unknown) == 1
    assert unknown[0][1]["candidate_kind"] == "new_mode_candidate"
    assert unknown[0][1]["source_path"] == str(media)


def test_model_services_reads_dinov2_classes_before_yolo(tmp_path):
    import torch

    from system.backend import model_services
    from tests.dinov2_fixtures import make_dinov2_payload

    checkpoint = tmp_path / "classifier.pt"
    torch.save(make_dinov2_payload(), checkpoint)
    manifest = tmp_path / "classifier.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "checkpoint": checkpoint.name,
            }
        ),
        encoding="utf-8",
    )

    classes = model_services.list_model_classes(str(manifest))

    assert [item.name for item in classes] == ["A", "B"]


def test_model_services_rejects_dinov3_manifest(tmp_path):
    from system.backend import model_services

    manifest = tmp_path / "legacy.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "checkpoint": "old.pt",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DINOv3.*不再支持"):
        model_services.list_model_classes(str(manifest))
