from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from system.backend.models import DetectionItem
from system.dinov3.classifier import DinoV3Observation
from system.dinov3.events import camera_id_for_path
from system.dinov3.feedback import HumanFeedbackStore
from system.dinov3.registry import SpeciesRegistry
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload

FP = "c" * 64


def _embedding(index=0):
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


class _ObservationDetector:
    def __init__(self, registry, observations):
        self.dinov3_registry = registry
        self._observations = list(observations)

    def drain_dinov3_observations(self):
        result = tuple(self._observations)
        self._observations = []
        return result


def _observation(*, registry_id=None, source="checkpoint", accepted=False):
    return DinoV3Observation(
        result_index=0,
        box_index=0,
        embedding=_embedding(),
        accepted=accepted,
        species="Unknown" if not accepted else "Known",
        source=source,
        registry_id=registry_id,
        registration_status="candidate" if registry_id else None,
        known_score=0.5,
        threshold=0.8,
        detection_confidence=0.9,
    )


def test_base_rejected_observation_is_persisted_and_existing_candidate_accumulates(tmp_path):
    from system.backend import services

    registry = SpeciesRegistry(tmp_path / "registry.db", model_fingerprint=FP)
    camera_dir = tmp_path / "camera-01"
    first_path = camera_dir / "first.jpg"
    first_item = DetectionItem(filename="first.jpg", path=str(first_path), file_type="jpg", date_taken="2026-09-09T00:00:00")
    detector = _ObservationDetector(registry, [_observation()])

    services._persist_dinov3_observations(detector, [first_path], [first_item], tmp_path)

    entries = registry.list()
    assert len(entries) == 1
    assert entries[0].event_count == 1
    assert entries[0].camera_count == 1

    second_path = camera_dir / "second.jpg"
    second_item = DetectionItem(filename="second.jpg", path=str(second_path), file_type="jpg", date_taken="2026-09-09T00:29:59")
    detector._observations = [_observation(registry_id=entries[0].id, source="registry")]
    services._persist_dinov3_observations(detector, [second_path], [second_item], tmp_path)

    updated = registry.get(entries[0].id)
    assert updated.event_count == 1
    event_rows = registry.list_events(updated.id)
    assert event_rows[0]["sample_count"] == 2
    registry.close()


def test_exact_30_minute_gap_creates_new_event(tmp_path):
    from system.backend import services

    registry = SpeciesRegistry(tmp_path / "registry.db", model_fingerprint=FP)
    camera_dir = tmp_path / "camera-02"
    first_path = camera_dir / "first.jpg"
    first_item = DetectionItem(filename="first.jpg", path=str(first_path), file_type="jpg", date_taken="2026-09-09T01:00:00")
    detector = _ObservationDetector(registry, [_observation()])
    services._persist_dinov3_observations(detector, [first_path], [first_item], tmp_path)
    entry = registry.list()[0]

    second_path = camera_dir / "second.jpg"
    second_item = DetectionItem(filename="second.jpg", path=str(second_path), file_type="jpg", date_taken="2026-09-09T01:30:00")
    detector._observations = [_observation(registry_id=entry.id, source="registry")]
    services._persist_dinov3_observations(detector, [second_path], [second_item], tmp_path)

    assert registry.get(entry.id).event_count == 2
    registry.close()


def test_backend_persists_checkpoint_accepted_observation_to_feedback_only(tmp_path):
    from system.backend import services

    registry = SpeciesRegistry(tmp_path / "registry.db", model_fingerprint=FP)
    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("Known", "Other"),
        threshold=0.8,
    )
    path = tmp_path / "camera-03" / "accepted.jpg"
    item = DetectionItem(
        filename=path.name,
        path=str(path),
        file_type="jpg",
        date_taken="2026-09-09T02:03:04",
    )
    observation = DinoV3Observation(
        result_index=0,
        box_index=2,
        embedding=_embedding(4),
        accepted=True,
        species="Known",
        source="checkpoint",
        registry_id=None,
        registration_status=None,
        known_score=0.91,
        threshold=0.8,
        detection_confidence=0.93,
        observation_id="obs-accepted",
        best_known_species="Known",
        bbox=(1.0, 2.0, 30.0, 40.0),
    )
    detector = _ObservationDetector(registry, [observation])
    detector.dinov3_feedback = feedback

    services._persist_dinov3_observations(detector, [path], [item], tmp_path)

    stored = feedback.get_observation("obs-accepted")
    assert stored.source_path == str(path)
    assert stored.media_kind == "image"
    assert stored.box_index == 2
    assert stored.frame_index is None
    assert stored.timestamp_seconds is None
    assert stored.bbox == (1.0, 2.0, 30.0, 40.0)
    assert stored.camera_id == camera_id_for_path(path, tmp_path)
    assert stored.captured_at == datetime(2026, 9, 9, 2, 3, 4)
    assert stored.predicted_species == "Known"
    assert stored.best_known_species == "Known"
    assert stored.accepted is True
    assert stored.prediction_source == "checkpoint"
    assert stored.registry_id is None
    assert stored.known_score == pytest.approx(0.91)
    assert stored.threshold == pytest.approx(0.8)
    assert np.allclose(stored.embedding, _embedding(4))
    assert registry.list() == []

    feedback.close()
    registry.close()


def test_dinov3_job_restrictions_require_detector_and_reject_full_video(tmp_path):
    from system.backend import services

    manifest = tmp_path / "head.neri.json"
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"placeholder")
    manifest.write_text(
        '{"backend":"dinov3","checkpoint":"head.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="探测模型"):
        services._validate_dinov3_job_options(None, str(manifest), "fast")
    with pytest.raises(ValueError, match="完整识别"):
        services._validate_dinov3_job_options("detector.pt", str(manifest), "all")

    services._validate_dinov3_job_options("detector.pt", str(manifest), "fast")


def _write_test_checkpoint(path: Path):
    import torch

    torch.save(
        make_multi_prototype_payload(
            classes=("species-a", "species-b"),
            threshold=0.4,
        ),
        path,
    )


def test_runtime_resolves_manifest_from_checkpoint_and_attaches_registry(tmp_path):
    from system.dinov3.runtime import load_dinov3_model, resolve_dinov3_manifest

    checkpoint = tmp_path / "head.pt"
    _write_test_checkpoint(checkpoint)
    manifest = tmp_path / "head.neri.json"
    manifest.write_text(
        '{"backend":"dinov3","checkpoint":"head.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )
    assert resolve_dinov3_manifest(checkpoint) == manifest.resolve()

    from system.dinov3.checkpoint import load_checkpoint
    fingerprint = load_checkpoint(checkpoint).fingerprint
    registry = SpeciesRegistry(tmp_path / "registry.db", model_fingerprint=fingerprint)
    runtime = load_dinov3_model(
        manifest,
        registry=registry,
        encoder_factory=lambda checkpoint, **kwargs: object(),
    )
    assert runtime.manifest_path == manifest.resolve()
    assert runtime.classifier.registry is registry
    assert runtime.checkpoint.feature_dim == 768
    registry.close()


def test_default_dinov3_state_root_honors_environment(monkeypatch, tmp_path):
    from system.dinov3.state import default_dinov3_state_root

    monkeypatch.setenv("NERI_DINOV3_STATE_DIR", str(tmp_path / "custom"))
    assert default_dinov3_state_root() == (tmp_path / "custom").resolve()


def test_load_detector_uses_native_dinov3_runtime(monkeypatch, tmp_path):
    import sys
    import types
    from types import SimpleNamespace
    from system.backend import services
    from system.dinov3 import runtime as dinov3_runtime

    detector_path = tmp_path / "detector.pt"
    detector_path.write_bytes(b"detector")
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"head")
    manifest = tmp_path / "head.neri.json"
    manifest.write_text(
        '{"backend":"dinov3","checkpoint":"head.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )

    classifier = object()
    registry = object()
    feedback = object()
    fake_runtime = SimpleNamespace(
        classifier=classifier,
        registry=registry,
        feedback=feedback,
    )
    monkeypatch.setattr(dinov3_runtime, "load_dinov3_model", lambda *args, **kwargs: fake_runtime)

    class FakeProcessor:
        def __init__(self, model_path):
            self.model = object() if model_path else None
            self.cls_model = None
            self.dinov3_classifier = None
            self.loaded_cls_paths = []

        def load_cls_model(self, path):
            self.loaded_cls_paths.append(path)
            self.cls_model = object()

        def load_dinov3_classifier(self, value):
            self.dinov3_classifier = value

    module = types.ModuleType("system.image_processor")
    module.ImageProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "system.image_processor", module)

    processor = services._load_detector(str(detector_path), str(manifest))

    assert processor.dinov3_classifier is classifier
    assert processor.dinov3_registry is registry
    assert processor.dinov3_feedback is feedback
    assert processor.loaded_cls_paths == []


def test_metadata_item_preserves_exif_time_for_dinov3_events(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from system.backend import services

    path = tmp_path / "camera" / "image.jpg"
    path.parent.mkdir()
    path.write_bytes(b"x")
    image = SimpleNamespace(width=10, height=20, close=lambda: None)
    captured = datetime(2026, 9, 9, 1, 2, 3)
    monkeypatch.setattr(
        services.ImageMetadataExtractor,
        "extract_metadata",
        lambda *args: (
            {
                "宽度": 10,
                "高度": 20,
                "拍摄日期": "2026-09-09",
                "拍摄时间": "01:02:03",
                "拍摄日期对象": captured,
            },
            image,
        ),
    )

    item = services._build_metadata_item(path)

    assert item.date_taken == "2026-09-09T01:02:03"
