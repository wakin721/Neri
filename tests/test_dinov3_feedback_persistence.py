from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from system.backend.models import DetectionItem
from system.dinov3.classifier import DinoV3Observation
from system.dinov3.feedback import HumanFeedbackStore
from system.dinov3.registry import SpeciesRegistry

FP = "d" * 64


def _embedding(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


class _Detector:
    def __init__(self, *, feedback, registry, observations):
        self.dinov3_feedback = feedback
        self.dinov3_registry = registry
        self._observations = list(observations)

    def drain_dinov3_observations(self):
        observations = tuple(self._observations)
        self._observations = []
        return observations


def _observation(
    observation_id: str,
    *,
    accepted: bool,
    species: str,
    box_index: int,
    embedding_index: int,
    result_index: int = 0,
    source: str = "checkpoint",
    registry_id: int | None = None,
) -> DinoV3Observation:
    return DinoV3Observation(
        result_index=result_index,
        box_index=box_index,
        embedding=_embedding(embedding_index),
        accepted=accepted,
        species=species,
        source=source,
        registry_id=registry_id,
        registration_status="candidate" if registry_id is not None else None,
        known_score=0.9 if accepted else 0.4,
        threshold=0.8,
        detection_confidence=0.95,
        observation_id=observation_id,
        best_known_species="Known",
        bbox=(10.0 + box_index, 20.0, 30.0, 40.0),
    )


def test_backend_persists_known_and_rejected_dinov3_observations(tmp_path):
    from system.backend.dinov3_feedback_service import persist_runtime_observations

    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("Known", "Other"),
        threshold=0.8,
    )
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    path = tmp_path / "camera-a" / "a.jpg"
    item = DetectionItem(
        filename="a.jpg",
        path=str(path),
        file_type="jpg",
        date_taken="2026-09-09T03:04:05",
    )
    detector = _Detector(
        feedback=feedback,
        registry=registry,
        observations=[
            _observation(
                "obs-known",
                accepted=True,
                species="Known",
                box_index=0,
                embedding_index=0,
            ),
            _observation(
                "obs-unknown",
                accepted=False,
                species="Unknown",
                box_index=1,
                embedding_index=1,
            ),
        ],
    )

    persist_runtime_observations(detector, [path], [item], tmp_path)

    known = feedback.get_observation("obs-known")
    unknown = feedback.get_observation("obs-unknown")
    assert known.accepted is True
    assert known.predicted_species == "Known"
    assert unknown.accepted is False
    assert unknown.predicted_species == "Unknown"
    assert known.captured_at == datetime(2026, 9, 9, 3, 4, 5)
    assert unknown.camera_id == "camera-a"

    entries = registry.list()
    assert len(entries) == 1
    assert entries[0].event_count == 1

    feedback.close()
    registry.close()


def test_fast_video_persistence_uses_original_path_and_frame_metadata(tmp_path):
    from system.backend.dinov3_feedback_service import persist_runtime_observations

    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("Known", "Other"),
        threshold=0.8,
    )
    temp_frame_a = tmp_path / "frames" / "frame_0.jpg"
    temp_frame_b = tmp_path / "frames" / "frame_1.jpg"
    original_video = tmp_path / "camera-b" / "clip.mp4"
    item = DetectionItem(
        filename="clip.mp4",
        path=str(original_video),
        file_type="mp4",
        date_taken="2026-09-09T04:00:00",
    )
    detector = _Detector(
        feedback=feedback,
        registry=None,
        observations=[
            _observation(
                "obs-frame-a",
                accepted=True,
                species="Known",
                box_index=0,
                embedding_index=2,
                result_index=0,
            ),
            _observation(
                "obs-frame-b",
                accepted=False,
                species="Unknown",
                box_index=0,
                embedding_index=3,
                result_index=1,
            ),
        ],
    )

    persist_runtime_observations(
        detector,
        [temp_frame_a, temp_frame_b],
        [item, item],
        tmp_path,
        source_paths=[original_video, original_video],
        frame_indices=[100, 200],
        timestamp_seconds=[4.0, 8.0],
    )

    first = feedback.get_observation("obs-frame-a")
    second = feedback.get_observation("obs-frame-b")
    assert first.source_path == str(original_video)
    assert second.source_path == str(original_video)
    assert first.media_kind == "video"
    assert first.frame_index == 100
    assert second.frame_index == 200
    assert first.timestamp_seconds == 4.0
    assert second.timestamp_seconds == 8.0

    feedback.close()
