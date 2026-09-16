from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from system.backend.dinov2_feedback_service import persist_runtime_observations
from system.backend.models import DetectionItem
from system.dinov2.checkpoint import DinoV2Rejection
from system.dinov2.registry import SpeciesRegistry
from system.dinov2.rejection import margin_route


FP = "9" * 64


def _vector(index: int = 0) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


class _Detector:
    def __init__(self, registry, observation) -> None:
        self.dinov2_registry = registry
        self.dinov2_feedback = None
        self._observation = observation

    def drain_dinov2_observations(self):
        observation, self._observation = self._observation, None
        return (observation,) if observation is not None else ()


def test_margin_rejection_routes_all_four_signal_quadrants() -> None:
    rejection = DinoV2Rejection(
        mode="multi_dual_margin",
        cosine_threshold=0.8,
        adjusted_distance_score_threshold=-0.4,
        margin_weight=2.0,
    )

    assert margin_route(0.9, 0.1, rejection) == "update_prototype"
    assert margin_route(0.9, -0.9, rejection) == "candidate"
    assert margin_route(0.2, 0.1, rejection) == "candidate"
    assert margin_route(0.2, -0.9, rejection) == "new_mode_candidate"


def test_inference_match_cannot_update_existing_registry_prototype(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    original = registry.record_unknown(
        _vector(),
        camera_id="camera-a",
        captured_at=datetime(2026, 9, 15, 0, 0),
        source_path="original.jpg",
    )
    for hour in range(1, 4):
        registry.record_observation(
            original.id,
            _vector(),
            camera_id="camera-a",
            captured_at=datetime(2026, 9, 15, hour, 0),
            source_path=f"original-{hour}.jpg",
        )
    registry.set_identity(original.id, common_name="豹猫")
    original = registry.register(original.id)
    observation = SimpleNamespace(
        result_index=0,
        box_index=0,
        embedding=_vector(),
        accepted=False,
        species="Unknown",
        source="registry",
        registry_id=original.id,
        registration_status="provisional",
        known_score=0.9,
        threshold=0.8,
        detection_confidence=0.9,
        registry_action="candidate",
        bbox=(0.0, 0.0, 10.0, 10.0),
        observation_id="",
        best_known_species="existing",
    )
    source = tmp_path / "camera-a" / "inference.jpg"
    item = DetectionItem(
        filename=source.name,
        path=str(source),
        file_type="jpg",
        date_taken="2026-09-15T01:00:00",
    )

    persist_runtime_observations(
        _Detector(registry, observation),
        [source],
        [item],
        tmp_path,
    )

    assert len(registry.list_events(original.id)) == 4
    entries = registry.list()
    assert len(entries) == 2
    assert entries[1].candidate_kind == "candidate"
    registry.close()


def test_both_failed_gates_create_a_new_mode_candidate(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    observation = SimpleNamespace(
        result_index=0,
        box_index=0,
        embedding=_vector(7),
        accepted=False,
        species="Unknown",
        source="checkpoint",
        registry_id=None,
        registration_status=None,
        known_score=0.2,
        threshold=0.8,
        detection_confidence=0.9,
        registry_action="new_mode_candidate",
        bbox=(0.0, 0.0, 10.0, 10.0),
        observation_id="",
        best_known_species="existing",
    )
    source = tmp_path / "camera-b" / "novel.jpg"
    item = DetectionItem(
        filename=source.name,
        path=str(source),
        file_type="jpg",
        date_taken="2026-09-15T02:00:00",
    )

    persist_runtime_observations(
        _Detector(registry, observation), [source], [item], tmp_path
    )

    entry = registry.list()[0]
    assert entry.candidate_kind == "new_mode_candidate"
    assert entry.display_name.startswith("新模式候选")
    registry.close()


def test_human_confirmation_still_updates_existing_registry_entry(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    original = registry.record_unknown(
        _vector(),
        camera_id="camera-a",
        captured_at=datetime(2026, 9, 15, 0, 0),
        source_path="original.jpg",
    )
    registry.set_identity(original.id, common_name="豹猫")

    updated, _, _ = registry.record_human_species(
        _vector(),
        common_name="豹猫",
        camera_id="camera-a",
        captured_at=datetime(2026, 9, 15, 1, 0),
        source_path="confirmed.jpg",
        preferred_entry_id=original.id,
    )

    assert updated.id == original.id
    assert updated.event_count == 2
    registry.close()
