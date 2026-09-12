from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import cv2
import numpy as np

from system.backend.dinov3_feedback_service import persist_runtime_observations
from system.backend.dinov3_registry_service import render_registry_example
from system.dinov3.registry import SpeciesRegistry
from system.dinov3.registry_examples import read_feedback_observation_example


FP = "e" * 64
BASE = datetime(2026, 9, 12, 12, 0, 0)


def _vector(index: int = 0) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _write_image(path, value: int = 180) -> None:
    image = np.full((120, 160, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_registry_event_example_survives_source_deletion(tmp_path):
    source = tmp_path / "event.jpg"
    _write_image(source)
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-a",
        captured_at=BASE,
        source_path=str(source),
        bbox=(30.0, 20.0, 130.0, 100.0),
    )
    event = registry.list_events(entry.id)[0]

    source.unlink()
    payload = render_registry_example(registry, entry.id, int(event["id"]))

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (320, 320, 3)
    assert registry.list_events(entry.id)[0]["has_example"] is True
    registry.close()


def test_provisional_species_keeps_event_example_after_sources_are_removed(tmp_path):
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    first_source = tmp_path / "event-0.jpg"
    _write_image(first_source, 100)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-0",
        captured_at=BASE,
        source_path=str(first_source),
        bbox=(30.0, 20.0, 130.0, 100.0),
    )
    sources = [first_source]
    for index in range(1, 4):
        source = tmp_path / f"event-{index}.jpg"
        _write_image(source, 100 + index)
        sources.append(source)
        registry.record_observation(
            entry.id,
            _vector(),
            camera_id=f"cam-{index}",
            captured_at=BASE + timedelta(hours=index),
            source_path=str(source),
            bbox=(30.0, 20.0, 130.0, 100.0),
        )

    registry.set_identity(entry.id, common_name="新物种", scientific_name="Species nova")
    registered = registry.register(entry.id)
    assert registered.status == "provisional"
    event = registry.list_events(entry.id)[0]

    for source in sources:
        source.unlink()
    payload = render_registry_example(registry, entry.id, int(event["id"]))

    assert payload
    assert registry.get(entry.id).status == "provisional"
    assert registry.list_events(entry.id)[0]["has_example"] is True
    registry.close()


def test_runtime_feedback_persists_checkpoint_learning_example(tmp_path):
    source = tmp_path / "checkpoint.jpg"
    _write_image(source)

    class FeedbackStub:
        def __init__(self):
            self.path = tmp_path / "feedback.sqlite3"
            self.observations = []

        def persist_observation(self, observation):
            self.observations.append(observation)

    feedback = FeedbackStub()
    observation = SimpleNamespace(
        result_index=0,
        observation_id="obs-checkpoint-1",
        box_index=0,
        bbox=(30.0, 20.0, 130.0, 100.0),
        species="盘羊",
        best_known_species="盘羊",
        accepted=True,
        source="checkpoint",
        registry_id=None,
        known_score=0.95,
        threshold=0.5,
        embedding=_vector(),
    )

    class DetectorStub:
        dinov3_feedback = feedback
        dinov3_registry = None

        def drain_dinov3_observations(self):
            return (observation,)

    item = SimpleNamespace(date_taken="2026-09-12 12:00:00")
    persist_runtime_observations(
        DetectorStub(),
        [source],
        [item],
        tmp_path,
    )

    source.unlink()
    payload = read_feedback_observation_example(
        feedback.path,
        observation.observation_id,
    )
    assert payload


def test_saving_same_species_name_does_not_merge_current_candidates(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint=FP,
        join_threshold=1.1,
    )
    first = registry.record_unknown(
        _vector(0),
        camera_id="cam-a",
        captured_at=BASE,
        source_path="a.jpg",
    )
    second = registry.record_unknown(
        _vector(1),
        camera_id="cam-b",
        captured_at=BASE + timedelta(hours=1),
        source_path="b.jpg",
    )

    registry.set_identity(first.id, common_name="赤麂")
    updated = registry.set_identity(second.id, common_name="赤麂")

    assert updated.id == second.id
    current_ids = [
        entry.id
        for entry in registry.list()
        if entry.status == "candidate" and entry.common_name == "赤麂"
    ]
    assert current_ids == [first.id, second.id]
    registry.close()


def test_new_named_candidates_do_not_merge_after_registry_reopen(tmp_path):
    registry_path = tmp_path / "registry.sqlite3"
    registry = SpeciesRegistry(
        registry_path,
        model_fingerprint=FP,
        join_threshold=1.1,
    )
    first = registry.record_unknown(
        _vector(10),
        camera_id="cam-a",
        captured_at=BASE,
        source_path="first.jpg",
    )
    second = registry.record_unknown(
        _vector(11),
        camera_id="cam-b",
        captured_at=BASE + timedelta(hours=1),
        source_path="second.jpg",
    )
    registry.set_identity(first.id, common_name="林麝")
    registry.set_identity(second.id, common_name="林麝")
    registry.close()

    reopened = SpeciesRegistry(
        registry_path,
        model_fingerprint=FP,
        join_threshold=1.1,
    )
    current_ids = [
        entry.id
        for entry in reopened.list()
        if entry.status == "candidate" and entry.common_name == "林麝"
    ]
    assert current_ids == [first.id, second.id]
    reopened.close()
