from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.registry import RegistrationConditionError, SpeciesRegistry
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload

FP = "d" * 64
BASE = datetime(2026, 9, 9, 0, 0, 0)


def _vector(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _candidate_with_events(
    registry: SpeciesRegistry,
    *,
    count: int = 4,
    vector_index: int = 10,
    cameras: tuple[str, ...] = ("cam-0",),
):
    entry = registry.record_unknown(
        _vector(vector_index),
        camera_id=cameras[0],
        captured_at=BASE,
        source_path="0.jpg",
    )
    for index in range(1, count):
        entry = registry.record_observation(
            entry.id,
            _vector(vector_index),
            camera_id=cameras[index % len(cameras)],
            captured_at=BASE + timedelta(hours=index),
            source_path=f"{index}.jpg",
        )
    return registry.set_identity(entry.id, common_name="新物种")


def _append_events(
    registry: SpeciesRegistry,
    entry_id: int,
    *,
    start: int,
    stop: int,
    vector_index: int = 10,
    cameras: tuple[str, ...],
):
    entry = None
    for index in range(start, stop):
        entry = registry.record_observation(
            entry_id,
            _vector(vector_index),
            camera_id=cameras[index % len(cameras)],
            captured_at=BASE + timedelta(hours=index),
            source_path=f"{index}.jpg",
        )
    return entry


def test_four_events_make_candidate_registerable_but_not_provisional_automatically(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.db",
        model_fingerprint=FP,
        consistency_threshold=0.5,
    )
    entry = _candidate_with_events(registry)

    assert entry.event_count == 4
    assert entry.status == "candidate"
    assert entry.can_register is True
    assert registry.prototype_bank(np.zeros(768, dtype=np.float32)).formal == ()
    assert registry.prototype_bank(np.zeros(768, dtype=np.float32)).provisional == ()

    registered = registry.register(entry.id)
    bank = registry.prototype_bank(np.zeros(768, dtype=np.float32))

    assert registered.status == "provisional"
    assert len(bank.formal) == 0
    assert len(bank.provisional) == 1
    assert bank.provisional[0].species.startswith("新物种")
    registry.close()


def test_provisional_is_assistive_then_ten_events_two_cameras_becomes_formal(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.db",
        model_fingerprint=FP,
        consistency_threshold=0.5,
    )
    entry = _candidate_with_events(registry)
    entry = registry.register(entry.id)
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=0.8))
    classifier = DinoV3Classifier(checkpoint, registry=registry)

    provisional = classifier.classify_features(_vector(10)[None, :])[0]
    assert provisional.species.startswith("新物种")
    assert provisional.accepted is False
    assert provisional.assistive_match is True
    assert provisional.source == "overlay"
    assert provisional.registration_status == "provisional"

    entry = _append_events(
        registry,
        entry.id,
        start=4,
        stop=10,
        cameras=("cam-0", "cam-1"),
    )
    assert entry.status == "confirmed"

    confirmed = classifier.classify_features(_vector(10)[None, :])[0]
    assert confirmed.species == "新物种"
    assert confirmed.accepted is True
    assert confirmed.assistive_match is False
    assert confirmed.source == "overlay"
    assert confirmed.registration_status == "confirmed"
    registry.close()


def test_twenty_events_three_cameras_promotes_confirmed_species_to_mature(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.db",
        model_fingerprint=FP,
        consistency_threshold=0.5,
    )
    entry = registry.register(_candidate_with_events(registry).id)
    entry = _append_events(
        registry,
        entry.id,
        start=4,
        stop=10,
        cameras=("cam-0", "cam-1"),
    )
    assert entry.status == "confirmed"

    entry = _append_events(
        registry,
        entry.id,
        start=10,
        stop=20,
        cameras=("cam-0", "cam-1", "cam-2"),
    )
    bank = registry.prototype_bank(np.zeros(768, dtype=np.float32))

    assert entry.status == "mature"
    learned = [record for record in bank.formal if record.registry_id == entry.id]
    assert 1 <= len(learned) <= 3
    registry.close()


def test_registration_blocks_candidate_when_eighty_percent_match_same_formal_species(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.db",
        model_fingerprint=FP,
        consistency_threshold=0.5,
    )
    entry = _candidate_with_events(registry, count=5)
    calls = 0

    def formal_matcher(_embedding):
        nonlocal calls
        calls += 1
        return "已有物种" if calls <= 4 else None

    with pytest.raises(RegistrationConditionError, match="已有物种"):
        registry.register(entry.id, formal_matcher=formal_matcher)
    registry.close()
