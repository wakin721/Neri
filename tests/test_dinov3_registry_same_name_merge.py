from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from system.dinov3.registry import RegistryEntryNotFound, SpeciesRegistry


FP = "d" * 64
BASE = datetime(2026, 9, 12, 10, 0, 0)


def _vector(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def test_named_entries_sort_before_unnamed_then_by_event_count(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)

    named_low = registry.record_unknown(
        _vector(10),
        camera_id="cam-a",
        captured_at=BASE,
        source_path="named-low-0.jpg",
    )
    named_low = registry.set_identity(named_low.id, common_name="B物种")

    unnamed_high = registry.record_unknown(
        _vector(11),
        camera_id="cam-b",
        captured_at=BASE,
        source_path="unknown-high-0.jpg",
    )
    registry.record_observation(
        unnamed_high.id,
        _vector(11),
        camera_id="cam-b",
        captured_at=BASE + timedelta(hours=1),
        source_path="unknown-high-1.jpg",
    )
    registry.record_observation(
        unnamed_high.id,
        _vector(11),
        camera_id="cam-b",
        captured_at=BASE + timedelta(hours=2),
        source_path="unknown-high-2.jpg",
    )

    named_high = registry.record_unknown(
        _vector(12),
        camera_id="cam-c",
        captured_at=BASE,
        source_path="named-high-0.jpg",
    )
    registry.record_observation(
        named_high.id,
        _vector(12),
        camera_id="cam-c",
        captured_at=BASE + timedelta(hours=1),
        source_path="named-high-1.jpg",
    )
    named_high = registry.set_identity(named_high.id, common_name="A物种")

    unnamed_low = registry.record_unknown(
        _vector(13),
        camera_id="cam-d",
        captured_at=BASE,
        source_path="unknown-low-0.jpg",
    )

    entries = [entry for entry in registry.list() if entry.status == "candidate"]
    assert [entry.id for entry in entries] == [
        named_high.id,
        named_low.id,
        unnamed_high.id,
        unnamed_low.id,
    ]
    assert [entry.event_count for entry in entries] == [2, 1, 3, 1]
    registry.close()


def test_explicit_same_name_candidate_merge_keeps_target_and_moves_events(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)

    target = registry.record_unknown(
        _vector(20),
        camera_id="cam-a",
        captured_at=BASE,
        source_path="camel-a.jpg",
    )
    target = registry.set_identity(target.id, common_name="家骆驼")

    source = registry.record_unknown(
        _vector(21),
        camera_id="cam-b",
        captured_at=BASE + timedelta(hours=1),
        source_path="camel-b.jpg",
    )
    source = registry.set_identity(source.id, common_name="家骆驼")
    source_event_ids = {event["id"] for event in registry.list_events(source.id)}

    merge = getattr(registry, "merge_candidate_into", None)
    assert callable(merge), "SpeciesRegistry.merge_candidate_into is required"

    merged = merge(source.id, target.id)
    assert merged.id == target.id
    assert merged.common_name == "家骆驼"
    assert merged.event_count == 2
    assert source_event_ids.issubset(
        {event["id"] for event in registry.list_events(target.id)}
    )
    with pytest.raises(RegistryEntryNotFound):
        registry.get(source.id)
    registry.close()
