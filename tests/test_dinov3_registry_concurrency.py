from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from system.dinov3.registry import SpeciesRegistry


FP = "8" * 64


def _vector(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def test_concurrent_registry_instances_allocate_unique_candidates(tmp_path) -> None:
    path = tmp_path / "registry.sqlite3"

    def create(index: int):
        registry = SpeciesRegistry(path, model_fingerprint=FP)
        try:
            return registry.record_unknown(
                _vector(index),
                camera_id=f"camera-{index}",
                captured_at=datetime(2026, 9, 15, index),
                source_path=f"{index}.jpg",
            )
        finally:
            registry.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        entries = list(executor.map(create, range(8)))

    assert len({entry.id for entry in entries}) == 8
    assert len({entry.candidate_number for entry in entries}) == 8


def test_list_and_match_use_bounded_query_counts(tmp_path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    for index in range(10):
        entry_id = registry._create()
        registry.record_observation(
            entry_id,
            _vector(index),
            camera_id=f"camera-{index}",
            captured_at=datetime(2026, 9, 15, index),
            source_path=f"{index}.jpg",
        )

    queries: list[str] = []
    registry._conn.set_trace_callback(queries.append)
    registry.list()
    assert len(queries) <= 4

    queries.clear()
    registry.match(_vector(20))
    assert len(queries) <= 4
    registry.close()
