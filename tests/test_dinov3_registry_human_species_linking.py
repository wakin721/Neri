from __future__ import annotations

import numpy as np

from system.dinov3.registry import SpeciesRegistry


def _unit(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _record_named(
    registry: SpeciesRegistry,
    embedding: np.ndarray,
    *,
    name: str,
    camera: str,
    path: str,
):
    return registry.record_human_species(
        embedding,
        common_name=name,
        camera_id=camera,
        captured_at=None,
        source_path=path,
        bbox=(0, 0, 20, 20),
    )


def test_manual_corrections_with_same_new_species_reuse_one_registry_entry(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint="c" * 64,
    )
    try:
        first, _, _ = _record_named(
            registry,
            _unit(0),
            name="赤麂",
            camera="camera-a",
            path="/tmp/red-muntjac-a.jpg",
        )
        second, _, _ = _record_named(
            registry,
            _unit(1),
            name="赤麂",
            camera="camera-b",
            path="/tmp/red-muntjac-b.jpg",
        )

        named_entries = [
            entry for entry in registry.list() if entry.common_name == "赤麂"
        ]
        assert second.id == first.id
        assert [entry.id for entry in named_entries] == [first.id]
        assert registry.get(first.id).event_count == 2
    finally:
        registry.close()


def test_manual_correction_reuses_same_species_after_candidate_is_registered(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint="d" * 64,
        consistency_threshold=0.0,
    )
    try:
        first_id = None
        for index in range(4):
            entry, _, _ = _record_named(
                registry,
                _unit(0),
                name="赤麂",
                camera=f"camera-{index % 2}",
                path=f"/tmp/red-muntjac-{index}.jpg",
            )
            first_id = entry.id if first_id is None else first_id
            assert entry.id == first_id

        registered = registry.register(first_id)
        assert registered.status == "provisional"

        later, _, _ = _record_named(
            registry,
            _unit(1),
            name="赤麂",
            camera="camera-later",
            path="/tmp/red-muntjac-later.jpg",
        )

        assert later.id == first_id
        assert later.status == "provisional"
        assert registry.get(first_id).event_count == 5
        assert len([entry for entry in registry.list() if entry.common_name == "赤麂"]) == 1
    finally:
        registry.close()
