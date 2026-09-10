from __future__ import annotations

import numpy as np

from system.dinov3.registry import SpeciesRegistry


def _unit(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def test_manual_corrections_with_same_new_species_reuse_one_registry_entry(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint="c" * 64,
    )
    try:
        first, _, _ = registry.record_human_species(
            _unit(0),
            common_name="赤麂",
            camera_id="camera-a",
            captured_at=None,
            source_path="/tmp/red-muntjac-a.jpg",
            bbox=(0, 0, 20, 20),
        )
        second, _, _ = registry.record_human_species(
            _unit(1),
            common_name="赤麂",
            camera_id="camera-b",
            captured_at=None,
            source_path="/tmp/red-muntjac-b.jpg",
            bbox=(0, 0, 20, 20),
        )

        named_entries = [
            entry for entry in registry.list() if entry.common_name == "赤麂"
        ]
        assert second.id == first.id
        assert [entry.id for entry in named_entries] == [first.id]
        assert registry.get(first.id).event_count == 2
    finally:
        registry.close()
