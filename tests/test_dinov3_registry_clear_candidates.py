from __future__ import annotations

import numpy as np
import pytest

from system.dinov3.registry import RegistryEntryNotFound, SpeciesRegistry
from tests.test_dinov3_feedback_store import FP


def _unit_feature(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _add_event(
    registry: SpeciesRegistry,
    entry_id: int,
    embedding: np.ndarray,
    *,
    camera_id: str,
    source_path: str,
) -> None:
    registry.record_observation(
        entry_id,
        embedding,
        camera_id=camera_id,
        captured_at=None,
        source_path=source_path,
    )


def test_delete_candidates_removes_only_unregistered_registry_entries(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint=FP,
    )
    formal_embedding = _unit_feature(0)
    formal = registry.record_unknown(
        formal_embedding,
        camera_id="formal-0",
        captured_at=None,
        source_path="formal-0.jpg",
    )
    for index in range(1, 4):
        _add_event(
            registry,
            formal.id,
            formal_embedding,
            camera_id=f"formal-{index}",
            source_path=f"formal-{index}.jpg",
        )
    registry.set_identity(formal.id, common_name="赤麂")
    formal = registry.register(formal.id, formal_matcher=lambda _embedding: None)
    assert formal.status == "provisional"

    candidate = registry.record_unknown(
        _unit_feature(1),
        camera_id="candidate-0",
        captured_at=None,
        source_path="candidate-0.jpg",
    )
    assert candidate.status == "candidate"

    delete_candidates = getattr(registry, "delete_candidates", None)
    assert callable(delete_candidates), "SpeciesRegistry.delete_candidates is required"
    deleted = delete_candidates()

    assert deleted == 1
    assert [(entry.id, entry.status) for entry in registry.list()] == [
        (formal.id, "provisional")
    ]
    with pytest.raises(RegistryEntryNotFound):
        registry.get(candidate.id)
    assert len(registry.list_events(formal.id)) == 4
    assert delete_candidates() == 0
    registry.close()


def test_registry_router_exposes_candidate_clear_endpoint():
    from system.dinov3.api import dinov3_registry_router

    route = next(
        (
            route
            for route in dinov3_registry_router().routes
            if route.path == "/api/dinov3/registry/candidates"
        ),
        None,
    )
    assert route is not None
    assert "DELETE" in route.methods
