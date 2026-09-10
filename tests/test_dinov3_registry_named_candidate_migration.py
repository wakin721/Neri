from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from system.dinov3.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)
from system.dinov3.registry import SpeciesRegistry


FP = "c" * 64
BASE = datetime(2026, 9, 10, 0, 0, 0)


def _vector(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _legacy_named_candidate(
    registry: SpeciesRegistry,
    index: int,
    *,
    common_name: str = "野猪",
    scientific_name: str = "Sus scrofa",
):
    entry = registry.record_unknown(
        _vector(index),
        camera_id=f"cam-{index}",
        captured_at=BASE + timedelta(hours=index),
        source_path=f"legacy-{index}.jpg",
    )
    return registry.set_identity(
        entry.id,
        common_name=common_name,
        scientific_name=scientific_name,
    )


def _feedback_observation(entry_id: int, index: int) -> FeedbackObservation:
    return FeedbackObservation(
        id=f"legacy-observation-{entry_id}",
        source_path=f"legacy-{index}.jpg",
        media_kind="image",
        box_index=0,
        frame_index=None,
        timestamp_seconds=None,
        bbox=(0.0, 0.0, 10.0, 10.0),
        camera_id=f"cam-{index}",
        captured_at=BASE + timedelta(hours=index),
        predicted_species="Unknown",
        best_known_species=None,
        accepted=False,
        prediction_source="unknown",
        registry_id=entry_id,
        known_score=0.0,
        threshold=0.5,
        embedding=_vector(index),
    )


def test_reopen_merges_legacy_same_named_candidates_and_redirects_feedback(tmp_path):
    registry_path = tmp_path / "registry.sqlite3"
    registry = SpeciesRegistry(registry_path, model_fingerprint=FP)
    duplicates = [_legacy_named_candidate(registry, index) for index in range(4)]
    survivor_id = duplicates[0].id

    registered = registry.record_unknown(
        _vector(20),
        camera_id="registered-0",
        captured_at=BASE,
        source_path="registered-0.jpg",
    )
    for index in range(1, 4):
        registry.record_observation(
            registered.id,
            _vector(20),
            camera_id=f"registered-{index}",
            captured_at=BASE + timedelta(hours=index),
            source_path=f"registered-{index}.jpg",
        )
    registry.set_identity(
        registered.id,
        common_name="野猪",
        scientific_name="Sus scrofa",
    )
    registered = registry.register(registered.id)
    assert registered.status == "provisional"
    registered_id = registered.id
    registry.close()

    feedback = HumanFeedbackStore(
        feedback_path_for_registry(registry_path),
        model_fingerprint=FP,
        checkpoint_classes=(),
        threshold=0.5,
    )
    for index, entry in enumerate(duplicates):
        observation = _feedback_observation(entry.id, index)
        feedback.persist_observation(observation)
        feedback.record_registry_feedback(
            observation.id,
            operation_id="legacy-batch-operation",
            registration_id=entry.id,
            previous_common_name="",
            previous_scientific_name="",
            confirmed_species="野猪",
        )
    feedback.close()

    migrated = SpeciesRegistry(registry_path, model_fingerprint=FP)
    boars = [entry for entry in migrated.list() if entry.common_name == "野猪"]
    candidates = [entry for entry in boars if entry.status == "candidate"]
    provisionals = [entry for entry in boars if entry.status == "provisional"]

    assert len(candidates) == 1
    assert candidates[0].id == survivor_id
    assert candidates[0].display_name == "野猪"
    assert candidates[0].event_count == 4
    assert candidates[0].prototype_count == 1
    assert [entry.id for entry in provisionals] == [registered_id]

    migration_marker = migrated._conn.execute(
        "SELECT value FROM metadata WHERE key='migration_named_candidate_merge_v1'"
    ).fetchone()
    assert migration_marker is not None and migration_marker[0] == "1"
    migrated.close()

    feedback = HumanFeedbackStore(
        feedback_path_for_registry(registry_path),
        model_fingerprint=FP,
        checkpoint_classes=(),
        threshold=0.5,
    )
    assignments = feedback.registry_assignments("legacy-batch-operation")
    assert len(assignments) == 1
    assert assignments[0]["registration_id"] == survivor_id
    assert assignments[0]["identity_restore_allowed"] is False

    redirects = feedback._conn.execute(
        """
        SELECT old_registration_id,new_registration_id
        FROM feedback_registry_assignment_redirects
        WHERE operation_id='legacy-batch-operation'
        ORDER BY old_registration_id
        """
    ).fetchall()
    assert [(int(row[0]), int(row[1])) for row in redirects] == [
        (entry.id, survivor_id) for entry in duplicates
    ]
    feedback.close()

    reopened = SpeciesRegistry(registry_path, model_fingerprint=FP)
    candidates_after_second_open = [
        entry
        for entry in reopened.list()
        if entry.common_name == "野猪" and entry.status == "candidate"
    ]
    assert len(candidates_after_second_open) == 1
    assert candidates_after_second_open[0].event_count == 4
    assert candidates_after_second_open[0].prototype_count == 1
    reopened.close()


def test_migration_skips_conflicting_scientific_names(tmp_path):
    registry_path = tmp_path / "registry.sqlite3"
    registry = SpeciesRegistry(registry_path, model_fingerprint=FP)
    first = _legacy_named_candidate(
        registry,
        30,
        common_name="赤麂",
        scientific_name="Muntiacus vaginalis",
    )
    second = _legacy_named_candidate(
        registry,
        31,
        common_name="赤麂",
        scientific_name="Muntiacus reevesi",
    )
    registry.close()

    reopened = SpeciesRegistry(registry_path, model_fingerprint=FP)
    red_muntjacs = [entry for entry in reopened.list() if entry.common_name == "赤麂"]
    assert [entry.id for entry in red_muntjacs] == [first.id, second.id]
    assert [entry.event_count for entry in red_muntjacs] == [1, 1]
    reopened.close()


def test_merged_assignment_is_not_allowed_to_restore_survivor_identity(tmp_path):
    from system.backend import dinov3_feedback_service

    restore = getattr(dinov3_feedback_service, "_restore_registry_assignments", None)
    assert callable(restore), "merged Registry assignments need a guarded restore helper"

    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = _legacy_named_candidate(registry, 40)
    restore(
        registry,
        [
            {
                "registration_id": entry.id,
                "previous_common_name": "",
                "previous_scientific_name": "",
                "assigned_common_name": "野猪",
                "identity_restore_allowed": False,
            }
        ],
    )
    assert registry.get(entry.id).common_name == "野猪"
    registry.close()
