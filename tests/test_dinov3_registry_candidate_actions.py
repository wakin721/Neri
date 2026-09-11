from __future__ import annotations

from system.backend.dinov3_feedback_service import _open_feedback_state
from tests.test_dinov3_api import (
    _client,
    _seed_registration,
    _write_model_bundle,
    _write_multi_prototype_bundle,
)


def test_registry_candidate_can_merge_into_same_name_checkpoint(monkeypatch, tmp_path):
    manifest = _write_multi_prototype_bundle(tmp_path)
    client = _client(monkeypatch, tmp_path)
    registration_id = _seed_registration(
        manifest,
        event_count=5,
        set_identity=False,
    )

    identity = client.patch(
        f"/api/dinov3/registry/{registration_id}/identity",
        json={
            "classification_model_path": str(manifest),
            "common_name": "A",
            "scientific_name": "",
        },
    )
    assert identity.status_code == 200

    merged = client.post(
        f"/api/dinov3/registry/{registration_id}/merge-checkpoint",
        json={
            "classification_model_path": str(manifest),
            "checkpoint_species": "A",
        },
    )

    assert merged.status_code == 200
    assert merged.json() == {
        "merged": True,
        "registration_id": registration_id,
        "checkpoint_species": "A",
        "event_count": 5,
    }

    missing = client.get(
        f"/api/dinov3/registry/{registration_id}",
        params={"classification_model_path": str(manifest)},
    )
    assert missing.status_code == 404

    catalog = client.get(
        "/api/dinov3/registry/catalog",
        params={"classification_model_path": str(manifest)},
    )
    assert catalog.status_code == 200
    checkpoint = next(
        entry
        for entry in catalog.json()
        if entry["status"] == "checkpoint" and entry["common_name"] == "A"
    )
    assert checkpoint["feedback_event_count"] == 5
    assert checkpoint["feedback_prototype_count"] == 1
    assert checkpoint["learning_status"] == "provisional"


def test_registry_candidate_can_be_marked_empty_and_removed(monkeypatch, tmp_path):
    manifest = _write_model_bundle(tmp_path)
    client = _client(monkeypatch, tmp_path)
    registration_id = _seed_registration(
        manifest,
        event_count=3,
        set_identity=False,
    )

    discarded = client.post(
        f"/api/dinov3/registry/{registration_id}/empty",
        json={"classification_model_path": str(manifest)},
    )

    assert discarded.status_code == 200
    assert discarded.json() == {
        "discarded": True,
        "registration_id": registration_id,
        "event_count": 3,
    }

    missing = client.get(
        f"/api/dinov3/registry/{registration_id}",
        params={"classification_model_path": str(manifest)},
    )
    assert missing.status_code == 404

    feedback, _ = _open_feedback_state(str(manifest))
    try:
        rows = feedback._conn.execute(
            "SELECT feedback_type FROM human_feedback WHERE active=1 ORDER BY id"
        ).fetchall()
        assert [str(row[0]) for row in rows] == ["empty", "empty", "empty"]
    finally:
        feedback.close()
