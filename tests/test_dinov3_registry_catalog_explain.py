from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.feedback import HumanFeedbackStore
from system.dinov3.registry import SpeciesRegistry
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload
from tests.test_dinov3_feedback_store import FP, make_observation
from tests.test_dinov3_feedback_validation_bridge import _isolate_validation, _request


def _unit_feature(index: int = 0) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def test_new_species_validation_routes_to_registry_not_checkpoint_feedback(
    monkeypatch, tmp_path
):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    registry_calls = []
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: [SimpleNamespace(id="obs-1")],
    )
    monkeypatch.setattr(
        services,
        "_checkpoint_species_for_model",
        lambda _model: {"Known"},
        raising=False,
    )
    monkeypatch.setattr(
        services,
        "_record_validation_registry_feedback",
        lambda model, observation, operation_id, confirmed_species: registry_calls.append(
            (model, observation.id, operation_id, confirmed_species)
        ),
        raising=False,
    )

    def checkpoint_feedback_must_not_receive_new_species(*_args, **_kwargs):
        raise AssertionError("new species must not be sent to checkpoint feedback")

    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        checkpoint_feedback_must_not_receive_new_species,
    )

    items = services.mark_validation_items(
        _request(
            tmp_path,
            action="update",
            species_name="赤麂",
            operation_id="op-new-species",
        )
    )

    assert len(items) == 1
    assert registry_calls == [
        ("model.neri.json", "obs-1", "op-new-species", "赤麂")
    ]


def test_checkpoint_species_validation_stays_in_human_feedback(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    feedback_calls = []
    registry_calls = []
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: [SimpleNamespace(id="obs-1")],
    )
    monkeypatch.setattr(
        services,
        "_checkpoint_species_for_model",
        lambda _model: {"Known"},
        raising=False,
    )
    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        lambda model, observation, operation_id, action, confirmed_species: feedback_calls.append(
            (model, observation.id, operation_id, action, confirmed_species)
        ),
    )
    monkeypatch.setattr(
        services,
        "_record_validation_registry_feedback",
        lambda *_args, **_kwargs: registry_calls.append(True),
        raising=False,
    )

    services.mark_validation_items(
        _request(
            tmp_path,
            action="update",
            species_name="Known",
            operation_id="op-known",
        )
    )

    assert feedback_calls == [
        ("model.neri.json", "obs-1", "op-known", "update", "Known")
    ]
    assert registry_calls == []


def test_registry_catalog_contains_checkpoint_and_local_registry_entries():
    import system.dinov3.api as api

    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=-1.0))
    local = {
        "id": 7,
        "candidate_number": 3,
        "status": "confirmed",
        "common_name": "赤麂",
        "scientific_name": "Muntiacus vaginalis",
        "event_count": 12,
        "camera_count": 2,
        "prototype_count": 2,
        "cluster_purity": 0.95,
        "embedding_consistency": 0.91,
        "conditions": {},
        "can_register": False,
        "display_name": "赤麂",
    }
    registry = SimpleNamespace(list=lambda status=None: [SimpleNamespace(as_dict=lambda: local)])

    builder = getattr(api, "build_registry_catalog", None)
    assert callable(builder), "registry catalog builder is required"
    catalog = builder(checkpoint, registry)

    checkpoint_rows = [item for item in catalog if item["status"] == "checkpoint"]
    registry_rows = [item for item in catalog if item["status"] != "checkpoint"]
    assert [item["display_name"] for item in checkpoint_rows] == list(checkpoint.classes)
    assert [item["prototype_count"] for item in checkpoint_rows] == list(
        checkpoint.prototypes_per_class
    )
    assert registry_rows == [local]


def test_feature_explanation_projects_nearest_two_species_without_embeddings():
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=-1.0))
    classifier = DinoV3Classifier(checkpoint)
    explain = getattr(classifier, "explain_feature", None)
    assert callable(explain), "DinoV3Classifier.explain_feature is required"

    payload = explain(_unit_feature(0))

    assert payload["projection"]["method"] == "nearest_two_species_axis"
    assert payload["projection"]["species"] == ["A", "B"]
    points = payload["projection"]["points"]
    assert any(point["kind"] == "current" for point in points)
    assert {point["species"] for point in points if point["kind"] == "prototype"} == {
        "A",
        "B",
    }
    assert len(payload["nearest_species"]) == 2
    assert payload["nearest_species"][0]["name"] == "A"
    assert payload["nearest_species"][0]["squared_distance"] == pytest.approx(0.0)
    assert "embedding" not in repr(payload).lower()
    for point in points:
        assert np.isfinite(float(point["x"]))
        assert np.isfinite(float(point["y"]))


def test_human_new_species_assignment_is_named_and_audited(tmp_path):
    observation = make_observation("obs-registry")
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint=FP,
    )
    initial = registry.record_unknown(
        observation.embedding,
        camera_id=observation.camera_id,
        captured_at=observation.captured_at,
        source_path=observation.source_path,
        bbox=observation.bbox,
        frame_index=observation.frame_index,
        timestamp_seconds=observation.timestamp_seconds,
    )
    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    feedback.persist_observation(observation)

    assign = getattr(registry, "record_human_species", None)
    assert callable(assign), "SpeciesRegistry.record_human_species is required"
    updated, previous_common, previous_scientific = assign(
        observation.embedding,
        common_name="赤麂",
        camera_id=observation.camera_id,
        captured_at=observation.captured_at,
        source_path=observation.source_path,
        bbox=observation.bbox,
        frame_index=observation.frame_index,
        timestamp_seconds=observation.timestamp_seconds,
        preferred_entry_id=initial.id,
    )
    feedback.record_registry_feedback(
        observation.id,
        operation_id="op-registry",
        registration_id=updated.id,
        previous_common_name=previous_common,
        previous_scientific_name=previous_scientific,
        confirmed_species="赤麂",
    )

    assert updated.id == initial.id
    assert updated.status == "candidate"
    assert updated.common_name == "赤麂"
    assert "赤麂" in updated.display_name
    record = feedback.active_feedback(observation.id)
    assert record is not None
    assert record.feedback_type == "registry"
    assert record.confirmed_species == "赤麂"
    assert record.positive_species is None
    assert feedback.registry_assignments("op-registry") == [
        {
            "registration_id": updated.id,
            "previous_common_name": previous_common,
            "previous_scientific_name": previous_scientific,
            "assigned_common_name": "赤麂",
        }
    ]

    feedback.close()
    registry.close()


def test_registry_and_feedback_routers_expose_catalog_and_observation_explain():
    from system.backend.dinov3_feedback_api import dinov3_feedback_router
    from system.dinov3.api import dinov3_registry_router

    registry_paths = {route.path for route in dinov3_registry_router().routes}
    feedback_paths = {route.path for route in dinov3_feedback_router().routes}

    assert "/api/dinov3/registry/catalog" in registry_paths
    assert "/api/dinov3/feedback/observations/{observation_id}/explain" in feedback_paths
    assert "/api/dinov3/feedback/observations/{observation_id}/example" in feedback_paths
