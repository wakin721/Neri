from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload
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
