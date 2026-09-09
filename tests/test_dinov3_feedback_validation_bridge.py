from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from system.backend.models import ValidationBatchMarkRequest


def _request(
    tmp_path,
    *,
    names=("a.jpg",),
    action="correct",
    species_name="Known",
    classification_model_path="model.neri.json",
    operation_id="op-1",
):
    root = tmp_path / "camera"
    root.mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = root / name
        path.write_bytes(b"x")
        paths.append(str(path))
    payload = {
        "input_path": str(root),
        "file_paths": paths,
        "action": action,
        "species_name": species_name,
    }
    if classification_model_path is not None:
        payload["classification_model_path"] = classification_model_path
    if operation_id is not None:
        payload["feedback_operation_id"] = operation_id
    return ValidationBatchMarkRequest(**payload)


def _isolate_validation(monkeypatch, services):
    persisted = []

    def persist(updates, _input_path):
        persisted.extend(list(updates))

    monkeypatch.setattr(services, "_persist_validation_updates", persist)
    training = types.ModuleType("system.training")
    training.get_queue = lambda: SimpleNamespace(enqueue=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "system.training", training)
    return persisted


def test_validation_request_preserves_feedback_context(tmp_path):
    request = _request(tmp_path)

    assert request.classification_model_path == "model.neri.json"
    assert request.feedback_operation_id == "op-1"


def test_single_box_file_mark_correct_creates_feedback(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    request = _request(tmp_path, action="correct")
    recorded = []
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: [SimpleNamespace(id="obs-1")],
    )
    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        lambda _model, observation, operation_id, action, confirmed_species: recorded.append(
            (observation.id, operation_id, action, confirmed_species)
        ),
    )

    services.mark_validation_items(request)

    assert recorded == [("obs-1", "op-1", "correct", None)]


def test_multi_box_file_mark_does_not_auto_feed(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    request = _request(tmp_path, action="correct")
    recorded = []
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: [
            SimpleNamespace(id="obs-1"),
            SimpleNamespace(id="obs-2"),
        ],
    )
    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        lambda *_args, **_kwargs: recorded.append(True),
    )

    services.mark_validation_items(request)

    assert recorded == []


def test_update_and_empty_map_to_explicit_feedback_semantics(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    recorded = []
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: [SimpleNamespace(id="obs-1")],
    )
    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        lambda _model, observation, operation_id, action, confirmed_species: recorded.append(
            (observation.id, operation_id, action, confirmed_species)
        ),
    )

    services.mark_validation_items(
        _request(tmp_path, action="update", species_name="Other", operation_id="op-update")
    )
    services.mark_validation_items(
        _request(tmp_path, action="empty", species_name=None, operation_id="op-empty")
    )

    assert recorded == [
        ("obs-1", "op-update", "update", "Other"),
        ("obs-1", "op-empty", "empty", None),
    ]


def test_batch_uses_one_shared_feedback_operation_id(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    request = _request(tmp_path, names=("a.jpg", "b.jpg"), operation_id="op-batch")
    recorded = []

    def observations(_model, file_path):
        return [SimpleNamespace(id=f"obs-{file_path.stem}")]

    monkeypatch.setattr(services, "_learnable_observations_for_file", observations)
    monkeypatch.setattr(
        services,
        "_record_validation_feedback",
        lambda _model, observation, operation_id, action, confirmed_species: recorded.append(
            (observation.id, operation_id, action, confirmed_species)
        ),
    )

    services.mark_validation_items(request)

    assert recorded == [
        ("obs-a", "op-batch", "correct", None),
        ("obs-b", "op-batch", "correct", None),
    ]


def test_omitted_classification_model_preserves_validation_without_feedback(monkeypatch, tmp_path):
    from system.backend import services

    persisted = _isolate_validation(monkeypatch, services)
    request = _request(
        tmp_path,
        classification_model_path=None,
        operation_id=None,
        action="update",
        species_name="Other",
    )
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("feedback state must not be opened")
        ),
        raising=False,
    )

    items = services.mark_validation_items(request)

    assert len(items) == 1
    assert items[0].validated is True
    assert persisted


def test_unverified_does_not_auto_feed(monkeypatch, tmp_path):
    from system.backend import services

    _isolate_validation(monkeypatch, services)
    request = _request(tmp_path, action="unverified")
    monkeypatch.setattr(
        services,
        "_learnable_observations_for_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unverified must not become learning evidence")
        ),
        raising=False,
    )

    services.mark_validation_items(request)
