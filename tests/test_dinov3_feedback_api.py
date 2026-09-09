from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from system.backend.dinov3_feedback_api import dinov3_feedback_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(dinov3_feedback_router())
    return TestClient(app)


def _box_payload(*, action: str, species_name: str | None = None) -> dict:
    payload = {
        "input_path": "C:/camera",
        "file_path": "C:/camera/a.jpg",
        "classification_model_path": "model.neri.json",
        "observation_id": "obs-1",
        "action": action,
        "feedback_operation_id": f"op-{action}",
    }
    if species_name is not None:
        payload["species_name"] = species_name
    return payload


def _contains_embedding(value) -> bool:
    if isinstance(value, dict):
        return "embedding" in value or any(
            _contains_embedding(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_embedding(item) for item in value)
    return False


def test_box_feedback_requires_matching_observation(monkeypatch):
    import system.backend.dinov3_feedback_api as api

    def missing(_request):
        raise KeyError("missing")

    monkeypatch.setattr(api, "apply_box_feedback", missing)
    response = _client().post(
        "/api/dinov3/feedback/box",
        json=_box_payload(action="correct"),
    )

    assert response.status_code in {400, 404}


def test_box_feedback_actions_return_public_payload_without_embedding(monkeypatch):
    import system.backend.dinov3_feedback_api as api

    calls = []

    def apply(request):
        calls.append(
            (
                request.action,
                request.species_name,
                request.feedback_operation_id,
            )
        )
        return {
            "item": {
                "filename": "a.jpg",
                "path": "C:/camera/a.jpg",
                "file_type": "jpg",
                "species": [request.species_name or "Known"],
                "detection_boxes": [
                    {
                        "species": request.species_name or "Known",
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "candidates": [
                            {
                                "name": "Known",
                                "observation_id": "obs-1",
                            }
                        ],
                    }
                ],
            },
            "operation_id": request.feedback_operation_id,
            "affected_species": [request.species_name or "Known"],
        }

    monkeypatch.setattr(api, "apply_box_feedback", apply)
    client = _client()
    cases = [
        ("correct", None),
        ("update", "Other"),
        ("empty", None),
    ]

    for action, species_name in cases:
        response = client.post(
            "/api/dinov3/feedback/box",
            json=_box_payload(action=action, species_name=species_name),
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["operation_id"] == f"op-{action}"
        assert isinstance(payload["affected_species"], list)
        assert _contains_embedding(payload) is False

    assert calls == [
        ("correct", None, "op-correct"),
        ("update", "Other", "op-update"),
        ("empty", None, "op-empty"),
    ]


def test_revert_feedback_returns_affected_species_without_embedding(monkeypatch):
    import system.backend.dinov3_feedback_api as api

    def revert(request):
        assert request.classification_model_path == "model.neri.json"
        assert request.feedback_operation_id == "op-1"
        return {
            "operation_id": request.feedback_operation_id,
            "affected_species": ["Known", "Other"],
        }

    monkeypatch.setattr(api, "revert_feedback_operation", revert)
    response = _client().post(
        "/api/dinov3/feedback/revert",
        json={
            "classification_model_path": "model.neri.json",
            "feedback_operation_id": "op-1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "operation_id": "op-1",
        "affected_species": ["Known", "Other"],
    }
    assert _contains_embedding(response.json()) is False


def test_revert_missing_operation_returns_not_found(monkeypatch):
    import system.backend.dinov3_feedback_api as api

    def missing(_request):
        raise KeyError("op-missing")

    monkeypatch.setattr(api, "revert_feedback_operation", missing)
    response = _client().post(
        "/api/dinov3/feedback/revert",
        json={
            "classification_model_path": "model.neri.json",
            "feedback_operation_id": "op-missing",
        },
    )

    assert response.status_code == 404
