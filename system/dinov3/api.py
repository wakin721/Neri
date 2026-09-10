"""FastAPI router for DINOv3 species registration state."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from system.backend.dinov3_registry_service import (
    load_checkpoint_for_model,
    open_registry_for_model,
    render_registry_example,
)
from system.backend.models import (
    DinoV3IdentityUpdateRequest,
    DinoV3RegisterRequest,
    DinoV3RegistryEntryResponse,
    DinoV3RegistryEventResponse,
)
from .checkpoint import CheckpointValidationError
from .registry import RegistrationConditionError, RegistryEntryNotFound
from .runtime import DinoV3ManifestError


def _run_with_registry(classification_model_path: str, action: Callable[[Any], Any]) -> Any:
    try:
        registry = open_registry_for_model(classification_model_path)
    except (DinoV3ManifestError, CheckpointValidationError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return action(registry)
    except RegistryEntryNotFound as exc:
        raise HTTPException(status_code=404, detail="DINOv3 registration not found") from exc
    except RegistrationConditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        registry.close()


def _register_with_duplicate_guard(
    registry: Any,
    registration_id: int,
    classification_model_path: str,
) -> dict[str, Any]:
    # Keep classifier/PyTorch imports out of module import time so the registry
    # API itself remains importable in environments where torch is unavailable.
    from .classifier import DinoV3Classifier

    checkpoint = load_checkpoint_for_model(classification_model_path)
    classifier = DinoV3Classifier(checkpoint, registry=registry)

    def formal_matcher(embedding):
        prediction = classifier.classify_features(embedding[None, :])[0]
        if not prediction.accepted:
            return None
        return prediction.species

    return registry.register(
        registration_id,
        formal_matcher=formal_matcher,
    ).as_dict()


def build_registry_catalog(
    checkpoint: Any,
    registry: Any,
    *,
    feedback: Any | None = None,
) -> list[dict[str, Any]]:
    """Merge checkpoint, human-feedback, and local Registry state for inspection."""
    result: list[dict[str, Any]] = []
    counts = tuple(int(value) for value in checkpoint.prototypes_per_class)
    raw_class_indices = checkpoint.prototype_class_indices
    if hasattr(raw_class_indices, "tolist"):
        class_indices = [int(value) for value in raw_class_indices.tolist()]
    else:
        class_indices = [int(value) for value in raw_class_indices]

    for index, species in enumerate(checkpoint.classes):
        base_indices = [
            prototype_index
            for prototype_index, class_index in enumerate(class_indices)
            if class_index == index
        ]
        clusters: list[dict[str, Any]] = [
            {
                "id": f"checkpoint:{index}:{local_index}",
                "label": f"Base #{local_index + 1}",
                "source": "checkpoint",
                "prototype_index": prototype_index,
                "event_count": 0,
                "camera_count": 0,
                "sample_count": 0,
                "mean_squared_distance": None,
                "active": True,
                "learning_status": None,
                "example_refs": [],
            }
            for local_index, prototype_index in enumerate(base_indices)
        ]
        feedback_event_count = 0
        feedback_prototype_count = 0
        learning_status = None
        if feedback is not None:
            state = feedback.learning_state(species)
            if state.positive_events > 0:
                feedback_event_count = int(state.positive_events)
                feedback_prototype_count = int(state.prototype_count)
                learning_status = str(state.status)
                clusters.extend(
                    feedback.cluster_details(species, checkpoint.feature_center)
                )
        result.append(
            {
                "id": -(index + 1),
                "candidate_number": index + 1,
                "status": "checkpoint",
                "common_name": species,
                "scientific_name": "",
                "event_count": 0,
                "camera_count": 0,
                "prototype_count": counts[index],
                "cluster_purity": 1.0,
                "embedding_consistency": 1.0,
                "conditions": {},
                "can_register": False,
                "display_name": species,
                "feedback_event_count": feedback_event_count,
                "feedback_prototype_count": feedback_prototype_count,
                "learning_status": learning_status,
                "clusters": clusters,
            }
        )

    for entry in registry.list():
        data = entry.as_dict()
        cluster_reader = getattr(registry, "cluster_details", None)
        if callable(cluster_reader):
            data.update(
                {
                    "feedback_event_count": 0,
                    "feedback_prototype_count": 0,
                    "learning_status": entry.status,
                    "clusters": cluster_reader(entry.id),
                }
            )
        result.append(data)
    return result


def _catalog_with_feedback(checkpoint: Any, registry: Any) -> list[dict[str, Any]]:
    from .feedback import HumanFeedbackStore, feedback_path_for_registry

    feedback_path = feedback_path_for_registry(registry.path)
    if not feedback_path.exists():
        return build_registry_catalog(checkpoint, registry)
    feedback = HumanFeedbackStore(
        feedback_path,
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )
    try:
        return build_registry_catalog(checkpoint, registry, feedback=feedback)
    finally:
        feedback.close()


def dinov3_registry_router() -> APIRouter:
    router = APIRouter(prefix="/api/dinov3", tags=["dinov3"])

    @router.get("/registry", response_model=list[DinoV3RegistryEntryResponse])
    def list_registry(
        classification_model_path: str = Query(..., min_length=1),
        status: str | None = Query(default=None),
    ):
        return _run_with_registry(
            classification_model_path,
            lambda registry: [entry.as_dict() for entry in registry.list(status=status)],
        )

    @router.get("/registry/catalog", response_model=list[DinoV3RegistryEntryResponse])
    def list_registry_catalog(
        classification_model_path: str = Query(..., min_length=1),
    ):
        checkpoint = load_checkpoint_for_model(classification_model_path)
        return _run_with_registry(
            classification_model_path,
            lambda registry: _catalog_with_feedback(checkpoint, registry),
        )

    @router.get("/registry/{registration_id}", response_model=DinoV3RegistryEntryResponse)
    def get_registry_entry(
        registration_id: int,
        classification_model_path: str = Query(..., min_length=1),
    ):
        return _run_with_registry(
            classification_model_path,
            lambda registry: registry.get(registration_id).as_dict(),
        )

    @router.get(
        "/registry/{registration_id}/events",
        response_model=list[DinoV3RegistryEventResponse],
    )
    def get_registry_events(
        registration_id: int,
        classification_model_path: str = Query(..., min_length=1),
    ):
        return _run_with_registry(
            classification_model_path,
            lambda registry: registry.list_events(registration_id),
        )

    @router.delete("/registry/{registration_id}")
    def delete_registry_entry(
        registration_id: int,
        classification_model_path: str = Query(..., min_length=1),
    ):
        _run_with_registry(
            classification_model_path,
            lambda registry: registry.delete(registration_id),
        )
        return {"deleted": True, "registration_id": registration_id}

    @router.get("/registry/{registration_id}/events/{event_id}/example")
    def get_registry_example(
        registration_id: int,
        event_id: int,
        classification_model_path: str = Query(..., min_length=1),
    ):
        try:
            content = _run_with_registry(
                classification_model_path,
                lambda registry: render_registry_example(
                    registry, registration_id, event_id
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )

    @router.patch("/registry/{registration_id}/identity", response_model=DinoV3RegistryEntryResponse)
    def update_registry_identity(registration_id: int, request: DinoV3IdentityUpdateRequest):
        return _run_with_registry(
            request.classification_model_path,
            lambda registry: registry.set_identity(
                registration_id,
                common_name=request.common_name,
                scientific_name=request.scientific_name,
            ).as_dict(),
        )

    @router.post("/registry/{registration_id}/register", response_model=DinoV3RegistryEntryResponse)
    def register_species(registration_id: int, request: DinoV3RegisterRequest):
        return _run_with_registry(
            request.classification_model_path,
            lambda registry: _register_with_duplicate_guard(
                registry,
                registration_id,
                request.classification_model_path,
            ),
        )

    return router
