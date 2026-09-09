"""FastAPI router for DINOv3 species registration state."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from system.backend.dinov3_registry_service import (
    load_checkpoint_for_model,
    open_registry_for_model,
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
