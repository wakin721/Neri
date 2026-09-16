"""FastAPI router for DINOv2 species registration state."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .checkpoint import CheckpointValidationError
try:
    from .registry import RegistrationConditionError, RegistryEntryNotFound
except ImportError:  # pragma: no cover - defensive for minimal builds
    class RegistryEntryNotFound(KeyError):
        pass
    class RegistrationConditionError(RuntimeError):
        pass
from .runtime import DinoV2ManifestError


class DinoV2ClusterExampleRefResponse(BaseModel):
    kind: str
    registration_id: int | None = None
    event_id: int | None = None
    observation_id: str | None = None


class DinoV2RegistryClusterResponse(BaseModel):
    id: str
    label: str
    source: str
    prototype_index: int
    event_count: int = 0
    camera_count: int = 0
    sample_count: int = 0
    mean_squared_distance: float | None = None
    active: bool = True
    learning_status: str | None = None
    example_refs: list[DinoV2ClusterExampleRefResponse] = Field(default_factory=list)


class DinoV2RegistryEntryResponse(BaseModel):
    id: int
    candidate_number: int
    status: str
    candidate_kind: str = "candidate"
    common_name: str = ""
    scientific_name: str = ""
    event_count: int
    camera_count: int
    prototype_count: int
    cluster_purity: float
    embedding_consistency: float
    conditions: dict[str, bool] = Field(default_factory=dict)
    can_register: bool
    display_name: str
    feedback_event_count: int = 0
    feedback_prototype_count: int = 0
    learning_status: str | None = None
    clusters: list[DinoV2RegistryClusterResponse] = Field(default_factory=list)


class DinoV2RegistryEventResponse(BaseModel):
    id: int
    event_key: str
    source_path: str
    camera_id: str
    started_at: str | None = None
    ended_at: str | None = None
    timestamp_missing: bool = False
    sample_count: int = 1
    bbox: list[float] | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    has_example: bool = False


class DinoV2IdentityUpdateRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    common_name: str = Field(..., min_length=1)
    scientific_name: str = ""


class DinoV2RegisterRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)


class _MergeCheckpointRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    checkpoint_species: str = Field(..., min_length=1)


class _MergeCandidateRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    target_registration_id: int


def _registry_service():
    from system.backend import dinov2_registry_service as service
    return service


def _run_with_registry(classification_model_path: str, action: Callable[[Any], Any]) -> Any:
    try:
        registry = _registry_service().open_registry_for_model(classification_model_path)
    except (DinoV2ManifestError, CheckpointValidationError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return action(registry)
    except RegistryEntryNotFound as exc:
        raise HTTPException(status_code=404, detail="DINOv2 registration not found") from exc
    except RegistrationConditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        registry.close()


def build_registry_catalog(checkpoint: Any, registry: Any, *, feedback: Any | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    counts = tuple(int(v) for v in checkpoint.prototypes_per_class)
    raw = checkpoint.prototype_class_indices
    indices = [int(v) for v in (raw.tolist() if hasattr(raw, "tolist") else raw)]
    for index, species in enumerate(checkpoint.classes):
        base_indices = [i for i, class_index in enumerate(indices) if class_index == index]
        clusters = [{
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
        } for local_index, prototype_index in enumerate(base_indices)]
        feedback_event_count = 0
        feedback_prototype_count = 0
        learning_status = None
        if feedback is not None:
            state = feedback.learning_state(species)
            if state.positive_events > 0:
                feedback_event_count = int(state.positive_events)
                feedback_prototype_count = int(state.prototype_count)
                learning_status = str(state.status)
                clusters.extend(feedback.cluster_details(species, checkpoint.feature_center))
        result.append({
            "id": -(index + 1),
            "candidate_number": index + 1,
            "status": "checkpoint",
            "candidate_kind": "checkpoint",
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
        })
    for entry in registry.list():
        data = entry.as_dict()
        data.setdefault("feedback_event_count", 0)
        data.setdefault("feedback_prototype_count", 0)
        data.setdefault("learning_status", entry.status)
        data.setdefault("clusters", [])
        result.append(data)
    return result


def _catalog_with_feedback(checkpoint: Any, registry: Any) -> list[dict[str, Any]]:
    from .feedback import HumanFeedbackStore, feedback_path_for_registry
    path = feedback_path_for_registry(registry.path)
    if not path.exists():
        return build_registry_catalog(checkpoint, registry)
    feedback = HumanFeedbackStore(
        path,
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=checkpoint.rejection,
        prototype_norm_power=checkpoint.prototype_norm_power,
    )
    try:
        return build_registry_catalog(checkpoint, registry, feedback=feedback)
    finally:
        feedback.close()


def _register_with_duplicate_guard(registry: Any, registration_id: int, classification_model_path: str):
    from .classifier import DinoV2Classifier
    checkpoint = _registry_service().load_checkpoint_for_model(classification_model_path)
    classifier = DinoV2Classifier(checkpoint, registry=registry)

    def matcher(embedding):
        prediction = classifier.classify_features(embedding[None, :])[0]
        return prediction.species if prediction.accepted else None

    return registry.register(registration_id, formal_matcher=matcher).as_dict()


def dinov2_registry_router() -> APIRouter:
    router = APIRouter(prefix="/api/dinov2", tags=["dinov2"])

    @router.get("/registry", response_model=list[DinoV2RegistryEntryResponse])
    def list_registry(classification_model_path: str = Query(..., min_length=1), status: str | None = Query(default=None)):
        return _run_with_registry(classification_model_path, lambda registry: [e.as_dict() for e in registry.list(status=status)])

    @router.get("/registry/catalog", response_model=list[DinoV2RegistryEntryResponse])
    def catalog(classification_model_path: str = Query(..., min_length=1)):
        checkpoint = _registry_service().load_checkpoint_for_model(classification_model_path)
        return _run_with_registry(classification_model_path, lambda registry: _catalog_with_feedback(checkpoint, registry))

    @router.delete("/registry/candidates")
    def clear_candidates(classification_model_path: str = Query(..., min_length=1)):
        return {"deleted": int(_run_with_registry(classification_model_path, lambda registry: registry.delete_candidates()))}

    @router.get("/registry/{registration_id}", response_model=DinoV2RegistryEntryResponse)
    def get_entry(registration_id: int, classification_model_path: str = Query(..., min_length=1)):
        return _run_with_registry(classification_model_path, lambda registry: registry.get(registration_id).as_dict())

    @router.get("/registry/{registration_id}/clusters", response_model=list[DinoV2RegistryClusterResponse])
    def clusters(registration_id: int, classification_model_path: str = Query(..., min_length=1)):
        return _run_with_registry(classification_model_path, lambda registry: registry.cluster_details(registration_id))

    @router.get("/registry/{registration_id}/events", response_model=list[DinoV2RegistryEventResponse])
    def events(registration_id: int, classification_model_path: str = Query(..., min_length=1)):
        return _run_with_registry(classification_model_path, lambda registry: registry.list_events(registration_id))

    @router.get("/registry/{registration_id}/events/{event_id}/example")
    def example(registration_id: int, event_id: int, classification_model_path: str = Query(..., min_length=1)):
        try:
            content = _run_with_registry(classification_model_path, lambda registry: _registry_service().render_registry_example(registry, registration_id, event_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})

    @router.delete("/registry/{registration_id}")
    def delete_entry(registration_id: int, classification_model_path: str = Query(..., min_length=1)):
        _run_with_registry(classification_model_path, lambda registry: registry.delete(registration_id))
        return {"deleted": True, "registration_id": registration_id}

    @router.patch("/registry/{registration_id}/identity", response_model=DinoV2RegistryEntryResponse)
    def identity(registration_id: int, request: DinoV2IdentityUpdateRequest):
        return _run_with_registry(request.classification_model_path, lambda registry: registry.set_identity(registration_id, common_name=request.common_name, scientific_name=request.scientific_name).as_dict())

    @router.post("/registry/{registration_id}/merge-checkpoint")
    def merge_checkpoint(registration_id: int, request: _MergeCheckpointRequest):
        return _run_with_registry(request.classification_model_path, lambda registry: _registry_service().merge_registry_candidate_into_checkpoint(registry, request.classification_model_path, registration_id, request.checkpoint_species))

    @router.post("/registry/{registration_id}/merge-candidate", response_model=DinoV2RegistryEntryResponse)
    def merge_candidate(registration_id: int, request: _MergeCandidateRequest):
        return _run_with_registry(request.classification_model_path, lambda registry: registry.merge_candidate_into(registration_id, request.target_registration_id).as_dict())

    @router.post("/registry/{registration_id}/empty")
    def empty(registration_id: int, request: DinoV2RegisterRequest):
        return _run_with_registry(request.classification_model_path, lambda registry: _registry_service().discard_registry_candidate_as_empty(registry, request.classification_model_path, registration_id))

    @router.post("/registry/{registration_id}/register", response_model=DinoV2RegistryEntryResponse)
    def register(registration_id: int, request: DinoV2RegisterRequest):
        return _run_with_registry(request.classification_model_path, lambda registry: _register_with_duplicate_guard(registry, registration_id, request.classification_model_path))

    return router
