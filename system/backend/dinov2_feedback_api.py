"""FastAPI routes for explicit DINOv2 human feedback."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field


class DinoV2BoxFeedbackRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    observation_id: str = Field(..., min_length=1)
    action: str
    species_name: str | None = None
    feedback_operation_id: str = Field(..., min_length=1)


class DinoV2SelectionFeedbackRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    observation_id: str = Field(..., min_length=1)
    species_name: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


class DinoV2FeedbackRevertRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


def _apply_selection_feedback(request: DinoV2SelectionFeedbackRequest) -> dict:
    from .dinov2_feedback_service import (
        _affected_learning_species,
        _assign_registry_species,
        _open_feedback_state,
        _path_identity,
    )

    feedback, feature_center = _open_feedback_state(request.classification_model_path)
    try:
        observation = feedback.get_observation(request.observation_id)
        if _path_identity(observation.source_path) != _path_identity(request.file_path):
            raise ValueError("DINOv2 observation does not belong to the selected file")
        confirmed = request.species_name.strip()
        if not confirmed:
            raise ValueError("species_name is required")
        if confirmed in feedback.checkpoint_classes:
            record = feedback.record_feedback(
                request.observation_id,
                operation_id=request.feedback_operation_id,
                action="update",
                confirmed_species=confirmed,
            )
            affected = _affected_learning_species(record)
            for species in sorted(affected):
                feedback.recompute_species(species, feature_center)
            registry_id = None
        else:
            entry = _assign_registry_species(
                feedback,
                request.classification_model_path,
                observation,
                operation_id=request.feedback_operation_id,
                confirmed_species=confirmed,
            )
            affected = set()
            registry_id = entry.id
        return {
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
            "registry_id": registry_id,
        }
    finally:
        feedback.close()


def dinov2_feedback_router() -> APIRouter:
    router = APIRouter(prefix="/api/dinov2", tags=["dinov2", "feedback"])

    @router.post("/feedback/box")
    def box(request: DinoV2BoxFeedbackRequest):
        from .dinov2_feedback_service import apply_box_feedback
        try:
            return apply_box_feedback(request)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/feedback/selection")
    def selection(request: DinoV2SelectionFeedbackRequest):
        try:
            return _apply_selection_feedback(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/feedback/observations/{observation_id}/explain")
    def explain(observation_id: str, classification_model_path: str = Query(..., min_length=1)):
        from .dinov2_feedback_service import explain_feedback_observation
        try:
            return explain_feedback_observation(classification_model_path, observation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/feedback/observations/{observation_id}/example")
    def example(observation_id: str, classification_model_path: str = Query(..., min_length=1)):
        from .dinov2_feedback_service import render_feedback_observation_example
        try:
            content = render_feedback_observation_example(classification_model_path, observation_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})

    @router.post("/feedback/revert")
    def revert(request: DinoV2FeedbackRevertRequest):
        from .dinov2_feedback_service import revert_feedback_operation
        try:
            return revert_feedback_operation(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
