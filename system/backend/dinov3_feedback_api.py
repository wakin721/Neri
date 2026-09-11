"""FastAPI routes for explicit DINOv3 human feedback."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .dinov3_feedback_service import (
    _affected_learning_species,
    _assign_registry_species,
    _open_feedback_state,
    _path_identity,
    apply_box_feedback,
    explain_feedback_observation,
    render_feedback_observation_example,
    revert_feedback_operation,
)
from .models import DinoV3BoxFeedbackRequest, DinoV3FeedbackRevertRequest


class DinoV3SelectedObservationFeedbackRequest(BaseModel):
    """Learning-only feedback for one explicitly selected DINOv3 observation."""

    file_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    observation_id: str = Field(..., min_length=1)
    species_name: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


def apply_selected_observation_feedback(
    request: DinoV3SelectedObservationFeedbackRequest,
) -> dict[str, object]:
    """Learn one selected observation without rewriting ecological validation data."""
    feedback, feature_center = _open_feedback_state(request.classification_model_path)
    try:
        observation = feedback.get_observation(request.observation_id)
        if _path_identity(observation.source_path) != _path_identity(request.file_path):
            raise ValueError("DINOv3 observation does not belong to the selected file")

        species_name = request.species_name.strip()
        if not species_name:
            raise ValueError("species_name is required")

        if species_name not in feedback.checkpoint_classes:
            registry_entry = _assign_registry_species(
                feedback,
                request.classification_model_path,
                observation,
                operation_id=request.feedback_operation_id,
                confirmed_species=species_name,
            )
            return {
                "operation_id": request.feedback_operation_id,
                "affected_species": [],
                "registry_id": registry_entry.id,
            }

        record = feedback.record_feedback(
            observation.id,
            operation_id=request.feedback_operation_id,
            action="update",
            confirmed_species=species_name,
        )
        affected = _affected_learning_species(record)
        for species in sorted(affected):
            feedback.recompute_species(species, feature_center)
        return {
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
            "registry_id": None,
        }
    finally:
        feedback.close()


def dinov3_feedback_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/dinov3/feedback",
        tags=["dinov3-feedback"],
    )

    @router.post("/box")
    def box_feedback(request: DinoV3BoxFeedbackRequest):
        try:
            return apply_box_feedback(request)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="DINOv3 observation not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/selection")
    def selected_observation_feedback(
        request: DinoV3SelectedObservationFeedbackRequest,
    ):
        try:
            return apply_selected_observation_feedback(request)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="DINOv3 observation not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/observations/{observation_id}/explain")
    def observation_explanation(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        try:
            return explain_feedback_observation(
                classification_model_path,
                observation_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="DINOv3 observation not found") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/observations/{observation_id}/example")
    def observation_example(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        try:
            content = render_feedback_observation_example(
                classification_model_path,
                observation_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="DINOv3 observation not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )

    @router.post("/revert")
    def revert_feedback(request: DinoV3FeedbackRevertRequest):
        try:
            return revert_feedback_operation(request)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="DINOv3 feedback operation not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
