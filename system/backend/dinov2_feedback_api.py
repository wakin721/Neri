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


class DinoV2FeedbackRevertRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


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
