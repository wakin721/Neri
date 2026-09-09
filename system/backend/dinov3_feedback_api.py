"""FastAPI routes for explicit DINOv3 box-level human feedback."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .dinov3_feedback_service import apply_box_feedback, revert_feedback_operation


class DinoV3BoxFeedbackRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    observation_id: str = Field(..., min_length=1)
    action: Literal["correct", "update", "empty", "unverified"]
    species_name: str | None = None
    feedback_operation_id: str = Field(..., min_length=1)


class DinoV3FeedbackRevertRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


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
