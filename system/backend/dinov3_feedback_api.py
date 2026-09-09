"""FastAPI routes for explicit DINOv3 box-level human feedback."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .dinov3_feedback_service import apply_box_feedback, revert_feedback_operation
from .models import DinoV3BoxFeedbackRequest, DinoV3FeedbackRevertRequest


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
