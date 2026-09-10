"""FastAPI routes for explicit DINOv3 box-level human feedback."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from .dinov3_feedback_service import (
    apply_box_feedback,
    explain_feedback_observation,
    render_feedback_observation_example,
    revert_feedback_operation,
)
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
