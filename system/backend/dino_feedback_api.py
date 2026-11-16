"""FastAPI routes for DINOv3 joint-model human feedback."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field


class DinoV3BoxFeedbackRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    observation_id: str = Field(..., min_length=1)
    action: str
    species_name: str | None = None
    feedback_operation_id: str = Field(..., min_length=1)


class DinoV3FeedbackRevertRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


class DinoV3LearningRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)


class DinoV3CalibrationRequest(DinoV3LearningRequest):
    calibration_path: str = Field(..., min_length=1)


class DinoRejectionClustersRequest(BaseModel):
    classification_model_path: str = Field(..., min_length=1)
    observation_ids: list[str] = Field(..., max_length=10000)


class DinoClusterFeedbackRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    classification_model_path: str = Field(..., min_length=1)
    cluster_id: str = Field(..., min_length=1)
    observation_ids: list[str] = Field(..., min_length=1, max_length=10000)
    species_name: str = Field(..., min_length=1)
    feedback_operation_id: str = Field(..., min_length=1)


def dino_feedback_router() -> APIRouter:
    router = APIRouter(prefix="/api/dino", tags=["dino", "feedback"])

    def learning_call(callback, *args):
        try:
            return callback(*args)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, OSError, sqlite3.OperationalError) as exc:
            raise HTTPException(status_code=409, detail=f"模型更新未完成：{exc}") from exc

    @router.post("/learning/status")
    def learning_status(request: DinoV3LearningRequest):
        from .dinov3_learning import learning_status as status
        return learning_call(status, request.classification_model_path)

    @router.post("/learning/update")
    def learning_update(request: DinoV3LearningRequest):
        from .dinov3_learning import update_model
        return learning_call(update_model, request.classification_model_path)

    @router.post("/learning/discard")
    def learning_discard(request: DinoV3LearningRequest):
        from .dinov3_learning import discard_pending
        return learning_call(discard_pending, request.classification_model_path)

    @router.post("/learning/retry")
    def learning_retry(request: DinoV3LearningRequest):
        from .dinov3_learning import schedule_learning, learning_status
        learning_call(schedule_learning, request.classification_model_path)
        return learning_call(learning_status, request.classification_model_path)

    @router.post("/learning/calibration")
    def learning_calibration(request: DinoV3CalibrationRequest):
        from .dinov3_learning import import_calibration
        return learning_call(import_calibration, request.classification_model_path, request.calibration_path)

    @router.post("/feedback/cluster")
    def cluster(_request: DinoClusterFeedbackRequest):
        raise HTTPException(
            status_code=400,
            detail="联合模型不支持按簇确认，请逐框人工校验。",
        )

    @router.post("/feedback/rejection-clusters")
    def rejection_clusters(_request: DinoRejectionClustersRequest):
        return {}

    @router.post("/feedback/box")
    def box(request: DinoV3BoxFeedbackRequest):
        from .dino_feedback_service import apply_box_feedback

        try:
            return apply_box_feedback(request)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/feedback/observations/{observation_id}/explain")
    def explain(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        from .dino_feedback_service import explain_feedback_observation

        try:
            return explain_feedback_observation(
                classification_model_path,
                observation_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/feedback/observations/{observation_id}/example")
    def example(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        from .dino_feedback_service import render_feedback_observation_example

        try:
            content = render_feedback_observation_example(
                classification_model_path,
                observation_id,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )

    @router.post("/feedback/revert")
    def revert(request: DinoV3FeedbackRevertRequest):
        from .dino_feedback_service import revert_feedback_operation

        try:
            return revert_feedback_operation(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
