"""FastAPI environment-maintenance surface for the optional DINOv2 component."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from system.dinov2.checkpoint import CheckpointValidationError, load_checkpoint
from system.dinov2.component import dinov2_component_paths, dinov2_component_status

from .dinov2_maintenance import start_dinov2_install, start_dinov2_remove
from .maintenance import schedule_backend_shutdown


class DinoV2InstallRequest(BaseModel):
    env_choice: str = Field(default="自动检测", min_length=1, max_length=64)
    package_source: str = Field(default="auto", min_length=1, max_length=32)


class DinoV2ComponentStatusResponse(BaseModel):
    installed: bool
    healthy: bool
    architecture: str
    component_version: int
    classifier_filename: str | None = None
    classifier_fingerprint: str | None = None
    classifier_head_type: str | None = None
    prototype_count: int | None = None
    message: str = ""


class DinoV2MaintenanceStartResponse(BaseModel):
    operation: str = ""
    state: str = "starting"
    message: str = ""
    progress: int = 0
    status_path: str | None = None
    log_path: str | None = None
    maintenance_pid: int | None = None


def _component_status_payload() -> dict[str, object]:
    payload = dict(dinov2_component_status())
    for key in (
        "classifier_filename",
        "classifier_fingerprint",
        "classifier_head_type",
        "prototype_count",
    ):
        payload.setdefault(key, None)
    if payload.get("healthy") is not True:
        return payload
    if all(
        payload.get(key) is not None
        for key in (
            "classifier_filename",
            "classifier_fingerprint",
            "classifier_head_type",
            "prototype_count",
        )
    ):
        return payload
    paths = dinov2_component_paths()
    try:
        checkpoint = load_checkpoint(paths.classifier)
    except (
        CheckpointValidationError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        payload["healthy"] = False
        payload["message"] = f"DINOv2 checkpoint 状态读取失败: {exc}"
        return payload
    payload["classifier_filename"] = paths.classifier.name
    payload["classifier_fingerprint"] = checkpoint.fingerprint
    payload["classifier_head_type"] = checkpoint.head_type
    payload["prototype_count"] = int(checkpoint.prototypes.shape[0])
    return payload


def dinov2_environment_router() -> APIRouter:
    router = APIRouter(prefix="/api/environment", tags=["environment", "dinov2"])

    @router.get("/dinov2-status", response_model=DinoV2ComponentStatusResponse)
    def status() -> DinoV2ComponentStatusResponse:
        return DinoV2ComponentStatusResponse(**_component_status_payload())

    @router.post(
        "/install-dinov2",
        response_model=DinoV2MaintenanceStartResponse,
        status_code=202,
    )
    def install(
        request: DinoV2InstallRequest,
        background_tasks: BackgroundTasks,
    ) -> DinoV2MaintenanceStartResponse:
        try:
            result = start_dinov2_install(
                request.env_choice,
                request.package_source,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(schedule_backend_shutdown)
        return DinoV2MaintenanceStartResponse(**result)

    @router.post(
        "/remove-dinov2",
        response_model=DinoV2MaintenanceStartResponse,
        status_code=202,
    )
    def remove(
        background_tasks: BackgroundTasks,
    ) -> DinoV2MaintenanceStartResponse:
        try:
            result = start_dinov2_remove()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(schedule_backend_shutdown)
        return DinoV2MaintenanceStartResponse(**result)

    return router
