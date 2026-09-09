"""FastAPI environment-maintenance surface for the optional DINOv3 component."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from system.dinov3.checkpoint import CheckpointValidationError, load_checkpoint
from system.dinov3.component import dinov3_component_paths, dinov3_component_status

from .dinov3_maintenance import start_dinov3_install, start_dinov3_remove
from .maintenance import schedule_backend_shutdown


class DinoV3InstallRequest(BaseModel):
    env_choice: str = Field(default="自动检测", min_length=1, max_length=64)
    package_source: str = Field(default="auto", min_length=1, max_length=32)


class DinoV3ComponentStatusResponse(BaseModel):
    installed: bool
    healthy: bool
    architecture: str
    component_version: int
    source_commit: str
    classifier_filename: str | None = None
    classifier_fingerprint: str | None = None
    classifier_head_type: str | None = None
    selection_k: int | None = None
    message: str = ""


class DinoV3MaintenanceStartResponse(BaseModel):
    operation: str = ""
    state: str = "starting"
    message: str = ""
    progress: int = 0
    status_path: str | None = None
    log_path: str | None = None
    maintenance_pid: int | None = None


def _component_status_payload() -> dict[str, object]:
    payload = dict(dinov3_component_status())
    payload.setdefault("classifier_filename", None)
    payload.setdefault("classifier_fingerprint", None)
    payload.setdefault("classifier_head_type", None)
    payload.setdefault("selection_k", None)
    if payload.get("healthy") is not True:
        return payload
    if all(
        payload.get(key) is not None
        for key in (
            "classifier_filename",
            "classifier_fingerprint",
            "classifier_head_type",
            "selection_k",
        )
    ):
        return payload
    paths = dinov3_component_paths()
    try:
        checkpoint = load_checkpoint(paths.classifier)
    except (CheckpointValidationError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        payload["healthy"] = False
        payload["message"] = f"DINOv3 checkpoint 状态读取失败: {exc}"
        return payload
    payload["classifier_filename"] = paths.classifier.name
    payload["classifier_fingerprint"] = checkpoint.fingerprint
    payload["classifier_head_type"] = checkpoint.head_type
    payload["selection_k"] = checkpoint.selection_k
    return payload


def dinov3_environment_router() -> APIRouter:
    router = APIRouter(prefix="/api/environment", tags=["environment", "dinov3"])

    @router.get("/dinov3-status", response_model=DinoV3ComponentStatusResponse)
    def status() -> DinoV3ComponentStatusResponse:
        return DinoV3ComponentStatusResponse(**_component_status_payload())

    @router.post("/install-dinov3", response_model=DinoV3MaintenanceStartResponse, status_code=202)
    def install(
        request: DinoV3InstallRequest,
        background_tasks: BackgroundTasks,
    ) -> DinoV3MaintenanceStartResponse:
        try:
            result = start_dinov3_install(request.env_choice, request.package_source)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(schedule_backend_shutdown)
        return DinoV3MaintenanceStartResponse(**result)

    @router.post("/remove-dinov3", response_model=DinoV3MaintenanceStartResponse, status_code=202)
    def remove(background_tasks: BackgroundTasks) -> DinoV3MaintenanceStartResponse:
        try:
            result = start_dinov3_remove()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(schedule_backend_shutdown)
        return DinoV3MaintenanceStartResponse(**result)

    return router
