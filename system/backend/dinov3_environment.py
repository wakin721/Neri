"""FastAPI environment-maintenance surface for the optional DINOv3 component."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from system.dinov3.component import dinov3_component_status

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
    message: str = ""


class DinoV3MaintenanceStartResponse(BaseModel):
    operation: str = ""
    state: str = "starting"
    message: str = ""
    progress: int = 0
    status_path: str | None = None
    log_path: str | None = None
    maintenance_pid: int | None = None


def dinov3_environment_router() -> APIRouter:
    router = APIRouter(prefix="/api/environment", tags=["environment", "dinov3"])

    @router.get("/dinov3-status", response_model=DinoV3ComponentStatusResponse)
    def status() -> DinoV3ComponentStatusResponse:
        return DinoV3ComponentStatusResponse(**dinov3_component_status())

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
