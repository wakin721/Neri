"""FastAPI entrypoint for the Flutter-based Neri client."""

import ctypes
import os
import threading
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.settings_manager import SettingsManager

from . import __version__
from .models import (
    CreateJobRequest,
    DetectionItem,
    HealthResponse,
    InstallPytorchRequest,
    InstallYoloDependenciesRequest,
    ModelClassInfo,
    MaintenanceStartResponse,
    MaintenanceStatusResponse,
    PytorchInstallPlanResponse,
    JobSummary,
    ReinstallPackageRequest,
    SettingsResponse,
    SettingsUpdateRequest,
    ValidationBatchMarkRequest,
    ValidationExportRequest,
    ValidationExportResponse,
    ValidationMarkRequest,
)
from .maintenance import (
    read_maintenance_status,
    resolve_pytorch_install_plan,
    schedule_backend_shutdown,
    start_package_reinstall,
    start_pytorch_install,
    start_yolo_dependencies_install,
)
from .services import (
    JobNotFoundError,
    ProcessingJobManager,
    classification_model_directory,
    detect_gpu_available,
    export_validation_data,
    list_available_classification_models,
    list_available_models,
    list_model_classes,
    load_species_types,
    mark_validation_items,
    mark_validation_item,
    missing_yolo_dependencies,
    model_directory,
    preview_media_item,
    preview_media_items,
)

app = FastAPI(
    title="Neri API",
    version=__version__,
    description="Python backend that exposes Neri infrared-camera processing capabilities to a Flutter Material 3 UI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = ProcessingJobManager()
settings_manager = SettingsManager()


def _start_parent_watchdog() -> None:
    """Exit this backend if the owning Flutter process disappears."""

    parent_pid_text = os.environ.get("NERI_PARENT_PID")
    if not parent_pid_text:
        return
    try:
        parent_pid = int(parent_pid_text)
    except ValueError:
        return
    if parent_pid <= 0:
        return

    def _watch_windows_parent() -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        synchronize = 0x00100000
        infinite = 0xFFFFFFFF
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            # ERROR_INVALID_PARAMETER means the parent PID no longer exists.
            # Other failures, such as access-denied across integrity levels,
            # should not make the backend immediately exit after startup.
            if ctypes.get_last_error() == 87:
                os._exit(0)
            return
        try:
            kernel32.WaitForSingleObject(handle, infinite)
        finally:
            kernel32.CloseHandle(handle)
        os._exit(0)

    def _watch_posix_parent() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except OSError:
                os._exit(0)
            time.sleep(1)

    target = _watch_windows_parent if os.name == "nt" else _watch_posix_parent
    thread = threading.Thread(
        target=target,
        name="neri-parent-watchdog",
        daemon=True,
    )
    thread.start()


_start_parent_watchdog()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple readiness response for the frontend."""

    return HealthResponse(version=__version__)


@app.post("/api/shutdown")
def shutdown(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Schedule the local backend to exit after the response is flushed."""

    background_tasks.add_task(schedule_backend_shutdown, 0.2)
    return {"status": "shutting_down"}


@app.get("/api/settings", response_model=SettingsResponse)
def settings() -> SettingsResponse:
    """Return Neri runtime settings and supported media formats."""

    stored_settings = settings_manager.load_settings() or {}
    quick_mark_settings = settings_manager.load_quick_mark_species() or {}
    if "quick_mark_list" not in stored_settings and isinstance(quick_mark_settings.get("list"), list):
        stored_settings["quick_mark_list"] = quick_mark_settings["list"]
    if "auto_sort" not in stored_settings and isinstance(quick_mark_settings.get("auto"), bool):
        stored_settings["auto_sort"] = quick_mark_settings["auto"]
    if "quick_mark_recent_history" not in stored_settings and isinstance(
        quick_mark_settings.get("recent_history"), list
    ):
        stored_settings["quick_mark_recent_history"] = quick_mark_settings["recent_history"]
    excluded_quick_mark_keys = {"list", "list_auto", "auto", "recent_history"}
    quick_mark_counts = {
        key: int(value)
        for key, value in quick_mark_settings.items()
        if key not in excluded_quick_mark_keys and isinstance(value, (int, float))
    }
    if "quick_mark_usage_counts" not in stored_settings and quick_mark_counts:
        stored_settings["quick_mark_usage_counts"] = quick_mark_counts
    available_models = list_available_models()
    available_classification_models = list_available_classification_models()
    saved_model = stored_settings.get("selected_model")
    saved_classification_model = stored_settings.get("selected_classification_model") or stored_settings.get("selected_cls_model")
    selected_model = None
    if isinstance(saved_model, str):
        selected_model = next(
            (model.path for model in available_models if model.name == saved_model or model.path == saved_model),
            None,
        )
    selected_model = selected_model or (available_models[0].path if available_models else None)

    selected_classification_model = None
    if isinstance(saved_classification_model, str) and saved_classification_model:
        selected_classification_model = next(
            (
                model.path
                for model in available_classification_models
                if model.name == saved_classification_model or model.path == saved_classification_model
            ),
            None,
        )

    return SettingsResponse(
        app_title=APP_TITLE,
        app_version=APP_VERSION,
        supported_image_extensions=list(SUPPORTED_IMAGE_EXTENSIONS),
        supported_video_extensions=list(SUPPORTED_VIDEO_EXTENSIONS),
        model_directory=str(model_directory()),
        classification_model_directory=str(classification_model_directory()),
        available_models=available_models,
        available_classification_models=available_classification_models,
        selected_model=selected_model,
        selected_classification_model=selected_classification_model,
        species_types=load_species_types(),
        gpu_available=detect_gpu_available(),
        missing_yolo_dependencies=missing_yolo_dependencies(),
        settings=stored_settings,
    )


@app.get("/api/models/classes", response_model=list[ModelClassInfo])
def model_classes(model_path: str = Query(..., min_length=1)) -> list[ModelClassInfo]:
    """Return the classes exposed by a selected YOLO model."""

    try:
        return list_model_classes(model_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/settings", response_model=SettingsResponse)
def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Persist advanced settings and return the refreshed settings snapshot."""

    current_settings = settings_manager.load_settings() or {}
    current_settings.update(request.settings)
    if not settings_manager.save_settings(current_settings):
        raise HTTPException(status_code=500, detail="Failed to save settings")
    quick_mark_settings = settings_manager.load_quick_mark_species() or {}
    quick_mark_list = current_settings.get("quick_mark_list")
    if isinstance(quick_mark_list, list):
        quick_mark_settings["list"] = quick_mark_list
    if isinstance(current_settings.get("auto_sort"), bool):
        quick_mark_settings["auto"] = current_settings["auto_sort"]
    recent_history = current_settings.get("quick_mark_recent_history")
    if isinstance(recent_history, list):
        quick_mark_settings["recent_history"] = recent_history
    usage_counts = current_settings.get("quick_mark_usage_counts")
    if isinstance(usage_counts, dict):
        for key, value in usage_counts.items():
            if isinstance(value, (int, float)):
                quick_mark_settings[str(key)] = int(value)
    settings_manager.save_quick_mark_species(quick_mark_settings)
    return settings()


@app.get("/api/environment/maintenance-status", response_model=MaintenanceStatusResponse)
def maintenance_status() -> MaintenanceStatusResponse:
    """Return the last environment-maintenance status."""

    return MaintenanceStatusResponse(**read_maintenance_status())


@app.get("/api/environment/pytorch-install-plan", response_model=PytorchInstallPlanResponse)
def pytorch_install_plan(env_choice: str = Query("自动检测", min_length=1)) -> PytorchInstallPlanResponse:
    """Resolve PyTorch target and Intel driver preflight data."""

    try:
        plan = resolve_pytorch_install_plan(env_choice)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PytorchInstallPlanResponse(**plan)


@app.post("/api/environment/install-pytorch", response_model=MaintenanceStartResponse, status_code=202)
def install_pytorch(request: InstallPytorchRequest, background_tasks: BackgroundTasks) -> MaintenanceStartResponse:
    """Start PyTorch installation, then restart the Python backend."""

    try:
        status = start_pytorch_install(
            request.env_choice,
            request.package_source,
            install_intel_driver=request.install_intel_driver,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(schedule_backend_shutdown)
    return MaintenanceStartResponse(operation=str(status["operation"]), message=str(status["message"]))


@app.post("/api/environment/install-yolo-dependencies", response_model=MaintenanceStartResponse, status_code=202)
def install_yolo_dependencies(
    request: InstallYoloDependenciesRequest,
    background_tasks: BackgroundTasks,
) -> MaintenanceStartResponse:
    """Install PyTorch and ultralytics, then restart the Python backend."""

    try:
        status = start_yolo_dependencies_install(
            request.env_choice,
            request.package_source,
            install_intel_driver=request.install_intel_driver,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(schedule_backend_shutdown)
    return MaintenanceStartResponse(operation=str(status["operation"]), message=str(status["message"]))


@app.post("/api/environment/reinstall-package", response_model=MaintenanceStartResponse, status_code=202)
def reinstall_package(
    request: ReinstallPackageRequest,
    background_tasks: BackgroundTasks,
) -> MaintenanceStartResponse:
    """Force-reinstall one Python package, then restart the Python backend."""

    try:
        status = start_package_reinstall(request.package, request.package_source)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(schedule_backend_shutdown)
    return MaintenanceStartResponse(operation=str(status["operation"]), message=str(status["message"]))


@app.post("/api/jobs", response_model=JobSummary, status_code=202)
def create_job(request: CreateJobRequest) -> JobSummary:
    """Start an asynchronous folder processing job."""

    if request.options.enable_detection:
        missing = missing_yolo_dependencies()
        if missing:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "missing_yolo_dependencies",
                    "message": "缺少 YOLO 处理依赖，请先安装依赖后再开始识别。",
                    "missing": missing,
                },
            )
    return job_manager.create_job(request)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobSummary)
def cancel_job(job_id: str) -> JobSummary:
    """Request cancellation for a queued or running processing job."""

    try:
        return job_manager.cancel_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/api/jobs/{job_id}/resume", response_model=JobSummary, status_code=202)
def resume_job(job_id: str) -> JobSummary:
    """Resume an interrupted processing job in place."""

    try:
        return job_manager.resume_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    """Delete one job from the progress list."""

    try:
        job_manager.delete_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.delete("/api/jobs", status_code=204)
def clear_jobs() -> None:
    """Clear all jobs from the progress list."""

    job_manager.clear_jobs()


@app.get("/api/preview", response_model=list[DetectionItem])
def preview(
    input_path: str = Query(..., min_length=1),
    output_dir: str | None = None,
    include_cached: bool = True,
) -> list[DetectionItem]:
    """List previewable media and cached DB detections without starting a job."""

    try:
        return preview_media_items(input_path, output_dir, include_cached=include_cached)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/preview/item", response_model=DetectionItem)
def preview_item(
    file_path: str = Query(..., min_length=1),
    input_path: str | None = None,
    output_dir: str | None = None,
) -> DetectionItem:
    """Return full metadata for one preview item."""

    try:
        return preview_media_item(file_path, input_path, output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/validation/mark", response_model=DetectionItem)
def mark_validation(request: ValidationMarkRequest) -> DetectionItem:
    """Persist one manual validation decision and return the refreshed item."""

    try:
        return mark_validation_item(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/validation/mark/batch", response_model=list[DetectionItem])
def mark_validation_batch(request: ValidationBatchMarkRequest) -> list[DetectionItem]:
    """Persist multiple manual validation decisions in one batch."""

    try:
        return mark_validation_items(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/validation/export", response_model=ValidationExportResponse)
def export_validation(request: ValidationExportRequest) -> ValidationExportResponse:
    """Export validation data using system.data_processor."""

    try:
        return export_validation_data(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs", response_model=list[JobSummary])
def list_jobs(include_results: bool = Query(True)) -> list[JobSummary]:
    """List all jobs known to this backend process."""

    jobs = job_manager.list_jobs()
    if include_results:
        return jobs
    return [job.model_copy(update={"results": []}, deep=True) for job in jobs]


@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    """Fetch progress and results for a single job."""

    try:
        return job_manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
