"""FastAPI entrypoint for the Flutter-based Neri client."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.settings_manager import SettingsManager

from . import __version__
from .models import (
    CreateJobRequest,
    DetectionItem,
    HealthResponse,
    JobSummary,
    SettingsUpdateRequest,
    SettingsResponse,
    ValidationExportRequest,
    ValidationExportResponse,
    ValidationMarkRequest,
)
from .services import (
    JobNotFoundError,
    ProcessingJobManager,
    classification_model_directory,
    detect_gpu_available,
    export_validation_data,
    list_available_classification_models,
    list_available_models,
    load_species_types,
    mark_validation_item,
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


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple readiness response for the frontend."""

    return HealthResponse(version=__version__)


@app.get("/api/settings", response_model=SettingsResponse)
def settings() -> SettingsResponse:
    """Return Neri runtime settings and supported media formats."""

    stored_settings = settings_manager.load_settings() or {}
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
        settings=stored_settings,
    )


@app.put("/api/settings", response_model=SettingsResponse)
def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Persist advanced settings and return the refreshed settings snapshot."""

    current_settings = settings_manager.load_settings() or {}
    current_settings.update(request.settings)
    if not settings_manager.save_settings(current_settings):
        raise HTTPException(status_code=500, detail="Failed to save settings")
    return settings()


@app.post("/api/jobs", response_model=JobSummary, status_code=202)
def create_job(request: CreateJobRequest) -> JobSummary:
    """Start an asynchronous folder processing job."""

    return job_manager.create_job(request)


@app.get("/api/preview", response_model=list[DetectionItem])
def preview(input_path: str = Query(..., min_length=1), output_dir: str | None = None) -> list[DetectionItem]:
    """List previewable media and cached DB detections without starting a job."""

    try:
        return preview_media_items(input_path, output_dir)
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
def list_jobs() -> list[JobSummary]:
    """List all jobs known to this backend process."""

    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    """Fetch progress and results for a single job."""

    try:
        return job_manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
