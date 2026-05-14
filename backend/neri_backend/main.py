"""FastAPI entrypoint for the Flutter-based Neri client."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.settings_manager import SettingsManager

from . import __version__
from .models import CreateJobRequest, HealthResponse, JobSummary, SettingsResponse
from .services import JobNotFoundError, ProcessingJobManager

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

    return SettingsResponse(
        app_title=APP_TITLE,
        app_version=APP_VERSION,
        supported_image_extensions=list(SUPPORTED_IMAGE_EXTENSIONS),
        supported_video_extensions=list(SUPPORTED_VIDEO_EXTENSIONS),
        settings=settings_manager.load_settings() or {},
    )


@app.post("/api/jobs", response_model=JobSummary, status_code=202)
def create_job(request: CreateJobRequest) -> JobSummary:
    """Start an asynchronous folder processing job."""

    return job_manager.create_job(request)


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
