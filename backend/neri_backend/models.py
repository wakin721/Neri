"""API schemas for the Neri backend."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Lifecycle states exposed to the Flutter frontend."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthResponse(BaseModel):
    """Runtime status for the backend process."""

    app: str = "Neri API"
    version: str
    status: Literal["ok"] = "ok"


class SettingsResponse(BaseModel):
    """Subset of desktop settings useful to the mobile/desktop frontend."""

    app_title: str
    app_version: str
    supported_image_extensions: list[str]
    supported_video_extensions: list[str]
    settings: dict[str, Any] = Field(default_factory=dict)


class ProcessingOptions(BaseModel):
    """Options that mirror the current desktop advanced processing controls."""

    model_path: str | None = Field(default=None, description="YOLO .pt model path. Uses the project default when empty.")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    use_fp16: bool = False
    enable_detection: bool = Field(
        default=False,
        description="When false, the job indexes supported files and EXIF metadata without loading YOLO.",
    )


class CreateJobRequest(BaseModel):
    """Request body for creating a processing job."""

    input_dir: str = Field(..., min_length=1)
    output_dir: str | None = None
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)


class DetectionItem(BaseModel):
    """A single indexed file and optional detection summary."""

    filename: str
    path: str
    file_type: str
    date_taken: str | None = None
    width: int | None = None
    height: int | None = None
    species: list[str] = Field(default_factory=list)
    confidence: float | None = None
    error: str | None = None


class JobSummary(BaseModel):
    """Progress and results for one backend processing job."""

    id: str
    state: JobState
    input_dir: str
    output_dir: str | None = None
    total: int = 0
    processed: int = 0
    message: str = ""
    results: list[DetectionItem] = Field(default_factory=list)
    error: str | None = None
    created_at: str
    updated_at: str
