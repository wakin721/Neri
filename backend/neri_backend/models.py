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


class ModelInfo(BaseModel):
    """A YOLO model discovered under the project model directory."""

    name: str
    path: str
    size_bytes: int | None = None


class ModelClassInfo(BaseModel):
    """A class exposed by the selected YOLO model."""

    id: int
    name: str
    display_name: str


class SettingsResponse(BaseModel):
    """Subset of desktop settings useful to the mobile/desktop frontend."""

    app_title: str
    app_version: str
    supported_image_extensions: list[str]
    supported_video_extensions: list[str]
    model_directory: str
    classification_model_directory: str
    available_models: list[ModelInfo] = Field(default_factory=list)
    available_classification_models: list[ModelInfo] = Field(default_factory=list)
    selected_model: str | None = None
    selected_classification_model: str | None = None
    species_types: dict[str, str] = Field(default_factory=dict)
    gpu_available: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingsUpdateRequest(BaseModel):
    """Persisted advanced settings sent from the Flutter settings screen."""

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
    selected_species_names: list[str] = Field(
        default_factory=list,
        description="Optional translated or raw class names to limit YOLO inference.",
    )


class CreateJobRequest(BaseModel):
    """Request body for creating a processing job."""

    input_dir: str = Field(
        ...,
        min_length=1,
        description="Folder path or a single supported media file path.",
    )
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
    size_bytes: int | None = None
    species: list[str] = Field(default_factory=list)
    confidence: float | None = None
    detection_boxes: list[dict[str, Any]] = Field(default_factory=list)
    detection_data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    validated: bool | None = None


class ValidationMarkRequest(BaseModel):
    """Manual validation update for one media file."""

    input_path: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    action: Literal["correct", "empty", "update", "unverified"] = "update"
    species_name: str | None = None
    species_count: str | None = None
    species_type: str | None = None
    remark: str | None = None


class ValidationExportRequest(BaseModel):
    """Export validated detection data through the existing data processor."""

    input_path: str = Field(..., min_length=1)
    output_path: str | None = None
    file_format: Literal["excel", "csv"] = "csv"
    confidence_settings: dict[str, float] = Field(default_factory=lambda: {"global": 0.25})
    columns_to_export: list[str] | None = None
    min_frame_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidationExportResponse(BaseModel):
    """Result of a validation export."""

    output_path: str
    file_format: str
    exported_count: int


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
