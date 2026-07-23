"""API schemas for the Neri backend."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    missing_yolo_dependencies: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingsUpdateRequest(BaseModel):
    """Persisted advanced settings sent from the Flutter settings screen."""

    settings: dict[str, Any] = Field(default_factory=dict)


class InstallPytorchRequest(BaseModel):
    """Request to reinstall PyTorch for a selected runtime target."""

    env_choice: str = Field(default="自动检测", min_length=1)
    package_source: str = Field(default="official", min_length=1)
    install_intel_driver: bool = False


class InstallYoloDependenciesRequest(BaseModel):
    """Request to install runtime dependencies required by YOLO processing."""

    env_choice: str = Field(default="自动检测", min_length=1)
    package_source: str = Field(default="official", min_length=1)
    install_intel_driver: bool = False


class ReinstallPackageRequest(BaseModel):
    """Request to force-reinstall one Python package."""

    package: str = Field(..., min_length=1, max_length=160)
    package_source: str = Field(default="official", min_length=1)


class MaintenanceStartResponse(BaseModel):
    """Acknowledgement for a backend maintenance operation."""

    accepted: bool = True
    operation: str
    message: str
    progress: int = Field(default=0, ge=0, le=100)
    status_path: str | None = None
    maintenance_pid: int | None = None


class PytorchInstallPlanResponse(BaseModel):
    """Resolved PyTorch installation target and Intel driver preflight info."""

    env_choice: str
    actual_env: str
    index_url: str
    is_xpu: bool = False
    intel_driver: dict[str, Any] = Field(default_factory=dict)
    needs_intel_driver: bool = False
    intel_driver_page_url: str
    intel_driver_download_url: str


class MaintenanceStatusResponse(BaseModel):
    """Last known backend maintenance status."""

    operation: str | None = None
    state: str = "idle"
    message: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    started_at: str | None = None
    updated_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None
    status_path: str | None = None
    maintenance_pid: int | None = None
    error: str | None = None


class InstalledPackageInfo(BaseModel):
    """One Python package visible to the running backend environment."""

    name: str
    version: str


class DebugLogInfo(BaseModel):
    """Metadata for a readable software log file."""

    name: str
    path: str
    size_bytes: int
    modified_at: str | None = None


class DebugLogContent(DebugLogInfo):
    """Readable log content, optionally truncated to its latest bytes."""

    content: str
    truncated: bool = False


class RuntimeDiagnostics(BaseModel):
    """Runtime details useful while debugging the local environment."""

    backend_version: str = ""
    pytorch_installed: bool = False
    pytorch_version: str | None = None
    pytorch_cuda_version: str | None = None
    gpu_available: bool = False
    gpu_devices: list[str] = Field(default_factory=list)
    hardware_gpus: list[str] = Field(default_factory=list)
    error: str | None = None


class ClearCacheRequest(BaseModel):
    """Request to clear selected local runtime files."""

    clear_logs: bool = False
    clear_software_cache: bool = False

    @model_validator(mode="after")
    def require_selection(self) -> "ClearCacheRequest":
        if not self.clear_logs and not self.clear_software_cache:
            raise ValueError("至少需要选择一项要清理的内容。")
        return self


class ClearCacheResponse(BaseModel):
    """Summary of a local cache cleanup operation."""

    cleared_files: int = 0
    cleared_directories: int = 0
    reclaimed_bytes: int = 0
    skipped: list[str] = Field(default_factory=list)


class ProcessingOptions(BaseModel):
    """Options that mirror the current desktop advanced processing controls."""

    model_path: str | None = Field(
        default=None,
        description="Optional YOLO detection model path.",
    )
    classification_model_path: str | None = Field(
        default=None,
        description="Optional classification model path; it can also run on full images without a detection model.",
    )
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    use_fp16: bool = False
    use_augment: bool = False
    use_agnostic_nms: bool = True
    batch_size: int = Field(default=16, ge=1, le=64)
    thread_count: int = Field(default=4, ge=1, le=8)
    imgsz: int = Field(default=1920, ge=320, le=4096)
    vid_stride: int = Field(default=1, ge=1, le=120)
    video_mode: Literal["all", "fast"] = "all"
    enable_detection: bool = Field(
        default=False,
        description="When false, the job indexes supported files and EXIF metadata without loading YOLO.",
    )
    selected_species_names: list[str] = Field(
        default_factory=list,
        description="Optional translated or raw class names to limit YOLO inference.",
    )

    @model_validator(mode="after")
    def validate_detection_models(self) -> "ProcessingOptions":
        self.model_path = self.model_path.strip() if self.model_path else None
        self.classification_model_path = (
            self.classification_model_path.strip()
            if self.classification_model_path
            else None
        )
        if (
            self.enable_detection
            and not self.model_path
            and not self.classification_model_path
        ):
            raise ValueError("启用检测时，探测模型和分类模型至少需要选择一个。")
        return self


class CreateJobRequest(BaseModel):
    """Request body for creating a processing job."""

    input_dir: str | None = Field(
        default=None,
        min_length=1,
        description="Folder path or a single supported media file path.",
    )
    input_paths: list[str] = Field(
        default_factory=list,
        description="Optional explicit list of supported media files or folders to process.",
    )
    output_dir: str | None = None
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    @model_validator(mode="after")
    def validate_inputs(self) -> "CreateJobRequest":
        self.input_dir = self.input_dir.strip() if self.input_dir else None
        self.input_paths = [
            path.strip()
            for path in self.input_paths
            if path and path.strip()
        ]
        if not self.input_dir and not self.input_paths:
            raise ValueError("input_dir 或 input_paths 至少需要提供一个。")
        return self


class DetectionItem(BaseModel):
    """A single indexed file and optional detection summary."""

    filename: str
    path: str
    file_type: str
    date_taken: str | None = None
    modified_at: str | None = None
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


class ValidationBatchMarkRequest(BaseModel):
    """Manual validation update for multiple media files."""

    input_path: str = Field(..., min_length=1)
    file_paths: list[str] = Field(..., min_length=1)
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
    export_favorite_photos: bool = False
    favorite_photo_paths: list[str] = Field(default_factory=list)
    min_frame_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidationExportResponse(BaseModel):
    """Result of a validation export."""

    output_path: str
    file_format: str
    exported_count: int
    favorite_output_dir: str | None = None
    favorite_exported_count: int = 0


class JobSummary(BaseModel):
    """Progress and results for one backend processing job."""

    id: str
    state: JobState
    input_dir: str
    output_dir: str | None = None
    total: int = 0
    processed: int = 0
    elapsed_seconds: float = 0.0
    speed: float = 0.0
    remaining_seconds: float | None = None
    message: str = ""
    results: list[DetectionItem] = Field(default_factory=list)
    error: str | None = None
    active: bool = False
    created_at: str
    updated_at: str
