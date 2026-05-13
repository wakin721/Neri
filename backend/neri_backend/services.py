"""Processing services used by the FastAPI application.

The service intentionally reuses the existing Neri domain modules instead of
forking the detection logic.  Lightweight metadata indexing is available by
default; YOLO inference can be enabled per job when the runtime has the model
and ML dependencies installed.
"""

from __future__ import annotations

import csv
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.metadata_extractor import ImageMetadataExtractor
from system.utils import resource_path

from .models import CreateJobRequest, DetectionItem, JobState, JobSummary


class JobNotFoundError(KeyError):
    """Raised when a requested job id is not present in memory."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for API responses."""

    return datetime.now(timezone.utc).isoformat()


class ProcessingJobManager:
    """Small in-memory job runner for local Neri deployments."""

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="neri-job")
        self._jobs: dict[str, JobSummary] = {}
        self._lock = threading.Lock()

    def create_job(self, request: CreateJobRequest) -> JobSummary:
        """Create a job and start it on a background worker."""

        job_id = uuid.uuid4().hex
        now = utc_now()
        job = JobSummary(
            id=job_id,
            state=JobState.QUEUED,
            input_dir=request.input_dir,
            output_dir=request.output_dir,
            message="任务已加入队列",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run_job, job_id, request)
        return job

    def get_job(self, job_id: str) -> JobSummary:
        """Return a snapshot of a job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy(deep=True)

    def list_jobs(self) -> list[JobSummary]:
        """Return all jobs with newest first."""

        with self._lock:
            jobs = [job.model_copy(deep=True) for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def _mutate_job(self, job_id: str, **changes: object) -> None:
        with self._lock:
            current = self._jobs[job_id]
            data = current.model_dump()
            data.update(changes)
            data["updated_at"] = utc_now()
            self._jobs[job_id] = JobSummary(**data)

    def _run_job(self, job_id: str, request: CreateJobRequest) -> None:
        try:
            input_dir = Path(request.input_dir).expanduser().resolve()
            if not input_dir.exists() or not input_dir.is_dir():
                raise ValueError(f"输入目录不存在或不是文件夹: {input_dir}")

            files = list(_iter_supported_files(input_dir))
            self._mutate_job(
                job_id,
                state=JobState.RUNNING,
                total=len(files),
                message=f"发现 {len(files)} 个支持的媒体文件",
            )

            detector = None
            if request.options.enable_detection:
                detector = _load_detector(request.options.model_path)

            results: list[DetectionItem] = []
            for index, path in enumerate(files, start=1):
                item = _build_metadata_item(path)
                if detector is not None and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    item = _detect_image(detector, path, item, request)
                results.append(item)
                self._mutate_job(
                    job_id,
                    processed=index,
                    results=results,
                    message=f"正在处理 {index}/{len(files)}: {path.name}",
                )

            exported_path = _export_results(request.output_dir, results)
            message = "处理完成"
            if exported_path is not None:
                message = f"处理完成，结果已导出到 {exported_path}"
            self._mutate_job(job_id, state=JobState.COMPLETED, processed=len(files), results=results, message=message)
        except Exception as exc:  # noqa: BLE001 - converted to an API-visible job failure
            self._mutate_job(job_id, state=JobState.FAILED, error=str(exc), message="处理失败")


def _iter_supported_files(input_dir: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def _build_metadata_item(path: Path) -> DetectionItem:
    if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        return DetectionItem(filename=path.name, path=str(path), file_type=path.suffix.lower().lstrip("."))

    try:
        metadata, image = ImageMetadataExtractor.extract_metadata(str(path), path.name)
        width = metadata.get("宽度")
        height = metadata.get("高度")
        if image is not None:
            width = width or image.width
            height = height or image.height
            image.close()
        date_taken = metadata.get("拍摄日期")
        return DetectionItem(
            filename=path.name,
            path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            date_taken=str(date_taken) if date_taken else None,
            width=int(width) if width else None,
            height=int(height) if height else None,
        )
    except Exception as exc:  # noqa: BLE001 - per-file errors should not fail the whole job
        return DetectionItem(filename=path.name, path=str(path), file_type=path.suffix.lower().lstrip("."), error=str(exc))


def _load_detector(model_path: str | None):
    from system.image_processor import ImageProcessor

    resolved_model_path = model_path or resource_path("res/model/11s_1225.pt")
    return ImageProcessor(resolved_model_path)


def _detect_image(detector, path: Path, item: DetectionItem, request: CreateJobRequest) -> DetectionItem:
    try:
        detections = detector.detect_batch_species(
            [str(path)],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
        )
        detection = detections[0] if detections else {}
        species = detection.get("物种名称", []) or detection.get("species", []) or detection.get("species_names", []) or []
        if isinstance(species, str):
            species = [species]
        confidence = detection.get("最低置信度") or detection.get("confidence") or detection.get("max_confidence")
        return item.model_copy(update={"species": species, "confidence": confidence})
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return item.model_copy(update={"error": f"检测失败: {exc}"})


def _export_results(output_dir: str | None, results: list[DetectionItem]) -> Path | None:
    if not output_dir:
        return None
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "neri_results.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "path", "file_type", "date_taken", "width", "height", "species", "confidence", "error"],
        )
        writer.writeheader()
        for item in results:
            row = item.model_dump()
            row["species"] = ",".join(item.species)
            writer.writerow(row)
    return output_path
