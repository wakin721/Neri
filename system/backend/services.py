"""Processing services used by the FastAPI application.

The service intentionally reuses the existing Neri domain modules instead of
forking the detection logic.  Lightweight metadata indexing is available by
default; YOLO inference can be enabled per job when the runtime has the model
and ML dependencies installed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.metadata_extractor import ImageMetadataExtractor
from system.utils import resource_path

from .models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ModelClassInfo,
    ModelInfo,
    ValidationExportRequest,
    ValidationExportResponse,
    ValidationMarkRequest,
)


class JobNotFoundError(KeyError):
    """Raised when a requested job id is not present in memory."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for API responses."""

    return datetime.now(timezone.utc).isoformat()


def model_directory() -> Path:
    """Return the project model directory used by the API."""

    return Path(resource_path("res/model")).resolve()


def classification_model_directory() -> Path:
    """Return the optional second-stage classification model directory."""

    return Path(resource_path("res/model_cls")).resolve()


def job_state_path() -> Path:
    """Return the persistent job progress file."""

    return Path("temp/job_state.json").resolve()


def list_available_models() -> list[ModelInfo]:
    """List all .pt files available under res/model."""

    directory = model_directory()
    return _list_model_files(directory, ("*.pt",))


def list_available_classification_models() -> list[ModelInfo]:
    """List classification model files under res/model_cls."""

    directory = classification_model_directory()
    return _list_model_files(directory, ("*.pt", "*.onnx", "*.engine"))


def list_model_classes(model_path: str | None) -> list[ModelClassInfo]:
    """Load a YOLO model and expose its class names for the settings UI."""

    missing = missing_yolo_dependencies()
    if missing:
        raise RuntimeError(f"缺少 YOLO 处理依赖: {', '.join(missing)}")

    resolved_model_path = _resolve_model_path(model_path)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {resolved_model_path}")
    return list(_cached_model_classes(str(resolved_model_path)))


@lru_cache(maxsize=8)
def _cached_model_classes(resolved_model_path: str) -> tuple[ModelClassInfo, ...]:
    detector = _load_detector(resolved_model_path)
    names = _detector_model_names(detector)
    translation_dict = getattr(detector, "translation_dict", {}) or {}
    class_infos: list[ModelClassInfo] = []

    for class_id, raw_name in sorted(names.items()):
        translated_name = str(translation_dict.get(raw_name, raw_name))
        display_name = (
            translated_name
            if translated_name == raw_name
            else f"{translated_name} ({raw_name})"
        )
        class_infos.append(
            ModelClassInfo(
                id=class_id,
                name=raw_name,
                display_name=display_name,
            )
        )
    return tuple(class_infos)


def _list_model_files(directory: Path, patterns: tuple[str, ...]) -> list[ModelInfo]:
    if not directory.exists() or not directory.is_dir():
        return []

    models: list[ModelInfo] = []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    for path in sorted(set(paths), key=lambda item: item.name.lower()):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        models.append(ModelInfo(name=path.name, path=str(path), size_bytes=size_bytes))
    return models


def load_species_types() -> dict[str, str]:
    """Load species type labels from res/species_database.db."""

    db_path = Path(resource_path("res/species_database.db"))
    if not db_path.exists():
        return {}

    species_types: dict[str, str] = {}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT 中文名, 物种类型 FROM species").fetchall()
        for name, species_type in rows:
            clean_name = str(name or "").strip()
            clean_type = str(species_type or "").strip()
            if clean_name and clean_type:
                species_types[clean_name] = clean_type
    except Exception:
        return {}
    return species_types


def detect_gpu_available() -> bool:
    """Best-effort hardware acceleration check used by the settings UI."""

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        xpu = getattr(torch, "xpu", None)
        xpu_available = bool(xpu is not None and xpu.is_available())
        return cuda_available or xpu_available
    except Exception:
        return False


def missing_yolo_dependencies() -> list[str]:
    """Return Python modules required before YOLO inference can run."""

    required_modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
    }
    return [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]


def preview_media_items(
    input_dir: str,
    output_dir: str | None = None,
    *,
    include_cached: bool = True,
) -> list[DetectionItem]:
    """Return previewable media without running YOLO.

    When include_cached is false this stays intentionally shallow: it only
    lists supported files and basic stat data so the UI can show a folder
    quickly, then fetch full metadata for the selected file on demand.
    """

    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    files = list(_resolve_supported_inputs(input_path))
    if not files:
        return []

    detection_db_roots = (
        _preview_detection_db_roots(input_path, output_dir, files)
        if include_cached
        else []
    )
    detection_index = (
        _load_detection_index(detection_db_roots, recursive=False) if include_cached else {}
    )
    validation_index = (
        _load_validation_index(detection_db_roots, recursive=False) if include_cached else {}
    )
    items: list[DetectionItem] = []
    for path in files:
        item = _build_fast_metadata_item(path)
        if include_cached:
            db_detection_data = detection_index.get(path.stem)
            if db_detection_data:
                item = _apply_detection_data(item, db_detection_data)
            item = _apply_validation_state(item, path.name, validation_index)
        items.append(item)
    return items


def preview_media_item(file_path: str, input_dir: str | None = None, output_dir: str | None = None) -> DetectionItem:
    """Return full metadata and cached detections for one preview item."""

    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"文件不存在: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的媒体文件: {path}")

    item = _build_metadata_item(path)
    roots: list[Path] = [path.parent]
    if input_dir:
        input_path = Path(input_dir).expanduser()
        roots.extend(_detection_db_search_roots(input_path, output_dir))
    validation_index = _load_validation_index(roots, recursive=False)
    db_detection_data = _load_detection_data_for_path(path, roots, recursive=False)
    if db_detection_data:
        item = _apply_detection_data(item, db_detection_data)
    item = _apply_validation_state(item, path.name, validation_index)
    return item


class ProcessingJobManager:
    """Small in-memory job runner for local Neri deployments."""

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="neri-job")
        self._jobs: dict[str, JobSummary] = {}
        self._job_requests: dict[str, CreateJobRequest] = {}
        self._cancelled_jobs: set[str] = set()
        self._deleted_jobs: set[str] = set()
        self._active_job_ids: set[str] = set()
        self._lock = threading.Lock()
        self._load_state()

    def create_job(
        self,
        request: CreateJobRequest,
        initial_results: list[DetectionItem] | None = None,
        resumed_from: str | None = None,
    ) -> JobSummary:
        """Create a job and start it on a background worker."""

        initial_results = _unique_detection_items(initial_results or [])
        job_id = uuid.uuid4().hex
        now = utc_now()
        initial_count = len(initial_results)
        job = JobSummary(
            id=job_id,
            state=JobState.QUEUED,
            input_dir=request.input_dir,
            output_dir=request.output_dir,
            processed=initial_count,
            message=(
                "继续任务已加入队列"
                if resumed_from is not None
                else ("预扫描已加入队列" if not request.options.enable_detection else "任务已加入队列")
            ),
            results=initial_results,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._job_requests[job_id] = request.model_copy(deep=True)
            self._cancelled_jobs.discard(job_id)
            self._deleted_jobs.discard(job_id)
            self._save_state_unlocked()
        self._executor.submit(self._run_job, job_id, request, initial_results)
        return job

    def cancel_job(self, job_id: str) -> JobSummary:
        """Request cancellation for a queued or running job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return job.model_copy(deep=True)
            self._cancelled_jobs.add(job_id)
            data = job.model_dump()
            data.update(
                {
                    "state": JobState.CANCELLED,
                    "message": "任务已停止",
                    "updated_at": utc_now(),
                }
            )
            self._jobs[job_id] = JobSummary(**data)
            self._save_state_unlocked()
            return self._snapshot_unlocked(job_id)

    def resume_job(self, job_id: str) -> JobSummary:
        """Continue a cancelled or failed job in-place."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job_id in self._active_job_ids:
                raise ValueError("任务仍在停止收尾，请稍后继续。")
            if job.state in {JobState.QUEUED, JobState.RUNNING}:
                raise ValueError("任务仍在运行，不能继续。")
            if job.state == JobState.COMPLETED:
                raise ValueError("任务已完成，无需继续。")
            request = self._job_requests.get(job_id)
            if request is None:
                raise ValueError("缺少原始任务参数，无法继续。")
            initial_results = [item.model_copy(deep=True) for item in job.results]
            data = job.model_dump()
            data.update(
                {
                    "state": JobState.QUEUED,
                    "message": "继续任务已加入队列",
                    "error": None,
                    "updated_at": utc_now(),
                }
            )
            self._jobs[job_id] = JobSummary(**data)
            self._cancelled_jobs.discard(job_id)
            self._deleted_jobs.discard(job_id)
            self._save_state_unlocked()
            request = request.model_copy(deep=True)
            snapshot = self._snapshot_unlocked(job_id)

        self._executor.submit(self._run_job, job_id, request, initial_results)
        return snapshot

    def delete_job(self, job_id: str) -> None:
        """Delete one job from the visible progress list."""

        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            self._cancelled_jobs.add(job_id)
            self._deleted_jobs.add(job_id)
            self._jobs.pop(job_id, None)
            self._job_requests.pop(job_id, None)
            self._save_state_unlocked()

    def clear_jobs(self) -> None:
        """Delete all jobs from the visible progress list."""

        with self._lock:
            self._cancelled_jobs.update(self._jobs.keys())
            self._deleted_jobs.update(self._jobs.keys())
            self._jobs.clear()
            self._job_requests.clear()
            self._save_state_unlocked()

    def get_job(self, job_id: str) -> JobSummary:
        """Return a snapshot of a job."""

        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            return self._snapshot_unlocked(job_id)

    def list_jobs(self) -> list[JobSummary]:
        """Return all jobs with newest first."""

        with self._lock:
            jobs = [self._snapshot_unlocked(job_id) for job_id in self._jobs]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def _snapshot_unlocked(self, job_id: str) -> JobSummary:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job.model_copy(
            update={"active": job_id in self._active_job_ids},
            deep=True,
        )

    def _mutate_job(self, job_id: str, *, persist: bool = True, **changes: object) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None or job_id in self._deleted_jobs:
                return
            data = current.model_dump()
            data.update(changes)
            data["updated_at"] = utc_now()
            self._jobs[job_id] = JobSummary(**data)
            if persist:
                self._save_state_unlocked()

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return (
                job_id in self._deleted_jobs
                or job_id in self._cancelled_jobs
                or job is not None
                and job.state == JobState.CANCELLED
            )

    def _mark_cancelled(self, job_id: str, results: list[DetectionItem] | None = None) -> None:
        changes: dict[str, object] = {
            "state": JobState.CANCELLED,
            "message": "任务已停止",
        }
        if results is not None:
            changes["results"] = results
            changes["processed"] = len(results)
        self._mutate_job(job_id, **changes)

    def _load_state(self) -> None:
        state_path = job_state_path()
        if not state_path.exists():
            return
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            raw_jobs = data.get("jobs", {})
            raw_requests = data.get("requests", {})
            if not isinstance(raw_jobs, dict) or not isinstance(raw_requests, dict):
                return
            for job_id, raw_job in raw_jobs.items():
                if not isinstance(raw_job, dict):
                    continue
                job = JobSummary(**raw_job).model_copy(update={"active": False})
                if job.state in {JobState.QUEUED, JobState.RUNNING}:
                    job = job.model_copy(
                        update={
                            "state": JobState.CANCELLED,
                            "message": "后端已重启，任务已暂停，可继续任务",
                            "updated_at": utc_now(),
                        }
                    )
                self._jobs[str(job_id)] = job
            for job_id, raw_request in raw_requests.items():
                if not isinstance(raw_request, dict):
                    continue
                self._job_requests[str(job_id)] = CreateJobRequest(**raw_request)
            self._save_state_unlocked()
        except Exception:
            self._jobs.clear()
            self._job_requests.clear()

    def _save_state_unlocked(self) -> None:
        state_path = job_state_path()
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "jobs": {
                    job_id: job.model_copy(update={"active": False}).model_dump()
                    for job_id, job in self._jobs.items()
                },
                "requests": {
                    job_id: request.model_dump()
                    for job_id, request in self._job_requests.items()
                    if job_id in self._jobs
                },
            }
            state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _run_job(
        self,
        job_id: str,
        request: CreateJobRequest,
        initial_results: list[DetectionItem] | None = None,
    ) -> None:
        with self._lock:
            if job_id in self._deleted_jobs:
                return
            self._active_job_ids.add(job_id)
        try:
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id, initial_results)
                return
            input_path = Path(request.input_dir).expanduser().resolve()
            if not input_path.exists():
                raise ValueError(f"输入路径不存在: {input_path}")

            files = list(_resolve_supported_inputs(input_path))
            if not files:
                raise ValueError(f"输入路径中没有支持的媒体文件: {input_path}")
            file_keys = {_path_key(path) for path in files}
            results = [
                item.model_copy(deep=True)
                for item in _unique_detection_items(initial_results or [])
                if _path_key(item.path) in file_keys
            ]
            skipped_paths = {_path_key(item.path) for item in results}
            remaining_files = [path for path in files if _path_key(path) not in skipped_paths]
            processed = len(results)

            self._mutate_job(
                job_id,
                state=JobState.RUNNING,
                total=len(files),
                processed=processed,
                results=results,
                message=(
                    f"继续任务：已完成 {processed}/{len(files)}，剩余 {len(remaining_files)} 个文件"
                    if processed > 0
                    else f"预扫描发现 {len(files)} 个支持的媒体文件"
                    if not request.options.enable_detection
                    else (
                        f"继续任务：已完成 {processed}/{len(files)}，剩余 {len(remaining_files)} 个文件"
                        if processed > 0
                        else f"发现 {len(files)} 个支持的媒体文件"
                    )
                ),
            )

            detection_db_roots = (
                _detection_db_search_roots(input_path, request.output_dir)
                if request.options.enable_detection
                else []
            )
            detection_index = (
                _load_detection_index(detection_db_roots) if request.options.enable_detection else {}
            )
            detector = None
            if request.options.enable_detection:
                detector = _load_detector(
                    request.options.model_path,
                    request.options.classification_model_path,
                )

            if detector is None:
                for path in remaining_files:
                    if self._is_cancelled(job_id):
                        self._mark_cancelled(job_id, results)
                        return
                    item = _build_fast_metadata_item(path)
                    results.append(item)
                    processed += 1
                    media_label = "照片" if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else "视频"
                    self._mutate_job(
                        job_id,
                        processed=processed,
                        results=results,
                        message=f"正在预扫描{media_label} {processed}/{len(files)}: {path.name}",
                    )
            else:
                image_files = [
                    path for path in remaining_files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
                video_files = [
                    path for path in remaining_files if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
                ]
                batch_size = max(1, int(request.options.batch_size or 1))

                for batch in _chunked(image_files, batch_size):
                    if self._is_cancelled(job_id):
                        self._mark_cancelled(job_id, results)
                        return
                    batch_items: list[DetectionItem] = []
                    for path in batch:
                        item = _build_metadata_item(path)
                        db_detection_data = detection_index.get(path.stem)
                        if db_detection_data:
                            item = _apply_detection_data(item, db_detection_data)
                        batch_items.append(item)

                    detected_items = _detect_image_batch(detector, batch, batch_items, request, input_path)
                    if self._is_cancelled(job_id):
                        self._mark_cancelled(job_id, results)
                        return
                    results.extend(detected_items)
                    processed += len(detected_items)
                    last_path = batch[min(len(batch), len(detected_items)) - 1] if detected_items else batch[-1]
                    self._mutate_job(
                        job_id,
                        persist=False,
                        processed=processed,
                        results=results,
                        message=(
                            f"正在批处理照片 {processed}/{len(files)} "
                            f"(batch={len(batch)}): {last_path.name}"
                        ),
                    )

                for path in video_files:
                    if self._is_cancelled(job_id):
                        self._mark_cancelled(job_id, results)
                        return
                    item = _build_metadata_item(path)
                    db_detection_data = detection_index.get(path.stem)
                    if db_detection_data:
                        item = _apply_detection_data(item, db_detection_data)

                    item = _detect_video(
                        detector,
                        path,
                        item,
                        request,
                        input_path,
                        cancelled=lambda: self._is_cancelled(job_id),
                    )
                    if self._is_cancelled(job_id):
                        self._mark_cancelled(job_id, results)
                        return
                    results.append(item)
                    processed += 1
                    self._mutate_job(
                        job_id,
                        processed=processed,
                        results=results,
                        message=f"正在处理视频 {processed}/{len(files)}: {path.name}",
                    )

            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id, results)
                return
            exported_path = _export_results(request.output_dir, results)
            message = "预扫描完成" if not request.options.enable_detection else "处理完成"
            if exported_path is not None:
                message = f"处理完成，结果已导出到 {exported_path}"
            self._mutate_job(job_id, state=JobState.COMPLETED, processed=len(files), results=results, message=message)
        except Exception as exc:  # noqa: BLE001 - converted to an API-visible job failure
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
            else:
                self._mutate_job(job_id, state=JobState.FAILED, error=str(exc), message="处理失败")
        finally:
            with self._lock:
                self._active_job_ids.discard(job_id)
                self._save_state_unlocked()


def _resolve_supported_inputs(input_path: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    if input_path.is_file():
        if input_path.suffix.lower() in supported:
            yield input_path
        return

    if input_path.is_dir():
        yield from _iter_supported_files(input_path)


def _chunked(paths: list[Path], size: int) -> Iterable[list[Path]]:
    size = max(1, size)
    for index in range(0, len(paths), size):
        yield paths[index : index + size]


def _unique_detection_items(items: Iterable[DetectionItem]) -> list[DetectionItem]:
    unique_items: list[DetectionItem] = []
    seen: set[str] = set()
    for item in items:
        key = _path_key(item.path)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item.model_copy(deep=True))
    return unique_items


def _path_key(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve()).casefold()
    except Exception:
        return text.replace("\\", "/").casefold()


def _iter_supported_files(input_dir: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    image_files: list[Path] = []
    video_files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in supported:
            continue
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(path)
        else:
            video_files.append(path)
    yield from image_files
    yield from video_files


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _file_modified_at(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return None


def _build_fast_metadata_item(path: Path) -> DetectionItem:
    return DetectionItem(
        filename=path.name,
        path=str(path),
        file_type=path.suffix.lower().lstrip("."),
        modified_at=_file_modified_at(path),
        size_bytes=_file_size(path),
    )


def _build_metadata_item(path: Path) -> DetectionItem:
    size_bytes = _file_size(path)

    if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        return DetectionItem(
            filename=path.name,
            path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            modified_at=_file_modified_at(path),
            size_bytes=size_bytes,
        )

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
            modified_at=_file_modified_at(path),
            width=int(width) if width else None,
            height=int(height) if height else None,
            size_bytes=size_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - per-file errors should not fail the whole job
        return DetectionItem(
            filename=path.name,
            path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            modified_at=_file_modified_at(path),
            size_bytes=size_bytes,
            error=str(exc),
        )


def _load_detector(model_path: str | None, classification_model_path: str | None = None):
    from system.image_processor import ImageProcessor

    resolved_model_path = _resolve_model_path(model_path)
    detector = ImageProcessor(str(resolved_model_path))
    resolved_classification_path = _resolve_classification_model_path(classification_model_path)
    if resolved_classification_path is not None:
        detector.load_cls_model(str(resolved_classification_path))
    return detector


def _detector_model_names(detector) -> dict[int, str]:
    model = getattr(detector, "model", None)
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for raw_class_id, raw_name in names.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError):
                continue
            normalized[class_id] = str(raw_name)
        return normalized
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def _selected_species_class_ids(detector, selected_species_names: list[str]) -> list[int] | None:
    selected_names = {
        _normalize_species_key(name)
        for name in selected_species_names
        if _normalize_species_key(name)
    }
    if not selected_names:
        return None

    translation_dict = getattr(detector, "translation_dict", {}) or {}
    matched_class_ids: list[int] = []
    for class_id, raw_name in _detector_model_names(detector).items():
        translated_name = str(translation_dict.get(raw_name, raw_name))
        display_name = (
            translated_name
            if translated_name == raw_name
            else f"{translated_name} ({raw_name})"
        )
        aliases = {
            _normalize_species_key(str(class_id)),
            _normalize_species_key(raw_name),
            _normalize_species_key(translated_name),
            _normalize_species_key(display_name),
        }
        if aliases & selected_names:
            matched_class_ids.append(class_id)

    return sorted(set(matched_class_ids)) if matched_class_ids else None


def _normalize_species_key(value: str) -> str:
    return value.strip().casefold()


def _resolve_model_path(model_path: str | None) -> Path:
    default_model_dir = model_directory()
    if model_path:
        raw_candidate = Path(model_path).expanduser()
        if raw_candidate.is_absolute():
            candidate = raw_candidate
        else:
            named_model = default_model_dir / raw_candidate
            candidate = named_model if named_model.exists() else Path(resource_path(str(raw_candidate)))
    else:
        candidate = default_model_dir

    if candidate.is_dir():
        preferred_model = candidate / "11s_1225.pt"
        if preferred_model.exists():
            return preferred_model
        models = sorted(candidate.glob("*.pt"))
        if models:
            return models[0]
        raise FileNotFoundError(f"模型文件夹中未找到 .pt 模型: {candidate}")

    return candidate


def _resolve_classification_model_path(model_path: str | None) -> Path | None:
    if not model_path:
        return None
    raw_candidate = Path(model_path).expanduser()
    if raw_candidate.is_absolute():
        candidate = raw_candidate
    else:
        default_model_dir = classification_model_directory()
        named_model = default_model_dir / raw_candidate
        candidate = named_model if named_model.exists() else Path(resource_path(str(raw_candidate)))
    return candidate if candidate.exists() and candidate.is_file() else None


def _detection_db_search_roots(input_path: Path, output_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    roots.append(input_path if input_path.is_dir() else input_path.parent)
    if output_dir:
        roots.append(Path(output_dir).expanduser())

    return _unique_existing_dirs(roots)


def _preview_detection_db_roots(
    input_path: Path,
    output_dir: str | None,
    files: list[Path],
) -> list[Path]:
    roots: list[Path] = [
        *_detection_db_search_roots(input_path, output_dir),
        *(path.parent for path in files),
    ]
    return _unique_existing_dirs(roots)


def _unique_existing_dirs(roots: Iterable[Path]) -> list[Path]:
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def _candidate_detection_dbs(
    path: Path,
    search_roots: list[Path],
    *,
    recursive: bool = True,
) -> list[Path]:
    return _candidate_detection_dbs_for_roots(
        [path.parent, *search_roots],
        recursive=recursive,
    )


def _candidate_detection_dbs_for_roots(
    roots: list[Path],
    *,
    recursive: bool = True,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            resolved_root = root
        if not resolved_root.exists() or not resolved_root.is_dir():
            continue

        try:
            from system.detection_db import get_db_path

            preferred = Path(get_db_path(str(resolved_root)))
            db_paths = [
                preferred,
                *sorted(resolved_root.glob("*.db"), key=lambda item: item.name.lower()),
            ]
            if recursive:
                db_paths.extend(
                    sorted(resolved_root.rglob("*.db"), key=lambda item: str(item).lower())
                )
        except Exception:
            db_paths = (
                sorted(resolved_root.rglob("*.db"), key=lambda item: str(item).lower())
                if recursive
                else sorted(resolved_root.glob("*.db"), key=lambda item: item.name.lower())
            )

        for db_path in db_paths:
            try:
                resolved_db = db_path.resolve()
            except OSError:
                resolved_db = db_path
            if resolved_db in seen or not resolved_db.exists() or not resolved_db.is_file():
                continue
            seen.add(resolved_db)
            candidates.append(resolved_db)

    return candidates


def _load_detection_index(
    search_roots: list[Path],
    *,
    recursive: bool = True,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for db_path in _candidate_detection_dbs_for_roots(search_roots, recursive=recursive):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT base_name, detection_json FROM detections").fetchall()
            for base_name, detection_json in rows:
                if base_name in index:
                    continue
                data = json.loads(detection_json)
                if isinstance(data, dict):
                    index[str(base_name)] = data
        except Exception:
            continue
    return index


def _load_validation_index(search_roots: list[Path], *, recursive: bool = True) -> dict[str, bool]:
    index: dict[str, bool] = {}
    for db_path in _candidate_detection_dbs_for_roots(search_roots, recursive=recursive):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT image_filename, is_validated FROM validation").fetchall()
            for image_filename, is_validated in rows:
                key = str(image_filename)
                if key not in index:
                    index[key] = bool(is_validated)
        except Exception:
            continue
    return index


def _load_detection_data_for_path(
    path: Path,
    search_roots: list[Path],
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    base_name = path.stem
    for db_path in _candidate_detection_dbs(path, search_roots, recursive=recursive):
        try:
            from system.detection_db import get_detection

            data = get_detection(str(db_path), base_name)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _apply_validation_state(
    item: DetectionItem,
    filename: str,
    validation_index: dict[str, bool],
) -> DetectionItem:
    if filename not in validation_index:
        return item
    return item.model_copy(update={"validated": validation_index[filename]})


def _value_from_keys(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _extract_species_list(data: dict[str, Any], boxes: list[dict[str, Any]] | None = None) -> list[str]:
    raw_species = _value_from_keys(data, ("物种名称", "species", "species_names"))
    species: list[str] = []
    if isinstance(raw_species, str):
        pieces = raw_species.replace("、", ",").replace("，", ",").split(",")
        species.extend(piece.strip() for piece in pieces)
    elif isinstance(raw_species, list):
        species.extend(str(item).strip() for item in raw_species)

    if boxes:
        species.extend(str(box.get("species", "")).strip() for box in boxes)

    ignored = {"", "空", "none", "None", "unknown", "Unknown"}
    unique_species: list[str] = []
    for item in species:
        if item in ignored or item in unique_species:
            continue
        unique_species.append(item)
    return unique_species


def _coerce_bbox(value: Any) -> list[float]:
    if isinstance(value, dict):
        value = [
            _value_from_keys(value, ("x1", "left", "xmin")),
            _value_from_keys(value, ("y1", "top", "ymin")),
            _value_from_keys(value, ("x2", "right", "xmax")),
            _value_from_keys(value, ("y2", "bottom", "ymax")),
        ]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    bbox: list[float] = []
    for item in value[:4]:
        number = _coerce_float(item)
        if number is None:
            return []
        bbox.append(number)
    return bbox


def _normalize_detection_boxes(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_boxes = _value_from_keys(data, ("检测框", "detect_results", "objects"))
    if not isinstance(raw_boxes, list):
        raw_tracks = data.get("tracks")
        raw_boxes = _boxes_from_tracks(raw_tracks)
    if not isinstance(raw_boxes, list):
        return []

    boxes: list[dict[str, Any]] = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, dict):
            continue
        bbox = _coerce_bbox(_value_from_keys(raw_box, ("边界框", "bbox", "box", "xyxy")))
        if not bbox:
            continue
        species = _value_from_keys(raw_box, ("物种", "species", "class_name", "name"))
        confidence = _coerce_float(_value_from_keys(raw_box, ("置信度", "confidence", "conf", "score")))
        candidates = _value_from_keys(raw_box, ("候选项", "candidates"))
        frame_index = _coerce_int(_value_from_keys(raw_box, ("frame_index", "frameIndex", "frame", "帧索引", "帧序号")))
        timestamp = _coerce_float(_value_from_keys(raw_box, ("timestamp", "time", "时间", "秒")))
        track_id = _value_from_keys(raw_box, ("track_id", "trackId", "id", "目标ID"))
        box_data = {
            "species": str(species or "Unknown"),
            "confidence": confidence,
            "bbox": bbox,
            "candidates": candidates if isinstance(candidates, list) else [],
        }
        if frame_index is not None:
            box_data["frame_index"] = frame_index
        if timestamp is not None:
            box_data["timestamp"] = timestamp
        if track_id is not None:
            box_data["track_id"] = str(track_id)
        boxes.append(box_data)
    return boxes


def _boxes_from_tracks(raw_tracks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tracks, dict):
        return []

    boxes: list[dict[str, Any]] = []
    for track_id, points in raw_tracks.items():
        if not isinstance(points, list):
            continue
        species_votes: dict[str, int] = {}
        for point in points:
            if not isinstance(point, dict):
                continue
            species = str(_value_from_keys(point, ("species", "物种", "name", "class_name")) or "").strip()
            if species:
                species_votes[species] = species_votes.get(species, 0) + 1
        voted_species = max(species_votes.items(), key=lambda item: item[1])[0] if species_votes else "Unknown"

        for point in points:
            if not isinstance(point, dict):
                continue
            bbox = _coerce_bbox(point.get("bbox"))
            if not bbox:
                continue
            species = str(_value_from_keys(point, ("species", "物种", "name", "class_name")) or voted_species)
            frame_index = _coerce_int(_value_from_keys(point, ("frame_index", "frameIndex", "frame", "帧索引", "帧序号")))
            timestamp = _coerce_float(_value_from_keys(point, ("timestamp", "time", "时间", "秒")))
            box_data = {
                "species": voted_species if species == "Unknown" else species,
                "confidence": _coerce_float(_value_from_keys(point, ("confidence", "置信度", "conf", "score"))),
                "bbox": bbox,
                "candidates": [],
                "track_id": str(track_id),
            }
            if frame_index is not None:
                box_data["frame_index"] = frame_index
            if timestamp is not None:
                box_data["timestamp"] = timestamp
            boxes.append(box_data)
    return boxes


def _apply_detection_data(item: DetectionItem, detection_data: dict[str, Any]) -> DetectionItem:
    boxes = _normalize_detection_boxes(detection_data)
    species = _extract_species_list(detection_data, boxes)
    confidence = _coerce_float(_value_from_keys(detection_data, ("最低置信度", "confidence", "max_confidence")))
    if confidence is None and boxes:
        box_confidences = [box["confidence"] for box in boxes if box.get("confidence") is not None]
        confidence = min(box_confidences) if box_confidences else None
    return item.model_copy(
        update={
            "species": species,
            "confidence": confidence,
            "detection_boxes": boxes,
            "detection_data": detection_data,
        }
    )


def _serialize_detector_output(detector, detection: dict[str, Any]) -> dict[str, Any]:
    confidence_raw = _value_from_keys(detection, ("最低置信度", "confidence", "max_confidence"))
    raw_species = _value_from_keys(detection, ("物种名称", "species", "species_names"))
    species = raw_species if isinstance(raw_species, str) else ",".join(_extract_species_list({"species": raw_species}))
    boxes: list[dict[str, Any]] = []
    all_confidences: list[float] = []
    all_classes: list[float] = []
    names_map: dict[Any, str] = {}

    results = detection.get("detect_results") or []
    for result in results:
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None:
            continue

        names = getattr(result, "names", {}) or {}
        candidates_data = getattr(result, "candidates_data", {}) or {}
        translation_dict = getattr(detector, "translation_dict", {}) or {}
        names_map = {
            class_id: translation_dict.get(english_name, english_name)
            for class_id, english_name in names.items()
        }
        try:
            all_confidences = result_boxes.conf.tolist()
            all_classes = result_boxes.cls.tolist()
        except Exception:
            pass

        for index, box in enumerate(result_boxes):
            try:
                cls_id = int(box.cls.item())
                raw_name = names.get(cls_id, str(cls_id))
                translated_name = translation_dict.get(raw_name, raw_name)
                box_confidence = float(box.conf.item())
                bbox = [float(value) for value in box.xyxy.tolist()[0]]
            except Exception:
                continue

            candidates = candidates_data.get(index, [])
            if candidates:
                best_candidate = candidates[0]
                translated_name = str(best_candidate.get("name", translated_name))
                box_confidence = _coerce_float(best_candidate.get("conf")) or box_confidence

            boxes.append(
                {
                    "物种": translated_name,
                    "置信度": box_confidence,
                    "边界框": bbox,
                    "候选项": candidates if isinstance(candidates, list) else [],
                }
            )

    return {
        "物种名称": species,
        "物种数量": detection.get("物种数量", ""),
        "最低置信度": confidence_raw,
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "检测框": boxes,
        "all_confidences": all_confidences,
        "all_classes": all_classes,
        "names_map": names_map,
    }


def _save_detection_data_for_path(path: Path, detection_data: dict[str, Any], input_path: Path) -> None:
    _save_detection_data_batch([(path, detection_data)], input_path)


def _save_detection_data_batch(detections: Iterable[tuple[Path, dict[str, Any]]], input_path: Path) -> None:
    payloads_by_root: dict[Path, list[tuple[str, str, dict[str, Any]]]] = {}
    validations_by_root: dict[Path, list[tuple[str, bool]]] = {}
    for path, detection_data in detections:
        roots = [path.parent]
        if input_path.is_dir():
            roots.append(input_path)

        seen: set[Path] = set()
        for root in roots:
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            if resolved in seen:
                continue
            seen.add(resolved)
            payloads_by_root.setdefault(resolved, []).append(
                (path.stem, path.name, detection_data)
            )
            validations_by_root.setdefault(resolved, []).append((path.name, False))

    if not payloads_by_root:
        return

    try:
        from system.detection_db import (
            get_db_path,
            init_db,
            upsert_detections_bulk,
            upsert_validation_bulk,
        )
    except Exception:
        return

    for root, payloads in payloads_by_root.items():
        try:
            db_path = get_db_path(str(root))
            init_db(db_path)
            upsert_detections_bulk(db_path, payloads)
            upsert_validation_bulk(db_path, validations_by_root.get(root, []))
        except Exception:
            continue


def _detect_image_batch(
    detector,
    paths: list[Path],
    items: list[DetectionItem],
    request: CreateJobRequest,
    input_path: Path,
) -> list[DetectionItem]:
    try:
        selected_class_ids = _selected_species_class_ids(
            detector,
            request.options.selected_species_names,
        )
        detections = detector.detect_batch_species(
            [str(path) for path in paths],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
            augment=request.options.use_augment,
            agnostic_nms=request.options.use_agnostic_nms,
            classes=selected_class_ids,
        )
        detected_items: list[DetectionItem] = []
        detection_payloads: list[tuple[Path, dict[str, Any]]] = []
        for index, (path, item) in enumerate(zip(paths, items)):
            detection = detections[index] if index < len(detections) else {}
            if not isinstance(detection, dict):
                detection = {}
            detection_data = _serialize_detector_output(detector, detection)
            detection_payloads.append((path, detection_data))
            detected_items.append(_apply_detection_data(item, detection_data))
        _save_detection_data_batch(detection_payloads, input_path)
        return detected_items
    except Exception as exc:  # noqa: BLE001 - preserve per-file metadata on batch failure
        return [item.model_copy(update={"error": f"批量检测失败: {exc}"}) for item in items]


def _detect_image(detector, path: Path, item: DetectionItem, request: CreateJobRequest, input_path: Path) -> DetectionItem:
    try:
        selected_class_ids = _selected_species_class_ids(
            detector,
            request.options.selected_species_names,
        )
        detections = detector.detect_batch_species(
            [str(path)],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
            augment=request.options.use_augment,
            agnostic_nms=request.options.use_agnostic_nms,
            classes=selected_class_ids,
        )
        detection = detections[0] if detections else {}
        detection_data = _serialize_detector_output(detector, detection)
        _save_detection_data_for_path(path, detection_data, input_path)
        return _apply_detection_data(item, detection_data)
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return item.model_copy(update={"error": f"检测失败: {exc}"})


def _detect_video(
    detector,
    path: Path,
    item: DetectionItem,
    request: CreateJobRequest,
    input_path: Path,
    cancelled=None,
) -> DetectionItem:
    if request.options.video_mode == "fast":
        return _detect_video_fast(detector, path, item, request, input_path, cancelled=cancelled)
    return _detect_video_track(detector, path, item, request, input_path, cancelled=cancelled)


def _detect_video_track(
    detector,
    path: Path,
    item: DetectionItem,
    request: CreateJobRequest,
    input_path: Path,
    cancelled=None,
) -> DetectionItem:
    try:
        selected_class_ids = _selected_species_class_ids(
            detector,
            request.options.selected_species_names,
        )
        output_dir = _video_output_dir(input_path, request.output_dir)
        extra_db_dir = str(input_path if input_path.is_dir() else input_path.parent)

        def status_callback(*_args):
            if cancelled is not None and cancelled():
                raise RuntimeError("用户取消任务")

        video_result = detector.detect_video_species(
            str(path),
            str(output_dir),
            use_fp16=request.options.use_fp16,
            iou=request.options.iou,
            conf=request.options.confidence,
            augment=request.options.use_augment,
            agnostic_nms=request.options.use_agnostic_nms,
            classes=selected_class_ids,
            status_callback=status_callback,
            vid_stride=request.options.vid_stride,
            extra_db_dir=extra_db_dir,
        )
        if video_result.get("status") != "success":
            return item.model_copy(update={"error": f"视频检测失败: {video_result.get('error', '未知错误')}"})

        detection_data: dict[str, Any] = {}
        json_path = video_result.get("json_path")
        if json_path:
            try:
                with Path(str(json_path)).open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    detection_data = loaded
            except Exception:
                detection_data = {}
        if not detection_data:
            detection_data = _load_detection_data_for_path(
                path,
                [output_dir, Path(extra_db_dir)],
                recursive=False,
            )
        if detection_data:
            _save_detection_data_for_path(path, detection_data, input_path)
            return _apply_detection_data(item, detection_data)
        return item
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return item.model_copy(update={"error": f"视频检测失败: {exc}"})


def _detect_video_fast(
    detector,
    path: Path,
    item: DetectionItem,
    request: CreateJobRequest,
    input_path: Path,
    cancelled=None,
) -> DetectionItem:
    import shutil
    import tempfile
    from collections import Counter

    import cv2

    temp_dir = Path(tempfile.mkdtemp(prefix="neri_video_frames_"))
    frame_paths: list[Path] = []
    frame_points: list[int] = []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")
        if cancelled is not None and cancelled():
            raise RuntimeError("用户取消任务")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = max(1, int(request.options.vid_stride or 1))
        if total_frames > 0:
            points = [
                min(total_frames - 1, max(0, round(total_frames * (index + 1) / (sample_count + 1))))
                for index in range(sample_count)
            ]
        else:
            points = [0]

        for index, point in enumerate(points):
            if cancelled is not None and cancelled():
                raise RuntimeError("用户取消任务")
            cap.set(cv2.CAP_PROP_POS_FRAMES, point)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = temp_dir / f"{path.stem}_frame_{index}.jpg"
            if cv2.imwrite(str(frame_path), frame):
                frame_paths.append(frame_path)
                frame_points.append(point)

        if not frame_paths:
            raise RuntimeError("无法从视频中提取有效帧")
        if cancelled is not None and cancelled():
            raise RuntimeError("用户取消任务")

        selected_class_ids = _selected_species_class_ids(
            detector,
            request.options.selected_species_names,
        )
        detections = detector.detect_batch_species(
            [str(frame_path) for frame_path in frame_paths],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
            augment=request.options.use_augment,
            agnostic_nms=request.options.use_agnostic_nms,
            classes=selected_class_ids,
        )

        species_votes: Counter[str] = Counter()
        max_counts: dict[str, int] = {}
        boxes: list[dict[str, Any]] = []
        min_confidence: float | None = None
        names_map: dict[Any, str] = {}
        all_confidences: list[float] = []
        all_classes: list[float] = []
        for index, detection in enumerate(detections):
            if cancelled is not None and cancelled():
                raise RuntimeError("用户取消任务")
            if not isinstance(detection, dict):
                detection = {}
            detection_data = _serialize_detector_output(detector, detection)
            frame_species = _extract_species_list(detection_data)
            for species in frame_species:
                species_votes[species] += 1
            raw_counts = str(detection_data.get("物种数量") or "")
            count_values = [value.strip() for value in raw_counts.replace("，", ",").split(",")]
            for species_index, species in enumerate(frame_species):
                count = 1
                if species_index < len(count_values):
                    count = int(count_values[species_index]) if count_values[species_index].isdigit() else 1
                max_counts[species] = max(max_counts.get(species, 0), count)

            confidence = _coerce_float(detection_data.get("最低置信度"))
            if confidence is not None:
                min_confidence = confidence if min_confidence is None else min(min_confidence, confidence)
            names_map.update(detection_data.get("names_map") or {})
            all_confidences.extend(detection_data.get("all_confidences") or [])
            all_classes.extend(detection_data.get("all_classes") or [])
            frame_index = frame_points[index] if index < len(frame_points) else index
            timestamp = frame_index / fps if fps else None
            for box in detection_data.get("检测框") or []:
                if not isinstance(box, dict):
                    continue
                box_data = dict(box)
                box_data["frame_index"] = frame_index
                box_data["timestamp"] = timestamp
                boxes.append(box_data)

        species_names = list(max_counts.keys())
        detection_data = {
            "video_source": str(path),
            "video_mode": "fast",
            "sampled_frames": frame_points,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "total_frames_processed": total_frames,
            "vid_stride": 1,
            "物种名称": ",".join(species_names) if species_names else "空",
            "物种数量": ",".join(str(max_counts[name]) for name in species_names) if species_names else "空",
            "最低置信度": f"{min_confidence:.3f}" if min_confidence is not None else None,
            "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "检测框": boxes,
            "all_confidences": all_confidences,
            "all_classes": all_classes,
            "names_map": names_map,
            "species_votes": dict(species_votes),
        }
        _save_detection_data_for_path(path, detection_data, input_path)
        return _apply_detection_data(item, detection_data)
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return item.model_copy(update={"error": f"视频快速检测失败: {exc}"})
    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _video_output_dir(input_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        target_dir = Path(output_dir).expanduser().resolve()
    else:
        target_dir = input_path if input_path.is_dir() else input_path.parent
    output_path = target_dir / "video_results"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


DEFAULT_EXPORT_COLUMNS = [
    "文件名",
    "格式",
    "拍摄日期",
    "拍摄时间",
    "工作天数",
    "物种名称",
    "学名",
    "目名",
    "目拉丁名",
    "科名",
    "科拉丁名",
    "属名",
    "属拉丁名",
    "物种类型",
    "物种数量",
    "最低置信度",
    "独立探测首只",
    "备注",
]

def _update_species_database(species_name_str: str, species_type_str: str) -> None:
    """如果前端传来了物种名称和物种类型，则自动更新至物种数据库以供后续使用。"""
    if not species_name_str or not species_type_str:
        return

    # 按中英文逗号分割并清理空格
    names = [n.strip() for n in species_name_str.replace("，", ",").split(",") if n.strip()]
    types = [t.strip() for t in species_type_str.replace("，", ",").split(",") if t.strip()]

    if not names or not types:
        return

    db_path = Path(resource_path("res/species_database.db"))
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            # 兼容：如果表不存在则自动创建，匹配 load_species_types 查询的 'species' 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS species (
                    中文名 TEXT PRIMARY KEY,
                    物种类型 TEXT
                )
            """)

            for i, name in enumerate(names):
                if name in ("空", "未知鸟"):
                    continue
                # 若类型的数量少于物种数量，则使用最后一个类型补齐
                t = types[i] if i < len(types) else types[-1]
                
                if t and t != "待补全" and t != "空":
                    row = cursor.execute("SELECT 中文名 FROM species WHERE 中文名 = ?", (name,)).fetchone()
                    if row:
                        cursor.execute("UPDATE species SET 物种类型 = ? WHERE 中文名 = ?", (t, name))
                    else:
                        cursor.execute("INSERT INTO species (中文名, 物种类型) VALUES (?, ?)", (name, t))
            conn.commit()
    except Exception as e:
        print(f"Failed to auto-update species_database.db: {e}")

def mark_validation_item(request: ValidationMarkRequest) -> DetectionItem:
    input_path = Path(request.input_path).expanduser().resolve()
    path = Path(request.file_path).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")
    if not path.exists() or not path.is_file():
        raise ValueError(f"文件不存在: {path}")

    roots = _detection_db_search_roots(input_path, None)
    existing_data = dict(_load_detection_data_for_path(path, roots))
    existing_item = _build_metadata_item(path)
    if existing_data:
        existing_item = _apply_detection_data(existing_item, existing_data)

    validated: bool | None = True
    detection_data = existing_data

    if request.action == "unverified":
        if detection_data.get("最低置信度") == "人工校验":
            for key in ["最低置信度", "物种名称", "物种数量", "备注", "检测时间"]:
                detection_data.pop(key, None)
        validated = None
    else:
        species_name = (request.species_name or "").strip()
        species_count = (request.species_count or "").strip()

        if request.action == "empty":
            species_name = "空"
            species_count = "空"
        elif request.action == "correct":
            species_name = species_name or str(detection_data.get("物种名称") or "").strip()
            if not species_name and existing_item.species:
                species_name = ",".join(existing_item.species)
            species_count = species_count or str(detection_data.get("物种数量") or "").strip()
            if not species_count:
                species_count = str(max(1, len(existing_item.detection_boxes))) if species_name and species_name != "空" else "空"

        if species_name:
            detection_data["物种名称"] = species_name
        if species_count:
            detection_data["物种数量"] = species_count
        if request.species_type is not None:
            species_type = request.species_type.strip()
            if species_type:
                detection_data["物种类型"] = species_type
                _update_species_database(detection_data.get("物种名称", ""), species_type)
            else:
                detection_data.pop("物种类型", None)
        if request.remark is not None:
            detection_data["备注"] = request.remark
        detection_data["最低置信度"] = "人工校验"
        detection_data["检测时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _persist_validation_update(path, input_path, detection_data, validated)
    return _reload_validation_item(path, input_path)


def export_validation_data(request: ValidationExportRequest) -> ValidationExportResponse:
    input_path = Path(request.input_path).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    files = list(_resolve_supported_inputs(input_path))
    if not files:
        raise ValueError(f"输入路径中没有支持的媒体文件: {input_path}")

    detection_roots = _detection_db_search_roots(input_path, None)
    image_info_list: list[dict[str, Any]] = []
    earliest_date: datetime | None = None

    for path in files:
        metadata = _build_export_metadata(path)
        detection_data = _load_detection_data_for_path(path, detection_roots)
        if detection_data:
            metadata.update(detection_data)
        image_info_list.append(metadata)

        date_taken = metadata.get("拍摄日期对象")
        if isinstance(date_taken, datetime) and (earliest_date is None or date_taken < earliest_date):
            earliest_date = date_taken

    from system.data_processor import DataProcessor

    confidence_settings = dict(request.confidence_settings or {"global": 0.25})
    processed_data = DataProcessor.process_independent_detection(
        image_info_list,
        confidence_settings,
        min_frame_ratio=request.min_frame_ratio,
    )
    if earliest_date:
        processed_data = DataProcessor.calculate_working_days(processed_data, earliest_date)

    output_path = _resolve_validation_export_path(input_path, request)
    columns = request.columns_to_export or DEFAULT_EXPORT_COLUMNS
    success = DataProcessor.export_to_excel(
        processed_data,
        str(output_path),
        confidence_settings,
        file_format=request.file_format,
        columns_to_export=columns,
        min_frame_ratio=request.min_frame_ratio,
    )
    if not success:
        raise RuntimeError("导出文件失败，请查看后端日志获取详情。")

    return ValidationExportResponse(
        output_path=str(output_path),
        file_format=request.file_format,
        exported_count=len(processed_data),
    )


def _reload_validation_item(path: Path, input_path: Path) -> DetectionItem:
    roots = _detection_db_search_roots(input_path, None)
    item = _build_metadata_item(path)
    detection_data = _load_detection_data_for_path(path, roots)
    if detection_data:
        item = _apply_detection_data(item, detection_data)
    validation_index = _load_validation_index(roots)
    return _apply_validation_state(item, path.name, validation_index)


def _persist_validation_update(
    path: Path,
    input_path: Path,
    detection_data: dict[str, Any],
    validated: bool | None,
) -> None:
    from system.detection_db import delete_validation, get_db_path, init_db, upsert_detection, upsert_validation

    roots = [path.parent]
    roots.append(input_path if input_path.is_dir() else input_path.parent)

    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)

        db_path = get_db_path(str(resolved))
        init_db(db_path)
        if detection_data:
            upsert_detection(db_path, path.stem, path.name, detection_data)
        if validated is None:
            delete_validation(db_path, path.name)
        else:
            upsert_validation(db_path, path.name, validated)


def _build_export_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        dt_obj = datetime.fromtimestamp(path.stat().st_mtime)
        return {
            "文件名": path.name,
            "格式": path.suffix.replace(".", "").upper(),
            "拍摄日期": dt_obj.strftime("%Y-%m-%d"),
            "拍摄时间": dt_obj.strftime("%H:%M:%S"),
            "拍摄日期对象": dt_obj,
        }

    image = None
    try:
        metadata, image = ImageMetadataExtractor.extract_metadata(str(path), path.name)
        return metadata
    except Exception:
        return {
            "文件名": path.name,
            "格式": path.suffix.replace(".", "").upper(),
        }
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


def _resolve_validation_export_path(input_path: Path, request: ValidationExportRequest) -> Path:
    extension = ".xlsx" if request.file_format == "excel" else ".csv"
    if request.output_path:
        output_path = Path(request.output_path).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(extension)
    else:
        target_dir = input_path if input_path.is_dir() else input_path.parent
        folder_name = target_dir.name or "neri"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = target_dir / f"{folder_name}_validation_data_{timestamp}{extension}"

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


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
            extrasaction="ignore",
        )
        writer.writeheader()
        for item in results:
            row = item.model_dump()
            row["species"] = ",".join(item.species)
            writer.writerow(row)
    return output_path
