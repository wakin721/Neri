"""Processing services used by the FastAPI application.

The service intentionally reuses the existing Neri domain modules instead of
forking the detection logic.  Lightweight metadata indexing is available by
default; YOLO inference can be enabled per job when the runtime has the model
and ML dependencies installed.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS, XPU_ENABLED
from system.metadata_extractor import ImageMetadataExtractor
from system.utils import resource_path

from .models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ModelClassInfo,
    ModelInfo,
    ValidationBatchMarkRequest,
    ValidationExportRequest,
    ValidationExportResponse,
    ValidationMarkRequest,
)

logger = logging.getLogger(__name__)


class JobNotFoundError(KeyError):
    """Raised when a requested job id is not present in memory."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for API responses."""

    return datetime.now(timezone.utc).isoformat()


def _clear_torch_runtime_cache() -> None:
    try:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        xpu = getattr(torch, "xpu", None) if XPU_ENABLED else None
        if xpu is not None and xpu.is_available() and hasattr(xpu, "empty_cache"):
            xpu.empty_cache()
    except Exception:
        pass


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
        xpu = getattr(torch, "xpu", None) if XPU_ENABLED else None
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
        _load_detection_index(
            detection_db_roots,
            recursive=False,
            filenames={path.name for path in files},
        )
        if include_cached
        else {}
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
            input_dir=_request_input_label(request),
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
            snapshot = self._snapshot_unlocked(job_id)
        _clear_torch_runtime_cache()
        return snapshot

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
        detector = None
        try:
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id, initial_results)
                return
            input_path, files = _resolve_job_inputs(request)
            file_keys = {_path_key(path) for path in files}
            results = [
                item.model_copy(deep=True)
                for item in _unique_detection_items(initial_results or [])
                if _path_key(item.path) in file_keys
            ]
            skipped_paths = {_path_key(item.path) for item in results}
            remaining_files = [path for path in files if _path_key(path) not in skipped_paths]
            processed = len(results)
            job_started_at = time.monotonic()
            start_processed = processed
            total_files = len(files)
            missing_file_keys: set[str] = set()

            def progress_metrics(current_processed: int) -> dict[str, object]:
                elapsed = max(0.0, time.monotonic() - job_started_at)
                completed_this_run = max(0, current_processed - start_processed)
                speed = completed_this_run / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_files - current_processed)
                remaining_seconds = remaining / speed if speed > 0 else None
                return {
                    "elapsed_seconds": elapsed,
                    "speed": speed,
                    "remaining_seconds": remaining_seconds,
                }

            def existing_paths(paths: Iterable[Path]) -> list[Path]:
                nonlocal total_files
                existing: list[Path] = []
                removed = 0
                for path in paths:
                    if path.is_file():
                        existing.append(path)
                        continue
                    key = _path_key(path)
                    if key in missing_file_keys:
                        continue
                    missing_file_keys.add(key)
                    removed += 1
                    logger.info("Skipping deleted input file: %s", path)
                if removed:
                    total_files = max(processed, total_files - removed)
                    self._mutate_job(
                        job_id,
                        persist=False,
                        total=total_files,
                        processed=min(processed, total_files),
                        results=results,
                        **progress_metrics(min(processed, total_files)),
                        message=f"已跳过 {len(missing_file_keys)} 个已删除文件",
                    )
                return existing

            self._mutate_job(
                job_id,
                state=JobState.RUNNING,
                total=total_files,
                processed=processed,
                results=results,
                **progress_metrics(processed),
                message=(
                    f"继续任务：已完成 {processed}/{total_files}，剩余 {len(remaining_files)} 个文件"
                    if processed > 0
                    else f"预扫描发现 {total_files} 个支持的媒体文件"
                    if not request.options.enable_detection
                    else (
                        f"继续任务：已完成 {processed}/{total_files}，剩余 {len(remaining_files)} 个文件"
                        if processed > 0
                        else f"发现 {total_files} 个支持的媒体文件"
                    )
                ),
            )

            detection_db_roots = (
                _preview_detection_db_roots(input_path, request.output_dir, files)
                if request.options.enable_detection
                else []
            )
            detection_index = (
                _load_detection_index(
                    detection_db_roots,
                    filenames={path.name for path in files},
                )
                if request.options.enable_detection
                else {}
            )
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
                    if not existing_paths([path]):
                        continue
                    item = _build_fast_metadata_item(path)
                    results.append(item)
                    processed += 1
                    media_label = "照片" if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else "视频"
                    self._mutate_job(
                        job_id,
                        processed=processed,
                        results=results,
                        **progress_metrics(processed),
                        message=f"正在预扫描{media_label} {processed}/{total_files}: {path.name}",
                    )
            else:
                image_files = [
                    path for path in remaining_files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
                video_files = [
                    path for path in remaining_files if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
                ]
                batch_size = max(1, int(request.options.batch_size or 1))

                image_batches = list(_chunked(image_files, batch_size))
                image_preload_workers = 1
                image_preload_lookahead = min(2, len(image_batches))
                preload_executor = (
                    ThreadPoolExecutor(max_workers=image_preload_workers)
                    if image_batches and hasattr(detector, "preload_batch_data")
                    else None
                )
                preload_futures: dict[int, Any] = {}

                def submit_preload(batch_index: int) -> None:
                    if preload_executor is None or batch_index >= len(image_batches):
                        return
                    if batch_index in preload_futures:
                        return
                    preload_futures[batch_index] = preload_executor.submit(
                        detector.preload_batch_data,
                        [str(path) for path in image_batches[batch_index]],
                    )

                def fill_preload_queue(current_batch_index: int) -> None:
                    if preload_executor is None:
                        return
                    end_index = min(
                        len(image_batches),
                        current_batch_index + image_preload_lookahead,
                    )
                    for preload_index in range(current_batch_index, end_index):
                        submit_preload(preload_index)

                def take_preloaded(batch_index: int):
                    future = preload_futures.pop(batch_index, None)
                    if future is None:
                        return None
                    try:
                        return future.result()
                    except Exception as exc:
                        logger.warning("Image batch preload failed: %s", exc)
                        return None

                fill_preload_queue(0)
                try:
                    for batch_index, original_batch in enumerate(image_batches):
                        if self._is_cancelled(job_id):
                            self._mark_cancelled(job_id, results)
                            return
                        batch_started = time.perf_counter()
                        fill_preload_queue(batch_index)
                        batch = existing_paths(original_batch)
                        metadata_paths = list(batch)
                        metadata_started = time.perf_counter()
                        batch_items: list[DetectionItem] = []
                        for path in batch:
                            item = _build_metadata_item(path)
                            db_detection_data = detection_index.get(path.stem)
                            if db_detection_data:
                                item = _apply_detection_data(item, db_detection_data)
                            batch_items.append(item)
                        metadata_elapsed = time.perf_counter() - metadata_started

                        preload_wait_started = time.perf_counter()
                        preloaded_data = take_preloaded(batch_index)
                        preload_wait_elapsed = time.perf_counter() - preload_wait_started
                        fill_preload_queue(batch_index + 1)
                        batch = existing_paths(batch)
                        if batch != original_batch:
                            # Preloaded indices refer to the original batch.
                            # Reload the surviving paths so deleted files cannot
                            # remain processable merely because they were read ahead.
                            preloaded_data = None
                            surviving_items = {
                                path: item
                                for path, item in zip(metadata_paths, batch_items)
                            }
                            batch_items = [
                                surviving_items[path]
                                for path in batch
                                if path in surviving_items
                            ]
                        if not batch:
                            continue
                        detect_started = time.perf_counter()
                        detected_items = _detect_image_batch(
                            detector,
                            batch,
                            batch_items,
                            request,
                            input_path,
                            preloaded_data=preloaded_data,
                        )
                        detect_elapsed = time.perf_counter() - detect_started
                        logger.info(
                            (
                                "Image batch timing: index=%d/%d size=%d "
                                "metadata=%.3fs preload_wait=%.3fs detect_save=%.3fs total=%.3fs"
                            ),
                            batch_index + 1,
                            len(image_batches),
                            len(batch),
                            metadata_elapsed,
                            preload_wait_elapsed,
                            detect_elapsed,
                            time.perf_counter() - batch_started,
                        )
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
                            **progress_metrics(processed),
                            message=(
                                f"正在批处理照片 {processed}/{total_files} "
                                f"(batch={len(batch)}): {last_path.name}"
                            ),
                        )
                finally:
                    if preload_executor is not None:
                        preload_executor.shutdown(wait=False, cancel_futures=True)

                if request.options.video_mode == "fast":
                    for original_batch in _chunked(video_files, batch_size):
                        if self._is_cancelled(job_id):
                            self._mark_cancelled(job_id, results)
                            return
                        batch = existing_paths(original_batch)
                        if not batch:
                            continue
                        batch_items: list[DetectionItem] = []
                        for path in batch:
                            item = _build_metadata_item(path)
                            db_detection_data = detection_index.get(path.stem)
                            if db_detection_data:
                                item = _apply_detection_data(item, db_detection_data)
                            batch_items.append(item)

                        detected_items = _detect_video_fast_batch(
                            detector,
                            batch,
                            batch_items,
                            request,
                            input_path,
                            cancelled=lambda: self._is_cancelled(job_id),
                        )
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
                            **progress_metrics(processed),
                            message=(
                                f"正在批量快速处理视频 {processed}/{total_files} "
                                f"(batch={len(batch)}): {last_path.name}"
                            ),
                        )
                else:
                    video_files = existing_paths(video_files)
                    video_items: list[DetectionItem] = []
                    for path in video_files:
                        item = _build_metadata_item(path)
                        db_detection_data = detection_index.get(path.stem)
                        if db_detection_data:
                            item = _apply_detection_data(item, db_detection_data)
                        video_items.append(item)

                    video_worker_count = _tracked_video_worker_count(
                        len(video_files),
                        request.options.thread_count,
                    )
                    if video_worker_count <= 1:
                        for path, item in zip(video_files, video_items):
                            if self._is_cancelled(job_id):
                                self._mark_cancelled(job_id, results)
                                return
                            if not existing_paths([path]):
                                continue

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
                                **progress_metrics(processed),
                                message=f"正在处理视频 {processed}/{total_files}: {path.name}",
                            )
                    else:
                        logger.info(
                            "Tracked video parallel workers: videos=%d workers=%d",
                            len(video_files),
                            video_worker_count,
                        )
                        base_results = list(results)
                        detected_video_items: list[DetectionItem | None] = [
                            None for _ in video_items
                        ]
                        worker_state = threading.local()
                        primary_detector_lock = threading.Lock()
                        primary_detector_available = True

                        def detector_for_worker():
                            nonlocal primary_detector_available
                            worker_detector = getattr(worker_state, "detector", None)
                            if worker_detector is not None:
                                return worker_detector
                            with primary_detector_lock:
                                if primary_detector_available:
                                    primary_detector_available = False
                                    worker_state.detector = detector
                                    return detector
                            worker_detector = _load_detector(
                                request.options.model_path,
                                request.options.classification_model_path,
                            )
                            worker_state.detector = worker_detector
                            return worker_detector

                        def detect_one_video(video_index: int) -> tuple[int, DetectionItem]:
                            _raise_if_cancelled(lambda: self._is_cancelled(job_id))
                            path = video_files[video_index]
                            if not path.is_file():
                                raise FileNotFoundError(f"文件已删除: {path}")
                            item = video_items[video_index]
                            detected_item = _detect_video(
                                detector_for_worker(),
                                path,
                                item,
                                request,
                                input_path,
                                cancelled=lambda: self._is_cancelled(job_id),
                            )
                            return video_index, detected_item

                        with ThreadPoolExecutor(
                            max_workers=video_worker_count,
                            thread_name_prefix="neri-video",
                        ) as video_executor:
                            future_to_index = {
                                video_executor.submit(detect_one_video, index): index
                                for index in range(len(video_files))
                            }
                            for future in as_completed(future_to_index):
                                if self._is_cancelled(job_id):
                                    self._mark_cancelled(job_id, results)
                                    return
                                video_index = future_to_index[future]
                                try:
                                    completed_index, item = future.result()
                                except Exception as exc:
                                    if self._is_cancelled(job_id):
                                        self._mark_cancelled(job_id, results)
                                        return
                                    completed_index = video_index
                                    if not video_files[completed_index].is_file():
                                        existing_paths(
                                            [video_files[completed_index]]
                                        )
                                        continue
                                    item = video_items[completed_index].model_copy(
                                        update={"error": f"视频检测失败: {exc}"}
                                    )
                                detected_video_items[completed_index] = item
                                processed += 1
                                current_video_results = [
                                    detected
                                    for detected in detected_video_items
                                    if detected is not None
                                ]
                                results = [*base_results, *current_video_results]
                                self._mutate_job(
                                    job_id,
                                    processed=processed,
                                    results=results,
                                    **progress_metrics(processed),
                                    message=(
                                        f"正在并发处理视频 {processed}/{total_files} "
                                        f"(workers={video_worker_count}): "
                                        f"{video_files[completed_index].name}"
                                    ),
                                )
                        _clear_torch_runtime_cache()

            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id, results)
                return
            results = [
                item
                for item in results
                if item.path and Path(item.path).is_file()
            ]
            total_files = len(results)
            exported_path = _export_results(request.output_dir, results)
            message = "预扫描完成" if not request.options.enable_detection else "处理完成"
            if exported_path is not None:
                message = f"处理完成，结果已导出到 {exported_path}"
            self._mutate_job(
                job_id,
                state=JobState.COMPLETED,
                total=total_files,
                processed=total_files,
                results=results,
                **progress_metrics(total_files),
                message=message,
            )
        except Exception as exc:  # noqa: BLE001 - converted to an API-visible job failure
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
            else:
                self._mutate_job(job_id, state=JobState.FAILED, error=str(exc), message="处理失败")
        finally:
            if detector is not None and hasattr(detector, "cleanup_runtime_cache"):
                try:
                    detector.cleanup_runtime_cache(clear_cuda_cache=self._is_cancelled(job_id))
                except Exception as exc:  # noqa: BLE001 - cleanup should not mask job status
                    logger.warning("Detector cleanup failed: %s", exc)
            with self._lock:
                self._active_job_ids.discard(job_id)
                self._save_state_unlocked()


def _request_input_label(request: CreateJobRequest) -> str:
    if request.input_paths:
        if len(request.input_paths) == 1:
            return request.input_paths[0]
        return f"已选择 {len(request.input_paths)} 个输入路径"
    return request.input_dir or ""


def _resolve_job_inputs(request: CreateJobRequest) -> tuple[Path, list[Path]]:
    raw_inputs = request.input_paths or (
        [request.input_dir] if request.input_dir else []
    )
    sources: list[Path] = []
    for raw_input in raw_inputs:
        if not raw_input:
            continue
        source = Path(raw_input).expanduser().resolve()
        if not source.exists():
            raise ValueError(f"输入路径不存在: {source}")
        sources.append(source)

    files: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        for path in _resolve_supported_inputs(source):
            key = _path_key(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)

    if not files:
        raise ValueError(
            f"输入路径中没有支持的媒体文件: {_request_input_label(request)}"
        )

    input_path = sources[0] if len(sources) == 1 else _common_input_root(files)
    return input_path, files


def _common_input_root(files: list[Path]) -> Path:
    if not files:
        return Path.cwd()
    roots = [path.parent for path in files]
    if len(roots) == 1:
        return roots[0]
    try:
        common_path = Path(os.path.commonpath([str(root) for root in roots]))
    except ValueError:
        return roots[0]
    return (
        common_path
        if common_path in roots and common_path.exists() and common_path.is_dir()
        else roots[0]
    )


def _resolve_supported_inputs(input_path: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    if input_path.is_file():
        if input_path.suffix.lower() in supported:
            yield input_path
        return

    if input_path.is_dir():
        yield from _iter_supported_files(input_path)


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _tracked_video_worker_count(item_count: int, thread_count: int) -> int:
    if item_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    configured = max(1, min(int(thread_count or 1), 8))
    cpu_budget = max(1, cpu_count // 2)
    # Full video tracking loads model state and decodes frames per worker, so
    # keep the cap conservative even when a larger value is requested.
    return max(1, min(item_count, configured, cpu_budget))


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
        if _is_generated_favorite_export_path(path):
            continue
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


def _is_generated_favorite_export_path(path: Path) -> bool:
    return any(part.endswith("_收藏照片") for part in path.parts)


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

    resolved_model_path = _resolve_model_path(model_path) if model_path else None
    resolved_classification_path = _resolve_classification_model_path(classification_model_path)
    if classification_model_path and resolved_classification_path is None:
        raise FileNotFoundError(f"分类模型文件不存在: {classification_model_path}")
    if resolved_model_path is None and resolved_classification_path is None:
        raise ValueError("探测模型和分类模型至少需要选择一个。")

    detector = ImageProcessor(
        str(resolved_model_path) if resolved_model_path is not None else None
    )
    if resolved_classification_path is not None:
        detector.load_cls_model(str(resolved_classification_path))
    if detector.model is None and detector.cls_model is None:
        raise RuntimeError("未能加载所选的探测模型或分类模型。")
    return detector


def _detector_model_names(detector) -> dict[int, str]:
    model = getattr(detector, "model", None)
    if model is None:
        model = getattr(detector, "cls_model", None)
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


def _selected_inference_class_ids(
    detector,
    selected_species_names: list[str],
) -> list[int] | None:
    if (
        getattr(detector, "model", None) is not None
        and getattr(detector, "cls_model", None) is not None
    ):
        # With both models enabled the species selection refers to the
        # classifier that provides the final name. Keep all detector classes
        # so potentially relevant crops are not discarded before classifying.
        return None
    return _selected_species_class_ids(detector, selected_species_names)


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
    filenames: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for db_path in _candidate_detection_dbs_for_roots(search_roots, recursive=recursive):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    "SELECT base_name, image_filename, detection_json FROM detections"
                ).fetchall()
            for base_name, image_filename, detection_json in rows:
                if filenames is not None and str(image_filename) not in filenames:
                    continue
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
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    (
                        "SELECT image_filename, detection_json "
                        "FROM detections WHERE base_name = ?"
                    ),
                    (base_name,),
                ).fetchone()
            if row is None or str(row[0]) != path.name:
                continue
            data = json.loads(row[1])
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
        selected_candidates_data = (
            getattr(result, "selected_candidates_data", {}) or {}
        )
        classification_filtered_boxes = (
            getattr(result, "classification_filtered_boxes", set()) or set()
        )
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
            if index in classification_filtered_boxes:
                continue
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
                best_candidate = selected_candidates_data.get(
                    index,
                    candidates[0],
                )
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
        "分类候选项": detection.get("分类候选项", []),
        "confidence_priority": detection.get("confidence_priority"),
        "confidence_weights": detection.get("confidence_weights"),
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
        if not path.is_file():
            logger.info("Not saving detection data for deleted file: %s", path)
            continue
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
    except Exception as exc:
        logger.warning("Detection DB bulk helpers unavailable: %s", exc)
        return

    for root, payloads in payloads_by_root.items():
        try:
            db_path = get_db_path(str(root))
            init_db(db_path)
            upsert_detections_bulk(db_path, payloads)
            upsert_validation_bulk(db_path, validations_by_root.get(root, []))
        except Exception as exc:
            logger.warning("Failed to save detection batch to SQLite at %s: %s", root, exc)
            continue


def _detect_image_batch(
    detector,
    paths: list[Path],
    items: list[DetectionItem],
    request: CreateJobRequest,
    input_path: Path,
    preloaded_data: Any = None,
) -> list[DetectionItem]:
    surviving_pairs = [
        (path, item)
        for path, item in zip(paths, items)
        if path.is_file()
    ]
    if not surviving_pairs:
        return []
    if len(surviving_pairs) != len(paths):
        paths = [path for path, _ in surviving_pairs]
        items = [item for _, item in surviving_pairs]
        preloaded_data = None
    try:
        selected_class_ids = _selected_inference_class_ids(
            detector,
            request.options.selected_species_names,
        )
        detect_started = time.perf_counter()
        detections = detector.detect_batch_species(
            [str(path) for path in paths],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
            augment=request.options.use_augment,
            agnostic_nms=request.options.use_agnostic_nms,
            classes=selected_class_ids,
            imgsz=request.options.imgsz,
            confidence_priority=request.options.confidence_priority,
            selected_species_names=request.options.selected_species_names,
            preloaded_data=preloaded_data,
        )
        detect_elapsed = time.perf_counter() - detect_started
        detected_items: list[DetectionItem] = []
        detection_payloads: list[tuple[Path, dict[str, Any]]] = []
        serialize_started = time.perf_counter()
        for index, (path, item) in enumerate(zip(paths, items)):
            detection = detections[index] if index < len(detections) else {}
            if not isinstance(detection, dict):
                detection = {}
            detection_data = _serialize_detector_output(detector, detection)
            detection_payloads.append((path, detection_data))
            detected_items.append(_apply_detection_data(item, detection_data))
        serialize_elapsed = time.perf_counter() - serialize_started
        save_started = time.perf_counter()
        _save_detection_data_batch(detection_payloads, input_path)
        save_elapsed = time.perf_counter() - save_started
        logger.info(
            (
                "Image batch backend timing: size=%d detect=%.3fs "
                "serialize=%.3fs save=%.3fs"
            ),
            len(paths),
            detect_elapsed,
            serialize_elapsed,
            save_elapsed,
        )
        return detected_items
    except Exception as exc:  # noqa: BLE001 - preserve per-file metadata on batch failure
        return [item.model_copy(update={"error": f"批量检测失败: {exc}"}) for item in items]


def _detect_image(detector, path: Path, item: DetectionItem, request: CreateJobRequest, input_path: Path) -> DetectionItem:
    try:
        selected_class_ids = _selected_inference_class_ids(
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
            imgsz=request.options.imgsz,
            confidence_priority=request.options.confidence_priority,
            selected_species_names=request.options.selected_species_names,
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
    classification_only = (
        getattr(detector, "model", None) is None
        and getattr(detector, "cls_model", None) is not None
    )
    if request.options.video_mode == "fast" or classification_only:
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
    output_dir: Path | None = None
    try:
        selected_class_ids = _selected_species_class_ids(
            detector,
            request.options.selected_species_names,
        )
        output_dir = _video_processing_temp_dir()
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
            imgsz=request.options.imgsz,
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
    finally:
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


def _detect_video_fast(
    detector,
    path: Path,
    item: DetectionItem,
    request: CreateJobRequest,
    input_path: Path,
    cancelled=None,
) -> DetectionItem:
    return _detect_video_fast_batch(
        detector,
        [path],
        [item],
        request,
        input_path,
        cancelled=cancelled,
    )[0]


def _detect_video_fast_batch(
    detector,
    paths: list[Path],
    items: list[DetectionItem],
    request: CreateJobRequest,
    input_path: Path,
    cancelled=None,
) -> list[DetectionItem]:
    import shutil
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="neri_video_frames_"))
    result_items: list[DetectionItem | None] = [None for _ in items]
    video_infos: list[dict[str, Any] | None] = [None for _ in items]
    frame_records: list[dict[str, Any]] = []

    try:
        batch_started = time.perf_counter()
        sample_started = time.perf_counter()
        sample_count = max(1, int(request.options.vid_stride or 1))
        classification_only = (
            getattr(detector, "model", None) is None
            and getattr(detector, "cls_model", None) is not None
        )
        frame_stride = (
            max(1, int(request.options.vid_stride or 1))
            if classification_only and request.options.video_mode == "all"
            else None
        )
        for video_index, (path, item) in enumerate(zip(paths, items)):
            try:
                _raise_if_cancelled(cancelled)
                info = _sample_fast_video_frames(
                    path,
                    temp_dir / f"video_{video_index}",
                    sample_count,
                    frame_stride=frame_stride,
                    cancelled=cancelled,
                )
                video_infos[video_index] = info
                for frame_order, (frame_path, frame_index) in enumerate(
                    zip(info["frame_paths"], info["frame_points"])
                ):
                    frame_records.append(
                        {
                            "video_index": video_index,
                            "frame_order": frame_order,
                            "frame_index": frame_index,
                            "path": frame_path,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - keep the rest of the video batch alive
                if cancelled is not None and cancelled():
                    raise
                result_items[video_index] = item.model_copy(
                    update={"error": f"视频快速检测失败: {exc}"}
                )

        sample_elapsed = time.perf_counter() - sample_started
        inference_elapsed = 0.0
        if frame_records:
            selected_class_ids = _selected_inference_class_ids(
                detector,
                request.options.selected_species_names,
            )
            detections_by_video: list[list[tuple[int, int, dict[str, Any]]]] = [
                [] for _ in items
            ]
            frame_batch_size = max(1, int(request.options.batch_size or 1))
            inference_started = time.perf_counter()
            for frame_batch in _chunked(frame_records, frame_batch_size):
                _raise_if_cancelled(cancelled)
                detections = detector.detect_batch_species(
                    [str(record["path"]) for record in frame_batch],
                    conf=request.options.confidence,
                    iou=request.options.iou,
                    use_fp16=request.options.use_fp16,
                    augment=request.options.use_augment,
                    agnostic_nms=request.options.use_agnostic_nms,
                    classes=selected_class_ids,
                    imgsz=request.options.imgsz,
                    confidence_priority=request.options.confidence_priority,
                    selected_species_names=request.options.selected_species_names,
                )
                _raise_if_cancelled(cancelled)
                for index, record in enumerate(frame_batch):
                    detection = detections[index] if index < len(detections) else {}
                    if not isinstance(detection, dict):
                        detection = {}
                    detections_by_video[record["video_index"]].append(
                        (record["frame_order"], record["frame_index"], detection)
                    )
            inference_elapsed = time.perf_counter() - inference_started
        else:
            detections_by_video = [[] for _ in items]

        detection_payloads: list[tuple[Path, dict[str, Any]]] = []
        for video_index, (path, item) in enumerate(zip(paths, items)):
            if result_items[video_index] is not None:
                continue
            info = video_infos[video_index]
            if info is None:
                result_items[video_index] = item.model_copy(
                    update={"error": "视频快速检测失败: 无法从视频中提取有效帧"}
                )
                continue
            frame_detections = sorted(
                detections_by_video[video_index],
                key=lambda record: record[0],
            )
            detection_data = _build_fast_video_detection_data(
                detector,
                path,
                info,
                frame_detections,
                video_mode=request.options.video_mode,
            )
            detection_payloads.append((path, detection_data))
            result_items[video_index] = _apply_detection_data(item, detection_data)
        serialize_elapsed = time.perf_counter() - serialize_started

        save_elapsed = 0.0
        if detection_payloads:
            save_started = time.perf_counter()
            _save_detection_data_batch(detection_payloads, input_path)
            save_elapsed = time.perf_counter() - save_started

        logger.info(
            (
                "Fast video batch timing: videos=%d frames=%d sample=%.3fs "
                "infer=%.3fs serialize=%.3fs save=%.3fs total=%.3fs"
            ),
            len(paths),
            len(frame_records),
            sample_elapsed,
            inference_elapsed,
            serialize_elapsed,
            save_elapsed,
            time.perf_counter() - batch_started,
        )

        return [
            result_item if result_item is not None else item
            for item, result_item in zip(items, result_items)
        ]
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return [
            item.model_copy(update={"error": f"视频批量快速检测失败: {exc}"})
            for item in items
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _raise_if_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("用户取消任务")


def _sample_fast_video_frames(
    path: Path,
    temp_dir: Path,
    sample_count: int,
    frame_stride: int | None = None,
    cancelled=None,
) -> dict[str, Any]:
    import cv2

    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frame_points: list[int] = []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")
        _raise_if_cancelled(cancelled)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and frame_stride is not None:
            points = list(range(0, total_frames, max(1, frame_stride)))
        elif total_frames > 0:
            points = [
                min(total_frames - 1, max(0, round(total_frames * (index + 1) / (sample_count + 1))))
                for index in range(sample_count)
            ]
        else:
            points = [0]

        for index, point in enumerate(points):
            _raise_if_cancelled(cancelled)
            cap.set(cv2.CAP_PROP_POS_FRAMES, point)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = temp_dir / f"frame_{index}.jpg"
            if cv2.imwrite(str(frame_path), frame):
                frame_paths.append(frame_path)
                frame_points.append(point)

        if not frame_paths:
            raise RuntimeError("无法从视频中提取有效帧")

        return {
            "frame_paths": frame_paths,
            "frame_points": frame_points,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        }
    finally:
        cap.release()


def _build_fast_video_detection_data(
    detector,
    path: Path,
    video_info: dict[str, Any],
    frame_detections: list[tuple[int, int, dict[str, Any]]],
    video_mode: str = "fast",
) -> dict[str, Any]:
    from collections import Counter

    species_votes: Counter[str] = Counter()
    max_counts: dict[str, int] = {}
    boxes: list[dict[str, Any]] = []
    min_confidence: float | None = None
    names_map: dict[Any, str] = {}
    all_confidences: list[float] = []
    all_classes: list[float] = []
    fps = video_info.get("fps") or 25

    for _frame_order, frame_index, detection in frame_detections:
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
        timestamp = frame_index / fps if fps else None
        for box in detection_data.get("检测框") or []:
            if not isinstance(box, dict):
                continue
            box_data = dict(box)
            box_data["frame_index"] = frame_index
            box_data["timestamp"] = timestamp
            boxes.append(box_data)

    species_names = list(max_counts.keys())
    total_frames = int(video_info.get("total_frames") or 0)
    return {
        "video_source": str(path),
        "video_mode": video_mode,
        "sampled_frames": video_info.get("frame_points") or [],
        "width": int(video_info.get("width") or 0),
        "height": int(video_info.get("height") or 0),
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


def _video_processing_temp_dir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="neri_video_track_"))


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
    items = mark_validation_items(
        ValidationBatchMarkRequest(
            input_path=request.input_path,
            file_paths=[request.file_path],
            action=request.action,
            species_name=request.species_name,
            species_count=request.species_count,
            species_type=request.species_type,
            remark=request.remark,
        )
    )
    if not items:
        raise ValueError("没有可标记的文件")
    return items[0]


def mark_validation_items(request: ValidationBatchMarkRequest) -> list[DetectionItem]:
    input_path = Path(request.input_path).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in request.file_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {path}")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        paths.append(path)
    if not paths:
        return []

    roots = _preview_detection_db_roots(input_path, None, paths)
    detection_indexes: dict[Path, dict[str, dict[str, Any]]] = {}

    def detection_data_for(path: Path) -> dict[str, Any]:
        for root in _unique_existing_dirs([path.parent, *roots]):
            index = detection_indexes.get(root)
            if index is None:
                index = _load_detection_index(
                    [root],
                    recursive=False,
                    filenames={path.name},
                )
                detection_indexes[root] = index
            data = index.get(path.stem)
            if isinstance(data, dict):
                return dict(data)
        return {}

    updates: list[tuple[Path, dict[str, Any], bool | None]] = []
    updated_items: list[DetectionItem] = []
    species_database_updates: set[tuple[str, str]] = set()
    for path in paths:
        detection_data, validated = _build_validation_update(
            request,
            path,
            detection_data_for(path),
        )
        updates.append((path, detection_data, validated))

        item = _build_fast_metadata_item(path)
        if detection_data:
            item = _apply_detection_data(item, detection_data)
        item = item.model_copy(update={"validated": validated})
        updated_items.append(item)

        if request.species_type is not None:
            species_type = request.species_type.strip()
            species_name = str(detection_data.get("物种名称") or "").strip()
            if species_name and species_type:
                species_database_updates.add((species_name, species_type))

    _persist_validation_updates(updates, input_path)
    for species_name, species_type in species_database_updates:
        _update_species_database(species_name, species_type)

    return updated_items


def _build_validation_update(
    request: ValidationMarkRequest | ValidationBatchMarkRequest,
    path: Path,
    existing_data: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
    existing_data = dict(existing_data)
    existing_item = _build_fast_metadata_item(path)
    if existing_data:
        existing_item = _apply_detection_data(existing_item, existing_data)
    validated: bool | None = True
    detection_data = existing_data

    if request.action == "unverified":
        if detection_data.get("最低置信度") == "人工校验":
            for key in ["最低置信度", "物种名称", "物种类型", "物种数量", "备注", "检测时间"]:
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
                species_count = (
                    str(max(1, len(existing_item.detection_boxes)))
                    if species_name and species_name != "空"
                    else "空"
                )

        if species_name:
            detection_data["物种名称"] = species_name
        if species_count:
            detection_data["物种数量"] = species_count
        if request.species_type is not None:
            species_type = request.species_type.strip()
            if species_type:
                detection_data["物种类型"] = species_type
            else:
                detection_data.pop("物种类型", None)
        if request.remark is not None:
            detection_data["备注"] = request.remark
        detection_data["最低置信度"] = "人工校验"
        detection_data["检测时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return detection_data, validated


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
        metadata["_source_path"] = str(path)
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

    processed_data.sort(key=_export_filename_sort_key)

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

    favorite_output_dir: Path | None = None
    favorite_exported_count = 0
    if request.export_favorite_photos:
        favorite_output_dir, favorite_exported_count = _export_favorite_photos(
            files,
            request.favorite_photo_paths,
            output_path,
            _export_species_by_path(processed_data),
        )

    return ValidationExportResponse(
        output_path=str(output_path),
        file_format=request.file_format,
        exported_count=len(processed_data),
        favorite_output_dir=str(favorite_output_dir) if favorite_output_dir else None,
        favorite_exported_count=favorite_exported_count,
    )


_FILENAME_SORT_PART_PATTERN = re.compile(r"\d+|\D+")


def _export_filename_sort_key(row: dict[str, Any]) -> tuple[tuple[int, int | str, int | str], ...]:
    filename = str(row.get("文件名") or row.get("filename") or "")
    parts: list[tuple[int, int | str, int | str]] = []
    for part in _FILENAME_SORT_PART_PATTERN.findall(filename):
        if part.isdigit():
            parts.append((0, int(part), len(part)))
        else:
            parts.append((1, part.casefold(), part))
    return tuple(parts)


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
    _persist_validation_updates([(path, detection_data, validated)], input_path)


def _persist_validation_updates(
    updates: Iterable[tuple[Path, dict[str, Any], bool | None]],
    input_path: Path,
) -> None:
    from system.detection_db import (
        delete_validation_bulk,
        get_db_path,
        init_db,
        upsert_detections_bulk,
        upsert_validation_bulk,
    )

    payloads_by_root: dict[Path, list[tuple[str, str, dict[str, Any]]]] = {}
    validation_upserts_by_root: dict[Path, list[tuple[str, bool]]] = {}
    validation_deletes_by_root: dict[Path, list[str]] = {}

    for path, detection_data, validated in updates:
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
            payloads_by_root.setdefault(resolved, []).append(
                (path.stem, path.name, detection_data)
            )
            if validated is None:
                validation_deletes_by_root.setdefault(resolved, []).append(
                    path.name
                )
            else:
                validation_upserts_by_root.setdefault(resolved, []).append(
                    (path.name, validated)
                )

    for root, payloads in payloads_by_root.items():
        db_path = get_db_path(str(root))
        init_db(db_path)
        upsert_detections_bulk(db_path, payloads)
        delete_validation_bulk(db_path, validation_deletes_by_root.get(root, []))
        upsert_validation_bulk(db_path, validation_upserts_by_root.get(root, []))


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


def _export_favorite_photos(
    files: Iterable[Path],
    favorite_photo_paths: Iterable[str],
    output_path: Path,
    species_by_path: dict[str, str],
) -> tuple[Path | None, int]:
    favorite_keys: list[str] = []
    seen_keys: set[str] = set()
    for path in favorite_photo_paths:
        key = _path_key(path)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        favorite_keys.append(key)
    if not favorite_keys:
        return None, 0

    available_images = {
        _path_key(path): path
        for path in files
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }
    selected_files = [
        available_images[key]
        for key in favorite_keys
        if key in available_images and available_images[key].is_file()
    ]
    if not selected_files:
        return None, 0

    target_dir = output_path.parent / f"{output_path.stem}_收藏照片"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in selected_files:
        species_folder = target_dir / _species_export_folder_name(
            species_by_path.get(_path_key(source))
        )
        species_folder.mkdir(parents=True, exist_ok=True)
        destination = _unique_favorite_export_path(species_folder, source.name)
        shutil.copy2(source, destination)
        copied += 1

    return target_dir, copied


def _export_species_by_path(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    species_by_path: dict[str, str] = {}
    for row in rows:
        key = _path_key(row.get("_source_path"))
        if not key:
            continue
        species = str(row.get("物种名称") or "").strip()
        if species:
            species_by_path[key] = species
    return species_by_path


def _species_export_folder_name(species: str | None) -> str:
    name = (species or "").strip() or "未知物种"
    name = re.sub(r"\s*[,，]\s*", "、", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = "未知物种"
    return name[:80]


def _unique_favorite_export_path(target_dir: Path, filename: str) -> Path:
    source_name = Path(filename)
    stem = source_name.stem or "photo"
    suffix = source_name.suffix
    candidate = target_dir / source_name.name
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


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
