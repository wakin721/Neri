"""Active backend facade for the versioned DINOv3 joint model.

The large legacy service module remains a temporary implementation detail for
non-DINO media/job behavior. Its DINO hooks select the validated joint-model
runtime; legacy classification checkpoint formats are not supported.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from system.dinov3.display import model_display_name

from . import services_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


class ProcessingJobManager(_legacy.ProcessingJobManager):
    """Persist resumable detection progress without making UI updates disk-bound."""

    _resume_checkpoint_interval_seconds = 1.0

    def __init__(self, *args, **kwargs) -> None:
        self._resume_checkpoint_at: dict[str, float] = {}
        from .job_state_store import JobStateStore

        self._state_store = JobStateStore(job_state_path())
        super().__init__(*args, **kwargs)

    def _load_state(self) -> None:
        data = self._state_store.load()
        if data is None:
            # Import the previous JSON checkpoint once; retain it for recovery.
            super()._load_state()
            return
        for job_id, raw_job in data["jobs"].items():
            job = JobSummary(**raw_job).model_copy(update={"active": False})
            self._jobs[job_id] = job
        self._job_requests = {
            job_id: CreateJobRequest(**request)
            for job_id, request in data["requests"].items()
        }
        self._state_store.remember(self._jobs, self._job_requests)
        for job_id, job in tuple(self._jobs.items()):
            if job.state in {JobState.QUEUED, JobState.RUNNING}:
                self._jobs[job_id] = job.model_copy(update={
                    "state": JobState.CANCELLED,
                    "message": "后端已重启，任务已暂停，可继续任务",
                    "updated_at": utc_now(),
                })
        self._save_state_unlocked()

    def _should_checkpoint_detection_progress(
        self,
        job_id: str,
        changes: dict[str, object],
    ) -> bool:
        if "processed" not in changes or "results" not in changes:
            return False
        try:
            next_processed = int(changes["processed"])
        except (TypeError, ValueError):
            return False
        with self._lock:
            request = self._job_requests.get(job_id)
            current = self._jobs.get(job_id)
            return bool(
                request is not None
                and request.options.enable_detection
                and current is not None
                and next_processed > current.processed
            )

    def _mutate_job(
        self,
        job_id: str,
        *,
        persist: bool = True,
        **changes: object,
    ) -> None:
        checkpoint_progress = self._should_checkpoint_detection_progress(job_id, changes)
        if not persist and checkpoint_progress:
            now = time.monotonic()
            last = self._resume_checkpoint_at.get(job_id)
            if last is None or now - last >= self._resume_checkpoint_interval_seconds:
                persist = True
        super()._mutate_job(job_id, persist=persist, **changes)
        if persist and checkpoint_progress:
            if self._state_store.has_saved(job_id, self._jobs.get(job_id)):
                self._resume_checkpoint_at[job_id] = time.monotonic()

    def _save_state_unlocked(self) -> None:
        """Commit changed jobs and new results without rewriting historical results."""
        try:
            self._state_store.save(self._jobs, self._job_requests)
        except Exception:
            logger.warning("Failed to checkpoint job progress", exc_info=True)

    def _remove_photo_features(self, job_ids) -> None:
        from system.dinov3.job_cache import DinoV3JobCache
        from system.dinov3.media_cache import DinoV3VideoCache

        for job_id in job_ids:
            try:
                DinoV3JobCache.remove(job_state_path(), job_id)
                DinoV3VideoCache.remove(job_state_path(), job_id)
            except OSError:
                logger.warning("Cannot remove DINOv3 features: %s", job_id, exc_info=True)

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            active = job_id in self._active_job_ids
        super().delete_job(job_id)
        if not active:
            self._remove_photo_features([job_id])

    def clear_jobs(self) -> None:
        with self._lock:
            inactive = set(self._jobs) - self._active_job_ids
        super().clear_jobs()
        self._remove_photo_features(inactive)

    def _resolve_inputs(self, job_id: str, request: CreateJobRequest) -> tuple[Path, list[Path]]:
        # Scope the snapshot to this task and the settings that affect discovery.
        # Continuing a task processes its original files; a new task discovers additions.
        input_key = json.dumps({
            "version": 1, "input_dir": request.input_dir, "input_paths": request.input_paths,
            "skip_videos": request.options.video_mode == "skip",
        }, sort_keys=True, ensure_ascii=False)
        try:
            cached = self._state_store.load_inputs(job_id, input_key)
        except Exception:
            logger.warning("Failed to read cached job inputs; scanning again", exc_info=True)
            cached = None
        if cached is not None:
            # Check only the selected roots. Do not mistake a disconnected
            # source directory for thousands of individually deleted files.
            for raw in request.input_paths or [request.input_dir]:
                if raw:
                    try:
                        Path(raw).expanduser().stat()
                    except (FileNotFoundError, NotADirectoryError) as exc:
                        raise ValueError(f"输入路径不存在: {raw}") from exc
            logger.info("Reusing cached job inputs: job_id=%s files=%d", job_id, len(cached[1]))
            return cached

        root, files = super()._resolve_inputs(job_id, request)
        _raise_if_cancelled(lambda: self._is_cancelled(job_id))
        with self._lock:
            if job_id not in self._jobs or job_id in self._deleted_jobs:
                raise RuntimeError("用户取消任务")
            try:
                self._save_state_unlocked()
                self._state_store.save_inputs(job_id, input_key, root, files)
            except Exception:
                # A cache write failure must not lose completed detection progress.
                logger.warning("Failed to cache discovered job inputs", exc_info=True)
        return root, files


def _dino_manifest_payload(model_path: str | Path | None):
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if not path.is_file() or not path.name.endswith(".neri.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("backend") != "dinov3":
        return None
    try:
        from system.dinov3.runtime import read_manifest

        return read_manifest(path)
    except ValueError as exc:
        raise ValueError(f"DINOv3联合模型无效: {exc}") from exc
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"DINOv3联合模型无效: {exc}") from exc


def _validate_dino_job_options(
    model_path: str | None,
    classification_model_path: str | None,
    video_mode: str,
) -> None:
    if (
        model_path
        and Path(model_path).suffix.lower() in {".tflite", ".litert"}
        and video_mode == "all"
    ):
        raise ValueError(
            "LiteRT 探测暂不支持完整视频追踪，请使用快速识别或跳过视频。"
        )
    payload = _dino_manifest_payload(classification_model_path)
    if payload is None:
        return
    if not model_path:
        raise ValueError(f"{model_display_name(payload)} 分类模型必须同时选择探测模型。")
    if video_mode == "all":
        raise ValueError(
            f"{model_display_name(payload)} 暂不支持完整视频追踪，请使用快速识别或跳过视频。"
        )


def _persist_dino_observations(*args, **kwargs):
    return None


def _load_detector(model_path: str | None, classification_model_path: str | None = None):
    import importlib.util

    from system.dino.image_processor import ImageProcessor

    resolved_model_path = _resolve_model_path(model_path) if model_path else None
    resolved_classification_path = _resolve_classification_model_path(
        classification_model_path
    )
    if classification_model_path and resolved_classification_path is None:
        raise FileNotFoundError(f"分类模型文件不存在: {classification_model_path}")
    if resolved_model_path is None and resolved_classification_path is None:
        raise ValueError("探测模型和分类模型至少需要选择一个。")

    suffix = resolved_model_path.suffix.lower() if resolved_model_path is not None else ""
    payload = _dino_manifest_payload(resolved_classification_path)
    if suffix in {".tflite", ".litert"} and payload is None:
        if resolved_classification_path is not None:
            raise ValueError(
                "LiteRT 探测暂不支持搭配分类模型；请取消分类模型或安装 PyTorch。"
            )
        from system.dino.litert_image_processor import LiteRtImageProcessor

        try:
            return LiteRtImageProcessor(str(resolved_model_path))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LiteRT 运行库未安装；请安装 requirements-privacy.txt 中的 "
                "ai-edge-litert，或安装 PyTorch 后使用 .pt 模型。"
            ) from exc
    if (
        resolved_model_path is not None
        and importlib.util.find_spec("torch") is None
        and resolved_classification_path is None
    ):
        raise RuntimeError(
            "当前环境未安装 PyTorch。请选择 .tflite LiteRT 探测模型，"
            "或在环境维护中安装 PyTorch。"
        )
    if (
        resolved_classification_path is not None
        and importlib.util.find_spec("torch") is None
        and (payload is None or payload.get("encoder_backend", "pytorch") != "litert")
    ):
        raise RuntimeError("分类模型需要 PyTorch；请安装 PyTorch 或取消分类模型。")

    if payload is not None and resolved_model_path is None:
        raise ValueError(f"{model_display_name(payload)} 分类模型必须同时选择探测模型。")
    if (payload is not None and payload.get("encoder_backend", "pytorch") != "litert"
            and importlib.util.find_spec("torch") is None):
        raise RuntimeError(f"{model_display_name(payload)} 分类需要 PyTorch；请先安装 PyTorch 或取消分类模型。")

    detector = ImageProcessor(
        str(resolved_model_path) if resolved_model_path is not None else None
    )
    if resolved_classification_path is not None:
        if payload is not None:
            from system.dinov3.runtime import load_dinov3_model

            runtime = load_dinov3_model(resolved_classification_path)
            detector.load_dino_classifier(runtime.classifier)
            detector.dino_registry = None
            detector.dino_feedback = None
            detector.dino_runtime = runtime
        else:
            detector.load_cls_model(str(resolved_classification_path))
    if (
        detector.model is None
        and getattr(detector, "cls_model", None) is None
        and getattr(detector, "dino_classifier", None) is None
    ):
        raise RuntimeError("未能加载所选的探测模型或分类模型。")
    return detector


def _learnable_observations_for_file(
    classification_model_path: str,
    file_path: Path,
):
    return []


def _checkpoint_species_for_model(classification_model_path: str):
    return None


def _record_validation_registry_feedback(
    classification_model_path: str,
    observation,
    operation_id: str,
    confirmed_species: str,
):
    return None


def _record_validation_feedback(
    classification_model_path: str,
    observation,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
):
    return None


# These assignments redirect DINO hooks inside the generic service implementation.
_legacy._validate_dino_job_options = _validate_dino_job_options
_legacy._dino_manifest_payload = _dino_manifest_payload
_legacy._persist_dino_observations = _persist_dino_observations
_legacy._load_detector = _load_detector
_legacy._learnable_observations_for_file = _learnable_observations_for_file
_legacy._checkpoint_species_for_model = _checkpoint_species_for_model
_legacy._record_validation_registry_feedback = _record_validation_registry_feedback
_legacy._record_validation_feedback = _record_validation_feedback


# Make the active facade self-contained: callers that import services directly
# must receive the same indexed/batched validation behavior as the FastAPI app.
# The installer is idempotent via marker attributes on each patched function.
def mark_validation_item(request):
    """Keep single-file marks on the same staged-learning path as batch marks."""
    items = mark_validation_items(ValidationBatchMarkRequest(
        input_path=request.input_path, file_paths=[request.file_path], action=request.action,
        species_name=request.species_name, species_count=request.species_count,
        species_type=request.species_type, remark=request.remark,
        classification_model_path=request.classification_model_path,
        feedback_operation_id=request.feedback_operation_id,
        confidence_settings=getattr(request, "confidence_settings", None),
        min_frame_ratio=getattr(request, "min_frame_ratio", 0.0)))
    if not items:
        raise ValueError("没有可标记的文件")
    return items[0]


import sys as _sys
from .runtime_patches import install_runtime_patches as _install_runtime_patches

_install_runtime_patches(_sys.modules[__name__])
del _install_runtime_patches
del _sys
