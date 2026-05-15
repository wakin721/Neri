"""Processing services used by the FastAPI application.

The service intentionally reuses the existing Neri domain modules instead of
forking the detection logic.  Lightweight metadata indexing is available by
default; YOLO inference can be enabled per job when the runtime has the model
and ML dependencies installed.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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


def list_available_models() -> list[ModelInfo]:
    """List all .pt files available under res/model."""

    directory = model_directory()
    return _list_model_files(directory, ("*.pt",))


def list_available_classification_models() -> list[ModelInfo]:
    """List classification model files under res/model_cls."""

    directory = classification_model_directory()
    return _list_model_files(directory, ("*.pt", "*.onnx", "*.engine"))


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


def preview_media_items(input_dir: str, output_dir: str | None = None) -> list[DetectionItem]:
    """Return media files and cached detections without running YOLO."""

    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    files = list(_resolve_supported_inputs(input_path))
    if not files:
        return []

    detection_db_roots = _detection_db_search_roots(input_path, output_dir)
    detection_index = _load_detection_index(detection_db_roots)
    validation_index = _load_validation_index(detection_db_roots)
    items: list[DetectionItem] = []
    for path in files:
        item = _build_fast_metadata_item(path)
        db_detection_data = detection_index.get(path.stem) or _load_detection_data_for_path(path, detection_db_roots)
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
    detection_index = _load_detection_index(roots)
    validation_index = _load_validation_index(roots)
    db_detection_data = detection_index.get(path.stem) or _load_detection_data_for_path(path, roots)
    if db_detection_data:
        item = _apply_detection_data(item, db_detection_data)
    item = _apply_validation_state(item, path.name, validation_index)
    return item


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
            input_path = Path(request.input_dir).expanduser().resolve()
            if not input_path.exists():
                raise ValueError(f"输入路径不存在: {input_path}")

            files = list(_resolve_supported_inputs(input_path))
            if not files:
                raise ValueError(f"输入路径中没有支持的媒体文件: {input_path}")

            self._mutate_job(
                job_id,
                state=JobState.RUNNING,
                total=len(files),
                message=f"发现 {len(files)} 个支持的媒体文件",
            )

            detection_db_roots = _detection_db_search_roots(input_path, request.output_dir)
            detection_index = _load_detection_index(detection_db_roots)
            detector = None
            if request.options.enable_detection:
                detector = _load_detector(request.options.model_path)

            results: list[DetectionItem] = []
            for index, path in enumerate(files, start=1):
                item = _build_metadata_item(path)
                db_detection_data = detection_index.get(path.stem) or _load_detection_data_for_path(path, detection_db_roots)
                if db_detection_data:
                    item = _apply_detection_data(item, db_detection_data)
                if detector is not None and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    item = _detect_image(detector, path, item, request, input_path)
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


def _resolve_supported_inputs(input_path: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    if input_path.is_file():
        if input_path.suffix.lower() in supported:
            yield input_path
        return

    if input_path.is_dir():
        yield from _iter_supported_files(input_path)


def _iter_supported_files(input_dir: Path) -> Iterable[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _build_fast_metadata_item(path: Path) -> DetectionItem:
    return DetectionItem(
        filename=path.name,
        path=str(path),
        file_type=path.suffix.lower().lstrip("."),
        size_bytes=_file_size(path),
    )


def _build_metadata_item(path: Path) -> DetectionItem:
    size_bytes = _file_size(path)

    if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        return DetectionItem(
            filename=path.name,
            path=str(path),
            file_type=path.suffix.lower().lstrip("."),
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
            width=int(width) if width else None,
            height=int(height) if height else None,
            size_bytes=size_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - per-file errors should not fail the whole job
        return DetectionItem(
            filename=path.name,
            path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            size_bytes=size_bytes,
            error=str(exc),
        )


def _load_detector(model_path: str | None):
    from system.image_processor import ImageProcessor

    resolved_model_path = _resolve_model_path(model_path)
    return ImageProcessor(str(resolved_model_path))


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


def _detection_db_search_roots(input_path: Path, output_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    roots.append(input_path if input_path.is_dir() else input_path.parent)
    if output_dir:
        roots.append(Path(output_dir).expanduser())

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


def _candidate_detection_dbs(path: Path, search_roots: list[Path]) -> list[Path]:
    return _candidate_detection_dbs_for_roots([path.parent, *search_roots])


def _candidate_detection_dbs_for_roots(roots: list[Path]) -> list[Path]:
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
                *sorted(resolved_root.rglob("*.db"), key=lambda item: str(item).lower()),
            ]
        except Exception:
            db_paths = sorted(resolved_root.rglob("*.db"), key=lambda item: str(item).lower())

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


def _load_detection_index(search_roots: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for db_path in _candidate_detection_dbs_for_roots(search_roots):
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


def _load_validation_index(search_roots: list[Path]) -> dict[str, bool]:
    index: dict[str, bool] = {}
    for db_path in _candidate_detection_dbs_for_roots(search_roots):
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


def _load_detection_data_for_path(path: Path, search_roots: list[Path]) -> dict[str, Any]:
    base_name = path.stem
    for db_path in _candidate_detection_dbs(path, search_roots):
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
        boxes.append(
            {
                "species": str(species or "Unknown"),
                "confidence": confidence,
                "bbox": bbox,
                "candidates": candidates if isinstance(candidates, list) else [],
            }
        )
    return boxes


def _boxes_from_tracks(raw_tracks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tracks, dict):
        return []

    boxes: list[dict[str, Any]] = []
    for points in raw_tracks.values():
        if not isinstance(points, list):
            continue
        best_point: dict[str, Any] | None = None
        best_confidence = -1.0
        for point in points:
            if not isinstance(point, dict):
                continue
            bbox = _coerce_bbox(point.get("bbox"))
            if not bbox:
                continue
            confidence = _coerce_float(_value_from_keys(point, ("confidence", "置信度", "conf", "score"))) or 0.0
            if best_point is None or confidence > best_confidence:
                best_point = point
                best_confidence = confidence
        if best_point is None:
            continue
        boxes.append(
            {
                "species": str(_value_from_keys(best_point, ("species", "物种", "name", "class_name")) or "Unknown"),
                "confidence": _coerce_float(_value_from_keys(best_point, ("confidence", "置信度", "conf", "score"))),
                "bbox": _coerce_bbox(best_point.get("bbox")),
                "candidates": [],
            }
        )
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
        try:
            from system.detection_db import get_db_path, init_db, upsert_detection, upsert_validation

            db_path = get_db_path(str(resolved))
            init_db(db_path)
            upsert_detection(db_path, path.stem, path.name, detection_data)
            upsert_validation(db_path, path.name, False)
        except Exception:
            continue


def _detect_image(detector, path: Path, item: DetectionItem, request: CreateJobRequest, input_path: Path) -> DetectionItem:
    try:
        detections = detector.detect_batch_species(
            [str(path)],
            conf=request.options.confidence,
            iou=request.options.iou,
            use_fp16=request.options.use_fp16,
        )
        detection = detections[0] if detections else {}
        detection_data = _serialize_detector_output(detector, detection)
        _save_detection_data_for_path(path, detection_data, input_path)
        return _apply_detection_data(item, detection_data)
    except Exception as exc:  # noqa: BLE001 - preserve metadata and report detection failure per file
        return item.model_copy(update={"error": f"检测失败: {exc}"})


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
