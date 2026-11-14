"""Indexed and batched validation helpers for Neri1.1."""
from __future__ import annotations

import logging
import json
from pathlib import Path
import time
from typing import Any, Sequence
import uuid
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def _is_joint_model(classification_model_path) -> bool:
    """Identify the joint Neri1.1 deployment for explicit reviewed supplements."""
    try:
        path = Path(classification_model_path)
        if not path.name.endswith(".neri.json"):
            return False
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("backend") == "dinov3"
        and payload.get("head_type") == "ridge_within"
    )


def make_mark_validation_items(services_module: Any):
    """Replace DINO-backed marking while preserving legacy operation ordering."""
    original = services_module.mark_validation_items

    # The facade re-exports functions originally defined in services_legacy.
    # Such functions keep their defining module's globals, so monkeypatching a
    # facade seam would otherwise be ignored whenever we deliberately fall back
    # to the legacy implementation. Mirror only the explicit validation seams
    # for the duration of that call and restore them afterwards.
    legacy_hook_names = (
        "_preview_detection_db_roots",
        "_unique_existing_dirs",
        "_load_detection_index",
        "_build_validation_update",
        "_build_fast_metadata_item",
        "_apply_detection_data",
        "_persist_validation_updates",
        "_update_species_database",
    )
    missing = object()

    def call_original(request):
        target_globals = original.__globals__
        previous: dict[str, object] = {}
        for name in legacy_hook_names:
            if not hasattr(services_module, name):
                continue
            previous[name] = target_globals.get(name, missing)
            target_globals[name] = getattr(services_module, name)
        try:
            return original(request)
        finally:
            for name, value in previous.items():
                if value is missing:
                    target_globals.pop(name, None)
                else:
                    target_globals[name] = value

    def mark_validation_items(request):
        model_path = request.classification_model_path
        if not model_path:
            return call_original(request)
        joint_model = _is_joint_model(model_path)
        if request.action == "unverified" and not joint_model:
            return call_original(request)

        started = time.perf_counter()
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

        detection_indexes: dict[Path, dict[str, dict[str, Any]]] = {}
        filenames = {path.name for path in paths}

        def detection_data_for(path: Path) -> dict[str, Any]:
            for root in services_module._unique_existing_dirs([path.parent]):
                index = detection_indexes.get(root)
                if index is None:
                    index = services_module._load_detection_index(
                        [root],
                        recursive=False,
                        filenames=filenames,
                    )
                    detection_indexes[root] = index
                data = index.get(path.stem)
                if isinstance(data, dict):
                    return dict(data)
            return {}

        operation_id = request.feedback_operation_id or uuid.uuid4().hex
        staged_boxes = []
        updates: list[tuple[Path, dict[str, Any], bool | None]] = []
        updated_items = []
        species_database_updates: set[tuple[str, str]] = set()
        for path in paths:
            detection_data, validated = services_module._build_validation_update(
                request,
                path,
                detection_data_for(path),
            )
            if joint_model:
                from .dinov3_image_feedback import prepare_image_feedback
                staged_boxes.extend((path, box, queue_action) for box, queue_action in
                                    prepare_image_feedback(detection_data, path, request.action, operation_id))
            updates.append((path, detection_data, validated))

            item = services_module._build_fast_metadata_item(path)
            if detection_data:
                item = services_module._apply_detection_data(item, detection_data)
            item = item.model_copy(update={"validated": validated})
            updated_items.append(item)

            if request.species_type is not None:
                species_type = request.species_type.strip()
                species_name = str(detection_data.get("物种名称") or "").strip()
                if species_name and species_type:
                    species_database_updates.add((species_name, species_type))

        persistence_started = time.perf_counter()
        from .dinov3_learning import _write_lock, stage_feedback, reconcile_image_samples
        withdrawn = 0
        ids_by_path = {}
        for path, box, _action in staged_boxes:
            ids_by_path.setdefault(path, set()).add(box["observation_id"])
        with (_write_lock() if joint_model else nullcontext()) as connection:
            if joint_model:
                for path, detection_data, _validated in updates:
                    ids = ids_by_path.get(path, set())
                    withdrawn += reconcile_image_samples(model_path, input_path, path, ids, operation_id, connection=connection)
            for path, box, queue_action in staged_boxes:
                stage_feedback(model_path, input_path, path, box, queue_action, operation_id,
                               connection=connection)
            services_module._persist_validation_updates(updates, input_path)
        if staged_boxes or withdrawn:
            from .dinov3_learning import schedule_learning
            schedule_learning(model_path)
        for species_name, species_type in species_database_updates:
            services_module._update_species_database(species_name, species_type)
        persisted_elapsed = time.perf_counter() - persistence_started

        try:
            from system.training import get_queue

            contribution_queue = get_queue()
            enqueue_many_async = getattr(contribution_queue, "enqueue_many_async", None)
            if callable(enqueue_many_async):
                enqueue_many_async(
                    (path, detection_data, validated is True)
                    for path, detection_data, validated in updates
                )
            else:
                for path, detection_data, validated in updates:
                    contribution_queue.enqueue(
                        path,
                        detection_data,
                        validated=validated is True,
                    )
        except Exception as error:
            logger.warning("Training queue unavailable (%s)", type(error).__name__)

        logger.info(
            (
                "Validation mark timing: targets=%d persisted=%.3fs "
                "model=%s total=%.3fs operation_id=%s"
            ),
            len(paths),
            persisted_elapsed,
            "dinov3" if joint_model else "other",
            time.perf_counter() - started,
            operation_id,
        )
        return updated_items

    setattr(mark_validation_items, "_neri_batched_dino_feedback", True)
    return mark_validation_items
