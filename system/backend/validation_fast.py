"""Indexed and batched DINOv2 validation feedback helpers."""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import uuid

logger = logging.getLogger(__name__)

_feedback_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="neri-dinov2-feedback",
)
_feedback_futures: dict[str, Future[int]] = {}
_feedback_futures_lock = threading.Lock()


def apply_validation_feedback_batch(
    classification_model_path: str,
    paths: Sequence[Path],
    *,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
    assign_registry_species: bool,
) -> int:
    """Apply conservative file-level DINOv2 feedback with one open store."""
    from .dinov2_feedback_service import (
        _affected_learning_species,
        _assign_registry_species,
        _open_feedback_state,
    )

    feedback, feature_center = _open_feedback_state(classification_model_path)
    affected_species: set[str] = set()
    applied = 0
    try:
        for path in paths:
            observations = feedback.observations_for_source_path(path)
            learnable = [
                observation
                for observation in observations
                if getattr(observation, "id", "")
            ]
            if len(learnable) != 1:
                continue
            observation = learnable[0]
            registry_species = None
            if assign_registry_species and confirmed_species:
                registry_species = confirmed_species
            elif action == "correct":
                predicted_species = str(
                    getattr(observation, "predicted_species", "") or ""
                ).strip()
                if (
                    predicted_species
                    and predicted_species != "Unknown"
                    and predicted_species not in feedback.checkpoint_classes
                ):
                    registry_species = predicted_species

            if registry_species:
                _assign_registry_species(
                    feedback,
                    classification_model_path,
                    observation,
                    operation_id=operation_id,
                    confirmed_species=registry_species,
                )
            else:
                record = feedback.record_feedback(
                    observation.id,
                    operation_id=operation_id,
                    action=action,
                    confirmed_species=confirmed_species,
                )
                affected_species.update(_affected_learning_species(record))
            applied += 1

        for species in sorted(affected_species):
            feedback.recompute_species(species, feature_center)
        return applied
    finally:
        feedback.close()


def schedule_validation_feedback_batch(
    classification_model_path: str,
    paths: Sequence[Path],
    *,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
    assign_registry_species: bool,
) -> Future[int]:
    """Queue DINOv2 learning without extending the validation save request."""
    future = _feedback_executor.submit(
        apply_validation_feedback_batch,
        classification_model_path,
        tuple(paths),
        operation_id=operation_id,
        action=action,
        confirmed_species=confirmed_species,
        assign_registry_species=assign_registry_species,
    )
    with _feedback_futures_lock:
        _feedback_futures[operation_id] = future

    def report_completion(completed: Future[int]) -> None:
        try:
            applied = completed.result()
            logger.info(
                "Background DINOv2 feedback completed: operation_id=%s applied=%d",
                operation_id,
                applied,
            )
        except Exception:
            logger.exception(
                "Background DINOv2 feedback failed: operation_id=%s",
                operation_id,
            )
        finally:
            with _feedback_futures_lock:
                if _feedback_futures.get(operation_id) is completed:
                    _feedback_futures.pop(operation_id, None)

    future.add_done_callback(report_completion)
    return future


def wait_for_validation_feedback_operation(operation_id: str) -> None:
    """Keep undo ordered behind an already queued feedback operation."""
    with _feedback_futures_lock:
        future = _feedback_futures.get(operation_id)
    if future is None:
        return
    try:
        future.result()
    except Exception:
        # The completion callback reports the original failure. Revert should
        # still inspect persisted feedback state in case it partially wrote.
        pass


def _checkpoint_species_for_model(
    classification_model_path: str,
) -> set[str] | None:
    from .dinov2_registry_service import load_checkpoint_for_model
    from system.dinov2.runtime import DinoV2ManifestError

    try:
        return set(load_checkpoint_for_model(classification_model_path).classes)
    except (DinoV2ManifestError, FileNotFoundError):
        return None


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
        "_checkpoint_species_for_model",
        "_learnable_observations_for_file",
        "_eligible_auto_feedback",
        "_record_validation_registry_feedback",
        "_record_validation_feedback",
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
        if not model_path or request.action == "unverified":
            return call_original(request)

        # The batched implementation depends on the indexed observation lookup
        # installed by runtime_patches. If a caller/test/extension overrides an
        # existing service seam, preserve the historical dynamic service behavior
        # instead of bypassing the override and opening a real model feedback store.
        observation_lookup = getattr(
            services_module,
            "_learnable_observations_for_file",
            None,
        )
        if observation_lookup is not None and not getattr(
            observation_lookup,
            "_neri_indexed_observation_lookup",
            False,
        ):
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

        roots = services_module._preview_detection_db_roots(input_path, None, paths)
        detection_indexes: dict[Path, dict[str, dict[str, Any]]] = {}
        filenames = {path.name for path in paths}

        def detection_data_for(path: Path) -> dict[str, Any]:
            for root in services_module._unique_existing_dirs([path.parent, *roots]):
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

        updates: list[tuple[Path, dict[str, Any], bool | None]] = []
        updated_items = []
        species_database_updates: set[tuple[str, str]] = set()
        for path in paths:
            detection_data, validated = services_module._build_validation_update(
                request,
                path,
                detection_data_for(path),
            )
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
        services_module._persist_validation_updates(updates, input_path)
        for species_name, species_type in species_database_updates:
            services_module._update_species_database(species_name, species_type)
        persisted_elapsed = time.perf_counter() - persistence_started

        operation_id = request.feedback_operation_id or uuid.uuid4().hex
        confirmed_species = (
            (request.species_name or "").strip() or None
            if request.action == "update"
            else None
        )
        checkpoint_species = (
            _checkpoint_species_for_model(model_path)
            if request.action == "update" and confirmed_species
            else set()
        )
        assign_registry_species = bool(
            request.action == "update"
            and confirmed_species
            and checkpoint_species is not None
            and confirmed_species not in checkpoint_species
        )

        schedule_validation_feedback_batch(
            model_path,
            paths,
            operation_id=operation_id,
            action=request.action,
            confirmed_species=confirmed_species,
            assign_registry_species=assign_registry_species,
        )

        try:
            from system.training import get_queue

            contribution_queue = get_queue()
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
                "dino_feedback=queued total=%.3fs operation_id=%s"
            ),
            len(paths),
            persisted_elapsed,
            time.perf_counter() - started,
            operation_id,
        )
        return updated_items

    setattr(mark_validation_items, "_neri_batched_dinov2_feedback", True)
    return mark_validation_items
