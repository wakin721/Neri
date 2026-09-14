"""Indexed and batched validation feedback helpers."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def apply_validation_feedback_batch(
    classification_model_path: str,
    paths: Sequence[Path],
    *,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
    assign_registry_species: bool,
) -> int:
    """Apply conservative file-level DINOv3 feedback with one open store."""
    from .dinov3_feedback_service import (
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
                if getattr(observation, 'id', '')
            ]
            if len(learnable) != 1:
                continue
            observation = learnable[0]
            if assign_registry_species and confirmed_species:
                _assign_registry_species(
                    feedback,
                    classification_model_path,
                    observation,
                    operation_id=operation_id,
                    confirmed_species=confirmed_species,
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


def _checkpoint_species_for_model(
    classification_model_path: str,
) -> set[str] | None:
    from .dinov3_registry_service import load_checkpoint_for_model
    from system.dinov3.runtime import DinoV3ManifestError

    try:
        return set(load_checkpoint_for_model(classification_model_path).classes)
    except (DinoV3ManifestError, FileNotFoundError):
        return None


def make_mark_validation_items(services_module: Any):
    """Wrap legacy marking while replacing only its per-file DINO feedback loop."""
    original = services_module.mark_validation_items

    def mark_validation_items(request):
        model_path = request.classification_model_path
        if not model_path or request.action == 'unverified':
            return original(request)

        operation_id = request.feedback_operation_id or uuid.uuid4().hex
        stripped = request.model_copy(
            update={
                'classification_model_path': None,
                'feedback_operation_id': None,
            }
        )
        started = time.perf_counter()
        items = original(stripped)
        persisted_elapsed = time.perf_counter() - started

        confirmed_species = (
            (request.species_name or '').strip() or None
            if request.action == 'update'
            else None
        )
        checkpoint_species = (
            _checkpoint_species_for_model(model_path)
            if request.action == 'update' and confirmed_species
            else set()
        )
        assign_registry_species = bool(
            request.action == 'update'
            and confirmed_species
            and checkpoint_species is not None
            and confirmed_species not in checkpoint_species
        )

        feedback_started = time.perf_counter()
        try:
            applied = apply_validation_feedback_batch(
                model_path,
                [Path(path).expanduser().resolve() for path in request.file_paths],
                operation_id=operation_id,
                action=request.action,
                confirmed_species=confirmed_species,
                assign_registry_species=assign_registry_species,
            )
        except Exception as exc:
            raise RuntimeError(f'DINOv3 自动反馈失败: {exc}') from exc
        feedback_elapsed = time.perf_counter() - feedback_started
        logger.info(
            (
                'Validation mark timing: targets=%d persisted=%.3fs '
                'dino_feedback=%.3fs total=%.3fs applied=%d'
            ),
            len(request.file_paths),
            persisted_elapsed,
            feedback_elapsed,
            time.perf_counter() - started,
            applied,
        )
        return items

    setattr(mark_validation_items, '_neri_batched_dinov3_feedback', True)
    return mark_validation_items
