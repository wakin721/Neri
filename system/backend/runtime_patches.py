from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REJECTED_UNKNOWN_LABEL = '拒识/Unknown'


def _candidate_is_dinov2_rejection(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    if candidate.get('accepted') is not False:
        return False
    return bool(
        candidate.get('observation_id')
        or candidate.get('predicted_species')
        or candidate.get('known_score') is not None
        or candidate.get('squared_distance') is not None
    )


def _first_rejected_candidate(candidates: Any) -> Mapping[str, Any] | None:
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if _candidate_is_dinov2_rejection(candidate):
            return candidate
    return None


def _box_bbox(box: Any) -> list[float] | None:
    try:
        values = box.xyxy.tolist()[0]
        if len(values) < 4:
            return None
        return [float(value) for value in values[:4]]
    except Exception:
        return None


def _split_species(value: Any) -> list[str]:
    text = str(value or '').strip()
    if not text or text == '空':
        return []
    return [
        part.strip()
        for part in text.replace('，', ',').replace('、', ',').split(',')
        if part.strip() and part.strip() != '空'
    ]


def preserve_dinov2_rejections(
    payload: dict[str, Any],
    detection: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore DINOv2 open-set rejections that ImageProcessor marks as filtered."""

    results = detection.get('detect_results') or []
    rejected_boxes: list[dict[str, Any]] = []

    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue
        candidates_data = getattr(result, 'candidates_data', {}) or {}
        filtered_indices = (
            getattr(result, 'classification_filtered_boxes', set()) or set()
        )
        for index in filtered_indices:
            try:
                box = boxes[index]
            except Exception:
                continue
            candidate = _first_rejected_candidate(candidates_data.get(index, []))
            if candidate is None:
                continue
            bbox = _box_bbox(box)
            if bbox is None:
                continue

            rejected_box: dict[str, Any] = {
                '物种': REJECTED_UNKNOWN_LABEL,
                '置信度': None,
                '边界框': bbox,
                '候选项': [],
            }
            for key in (
                'observation_id',
                'registry_id',
                'predicted_species',
                'known_score',
                'threshold',
                'squared_distance',
                'class_margin',
                'adjusted_distance_score',
                'score_threshold',
                'registry_action',
                'assistive_match',
                'source',
                'nearest_prototype_index',
            ):
                value = candidate.get(key)
                if value is not None:
                    rejected_box[key] = value
            rejected_boxes.append(rejected_box)

    if not rejected_boxes:
        return payload

    boxes = payload.setdefault('检测框', [])
    if not isinstance(boxes, list):
        boxes = []
        payload['检测框'] = boxes
    boxes.extend(rejected_boxes)

    species = _split_species(payload.get('物种名称'))
    had_rejected_label = REJECTED_UNKNOWN_LABEL in species
    if not had_rejected_label:
        species.append(REJECTED_UNKNOWN_LABEL)
    payload['物种名称'] = ','.join(species)
    payload['拒识数量'] = len(rejected_boxes)

    count = str(payload.get('物种数量') or '').strip()
    if not count or count == '空':
        payload['物种数量'] = str(len(rejected_boxes))
    elif not had_rejected_label:
        count_parts = [
            part.strip()
            for part in count.replace('，', ',').replace('、', ',').split(',')
            if part.strip()
        ]
        known_species_count = len(species) - 1
        if len(count_parts) == known_species_count:
            count_parts.append(str(len(rejected_boxes)))
            payload['物种数量'] = ','.join(count_parts)

    return payload


def install_runtime_patches(services_module: Any, *, patch_preview: bool = True) -> None:
    """Install compatibility-preserving backend patches before main_core imports."""

    serializer = getattr(services_module, '_serialize_detector_output', None)
    if callable(serializer) and not getattr(
        serializer,
        '_neri_preserves_open_set_rejections',
        False,
    ):
        original_serializer = serializer

        def patched_serializer(
            detector: Any,
            detection: dict[str, Any],
        ) -> dict[str, Any]:
            payload = original_serializer(detector, detection)
            return preserve_dinov2_rejections(payload, detection)

        setattr(patched_serializer, '_neri_preserves_open_set_rejections', True)
        services_module._serialize_detector_output = patched_serializer

    if not patch_preview:
        return

    from system.dinov2.feedback_index import install_feedback_store_patches

    install_feedback_store_patches()

    detection_loader = getattr(services_module, '_load_detection_index', None)
    if callable(detection_loader) and not getattr(
        detection_loader,
        '_neri_filtered_detection_sql',
        False,
    ):
        original_detection_loader = detection_loader

        def patched_detection_loader(
            roots,
            recursive=False,
            filenames=None,
        ):
            if filenames is None:
                return original_detection_loader(
                    roots,
                    recursive=recursive,
                    filenames=None,
                )
            from .preview_fast import load_detection_index_for_filenames

            db_paths = services_module._candidate_detection_dbs_for_roots(
                roots,
                recursive=recursive,
            )
            return load_detection_index_for_filenames(db_paths, set(filenames))

        setattr(patched_detection_loader, '_neri_filtered_detection_sql', True)
        services_module._load_detection_index = patched_detection_loader

    validation_loader = getattr(services_module, '_load_validation_index', None)
    if callable(validation_loader) and not getattr(
        validation_loader,
        '_neri_filtered_validation_sql',
        False,
    ):
        original_validation_loader = validation_loader

        def patched_validation_loader(
            roots,
            recursive=False,
            filenames=None,
        ):
            if filenames is None:
                return original_validation_loader(
                    roots,
                    recursive=recursive,
                )
            from .preview_fast import load_validation_index_for_filenames

            db_paths = services_module._candidate_detection_dbs_for_roots(
                roots,
                recursive=recursive,
            )
            return load_validation_index_for_filenames(db_paths, set(filenames))

        setattr(patched_validation_loader, '_neri_filtered_validation_sql', True)
        services_module._load_validation_index = patched_validation_loader

    preview = getattr(services_module, 'preview_media_items', None)
    if callable(preview) and not getattr(
        preview,
        '_neri_filtered_preview_sql',
        False,
    ):
        from .preview_fast import make_preview_media_items

        services_module.preview_media_items = make_preview_media_items(services_module)

    preview_item = getattr(services_module, 'preview_media_item', None)
    if callable(preview_item) and not getattr(
        preview_item,
        '_neri_filtered_single_preview_sql',
        False,
    ):
        from .preview_fast import make_preview_media_item

        services_module.preview_media_item = make_preview_media_item(services_module)

    reload_item = getattr(services_module, '_reload_validation_item', None)
    if callable(reload_item) and not getattr(
        reload_item,
        '_neri_filtered_validation_reload_sql',
        False,
    ):
        from .preview_fast import make_reload_validation_item

        services_module._reload_validation_item = make_reload_validation_item(
            services_module
        )

    mark_items = getattr(services_module, 'mark_validation_items', None)
    if callable(mark_items) and not getattr(
        mark_items,
        '_neri_batched_dinov2_feedback',
        False,
    ):
        from .validation_fast import make_mark_validation_items

        services_module.mark_validation_items = make_mark_validation_items(
            services_module
        )

    learnable = getattr(services_module, '_learnable_observations_for_file', None)
    if callable(learnable) and not getattr(
        learnable,
        '_neri_indexed_observation_lookup',
        False,
    ):

        def indexed_learnable_observations(
            classification_model_path,
            file_path,
        ):
            from .dinov2_feedback_service import _open_feedback_state

            feedback, _feature_center = _open_feedback_state(
                classification_model_path
            )
            try:
                return feedback.observations_for_source_path(file_path)
            finally:
                feedback.close()

        setattr(
            indexed_learnable_observations,
            '_neri_indexed_observation_lookup',
            True,
        )
        services_module._learnable_observations_for_file = (
            indexed_learnable_observations
        )

    from . import dinov2_feedback_service

    persistence = getattr(
        dinov2_feedback_service,
        'persist_runtime_observations',
        None,
    )
    if callable(persistence) and not getattr(
        persistence,
        '_neri_batched_runtime_persistence',
        False,
    ):
        from .dinov2_persistence_fast import persist_runtime_observations

        dinov2_feedback_service.persist_runtime_observations = (
            persist_runtime_observations
        )
