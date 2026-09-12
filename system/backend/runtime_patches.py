from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REJECTED_UNKNOWN_LABEL = '拒识/Unknown'


def _candidate_is_dinov3_rejection(candidate: Any) -> bool:
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
        if _candidate_is_dinov3_rejection(candidate):
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


def preserve_dinov3_rejections(
    payload: dict[str, Any],
    detection: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore DINOv3 open-set rejections that ImageProcessor marks as filtered.

    Normal low-confidence classifier filtering remains untouched. Only candidates
    carrying the native DINOv3 rejection metadata and ``accepted == False`` are
    surfaced as the explicit ``拒识/Unknown`` state.
    """

    results = detection.get('detect_results') or []
    rejected_boxes: list[dict[str, Any]] = []

    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue
        candidates_data = getattr(result, 'candidates_data', {}) or {}
        filtered_indices = getattr(result, 'classification_filtered_boxes', set()) or set()
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
                # Intentionally null: the frontend confidence threshold is a
                # closed-set filter and must not turn a rejection back into 空.
                '置信度': None,
                '边界框': bbox,
                # Do not feed open-set alternatives back into the generic
                # closed-set candidate filter. Preserve DINO metadata directly.
                '候选项': [],
            }
            for key in (
                'observation_id',
                'registry_id',
                'predicted_species',
                'known_score',
                'threshold',
                'squared_distance',
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
    if REJECTED_UNKNOWN_LABEL not in species:
        species.append(REJECTED_UNKNOWN_LABEL)
    payload['物种名称'] = ','.join(species)
    payload['拒识数量'] = len(rejected_boxes)

    count = str(payload.get('物种数量') or '').strip()
    if not count or count == '空':
        payload['物种数量'] = str(len(rejected_boxes))

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
            return preserve_dinov3_rejections(payload, detection)

        setattr(patched_serializer, '_neri_preserves_open_set_rejections', True)
        services_module._serialize_detector_output = patched_serializer

    if not patch_preview:
        return
    preview = getattr(services_module, 'preview_media_items', None)
    if callable(preview) and not getattr(preview, '_neri_filtered_preview_sql', False):
        from .preview_fast import make_preview_media_items

        services_module.preview_media_items = make_preview_media_items(services_module)
