"""Preserve generic detector person/vehicle categories and human overrides."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

_ALIASES = {
    "animal": "animal", "动物": "animal",
    "person": "person", "human": "person", "people": "person", "人员": "person", "人": "person",
    "vehicle": "vehicle", "车辆": "vehicle",
}
_LABELS = {"person": "人员", "vehicle": "车辆"}
_HUMAN = {"confirmed", "image_label"}


def category_from_label(label):
    return _ALIASES.get(str(label or "").strip().casefold())


def category_for_box(result, box):
    """Resolve names, never assume a numeric class ID means person."""
    try:
        index = int(box.cls.item())
        names = result.names
        label = names.get(index, names.get(str(index))) if isinstance(names, dict) else names[index]
        return category_from_label(label)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def detector_category_candidate(category, confidence, *, observation_id=None, track_id=None,
                                event_id=None, complete=True):
    name = _LABELS[category]
    candidate = dict(name=name, conf=float(confidence), raw_det_conf=float(confidence), accepted=True,
        predicted_species=name, best_known_species=name, source="detector_category",
        classification_source="detector_category", detector_category=category,
        confidence_kind="detector_confidence", gate_mode="detector",
        sequence_pending=not complete, sequence_status="complete" if complete else "pending")
    for key, value in (("observation_id", observation_id), ("track_id", track_id), ("event_id", event_id)):
        if value is not None:
            candidate[key] = value
    return candidate


def detector_category_box(category, confidence, bbox, **kwargs):
    candidate = detector_category_candidate(category, confidence, **kwargs)
    return {**candidate, "物种": candidate["name"], "置信度": candidate["conf"],
            "边界框": list(bbox), "候选项": [candidate]}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def normalize_detector_categories(data, *, human_confirmed=False):
    """Repair safe, auto-generated legacy views without changing stored records.

    Raw class vectors are used only when every final box still aligns with them.
    Explicit human labels are authoritative, even if they disagree with a detector.
    """
    if (not isinstance(data, dict) or human_confirmed or data.get("最低置信度") == "人工校验"
            or data.get("feedback_status") in _HUMAN):
        return data
    annotation = data.get("image_annotation")
    if isinstance(annotation, dict) and annotation.get("source") == "human":
        return data
    boxes = data.get("检测框")
    if not isinstance(boxes, list) or not boxes:
        return data
    dino = (data.get("sequence_completed") is True or data.get("sequence_status") is not None
            or any(isinstance(box, dict) and (str(box.get("observation_id", "")).startswith("dinov3:")
                       or box.get("known_score") is not None) for box in boxes))
    if not dino:
        return data
    names, classes, confidences = data.get("names_map"), data.get("all_classes"), data.get("all_confidences")
    aligned = (isinstance(names, dict) and isinstance(classes, list) and isinstance(confidences, list)
               and len(classes) == len(confidences) == len(boxes))
    changed = None
    for index, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("feedback_status") in _HUMAN:
            continue
        if box.get("classification_source") == "detector_category":
            continue
        category = category_from_label(box.get("detector_category"))
        confidence = _number(box.get("raw_det_conf"))
        if aligned:
            raw_id = _number(classes[index])
            if raw_id is not None and raw_id.is_integer():
                category = category_from_label(names.get(str(int(raw_id)), names.get(int(raw_id))))
                confidence = _number(confidences[index])
        if category not in _LABELS or confidence is None or not 0 <= confidence <= 1:
            continue
        bbox = box.get("边界框") or box.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(_number(v) is None for v in bbox):
            continue
        updated = detector_category_box(category, confidence, bbox,
            observation_id=box.get("observation_id"), track_id=box.get("track_id"),
            event_id=box.get("event_id"), complete=data.get("sequence_completed") is True)
        updated["original_classification"] = dict(species=box.get("物种"), confidence=box.get("置信度"))
        for key in ("frame_index", "timestamp", "box_index"):
            if key in box:
                updated[key] = box[key]
        if changed is None:
            changed = deepcopy(data)
        changed["检测框"][index] = updated
    if changed is None:
        return data
    visible = [box for box in changed["检测框"] if isinstance(box, dict)
               and str(box.get("物种") or "").strip() not in {"", "空", "无"}]
    if any(box.get("frame_index") is not None for box in visible):
        frames = {}
        for box in visible:
            frame = box.get("frame_index", box.get("timestamp"))
            counts_in_frame = frames.setdefault(frame, Counter())
            counts_in_frame[str(box.get("物种"))] += 1
        counts = Counter()
        for per_frame in frames.values():
            for species, count in per_frame.items():
                counts[species] = max(counts.get(species, 0), count)
    else:
        counts = Counter(str(box.get("物种")) for box in visible)
    scores = [_number(box.get("置信度")) for box in visible]
    scores = [value for value in scores if value is not None]
    changed.update(物种名称=",".join(counts) if counts else "空", 物种数量=",".join(map(str, counts.values())) if counts else "空",
                   最低置信度=f"{min(scores):.3f}" if scores else None,
                   detector_category_guard=True)
    changed.pop("物种类型", None)
    return changed
