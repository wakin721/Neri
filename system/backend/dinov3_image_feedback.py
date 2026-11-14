"""Human image labels and quantity-based admission of reviewed crops."""
from __future__ import annotations

from copy import deepcopy
import math
import re
from pathlib import Path

from system.dinov3.observation_support import is_dinov3_box_mapping, make_dinov3_observation_id

_UNKNOWN = {"", "unknown", "unknow", "拒识/unknown", "未知", "未知物种", "待确认", "待核实", "无", "empty"}
_REASONS = {
    "eligible": "已按人工单物种标注更新检测框，符合数量条件的裁图将自动学习。",
    "count_exceeded": "已更新框上的物种；检测框数超过人工数量，本图不自动学习，请逐框核对。",
    "multiple_species": "整图包含多个物种，未向检测框分配物种，也未进行整图学习；可逐框确认。",
    "invalid_count": "已保存人工标注；数量须为明确的正整数，当前未自动学习。",
    "unknown_species": "已保存标注；物种尚未明确，当前未自动学习。",
    "no_boxes": "已保存人工物种和数量；没有可用的检测框，未加入学习。",
    "invalid_boxes": "已保存人工标注；检测框坐标无效或不是照片，本图未自动学习。",
    "empty": "已保存为空照片，并撤回该图对应的学习样本。",
    "unverified": "已取消人工校验，并撤回该图对应的学习样本。",
    "excluded": "已保存整图标注；这些检测框已设置不参与学习。",
}


def _valid_crop(box):
    bbox = box.get("边界框") or box.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or box.get("frame_index") is not None:
        return False
    try:
        left, top, right, bottom = map(float, bbox)
        return all(math.isfinite(v) for v in (left, top, right, bottom)) and right > left and bottom > top
    except (TypeError, ValueError):
        return False


def _original(box):
    original = box.get("image_annotation_original")
    return deepcopy(original if isinstance(original, dict) else box)


def prepare_image_feedback(data: dict, path: Path, action: str, operation_id: str):
    """Return (box, queue_action) pairs without replacing human image counts."""
    boxes = deepcopy(data.get("检测框") or [])
    targets = [(index, box) for index, box in enumerate(boxes)
               if isinstance(box, dict) and is_dinov3_box_mapping(box, data)]
    species_text = str(data.get("物种名称") or "").strip()
    count_text = str(data.get("物种数量") or "").strip()
    names = list(dict.fromkeys(p.strip() for p in re.split(r"[,，、;；\n]+", species_text) if p.strip()))
    count = int(count_text) if re.fullmatch(r"\d+", count_text) else None

    def excluded(box):
        return (box.get("label_source") != "image_annotation"
                and box.get("feedback_status") == "confirmed" and box.get("物种") == "空")

    candidates = [(index, box) for index, box in targets if not excluded(box)]
    box_count = len(candidates)
    if action == "unverified":
        status = "unverified"
    elif action == "empty" or species_text == "空" or count == 0:
        status = "empty"
    elif len(names) > 1:
        status = "multiple_species"
    elif not names or names[0].casefold() in _UNKNOWN:
        status = "unknown_species"
    elif count is None:
        status = "invalid_count"
    elif box_count == 0:
        status = "no_boxes"
    elif box_count > count:
        status = "count_exceeded"
    elif any(not _valid_crop(box) for _, box in candidates):
        status = "invalid_boxes"
    else:
        status = "eligible"

    if status == "empty":
        data["物种名称"] = species_text = "空"
    if status == "unverified":
        data.pop("image_annotation", None)
    else:
        data["image_annotation"] = {
            "species": species_text, "count": count_text, "source": "human", "operation_id": operation_id,
        }
    staged = []
    for index, box in targets:
        if status in {"unverified", "multiple_species", "unknown_species"}:
            original = _original(box)
            if status == "unverified":
                original.pop("feedback_status", None)
            original["learning_eligible"] = False
            boxes[index] = box = original
            queue_action = "unverified"
        elif status == "empty":
            box.setdefault("image_annotation_original", _original(box))
            box.update(物种="空", feedback_status="confirmed", label_source="image_annotation", learning_eligible=False)
            queue_action = "unverified"
        elif excluded(box):
            box["learning_eligible"] = False
            queue_action = "unverified"
        else:
            box.setdefault("image_annotation_original", _original(box))
            eligible = status == "eligible" and box.get("learning_excluded") is not True
            box.update(物种=names[0], feedback_status="confirmed" if eligible else "image_label",
                       label_source="image_annotation", learning_eligible=eligible)
            queue_action = "update" if eligible else "unverified"
        box["observation_id"] = box.get("observation_id") or make_dinov3_observation_id(
            path, index, box.get("边界框") or box.get("bbox"))
        staged.append((box, queue_action))
    learning_count = sum(queue_action == "update" for _, queue_action in staged)
    if status == "eligible" and learning_count == 0:
        status = "excluded"
    data["检测框"] = boxes
    data["learning_feedback"] = {"status": status, "box_count": box_count, "manual_count": count, "learning_count": learning_count,
                                 "message": _REASONS[status]}
    return staged
