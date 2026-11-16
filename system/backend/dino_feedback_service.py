"""DINOv3 human feedback and shared detection payload helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _path_identity(value: str | Path) -> str:
    text = str(value or "").strip()
    try:
        return str(Path(text).expanduser().resolve()).replace("\\", "/").casefold()
    except Exception:
        return text.replace("\\", "/").casefold()


def _recompute_detection_summary(detection_data: dict) -> None:
    annotation = detection_data.get("image_annotation")
    if isinstance(annotation, dict) and annotation.get("source") == "human":
        detection_data["物种名称"] = annotation["species"]
        detection_data["物种数量"] = annotation["count"]
        return
    counts: Counter[str] = Counter()
    by_frame: dict[object, Counter[str]] = {}
    boxes = detection_data.get("检测框")
    if isinstance(boxes, list):
        for box in boxes:
            if isinstance(box, dict):
                species = str(box.get("物种") or "").strip()
                if species not in {"", "空", "Unknown", "unknown"}:
                    if box.get("frame_index") is not None:
                        frame = by_frame.setdefault(box["frame_index"], Counter())
                        frame[species] += 1
                    else:
                        counts[species] += 1
    for frame in by_frame.values():
        for species, count in frame.items():
            counts[species] = max(counts[species], count)
    detection_data["物种名称"] = ",".join(counts.keys()) if counts else "空"
    detection_data["物种数量"] = (
        ",".join(str(counts[name]) for name in counts) if counts else "空"
    )


def explain_feedback_observation(
    classification_model_path: str,
    observation_id: str,
) -> dict[str, Any]:
    from system.dinov3.observation_support import (
        explain_dinov3_observation,
        is_dinov3_observation_id,
    )

    if not is_dinov3_observation_id(observation_id):
        raise ValueError("该检测框不属于联合模型")
    return explain_dinov3_observation(classification_model_path, observation_id)


def render_feedback_observation_example(
    classification_model_path: str,
    observation_id: str,
) -> bytes:
    from system.dinov3.observation_support import (
        is_dinov3_observation_id,
        render_dinov3_observation_example,
    )

    if not is_dinov3_observation_id(observation_id):
        raise ValueError("该检测框不属于联合模型")
    return render_dinov3_observation_example(observation_id)


def apply_box_feedback(request: Any) -> dict[str, Any]:
    from system.dinov3.observation_support import apply_dinov3_box_feedback

    return apply_dinov3_box_feedback(request)


def revert_feedback_operation(request: Any) -> dict[str, Any]:
    from system.dinov3.observation_support import revert_dinov3_feedback_operation

    return revert_dinov3_feedback_operation(request)
