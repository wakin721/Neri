"""The explicit contribution policy, independent of ordinary app settings."""
from __future__ import annotations

import math
import re
import unicodedata

AGREEMENT_VERSION = "2026-09-05"
EMPTY_CATEGORY = "空照片"
MAX_EMPTY_PER_FOLDER = 3
MAX_IMAGE_EDGE = 2048
JPEG_QUALITY = 82
MAX_IMAGE_BYTES = 16 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNKNOWN_LABELS = {"", "unknown", "none", "未检测", "错误", "需人工校验", "未知"}
SENSITIVE_LABELS = {"人", "人类", "人员", "行人", "人脸", "person", "people", "human",
                    "车", "车辆", "汽车", "摩托车", "车牌", "vehicle", "car", "truck", "motorcycle"}


def contribution_payload(data: dict) -> dict | None:
    """Allowlist human labels and model boxes; never retain free-text notes/paths."""
    raw = data.get("物种名称", "")
    if not isinstance(raw, str):
        return None
    labels = list(dict.fromkeys(x.strip() for x in re.split(r"[,，、]", raw) if x.strip()))
    if not labels or any(x.lower() in UNKNOWN_LABELS | SENSITIVE_LABELS for x in labels):
        return None
    if any(len(x) > 80 or x.endswith('.') or re.search(r'[\\/:*?"<>|%\x00-\x1f\x7f]', x) or x in {".", ".."} for x in labels):
        return None
    empty = labels == ["空"]
    if "空" in labels and not empty:
        return None
    if len(labels) > 8:
        return None
    if len({unicodedata.normalize('NFC', x).casefold() for x in labels}) != len(labels):
        return None
    boxes = []
    raw_boxes = data.get("检测框") or data.get("detect_results") or data.get("objects") or []
    if not isinstance(raw_boxes, list):
        raw_boxes = []
    for box in raw_boxes[:1000]:
        if not isinstance(box, dict):
            continue
        label = str(box.get("物种", box.get("species", box.get("class_name", "")))).strip()
        if label.lower() in SENSITIVE_LABELS:
            return None
        coords = box.get("边界框", box.get("bbox", box.get("xyxy")))
        try:
            coords = [float(x) for x in coords[:4]]
            confidence = float(box.get("置信度", box.get("confidence", 0)))
        except (TypeError, ValueError, OverflowError, KeyError):
            continue
        if len(coords) != 4 or not all(math.isfinite(x) for x in coords + [confidence]):
            continue
        if len(label) > 80 or re.search(r'[\\/\x00-\x1f]', label):
            continue
        boxes.append({"species": label, "bbox": coords, "confidence": max(0, min(1, confidence))})
    count_text = str(data.get("物种数量", ""))
    counts = []
    if re.fullmatch(r"\d{1,6}(?:[,，、]\d{1,6})*", count_text):
        counts = [int(x) for x in re.split(r"[,，、]", count_text)]
        if len(counts) != len(labels):
            counts = []
    return {"species": [EMPTY_CATEGORY] if empty else labels,
            "species_counts": [] if empty else counts,
            "empty": empty, "model_boxes": [] if empty else boxes}
