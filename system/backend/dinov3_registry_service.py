"""Backend helpers for opening the registry associated with a DINOv3 model."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov3.checkpoint import DinoV3Checkpoint, load_checkpoint
from system.dinov3.registry import SpeciesRegistry, registry_path_for_fingerprint
from system.dinov3.runtime import DinoV3ManifestError, resolve_dinov3_manifest
from system.dinov3.state import default_dinov3_state_root


def load_checkpoint_for_model(classification_model_path: str) -> DinoV3Checkpoint:
    manifest_path = resolve_dinov3_manifest(classification_model_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV3ManifestError(f"Invalid DINOv3 manifest: {manifest_path}") from exc
    checkpoint_name = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise DinoV3ManifestError("DINOv3 manifest checkpoint is missing")
    return load_checkpoint(manifest_path.parent / checkpoint_name)


def open_registry_for_model(classification_model_path: str) -> SpeciesRegistry:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    path = registry_path_for_fingerprint(
        default_dinov3_state_root(),
        checkpoint.fingerprint,
    )
    return SpeciesRegistry(path, model_fingerprint=checkpoint.fingerprint)


def _read_registry_frame(event: dict[str, object]) -> np.ndarray:
    source = Path(str(event.get("source_path") or "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"例图源文件不存在: {source}")
    if source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(source))
        try:
            frame_index = event.get("frame_index")
            timestamp_seconds = event.get("timestamp_seconds")
            if frame_index is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            elif timestamp_seconds is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise FileNotFoundError(f"无法读取例图视频帧: {source}")
        return frame
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"无法读取例图: {source}")
    return frame


def render_registry_example(
    registry: SpeciesRegistry, registration_id: int, event_id: int
) -> bytes:
    event = next(
        (item for item in registry.list_events(registration_id) if item["id"] == event_id),
        None,
    )
    if event is None:
        raise FileNotFoundError("DINOv3 registry event not found")
    bbox = event.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    frame = _read_registry_frame(event)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise FileNotFoundError("该历史事件的裁切框无效")
    pad_w = int(box_width * 0.1)
    pad_h = int(box_height * 0.1)
    left = max(0, int(x1) - pad_w)
    top = max(0, int(y1) - pad_h)
    right = min(width, int(x2) + pad_w)
    bottom = min(height, int(y2) + pad_h)
    if right <= left or bottom <= top:
        raise FileNotFoundError("该历史事件的裁切框超出图像范围")
    crop = frame[top:bottom, left:right]
    crop_h, crop_w = crop.shape[:2]
    if crop_h != crop_w:
        size = max(crop_h, crop_w)
        pad_top = (size - crop_h) // 2
        pad_bottom = size - crop_h - pad_top
        pad_left = (size - crop_w) // 2
        pad_right = size - crop_w - pad_left
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[114, 114, 114],
        )
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("裁切例图编码失败")
    return encoded.tobytes()
