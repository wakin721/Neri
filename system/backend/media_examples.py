"""Render cropped media examples for validation review."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from system.config import SUPPORTED_VIDEO_EXTENSIONS


def _read_media_frame(event: dict[str, object]) -> np.ndarray:
    source = Path(str(event.get("source_path") or "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"例图源文件不存在: {source}")
    if source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(source))
        try:
            if event.get("frame_index") is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(event["frame_index"]))
            elif event.get("timestamp_seconds") is not None:
                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    float(event["timestamp_seconds"]) * 1000.0,
                )
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


def render_media_example(
    *,
    source_path: str,
    bbox,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bytes:
    values = list(bbox) if bbox is not None else []
    if len(values) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    frame = _read_media_frame(
        {
            "source_path": source_path,
            "bbox": values,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
        }
    )
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, values)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise FileNotFoundError("该历史事件的裁切框无效")
    pad_w, pad_h = int(box_width * 0.1), int(box_height * 0.1)
    left, top = max(0, int(x1) - pad_w), max(0, int(y1) - pad_h)
    right, bottom = min(width, int(x2) + pad_w), min(height, int(y2) + pad_h)
    if right <= left or bottom <= top:
        raise FileNotFoundError("该历史事件的裁切框超出图像范围")
    crop = frame[top:bottom, left:right]
    ch, cw = crop.shape[:2]
    if ch != cw:
        size = max(ch, cw)
        pt, pl = (size - ch) // 2, (size - cw) // 2
        crop = cv2.copyMakeBorder(
            crop,
            pt,
            size - ch - pt,
            pl,
            size - cw - pl,
            cv2.BORDER_CONSTANT,
            value=[114, 114, 114],
        )
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("裁切例图编码失败")
    return encoded.tobytes()
