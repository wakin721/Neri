"""Local upload-copy redaction, adapted from Ultralytics' object-blurring guide.

https://docs.ultralytics.com/guides/object-blurring/
Use LiteRT prediction rather than ObjectBlurrer's tracking/annotation for independent
photos: never draw labels, keep tracker state, or send images to an inference API.
COCO detects whole people/vehicles, not separate faces/plates. Redact the whole
object conservatively, plus screens/books; this is not an anonymity guarantee.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

from PIL import Image

MODEL_NAME = "yolo26n_w8a32.tflite"
MODEL_URL = "https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_w8a32.tflite"
MODEL_SHA256 = "d9cef07ce652ccfa9ce58e4ac8a4df98ff037739a9dad20a8afcae21b545df73"
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_DETECTIONS = 1000
PRIVACY_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 62: "tv",
    63: "laptop", 67: "cell phone", 73: "book",
}
PROCESSING_VERSION = "metadata_removed_local_litert_blurring_v1"


class PrivacyProcessingError(RuntimeError):
    """Retry the sample without requesting any remote upload capabilities."""


def _verified_model(path: Path) -> bool:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MODEL_BYTES:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == MODEL_SHA256


class PrivacyBlurrer:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        # Status polling must not import ML runtimes, download, or decode images.
        return {
            "required": True,
            "model": MODEL_NAME,
            "runtime": "LiteRT",
            "requires_pytorch": False,
            "dependencies_available": all(
                importlib.util.find_spec(name) is not None
                for name in ("ai_edge_litert", "cv2", "numpy")
            ),
            "model_cached": (self.cache_dir / MODEL_NAME).is_file(),
        }

    def _model_path(self) -> Path:
        path = self.cache_dir / MODEL_NAME
        if _verified_model(path):
            return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.cache_dir, suffix=".part")
        try:
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Neri"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_MODEL_BYTES:
                            raise PrivacyProcessingError("privacy_model_too_large")
                        output.write(chunk)
            if not _verified_model(Path(temporary)):
                raise PrivacyProcessingError("privacy_model_checksum_mismatch")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return path

    def _load_model(self):
        from .litert_detector import LiteRtDetector

        if not self.status()["dependencies_available"]:
            raise PrivacyProcessingError("privacy_runtime_unavailable")
        model = LiteRtDetector(self._model_path(), classes=PRIVACY_CLASSES, max_detections=MAX_DETECTIONS)
        if any(model.names.get(key) != name for key, name in PRIVACY_CLASSES.items()):
            raise PrivacyProcessingError("privacy_model_classes_mismatch")
        return model

    def __call__(self, image: Image.Image) -> tuple[Image.Image, int]:
        with self._lock:
            try:
                import cv2
                import numpy as np

                if self._model is None:
                    self._model = self._load_model()
                # Never pass source filenames (including paths/GPS) to inference.
                frame = np.asarray(image.convert("RGB")).copy()
                results = np.asarray(self._model.predict(frame))
                if results.ndim != 2 or results.shape[1] != 6:
                    raise PrivacyProcessingError("privacy_detection_incomplete")
                if len(results) >= MAX_DETECTIONS or not np.isfinite(results).all():
                    raise PrivacyProcessingError("privacy_detection_incomplete")
                height, width = frame.shape[:2]
                regions = 0
                for x1, y1, x2, y2, confidence, class_id in results:
                    if class_id not in PRIVACY_CLASSES or not 0 <= confidence <= 1:
                        raise PrivacyProcessingError("privacy_detection_invalid")
                    if x2 <= x1 or y2 <= y1:
                        raise PrivacyProcessingError("privacy_detection_invalid")
                    # Expand edges to cover borders of faces/plates near a box.
                    dx, dy = max(4, (x2 - x1) * 0.1), max(4, (y2 - y1) * 0.1)
                    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
                    right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
                    if right <= left or bottom <= top:
                        raise PrivacyProcessingError("privacy_detection_invalid")
                    roi = frame[top:bottom, left:right]
                    # Coarse mosaic before the guide's box blur removes fine
                    # face/plate detail even when the containing object is large.
                    small = cv2.resize(roi, (min(8, max(1, roi.shape[1] // 24)),
                                             min(8, max(1, roi.shape[0] // 24))),
                                       interpolation=cv2.INTER_AREA)
                    mosaic = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                    frame[top:bottom, left:right] = cv2.blur(mosaic, (51, 51))
                    regions += 1
                return Image.fromarray(frame), regions
            except PrivacyProcessingError:
                raise
            except Exception:
                # Discard possibly broken predictor state; never fall back to
                # the original pixels and never expose paths in an error.
                self._model = None
                raise PrivacyProcessingError("privacy_processing_unavailable") from None
