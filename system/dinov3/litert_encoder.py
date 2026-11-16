"""Offline, float32 DINOv3 ViT-B/16 extraction using LiteRT on CPU."""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Iterable, Literal

import numpy as np

from system.dino.preprocess import IMAGE_SIZE, ImageInput, preprocess_batch_nhwc

from .encoder import FEATURE_DIM, sha256


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"DINOv3 {name} must be a positive integer")
    return int(value)


class LiteRtDinoV3Encoder:
    """Torch-free LiteRT CPU encoder matching DinoV3Encoder's float32 CLS contract."""

    def __init__(
        self,
        weights,
        encoder_litert_sha256: str | None = None,
        *,
        device=None,
        batch_size: int = 16,
        num_threads: int | None = None,
        interpreter_factory=None,
        **_kwargs,
    ):
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if encoder_litert_sha256 is not None and sha256(self.weights) != encoder_litert_sha256:
            raise ValueError("DINOv3 LiteRT encoder weight SHA-256 does not match the model")
        self.batch_size = _validate_positive_int(batch_size, "batch_size")
        if num_threads is None:
            threads = min(4, os.cpu_count() or 1)
        else:
            threads = _validate_positive_int(num_threads, "num_threads")
        self.num_threads = threads
        # LiteRT encoder always runs in float32 on CPU regardless of caller preferences.
        del device
        self.device = "cpu"
        self._lock = threading.RLock()

        factory = interpreter_factory
        if factory is None:
            from ai_edge_litert.interpreter import Interpreter

            factory = Interpreter

        with self._lock:
            self.interpreter = factory(
                model_path=str(self.weights),
                num_threads=threads,
            )
            inputs = self.interpreter.get_input_details()
            outputs = self.interpreter.get_output_details()
            if len(inputs) != 1 or not outputs:
                raise ValueError("Invalid LiteRT DINOv3 encoder input/output signature")
            self._input_index = inputs[0]["index"]
            self._output_index = outputs[0]["index"]
            self._allocated_batch: int | None = None
            self._ensure_batch(1)
            inputs = self.interpreter.get_input_details()
            outputs = self.interpreter.get_output_details()
            input_shape = tuple(int(value) for value in inputs[0]["shape"])
            output_shape = tuple(int(value) for value in outputs[0]["shape"])
            if len(input_shape) != 4 or input_shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError("LiteRT DINOv3 encoder input must have NHWC shape [B,224,224,3]")
            if inputs[0]["dtype"] != np.float32:
                raise ValueError("LiteRT DINOv3 encoder input must be float32")
            if len(output_shape) != 2 or output_shape[1] != FEATURE_DIM:
                raise ValueError("LiteRT DINOv3 encoder output must have shape [B,768]")
            if outputs[0]["dtype"] != np.float32:
                raise ValueError("LiteRT DINOv3 encoder output must be float32")

    @property
    def use_fp16(self) -> bool:
        return False

    @use_fp16.setter
    def use_fp16(self, _value: object) -> None:
        # Detector preferences do not override the calibrated float32 CPU encoder.
        pass

    def _ensure_batch(self, batch: int) -> None:
        if self._allocated_batch != batch:
            self.interpreter.resize_tensor_input(
                self._input_index, [batch, IMAGE_SIZE, IMAGE_SIZE, 3]
            )
            self.interpreter.allocate_tensors()
            self._allocated_batch = batch

    def encode(
        self,
        crops: Iterable[ImageInput],
        *,
        array_color: Literal["rgb", "bgr"] = "rgb",
    ) -> np.ndarray:
        if array_color not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported array color order: {array_color}")
        items = list(crops)
        if not items:
            raise ValueError("At least one crop is required")
        batch = preprocess_batch_nhwc(items, array_color=array_color)
        return self._encode_nhwc(batch)

    def encode_preprocessed(self, arrays: np.ndarray | Iterable[np.ndarray]) -> np.ndarray:
        """Encode NHWC float32 arrays produced by preprocess_image_nhwc or preprocess_batch_nhwc."""
        if isinstance(arrays, np.ndarray):
            if arrays.ndim != 4 or not len(arrays):
                raise ValueError("Expected finite float32 preprocessed crops [N,224,224,3]")
            return self._encode_nhwc(arrays)
        items = list(arrays)
        if not items:
            raise ValueError("At least one preprocessed crop is required")
        prepared = []
        for item in items:
            arr = np.asarray(item)
            if arr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or arr.dtype != np.float32:
                raise ValueError("Expected finite float32 preprocessed crops [224,224,3]")
            prepared.append(arr)
        return self._encode_nhwc(np.stack(prepared, axis=0))

    def encode_nhwc(self, nhwc_batch: np.ndarray) -> np.ndarray:
        return self._encode_nhwc(nhwc_batch)

    def _encode_nhwc(self, nhwc_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(nhwc_batch)
        if (
            batch.ndim != 4
            or batch.shape[0] < 1
            or batch.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3)
            or batch.dtype != np.float32
            or not bool(np.isfinite(batch).all())
        ):
            raise ValueError("Expected finite float32 preprocessed crops [N,224,224,3]")
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        output = []
        with self._lock:
            for begin in range(0, len(batch), self.batch_size):
                chunk = np.ascontiguousarray(batch[begin : begin + self.batch_size], dtype=np.float32)
                self._ensure_batch(len(chunk))
                self.interpreter.set_tensor(self._input_index, chunk)
                self.interpreter.invoke()
                values = np.asarray(self.interpreter.get_tensor(self._output_index)).copy()
                if (
                    values.dtype != np.float32
                    or values.shape != (len(chunk), FEATURE_DIM)
                    or not bool(np.isfinite(values).all())
                ):
                    raise ValueError("DINOv3 LiteRT returned invalid CLS features")
                norms = np.linalg.norm(values, axis=1)
                if not bool(np.isfinite(norms).all()):
                    raise ValueError("DINOv3 LiteRT returned invalid CLS features")
                if bool((norms <= 1e-12).any()):
                    raise ValueError("DINOv3 LiteRT returned zero CLS features")
                if not bool(np.allclose(norms, 1.0, atol=1e-5)):
                    raise ValueError("DINOv3 LiteRT returned non-unit-norm CLS features")
                # TFLite graph already includes L2_NORMALIZATION; preserve exact FP32 outputs.
                output.append(values)
        return np.concatenate(output, axis=0).astype(np.float32, copy=False)
