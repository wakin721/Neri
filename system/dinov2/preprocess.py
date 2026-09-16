"""Benchmark-exact image preprocessing for Neri's DINOv2 runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGE_SIZE = 224
LETTERBOX_RGB = (124, 116, 104)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageInput = str | Path | Image.Image | np.ndarray


def _as_rgb_image(
    value: ImageInput,
    *,
    array_color: Literal["rgb", "bgr"] = "rgb",
) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, np.ndarray):
        if array_color not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported array color order: {array_color}")
        array = np.asarray(value)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Expected an HxWx3 or HxWx4 image array")
        array = array[:, :, :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array_color == "bgr":
            array = array[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    raise TypeError(f"Unsupported image input: {type(value).__name__}")


def letterbox_image(image: Image.Image, *, size: int = IMAGE_SIZE) -> Image.Image:
    if size <= 0:
        raise ValueError("Letterbox size must be positive")
    image = image.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image dimensions must be positive")
    small = ImageOps.contain(
        image,
        (size, size),
        method=Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (size, size), LETTERBOX_RGB)
    canvas.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    return canvas


def preprocess_image(
    value: ImageInput,
    *,
    array_color: Literal["rgb", "bgr"] = "rgb",
) -> torch.Tensor:
    image = letterbox_image(_as_rgb_image(value, array_color=array_color))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std
