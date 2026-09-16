from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps

from system.dinov2.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_RGB,
    preprocess_image,
)


def _benchmark_reference(image: Image.Image) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    small = ImageOps.contain(image, (224, 224), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), LETTERBOX_RGB)
    canvas.paste(small, ((224 - small.width) // 2, (224 - small.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def test_preprocess_matches_benchmark_letterbox_exactly_for_odd_dimensions():
    data = np.zeros((173, 317, 3), dtype=np.uint8)
    data[..., 0] = np.arange(317, dtype=np.uint8)[None, :]
    data[..., 1] = 91
    data[..., 2] = 203
    image = Image.fromarray(data, mode="RGB")

    actual = preprocess_image(image)
    expected = _benchmark_reference(image)

    assert actual.shape == (3, 224, 224)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_preprocess_bgr_array_matches_rgb_reference():
    rgb = np.zeros((37, 19, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    rgb[..., 1] = 17
    rgb[..., 2] = 61
    bgr = rgb[..., ::-1].copy()

    actual = preprocess_image(bgr, array_color="bgr")
    expected = preprocess_image(rgb, array_color="rgb")

    assert torch.equal(actual, expected)


def test_preprocess_rejects_invalid_array_color():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        preprocess_image(image, array_color="xyz")
    except ValueError as exc:
        assert "color" in str(exc).lower()
    else:
        raise AssertionError("invalid array color should fail")
