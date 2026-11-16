"""Unit and regression tests for the torch-free LiteRT DINOv3 runtime."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from threadpoolctl import threadpool_limits

from system.backend import dinov3_learning as learning
from system.dino.preprocess import (
    preprocess_batch_nhwc,
    preprocess_image,
    preprocess_image_nhwc,
)
from system.dinov3.deployment import publish_model
from system.dinov3.encoder import FEATURE_DIM, PREPROCESSING, sha256
from system.dinov3.experiment import EXPERIMENT_RECIPE
from system.dinov3.litert_encoder import LiteRtDinoV3Encoder
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.observation_support import make_dinov3_observation_id
from system.dinov3.runtime import (
    load_dinov3_model,
    load_manifest_encoder,
    read_manifest,
)


class FakeEncoderInterpreter:
    """Thread-checking fake LiteRT interpreter for [B,224,224,3] -> [B,768]."""

    def __init__(
        self,
        *,
        output_fn=None,
        input_shape=(1, 224, 224, 3),
        output_shape=(1, FEATURE_DIM),
        input_dtype=np.float32,
        output_dtype=np.float32,
        delay_s: float = 0.0,
    ):
        self.output_fn = output_fn
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.delay_s = delay_s
        self.resized_batches: list[int] = []
        self.inputs: list[np.ndarray] = []
        self._allocated = False
        self._current_input: np.ndarray | None = None
        self._current_output: np.ndarray | None = None
        self._active = 0
        self.concurrent_invocations = 0
        self._probe_lock = threading.Lock()

    def resize_tensor_input(self, index: int, shape: list[int]) -> None:
        assert index == 0
        self.input_shape = list(shape)
        self.output_shape = [int(shape[0]), int(self.output_shape[1])]
        self.resized_batches.append(int(shape[0]))
        self._allocated = False

    def allocate_tensors(self) -> None:
        self._allocated = True

    def get_input_details(self):
        return [
            {
                "index": 0,
                "shape": np.asarray(self.input_shape, dtype=np.int32),
                "dtype": self.input_dtype,
            }
        ]

    def get_output_details(self):
        return [
            {
                "index": 1,
                "shape": np.asarray(self.output_shape, dtype=np.int32),
                "dtype": self.output_dtype,
            }
        ]

    def set_tensor(self, index: int, tensor: np.ndarray) -> None:
        assert index == 0 and self._allocated
        self._current_input = tensor.copy()
        self.inputs.append(self._current_input)

    def invoke(self) -> None:
        assert self._allocated and self._current_input is not None
        with self._probe_lock:
            self._active += 1
            if self._active > 1:
                self.concurrent_invocations += 1
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.output_fn is not None:
                self._current_output = self.output_fn(self._current_input)
            else:
                out = np.zeros((len(self._current_input), FEATURE_DIM), dtype=np.float32)
                out[:, 0] = 2.0
                out[:, 1] = self._current_input[:, 0, 0, 0]
                norms = np.linalg.norm(out, axis=1, keepdims=True)
                self._current_output = (out / norms).astype(np.float32)
        finally:
            with self._probe_lock:
                self._active -= 1

    def get_tensor(self, index: int) -> np.ndarray:
        assert index == 1 and self._current_output is not None
        return self._current_output


def _make_litert_package(tmp_path: Path, *, recipe: str = EXPERIMENT_RECIPE):
    pkg = tmp_path / "litert_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    tflite = pkg / "encoder.tflite"
    tflite.write_bytes(b"fake-litert-encoder")
    pth_provenance = "a" * 64

    rng = np.random.default_rng(7)
    raw = rng.normal(0, 0.01, (16, FEATURE_DIM)).astype(np.float32)
    raw[:8, 0] += 1.0
    raw[8:, 1] += 1.0
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    unknown = rng.normal(size=(8, FEATURE_DIM)).astype(np.float32)
    unknown /= np.linalg.norm(unknown, axis=1, keepdims=True)
    labels = np.asarray(["A"] * 8 + ["B"] * 8)
    cameras = np.asarray([f"c{i // 4}" for i in range(16)])
    sequences = np.asarray([f"s{i // 2}" for i in range(16)])
    cal = dict(
        features=np.r_[raw, unknown],
        labels=np.r_[labels, ["U"] * 8],
        cameras=np.asarray(["cal"] * 24),
        sequences=np.asarray([f"q{i // 2}" for i in range(24)]),
        contents=np.asarray([f"calcontent{i}:0" for i in range(24)]),
        encoder_sha256=np.asarray(pth_provenance),
    )
    with threadpool_limits(limits=1):
        model = DinoV3RidgeWithinModel(inference_recipe=recipe).fit(
            raw,
            labels,
            cameras,
            sequences,
            origin=np.zeros(FEATURE_DIM),
            transform=np.eye(FEATURE_DIM),
            metadata=dict(encoder_sha256=pth_provenance, preprocessing=PREPROCESSING),
        )
        model.calibrate(
            cal["features"],
            cal["labels"],
            cal["cameras"],
            cal["sequences"],
            contents=cal["contents"],
        )

    ckpt = pkg / "model.neri-model"
    model.save(ckpt)
    cal_path = pkg / "calibration.npz"
    np.savez_compressed(cal_path, **cal)
    payload = {
        "schema_version": 1,
        "backend": "dinov3",
        "architecture": "dinov3_vitb16",
        "encoder_backend": "litert",
        "encoder_weights": tflite.name,
        "encoder_litert_sha256": sha256(tflite),
        "encoder_sha256": pth_provenance,
        "checkpoint": ckpt.name,
        "checkpoint_sha256": sha256(ckpt),
        "calibration": cal_path.name,
        "calibration_sha256": sha256(cal_path),
        "head_type": "ridge_within_experiment" if recipe == EXPERIMENT_RECIPE else "ridge_within",
        "inference_recipe": recipe,
        "feature_dim": FEATURE_DIM,
        "preprocessing": PREPROCESSING,
        "event_aggregation": "complete_sequence",
        "requires_detector": True,
        "supports_video_fast": False,
        "supports_video_all": False,
        "display_name": "Neri1.2preview (LiteRT)",
    }
    manifest = pkg / "Neri1.2preview-litert.neri.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest, payload, model, cal, raw


def test_numpy_preprocessing_is_bitwise_equal_to_pytorch(tmp_path):
    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, (37, 83, 3), dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    pil = Image.fromarray(rgb, mode="RGB")
    img_path = tmp_path / "sample.png"
    pil.save(img_path)

    nhwc_rgb = preprocess_image_nhwc(rgb, array_color="rgb")
    nhwc_bgr = preprocess_image_nhwc(bgr, array_color="bgr")
    nhwc_pil = preprocess_image_nhwc(pil)
    nhwc_path = preprocess_image_nhwc(img_path)

    assert nhwc_rgb.shape == (224, 224, 3) and nhwc_rgb.dtype == np.float32
    np.testing.assert_array_equal(nhwc_rgb, nhwc_bgr)
    np.testing.assert_array_equal(nhwc_rgb, nhwc_pil)
    np.testing.assert_array_equal(nhwc_rgb, nhwc_path)

    batch = preprocess_batch_nhwc([rgb, bgr, pil, str(img_path)], array_color="rgb")
    assert batch.shape == (4, 224, 224, 3) and batch.dtype == np.float32

    pt = preprocess_image(rgb, array_color="rgb").numpy().transpose(1, 2, 0)
    np.testing.assert_array_equal(nhwc_rgb, pt)


def test_litert_encoder_inputs_batching_normalization_and_preferences(tmp_path):
    tflite = tmp_path / "encoder.tflite"
    tflite.write_bytes(b"litert-weights")
    fake = FakeEncoderInterpreter()
    encoder = LiteRtDinoV3Encoder(
        tflite,
        sha256(tflite),
        device="cuda",
        batch_size=2,
        num_threads=2,
        interpreter_factory=lambda **_kw: fake,
    )
    assert encoder.device == "cpu"
    assert encoder.use_fp16 is False
    encoder.use_fp16 = True
    assert encoder.use_fp16 is False

    rgb = np.full((25, 45, 3), (30, 60, 90), dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    pil = Image.fromarray(rgb, mode="RGB")
    img_path = tmp_path / "crop.png"
    pil.save(img_path)

    out_rgb = encoder.encode([rgb, pil, img_path], array_color="rgb")
    out_bgr = encoder.encode([bgr], array_color="bgr")
    assert out_rgb.shape == (3, FEATURE_DIM) and out_rgb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out_rgb, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out_rgb[0], out_rgb[1])
    np.testing.assert_array_equal(out_rgb[0], out_rgb[2])
    np.testing.assert_array_equal(out_rgb[0], out_bgr[0])
    assert fake.resized_batches == [1, 2, 1]

    pre_list = [preprocess_image_nhwc(rgb), preprocess_image_nhwc(bgr, array_color="bgr")]
    pre_batch = preprocess_batch_nhwc([rgb, pil])
    np.testing.assert_array_equal(encoder.encode_preprocessed(pre_list), out_rgb[:2])
    np.testing.assert_array_equal(encoder.encode_preprocessed(pre_batch), out_rgb[:2])
    np.testing.assert_array_equal(encoder.encode_nhwc(pre_batch), out_rgb[:2])


def test_litert_encoder_validation_and_thread_safety(tmp_path):
    tflite = tmp_path / "encoder.tflite"
    tflite.write_bytes(b"litert-weights")

    with pytest.raises(ValueError, match="SHA-256"):
        LiteRtDinoV3Encoder(
            tflite,
            "0" * 64,
            interpreter_factory=lambda **_kw: pytest.fail("must verify hash first"),
        )
    with pytest.raises(FileNotFoundError):
        LiteRtDinoV3Encoder(tmp_path / "missing.tflite")

    bad_shape = FakeEncoderInterpreter(output_shape=(1, 128))
    with pytest.raises(ValueError, match="768"):
        LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: bad_shape)

    nan_interp = FakeEncoderInterpreter(
        output_fn=lambda x: np.full((len(x), FEATURE_DIM), np.nan, dtype=np.float32)
    )
    nan_enc = LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: nan_interp)
    with pytest.raises(ValueError, match="invalid CLS"):
        nan_enc.encode([np.zeros((12, 12, 3), np.uint8)])

    zero_interp = FakeEncoderInterpreter(
        output_fn=lambda x: np.zeros((len(x), FEATURE_DIM), dtype=np.float32)
    )
    zero_enc = LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: zero_interp)
    with pytest.raises(ValueError, match="zero CLS"):
        zero_enc.encode([np.zeros((12, 12, 3), np.uint8)])

    wrong_dim = FakeEncoderInterpreter(
        output_fn=lambda x: np.ones((len(x), 10), dtype=np.float32)
    )
    wrong_enc = LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: wrong_dim)
    with pytest.raises(ValueError, match="invalid CLS"):
        wrong_enc.encode([np.zeros((12, 12, 3), np.uint8)])

    non_unit_interp = FakeEncoderInterpreter(
        output_fn=lambda x: np.full((len(x), FEATURE_DIM), 0.1, dtype=np.float32)
    )
    non_unit_enc = LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: non_unit_interp)
    with pytest.raises(ValueError, match="non-unit-norm"):
        non_unit_enc.encode([np.zeros((12, 12, 3), np.uint8)])

    near_unit_raw = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    near_unit_raw[0, 0] = np.float32(1.0 + 3e-6)
    near_unit_raw[1, 1] = np.float32(1.0 - 4e-6)
    near_unit_interp = FakeEncoderInterpreter(output_fn=lambda _x: near_unit_raw.copy())
    near_unit_enc = LiteRtDinoV3Encoder(tflite, interpreter_factory=lambda **_kw: near_unit_interp)
    near_unit_out = near_unit_enc.encode([np.zeros((12, 12, 3), np.uint8)] * 2)
    np.testing.assert_array_equal(near_unit_out, near_unit_raw)
    assert float(np.linalg.norm(near_unit_out[0])) != 1.0

    with pytest.raises(ValueError, match="At least one crop"):
        zero_enc.encode([])
    with pytest.raises(ValueError, match="Unsupported array color"):
        zero_enc.encode([np.zeros((8, 8, 3), np.uint8)], array_color="rgba")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected finite float32"):
        zero_enc.encode_preprocessed([np.zeros((3, 224, 224), np.float32)])
    with pytest.raises(ValueError, match="Expected finite float32"):
        zero_enc.encode_nhwc(np.full((1, 224, 224, 3), np.inf, dtype=np.float32))

    concurrent_interp = FakeEncoderInterpreter(delay_s=0.01)
    concurrent_enc = LiteRtDinoV3Encoder(
        tflite,
        batch_size=2,
        interpreter_factory=lambda **_kw: concurrent_interp,
    )
    crop = np.full((16, 16, 3), 80, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: concurrent_enc.encode([crop, crop]), range(8)))
    assert concurrent_interp.concurrent_invocations == 0
    for res in results[1:]:
        np.testing.assert_array_equal(res, results[0])


def test_manifest_contract_branching_and_cross_backend_fingerprint(tmp_path):
    manifest, payload, _model, cal, raw = _make_litert_package(tmp_path)
    read_back = read_manifest(manifest)
    assert read_back == payload
    assert "encoder_code" not in read_back

    bad = dict(payload, encoder_backend="onnx")
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder_backend"):
        read_manifest(manifest)

    bad = {k: v for k, v in payload.items() if k != "encoder_litert_sha256"}
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder_litert_sha256"):
        read_manifest(manifest)

    pth = manifest.parent / "wrong.pth"
    pth.write_bytes(b"not-tflite")
    bad = dict(payload, encoder_weights=pth.name, encoder_litert_sha256=sha256(pth))
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.tflite"):
        read_manifest(manifest)

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    bad_hash_payload = dict(payload, encoder_litert_sha256="b" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        load_manifest_encoder(manifest, bad_hash_payload)

    fake_lrt = FakeEncoderInterpreter()
    lrt_runtime = load_dinov3_model(
        manifest,
        device="cuda",
        encoder_factory=lambda w, s, device=None, **kw: LiteRtDinoV3Encoder(
            w, s, device=device, interpreter_factory=lambda **_k: fake_lrt, **kw
        ),
    )
    assert lrt_runtime.classifier.encoder.device == "cpu"
    assert payload["encoder_litert_sha256"] in lrt_runtime.classifier.model_fingerprint

    alt_tflite = manifest.parent / "encoder_fp16.tflite"
    alt_tflite.write_bytes(b"different-tflite-variant")
    alt_payload = dict(
        payload,
        encoder_weights=alt_tflite.name,
        encoder_litert_sha256=sha256(alt_tflite),
    )
    alt_manifest = manifest.parent / "alt-litert.neri.json"
    alt_manifest.write_text(json.dumps(alt_payload), encoding="utf-8")
    alt_runtime = load_dinov3_model(
        alt_manifest,
        encoder_factory=lambda *a, **kw: SimpleNamespace(device="cpu", use_fp16=False),
    )
    assert alt_runtime.classifier.model_fingerprint != lrt_runtime.classifier.model_fingerprint

    code_dir = manifest.parent / "encoder-code"
    (code_dir / "dinov3" / "hub").mkdir(parents=True)
    (code_dir / "dinov3" / "hub" / "backbones.py").write_text("# mock", encoding="utf-8")
    (code_dir / "LICENSE.md").write_text("mock", encoding="utf-8")
    pt_weights = manifest.parent / "encoder.pth"
    pt_weights.write_bytes(b"pt-weights")
    pt_payload = {
        k: v
        for k, v in payload.items()
        if k not in {"encoder_backend", "encoder_litert_sha256"}
    }
    pt_payload.update(
        encoder_weights=pt_weights.name,
        encoder_code=code_dir.name,
        encoder_sha256=sha256(pt_weights),
    )
    with threadpool_limits(limits=1):
        pt_model = DinoV3RidgeWithinModel(inference_recipe=EXPERIMENT_RECIPE).fit(
            raw,
            np.asarray(["A"] * 8 + ["B"] * 8),
            np.asarray([f"c{i // 4}" for i in range(16)]),
            np.asarray([f"s{i // 2}" for i in range(16)]),
            origin=np.zeros(FEATURE_DIM),
            transform=np.eye(FEATURE_DIM),
            metadata=dict(encoder_sha256=sha256(pt_weights), preprocessing=PREPROCESSING),
        )
        pt_model.calibrate(
            cal["features"],
            cal["labels"],
            cal["cameras"],
            cal["sequences"],
            contents=cal["contents"],
        )
    pt_ckpt = manifest.parent / "pt.neri-model"
    pt_model.save(pt_ckpt)
    pt_payload["checkpoint"] = pt_ckpt.name
    pt_payload["checkpoint_sha256"] = sha256(pt_ckpt)
    pt_manifest = manifest.parent / "pt.neri.json"
    pt_manifest.write_text(json.dumps(pt_payload), encoding="utf-8")
    pt_read = read_manifest(pt_manifest)
    assert "encoder_backend" not in pt_read
    pt_runtime = load_dinov3_model(
        pt_manifest,
        encoder_factory=lambda *a, **kw: SimpleNamespace(device="cpu", use_fp16=False),
    )
    assert pt_runtime.classifier.model_fingerprint != lrt_runtime.classifier.model_fingerprint

    pt_bad = dict(pt_payload, encoder_weights=payload["encoder_weights"])
    pt_manifest.write_text(json.dumps(pt_bad), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.pth"):
        read_manifest(pt_manifest)


def test_publish_and_learning_preserve_litert_fields_and_calibration(tmp_path, monkeypatch):
    import cv2
    import system.dinov3.litert_encoder as litert_mod

    manifest, payload, model, cal, _raw = _make_litert_package(tmp_path)
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))

    published = publish_model(
        model,
        manifest,
        existing_manifest=payload,
        display_name=payload["display_name"],
        calibration_data=cal,
    )
    next_payload = read_manifest(published)
    assert next_payload["encoder_backend"] == "litert"
    assert next_payload["encoder_litert_sha256"] == payload["encoder_litert_sha256"]
    assert next_payload["encoder_weights"] == payload["encoder_weights"]
    assert next_payload["encoder_sha256"] == payload["encoder_sha256"]
    assert "encoder_code" not in next_payload
    assert next_payload["calibration"] != payload["calibration"]

    publish_model(
        model,
        manifest,
        existing_manifest=next_payload,
        display_name=next_payload["display_name"],
    )
    inherited = read_manifest(manifest)
    assert inherited["encoder_backend"] == "litert"
    assert inherited["calibration"] == next_payload["calibration"]
    assert inherited["calibration_sha256"] == next_payload["calibration_sha256"]

    class FakeLiteRtEncoder:
        calls = 0

        def __init__(self, weights, litert_sha, **_kwargs):
            assert Path(weights).name == "encoder.tflite"
            assert litert_sha == payload["encoder_litert_sha256"]

        def encode(self, crops, **_kwargs):
            type(self).calls += 1
            out = np.zeros((len(crops), FEATURE_DIM), dtype=np.float32)
            out[:, 2] = 1.0
            return out

    monkeypatch.setattr(litert_mod, "LiteRtDinoV3Encoder", FakeLiteRtEncoder)
    photo = tmp_path / "new_species.jpg"
    cv2.imwrite(str(photo), np.full((160, 160, 3), 130, dtype=np.uint8))
    bbox = [20, 20, 140, 140]
    box = {
        "物种": "C",
        "边界框": bbox,
        "observation_id": make_dinov3_observation_id(photo, 0, bbox),
    }
    learning.stage_feedback(manifest, photo.parent, photo, box, "update", "op-litert")
    result = learning.update_model(manifest)
    assert result["updated_count"] == 1
    assert FakeLiteRtEncoder.calls == 1

    updated_payload = read_manifest(manifest)
    assert updated_payload["encoder_backend"] == "litert"
    assert updated_payload["encoder_litert_sha256"] == payload["encoder_litert_sha256"]
    assert updated_payload["encoder_weights"] == payload["encoder_weights"]
    assert "encoder_code" not in updated_payload

    runtime = load_dinov3_model(
        manifest,
        encoder_factory=lambda *a, **kw: SimpleNamespace(device="cpu", use_fp16=False),
    )
    probe = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    probe[0, 2] = 1.0
    assert runtime.classifier.classify_features(probe)[0].best_known_species == "C"


def test_torch_free_subprocess_executes_runtime_deployment_and_learning(tmp_path):
    manifest, _payload, _model, _cal, _raw = _make_litert_package(tmp_path)
    script = r"""
import builtins
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, sys.argv[3])

_real_import = builtins.__import__
def _block_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch is intentionally unavailable in this test", name="torch")
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _block_torch

from system.dino.preprocess import preprocess_image_nhwc, preprocess_batch_nhwc
import system.dinov3.litert_encoder as litert_mod
from system.dinov3.runtime import load_dinov3_model, read_manifest
from system.dinov3.deployment import publish_model
from system.backend import dinov3_learning as learning
from system.dinov3.observation_support import make_dinov3_observation_id

class SubprocInterpreter:
    def __init__(self, **_kw):
        self._b = 1
    def resize_tensor_input(self, _idx, shape):
        self._b = int(shape[0])
    def allocate_tensors(self):
        pass
    def get_input_details(self):
        return [{"index": 0, "shape": np.asarray([self._b, 224, 224, 3]), "dtype": np.float32}]
    def get_output_details(self):
        return [{"index": 1, "shape": np.asarray([self._b, 768]), "dtype": np.float32}]
    def set_tensor(self, _idx, tensor):
        self._x = tensor
    def invoke(self):
        out = np.zeros((len(self._x), 768), dtype=np.float32)
        out[:, 2] = 1.0
        self._y = out
    def get_tensor(self, _idx):
        return self._y

_orig_cls = litert_mod.LiteRtDinoV3Encoder
litert_mod.LiteRtDinoV3Encoder = lambda w, s=None, **kw: _orig_cls(
    w, s, interpreter_factory=lambda **_k: SubprocInterpreter(), **kw
)

manifest = Path(sys.argv[1])
state_dir = Path(sys.argv[2])
os.environ["NERI_DINO_STATE_DIR"] = str(state_dir)

rt = load_dinov3_model(manifest, device="cuda")
assert rt.classifier.encoder.device == "cpu"
assert rt.classifier.encoder.use_fp16 is False
crop = np.full((32, 48, 3), 120, dtype=np.uint8)
nhwc = preprocess_image_nhwc(crop, array_color="bgr")
assert nhwc.shape == (224, 224, 3)
batch = preprocess_batch_nhwc([crop, crop])
emb_a = rt.classifier.encoder.encode([crop, crop])
emb_b = rt.classifier.encoder.encode_preprocessed(batch)
np.testing.assert_array_equal(emb_a, emb_b)
preds = rt.classifier.classify_crops([crop])
assert len(preds) == 1

photo = state_dir.parent / "subproc_c.png"
from PIL import Image
Image.fromarray(np.full((120, 120, 3), 130, dtype=np.uint8), mode="RGB").save(photo)
bbox = [10, 10, 100, 100]
box = {"物种": "C", "边界框": bbox, "observation_id": make_dinov3_observation_id(photo, 0, bbox)}
learning.stage_feedback(manifest, photo.parent, photo, box, "update", "op-subproc")
res = learning.update_model(manifest)
assert res["updated_count"] == 1
updated = read_manifest(manifest)
assert updated["encoder_backend"] == "litert"
assert "encoder_code" not in updated
publish_model(rt.model, manifest, existing_manifest=updated, display_name=updated["display_name"])
assert "torch" not in sys.modules
print("TORCH_FREE_OK")
"""
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script, str(manifest), str(tmp_path / "sub_state"), str(repo_root)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TORCH_FREE_OK" in proc.stdout


def test_real_litert_asset_single_crop_smoke_if_available():
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        pytest.skip("ai_edge_litert is not installed in this interpreter")
    repo_root = Path(__file__).resolve().parents[1]
    tflite = repo_root / "res/model/cls/bundled/Neri1.2preview-litert-assets/encoder.tflite"
    if not tflite.is_file():
        tflite = repo_root / "runs/Neri1.2preview-litert/dinov3_vitb16_fp32.tflite"
    if not tflite.is_file():
        pytest.skip("runs/Neri1.2preview-litert/dinov3_vitb16_fp32.tflite not present")
    encoder = LiteRtDinoV3Encoder(tflite, num_threads=2, batch_size=1)
    assert encoder.device == "cpu" and encoder.use_fp16 is False
    crop = np.full((64, 64, 3), 114, dtype=np.uint8)
    features = encoder.encode([crop])
    assert features.shape == (1, FEATURE_DIM)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)
