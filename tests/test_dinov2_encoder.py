from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from system.dinov2.checkpoint import validate_checkpoint
from system.dinov2.encoder import (
    DinoV2Encoder,
    EncoderAssetError,
    _default_model_factory,
    resolve_encoder_dir,
)
from tests.dinov2_fixtures import make_dinov2_payload


class FakeDinoV2Model:
    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *, pixel_values):
        batch = pixel_values.shape[0]
        hidden = torch.zeros((batch, 2, 768), dtype=torch.float32, device=pixel_values.device)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        hidden[:, 1, 2] = 100.0
        return SimpleNamespace(last_hidden_state=hidden)


def _checkpoint_with_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"dinov2-test-weights")
    payload = make_dinov2_payload()
    payload["encoder_weights"] = "model/model.safetensors"
    payload["encoder_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    checkpoint = validate_checkpoint(payload, path=tmp_path / "classifier.pt")
    return checkpoint, model_dir


def test_resolve_encoder_dir_prefers_explicit_model_directory(tmp_path):
    checkpoint, model_dir = _checkpoint_with_model(tmp_path)
    assert resolve_encoder_dir(checkpoint, encoder_dir=model_dir) == model_dir.resolve()


def test_encoder_uses_cls_token_and_l2_normalizes(tmp_path):
    checkpoint, model_dir = _checkpoint_with_model(tmp_path)
    encoder = DinoV2Encoder(
        checkpoint,
        encoder_dir=model_dir,
        model_factory=lambda path: FakeDinoV2Model(),
        device="cpu",
    )
    image = np.zeros((16, 24, 3), dtype=np.uint8)

    features = encoder.encode([image])

    assert features.shape == (1, 768)
    assert features.dtype == np.float32
    assert features[0, 0] == pytest.approx(0.6)
    assert features[0, 1] == pytest.approx(0.8)
    assert features[0, 2] == pytest.approx(0.0)
    assert np.linalg.norm(features[0]) == pytest.approx(1.0)


def test_encoder_rejects_weight_hash_mismatch(tmp_path):
    checkpoint, model_dir = _checkpoint_with_model(tmp_path)
    (model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(EncoderAssetError, match="hash mismatch"):
        DinoV2Encoder(
            checkpoint,
            encoder_dir=model_dir,
            model_factory=lambda path: FakeDinoV2Model(),
            device="cpu",
        )


def test_default_model_factory_uses_offline_safe_transformers_flags(tmp_path, monkeypatch):
    observed = {}
    sentinel = object()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            observed["path"] = path
            observed.update(kwargs)
            return sentinel

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    result = _default_model_factory(tmp_path)

    assert result is sentinel
    assert observed["path"] == str(tmp_path.resolve())
    assert observed["local_files_only"] is True
    assert observed["trust_remote_code"] is False


def test_encoder_rejects_requested_cuda_when_unavailable(tmp_path, monkeypatch):
    checkpoint, model_dir = _checkpoint_with_model(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EncoderAssetError, match="CUDA"):
        DinoV2Encoder(
            checkpoint,
            encoder_dir=model_dir,
            model_factory=lambda path: FakeDinoV2Model(),
            device="cuda",
        )


def test_encoder_requires_nonempty_crop_batch(tmp_path):
    checkpoint, model_dir = _checkpoint_with_model(tmp_path)
    encoder = DinoV2Encoder(
        checkpoint,
        encoder_dir=model_dir,
        model_factory=lambda path: FakeDinoV2Model(),
        device="cpu",
    )
    with pytest.raises(ValueError, match="crop"):
        encoder.encode([])
