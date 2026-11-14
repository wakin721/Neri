from pathlib import Path

import numpy as np
import pytest
import torch

from system.dino.preprocess import preprocess_image
from system.dinov3.encoder import DinoV3Encoder, sha256


class FrozenBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token = torch.nn.Parameter(torch.arange(1, 769, dtype=torch.float32))
        self.batches = []

    def forward_features(self, batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert batch.dtype == torch.float32
        assert not torch.backends.cuda.matmul.allow_tf32
        self.batches.append(batch.detach().clone())
        return {"x_norm_clstoken": self.token[None].expand(len(batch), -1),
                "x_norm_regtokens": torch.full((len(batch), 4, 768), float("nan"))}


def test_frozen_cls_only_preprocessing_hash_and_precision(tmp_path):
    path = tmp_path / "encoder.pth"
    backbone = FrozenBackbone()
    torch.save(backbone.state_dict(), path)
    encoder = DinoV3Encoder(path, tmp_path, sha256(path), device="cpu", batch_size=2,
                            model_factory=lambda source: backbone)
    image = np.full((31, 99, 3), (22, 44, 88), dtype=np.uint8)
    old = torch.backends.cuda.matmul.allow_tf32
    values = encoder.encode([image] * 3)
    assert values.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
    torch.testing.assert_close(backbone.batches[0][0], preprocess_image(image))
    assert [len(batch) for batch in backbone.batches] == [2, 1]
    preprocessed = encoder.encode_preprocessed([preprocess_image(image)] * 3)
    np.testing.assert_array_equal(preprocessed, values)
    assert not backbone.token.requires_grad
    assert torch.backends.cuda.matmul.allow_tf32 == old
    encoder.use_fp16 = True
    assert encoder.use_fp16 is False
    with pytest.raises(ValueError, match="SHA-256"):
        DinoV3Encoder(path, tmp_path, "a" * 64, model_factory=lambda _: pytest.fail("must verify first"))


def test_precision_restored_after_failed_forward(tmp_path):
    path = tmp_path / "encoder.pth"
    model = FrozenBackbone()
    torch.save(model.state_dict(), path)
    encoder = DinoV3Encoder(path, tmp_path, sha256(path), device="cpu", model_factory=lambda _: model)
    def fail(batch):
        raise RuntimeError("forward failed")
    model.forward_features = fail
    old = torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError, match="forward failed"):
        encoder.encode([np.zeros((5, 5, 3), np.uint8)])
    assert torch.backends.cuda.matmul.allow_tf32 == old


@pytest.mark.parametrize("raw", ["../head", "C:/head", "a\\head", "/head", "a//head", "a/./head"])
def test_manifest_assets_stay_inside_package(tmp_path, raw):
    from system.dinov3.runtime import resolve_manifest_asset
    with pytest.raises(ValueError):
        resolve_manifest_asset(tmp_path / "joint.neri.json", raw)
