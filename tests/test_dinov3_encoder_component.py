from pathlib import Path
from types import SimpleNamespace

import system.dinov3.encoder as encoder
from system.dinov3.checkpoint import DinoV3Checkpoint


def _checkpoint(tmp_path: Path) -> DinoV3Checkpoint:
    return DinoV3Checkpoint(
        path=tmp_path / 'head.pt',
        backbone='dinov3_vitb16',
        feature_dim=768,
        classes=('species',),
        head_weight=None,
        head_bias=None,
        prototypes=None,
        threshold=0.4,
        encoder_weights='models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        encoder_sha256='73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c',
        preprocessing='letterbox224_imagenet',
        event_aggregation='mean_l2_normalized_crop_embeddings',
        fingerprint='f' * 64,
    )


def test_resolve_encoder_weights_falls_back_to_installed_component(tmp_path, monkeypatch):
    installed = tmp_path / 'res' / 'model' / 'DINOv3'
    backbone = installed / 'dinov3-vitb16' / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    backbone.parent.mkdir(parents=True)
    backbone.write_bytes(b'weights')
    monkeypatch.setattr(
        encoder,
        'dinov3_component_paths',
        lambda: SimpleNamespace(backbone=backbone, source_root=installed / 'source'),
        raising=False,
    )

    resolved = encoder.resolve_encoder_weights(_checkpoint(tmp_path))

    assert resolved == backbone.resolve()


def test_default_model_factory_prefers_installed_source(tmp_path, monkeypatch):
    installed = tmp_path / 'res' / 'model' / 'DINOv3'
    source_root = installed / 'source'
    (source_root / 'dinov3').mkdir(parents=True)
    monkeypatch.delenv('NERI_DINOV3_SOURCE', raising=False)
    monkeypatch.setattr(
        encoder,
        'dinov3_component_paths',
        lambda: SimpleNamespace(source_root=source_root),
        raising=False,
    )
    observed = {}
    sentinel = object()

    def fake_import(name):
        observed['name'] = name
        observed['path_present'] = str(source_root.resolve()) in encoder.sys.path
        return SimpleNamespace(dinov3_vitb16=sentinel)

    monkeypatch.setattr(encoder.importlib, 'import_module', fake_import)

    factory = encoder._default_model_factory()

    assert factory is sentinel
    assert observed == {'name': 'dinov3.hub.backbones', 'path_present': True}
