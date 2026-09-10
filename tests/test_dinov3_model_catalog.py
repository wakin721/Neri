import json
from pathlib import Path

import system.model_sync.catalog as catalog
from system.model_sync.catalog import discover_models, resolve_saved_model_path
from system.model_sync.layout import get_model_layout


def test_manifest_exposes_dinov3_capabilities(tmp_path):
    layout = get_model_layout(tmp_path)
    head = layout.cls_user / 'head.pt'
    head.write_bytes(b'x')
    manifest = layout.cls_user / 'head.neri.json'
    manifest.write_text(
        json.dumps({
            'backend': 'dinov3',
            'display_name': 'DINOv3 reviewed',
            'checkpoint': 'head.pt',
            'architecture': 'dinov3_vitb16',
            'feature_dim': 768,
            'requires_detector': True,
            'supports_video_fast': True,
            'supports_video_all': False,
        }),
        encoding='utf-8',
    )
    models = discover_models(layout, 'cls')
    assert len(models) == 1
    model = models[0]
    assert model.backend == 'dinov3'
    assert model.feature_dim == 768
    assert model.requires_detector
    assert not model.supports_video_all
    assert resolve_saved_model_path(str(head.resolve()), models) == str(manifest.resolve())


def test_healthy_installed_component_uses_only_declared_manifest(tmp_path, monkeypatch):
    layout = get_model_layout(tmp_path)
    root = layout.dinov3_root
    root.mkdir(parents=True)

    active_head = root / 'multi_prototype.pt'
    active_head.write_bytes(b'head')
    active_manifest = root / 'active-multi.neri.json'
    active_manifest.write_text(
        json.dumps({
            'backend': 'dinov3',
            'display_name': 'DINOv3 ViT-B/16 · Multi-prototype',
            'checkpoint': active_head.name,
            'architecture': 'dinov3_vitb16',
            'feature_dim': 768,
            'requires_detector': True,
            'supports_video_fast': True,
            'supports_video_all': False,
        }),
        encoding='utf-8',
    )

    decoy_head = root / 'decoy.pt'
    decoy_head.write_bytes(b'decoy')
    decoy_manifest = root / 'decoy.neri.json'
    decoy_manifest.write_text(
        json.dumps({
            'backend': 'dinov3',
            'display_name': 'must not be discovered',
            'checkpoint': decoy_head.name,
            'architecture': 'dinov3_vitb16',
            'feature_dim': 768,
        }),
        encoding='utf-8',
    )
    (root / 'install.json').write_text(
        json.dumps({
            'classifier': {
                'filename': active_head.name,
                'manifest': active_manifest.name,
            }
        }),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        catalog,
        'dinov3_component_status',
        lambda **kwargs: {'installed': True, 'healthy': True},
        raising=False,
    )

    models = discover_models(layout, 'cls')
    installed = [model for model in models if model.path.startswith(str(root.resolve()))]

    assert len(installed) == 1
    assert installed[0].path == str(active_manifest.resolve())
    assert installed[0].backend == 'dinov3'


def test_unhealthy_installed_component_is_hidden(tmp_path, monkeypatch):
    layout = get_model_layout(tmp_path)
    root = layout.dinov3_root
    root.mkdir(parents=True)
    head = root / 'head.pt'
    head.write_bytes(b'head')
    (root / 'head.neri.json').write_text(
        json.dumps({'backend': 'dinov3', 'checkpoint': head.name}),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        catalog,
        'dinov3_component_status',
        lambda **kwargs: {'installed': True, 'healthy': False},
        raising=False,
    )

    assert discover_models(layout, 'cls') == []
