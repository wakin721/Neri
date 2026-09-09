from __future__ import annotations

import json

import system.model_sync.catalog as catalog
from system.model_sync.catalog import discover_models
from system.model_sync.layout import get_model_layout


def test_healthy_dinov3_install_is_visible_in_classification_catalog(tmp_path, monkeypatch):
    layout = get_model_layout(tmp_path)
    root = layout.dinov3_root
    root.mkdir(parents=True)
    checkpoint = root / "multi_prototype.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = root / "multi_prototype.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "backend": "dinov3",
                "display_name": "DINOv3 ViT-B/16 Multi-prototype",
                "checkpoint": checkpoint.name,
                "architecture": "dinov3_vitb16",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    (root / "install.json").write_text(
        json.dumps({"classifier": {"manifest": manifest.name}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        catalog,
        "dinov3_component_status",
        lambda **kwargs: {"installed": True, "healthy": True},
    )

    models = discover_models(layout, "cls")

    assert any(model.backend == "dinov3" for model in models)
