from __future__ import annotations

import json


def test_layout_uses_dinov2_component_root(tmp_path):
    from system.model_sync.layout import get_model_layout

    layout = get_model_layout(tmp_path)

    assert layout.dinov2_root == (tmp_path.resolve() / "model" / "DINOv2")
    assert not hasattr(layout, "dinov3_root")


def test_manifest_exposes_dinov2_capabilities(tmp_path):
    from system.model_sync.catalog import discover_models, resolve_saved_model_path
    from system.model_sync.layout import get_model_layout

    layout = get_model_layout(tmp_path)
    head = layout.cls_user / "head.pt"
    head.write_bytes(b"x")
    manifest = layout.cls_user / "head.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "display_name": "DINOv2 reviewed",
                "checkpoint": "head.pt",
                "architecture": "dinov2_vitb14",
                "feature_dim": 768,
                "requires_detector": True,
                "supports_video_fast": True,
                "supports_video_all": False,
            }
        ),
        encoding="utf-8",
    )

    models = discover_models(layout, "cls")

    assert len(models) == 1
    model = models[0]
    assert model.backend == "dinov2"
    assert model.architecture == "dinov2_vitb14"
    assert model.feature_dim == 768
    assert resolve_saved_model_path(str(head.resolve()), models) == str(
        manifest.resolve()
    )


def test_dinov3_manifest_is_not_selectable(tmp_path):
    from system.model_sync.catalog import discover_models
    from system.model_sync.layout import get_model_layout

    layout = get_model_layout(tmp_path)
    head = layout.cls_user / "old.pt"
    head.write_bytes(b"x")
    (layout.cls_user / "old.neri.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "checkpoint": "old.pt",
                "architecture": "dinov3_vitb16",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )

    models = discover_models(layout, "cls")

    assert models == []


def test_healthy_installed_component_uses_only_declared_manifest(
    tmp_path,
    monkeypatch,
):
    import system.model_sync.catalog as catalog
    from system.model_sync.catalog import discover_models
    from system.model_sync.layout import get_model_layout

    layout = get_model_layout(tmp_path)
    root = layout.dinov2_root
    root.mkdir(parents=True)
    active = root / "classifier.pt"
    active.write_bytes(b"head")
    manifest = root / "active.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "display_name": "DINOv2 ViT-B/14",
                "checkpoint": active.name,
                "architecture": "dinov2_vitb14",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    decoy = root / "decoy.pt"
    decoy.write_bytes(b"decoy")
    (root / "decoy.neri.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "checkpoint": decoy.name,
                "architecture": "dinov2_vitb14",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    (root / "install.json").write_text(
        json.dumps(
            {
                "classifier": {
                    "filename": active.name,
                    "manifest": manifest.name,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        catalog,
        "dinov2_component_status",
        lambda **_kwargs: {"installed": True, "healthy": True},
    )

    models = discover_models(layout, "cls")
    installed = [
        model for model in models if model.path.startswith(str(root.resolve()))
    ]

    assert len(installed) == 1
    assert installed[0].path == str(manifest.resolve())
    assert installed[0].backend == "dinov2"
