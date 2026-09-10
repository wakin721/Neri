from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from system.dinov3 import component


def _write_component(
    root: Path,
    *,
    selection_k: int,
    classifier_sha256: str,
) -> component.DinoV3ComponentPaths:
    paths = component.dinov3_component_paths(root=root)
    (paths.source_package / "hub").mkdir(parents=True)
    (paths.source_package / "models").mkdir(parents=True)
    (paths.source_package / "__init__.py").write_text("", encoding="utf-8")
    (paths.source_package / "hub" / "backbones.py").write_text("", encoding="utf-8")
    (paths.source_package / "models" / "vision_transformer.py").write_text("", encoding="utf-8")
    paths.backbone.parent.mkdir(parents=True, exist_ok=True)
    paths.backbone.write_bytes(b"backbone")
    paths.classifier.write_bytes(b"classifier")
    paths.license.write_text("license", encoding="utf-8")
    paths.model_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "checkpoint": component.DINO_CLASSIFIER_FILENAME,
                "architecture": component.DINO_ARCHITECTURE_ID,
                "feature_dim": 768,
                "selection_k": selection_k,
            }
        ),
        encoding="utf-8",
    )
    paths.install_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "component": "dinov3",
                "component_version": component.DINO_COMPONENT_VERSION,
                "architecture": component.DINO_ARCHITECTURE_ID,
                "source": {"commit": component.DINO_SOURCE_COMMIT},
                "backbone": {
                    "filename": f"dinov3-vitb16/{component.DINO_BACKBONE_FILENAME}",
                    "sha256": component.DINO_BACKBONE_SHA256,
                },
                "classifier": {
                    "filename": component.DINO_CLASSIFIER_FILENAME,
                    "sha256": classifier_sha256,
                    "head_type": "multi_prototype",
                    "selection_k": selection_k,
                    "manifest": component.DINO_MODEL_MANIFEST_FILENAME,
                },
                "files": [
                    {
                        "path": component.DINO_CLASSIFIER_FILENAME,
                        "sha256": classifier_sha256,
                        "size": paths.classifier.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paths


def _patch_static_integrity(monkeypatch, paths, classifier_sha256: str) -> None:
    monkeypatch.setattr(component, "_validate_file_inventory", lambda *_args: None)
    monkeypatch.setattr(
        component,
        "_sha256_file",
        lambda path: (
            component.DINO_BACKBONE_SHA256
            if Path(path) == paths.backbone
            else classifier_sha256
        ),
    )


def test_component_health_accepts_manifest_declared_selection_k_and_classifier_hash(
    tmp_path,
    monkeypatch,
):
    classifier_sha256 = "b" * 64
    paths = _write_component(
        tmp_path / "DINOv3",
        selection_k=5,
        classifier_sha256=classifier_sha256,
    )
    _patch_static_integrity(monkeypatch, paths, classifier_sha256)
    monkeypatch.setattr(
        component,
        "load_checkpoint",
        lambda _path: SimpleNamespace(
            head_type="multi_prototype",
            selection_k=5,
            fingerprint=classifier_sha256,
        ),
    )

    healthy, message = component._component_health(paths)

    assert healthy is True, message


def test_component_status_without_pytorch_reports_assets_without_runtime_error(
    tmp_path,
    monkeypatch,
):
    classifier_sha256 = component.DINO_CLASSIFIER_SHA256
    paths = _write_component(
        tmp_path / "DINOv3",
        selection_k=3,
        classifier_sha256=classifier_sha256,
    )
    _patch_static_integrity(monkeypatch, paths, classifier_sha256)

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "torch" else real_find_spec(name, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        component,
        "load_checkpoint",
        lambda _path: (_ for _ in ()).throw(
            RuntimeError("PyTorch is required to load or validate a DINOv3 checkpoint")
        ),
    )

    status = component.dinov3_component_status(root=paths.root)

    assert status["healthy"] is True
    assert status["classifier_filename"] == component.DINO_CLASSIFIER_FILENAME
    assert status["classifier_fingerprint"] == classifier_sha256
    assert status["classifier_head_type"] == "multi_prototype"
    assert status["selection_k"] == 3
    assert "PyTorch" in status["message"]
