from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tests.dinov2_fixtures import make_dinov2_payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_component(root: Path):
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"dinov2-weights")
    payload = make_dinov2_payload()
    payload["encoder_weights"] = "model/model.safetensors"
    payload["encoder_sha256"] = _sha(weights)
    classifier = root / "classifier.pt"
    torch.save(payload, classifier)
    manifest = root / "classifier.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "display_name": "DINOv2 test",
                "checkpoint": classifier.name,
                "architecture": "dinov2_vitb14",
                "feature_dim": 768,
                "encoder_sha256": payload["encoder_sha256"],
                "preprocessing": payload["preprocessing"],
                "event_aggregation": payload["event_aggregation"],
            }
        ),
        encoding="utf-8",
    )
    (root / "LICENSE_DINOv2.md").write_text("license", encoding="utf-8")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "install.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    (root / "install.json").write_text(
        json.dumps(
            {
                "component_version": 1,
                "architecture": "dinov2_vitb14",
                "classifier": {
                    "filename": classifier.name,
                    "manifest": manifest.name,
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_component_health_accepts_complete_offline_bundle(tmp_path):
    from system.dinov2.component import dinov2_component_status

    root = tmp_path / "DINOv2"
    root.mkdir()
    _write_component(root)
    status = dinov2_component_status(root=root)

    assert status["installed"] is True
    assert status["healthy"] is True
    assert status["architecture"] == "DINOv2 ViT-B/14"
    assert status["classifier_head_type"] == "multi_prototype"


def test_component_health_rejects_undeclared_extra_file(tmp_path):
    from system.dinov2.component import dinov2_component_status

    root = tmp_path / "DINOv2"
    root.mkdir()
    _write_component(root)
    (root / "rogue.bin").write_bytes(b"x")

    status = dinov2_component_status(root=root)

    assert status["healthy"] is False
    assert "完整" in status["message"] or "清单" in status["message"]


def test_component_health_rejects_model_weight_hash_mismatch(tmp_path):
    from system.dinov2.component import dinov2_component_status

    root = tmp_path / "DINOv2"
    root.mkdir()
    _write_component(root)
    (root / "model" / "model.safetensors").write_bytes(
        b"x" * len(b"dinov2-weights")
    )

    status = dinov2_component_status(root=root)

    assert status["healthy"] is False
    assert "SHA-256" in status["message"]
