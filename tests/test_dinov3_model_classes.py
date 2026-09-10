from __future__ import annotations

import json

import torch

from system.backend.services import list_model_classes
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def test_dinov3_manifest_model_classes_are_loaded_from_checkpoint_without_yolo(
    monkeypatch,
    tmp_path,
):
    model_root = tmp_path / "DINOv3"
    model_root.mkdir(parents=True)
    checkpoint_path = model_root / "multi_prototype.pt"
    torch.save(
        make_multi_prototype_payload(classes=("TestSpecies", "AnotherSpecies")),
        checkpoint_path,
    )
    manifest_path = model_root / "multi_prototype.neri.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "checkpoint": checkpoint_path.name,
                "architecture": "dinov3_vitb16",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "system.backend.services._resolve_model_path",
        lambda _path: manifest_path,
    )

    def unexpected_yolo_call(*_args, **_kwargs):
        raise AssertionError("DINOv3 class loading must not use YOLO")

    monkeypatch.setattr(
        "system.backend.services.missing_yolo_dependencies",
        unexpected_yolo_call,
    )
    monkeypatch.setattr(
        "system.backend.services._load_detector",
        unexpected_yolo_call,
    )

    classes = list_model_classes(str(manifest_path))

    assert [(item.id, item.name, item.display_name) for item in classes] == [
        (0, "TestSpecies", "TestSpecies"),
        (1, "AnotherSpecies", "AnotherSpecies"),
    ]
