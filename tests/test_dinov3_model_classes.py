from __future__ import annotations

import json
from pathlib import Path

from system.backend.services import list_model_classes


def test_dinov3_manifest_model_classes_are_loaded_from_checkpoint(monkeypatch, tmp_path):
    model_root = tmp_path / "DINOv3"
    model_root.mkdir(parents=True)
    (model_root / "multi_prototype.neri.json").write_text(
        json.dumps(
            {
                "backend": "dinov3",
                "checkpoint": "multi_prototype.pt",
                "architecture": "dinov3_vitb16",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "system.backend.services._resolve_model_path",
        lambda _path: model_root / "multi_prototype.neri.json",
    )
    monkeypatch.setattr(
        "system.backend.services._load_dinov3_model_classes",
        lambda _manifest: [
            {"id": 0, "name": "TestSpecies", "display_name": "测试物种"}
        ],
        raising=False,
    )

    classes = list_model_classes(str(model_root / "multi_prototype.neri.json"))

    assert len(classes) == 1
    assert classes[0].name == "TestSpecies"
