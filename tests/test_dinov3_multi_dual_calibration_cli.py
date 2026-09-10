from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def _unit(index: int, dim: int = 768) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def test_calibration_cli_writes_derived_manifest_without_touching_original(tmp_path):
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "head_type": "multi_prototype",
            "feature_dim": 768,
            "classes": ["A", "B"],
            "feature_center": torch.zeros(768),
            "prototypes": torch.stack(
                [
                    torch.from_numpy(_unit(0) * 0.8),
                    torch.from_numpy(_unit(1) * 0.8),
                ]
            ),
            "data_fingerprint": "synthetic",
        },
        checkpoint,
    )
    manifest = tmp_path / "head.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "dinov3",
                "checkpoint": "head.pt",
                "feature_dim": 768,
            }
        ),
        encoding="utf-8",
    )

    rows = []
    features = []
    for index, class_index in enumerate([0, 0, 0, 0, 1, 1, 1, 1]):
        rows.append(
            {
                "image_id": f"v{index}",
                "species": "A" if class_index == 0 else "B",
                "split": "val",
            }
        )
        features.append(_unit(class_index))
    proxy = [
        _unit(0) * 0.7 + _unit(2) * 0.714,
        _unit(1) * 0.7 + _unit(2) * 0.714,
    ]
    proxy = [value / np.linalg.norm(value) for value in proxy]
    for index, value in enumerate(proxy):
        rows.append(
            {
                "image_id": f"u{index}",
                "species": f"U{index}",
                "split": "proxy_unknown_val",
            }
        )
        features.append(value.astype(np.float32))

    (tmp_path / "splits.json").write_text(json.dumps(rows), encoding="utf-8")
    np.savez(
        tmp_path / "image_features.npz",
        features=np.stack(features),
        image_ids=np.asarray([row["image_id"] for row in rows]),
    )

    output = tmp_path / "head.multi-dual.neri.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_dinov3_multi_dual.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--source-run",
            str(tmp_path),
            "--out-manifest",
            str(output),
            "--max-known-frr",
            "0.05",
            "--grid-size",
            "32",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    original = json.loads(manifest.read_text(encoding="utf-8"))
    derived = json.loads(output.read_text(encoding="utf-8"))
    assert "rejection" not in original
    assert derived["rejection"]["mode"] == "multi_dual"
    assert "cosine_threshold" in derived["rejection"]
    assert "squared_distance_threshold" in derived["rejection"]
    assert derived["rejection_calibration"]["known_validation_images"] == 8
    assert derived["rejection_calibration"]["proxy_unknown_images"] == 2
    assert derived["rejection_calibration"]["test_isolation"].startswith("test and unknown_test")
