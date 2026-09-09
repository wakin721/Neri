from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from system.dinov3 import component


class MirrorCloud:
    def __init__(self, tree: Path) -> None:
        self.tree = tree
        self.download_tree_calls: list[str] = []

    def download_tree(self, relative_path: str, target: Path, on_progress=None) -> None:
        self.download_tree_calls.append(relative_path)
        source = self.tree / relative_path if relative_path else self.tree
        shutil.copytree(source, target, dirs_exist_ok=True)

    # Kept so the RED test exercises the current partial installer rather than
    # failing merely because a method is absent on the fake.
    def download_file(self, relative_path: str, target: Path, on_progress=None) -> None:
        source = self.tree / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _cloud_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "cloud"
    (tree / "source" / "dinov3" / "hub").mkdir(parents=True)
    (tree / "source" / "dinov3" / "models").mkdir(parents=True)
    (tree / "source" / "dinov3" / "__init__.py").write_text("# package\n", encoding="utf-8")
    (tree / "source" / "dinov3" / "hub" / "backbones.py").write_text("# hub\n", encoding="utf-8")
    (tree / "source" / "dinov3" / "models" / "vision_transformer.py").write_text(
        "# vit\n",
        encoding="utf-8",
    )
    backbone_dir = tree / "dinov3-vitb16"
    backbone_dir.mkdir()
    (backbone_dir / component.DINO_BACKBONE_FILENAME).write_bytes(b"cloud-backbone")
    (tree / "multi_prototype.pt").write_bytes(b"cloud-classifier")
    manifest_bytes = (
        b'{"schema_version":1,"backend":"dinov3","checkpoint":"multi_prototype.pt",'
        b'"architecture":"dinov3_vitb16","feature_dim":768}\n'
    )
    (tree / "multi_prototype.neri.json").write_bytes(manifest_bytes)
    (tree / "LICENSE_DINOv3.md").write_text("cloud-license\n", encoding="utf-8")
    (tree / "remote-extra.txt").write_text("must mirror\n", encoding="utf-8")
    (tree / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "component": "dinov3",
                "component_version": 2,
                "architecture": "dinov3_vitb16",
                "source": {"commit": component.DINO_SOURCE_COMMIT},
                "backbone": {
                    "filename": f"dinov3-vitb16/{component.DINO_BACKBONE_FILENAME}",
                    "sha256": component.DINO_BACKBONE_SHA256,
                },
                "classifier": {
                    "filename": "multi_prototype.pt",
                    "sha256": "4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43",
                    "head_type": "multi_prototype",
                    "selection_k": 3,
                    "manifest": "multi_prototype.neri.json",
                },
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    return tree


def test_install_downloads_authoritative_root_as_complete_mirror(tmp_path, monkeypatch):
    cloud_tree = _cloud_tree(tmp_path)
    cloud = MirrorCloud(cloud_tree)
    target = tmp_path / "res" / "model" / "DINOv3"
    target.mkdir(parents=True)
    (target / "local-only.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(component, "_component_health", lambda _paths: (True, "ok"))

    component.install_dinov3_component(root=target, cloud_client=cloud)

    assert cloud.download_tree_calls == [""]
    assert (target / "remote-extra.txt").read_text(encoding="utf-8") == "must mirror\n"
    assert not (target / "local-only.txt").exists()


def test_install_preserves_cloud_manifest_bytes_without_normalization(tmp_path, monkeypatch):
    cloud_tree = _cloud_tree(tmp_path)
    original = (cloud_tree / "multi_prototype.neri.json").read_bytes()
    target = tmp_path / "DINOv3"

    # The current installer prefers a seed; a poisoned seed makes that behavior
    # observable. Strict mirroring must ignore it entirely.
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / component.DINO_CLASSIFIER_FILENAME).write_bytes(b"poisoned-seed")
    (seed / component.DINO_MODEL_MANIFEST_FILENAME).write_bytes(b'{"backend":"dinov3"}\n')
    monkeypatch.setattr(component, "_component_health", lambda _paths: (True, "ok"))

    component.install_dinov3_component(
        root=target,
        cloud_client=MirrorCloud(cloud_tree),
        seed_dir=seed,
    )

    assert (target / "multi_prototype.neri.json").read_bytes() == original
    assert (target / "multi_prototype.pt").read_bytes() == b"cloud-classifier"


def test_post_activation_validation_failure_restores_previous_component(tmp_path, monkeypatch):
    cloud_tree = _cloud_tree(tmp_path)
    target = tmp_path / "DINOv3"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("healthy-old", encoding="utf-8")
    checks = iter([(True, "staged ok"), (False, "active smoke failed")])
    monkeypatch.setattr(component, "_component_health", lambda _paths: next(checks))

    with pytest.raises(RuntimeError, match="active smoke failed"):
        component.install_dinov3_component(
            root=target,
            cloud_client=MirrorCloud(cloud_tree),
        )

    assert marker.read_text(encoding="utf-8") == "healthy-old"
    assert not (target / "remote-extra.txt").exists()
