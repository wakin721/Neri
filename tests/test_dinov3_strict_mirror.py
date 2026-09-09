from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from system.dinov3 import component
from system.dinov3.checkpoint import CheckpointValidationError


class MirrorCloud:
    def __init__(self, tree: Path) -> None:
        self.tree = tree
        self.download_tree_calls: list[str] = []

    def download_tree(self, relative_path: str, target: Path, on_progress=None) -> None:
        self.download_tree_calls.append(relative_path)
        source = self.tree / relative_path if relative_path else self.tree
        shutil.copytree(source, target, dirs_exist_ok=True)

    def download_file(self, relative_path: str, target: Path, on_progress=None) -> None:
        source = self.tree / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cloud_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "cloud"
    (tree / "source" / "dinov3" / "hub").mkdir(parents=True)
    (tree / "source" / "dinov3" / "models").mkdir(parents=True)
    (tree / "source" / "dinov3" / "__init__.py").write_text("# package\n", encoding="utf-8")
    (tree / "source" / "dinov3" / "hub" / "backbones.py").write_text(
        "def dinov3_vitb16(*, pretrained=True):\n"
        "    if pretrained is not False:\n"
        "        raise RuntimeError('pretrained must be false')\n"
        "    return object()\n",
        encoding="utf-8",
    )
    (tree / "source" / "dinov3" / "models" / "vision_transformer.py").write_text("# vit\n", encoding="utf-8")
    backbone_dir = tree / "dinov3-vitb16"
    backbone_dir.mkdir()
    (backbone_dir / component.DINO_BACKBONE_FILENAME).write_bytes(b"cloud-backbone")
    (tree / component.DINO_CLASSIFIER_FILENAME).write_bytes(b"cloud-classifier")
    (tree / component.DINO_MODEL_MANIFEST_FILENAME).write_bytes(
        b'{"schema_version":1,"backend":"dinov3","checkpoint":"multi_prototype.pt",'
        b'"architecture":"dinov3_vitb16","feature_dim":768}\n'
    )
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
                    "filename": component.DINO_CLASSIFIER_FILENAME,
                    "sha256": component.DINO_CLASSIFIER_SHA256,
                    "head_type": "multi_prototype",
                    "selection_k": 3,
                    "manifest": component.DINO_MODEL_MANIFEST_FILENAME,
                },
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    return tree


def _write_full_inventory(tree: Path) -> None:
    payload = json.loads((tree / "install.json").read_text(encoding="utf-8"))
    files = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.name == "install.json":
            continue
        relative = path.relative_to(tree).as_posix()
        if path.name == component.DINO_BACKBONE_FILENAME:
            sha256 = component.DINO_BACKBONE_SHA256
        elif path.name == component.DINO_CLASSIFIER_FILENAME:
            sha256 = component.DINO_CLASSIFIER_SHA256
        else:
            sha256 = _sha(path)
        files.append({"path": relative, "sha256": sha256, "size": path.stat().st_size})
    payload["files"] = files
    (tree / "install.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_component_hash(path: Path) -> str:
    path = Path(path)
    if path.name == component.DINO_BACKBONE_FILENAME:
        return component.DINO_BACKBONE_SHA256
    if path.name == component.DINO_CLASSIFIER_FILENAME:
        return component.DINO_CLASSIFIER_SHA256
    return _sha(path)


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
    original = (cloud_tree / component.DINO_MODEL_MANIFEST_FILENAME).read_bytes()
    target = tmp_path / "DINOv3"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / component.DINO_CLASSIFIER_FILENAME).write_bytes(b"poisoned-seed")
    (seed / component.DINO_MODEL_MANIFEST_FILENAME).write_bytes(b'{"backend":"dinov3"}\n')
    monkeypatch.setattr(component, "_component_health", lambda _paths: (True, "ok"))
    component.install_dinov3_component(root=target, cloud_client=MirrorCloud(cloud_tree), seed_dir=seed)
    assert (target / component.DINO_MODEL_MANIFEST_FILENAME).read_bytes() == original
    assert (target / component.DINO_CLASSIFIER_FILENAME).read_bytes() == b"cloud-classifier"


def test_post_activation_validation_failure_restores_previous_component(tmp_path, monkeypatch):
    cloud_tree = _cloud_tree(tmp_path)
    target = tmp_path / "DINOv3"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("healthy-old", encoding="utf-8")
    checks = iter([(True, "staged ok"), (False, "active smoke failed")])
    monkeypatch.setattr(component, "_component_health", lambda _paths: next(checks))
    with pytest.raises(RuntimeError, match="active smoke failed"):
        component.install_dinov3_component(root=target, cloud_client=MirrorCloud(cloud_tree))
    assert marker.read_text(encoding="utf-8") == "healthy-old"
    assert not (target / "remote-extra.txt").exists()


def test_health_rejects_declared_file_hash_mismatch(tmp_path, monkeypatch):
    tree = _cloud_tree(tmp_path)
    _write_full_inventory(tree)
    payload = json.loads((tree / "install.json").read_text(encoding="utf-8"))
    for item in payload["files"]:
        if item["path"] == "remote-extra.txt":
            item["sha256"] = "0" * 64
    (tree / "install.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(component, "_sha256_file", _fake_component_hash)
    status = component.dinov3_component_status(root=tree)
    assert status["healthy"] is False
    assert "remote-extra.txt" in status["message"]
    assert "SHA-256" in status["message"]


def test_health_rejects_duplicate_or_unsafe_inventory_path(tmp_path, monkeypatch):
    tree = _cloud_tree(tmp_path)
    _write_full_inventory(tree)
    payload = json.loads((tree / "install.json").read_text(encoding="utf-8"))
    payload["files"].append(dict(payload["files"][0]))
    payload["files"].append({"path": "../escape.txt", "sha256": "0" * 64, "size": 0})
    (tree / "install.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(component, "_sha256_file", _fake_component_hash)
    status = component.dinov3_component_status(root=tree)
    assert status["healthy"] is False
    assert "files" in status["message"].lower()


def test_integrity_failure_preserves_previous_component(tmp_path, monkeypatch):
    tree = _cloud_tree(tmp_path)
    _write_full_inventory(tree)
    payload = json.loads((tree / "install.json").read_text(encoding="utf-8"))
    payload["files"] = [item for item in payload["files"] if item["path"] != "remote-extra.txt"]
    (tree / "install.json").write_text(json.dumps(payload), encoding="utf-8")
    target = tmp_path / "installed"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("healthy-old", encoding="utf-8")
    monkeypatch.setattr(component, "_sha256_file", _fake_component_hash)
    with pytest.raises(RuntimeError, match="files"):
        component.install_dinov3_component(root=target, cloud_client=MirrorCloud(tree))
    assert marker.read_text(encoding="utf-8") == "healthy-old"


def test_health_rejects_invalid_multi_prototype_checkpoint(tmp_path, monkeypatch):
    tree = _cloud_tree(tmp_path)
    _write_full_inventory(tree)
    monkeypatch.setattr(component, "_sha256_file", _fake_component_hash)

    def invalid_checkpoint(_path):
        raise CheckpointValidationError("invalid multi-prototype checkpoint")

    monkeypatch.setattr(component, "load_checkpoint", invalid_checkpoint, raising=False)
    status = component.dinov3_component_status(root=tree)
    assert status["healthy"] is False
    assert "checkpoint" in status["message"].lower()


def test_source_smoke_failure_preserves_previous_component(tmp_path, monkeypatch):
    tree = _cloud_tree(tmp_path)
    (tree / "source" / "dinov3" / "hub" / "backbones.py").write_text(
        "def dinov3_vitb16(*, pretrained=True):\n"
        "    if pretrained is not False:\n"
        "        raise RuntimeError('pretrained must be false')\n"
        "    raise RuntimeError('source factory invoked')\n",
        encoding="utf-8",
    )
    _write_full_inventory(tree)
    target = tmp_path / "installed"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("healthy-old", encoding="utf-8")
    monkeypatch.setattr(component, "_sha256_file", _fake_component_hash)
    monkeypatch.setattr(
        component,
        "load_checkpoint",
        lambda _path: SimpleNamespace(head_type="multi_prototype", selection_k=3),
    )

    with pytest.raises(RuntimeError, match="source factory invoked"):
        component.install_dinov3_component(root=target, cloud_client=MirrorCloud(tree))

    assert marker.read_text(encoding="utf-8") == "healthy-old"
    assert not (target / "remote-extra.txt").exists()
