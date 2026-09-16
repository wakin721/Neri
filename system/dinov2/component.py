"""Lifecycle helpers for Neri's optional offline DINOv2 runtime component."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from system.model_sync.layout import get_model_layout

from .checkpoint import (
    DINO_ARCHITECTURE_ID,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
    load_checkpoint,
)

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv2 ViT-B/14"
DINO_CLASSIFIER_FILENAME = "classifier.pt"
DINO_MODEL_MANIFEST_FILENAME = "classifier.neri.json"
DINO_LICENSE_FILENAME = "LICENSE_DINOv2.md"


@dataclass(frozen=True)
class DinoV2ComponentPaths:
    root: Path
    model_dir: Path
    model_config: Path
    model_weights: Path
    preprocessor_config: Path
    classifier: Path
    model_manifest: Path
    license: Path
    install_manifest: Path


def _component_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else get_model_layout().dinov2_root


def dinov2_component_paths(*, root: Path | None = None) -> DinoV2ComponentPaths:
    component_root = _component_root(root)
    model_dir = component_root / "model"
    return DinoV2ComponentPaths(
        root=component_root,
        model_dir=model_dir,
        model_config=model_dir / "config.json",
        model_weights=model_dir / "model.safetensors",
        preprocessor_config=model_dir / "preprocessor_config.json",
        classifier=component_root / DINO_CLASSIFIER_FILENAME,
        model_manifest=component_root / DINO_MODEL_MANIFEST_FILENAME,
        license=component_root / DINO_LICENSE_FILENAME,
        install_manifest=component_root / "install.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_inventory_path(raw) -> str:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or "\\" in raw
        or raw.startswith("/")
        or raw.endswith("/")
        or "//" in raw
    ):
        raise ValueError("DINOv2 install.json files 包含不安全路径。")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() == "install.json"
    ):
        raise ValueError("DINOv2 install.json files 包含不安全路径。")
    return pure.as_posix()


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 不是 JSON 对象。")
    return value


def _validate_file_inventory(
    paths: DinoV2ComponentPaths,
    install_manifest: dict,
) -> None:
    items = install_manifest.get("files")
    if not isinstance(items, list) or not items:
        raise ValueError("DINOv2 install.json files 必须是完整的非空文件清单。")
    declared: set[str] = set()
    root = paths.root.resolve()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("DINOv2 install.json files 条目格式无效。")
        relative = _safe_inventory_path(item.get("path"))
        if relative in declared:
            raise ValueError(f"DINOv2 install.json files 包含重复路径: {relative}")
        raw_sha = item.get("sha256")
        size = item.get("size")
        if (
            not isinstance(raw_sha, str)
            or len(raw_sha) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in raw_sha)
        ):
            raise ValueError(f"DINOv2 install.json files SHA-256 无效: {relative}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"DINOv2 install.json files size 无效: {relative}")
        candidate = paths.root.joinpath(*PurePosixPath(relative).parts)
        if candidate.is_symlink():
            raise ValueError(f"DINOv2 install.json files 不允许符号链接: {relative}")
        try:
            candidate.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"DINOv2 install.json files 包含不安全路径: {relative}"
            ) from exc
        if not candidate.is_file():
            raise ValueError(f"DINOv2 install.json files 缺少文件: {relative}")
        if candidate.stat().st_size != size:
            raise ValueError(f"DINOv2 install.json files size 不匹配: {relative}")
        if _sha256_file(candidate).lower() != raw_sha.lower():
            raise ValueError(f"DINOv2 install.json files SHA-256 不匹配: {relative}")
        declared.add(relative)

    actual = {
        path.relative_to(paths.root).as_posix()
        for path in paths.root.rglob("*")
        if path.is_file()
        and path != paths.install_manifest
        and not (
            "__pycache__" in path.parts
            and path.suffix.lower() in {".pyc", ".pyo"}
        )
    }
    if actual != declared:
        raise ValueError("DINOv2 install.json files 未覆盖完整清单。")


def _component_health(paths: DinoV2ComponentPaths) -> tuple[bool, str]:
    required = (
        paths.model_config,
        paths.model_weights,
        paths.preprocessor_config,
        paths.classifier,
        paths.model_manifest,
        paths.license,
        paths.install_manifest,
    )
    missing = [
        path.relative_to(paths.root).as_posix()
        for path in required
        if not path.is_file()
    ]
    if missing:
        return False, "DINOv2 安装不完整，缺少: " + "、".join(missing)
    try:
        install = _read_json_object(paths.install_manifest)
        _validate_file_inventory(paths, install)
        if install.get("architecture") != DINO_ARCHITECTURE_ID:
            raise ValueError("DINOv2 install.json architecture 不匹配。")

        manifest = _read_json_object(paths.model_manifest)
        if (
            manifest.get("schema_version") != 2
            or manifest.get("backend") != "dinov2"
            or manifest.get("architecture") != DINO_ARCHITECTURE_ID
            or manifest.get("feature_dim") != DINO_FEATURE_DIM
        ):
            raise ValueError("DINOv2 manifest 模型契约不匹配。")

        checkpoint = load_checkpoint(paths.classifier)
        if manifest.get("checkpoint") != paths.classifier.name:
            raise ValueError("DINOv2 manifest checkpoint 不匹配。")
        if (
            manifest.get("encoder_sha256") != checkpoint.encoder_sha256
            or manifest.get("preprocessing") != DINO_PREPROCESSING
            or manifest.get("event_aggregation") != DINO_EVENT_AGGREGATION
        ):
            raise ValueError("DINOv2 manifest 与 checkpoint 不匹配。")
        if _sha256_file(paths.model_weights).lower() != checkpoint.encoder_sha256.lower():
            raise ValueError(
                "DINOv2 model.safetensors SHA-256 与 checkpoint 不匹配。"
            )
        if Path(checkpoint.encoder_weights).name != paths.model_weights.name:
            raise ValueError("DINOv2 encoder_weights 不匹配。")
    except Exception as exc:
        return False, str(exc)
    return True, "DINOv2 ViT-B/14 已安装。"


def dinov2_component_status(*, root: Path | None = None) -> dict:
    paths = dinov2_component_paths(root=root)
    installed = paths.root.is_dir() and any(paths.root.iterdir())
    healthy, message = (
        _component_health(paths) if installed else (False, "DINOv2 未安装。")
    )
    result = {
        "installed": installed,
        "healthy": healthy,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "classifier_filename": paths.classifier.name,
        "message": message,
    }
    if healthy:
        checkpoint = load_checkpoint(paths.classifier)
        result.update(
            {
                "classifier_fingerprint": checkpoint.fingerprint,
                "classifier_head_type": checkpoint.head_type,
                "prototype_count": int(checkpoint.prototypes.shape[0]),
            }
        )
    return result


def remove_dinov2_component(*, root: Path | None = None) -> dict:
    paths = dinov2_component_paths(root=root)
    if paths.root.exists():
        shutil.rmtree(paths.root)
    return dinov2_component_status(root=paths.root)


def install_dinov2_component(
    *,
    root: Path | None = None,
    distribution_client=None,
    seed_dir: Path | None = None,
    on_progress=None,
) -> dict:
    paths = dinov2_component_paths(root=root)
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".dinov2-stage-", dir=paths.root.parent)
    )
    try:
        if seed_dir is not None:
            source = Path(seed_dir)
            for child in source.iterdir():
                target = staging / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
        else:
            if distribution_client is None:
                from .public_distribution import DinoV2DistributionClient

                distribution_client = DinoV2DistributionClient()
            distribution_client.download_tree("", staging, on_progress=on_progress)

        healthy, message = _component_health(
            dinov2_component_paths(root=staging)
        )
        if not healthy:
            raise RuntimeError(message)

        backup = paths.root.with_name(paths.root.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        if paths.root.exists():
            paths.root.replace(backup)
        staging.replace(paths.root)
        if backup.exists():
            shutil.rmtree(backup)
        return dinov2_component_status(root=paths.root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
