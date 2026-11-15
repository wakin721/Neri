"""Lifecycle helpers for Neri's offline DINOv3 (Neri1.2preview) component."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from system.model_sync.layout import get_model_layout

from .encoder import sha256
from .runtime import read_manifest, resolve_manifest_asset
from .experiment import PREVIEW_DISPLAY_NAME, PRODUCTION_RECIPE, head_type

COMPONENT_VERSION = 1
DINO_COMPONENT_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_COMPONENT_DISPLAY_NAME = PREVIEW_DISPLAY_NAME
_MANIFEST_SUFFIX = ".neri.json"


@dataclass(frozen=True)
class DinoV3ComponentPaths:
    root: Path
    manifest: Path | None


def _component_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return get_model_layout().dinov3_root.resolve()


def _component_paths(root: Path | None = None) -> DinoV3ComponentPaths:
    component_root = _component_root(root)
    manifests = sorted(
        component_root.glob(f"*{_MANIFEST_SUFFIX}"),
        key=lambda path: path.name.casefold(),
    )
    return DinoV3ComponentPaths(
        root=component_root,
        manifest=manifests[0] if len(manifests) == 1 else None,
    )


def _select_valid_manifest(root: Path) -> tuple[Path, dict]:
    candidates = sorted(
        root.glob(f"*{_MANIFEST_SUFFIX}"),
        key=lambda path: path.name.casefold(),
    )
    valid: list[tuple[Path, dict]] = []
    legacy = False
    for candidate in candidates:
        try:
            valid.append((candidate, read_manifest(candidate)))
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                previous = json.loads(candidate.read_text(encoding="utf-8"))
                legacy |= (isinstance(previous, dict) and previous.get("backend") == "dinov3"
                           and previous.get("schema_version") == 1
                           and previous.get("head_type") is None
                           and previous.get("event_aggregation") == "mean_l2_normalized_crop_embeddings")
            except (OSError, ValueError):
                pass
            continue
    if not valid:
        if legacy:
            raise ValueError("检测到旧版 DINOv3 多原型组件，需迁移为当前联合模型格式；独立安装的用户模型不受影响。")
        raise ValueError("未找到有效的 Neri1.2preview 模型清单。")
    if len(valid) > 1:
        raise ValueError("DINOv3 组件中包含多个模型清单。")
    return valid[0]


def _model_manifest_summary(checkpoint: Path) -> tuple[int, int]:
    with zipfile.ZipFile(checkpoint, mode="r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("模型文件缺少 manifest.json。")
        payload = json.loads(archive.read("manifest.json").decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("模型 manifest 不是 JSON 对象。")
    if (
        payload.get("backend") != "dinov3"
        or payload.get("head_type") != head_type(payload.get("inference_recipe", PRODUCTION_RECIPE))
        or payload.get("schema_version") != 1
        or payload.get("is_calibrated") is not True
        or payload.get("dimension") != 768
    ):
        raise ValueError("模型文件契约不匹配。")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("模型文件没有已知物种。")
    return len(classes), int(payload["dimension"])


def _component_health(
    root: Path,
) -> tuple[bool, str, dict | None, int | None, int | None]:
    if not root.is_dir() or not any(root.iterdir()):
        return False, "Neri1.2preview 未安装。", None, None, None
    try:
        manifest_path, payload = _select_valid_manifest(root)
        checkpoint = resolve_manifest_asset(manifest_path, payload["checkpoint"])
        weights = resolve_manifest_asset(manifest_path, payload["encoder_weights"])
        code = resolve_manifest_asset(manifest_path, payload["encoder_code"])
        if sha256(checkpoint) != payload["checkpoint_sha256"]:
            raise ValueError("模型文件 SHA-256 不匹配。")
        if sha256(weights) != payload["encoder_sha256"]:
            raise ValueError("编码器权重 SHA-256 不匹配。")
        if "calibration" in payload:
            calibration = resolve_manifest_asset(manifest_path, payload["calibration"])
            if sha256(calibration) != payload["calibration_sha256"]:
                raise ValueError("校准集 SHA-256 不匹配。")
        if not code.is_dir():
            raise ValueError("编码器源码目录不存在。")
        if not (code / "LICENSE.md").is_file():
            raise ValueError("编码器源码缺少 LICENSE.md。")
        if not (code / "dinov3" / "hub" / "backbones.py").is_file():
            raise ValueError("编码器源码不完整。")
        class_count, feature_dim = _model_manifest_summary(checkpoint)
    except Exception as exc:  # noqa: BLE001 - status boundary
        return False, str(exc), None, None, None
    return True, f"{payload.get('display_name') or DINO_COMPONENT_DISPLAY_NAME} 已安装。", payload, class_count, feature_dim


def dinov3_component_status(*, root: Path | None = None) -> dict:
    component_root = _component_root(root)
    installed = component_root.is_dir() and any(component_root.iterdir())
    healthy, message, payload, class_count, feature_dim = _component_health(
        component_root
    )
    manifest_name = None
    if healthy:
        try:
            manifest_name = _select_valid_manifest(component_root)[0].name
        except (OSError, ValueError):
            manifest_name = None
    result = {
        "installed": installed,
        "healthy": healthy,
        "architecture": DINO_COMPONENT_ARCHITECTURE,
        "component_version": COMPONENT_VERSION,
        "display_name": (
            str(payload.get("display_name") or DINO_COMPONENT_DISPLAY_NAME)
            if payload
            else DINO_COMPONENT_DISPLAY_NAME
        ),
        "class_count": class_count,
        "feature_dim": feature_dim,
        "manifest": manifest_name,
        "message": message,
    }
    return result


def installed_dinov3_manifests(*, root: Path | None = None) -> list[Path]:
    component_root = _component_root(root)
    result: list[Path] = []
    for candidate in sorted(
        component_root.glob(f"*{_MANIFEST_SUFFIX}"),
        key=lambda path: path.name.casefold(),
    ):
        try:
            read_manifest(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        result.append(candidate)
    return result


def remove_dinov3_component(*, root: Path | None = None) -> dict:
    component_root = _component_root(root)
    if component_root.exists():
        shutil.rmtree(component_root)
    return dinov3_component_status(root=component_root)


def install_dinov3_component(
    *,
    root: Path | None = None,
    distribution_client=None,
    seed_dir: Path | None = None,
    on_progress=None,
    on_event=None,
) -> dict:
    component_root = _component_root(root)
    component_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".dinov3-stage-", dir=component_root.parent)
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
                from .distribution import DinoV3DistributionClient

                distribution_client = DinoV3DistributionClient(
                    cache_dir=component_root.parent / '.dinov3-download-cache',
                    on_event=on_event)
            distribution_client.download_tree("", staging, on_progress=on_progress)

        healthy, message, _payload, _classes, _dim = _component_health(staging)
        if not healthy:
            raise RuntimeError(message)

        backup = component_root.with_name(component_root.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        if component_root.exists():
            component_root.replace(backup)
        try:
            staging.replace(component_root)
        except Exception:
            if backup.exists() and not component_root.exists():
                backup.replace(component_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return dinov3_component_status(root=component_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
