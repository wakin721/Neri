"""Lifecycle helpers for Neri's offline DINOv3 joint-model component."""
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
from .runtime_selection import (
    RUNTIME_CHOICES,
    expected_encoder_backend,
    resolve_runtime_choice,
    runtime_choice_from_encoder_backend,
    validate_resolved_runtime,
    validate_runtime_choice,
)

COMPONENT_VERSION = 1
DINO_COMPONENT_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_COMPONENT_DISPLAY_NAME = PREVIEW_DISPLAY_NAME
_MANIFEST_SUFFIX = ".neri.json"
_RUNTIME_META_FILE = ".dinov3_runtime.json"


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
        raise ValueError("未找到有效的联合模型清单。")
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
        return False, "联合模型组件未安装。", None, None, None
    try:
        manifest_path, payload = _select_valid_manifest(root)
        checkpoint = resolve_manifest_asset(manifest_path, payload["checkpoint"])
        weights = resolve_manifest_asset(manifest_path, payload["encoder_weights"])
        if sha256(checkpoint) != payload["checkpoint_sha256"]:
            raise ValueError("模型文件 SHA-256 不匹配。")
        encoder_backend = payload.get("encoder_backend", "pytorch")
        if encoder_backend == "litert":
            if not weights.is_file() or weights.suffix.lower() not in {".tflite", ".litert"}:
                raise ValueError("LiteRT 编码器权重文件无效。")
            if sha256(weights) != payload["encoder_litert_sha256"]:
                raise ValueError("LiteRT 编码器权重 SHA-256 不匹配。")
        else:
            if not weights.is_file() or weights.suffix.lower() not in {".pth", ".pt"}:
                raise ValueError("编码器权重文件无效。")
            if sha256(weights) != payload["encoder_sha256"]:
                raise ValueError("编码器权重 SHA-256 不匹配。")
            code = resolve_manifest_asset(manifest_path, payload["encoder_code"])
            if not code.is_dir():
                raise ValueError("编码器源码目录不存在。")
            if not (code / "LICENSE.md").is_file():
                raise ValueError("编码器源码缺少 LICENSE.md。")
            if not (code / "dinov3" / "hub" / "backbones.py").is_file():
                raise ValueError("编码器源码不完整。")
        if "calibration" in payload:
            calibration = resolve_manifest_asset(manifest_path, payload["calibration"])
            if sha256(calibration) != payload["calibration_sha256"]:
                raise ValueError("校准集 SHA-256 不匹配。")
        class_count, feature_dim = _model_manifest_summary(checkpoint)
    except Exception as exc:  # noqa: BLE001 - status boundary
        return False, str(exc), None, None, None
    return True, f"{payload.get('display_name') or DINO_COMPONENT_DISPLAY_NAME} 已安装。", payload, class_count, feature_dim


def _read_runtime_metadata(root: Path, resolved_runtime: str | None) -> str | None:
    if resolved_runtime is None:
        return None
    meta_path = root / _RUNTIME_META_FILE
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("resolved_runtime_choice") != resolved_runtime:
        return None
    choice = meta.get("runtime_choice")
    return choice if isinstance(choice, str) and choice in RUNTIME_CHOICES else None


def _write_runtime_metadata(root: Path, runtime_choice: str, resolved_runtime: str) -> None:
    meta_path = root / _RUNTIME_META_FILE
    meta_path.write_text(
        json.dumps(
            {
                "runtime_choice": runtime_choice,
                "resolved_runtime_choice": resolved_runtime,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
    encoder_backend = (
        str(payload.get("encoder_backend", "pytorch")) if healthy and payload else None
    )
    resolved_runtime = runtime_choice_from_encoder_backend(encoder_backend)
    recorded_choice = _read_runtime_metadata(component_root, resolved_runtime)
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
        "encoder_backend": encoder_backend,
        "runtime_choice": recorded_choice,
        "resolved_runtime_choice": resolved_runtime,
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


def _emit_event(on_event, **event) -> None:
    """Forward a lifecycle event without letting diagnostics fail an install."""
    if on_event is not None:
        try:
            on_event(event)
        except Exception:
            pass


def _cleanup_backup(backup: Path, *, on_event=None) -> bool:
    """Delete a backup tree or defer cleanup without force-removing locked files."""
    try:
        shutil.rmtree(backup)
        return True
    except OSError as exc:
        _emit_event(
            on_event,
            phase="cleanup",
            state="deferred",
            action="deferred_cleanup",
            message=f"deferred cleanup: {backup.name}",
            path=str(backup),
            exception_type=type(exc).__name__,
            reason=str(exc),
        )
        return False


def _prepare_backup_path(component_root: Path, *, on_event=None) -> Path:
    """Return a free backup sibling path without overwriting locked backups."""
    default_backup = component_root.with_name(component_root.name + ".old")
    if not default_backup.exists() and not default_backup.is_symlink():
        return default_backup
    if _cleanup_backup(default_backup, on_event=on_event):
        return default_backup
    index = 1
    while True:
        candidate = component_root.with_name(f"{component_root.name}.old.{index}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        index += 1


def install_dinov3_component(
    *,
    root: Path | None = None,
    distribution_client=None,
    seed_dir: Path | None = None,
    runtime_choice: str | None = None,
    resolved_runtime_choice: str | None = None,
    on_progress=None,
    on_event=None,
) -> dict:
    validated_choice = (
        validate_runtime_choice(runtime_choice) if runtime_choice is not None else None
    )
    resolved_runtime = (
        validate_resolved_runtime(resolved_runtime_choice)
        if resolved_runtime_choice is not None
        else None
    )
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
            if distribution_client is None and validated_choice is None:
                validated_choice = "auto"
            if validated_choice is not None and resolved_runtime is None:
                resolved_runtime = resolve_runtime_choice(validated_choice)
            if distribution_client is None:
                from .distribution import DinoV3DistributionClient

                distribution_client = DinoV3DistributionClient(
                    cache_dir=component_root.parent / '.dinov3-download-cache',
                    on_event=on_event,
                    runtime=resolved_runtime)
            if resolved_runtime is None:
                distribution_client.download_tree("", staging, on_progress=on_progress)
            else:
                distribution_client.download_tree(
                    "",
                    staging,
                    on_progress=on_progress,
                    runtime=resolved_runtime,
                )

        healthy, message, payload, _classes, _dim = _component_health(staging)
        if not healthy:
            raise RuntimeError(message)
        actual_backend = str((payload or {}).get("encoder_backend", "pytorch"))
        if seed_dir is None and resolved_runtime is not None:
            expected_backend = expected_encoder_backend(resolved_runtime)
            if actual_backend != expected_backend:
                raise RuntimeError(
                    f"下载的 DINOv3 组件 ({actual_backend}) 与目标运行时 ({resolved_runtime}) 不匹配。"
                )
        if validated_choice is not None:
            actual_resolved = runtime_choice_from_encoder_backend(actual_backend)
            if actual_resolved is not None:
                _write_runtime_metadata(staging, validated_choice, actual_resolved)

        backup: Path | None = None
        if component_root.exists():
            backup = _prepare_backup_path(component_root, on_event=on_event)
            component_root.replace(backup)
        try:
            staging.replace(component_root)
        except Exception:
            if backup is not None and backup.exists() and not component_root.exists():
                backup.replace(component_root)
            raise
        if backup is not None and backup.exists():
            _cleanup_backup(backup, on_event=on_event)
        return dinov3_component_status(root=component_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
