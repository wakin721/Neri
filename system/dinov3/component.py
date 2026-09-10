"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from system.model_sync.layout import get_model_layout

from .checkpoint import CheckpointValidationError, load_checkpoint

DINO_COMPONENT_VERSION = 2
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_ARCHITECTURE_ID = "dinov3_vitb16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_BACKBONE_FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINO_BACKBONE_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
DINO_CLASSIFIER_FILENAME = "multi_prototype.pt"
DINO_CLASSIFIER_SHA256 = "4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43"
DINO_MODEL_MANIFEST_FILENAME = "multi_prototype.neri.json"
DINO_CLOUD_ROOT = "/Neri_Data/Model/DINOv3"
DINO_DEFAULT_CLOUD_ORIGIN = "https://cloud.myneri.top"
DINO_CLOUD_ORIGIN_ENV = "NERI_DINOV3_CLOUD_ORIGIN"

ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class DinoV3ComponentPaths:
    root: Path
    source_root: Path
    source_package: Path
    backbone_dir: Path
    backbone: Path
    classifier: Path
    model_manifest: Path
    license: Path
    install_manifest: Path


class DinoV3CloudError(RuntimeError):
    """Raised when NeriCloud cannot provide a required DINOv3 asset."""


class DinoV3CloudClient:
    """Small OpenList client scoped to Neri's DINOv3 component directory."""

    def __init__(self, origin: str | None = None, *, opener=None):
        configured = origin or os.environ.get(DINO_CLOUD_ORIGIN_ENV) or DINO_DEFAULT_CLOUD_ORIGIN
        parsed = urllib.parse.urlsplit(configured.strip())
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("DINOv3 NeriCloud 地址无效。")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("DINOv3 NeriCloud 必须使用 HTTPS。")
        self.origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
        self.opener = opener or urllib.request.urlopen

    @staticmethod
    def _relative_parts(relative_path: str) -> tuple[str, ...]:
        raw = str(relative_path or "").replace("\\", "/").strip("/")
        if not raw:
            return ()
        parts = PurePosixPath(raw).parts
        if any(part in {"", ".", ".."} or "/" in part or "\\" in part for part in parts):
            raise ValueError("DINOv3 NeriCloud 路径无效。")
        return tuple(parts)

    def mount_path(self, relative_path: str = "") -> str:
        parts = self._relative_parts(relative_path)
        if not parts:
            return DINO_CLOUD_ROOT
        return DINO_CLOUD_ROOT + "/" + "/".join(parts)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.origin + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Neri DINOv3 component installer"},
            method="POST",
        )
        try:
            with self.opener(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except Exception as exc:
            raise DinoV3CloudError(f"NeriCloud 请求失败: {endpoint}") from exc
        if len(raw) > 8 * 1024 * 1024:
            raise DinoV3CloudError("NeriCloud API 响应过大。")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DinoV3CloudError("NeriCloud 返回了无效 JSON。") from exc
        if not isinstance(value, dict):
            raise DinoV3CloudError("NeriCloud API 响应格式无效。")
        code = value.get("code")
        if code not in (None, 0, 200):
            raise DinoV3CloudError(str(value.get("message") or value.get("msg") or f"NeriCloud error {code}"))
        data = value.get("data")
        if not isinstance(data, dict):
            raise DinoV3CloudError("NeriCloud API 缺少 data。")
        return data

    def list_directory(self, relative_path: str) -> list[dict[str, Any]]:
        data = self._post_json(
            "/api/fs/list",
            {"path": self.mount_path(relative_path), "password": "", "page": 1, "per_page": 0, "refresh": False},
        )
        content = data.get("content")
        if content is None:
            return []
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise DinoV3CloudError("NeriCloud 目录列表格式无效。")
        return content

    def _file_link(self, relative_path: str) -> tuple[str, dict[str, str]]:
        data = self._post_json("/api/fs/link", {"path": self.mount_path(relative_path), "password": ""})
        url = data.get("url") or data.get("download_url") or data.get("raw_url")
        headers = data.get("header") or data.get("headers") or {}
        if not isinstance(url, str) or not url.strip():
            raise DinoV3CloudError(f"NeriCloud 未返回下载链接: {relative_path}")
        url = url.strip()
        if url.startswith("//"):
            url = urllib.parse.urlsplit(self.origin).scheme + ":" + url
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise DinoV3CloudError("NeriCloud 下载链接无效。")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise DinoV3CloudError("NeriCloud 下载链接必须使用 HTTPS。")
        safe_headers: dict[str, str] = {}
        if isinstance(headers, dict):
            for key, value in headers.items():
                name = str(key)
                if name.lower() in {"user-agent", "referer", "cookie", "authorization", "accept"}:
                    safe_headers[name] = str(value)
        safe_headers.setdefault("User-Agent", "Neri DINOv3 component installer")
        return url, safe_headers

    def download_file(
        self,
        relative_path: str,
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        url, headers = self._file_link(relative_path)
        request = urllib.request.Request(url, headers=headers, method="GET")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".download")
        temp.unlink(missing_ok=True)
        try:
            with self.opener(request, timeout=90) as response, temp.open("wb") as output:
                raw_total = response.headers.get("Content-Length") if hasattr(response, "headers") else None
                total = int(raw_total) if raw_total and str(raw_total).isdigit() else None
                received = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received, total)
            temp.replace(target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def download_tree(
        self,
        relative_path: str,
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for item in self.list_directory(relative_path):
            name = item.get("name")
            if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise DinoV3CloudError("NeriCloud 返回了不安全的文件名。")
            child_relative = "/".join(filter(None, (relative_path.strip("/"), name)))
            child_target = target / name
            if bool(item.get("is_dir")):
                self.download_tree(child_relative, child_target, on_progress=on_progress)
            else:
                self.download_file(child_relative, child_target, on_progress=on_progress)


def _component_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else get_model_layout().dinov3_root


def dinov3_component_paths(*, root: Path | None = None) -> DinoV3ComponentPaths:
    component_root = _component_root(root)
    source_root = component_root / "source"
    backbone_dir = component_root / "dinov3-vitb16"
    return DinoV3ComponentPaths(
        root=component_root,
        source_root=source_root,
        source_package=source_root / "dinov3",
        backbone_dir=backbone_dir,
        backbone=backbone_dir / DINO_BACKBONE_FILENAME,
        classifier=component_root / DINO_CLASSIFIER_FILENAME,
        model_manifest=component_root / DINO_MODEL_MANIFEST_FILENAME,
        license=component_root / "LICENSE_DINOv3.md",
        install_manifest=component_root / "install.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 不是 JSON 对象。")
    return value


def _inventory_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\\" in raw:
        raise ValueError("DINOv3 install.json files 包含不安全路径。")
    if raw.startswith("/") or raw.endswith("/") or "//" in raw:
        raise ValueError("DINOv3 install.json files 包含不安全路径。")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("DINOv3 install.json files 包含不安全路径。")
    normalized = pure.as_posix()
    if normalized == "install.json":
        raise ValueError("DINOv3 install.json files 不应包含 install.json。")
    return normalized


def _is_runtime_bytecode_cache(path: Path) -> bool:
    relative = path.relative_to(path.anchor) if path.is_absolute() else path
    return "__pycache__" in relative.parts and path.suffix.lower() in {".pyc", ".pyo"}


def _validate_file_inventory(paths: DinoV3ComponentPaths, install_manifest: dict[str, Any]) -> None:
    items = install_manifest.get("files")
    if not isinstance(items, list) or not items:
        raise ValueError("DINOv3 install.json files 必须是完整的非空文件清单。")
    declared: dict[str, dict[str, Any]] = {}
    root = paths.root.resolve()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("DINOv3 install.json files 条目格式无效。")
        relative = _inventory_path(item.get("path"))
        if relative in declared:
            raise ValueError(f"DINOv3 install.json files 包含重复路径: {relative}")
        raw_sha = item.get("sha256")
        if not isinstance(raw_sha, str) or len(raw_sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in raw_sha):
            raise ValueError(f"DINOv3 install.json files SHA-256 无效: {relative}")
        size = item.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(f"DINOv3 install.json files size 无效: {relative}")
        candidate = paths.root.joinpath(*PurePosixPath(relative).parts)
        if candidate.is_symlink():
            raise ValueError(f"DINOv3 install.json files 不允许符号链接: {relative}")
        try:
            candidate.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"DINOv3 install.json files 包含不安全路径: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"DINOv3 install.json files 缺少文件: {relative}")
        if size is not None and candidate.stat().st_size != size:
            raise ValueError(f"DINOv3 install.json files size 不匹配: {relative}")
        if _sha256_file(candidate).lower() != raw_sha.lower():
            raise ValueError(f"DINOv3 install.json files SHA-256 不匹配: {relative}")
        declared[relative] = item
    actual = {
        path.relative_to(paths.root).as_posix()
        for path in paths.root.rglob("*")
        if path.is_file()
        and path != paths.install_manifest
        and not _is_runtime_bytecode_cache(path)
    }
    declared_paths = set(declared)
    if actual != declared_paths:
        missing_from_manifest = sorted(actual - declared_paths)
        missing_from_disk = sorted(declared_paths - actual)
        details: list[str] = []
        if missing_from_manifest:
            details.append("未列出: " + "、".join(missing_from_manifest[:4]))
        if missing_from_disk:
            details.append("缺失: " + "、".join(missing_from_disk[:4]))
        raise ValueError("DINOv3 install.json files 未覆盖完整镜像" + ("（" + "；".join(details) + "）" if details else "。"))


def _component_health(paths: DinoV3ComponentPaths) -> tuple[bool, str]:
    required_files = (
        paths.source_package / "__init__.py",
        paths.source_package / "hub" / "backbones.py",
        paths.source_package / "models" / "vision_transformer.py",
        paths.backbone,
        paths.classifier,
        paths.model_manifest,
        paths.license,
        paths.install_manifest,
    )
    missing = [path.relative_to(paths.root).as_posix() for path in required_files if not path.is_file()]
    if missing:
        return False, "DINOv3 安装不完整，缺少: " + "、".join(missing[:4])
    try:
        install_manifest = _read_json_object(paths.install_manifest)
        if install_manifest.get("schema_version") != 1:
            return False, "DINOv3 install.json schema 不受支持。"
        if install_manifest.get("component") != "dinov3":
            return False, "DINOv3 install.json component 不匹配。"
        if install_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 install.json architecture 不匹配。"
        source = install_manifest.get("source")
        if not isinstance(source, dict) or source.get("commit") != DINO_SOURCE_COMMIT:
            return False, "DINOv3 source commit 不匹配。"
        backbone_meta = install_manifest.get("backbone")
        if not isinstance(backbone_meta, dict):
            return False, "DINOv3 backbone 元数据缺失。"
        if backbone_meta.get("sha256") != DINO_BACKBONE_SHA256:
            return False, "DINOv3 backbone 元数据不匹配。"
        classifier_meta = install_manifest.get("classifier")
        if not isinstance(classifier_meta, dict):
            return False, "DINOv3 classifier 元数据缺失。"
        if classifier_meta.get("filename") != DINO_CLASSIFIER_FILENAME:
            return False, "DINOv3 classifier 文件名不匹配。"
        classifier_sha256 = classifier_meta.get("sha256")
        if (
            not isinstance(classifier_sha256, str)
            or len(classifier_sha256) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in classifier_sha256)
        ):
            return False, "DINOv3 classifier SHA-256 元数据无效。"
        classifier_sha256 = classifier_sha256.lower()
        if classifier_meta.get("head_type") != "multi_prototype":
            return False, "DINOv3 classifier head_type 不匹配。"
        selection_k = classifier_meta.get("selection_k")
        if (
            isinstance(selection_k, bool)
            or not isinstance(selection_k, int)
            or selection_k <= 0
        ):
            return False, "DINOv3 classifier selection_k 必须是正整数。"
        if classifier_meta.get("manifest") != DINO_MODEL_MANIFEST_FILENAME:
            return False, "DINOv3 classifier manifest 不匹配。"
        _validate_file_inventory(paths, install_manifest)
        model_manifest = _read_json_object(paths.model_manifest)
        if model_manifest.get("schema_version") != 1 or model_manifest.get("backend") != "dinov3":
            return False, "DINOv3 模型 manifest 不受支持。"
        if model_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 模型 architecture 不匹配。"
        if model_manifest.get("feature_dim") != 768:
            return False, "DINOv3 模型 feature_dim 不匹配。"
        if model_manifest.get("checkpoint") != DINO_CLASSIFIER_FILENAME:
            return False, "DINOv3 模型 checkpoint 不匹配。"
        manifest_selection_k = model_manifest.get("selection_k")
        if manifest_selection_k not in (None, selection_k):
            return False, "DINOv3 模型 manifest selection_k 与 install.json 不匹配。"
    except ValueError as exc:
        return False, str(exc)

    if _sha256_file(paths.backbone).lower() != DINO_BACKBONE_SHA256:
        return False, "DINOv3 backbone SHA-256 校验失败。"
    if _sha256_file(paths.classifier).lower() != classifier_sha256:
        return False, "DINOv3 classifier SHA-256 校验失败。"
    if importlib.util.find_spec("torch") is None:
        return True, "DINOv3 组件文件完整；PyTorch 未安装，推理暂不可用。"
    try:
        checkpoint = load_checkpoint(paths.classifier)
    except (CheckpointValidationError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return False, f"DINOv3 checkpoint 校验失败: {exc}"
    if (
        checkpoint.head_type != "multi_prototype"
        or checkpoint.selection_k != selection_k
    ):
        return False, "DINOv3 checkpoint 与 install.json Multi-prototype 元数据不匹配。"
    return True, f"DINOv3 ViT-B/16 Multi-prototype K={selection_k} 已安装。"


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]:
    paths = dinov3_component_paths(root=root)
    installed = paths.root.is_dir()
    healthy = False
    message = "DINOv3 未安装。"
    classifier_filename: str | None = None
    classifier_fingerprint: str | None = None
    classifier_head_type: str | None = None
    selection_k: int | None = None
    if installed:
        try:
            healthy, message = _component_health(paths)
            if healthy:
                install_manifest = _read_json_object(paths.install_manifest)
                classifier_meta = install_manifest.get("classifier")
                if isinstance(classifier_meta, dict):
                    raw_filename = classifier_meta.get("filename")
                    if isinstance(raw_filename, str) and raw_filename:
                        classifier_filename = raw_filename
                    raw_head_type = classifier_meta.get("head_type")
                    if isinstance(raw_head_type, str) and raw_head_type:
                        classifier_head_type = raw_head_type
                    raw_selection_k = classifier_meta.get("selection_k")
                    if (
                        isinstance(raw_selection_k, int)
                        and not isinstance(raw_selection_k, bool)
                        and raw_selection_k > 0
                    ):
                        selection_k = raw_selection_k
                    classifier_fingerprint = _sha256_file(paths.classifier).lower()
        except (OSError, ValueError) as exc:
            healthy = False
            message = f"DINOv3 健康检查失败: {exc}"
    return {
        "installed": installed,
        "healthy": healthy,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "source_commit": DINO_SOURCE_COMMIT,
        "classifier_filename": classifier_filename,
        "classifier_fingerprint": classifier_fingerprint,
        "classifier_head_type": classifier_head_type,
        "selection_k": selection_k,
        "message": message,
    }


def _smoke_test_source(paths: DinoV3ComponentPaths) -> None:
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from dinov3.hub.backbones import dinov3_vitb16\n"
        "dinov3_vitb16(pretrained=False)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code, str(paths.source_root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"DINOv3 source smoke test failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown source error").strip()
        raise RuntimeError(f"DINOv3 source smoke test failed: {detail}")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def install_dinov3_component(
    *,
    root: Path | None = None,
    cloud_client: DinoV3CloudClient | Any | None = None,
    seed_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Mirror, validate and atomically activate the NeriCloud DINOv3 tree.

    ``seed_dir`` is accepted only for call-site compatibility with older Neri
    builds. Strict mirror installation deliberately ignores it.
    """
    del seed_dir
    target = _component_root(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud = cloud_client or DinoV3CloudClient()
    stage = Path(tempfile.mkdtemp(prefix=".DINOv3-sync-", dir=str(target.parent))).resolve()
    backup = target.parent / ".DINOv3-backup"

    def progress(value: int, message: str) -> None:
        if on_progress is not None:
            on_progress(max(0, min(100, int(value))), message)

    activated = False
    had_previous = target.exists() or target.is_symlink()
    try:
        progress(10, "正在同步 NeriCloud DINOv3 组件...")
        cloud.download_tree("", stage)
        progress(75, "正在校验 DINOv3 镜像完整性...")
        staged_paths = dinov3_component_paths(root=stage)
        healthy, message = _component_health(staged_paths)
        if not healthy:
            raise RuntimeError(message)
        progress(82, "正在验证 DINOv3 source...")
        _smoke_test_source(staged_paths)
        _remove_path(backup)
        if had_previous:
            target.rename(backup)
        try:
            stage.rename(target)
            activated = True
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        progress(90, "正在验证已激活的 DINOv3 组件...")
        active_paths = dinov3_component_paths(root=target)
        active_healthy, active_message = _component_health(active_paths)
        if not active_healthy:
            _remove_path(target)
            activated = False
            if backup.exists():
                backup.rename(target)
            raise RuntimeError(active_message)
        _remove_path(backup)
        progress(100, "DINOv3 ViT-B/16 同步完成。")
        return target
    except Exception:
        if activated:
            _remove_path(target)
            if backup.exists():
                backup.rename(target)
        raise
    finally:
        _remove_path(stage)


def remove_dinov3_component(*, root: Path | None = None) -> None:
    """Remove mirrored DINOv3 assets without touching shared or learned state."""
    _remove_path(_component_root(root))
