"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from system.model_sync.layout import get_model_layout

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_ARCHITECTURE_ID = "dinov3_vitb16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_BACKBONE_FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINO_BACKBONE_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
DINO_CLASSIFIER_FILENAME = "dinov3_classifier_merged_reviewed_20260908.pt"
DINO_CLASSIFIER_SHA256 = "b2f334da61c9feee51cff51bcded0af16878dcd5d72ea4f2e3e47b7f2adab76a"
DINO_MODEL_MANIFEST_FILENAME = "dinov3_classifier_merged_reviewed_20260908.neri.json"
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
        return DINO_CLOUD_ROOT if not parts else DINO_CLOUD_ROOT + "/" + "/".join(parts)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.origin + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Neri DINOv3 component installer",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except Exception as exc:  # noqa: BLE001 - normalize remote/network errors
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
            {
                "path": self.mount_path(relative_path),
                "password": "",
                "page": 1,
                "per_page": 0,
                "refresh": False,
            },
        )
        content = data.get("content")
        if content is None:
            return []
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise DinoV3CloudError("NeriCloud 目录列表格式无效。")
        return content

    def _file_link(self, relative_path: str) -> tuple[str, dict[str, str]]:
        data = self._post_json(
            "/api/fs/link",
            {"path": self.mount_path(relative_path), "password": ""},
        )
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
    """Return canonical installed paths for the optional DINOv3 component."""

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
        if install_manifest.get("component") not in (None, "dinov3"):
            return False, "DINOv3 install.json component 不匹配。"
        if install_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 install.json architecture 不匹配。"
        source = install_manifest.get("source")
        if not isinstance(source, dict) or source.get("commit") != DINO_SOURCE_COMMIT:
            return False, "DINOv3 source commit 不匹配。"
        backbone_meta = install_manifest.get("backbone")
        if not isinstance(backbone_meta, dict) or backbone_meta.get("sha256") != DINO_BACKBONE_SHA256:
            return False, "DINOv3 backbone 元数据不匹配。"

        model_manifest = _read_json_object(paths.model_manifest)
        if model_manifest.get("schema_version") != 1 or model_manifest.get("backend") != "dinov3":
            return False, "DINOv3 模型 manifest 不受支持。"
        if model_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 模型 architecture 不匹配。"
        if model_manifest.get("feature_dim") != 768:
            return False, "DINOv3 模型 feature_dim 不匹配。"
        if model_manifest.get("checkpoint") != DINO_CLASSIFIER_FILENAME:
            return False, "DINOv3 classifier 文件名不匹配。"
        if model_manifest.get("encoder_sha256") != DINO_BACKBONE_SHA256:
            return False, "DINOv3 模型 backbone 校验值不匹配。"
    except ValueError as exc:
        return False, str(exc)

    if _sha256_file(paths.backbone).lower() != DINO_BACKBONE_SHA256:
        return False, "DINOv3 backbone SHA-256 校验失败。"
    if _sha256_file(paths.classifier).lower() != DINO_CLASSIFIER_SHA256:
        return False, "DINOv3 classifier SHA-256 校验失败。"
    return True, "DINOv3 ViT-B/16 已安装。"


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]:
    """Return the optional component's current installation state."""

    paths = dinov3_component_paths(root=root)
    installed = paths.root.is_dir()
    healthy = False
    message = "DINOv3 未安装。"
    if installed:
        try:
            healthy, message = _component_health(paths)
        except OSError as exc:
            healthy = False
            message = f"DINOv3 健康检查失败: {exc}"
    return {
        "installed": installed,
        "healthy": healthy,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "source_commit": DINO_SOURCE_COMMIT,
        "message": message,
    }


def _default_seed_dir() -> Path:
    return get_model_layout().resource_root / "install" / "dinov3"


def _copy_seed_asset(seed_dir: Path, fallback: Path | None, target: Path) -> bool:
    source = seed_dir / target.name
    if not source.is_file() and fallback is not None and fallback.is_file():
        source = fallback
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _normalize_model_manifest(path: Path) -> None:
    payload = _read_json_object(path)
    payload["checkpoint"] = DINO_CLASSIFIER_FILENAME
    payload["architecture"] = DINO_ARCHITECTURE_ID
    payload["feature_dim"] = 768
    payload["encoder_weights"] = f"dinov3-vitb16/{DINO_BACKBONE_FILENAME}"
    payload["encoder_sha256"] = DINO_BACKBONE_SHA256
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install_dinov3_component(
    *,
    root: Path | None = None,
    cloud_client: DinoV3CloudClient | Any | None = None,
    seed_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Assemble and atomically activate the optional DINOv3 component."""

    target = _component_root(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud = cloud_client or DinoV3CloudClient()
    seeds = Path(seed_dir).resolve() if seed_dir is not None else _default_seed_dir()
    stage = Path(tempfile.mkdtemp(prefix=".DINOv3-install-", dir=str(target.parent))).resolve()
    backup = target.parent / ".DINOv3-backup"

    def progress(value: int, message: str) -> None:
        if on_progress is not None:
            on_progress(max(0, min(100, int(value))), message)

    try:
        stage_paths = dinov3_component_paths(root=stage)
        progress(10, "正在下载 DINOv3 安装清单...")
        cloud.download_file("install.json", stage_paths.install_manifest)
        progress(20, "正在下载 DINOv3 License...")
        cloud.download_file("LICENSE_DINOv3.md", stage_paths.license)
        progress(30, "正在下载 DINOv3 官方 source...")
        cloud.download_tree("source", stage_paths.source_root)
        progress(55, "正在下载 DINOv3 ViT-B/16 backbone...")
        cloud.download_file(
            f"dinov3-vitb16/{DINO_BACKBONE_FILENAME}",
            stage_paths.backbone,
        )

        layout = get_model_layout() if root is None else None
        manifest_fallback = (
            layout.cls_user / DINO_MODEL_MANIFEST_FILENAME if layout is not None else None
        )
        classifier_fallback = (
            layout.cls_user / DINO_CLASSIFIER_FILENAME if layout is not None else None
        )
        progress(75, "正在部署 DINOv3 分类头...")
        if not _copy_seed_asset(seeds, classifier_fallback, stage_paths.classifier):
            # The cloud fallback keeps clean installations self-contained when
            # the small reviewed seed was not bundled by an older package.
            cloud.download_file(DINO_CLASSIFIER_FILENAME, stage_paths.classifier)
        if not _copy_seed_asset(seeds, manifest_fallback, stage_paths.model_manifest):
            cloud.download_file(DINO_MODEL_MANIFEST_FILENAME, stage_paths.model_manifest)
        _normalize_model_manifest(stage_paths.model_manifest)

        progress(85, "正在校验 DINOv3 组件完整性...")
        healthy, message = _component_health(stage_paths)
        if not healthy:
            raise RuntimeError(message)

        backup_removed = False
        if backup.exists():
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            backup_removed = True
        del backup_removed
        progress(100, "DINOv3 ViT-B/16 安装完成。")
        return target
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def remove_dinov3_component(*, root: Path | None = None) -> None:
    """Remove DINOv3-specific runtime assets without touching shared packages."""

    component_root = _component_root(root)
    if not component_root.exists():
        return
    if component_root.is_symlink() or component_root.is_file():
        component_root.unlink()
        return
    shutil.rmtree(component_root)
