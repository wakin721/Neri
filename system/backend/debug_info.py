"""Read-only diagnostics for the local backend."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import subprocess
from typing import Any

from . import __version__
from .maintenance import maintenance_log_path, maintenance_status_path, project_root

LOG_EXTENSIONS = {".log", ".txt", ".json"}


def installed_packages() -> list[dict[str, str]]:
    """Return installed Python packages for the running backend environment."""

    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        packages.append({"name": name, "version": distribution.version or ""})
    return sorted(packages, key=lambda item: item["name"].lower())


def runtime_diagnostics() -> dict[str, Any]:
    """Return PyTorch and GPU diagnostics for the running backend."""

    diagnostics: dict[str, Any] = {
        "backend_version": __version__,
        "pytorch_installed": False,
        "pytorch_version": None,
        "pytorch_cuda_version": None,
        "gpu_available": False,
        "gpu_devices": [],
        "hardware_gpus": _hardware_gpu_names(),
        "error": None,
    }
    try:
        import torch

        diagnostics["pytorch_installed"] = True
        diagnostics["pytorch_version"] = str(getattr(torch, "__version__", ""))
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        diagnostics["pytorch_cuda_version"] = str(cuda_version) if cuda_version else None

        gpu_devices: list[str] = []
        try:
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    gpu_devices.append(f"CUDA: {torch.cuda.get_device_name(index)}")
        except Exception as exc:  # noqa: BLE001 - diagnostic best effort
            diagnostics["error"] = str(exc)

        xpu = getattr(torch, "xpu", None)
        try:
            if xpu is not None and xpu.is_available():
                for index in range(xpu.device_count()):
                    get_name = getattr(xpu, "get_device_name", None)
                    name = get_name(index) if callable(get_name) else f"XPU {index}"
                    gpu_devices.append(f"XPU: {name}")
        except Exception as exc:  # noqa: BLE001 - diagnostic best effort
            diagnostics["error"] = str(exc)

        diagnostics["gpu_devices"] = gpu_devices
        diagnostics["gpu_available"] = bool(gpu_devices)
    except Exception as exc:  # noqa: BLE001 - torch may be absent or broken
        diagnostics["error"] = str(exc)
        try:
            diagnostics["pytorch_version"] = metadata.version("torch")
            diagnostics["pytorch_installed"] = True
        except metadata.PackageNotFoundError:
            pass

    return diagnostics


def list_debug_logs() -> list[dict[str, Any]]:
    """Return software logs that are safe to expose through the local UI."""

    files: dict[Path, Path] = {}
    root = project_root()
    for logs_root in (root / "temp", root / "logs"):
        if not logs_root.exists():
            continue
        for path in logs_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in LOG_EXTENSIONS:
                files[path.resolve()] = path

    for path in (maintenance_log_path(), maintenance_status_path()):
        if path.exists() and path.is_file():
            files[path.resolve()] = path

    return sorted(
        (_log_info(path) for path in files),
        key=lambda item: item.get("modified_at") or "",
        reverse=True,
    )


def read_debug_log(path_text: str, max_bytes: int = 200_000) -> dict[str, Any]:
    """Read the tail of a whitelisted log file."""

    path = _resolve_allowed_log_path(path_text)
    size = path.stat().st_size
    truncated = size > max_bytes
    with path.open("rb") as file:
        if truncated:
            file.seek(-max_bytes, 2)
        content = file.read().decode("utf-8", errors="replace")
    if truncated:
        content = f"... 仅显示最后 {max_bytes} 字节\n{content}"
    return {**_log_info(path), "content": content, "truncated": truncated}


def _resolve_allowed_log_path(path_text: str) -> Path:
    requested = Path(path_text)
    if not requested.is_absolute():
        requested = project_root() / requested
    path = requested.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("日志文件不存在。")
    if path.suffix.lower() not in LOG_EXTENSIONS:
        raise ValueError("不支持查看该类型的日志文件。")
    allowed_roots = [
        (project_root() / "temp").resolve(),
        (project_root() / "logs").resolve(),
    ]
    if any(_is_relative_to(path, root) for root in allowed_roots):
        return path
    allowed_files = {
        maintenance_log_path().resolve(),
        maintenance_status_path().resolve(),
    }
    if path in allowed_files:
        return path
    raise ValueError("日志路径不在允许的调试目录中。")


def _log_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": _display_name(path),
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _display_name(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return path.name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _hardware_gpu_names() -> list[str]:
    if not _is_windows():
        return []
    names = _powershell_property_values("Win32_VideoController", "Name")
    if names:
        return names
    return _wmic_property_values(["wmic", "path", "win32_VideoController", "get", "Name"])


def _powershell_property_values(cim_class: str, property_name: str) -> list[str]:
    command = f"Get-CimInstance {cim_class} | Select-Object -ExpandProperty {property_name}"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return _clean_command_lines(result.stdout)


def _wmic_property_values(command: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line for line in _clean_command_lines(result.stdout) if line.lower() != "name"]


def _clean_command_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _is_windows() -> bool:
    return os.name == "nt"
