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
LEGACY_TEMP_LOG_EXTENSIONS = {".log", ".txt"}
LEGACY_TEMP_LOG_NAMES = {
    "backend_maintenance.log",
    "backend_maintenance_status.json",
    "crash_startup_report.json",
    "crash_watchdog.tmp",
}
LEGACY_TEMP_LOG_PREFIXES = (
    "crash_watchdog_",
    "crash_startup_report_",
)
DEFAULT_LOG_TAIL_BYTES = 32_000
CACHE_FILE_NAMES = {
    "job_state.json",
    "crash_startup_report.json",
}
CACHE_FILE_PREFIXES = (
    "crash_watchdog_",
    "crash_startup_report_",
)
CACHE_PRESERVED_TEMP_NAMES = {
    "settings.json",
    "quick_mark.json",
}


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
    search_roots = (
        (root / "logs", LOG_EXTENSIONS),
    )
    for logs_root, extensions in search_roots:
        if not logs_root.exists():
            continue
        for path in logs_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                files[path.resolve()] = path

    for path in (maintenance_log_path(), maintenance_status_path()):
        if path.exists() and path.is_file():
            files[path.resolve()] = path

    return sorted(
        (_log_info(path) for path in files),
        key=lambda item: item.get("modified_at") or "",
        reverse=True,
    )


def read_debug_log(
    path_text: str,
    max_bytes: int = DEFAULT_LOG_TAIL_BYTES,
) -> dict[str, Any]:
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


def clear_debug_storage(
    *,
    clear_logs: bool,
    clear_software_cache: bool,
) -> dict[str, Any]:
    """Clear selected local runtime files under controlled project paths."""

    cleared_files = 0
    cleared_directories = 0
    reclaimed_bytes = 0
    skipped: list[str] = []

    def delete_file(path: Path) -> None:
        nonlocal cleared_files, reclaimed_bytes
        try:
            size = path.stat().st_size
            path.unlink()
            cleared_files += 1
            reclaimed_bytes += size
        except Exception as exc:  # noqa: BLE001 - cleanup should be best effort
            skipped.append(f"{_display_name(path)}：{exc}")

    def delete_directory(path: Path) -> None:
        nonlocal cleared_directories, reclaimed_bytes
        try:
            size = _directory_size(path)
            for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
            cleared_directories += 1
            reclaimed_bytes += size
        except Exception as exc:  # noqa: BLE001 - cleanup should be best effort
            skipped.append(f"{_display_name(path)}：{exc}")

    root = project_root()
    if clear_logs:
        log_roots = (
            (root / "logs", LOG_EXTENSIONS),
        )
        for log_root, extensions in log_roots:
            if not log_root.exists():
                continue
            for path in sorted(log_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    delete_file(path)
        legacy_temp_root = root / "temp"
        if legacy_temp_root.exists():
            for path in sorted(legacy_temp_root.rglob("*")):
                if path.is_file() and _is_legacy_temp_log_file(path):
                    delete_file(path)

    if clear_software_cache:
        temp_root = root / "temp"
        if temp_root.exists():
            for path in sorted(temp_root.iterdir()):
                if path.name in CACHE_PRESERVED_TEMP_NAMES or path.suffix.lower() == ".db":
                    continue
                if path.is_file() and _is_cache_file(path):
                    delete_file(path)
                elif path.is_dir() and _is_cache_directory(path):
                    delete_directory(path)

        for pycache in sorted((root / "system").rglob("__pycache__")):
            if pycache.is_dir():
                delete_directory(pycache)

    return {
        "cleared_files": cleared_files,
        "cleared_directories": cleared_directories,
        "reclaimed_bytes": reclaimed_bytes,
        "skipped": skipped,
    }


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


def _is_cache_file(path: Path) -> bool:
    name = path.name
    return (
        name in CACHE_FILE_NAMES
        or any(name.startswith(prefix) for prefix in CACHE_FILE_PREFIXES)
        or path.suffix.lower() in {".tmp", ".cache", ".bak"}
    )


def _is_legacy_temp_log_file(path: Path) -> bool:
    name = path.name
    return (
        name in LEGACY_TEMP_LOG_NAMES
        or any(name.startswith(prefix) for prefix in LEGACY_TEMP_LOG_PREFIXES)
        or path.suffix.lower() in LEGACY_TEMP_LOG_EXTENSIONS
    )


def _is_cache_directory(path: Path) -> bool:
    return path.name == "__pycache__" or path.name.lower().endswith("_cache")


def _directory_size(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


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
