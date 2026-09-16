"""Environment-maintenance worker for the optional DINOv2 component."""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from typing import Any, Sequence

from system.dinov2.component import install_dinov2_component, remove_dinov2_component

from . import maintenance as _maintenance

DINO_OPTIONAL_DEPENDENCIES = {
    "transformers": "transformers>=4.40",
    "safetensors": "safetensors>=0.4",
}


def _missing_dinov2_dependencies() -> list[str]:
    return [
        package_spec
        for module_name, package_spec in DINO_OPTIONAL_DEPENDENCIES.items()
        if importlib.util.find_spec(module_name) is None
    ]


def _install_dinov2_python_dependencies(
    package_source: str = "auto",
    *,
    progress_start: int = 35,
    progress_end: int = 55,
) -> None:
    missing = _missing_dinov2_dependencies()
    if not missing:
        return
    python_exe = _maintenance.toolkit_python()
    _, source_label, source_url = _maintenance.resolve_package_source(package_source)
    command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--progress-bar",
        "raw",
        "--timeout",
        str(_maintenance.PIP_NETWORK_TIMEOUT_SECONDS),
        "--retries",
        str(_maintenance.PIP_NETWORK_RETRIES),
        "--cache-dir",
        str(_maintenance.maintenance_pip_cache_dir()),
        "--upgrade",
        *missing,
        "-i",
        source_url,
    ]
    message = f"正在从{source_label}安装 DINOv2 Python 依赖..."
    _maintenance._write_status(
        {"state": "running", "message": message, "progress": progress_start}
    )
    _maintenance._run_commands(
        [command],
        progress_start=progress_start,
        progress_end=progress_end,
        progress_message=message,
    )


def _start(
    operation: str,
    *,
    env_choice: str = "自动检测",
    package_source: str = "auto",
) -> dict[str, Any]:
    python_exe = _maintenance.toolkit_python()
    if not python_exe.exists():
        raise FileNotFoundError(f"未找到嵌入式 Python: {python_exe}")
    current = _maintenance.read_maintenance_status()
    if current.get("state") in _maintenance.ACTIVE_STATES:
        raise RuntimeError("已有环境维护任务正在执行，请等待其完成。")
    source_key, _, _ = _maintenance.resolve_package_source(package_source)
    message = (
        "DINOv2 安装已开始。"
        if operation == "install_dinov2"
        else "DINOv2 删除已开始。"
    )
    status = {
        "operation": operation,
        "state": "starting",
        "message": message,
        "progress": 0,
        "started_at": _maintenance._utc_now(),
        "updated_at": _maintenance._utc_now(),
        "log_path": str(_maintenance.maintenance_log_path()),
        "status_path": str(_maintenance.maintenance_status_path()),
    }
    _maintenance._write_status(status, replace=True)
    command = [
        str(python_exe),
        "-m",
        "system.backend.dinov2_maintenance",
        "--operation",
        operation,
        "--backend-pid",
        str(os.getpid()),
        "--env-choice",
        (env_choice or "自动检测").strip(),
        "--package-source",
        source_key,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(_maintenance.project_root()),
        creationflags=_maintenance._creation_flags(),
        close_fds=True,
    )
    status["maintenance_pid"] = process.pid
    _maintenance._write_status({"maintenance_pid": process.pid})
    return status


def start_dinov2_install(
    env_choice: str = "自动检测",
    package_source: str = "auto",
) -> dict[str, Any]:
    return _start(
        "install_dinov2",
        env_choice=env_choice,
        package_source=package_source,
    )


def start_dinov2_remove() -> dict[str, Any]:
    return _start("remove_dinov2")


def _progress(start: int, end: int):
    span = max(0, end - start)

    def callback(value: int, message: str) -> None:
        progress = start + round(span * max(0, min(100, int(value))) / 100)
        _maintenance._write_status(
            {"state": "running", "message": message, "progress": progress}
        )

    return callback


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        choices=["install_dinov2", "remove_dinov2"],
        required=True,
    )
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--env-choice", default="自动检测")
    parser.add_argument("--package-source", default="auto")
    args = parser.parse_args(argv)

    _maintenance._write_status(
        {
            "state": "running",
            "message": "正在准备 DINOv2 环境维护...",
            "progress": 2,
        }
    )
    try:
        _maintenance._wait_for_process_exit(args.backend_pid)
        _maintenance._wait_for_port_release()
        if args.operation == "install_dinov2":
            if importlib.util.find_spec("torch") is None:
                _maintenance._write_status(
                    {
                        "message": "未检测到 PyTorch，正在安装共享 PyTorch 运行环境...",
                        "progress": 5,
                    }
                )
                _maintenance._run_pytorch_install(
                    args.env_choice,
                    args.package_source,
                    progress_start=5,
                    progress_end=35,
                )
            _install_dinov2_python_dependencies(
                args.package_source,
                progress_start=35,
                progress_end=55,
            )
            install_dinov2_component(on_progress=_progress(55, 92))
            final_message = "DINOv2 ViT-B/14 安装完成，Python 后端已重启。"
        else:
            _maintenance._write_status(
                {"message": "正在删除 DINOv2 模型资产...", "progress": 55}
            )
            remove_dinov2_component()
            final_message = "DINOv2 已删除；共享 PyTorch 环境已保留。"

        _maintenance._write_status(
            {
                "state": "restarting",
                "message": "正在重启 Python 后端...",
                "progress": 95,
                "return_code": 0,
            }
        )
        _maintenance._start_backend()
        _maintenance._write_status(
            {
                "state": "completed",
                "message": final_message,
                "progress": 100,
                "return_code": 0,
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - maintenance boundary
        _maintenance._write_status(
            {
                "state": "failed",
                "message": f"DINOv2 环境维护失败: {exc}",
                "progress": 100,
                "return_code": 1,
                "error": str(exc),
            }
        )
        try:
            _maintenance._start_backend()
        except Exception as restart_exc:  # noqa: BLE001
            _maintenance._write_status(
                {
                    "message": f"DINOv2 环境维护失败，且 Python 后端恢复失败: {restart_exc}",
                    "error": f"{exc}; restart failed: {restart_exc}",
                }
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
