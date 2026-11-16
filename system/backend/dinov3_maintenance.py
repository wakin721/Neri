"""Environment-maintenance worker for the offline DINOv3 component."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from system.dinov3.component import (
    _component_root,
    install_dinov3_component,
    remove_dinov3_component,
)
from system.dinov3.runtime_selection import (
    RUNTIME_CHOICES,
    resolve_runtime_choice,
    validate_runtime_choice,
)
from system.python_runtime import PYTHON_ISOLATION_ARGS, python_environment, python_module_command
from system.dinov3.download_diagnostics import event_line, exception_details, redact, sanitize_tree

from . import maintenance as _maintenance

DINO_OPTIONAL_DEPENDENCIES = {
    "numpy": "numpy",
    "PIL": "pillow>=11.2.1",
    "cv2": "opencv-python>=4.5.5",
}
DINO_CPU_OPTIONAL_DEPENDENCIES = {
    **DINO_OPTIONAL_DEPENDENCIES,
    "ai_edge_litert": "ai-edge-litert==2.3.0",
}
_LEGACY_PREVIEW_FILENAMES = frozenset(
    {
        "neri1.2preview.neri.json",
        "neri1.2preview-litert.neri.json",
    }
)
_LEGACY_PREVIEW_DISPLAY_NAMES = frozenset(
    {
        "neri1.2preview",
        "neri1.2preview·litert",
        "neri1.2preview（litert fp32 cpu）",
    }
)


def _installed_torch_supports_gpu() -> bool:
    # Loading torch here would lock its DLLs during the following pip repair.
    # subprocess.run waits for the probe to exit before any packages are changed.
    if importlib.util.find_spec("torch") is None:
        return False
    probe = (
        "import json; import torch; "
        "xpu=getattr(torch,'xpu',None); "
        "available=bool(torch.cuda.is_available()) or "
        f"({bool(_maintenance.XPU_ENABLED)!r} and xpu is not None and bool(xpu.is_available())); "
        "print(json.dumps({'gpu_available':bool(available)}))"
    )
    try:
        result = subprocess.run(
            [str(_maintenance.toolkit_python()), *PYTHON_ISOLATION_ARGS, "-c", probe],
            capture_output=True, text=True, timeout=60,
            env=python_environment(), creationflags=_maintenance._creation_flags(),
        )
        if result.returncode != 0:
            return False
        value = json.loads(result.stdout)
        return isinstance(value, dict) and value.get('gpu_available') is True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def _resolve_gpu_pytorch_env_choice(env_choice: str) -> tuple[str, dict[str, Any]]:
    plan = _maintenance.resolve_pytorch_install_plan(env_choice)
    cpu_index = _maintenance.PYTORCH_CPU_INDEX_URL.rstrip("/")
    if plan["index_url"].rstrip("/") == cpu_index and (env_choice or "").strip() != "自动检测":
        auto_plan = _maintenance.resolve_pytorch_install_plan("自动检测")
        if auto_plan["index_url"].rstrip("/") != cpu_index:
            return "自动检测", auto_plan
    return env_choice, plan


def _ensure_gpu_pytorch_environment(
    env_choice: str,
    package_source: str,
    *,
    progress_start: int = 5,
    progress_end: int = 35,
) -> None:
    effective_env_choice, plan = _resolve_gpu_pytorch_env_choice(env_choice)
    cpu_index = _maintenance.PYTORCH_CPU_INDEX_URL.rstrip("/")
    torch_missing = importlib.util.find_spec("torch") is None
    needs_gpu_repair = (
        plan["index_url"].rstrip("/") != cpu_index
        and not _installed_torch_supports_gpu()
    )
    if not torch_missing and not needs_gpu_repair:
        return
    message = (
        "未检测到 PyTorch，正在安装共享 PyTorch 运行环境..."
        if torch_missing
        else "检测到当前 PyTorch 不支持 GPU，正在安装 GPU 版本 PyTorch 运行环境..."
    )
    _maintenance._write_status({"message": message, "progress": progress_start})
    _maintenance._run_pytorch_install(
        effective_env_choice,
        package_source,
        progress_start=progress_start,
        progress_end=progress_end,
    )


def _settings_file_path() -> Path:
    return _maintenance.project_root() / "temp" / "settings.json"


def _is_legacy_preview_selection(saved_value: str) -> bool:
    trimmed = saved_value.strip()
    if not trimmed:
        return False
    folded = trimmed.casefold()
    if folded in _LEGACY_PREVIEW_DISPLAY_NAMES or folded in _LEGACY_PREVIEW_FILENAMES:
        return True
    normalized = trimmed.replace("\\", "/").casefold()
    return any(
        normalized.endswith(suffix)
        for suffix in (
            "/model/cls/bundled/neri1.2preview-litert.neri.json",
            "/model/cls/bundled/neri1.2preview.neri.json",
            "/model/dinov3/neri1.2preview-litert.neri.json",
            "/model/dinov3/neri1.2preview.neri.json",
        )
    )


def _capture_selected_classification_context() -> dict[str, Any] | None:
    settings_path = _settings_file_path()
    if not settings_path.is_file():
        return None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(settings, dict):
        return None
    saved = (
        settings.get("selected_classification_model")
        if "selected_classification_model" in settings
        else settings.get("selected_cls_model")
    )
    if not isinstance(saved, str) or not saved.strip():
        return None
    raw_value = saved.strip()
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = _maintenance.project_root() / candidate
    previous_family: str | None = None
    if candidate.is_file() and candidate.name.casefold().endswith(".neri.json"):
        from system.dinov3.shared_learning import family_id

        previous_family = family_id(candidate)
    return {
        "settings_path": settings_path,
        "previous_family": previous_family,
        "legacy_preview": _is_legacy_preview_selection(raw_value),
    }


def _migrate_selected_classification_model(
    context: dict[str, Any] | None,
    installed_status: dict[str, Any],
) -> None:
    if not context or not isinstance(installed_status, dict):
        return
    manifest_name = installed_status.get("manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        return
    from system.dinov3.shared_learning import PREVIEW_BASE, PREVIEW_ENCODER, family_descriptor

    new_manifest = (_component_root() / manifest_name).resolve()
    if not new_manifest.is_file():
        return
    try:
        new_payload = json.loads(new_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    descriptor = family_descriptor(new_manifest, new_payload)
    if descriptor is None:
        return
    should_migrate = False
    if context["previous_family"] is not None:
        should_migrate = descriptor["id"] == context["previous_family"]
    elif context["legacy_preview"]:
        should_migrate = (
            new_payload.get("encoder_sha256") == PREVIEW_ENCODER
            and descriptor.get("base_sha256") == PREVIEW_BASE
        )
    if not should_migrate:
        return
    settings_path: Path = context["settings_path"]
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(settings, dict):
        return
    target_value = str(new_manifest)
    if settings.get("selected_classification_model") == target_value:
        return
    settings["selected_classification_model"] = target_value
    settings_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def _missing_dino_dependencies(*, runtime: str = "gpu") -> list[str]:
    dependencies = (
        DINO_CPU_OPTIONAL_DEPENDENCIES if runtime == "cpu" else DINO_OPTIONAL_DEPENDENCIES
    )
    return [
        package_spec
        for module_name, package_spec in dependencies.items()
        if importlib.util.find_spec(module_name) is None
    ]


def _install_dino_python_dependencies(
    package_source: str = "auto",
    *,
    runtime: str = "gpu",
    progress_start: int = 35,
    progress_end: int = 55,
) -> None:
    missing = _missing_dino_dependencies(runtime=runtime)
    if not missing:
        return
    python_exe = _maintenance.toolkit_python()
    _, source_label, source_url = _maintenance.resolve_package_source(package_source)
    command = [
        str(python_exe),
        *PYTHON_ISOLATION_ARGS,
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
    message = f"正在从{source_label}安装 DINOv3 Python 依赖..."
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
    runtime_choice: str = "auto",
) -> dict[str, Any]:
    validated_runtime = validate_runtime_choice(runtime_choice)
    python_exe = _maintenance.toolkit_python()
    if not python_exe.exists():
        raise FileNotFoundError(f"未找到嵌入式 Python: {python_exe}")
    current = _maintenance.read_maintenance_status()
    if current.get("state") in _maintenance.ACTIVE_STATES:
        raise RuntimeError("已有环境维护任务正在执行，请等待其完成。")
    source_key, _, _ = _maintenance.resolve_package_source(package_source)
    message = (
        "联合模型组件安装已开始。"
        if operation == "install_dinov3"
        else "联合模型组件删除已开始。"
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
    if operation == "install_dinov3":
        status["runtime_choice"] = validated_runtime
    _maintenance._write_status(status, replace=True)
    command = python_module_command(
        python_exe,
        _maintenance.project_root(),
        "system.backend.dinov3_maintenance",
        "--operation",
        operation,
        "--backend-pid",
        str(os.getpid()),
        "--env-choice",
        (env_choice or "自动检测").strip(),
        "--package-source",
        source_key,
        "--runtime-choice",
        validated_runtime,
    )
    process = subprocess.Popen(
        command,
        cwd=str(_maintenance.project_root()),
        env=python_environment(),
        creationflags=_maintenance._creation_flags(),
        close_fds=True,
    )
    status["maintenance_pid"] = process.pid
    _maintenance._write_status({"maintenance_pid": process.pid})
    return status


def start_dinov3_install(
    env_choice: str = "自动检测",
    package_source: str = "auto",
    runtime_choice: str = "auto",
) -> dict[str, Any]:
    return _start(
        "install_dinov3",
        env_choice=env_choice,
        package_source=package_source,
        runtime_choice=runtime_choice,
    )


def start_dinov3_remove() -> dict[str, Any]:
    return _start("remove_dinov3")


def _progress(start: int, end: int):
    span = max(0, end - start)

    def callback(value: int, message: str) -> None:
        progress = start + round(span * max(0, min(100, int(value))) / 100)
        _maintenance._write_status(
            {"state": "running", "message": message, "progress": progress}
        )

    return callback


def _download_event(event: dict) -> None:
    log = _maintenance.maintenance_log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(event_line(event) + '\n')
    fields = {f'download_{key}': event[key] for key in
              ('phase', 'file', 'attempt', 'http_status', 'server_detail', 'offset') if key in event}
    if fields:
        _maintenance._write_status(fields)


def _failure(exc: Exception, *, phase: str) -> dict:
    details = exception_details(exc)
    details.update(sanitize_tree(getattr(exc, 'details', {})))
    try:
        _download_event({**details, 'state': 'failed', 'phase': details.get('phase', phase)})
    except OSError:
        pass
    return details


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        choices=["install_dinov3", "remove_dinov3"],
        required=True,
    )
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--env-choice", default="自动检测")
    parser.add_argument("--package-source", default="auto")
    parser.add_argument("--runtime-choice", choices=list(RUNTIME_CHOICES), default="auto")
    args = parser.parse_args(argv)

    _maintenance._write_status(
        {
            "state": "running",
            "message": "正在准备联合模型组件环境维护...",
            "progress": 2,
        }
    )
    try:
        _maintenance._wait_for_process_exit(args.backend_pid)
        _maintenance._wait_for_port_release()
        if args.operation == "install_dinov3":
            resolved_runtime = resolve_runtime_choice(args.runtime_choice)
            _maintenance._write_status(
                {
                    "runtime_choice": args.runtime_choice,
                    "resolved_runtime_choice": resolved_runtime,
                }
            )
            if resolved_runtime == "gpu":
                _ensure_gpu_pytorch_environment(
                    args.env_choice,
                    args.package_source,
                    progress_start=5,
                    progress_end=35,
                )
            _install_dino_python_dependencies(
                args.package_source,
                runtime=resolved_runtime,
                progress_start=35,
                progress_end=55,
            )
            selection_context = _capture_selected_classification_context()
            installed_status = install_dinov3_component(
                runtime_choice=args.runtime_choice,
                resolved_runtime_choice=resolved_runtime,
                on_progress=_progress(55, 92),
                on_event=_download_event,
            )
            _migrate_selected_classification_model(selection_context, installed_status)
            final_message = "联合模型组件安装完成，Python 后端已重启。"
        else:
            _maintenance._write_status(
                {"message": "正在删除联合模型组件资产...", "progress": 55}
            )
            remove_dinov3_component()
            final_message = "联合模型组件已删除；共享 PyTorch 环境已保留。"

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
        details = _failure(exc, phase='maintenance')
        current = _maintenance.read_maintenance_status()
        _maintenance._write_status(
            {
                "state": "failed",
                "message": f"联合模型组件环境维护失败: {redact(exc)}",
                "progress": min(99, max(0, int(current.get('progress') or 0))),
                "return_code": 1,
                "error": redact(exc),
                "error_details": details,
                "download_cache_preserved": args.operation == 'install_dinov3',
            }
        )
        try:
            _maintenance._start_backend()
        except Exception as restart_exc:  # noqa: BLE001
            _failure(restart_exc, phase='backend_restart')
            _maintenance._write_status(
                {
                    "message": f"联合模型组件环境维护失败，且 Python 后端恢复失败: {redact(restart_exc)}",
                    "error": f"{redact(exc)}; restart failed: {redact(restart_exc)}",
                }
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
