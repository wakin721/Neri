"""Backend-only staged deployment utility for mixed-media DINOv3 sequences.

Supports read-only planning (default), transactional deployment with SHA-256
backups and automatic rollback, optional ``--enable-fast-video`` settings update,
and validated manual rollback from ``repair-backups/mixed-media-*``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys
from typing import Any, Sequence
import uuid

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

EXPLICIT_FILES: tuple[str, ...] = (
    "system/metadata_extractor.py",
    "system/backend/services.py",
    "system/backend/services_legacy.py",
    "system/backend/video_frames.py",
    "system/backend/media_time.py",
    "system/dinov3/classifier.py",
    "system/dinov3/experiment.py",
    "system/dinov3/model.py",
    "system/dinov3/processor.py",
    "system/dinov3/evidence.py",
    "system/dinov3/media_cache.py",
    "system/dinov3/mixed_media.py",
    "system/dinov3/video.py",
    "system/dinov3/video_reference.py",
    "scripts/verify_mixed_media_installation.py",
    "docs/dinov3-mixed-media-sequences-20261116.md",
)

SETTINGS_REL_PATH = "temp/settings.json"
_ACTIVE_EXE_NAMES = frozenset({"neri.exe", "neri_flutter.exe", "python.exe", "pythonw.exe"})


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a regular file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of an in-memory byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction(path)) if isjunction is not None else False


def _validate_relative_parts(relative: str | Path) -> tuple[str, ...]:
    raw = str(relative).strip()
    if not raw or "\x00" in raw:
        raise ValueError(f"Unsafe empty or null path: {relative!r}")
    win = PureWindowsPath(raw)
    posix = PurePosixPath(raw.replace("\\", "/"))
    if win.drive or win.is_absolute() or posix.is_absolute():
        raise ValueError(f"Unsafe absolute path: {relative!r}")
    parts = tuple(part for part in posix.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe path traversal: {relative!r}")
    return parts


def resolve_safe_path(root: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` inside ``root`` while rejecting traversal and symlinks."""
    root_resolved = Path(root).resolve(strict=True)
    parts = _validate_relative_parts(relative)
    current = root_resolved
    for part in parts:
        current = current / part
        if _is_link_or_junction(current):
            raise ValueError(f"Unsafe symlink or junction: {current}")
        if current.exists():
            resolved_part = current.resolve(strict=True)
            if not resolved_part.is_relative_to(root_resolved):
                raise ValueError(f"Unsafe symlink escape: {current}")
    final_path = root_resolved.joinpath(*parts)
    if not final_path.resolve(strict=False).is_relative_to(root_resolved):
        raise ValueError(f"Unsafe path escape: {relative!r}")
    return final_path


def active_processes(installation_root: Path) -> list[dict[str, Any]]:
    """Return active Neri or Python processes executing inside ``installation_root``."""
    if psutil is None:
        return []
    root_resolved = Path(installation_root).resolve(strict=True)
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name"]):
        pid = getattr(process, "pid", None) or (process.info or {}).get("pid")
        name = ((process.info or {}).get("name") or "").lower()
        if pid == current_pid or name not in _ACTIVE_EXE_NAMES:
            continue
        try:
            exe = process.exe()
            if exe and Path(exe).resolve().is_relative_to(root_resolved):
                found.append({"pid": pid, "name": (process.info or {}).get("name")})
        except (psutil.Error, OSError):
            continue
    return found


def _atomic_write_bytes(target: Path, data: bytes, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.deploy-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with tmp_path.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256_file(tmp_path) != expected_sha256:
            raise OSError(f"Staged write hash mismatch for {target}")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_copy_replace(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.deploy-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copy2(source, tmp_path)
        if sha256_file(tmp_path) != expected_sha256:
            raise OSError(f"Temporary target hash mismatch for {target}")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_report(report_path: Path | None, payload: dict[str, Any]) -> None:
    if report_path is None:
        return
    resolved = Path(report_path)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(resolved, encoded, sha256_bytes(encoded))


def _load_settings_json(settings_path: Path) -> tuple[dict[str, Any], str]:
    raw_bytes = settings_path.read_bytes()
    old_sha256 = sha256_bytes(raw_bytes)
    parsed = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {settings_path}")
    return parsed, old_sha256


def _protected_paths(installation_root: Path, *, enable_fast_video: bool) -> dict[str, str]:
    protected: dict[str, str] = {}
    settings_target = resolve_safe_path(installation_root, SETTINGS_REL_PATH)
    if settings_target.is_file():
        if not settings_target.is_file() or _is_link_or_junction(settings_target):
            raise ValueError(f"Unsafe settings path: {settings_target}")
        settings_data, settings_hash = _load_settings_json(settings_target)
        if not enable_fast_video:
            protected[SETTINGS_REL_PATH] = settings_hash
        selected = settings_data.get("selected_classification_model")
        if isinstance(selected, str) and selected.strip():
            selected_path = Path(selected)
            install_resolved = installation_root.resolve(strict=True)
            if selected_path.is_absolute():
                try:
                    rel = selected_path.resolve(strict=False).relative_to(install_resolved)
                except ValueError:
                    rel = None
                if rel is not None:
                    safe_model = resolve_safe_path(installation_root, rel)
                    if safe_model.is_file():
                        protected[rel.as_posix()] = sha256_file(safe_model)
    return protected


def _prepare_settings_change(
    installation_root: Path,
    *,
    enable_fast_video: bool,
) -> tuple[dict[str, Any] | None, bytes | None, dict[str, Any] | None]:
    if not enable_fast_video:
        return None, None, None
    target = resolve_safe_path(installation_root, SETTINGS_REL_PATH)
    existing_data: dict[str, Any] | None = None
    old_hash: str | None = None
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Installation settings path is not a regular file: {target}")
        existing_data, old_hash = _load_settings_json(target)
        if existing_data.get("video_mode") == "fast":
            return None, None, existing_data
        updated = dict(existing_data)
    else:
        updated = {}
    updated["video_mode"] = "fast"
    encoded = (json.dumps(updated, ensure_ascii=False, indent=4) + "\n").encode("utf-8")
    new_hash = sha256_bytes(encoded)
    change = {
        "path": SETTINGS_REL_PATH,
        "kind": "settings",
        "action": "create" if old_hash is None else "update",
        "source": None,
        "sha256": new_hash,
        "old_sha256": old_hash,
    }
    return change, encoded, existing_data


def _collect_plan(
    source_root: Path,
    installation_root: Path,
    *,
    enable_fast_video: bool,
) -> tuple[Path, Path, list[dict[str, Any]], list[str], bytes | None, dict[str, Any] | None, dict[str, str]]:
    source = Path(source_root).resolve(strict=True)
    install = Path(installation_root).resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not install.is_dir():
        raise NotADirectoryError(install)
    if source == install:
        raise ValueError("source_root and installation_root must be distinct directories")

    changes: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for rel in EXPLICIT_FILES:
        src = resolve_safe_path(source, rel)
        if not src.is_file():
            raise FileNotFoundError(f"Required source file missing: {src}")
        dst = resolve_safe_path(install, rel)
        if dst.exists() and not dst.is_file():
            raise ValueError(f"Installation target is not a regular file: {dst}")
        new_hash = sha256_file(src)
        old_hash = sha256_file(dst) if dst.is_file() else None
        if old_hash == new_hash:
            unchanged.append(rel)
        else:
            changes.append({
                "path": rel,
                "kind": "code",
                "action": "create" if old_hash is None else "update",
                "source": str(src),
                "sha256": new_hash,
                "old_sha256": old_hash,
            })

    protected = _protected_paths(install, enable_fast_video=enable_fast_video)
    settings_change, settings_bytes, orig_settings = _prepare_settings_change(
        install, enable_fast_video=enable_fast_video,
    )
    if settings_change is not None:
        changes.append(settings_change)
    elif enable_fast_video:
        unchanged.append(SETTINGS_REL_PATH)

    return source, install, changes, unchanged, settings_bytes, orig_settings, protected


def plan_deployment(
    source_root: Path,
    installation_root: Path,
    *,
    enable_fast_video: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Compute hashes and planned updates without modifying ``installation_root``."""
    source, install, changes, unchanged, settings_bytes, _orig, protected = _collect_plan(
        source_root, installation_root, enable_fast_video=enable_fast_video,
    )
    staged_settings_path: str | None = None
    if settings_bytes is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        runs_rel = f"runs/mixed-media-plan-{stamp}/{SETTINGS_REL_PATH}"
        scratch_target = resolve_safe_path(source, runs_rel)
        _atomic_write_bytes(scratch_target, settings_bytes, sha256_bytes(settings_bytes))
        staged_settings_path = str(scratch_target)

    report = {
        "mode": "plan",
        "source_root": str(source),
        "installation_root": str(install),
        "enable_fast_video": enable_fast_video,
        "files_total": len(EXPLICIT_FILES),
        "files_to_update": len(changes),
        "unchanged_files": unchanged,
        "changes": changes,
        "protected": protected,
        "staged_settings_path": staged_settings_path,
        "active_processes": active_processes(install),
    }
    _write_report(report_path, report)
    return report


def _rollback_changes(
    installation_root: Path,
    backup_dir: Path,
    changes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    restored: list[str] = []
    removed: list[str] = []
    skipped_modified_new_files: list[str] = []
    for change in reversed(changes):
        rel = change["path"]
        target = resolve_safe_path(installation_root, rel)
        old_hash = change["old_sha256"]
        if old_hash is None:
            if target.is_file():
                if sha256_file(target) == change["sha256"]:
                    target.unlink()
                    removed.append(rel)
                else:
                    skipped_modified_new_files.append(rel)
            continue
        backup_rel = change.get("backup_file")
        if not isinstance(backup_rel, str):
            raise ValueError(f"Missing backup_file for {rel}")
        backup_file = resolve_safe_path(backup_dir, backup_rel)
        if not backup_file.is_file() or sha256_file(backup_file) != old_hash:
            raise ValueError(f"Backup verification failed for {rel}")
        _atomic_copy_replace(backup_file, target, old_hash)
        if sha256_file(target) != old_hash:
            raise OSError(f"Restored file verification failed for {rel}")
        restored.append(rel)
    return {
        "restored": restored,
        "removed": removed,
        "skipped_modified_new_files": skipped_modified_new_files,
    }


def apply_deployment(
    source_root: Path,
    installation_root: Path,
    *,
    enable_fast_video: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Stage, back up, verify, and atomically deploy the explicit mixed-media files."""
    source, install, changes, unchanged, settings_bytes, orig_settings, protected = _collect_plan(
        source_root, installation_root, enable_fast_video=enable_fast_video,
    )
    if running := active_processes(install):
        raise RuntimeError(f"Close Neri before installation (active processes): {running}")

    if not changes:
        report = {
            "mode": "apply",
            "applied": True,
            "source_root": str(source),
            "installation_root": str(install),
            "enable_fast_video": enable_fast_video,
            "files_updated": 0,
            "unchanged_files": unchanged,
            "backup_dir": None,
            "inventory_path": None,
            "settings_updated": False,
            "protected": protected,
            "verified": True,
        }
        _write_report(report_path, report)
        return report

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = resolve_safe_path(install, f"repair-backups/mixed-media-{stamp}")
    backup_dir.mkdir(parents=True, exist_ok=False)

    inventory_changes: list[dict[str, Any]] = []
    for change in changes:
        rel = change["path"]
        staged = resolve_safe_path(backup_dir, f"staging/{rel}")
        if change["kind"] == "settings":
            assert settings_bytes is not None
            _atomic_write_bytes(staged, settings_bytes, change["sha256"])
        else:
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(change["source"], staged)
            if sha256_file(staged) != change["sha256"]:
                raise OSError(f"Staged file verification failed: {rel}")

        backup_rel: str | None = None
        if change["old_sha256"] is not None:
            backup_rel = f"files/{rel}"
            saved = resolve_safe_path(backup_dir, backup_rel)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_safe_path(install, rel), saved)
            if sha256_file(saved) != change["old_sha256"]:
                raise OSError(f"Backup verification failed: {rel}")

        inventory_changes.append({**change, "backup_file": backup_rel})

    inventory = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "installation_root": str(install),
        "source_root": str(source),
        "backup_dir": str(backup_dir),
        "enable_fast_video": enable_fast_video,
        "changes": inventory_changes,
        "protected": protected,
    }
    inventory_path = resolve_safe_path(backup_dir, "inventory.json")
    inv_bytes = (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(inventory_path, inv_bytes, sha256_bytes(inv_bytes))

    if running := active_processes(install):
        raise RuntimeError(f"Neri started during backup; no installed files were changed: {running}")

    applied: list[dict[str, Any]] = []
    try:
        for change in inventory_changes:
            rel = change["path"]
            target = resolve_safe_path(install, rel)
            staged = resolve_safe_path(backup_dir, f"staging/{rel}")
            if change["old_sha256"] is not None:
                if not target.is_file() or sha256_file(target) != change["old_sha256"]:
                    raise OSError(f"Installed file changed during staging: {rel}")
            elif target.exists():
                raise OSError(f"Target file appeared unexpectedly during staging: {rel}")
            _atomic_copy_replace(staged, target, change["sha256"])
            applied.append(change)
            if sha256_file(target) != change["sha256"]:
                raise OSError(f"Post-install SHA-256 verification failed: {rel}")

        for rel, expected_hash in protected.items():
            protected_target = resolve_safe_path(install, rel)
            if not protected_target.is_file() or sha256_file(protected_target) != expected_hash:
                raise OSError(f"Protected file changed unexpectedly: {rel}")

        if settings_bytes is not None and orig_settings is not None:
            final_settings, _ = _load_settings_json(resolve_safe_path(install, SETTINGS_REL_PATH))
            if final_settings != {**orig_settings, "video_mode": "fast"}:
                raise OSError("Settings JSON semantic verification failed")
    except BaseException as exc:
        rollback_info = _rollback_changes(install, backup_dir, applied)
        _write_report(report_path, {
            "mode": "apply",
            "applied": False,
            "rolled_back": True,
            "error": str(exc),
            "source_root": str(source),
            "installation_root": str(install),
            "backup_dir": str(backup_dir),
            "inventory_path": str(inventory_path),
            "rollback": rollback_info,
        })
        raise

    report = {
        "mode": "apply",
        "applied": True,
        "source_root": str(source),
        "installation_root": str(install),
        "enable_fast_video": enable_fast_video,
        "files_updated": len(inventory_changes),
        "unchanged_files": unchanged,
        "backup_dir": str(backup_dir),
        "inventory_path": str(inventory_path),
        "changes": inventory_changes,
        "settings_updated": settings_bytes is not None,
        "protected": protected,
        "verified": True,
    }
    _write_report(report_path, report)
    return report


def _validate_rollback_inventory(install: Path, rollback_from: Path) -> tuple[Path, dict[str, Any]]:
    if _is_link_or_junction(Path(rollback_from)):
        raise ValueError(f"Unsafe symlink or junction rollback directory: {rollback_from}")
    backup_resolved = Path(rollback_from).resolve(strict=True)
    if not backup_resolved.is_dir():
        raise NotADirectoryError(backup_resolved)
    repair_backups = resolve_safe_path(install, "repair-backups")
    try:
        rel_backup = backup_resolved.relative_to(install)
    except ValueError as exc:
        raise ValueError(f"Rollback directory must reside inside installation/repair-backups: {rollback_from}") from exc
    if not backup_resolved.is_relative_to(repair_backups) or len(rel_backup.parts) < 2:
        raise ValueError(f"Rollback directory must reside inside installation/repair-backups: {rollback_from}")
    safe_backup = resolve_safe_path(install, rel_backup)

    inventory_path = resolve_safe_path(safe_backup, "inventory.json")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Rollback inventory not found: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or inventory.get("version") != 1:
        raise ValueError("Invalid rollback inventory format")
    inv_install = Path(inventory.get("installation_root", "")).resolve(strict=False)
    if inv_install != install:
        raise ValueError(
            f"Inventory installation_root mismatch: expected {install}, got {inv_install}"
        )
    changes = inventory.get("changes")
    if not isinstance(changes, list):
        raise ValueError("Invalid rollback inventory changes list")
    for change in changes:
        if not isinstance(change, dict) or not isinstance(change.get("path"), str):
            raise ValueError("Invalid rollback change entry")
        resolve_safe_path(install, change["path"])
        old_hash = change.get("old_sha256")
        if old_hash is None:
            if not isinstance(change.get("sha256"), str) or len(change["sha256"]) != 64:
                raise ValueError(f"Invalid applied sha256 in inventory for {change['path']}")
        else:
            backup_rel = change.get("backup_file")
            if not isinstance(backup_rel, str):
                raise ValueError(f"Missing backup_file in inventory for {change['path']}")
            backup_file = resolve_safe_path(safe_backup, backup_rel)
            if not backup_file.is_file() or sha256_file(backup_file) != old_hash:
                raise ValueError(f"Backup file missing or corrupt for {change['path']}")
    return safe_backup, inventory


def rollback_deployment(
    installation_root: Path,
    rollback_from: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate ``inventory.json`` and restore an installation from a backup directory."""
    install = Path(installation_root).resolve(strict=True)
    if not install.is_dir():
        raise NotADirectoryError(install)
    safe_backup, inventory = _validate_rollback_inventory(install, rollback_from)
    if running := active_processes(install):
        raise RuntimeError(f"Close Neri before rollback (active processes): {running}")

    outcome = _rollback_changes(install, safe_backup, inventory["changes"])
    report = {
        "mode": "rollback",
        "rolled_back": True,
        "installation_root": str(install),
        "backup_dir": str(safe_backup),
        "inventory_path": str(safe_backup / "inventory.json"),
        **outcome,
    }
    _write_report(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--installation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--plan", action="store_true", help="Plan-only mode (default).")
    mode_group.add_argument("--apply", action="store_true", help="Apply staged updates atomically.")
    mode_group.add_argument("--rollback-from", type=Path, help="Restore from a repair-backups directory.")
    parser.add_argument(
        "--enable-fast-video",
        action="store_true",
        help="Update temp/settings.json video_mode to 'fast' while preserving all other keys.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.rollback_from is not None:
        result = rollback_deployment(
            args.installation_root,
            args.rollback_from,
            report_path=args.report,
        )
    elif args.apply:
        result = apply_deployment(
            args.source_root,
            args.installation_root,
            enable_fast_video=args.enable_fast_video,
            report_path=args.report,
        )
    else:
        result = plan_deployment(
            args.source_root,
            args.installation_root,
            enable_fast_video=args.enable_fast_video,
            report_path=args.report,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
