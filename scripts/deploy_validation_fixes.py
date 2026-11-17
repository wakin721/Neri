"""Guarded staged deployment utility for validation correctness and UI fixes.

Reuses ``scripts.deploy_mixed_media`` path safety, process checks, atomic
writes, SHA-256 backups, protected-settings checks, and rollback while
extending the explicit backend file set and staging compiled Windows Release UI
artifacts verified against the version-locked Flutter accessibility hotfix.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import shutil
import struct
import sys
from typing import Any, Iterator, Sequence

if __package__ in (None, ""):
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import scripts.deploy_mixed_media as deploy_mixed_media
import scripts.flutter_accessibility_hotfix as flutter_accessibility_hotfix

psutil = deploy_mixed_media.psutil
_INITIAL_PSUTIL = psutil

RELEASE_UI_REL_DIR = "frontend/build/windows/x64/runner/Release"
REQUIRED_UI_FILES: tuple[str, ...] = (
    "Neri.exe",
    "data/app.so",
    "flutter_windows.dll",
)
ENGINE_REL_PATH = "flutter_windows.dll"
_RESERVED_UI_ROOT_DIRS = frozenset({
    "docs",
    "repair-backups",
    "res",
    "scripts",
    "system",
    "temp",
    "toolkit",
})

BACKEND_EXPLICIT_FILES: tuple[str, ...] = tuple(dict.fromkeys((
    *deploy_mixed_media.EXPLICIT_FILES,
    "system/backend/preview_fast.py",
    "system/dinov3/detection_categories.py",
    "system/dinov3/context.py",
    "system/dinov3/job_cache.py",
    "system/dino/image_processor.py",
    "docs/validation-correctness-20261117.md",
)))
EXPLICIT_FILES: tuple[str, ...] = BACKEND_EXPLICIT_FILES
SETTINGS_REL_PATH = deploy_mixed_media.SETTINGS_REL_PATH

sha256_file = deploy_mixed_media.sha256_file
sha256_bytes = deploy_mixed_media.sha256_bytes
resolve_safe_path = deploy_mixed_media.resolve_safe_path


@contextmanager
def _synced_base_state() -> Iterator[None]:
    prev_explicit = deploy_mixed_media.EXPLICIT_FILES
    prev_psutil = deploy_mixed_media.psutil
    deploy_mixed_media.EXPLICIT_FILES = EXPLICIT_FILES
    if psutil is not _INITIAL_PSUTIL:
        deploy_mixed_media.psutil = psutil
    try:
        yield
    finally:
        deploy_mixed_media.EXPLICIT_FILES = prev_explicit
        deploy_mixed_media.psutil = prev_psutil


def active_processes(installation_root: Path) -> list[dict[str, Any]]:
    """Return active Neri or Python processes executing inside ``installation_root``."""
    with _synced_base_state():
        return deploy_mixed_media.active_processes(installation_root)


def verify_release_engine_hotfix(engine_path: Path) -> str:
    """Verify Release ``flutter_windows.dll`` already carries the known accessibility hotfix."""
    if deploy_mixed_media._is_link_or_junction(engine_path):
        raise ValueError(f"Unsafe symlink or junction engine artifact: {engine_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"Required compiled UI artifact missing: {engine_path}")
    existing_bytes = engine_path.read_bytes()
    try:
        guarded_bytes = flutter_accessibility_hotfix.patched_bytes(existing_bytes)
    except (ValueError, struct.error) as exc:
        raise ValueError(
            f"Release engine {engine_path} failed version-locked hotfix check: {exc}"
        ) from exc
    if guarded_bytes != existing_bytes:
        raise ValueError(
            f"Release engine {engine_path} is missing the required accessibility hotfix; "
            "apply scripts/flutter_accessibility_hotfix.py before deploying."
        )
    return sha256_bytes(existing_bytes)


def collect_release_ui_files(
    source_root: Path,
    *,
    explicit_files: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    """Discover and validate regular compiled UI files under the Windows Release directory."""
    release_root = resolve_safe_path(source_root, RELEASE_UI_REL_DIR)
    if not release_root.exists():
        raise FileNotFoundError(f"Compiled Windows Release directory not found: {release_root}")
    if not release_root.is_dir():
        raise NotADirectoryError(f"Compiled Windows Release path is not a directory: {release_root}")

    release_resolved = release_root.resolve(strict=True)
    backend_set = set(EXPLICIT_FILES if explicit_files is None else explicit_files)
    collected: list[tuple[str, Path]] = []

    def _walk(directory: Path) -> None:
        if deploy_mixed_media._is_link_or_junction(directory):
            raise ValueError(f"Unsafe symlink or junction in Release UI: {directory}")
        if not directory.resolve(strict=True).is_relative_to(release_resolved):
            raise ValueError(f"Unsafe path escape in Release UI: {directory}")
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if deploy_mixed_media._is_link_or_junction(entry):
                raise ValueError(f"Unsafe symlink or junction in Release UI: {entry}")
            resolved_entry = entry.resolve(strict=True)
            if not resolved_entry.is_relative_to(release_resolved):
                raise ValueError(f"Unsafe path escape in Release UI: {entry}")
            if entry.is_dir():
                _walk(entry)
            elif entry.is_file():
                rel = entry.relative_to(release_resolved).as_posix()
                safe_src = resolve_safe_path(release_resolved, rel)
                top_dir = rel.split("/", 1)[0]
                if rel in backend_set or rel == SETTINGS_REL_PATH or top_dir in _RESERVED_UI_ROOT_DIRS:
                    raise ValueError(f"Compiled UI artifact conflicts with reserved path: {rel}")
                collected.append((rel, safe_src))
            else:
                raise ValueError(f"Unsupported non-regular entry in Release UI: {entry}")

    _walk(release_resolved)

    discovered = {rel for rel, _ in collected}
    missing = [req for req in REQUIRED_UI_FILES if req not in discovered]
    if missing:
        raise FileNotFoundError(
            f"Required compiled UI artifact(s) missing in {release_resolved}: {', '.join(missing)}"
        )
    return collected


def _collect_plan(
    source_root: Path,
    installation_root: Path,
) -> tuple[Path, Path, list[dict[str, Any]], list[str], dict[str, str], int, int, str]:
    with _synced_base_state():
        source, install, changes, unchanged, settings_bytes, orig_settings, protected = (
            deploy_mixed_media._collect_plan(
                source_root,
                installation_root,
                enable_fast_video=False,
            )
        )
    assert settings_bytes is None and orig_settings is None
    backend_count = len(EXPLICIT_FILES)

    ui_files = collect_release_ui_files(source)
    engine_path = resolve_safe_path(source, f"{RELEASE_UI_REL_DIR}/{ENGINE_REL_PATH}")
    engine_sha256 = verify_release_engine_hotfix(engine_path)

    for rel, src in ui_files:
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
                "kind": "ui",
                "action": "create" if old_hash is None else "update",
                "source": str(src),
                "sha256": new_hash,
                "old_sha256": old_hash,
            })

    return source, install, changes, unchanged, protected, backend_count, len(ui_files), engine_sha256


def plan_deployment(
    source_root: Path,
    installation_root: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Compute hashes and planned updates without modifying ``installation_root``."""
    source, install, changes, unchanged, protected, backend_count, ui_count, engine_sha256 = _collect_plan(
        source_root,
        installation_root,
    )
    report = {
        "mode": "plan",
        "source_root": str(source),
        "installation_root": str(install),
        "files_total": backend_count + ui_count,
        "backend_files_total": backend_count,
        "ui_files_total": ui_count,
        "files_to_update": len(changes),
        "unchanged_files": unchanged,
        "changes": changes,
        "protected": protected,
        "engine_hotfix_verified": True,
        "engine_sha256": engine_sha256,
        "active_processes": active_processes(install),
    }
    deploy_mixed_media._write_report(report_path, report)
    return report


def _stage_and_backup_changes(
    install: Path,
    backup_dir: Path,
    changes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory_changes: list[dict[str, Any]] = []
    for change in changes:
        rel = change["path"]
        staged = resolve_safe_path(backup_dir, f"staging/{rel}")
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
    return inventory_changes


def apply_deployment(
    source_root: Path,
    installation_root: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Stage, back up, verify, and atomically deploy explicit backend and compiled UI files."""
    source, install, changes, unchanged, protected, backend_count, ui_count, engine_sha256 = _collect_plan(
        source_root,
        installation_root,
    )
    if running := active_processes(install):
        raise RuntimeError(f"Close Neri before installation (active processes): {running}")

    if not changes:
        report = {
            "mode": "apply",
            "applied": True,
            "source_root": str(source),
            "installation_root": str(install),
            "files_total": backend_count + ui_count,
            "backend_files_total": backend_count,
            "ui_files_total": ui_count,
            "files_updated": 0,
            "unchanged_files": unchanged,
            "backup_dir": None,
            "inventory_path": None,
            "protected": protected,
            "engine_hotfix_verified": True,
            "engine_sha256": engine_sha256,
            "verified": True,
        }
        deploy_mixed_media._write_report(report_path, report)
        return report

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = resolve_safe_path(install, f"repair-backups/validation-fixes-{stamp}")
    backup_dir.mkdir(parents=True, exist_ok=False)

    inventory_changes = _stage_and_backup_changes(install, backup_dir, changes)
    inventory = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "installation_root": str(install),
        "source_root": str(source),
        "backup_dir": str(backup_dir),
        "engine_sha256": engine_sha256,
        "changes": inventory_changes,
        "protected": protected,
    }
    inventory_path = resolve_safe_path(backup_dir, "inventory.json")
    inv_bytes = (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    deploy_mixed_media._atomic_write_bytes(inventory_path, inv_bytes, sha256_bytes(inv_bytes))

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
            deploy_mixed_media._atomic_copy_replace(staged, target, change["sha256"])
            applied.append(change)
            if sha256_file(target) != change["sha256"]:
                raise OSError(f"Post-install SHA-256 verification failed: {rel}")

        for rel, expected_hash in protected.items():
            protected_target = resolve_safe_path(install, rel)
            if not protected_target.is_file() or sha256_file(protected_target) != expected_hash:
                raise OSError(f"Protected file changed unexpectedly: {rel}")
    except BaseException as exc:
        rollback_info = deploy_mixed_media._rollback_changes(install, backup_dir, applied)
        deploy_mixed_media._write_report(report_path, {
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
        "files_total": backend_count + ui_count,
        "backend_files_total": backend_count,
        "ui_files_total": ui_count,
        "files_updated": len(inventory_changes),
        "unchanged_files": unchanged,
        "backup_dir": str(backup_dir),
        "inventory_path": str(inventory_path),
        "changes": inventory_changes,
        "protected": protected,
        "engine_hotfix_verified": True,
        "engine_sha256": engine_sha256,
        "verified": True,
    }
    deploy_mixed_media._write_report(report_path, report)
    return report


def rollback_deployment(
    installation_root: Path,
    rollback_from: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate ``inventory.json`` and restore an installation from a backup directory."""
    with _synced_base_state():
        return deploy_mixed_media.rollback_deployment(
            installation_root,
            rollback_from,
            report_path=report_path,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=False)
    parser.add_argument("--installation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--plan", action="store_true", help="Plan-only mode (default).")
    mode_group.add_argument("--apply", action="store_true", help="Apply staged updates atomically.")
    mode_group.add_argument("--rollback-from", type=Path, help="Restore from a repair-backups directory.")
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
    else:
        if args.source_root is None:
            parser.error("--source-root is required for --plan and --apply")
        if args.apply:
            result = apply_deployment(
                args.source_root,
                args.installation_root,
                report_path=args.report,
            )
        else:
            result = plan_deployment(
                args.source_root,
                args.installation_root,
                report_path=args.report,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
