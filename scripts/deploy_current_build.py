"""Install a compiled Neri UI and backend with verified backups and rollback.

Uses direct file operations so PowerShell drive-provider failures do not prevent
an update. Model assets, user settings and learning databases are not deployed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import psutil


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def active_processes(root):
    found = []
    for process in psutil.process_iter(["pid", "name"]):
        if (process.info.get("name") or "").lower() not in {"neri.exe", "neri_flutter.exe", "python.exe", "pythonw.exe"}:
            continue
        try:
            if Path(process.exe()).resolve().is_relative_to(root):
                found.append({"pid": process.pid, "name": process.info["name"]})
        except (psutil.Error, OSError):
            continue
    return found


def deploy(source, installation):
    source, installation = Path(source).resolve(), Path(installation).resolve()
    if not installation.is_dir():
        raise FileNotFoundError(installation)
    if running := active_processes(installation):
        raise RuntimeError(f"Close Neri before installation: {running}")
    release = inside(source, "frontend/build/windows/x64/runner/Release")
    if not (release / "Neri.exe").is_file():
        raise FileNotFoundError("Compile the Windows Release first")
    planned = {file.relative_to(release): file for file in release.rglob("*") if file.is_file()}
    system = inside(source, "system")
    for file in system.rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix not in {".pyc", ".pyo"}:
            planned[Path("system") / file.relative_to(system)] = file
    for relative in ("docs/dinov3-ridge-within.md", "scripts/deploy_current_build.py"):
        if (source / relative).is_file():
            planned[Path(relative)] = source / relative
    changes = []
    for relative, src in sorted(planned.items()):
        src.resolve().relative_to(source)
        target = inside(installation, relative)
        new_hash = digest(src)
        old_hash = digest(target) if target.is_file() else None
        if old_hash != new_hash:
            changes.append(dict(path=relative.as_posix(), source=str(src), sha256=new_hash, old_sha256=old_hash))
    protected = [installation / "temp/settings.json"]
    if protected[0].is_file():
        settings = json.loads(protected[0].read_text("utf-8"))
        selected = settings.get("selected_classification_model")
        if selected:
            manifest = Path(selected).resolve()
            manifest.relative_to(installation)
            protected.append(manifest)
    fingerprints = {str(path): digest(path) for path in protected if path.is_file()}
    backup = inside(installation, "repair-backups/rules-update-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup.mkdir(parents=True)
    for change in changes:
        relative = Path(change["path"])
        staged = inside(backup, Path("staging") / relative)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(change["source"], staged)
        if digest(staged) != change["sha256"]:
            raise OSError(f"Staged file differs: {relative}")
        if change["old_sha256"] is not None:
            saved = inside(backup, Path("files") / relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inside(installation, relative), saved)
            if digest(saved) != change["old_sha256"]:
                raise OSError(f"Backup differs: {relative}")
    inventory = dict(installation=str(installation), source=str(source), changes=changes, protected=fingerprints)
    (backup / "inventory.json").write_text(json.dumps(inventory, indent=2), "utf-8")
    if active_processes(installation):
        raise RuntimeError("Neri started during backup; no installed files were changed")
    applied = []
    try:
        for change in changes:
            relative = Path(change["path"])
            target = inside(installation, relative)
            if change["old_sha256"] is not None and digest(target) != change["old_sha256"]:
                raise OSError(f"Installed file changed during backup: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(inside(backup, Path("staging") / relative), target)
            applied.append(change)
            if digest(target) != change["sha256"]:
                raise OSError(f"Installed file verification failed: {relative}")
        if any(digest(path) != expected for path, expected in fingerprints.items()):
            raise OSError("Settings or selected model changed during installation")
    except BaseException:
        for change in reversed(applied):
            relative = Path(change["path"])
            target = inside(installation, relative)
            if change["old_sha256"] is None:
                target.unlink(missing_ok=True)
            else:
                restore = inside(backup, Path("restore") / relative)
                restore.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(inside(backup, Path("files") / relative), restore)
                os.replace(restore, target)
        raise
    report = dict(files_updated=len(changes), backup=str(backup), settings_preserved=True, selected_model_preserved=True)
    (backup / "verification.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="E:/Files/Neri")
    parser.add_argument("--installation", default="D:/Application/Neri")
    args = parser.parse_args()
    deploy(args.source, args.installation)
