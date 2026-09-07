"""Canonical local model layout and non-destructive legacy migration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from system.utils import resource_path

_DETECT_EXTENSIONS = frozenset({".pt"})
_CLS_EXTENSIONS = frozenset({".pt", ".onnx", ".engine"})
_LEGACY_DETECT_STAGING = ".neri-legacy-model"


@dataclass(frozen=True)
class ModelLayout:
    resource_root: Path
    root: Path
    detect_user: Path
    detect_sync: Path
    cls_user: Path
    cls_sync: Path
    tracker: Path
    state_file: Path


@dataclass(frozen=True)
class MigrationReport:
    moved: int
    collisions: tuple[str, ...]


def get_model_layout(resource_root: Path | None = None) -> ModelLayout:
    resolved_resource_root = (
        Path(resource_root).expanduser().resolve()
        if resource_root is not None
        else Path(resource_path("res")).resolve()
    )
    root = resolved_resource_root / "Model"
    layout = ModelLayout(
        resource_root=resolved_resource_root,
        root=root,
        detect_user=root / "detect" / "user",
        detect_sync=root / "detect" / "sync",
        cls_user=root / "cls" / "user",
        cls_sync=root / "cls" / "sync",
        tracker=root / "tracker.yaml",
        state_file=root / ".sync-state.json",
    )
    for directory in (
        layout.detect_user,
        layout.detect_sync,
        layout.cls_user,
        layout.cls_sync,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def _case_insensitive_paths() -> bool:
    return os.path.normcase("Model") == os.path.normcase("model")


def _exact_child(root: Path, name: str) -> Path | None:
    if not root.is_dir():
        return None
    try:
        for child in root.iterdir():
            if child.name == name:
                return child
    except OSError:
        return None
    return None


def _looks_like_canonical_root(path: Path) -> bool:
    return (path / "detect").is_dir() or (path / "cls").is_dir()


def _prepare_legacy_detect_source(resource_root: Path) -> Path | None:
    """Separate legacy top-level detect files from canonical content on Windows."""

    resource_root.mkdir(parents=True, exist_ok=True)
    staging = resource_root / _LEGACY_DETECT_STAGING
    if staging.is_dir():
        return staging

    legacy_exact = _exact_child(resource_root, "model")
    canonical_exact = _exact_child(resource_root, "Model")

    if not _case_insensitive_paths():
        return legacy_exact

    candidate = legacy_exact or canonical_exact
    if candidate is None:
        return None

    if not _looks_like_canonical_root(candidate):
        candidate.rename(staging)
        return staging

    staged_any = False
    for source in sorted(candidate.iterdir(), key=lambda path: path.name.casefold()):
        if not source.is_file():
            continue
        if source.suffix.lower() not in _DETECT_EXTENSIONS and source.name != "tracker.yaml":
            continue
        staging.mkdir(parents=True, exist_ok=True)
        target = staging / source.name
        if target.exists():
            target = _collision_target(target)
        shutil.move(str(source), str(target))
        staged_any = True
    return staging if staged_any else None


def _collision_target(target: Path) -> Path:
    index = 1
    while True:
        candidate = target.with_name(f"{target.stem}.legacy-{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _move_allowed_files(
    source_dir: Path | None,
    target_dir: Path,
    extensions: frozenset[str],
    collisions: list[str],
) -> int:
    if source_dir is None or not source_dir.is_dir():
        return 0
    moved = 0
    for source in sorted(source_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not source.is_file() or source.suffix.lower() not in extensions:
            continue
        target = target_dir / source.name
        if target.exists():
            target = _collision_target(target)
            collisions.append(str(target))
        shutil.move(str(source), str(target))
        moved += 1
    return moved


def migrate_legacy_layout(resource_root: Path | None = None) -> MigrationReport:
    resolved_resource_root = (
        Path(resource_root).expanduser().resolve()
        if resource_root is not None
        else Path(resource_path("res")).resolve()
    )
    legacy_detect = _prepare_legacy_detect_source(resolved_resource_root)
    layout = get_model_layout(resolved_resource_root)
    collisions: list[str] = []
    moved = 0
    legacy_cls = layout.resource_root / "model_cls"

    moved += _move_allowed_files(
        legacy_detect,
        layout.detect_user,
        _DETECT_EXTENSIONS,
        collisions,
    )
    moved += _move_allowed_files(
        legacy_cls,
        layout.cls_user,
        _CLS_EXTENSIONS,
        collisions,
    )

    if not layout.tracker.exists():
        candidates = [legacy_cls / "tracker.yaml"]
        if legacy_detect is not None:
            candidates.append(legacy_detect / "tracker.yaml")
        for source in candidates:
            if source.is_file():
                layout.tracker.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(layout.tracker))
                moved += 1
                break

    return MigrationReport(moved=moved, collisions=tuple(collisions))


def resolve_tracker_config(resource_root: Path | None = None) -> str:
    layout = get_model_layout(resource_root)
    return str(layout.tracker) if layout.tracker.is_file() else "botsort.yaml"
