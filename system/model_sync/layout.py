"""Canonical local model layout and non-destructive legacy migration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from system.utils import resource_path

_DETECT_EXTENSIONS = frozenset({".pt"})
_CLS_EXTENSIONS = frozenset({".pt", ".onnx", ".engine"})


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


def _collision_target(target: Path) -> Path:
    index = 1
    while True:
        candidate = target.with_name(f"{target.stem}.legacy-{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _move_allowed_files(
    source_dir: Path,
    target_dir: Path,
    extensions: frozenset[str],
    collisions: list[str],
) -> int:
    if not source_dir.is_dir():
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
    layout = get_model_layout(resource_root)
    collisions: list[str] = []
    moved = 0
    legacy_detect = layout.resource_root / "model"
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
        for source in (
            legacy_cls / "tracker.yaml",
            legacy_detect / "tracker.yaml",
        ):
            if source.is_file():
                layout.tracker.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(layout.tracker))
                moved += 1
                break

    return MigrationReport(moved=moved, collisions=tuple(collisions))


def resolve_tracker_config(resource_root: Path | None = None) -> str:
    layout = get_model_layout(resource_root)
    return str(layout.tracker) if layout.tracker.is_file() else "botsort.yaml"
