"""Canonical local model layout and non-destructive legacy migration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from system.utils import resource_path

_DETECT_EXTENSIONS = frozenset({".pt"})
_CLS_EXTENSIONS = frozenset({".pt", ".onnx", ".engine"})
_CASE_STAGING_NAME = ".neri-model-case-normalize"


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


def _resolved_resource_root(resource_root: Path | None) -> Path:
    return (
        Path(resource_root).expanduser().resolve()
        if resource_root is not None
        else Path(resource_path("res")).resolve()
    )


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


def _collision_target(target: Path) -> Path:
    index = 1
    while True:
        candidate = target.with_name(f"{target.stem}.legacy-{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _merge_tree(source: Path, target: Path) -> None:
    """Merge an old canonical tree without replacing existing local data."""

    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir(), key=lambda path: path.name.casefold()):
        destination = target / child.name
        if child.is_dir():
            if destination.exists() and destination.is_dir():
                _merge_tree(child, destination)
            elif not destination.exists():
                child.rename(destination)
            else:
                raise OSError(f"Cannot merge model directory into file: {destination}")
            continue

        if destination.exists():
            destination = _collision_target(destination)
        shutil.move(str(child), str(destination))

    try:
        source.rmdir()
    except OSError:
        pass


def _normalize_canonical_root_case(resource_root: Path) -> None:
    """Normalize alpha2's ``res/Model`` directory to lowercase ``res/model``.

    A two-step rename is required on case-insensitive filesystems such as
    Windows NTFS. The staging directory also makes an interrupted case-only
    rename recoverable on the next startup.
    """

    resource_root.mkdir(parents=True, exist_ok=True)
    lowercase = _exact_child(resource_root, "model")
    uppercase = _exact_child(resource_root, "Model")
    staging = _exact_child(resource_root, _CASE_STAGING_NAME)

    if staging is not None:
        if lowercase is None:
            staging.rename(resource_root / "model")
            lowercase = _exact_child(resource_root, "model")
        elif staging != lowercase:
            _merge_tree(staging, lowercase)

    if uppercase is None:
        return

    lowercase = _exact_child(resource_root, "model")
    if lowercase is not None and lowercase != uppercase:
        _merge_tree(uppercase, lowercase)
        return

    staging_path = resource_root / _CASE_STAGING_NAME
    if staging_path.exists():
        raise OSError(f"Model case-normalization staging path already exists: {staging_path}")
    uppercase.rename(staging_path)
    staging_path.rename(resource_root / "model")


def _build_layout(resource_root: Path) -> ModelLayout:
    root = resource_root / "model"
    return ModelLayout(
        resource_root=resource_root,
        root=root,
        detect_user=root / "detect" / "user",
        detect_sync=root / "detect" / "sync",
        cls_user=root / "cls" / "user",
        cls_sync=root / "cls" / "sync",
        tracker=root / "tracker.yaml",
        state_file=root / ".sync-state.json",
    )


def _ensure_layout_directories(layout: ModelLayout) -> None:
    for directory in (
        layout.detect_user,
        layout.detect_sync,
        layout.cls_user,
        layout.cls_sync,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_model_layout(resource_root: Path | None = None) -> ModelLayout:
    resolved_resource_root = _resolved_resource_root(resource_root)
    _normalize_canonical_root_case(resolved_resource_root)
    layout = _build_layout(resolved_resource_root)
    _ensure_layout_directories(layout)
    return layout


def _looks_like_canonical_root(path: Path) -> bool:
    return (
        (path / "detect").is_dir()
        or (path / "cls").is_dir()
        or (path / ".sync-state.json").is_file()
    )


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
            collisions.append(target.name)
        shutil.move(str(source), str(target))
        moved += 1
    return moved


def _migrate_tracker(
    layout: ModelLayout,
    legacy_cls: Path,
    *,
    had_canonical_structure: bool,
) -> int:
    """Migrate the old tracker while preserving an established canonical copy."""

    cls_tracker = legacy_cls / "tracker.yaml"
    flat_tracker = layout.tracker

    if had_canonical_structure and flat_tracker.is_file():
        return 0

    if cls_tracker.is_file():
        if flat_tracker.is_file():
            backup = _collision_target(flat_tracker)
            shutil.move(str(flat_tracker), str(backup))
        flat_tracker.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cls_tracker), str(flat_tracker))
        return 1

    # In old releases res/model/tracker.yaml was already at the same physical
    # path as the alpha3 canonical tracker. When no model_cls tracker exists,
    # leaving it in place is the non-destructive fallback migration.
    return 0


def migrate_legacy_layout(resource_root: Path | None = None) -> MigrationReport:
    resolved_resource_root = _resolved_resource_root(resource_root)
    _normalize_canonical_root_case(resolved_resource_root)
    layout = _build_layout(resolved_resource_root)
    had_canonical_structure = _looks_like_canonical_root(layout.root)
    legacy_cls = layout.resource_root / "model_cls"

    _ensure_layout_directories(layout)
    collisions: list[str] = []
    moved = 0

    # The alpha3 canonical root intentionally reuses the historic lowercase
    # res/model directory. Only immediate .pt files are legacy detection
    # models; canonical detect/cls subdirectories are never traversed here.
    moved += _move_allowed_files(
        layout.root,
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
    moved += _migrate_tracker(
        layout,
        legacy_cls,
        had_canonical_structure=had_canonical_structure,
    )

    return MigrationReport(moved=moved, collisions=tuple(collisions))


def resolve_tracker_config(resource_root: Path | None = None) -> str:
    layout = get_model_layout(resource_root)
    return str(layout.tracker) if layout.tracker.is_file() else "botsort.yaml"
