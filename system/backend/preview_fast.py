from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SQLITE_IN_CHUNK = 400


def _chunks(values: Iterable[str], size: int = _SQLITE_IN_CHUNK):
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _in_clause(size: int) -> str:
    return ','.join('?' for _ in range(size))


def load_detection_index_for_filenames(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> dict[str, dict[str, Any]]:
    """Load detection payloads only for the requested media filenames."""
    if not filenames:
        return {}

    ordered_filenames = sorted(filenames)
    detection_index: dict[str, dict[str, Any]] = {}
    for raw_db_path in db_paths:
        try:
            with sqlite3.connect(str(Path(raw_db_path))) as conn:
                for chunk in _chunks(ordered_filenames):
                    placeholders = _in_clause(len(chunk))
                    rows = conn.execute(
                        'SELECT base_name, image_filename, detection_json '
                        f'FROM detections WHERE image_filename IN ({placeholders})',
                        chunk,
                    ).fetchall()
                    for base_name, _image_filename, detection_json in rows:
                        key = str(base_name)
                        if key in detection_index:
                            continue
                        try:
                            data = json.loads(detection_json)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if isinstance(data, dict):
                            detection_index[key] = data
        except (OSError, sqlite3.Error):
            continue
    return detection_index


def load_validation_index_for_filenames(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> dict[str, bool]:
    """Load validation flags only for the requested media filenames."""
    if not filenames:
        return {}

    ordered_filenames = sorted(filenames)
    validation_index: dict[str, bool] = {}
    for raw_db_path in db_paths:
        try:
            with sqlite3.connect(str(Path(raw_db_path))) as conn:
                for chunk in _chunks(ordered_filenames):
                    placeholders = _in_clause(len(chunk))
                    rows = conn.execute(
                        'SELECT image_filename, is_validated '
                        f'FROM validation WHERE image_filename IN ({placeholders})',
                        chunk,
                    ).fetchall()
                    for image_filename, is_validated in rows:
                        validation_index.setdefault(
                            str(image_filename),
                            bool(is_validated),
                        )
        except (OSError, sqlite3.Error):
            continue
    return validation_index


def load_preview_indexes(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    """Load only rows needed by the current preview file list."""
    paths = tuple(Path(path) for path in db_paths)
    return (
        load_detection_index_for_filenames(paths, filenames),
        load_validation_index_for_filenames(paths, filenames),
    )


def make_preview_media_items(services_module: Any):
    """Build a drop-in replacement for services.preview_media_items."""

    def preview_media_items(
        input_dir: str,
        output_dir: str | None = None,
        *,
        include_cached: bool = True,
    ):
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists():
            raise ValueError(f'输入路径不存在: {input_path}')

        files = list(services_module._resolve_supported_inputs(input_path))
        if not files:
            return []

        detection_index: dict[str, dict[str, Any]] = {}
        validation_index: dict[str, bool] = {}
        if include_cached:
            roots = services_module._preview_detection_db_roots(
                input_path,
                output_dir,
                files,
            )
            db_paths = services_module._candidate_detection_dbs_for_roots(
                roots,
                recursive=False,
            )
            detection_index, validation_index = load_preview_indexes(
                db_paths,
                {path.name for path in files},
            )

        items = []
        for path in files:
            item = services_module._build_fast_metadata_item(path)
            if include_cached:
                data = detection_index.get(path.stem)
                if data:
                    item = services_module._apply_detection_data(item, data)
                item = services_module._apply_validation_state(
                    item,
                    path.name,
                    validation_index,
                )
            items.append(item)
        return items

    setattr(preview_media_items, '_neri_filtered_preview_sql', True)
    return preview_media_items


def make_preview_media_item(services_module: Any):
    """Build a single-item preview loader with targeted validation SQL."""

    def preview_media_item(
        file_path: str,
        input_dir: str | None = None,
        output_dir: str | None = None,
    ):
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f'文件不存在: {path}')
        supported = tuple(
            getattr(services_module, 'SUPPORTED_IMAGE_EXTENSIONS', ())
        ) + tuple(getattr(services_module, 'SUPPORTED_VIDEO_EXTENSIONS', ()))
        if supported and path.suffix.lower() not in supported:
            raise ValueError(f'不支持的媒体文件: {path}')

        item = services_module._build_metadata_item(path)
        roots: list[Path] = [path.parent]
        if input_dir:
            input_path = Path(input_dir).expanduser()
            roots.extend(
                services_module._detection_db_search_roots(
                    input_path,
                    output_dir,
                )
            )
        db_paths = services_module._candidate_detection_dbs_for_roots(
            roots,
            recursive=False,
        )
        validation_index = load_validation_index_for_filenames(
            db_paths,
            {path.name},
        )
        data = services_module._load_detection_data_for_path(
            path,
            roots,
            recursive=False,
        )
        if data:
            item = services_module._apply_detection_data(item, data)
        return services_module._apply_validation_state(
            item,
            path.name,
            validation_index,
        )

    setattr(preview_media_item, '_neri_filtered_single_preview_sql', True)
    return preview_media_item


def make_reload_validation_item(services_module: Any):
    """Build a validation reload helper that avoids a full validation scan."""

    def reload_validation_item(path: Path, input_path: Path):
        roots = services_module._detection_db_search_roots(input_path, None)
        item = services_module._build_metadata_item(path)
        data = services_module._load_detection_data_for_path(path, roots)
        if data:
            item = services_module._apply_detection_data(item, data)
        db_paths = services_module._candidate_detection_dbs_for_roots(
            roots,
            recursive=True,
        )
        validation_index = load_validation_index_for_filenames(
            db_paths,
            {path.name},
        )
        return services_module._apply_validation_state(
            item,
            path.name,
            validation_index,
        )

    setattr(
        reload_validation_item,
        '_neri_filtered_validation_reload_sql',
        True,
    )
    return reload_validation_item
