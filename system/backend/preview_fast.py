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


def load_preview_indexes(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    """Load only rows needed by the current preview file list.

    The detections table already has ``idx_det_imgfile`` on image_filename, so
    filtering in SQL avoids transferring and parsing every historical mark on
    each validation-page refresh.
    """

    if not filenames:
        return {}, {}

    ordered_filenames = sorted(filenames)
    detection_index: dict[str, dict[str, Any]] = {}
    validation_index: dict[str, bool] = {}

    for raw_db_path in db_paths:
        db_path = Path(raw_db_path)
        try:
            with sqlite3.connect(str(db_path)) as conn:
                for chunk in _chunks(ordered_filenames):
                    placeholders = _in_clause(len(chunk))
                    rows = conn.execute(
                        'SELECT base_name, image_filename, detection_json '
                        f'FROM detections WHERE image_filename IN ({placeholders})',
                        chunk,
                    ).fetchall()
                    for base_name, image_filename, detection_json in rows:
                        key = str(base_name)
                        if key in detection_index:
                            continue
                        try:
                            data = json.loads(detection_json)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if isinstance(data, dict):
                            detection_index[key] = data

                for chunk in _chunks(ordered_filenames):
                    placeholders = _in_clause(len(chunk))
                    rows = conn.execute(
                        'SELECT image_filename, is_validated '
                        f'FROM validation WHERE image_filename IN ({placeholders})',
                        chunk,
                    ).fetchall()
                    for image_filename, is_validated in rows:
                        key = str(image_filename)
                        validation_index.setdefault(key, bool(is_validated))
        except (OSError, sqlite3.Error):
            continue

    return detection_index, validation_index


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
