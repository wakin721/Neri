"""Lightweight directory discovery, independent of preview metadata and DBs."""

import os
from pathlib import Path

from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


def preview_directories(input_dir: str) -> list[str]:
    try:
        root = Path(input_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f'输入文件夹不存在: {root}')
    except OSError as exc:
        raise ValueError(f'无法读取输入文件夹: {input_dir}') from exc
    supported = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
    pending = [root]
    media_dirs: set[Path] = set()
    ancestors: set[Path] = set()
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                has_media = False
                for entry in entries:
                    # Do not follow links/junctions into other trees or cycles.
                    if entry.is_dir(follow_symlinks=False):
                        child = Path(entry.path)
                        if not child.is_junction():
                            pending.append(child)
                    elif not has_media and entry.name.lower().endswith(supported):
                        has_media = entry.is_file(follow_symlinks=False)
                if has_media and directory != root:
                    media_dirs.add(directory)
                    parent = directory.parent
                    while parent != root and parent != parent.parent:
                        ancestors.add(parent)
                        parent = parent.parent
        except OSError as exc:
            if directory == root:
                raise ValueError(f'无法读取输入文件夹: {root}') from exc
            # One inaccessible camera must not hide all the accessible cameras.
            continue
    return sorted((str(path) for path in media_dirs - ancestors), key=os.path.normcase)
