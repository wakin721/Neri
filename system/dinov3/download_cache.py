"""Crash-tolerant partial files; completion still requires the published hash."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def partial_paths(target: Path):
    part = target.with_name(target.name + '.download')
    return part, part.with_name(part.name + '.json')


def discard_partial(target: Path):
    for path in partial_paths(target):
        path.unlink(missing_ok=True)


def restore_partial(target: Path, size: int, sha256: str, segment_size: int):
    part, metadata = partial_paths(target)
    digest = hashlib.sha256()
    try:
        if metadata.stat().st_size > 4096:
            raise ValueError('Invalid partial metadata')
        state = json.loads(metadata.read_text('utf-8'))
        offset = state['offset']
        if (state['size'] != size or state['sha256'] != sha256
                or state['segment_size'] != segment_size or type(offset) is not int
                or not 0 <= offset <= size or (offset != size and offset % segment_size)
                or part.stat().st_size < offset):
            raise ValueError('Partial identity differs')
        with part.open('rb') as stream:
            remaining = offset
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('Short partial')
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != state['prefix_sha256']:
            raise ValueError('Partial checksum differs')
        # A crash may leave bytes written after the last committed checkpoint.
        with part.open('r+b') as stream:
            stream.truncate(offset)
        return part, metadata, offset, digest
    except (OSError, ValueError, KeyError, TypeError):
        discard_partial(target)
        return part, metadata, 0, hashlib.sha256()


def checkpoint_partial(metadata, *, size, sha256, segment_size, offset, digest):
    state = dict(size=size, sha256=sha256, segment_size=segment_size,
                 offset=offset, prefix_sha256=digest.hexdigest())
    temporary = metadata.with_name(metadata.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(metadata)
