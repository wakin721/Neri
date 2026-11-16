"""Offer OneDrive mirrors while keeping local manifests and proxy fallback."""
from __future__ import annotations

from pathlib import PurePosixPath

from .storage import StorageError


class ComponentMirrorStore:
    def __init__(self, local, drive, *, gpu_root=None, cpu_root=None):
        self.local = local
        self.drive = drive
        self.roots = {
            '/Neri_Data/Model/DINOv3/cpu': self._root(cpu_root),
            '/Neri_Data/Model/DINOv3': self._root(gpu_root),
        }

    @staticmethod
    def _root(value):
        if value is None:
            return None
        if (not isinstance(value, str) or not value.startswith('/Neri_Data/Model/')
                or any(c in value for c in ('\\', '\x00', ':'))
                or any(p in {'', '.', '..'} for p in value.strip('/').split('/'))):
            raise ValueError('invalid_component_drive_root')
        return value.rstrip('/')

    def __getattr__(self, name):
        return getattr(self.local, name)

    def resolve_direct_link(self, path):
        for prefix, root in self.roots.items():
            if path == prefix or path.startswith(prefix + '/'):
                if root is None:
                    return self.local.resolve_link(path)
                relative = path[len(prefix)+1:]
                if (not relative or any(c in relative for c in ('\\', '\x00', ':'))
                        or any(p in {'', '.', '..'} for p in relative.split('/'))):
                    raise StorageError('invalid_component_mirror_path')
                return self.drive.resolve_link(root + '/' + PurePosixPath(relative).as_posix())
        return self.local.resolve_link(path)
