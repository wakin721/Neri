"""Build the small, reviewable cloud-service deployment bundle (no weights)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(output):
    repo = Path(__file__).resolve().parents[1]
    sources = list((repo/'server/model_distribution').glob('*.py')) + [
        repo/'scripts/verify_dinov3_cloud.py',
        repo/'system/dinov3/distribution.py',
        repo/'system/dinov3/download_cache.py',
        repo/'system/dinov3/download_diagnostics.py',
        repo/'system/dinov3/runtime_selection.py',
        repo/'system/model_sync/protocol.py',
    ]
    files = {path.relative_to(repo).as_posix(): path.read_bytes() for path in sources}
    for name in ('server/__init__.py', 'system/__init__.py', 'system/dinov3/__init__.py',
                 'system/model_sync/__init__.py'):
        files.setdefault(name, b'')
    receipt = {name: dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
               for name, data in files.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr('service_inventory.json', json.dumps(dict(files=receipt), indent=2))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', 'ascii')
    print(json.dumps(dict(archive=str(output), files=len(files), sha256=digest)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
