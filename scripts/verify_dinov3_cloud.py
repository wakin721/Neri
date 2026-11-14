"""Download every published preview file using the desktop's actual client."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.dinov3.distribution import DinoV3DistributionClient


def verify(origin, root):
    if root.exists():
        raise FileExistsError(root)
    client = DinoV3DistributionClient(origin)
    manifest = client.fetch_manifest()
    client.download_tree('', root)
    receipt = json.loads((root/'package_inventory.json').read_text('utf-8'))
    rows = {row.relative_path: row for row in manifest.files}
    if set(rows) != set(receipt['files']) | {'package_inventory.json'}:
        raise ValueError('Downloaded inventory differs')
    for name, expected in receipt['files'].items():
        if rows[name].size != expected['size'] or rows[name].sha256 != expected['sha256']:
            raise ValueError('Published identity differs: ' + name)
    payload = json.loads((root/'Neri1.2preview.neri.json').read_text('utf-8'))
    if payload['display_name'] != 'Neri1.2preview' or payload['inference_recipe'] != 'experiment_full':
        raise ValueError('Unexpected downloaded model')
    for key in ('checkpoint', 'encoder_weights', 'encoder_code', 'calibration'):
        asset = (root/payload[key]).resolve(strict=True)
        asset.relative_to(root.resolve())
    return dict(passed=True, origin=origin, manifest_id=manifest.manifest_id,
                files=len(rows), total_bytes=sum(row.size for row in rows.values()),
                model=payload['display_name'], known_species=receipt['known_species'],
                checkpoint_sha256=payload['checkpoint_sha256'],
                calibration_sha256=payload['calibration_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', default='https://myneri.top/api/models/v1')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--serve-local', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    process = None
    log = None
    try:
        if args.serve_local:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env = dict(os.environ, NERI_OPENLIST_TOKEN='isolated-preflight',
                       NERI_MODEL_STATE=str(args.root.parent/'preflight-state'),
                       NERI_DINOV3_ROOT=str(args.serve_local.resolve(strict=True)),
                       NERI_MODEL_REQUESTS_PER_MINUTE='1200', PYTHONDONTWRITEBYTECODE='1')
            args.report.parent.mkdir(parents=True, exist_ok=True)
            log = args.report.with_suffix('.server.log').open('wb')
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn',
                'server.model_distribution.asgi:app', '--host', '127.0.0.1',
                '--port', str(port), '--no-access-log'],
                cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=log)
            for attempt in range(100):
                if process.poll() is not None:
                    raise RuntimeError('Preflight server exited; inspect server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except OSError:
                    if attempt == 99:
                        raise
                    time.sleep(.1)
            args.origin = f'http://127.0.0.1:{port}/v1'
        result = verify(args.origin, args.root)
        args.report.write_text(json.dumps(result, indent=2)+'\n', 'utf-8')
        print(json.dumps(result, indent=2))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
