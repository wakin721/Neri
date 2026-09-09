from __future__ import annotations

import hashlib

from system.dinov3 import component


def test_runtime_pycache_does_not_invalidate_strict_component_inventory(tmp_path):
    root = tmp_path / "DINOv3"
    root.mkdir()
    authoritative = root / "source" / "dinov3" / "hub" / "backbones.py"
    authoritative.parent.mkdir(parents=True)
    authoritative.write_text("# authoritative source\n", encoding="utf-8")
    digest = hashlib.sha256(authoritative.read_bytes()).hexdigest()
    install_manifest = {
        "files": [
            {
                "path": "source/dinov3/hub/backbones.py",
                "sha256": digest,
                "size": authoritative.stat().st_size,
            }
        ]
    }

    pycache = authoritative.parent / "__pycache__"
    pycache.mkdir()
    (pycache / "backbones.cpython-312.pyc").write_bytes(b"runtime bytecode")

    component._validate_file_inventory(
        component.dinov3_component_paths(root=root),
        install_manifest,
    )
