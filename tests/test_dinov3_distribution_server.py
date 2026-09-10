from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from starlette.requests import Request

from server.model_distribution.manifest import ManifestEntry
from server.model_distribution.storage import RemoteEntry, UpstreamLink


class _RecursiveStore:
    def __init__(self):
        self.files = {
            "/Neri_Data/Model/DINOv3/install.json": b"install",
            "/Neri_Data/Model/DINOv3/source/dinov3/__init__.py": b"package",
            "/Neri_Data/Model/DINOv3/source/dinov3/models/vit.py": b"model",
            "/Neri_Data/Model/cls/unrelated.pt": b"ignore",
        }

    def list_dir(self, path: str):
        prefix = path.rstrip("/") + "/"
        children: dict[str, bool] = {}
        for full in self.files:
            if not full.startswith(prefix):
                continue
            rest = full[len(prefix) :]
            first, separator, _tail = rest.partition("/")
            children[first] = bool(separator)
        result = []
        for name, is_dir in sorted(children.items()):
            full = prefix + name
            data = b"" if is_dir else self.files[full]
            result.append(RemoteEntry(name, len(data), "1", is_dir))
        return result

    def resolve_link(self, path: str):
        return UpstreamLink("https://example.1drv.com/" + path.replace("/", "_"), {})

    def iter_bytes(self, link, range_header=None):
        encoded = link.url.removeprefix("https://example.1drv.com/")
        for path, data in self.files.items():
            if path.replace("/", "_") == encoded:
                yield data
                return
        raise KeyError(encoded)


class _Budget:
    def check_request(self, _ip):
        return None

    def reserve_proxy_bytes(self, _ip, _size):
        return None


class DinoV3DistributionServerTests(unittest.TestCase):
    def test_dinov3_manifest_recurses_only_component_tree(self):
        from server.model_distribution.service import DistributionService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = DistributionService(Path(temp_dir), _RecursiveStore())
            snapshot = service.dinov3_manifest()

        self.assertEqual(
            [entry.path for entry in snapshot.files],
            [
                "DINOv3/install.json",
                "DINOv3/source/dinov3/__init__.py",
                "DINOv3/source/dinov3/models/vit.py",
            ],
        )
        self.assertEqual(
            snapshot.files[0].sha256,
            hashlib.sha256(b"install").hexdigest(),
        )

    def test_app_exposes_anonymous_dinov3_manifest_and_direct_capability(self):
        from server.model_distribution.app import DinoV3DirectRequest, create_app
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.service import DistributionService

        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir)
            service = DistributionService(state, _RecursiveStore(), ttl_seconds=60)
            app: FastAPI = create_app(
                DistributionConfig(state, openlist_token="server-only-secret"),
                service=service,
                budget=_Budget(),
            )
            routes = {getattr(route, "path", None): route for route in app.routes}
            request = Request({"type": "http", "headers": [], "client": ("203.0.113.5", 1234)})

            manifest = routes["/v1/dinov3/manifest"].endpoint(request)
            entry = manifest["files"][0]
            capability = routes["/v1/dinov3/direct"].endpoint(
                DinoV3DirectRequest(
                    manifest_id=manifest["manifest_id"],
                    path=entry["path"],
                    sha256=entry["sha256"],
                ),
                request,
            )

        self.assertEqual(capability["path"], "DINOv3/install.json")
        self.assertEqual(capability["sha256"], entry["sha256"])
        self.assertTrue(capability["proxy_token"])
        self.assertNotIn("server-only-secret", repr(manifest) + repr(capability))


if __name__ == "__main__":
    unittest.main()
