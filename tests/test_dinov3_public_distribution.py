from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from system.dinov3 import DinoV3CloudClient


class _Response:
    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None, status: int = 200):
        self._body = body
        self._offset = 0
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._body):
            return b""
        if size is None or size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class DinoV3PublicDistributionClientTests(unittest.TestCase):
    def test_download_tree_uses_public_distribution_without_openlist_guest_login(self) -> None:
        payloads = {
            "DINOv3/install.json": b'{"component":"dinov3"}',
            "DINOv3/source/dinov3/__init__.py": b"# dinov3\n",
        }
        manifest_id = "a" * 64
        tokens = {
            "DINOv3/install.json": "install_proxy_token_1234567890",
            "DINOv3/source/dinov3/__init__.py": "source_proxy_token_1234567890",
        }
        requests: list[tuple[str, str, str | None]] = []

        def opener(request, timeout=20):
            authorization = request.get_header("Authorization")
            requests.append((request.get_method(), request.full_url, authorization))
            if request.full_url == "https://models.example/v1/dinov3/manifest":
                body = {
                    "schema_version": 1,
                    "manifest_id": manifest_id,
                    "files": [
                        {
                            "path": path,
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                        for path, data in payloads.items()
                    ],
                }
                return _Response(json.dumps(body).encode("utf-8"))
            if request.full_url == "https://models.example/v1/dinov3/direct":
                body = json.loads(request.data.decode("utf-8"))
                path = body["path"]
                data = payloads[path]
                return _Response(
                    json.dumps(
                        {
                            "path": path,
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "direct_url": None,
                            "direct_headers": {},
                            "proxy_token": tokens[path],
                            "expires_at": 9999999999.0,
                        }
                    ).encode("utf-8")
                )
            for path, token in tokens.items():
                if request.full_url == f"https://models.example/v1/proxy/{token}":
                    data = payloads[path]
                    return _Response(data, headers={"Content-Length": str(len(data))})
            raise AssertionError(f"unexpected request: {request.full_url}")

        client = DinoV3CloudClient(origin="https://models.example/v1", opener=opener)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "DINOv3"
            client.download_tree("", target)
            self.assertEqual((target / "install.json").read_bytes(), payloads["DINOv3/install.json"])
            self.assertEqual(
                (target / "source" / "dinov3" / "__init__.py").read_bytes(),
                payloads["DINOv3/source/dinov3/__init__.py"],
            )

        self.assertTrue(requests)
        self.assertTrue(all("/api/fs/" not in url for _, url, _ in requests))
        self.assertTrue(all(auth is None for _, _, auth in requests))


if __name__ == "__main__":
    unittest.main()
