import json
import tempfile
import unittest
from pathlib import Path

from system.model_sync.protocol import ProtocolError, parse_capability, parse_manifest


class ProtocolTests(unittest.TestCase):
    def test_parses_valid_manifest(self):
        manifest = parse_manifest({
            'schema_version': 1,
            'manifest_id': 'a' * 64,
            'files': [
                {'path': 'detect/a.pt', 'size': 3, 'sha256': 'b' * 64},
                {'path': 'cls/c.onnx', 'size': 4, 'sha256': 'c' * 64},
                {'path': 'tracker.yaml', 'size': 5, 'sha256': 'd' * 64},
            ],
        })
        self.assertEqual(manifest.files[0].path, 'detect/a.pt')
        self.assertEqual(len(manifest.files), 3)

    def test_rejects_invalid_or_duplicate_paths(self):
        base = {'schema_version': 1, 'manifest_id': 'a' * 64}
        for path in ['../a.pt', '/abs.pt', r'detect\\a.pt', 'detect/a.onnx', 'cls/readme.txt']:
            with self.subTest(path=path), self.assertRaises(ProtocolError):
                parse_manifest({**base, 'files': [{'path': path, 'size': 1, 'sha256': 'b' * 64}]})
        with self.assertRaises(ProtocolError):
            parse_manifest({**base, 'files': [
                {'path': 'detect/A.pt', 'size': 1, 'sha256': 'b' * 64},
                {'path': 'detect/a.pt', 'size': 1, 'sha256': 'c' * 64},
            ]})

    def test_discards_invalid_direct_url_but_keeps_proxy_token(self):
        capability = parse_capability({
            'path': 'detect/a.pt',
            'size': 3,
            'sha256': 'b' * 64,
            'direct_url': 'https://evil.example/a',
            'direct_headers': {'Authorization': 'secret', 'User-Agent': 'ua'},
            'proxy_token': 'A' * 43,
            'expires_at': 1000.0,
        })
        self.assertIsNone(capability.direct_url)
        self.assertEqual(capability.direct_headers, {})
        self.assertEqual(capability.proxy_token, 'A' * 43)


class _FakeResponse:
    def __init__(self, body=b'', chunks=None):
        self.body = body
        self.pos = 0
        self.chunks = list(chunks or [])
        self.headers = {'Content-Length': str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if self.chunks:
            return self.chunks.pop(0)
        if self.pos >= len(self.body):
            return b''
        if size < 0:
            size = len(self.body) - self.pos
        result = self.body[self.pos:self.pos + size]
        self.pos += len(result)
        return result


class ClientTests(unittest.TestCase):
    def test_fetches_manifest_and_builds_proxy_url(self):
        from system.model_sync.client import ModelDistributionClient
        payload = json.dumps({'schema_version': 1, 'manifest_id': 'a' * 64, 'files': []}).encode()
        client = ModelDistributionClient(
            base_url='https://myneri.top/api/models/v1',
            opener=lambda request, timeout=0: _FakeResponse(payload),
        )
        self.assertEqual(client.fetch_manifest().manifest_id, 'a' * 64)
        self.assertEqual(
            client.proxy_url('A' * 43),
            'https://myneri.top/api/models/v1/proxy/' + 'A' * 43,
        )

    def test_download_to_streams_and_reports_progress(self):
        from system.model_sync.client import ModelDistributionClient
        progress = []
        client = ModelDistributionClient(
            opener=lambda request, timeout=0: _FakeResponse(chunks=[b'ab', b'cd', b''])
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'x.part'
            client.download_to(
                'https://foo.1drv.com/a',
                {},
                target,
                lambda got, total: progress.append((got, total)),
            )
            self.assertEqual(target.read_bytes(), b'abcd')
        self.assertEqual(progress[-1][0], 4)


class RedirectSafetyTests(unittest.TestCase):
    def test_direct_redirect_handler_rejects_non_microsoft_hop(self):
        import urllib.error
        from system.model_sync.client import SafeDirectRedirectHandler

        handler = SafeDirectRedirectHandler()
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(
                request=None,
                fp=None,
                code=302,
                msg='Found',
                headers={},
                newurl='https://evil.example/model.pt',
            )


if __name__ == '__main__':
    unittest.main()
