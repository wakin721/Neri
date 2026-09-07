import hashlib
import tempfile
import unittest
from pathlib import Path

from system.model_sync.layout import get_model_layout
from system.model_sync.protocol import DownloadCapability, ModelManifest, ModelManifestEntry


class FakeClient:
    def __init__(self, manifest, direct_bytes=None, proxy_bytes=None, fail_direct=False, fail_proxy=False):
        self.manifest = manifest
        self.direct_bytes = direct_bytes or {}
        self.proxy_bytes = proxy_bytes or {}
        self.fail_direct = fail_direct
        self.fail_proxy = fail_proxy
        self.direct_attempts = 0
        self.proxy_attempts = 0

    def fetch_manifest(self):
        return self.manifest

    def request_download(self, manifest_id, entry):
        return DownloadCapability(
            entry.path, entry.size, entry.sha256,
            'https://foo.1drv.com/' + entry.path.replace('/', '_'), {},
            'A' * 43, 9999999999.0,
        )

    def proxy_url(self, token):
        return 'https://myneri.top/api/models/v1/proxy/' + token

    def download_to(self, url, headers, target, on_progress=None):
        is_proxy = '/proxy/' in url
        if is_proxy:
            self.proxy_attempts += 1
            if self.fail_proxy:
                raise RuntimeError('proxy fail')
            data = next(iter(self.proxy_bytes.values())) if self.proxy_bytes else b''
        else:
            self.direct_attempts += 1
            if self.fail_direct:
                raise RuntimeError('direct fail')
            data = next(iter(self.direct_bytes.values())) if self.direct_bytes else b''
        Path(target).write_bytes(data)
        if on_progress:
            on_progress(len(data), len(data))


def entry(path, data):
    return ModelManifestEntry(path, len(data), hashlib.sha256(data).hexdigest())


class ManagerTests(unittest.TestCase):
    def test_direct_success_atomically_replaces_target(self):
        from system.model_sync.manager import ModelSyncManager
        data = b'new-model'
        manifest = ModelManifest('a' * 64, (entry('detect/a.pt', data),))
        with tempfile.TemporaryDirectory() as td:
            layout = get_model_layout(Path(td) / 'res')
            target = layout.detect_sync / 'a.pt'
            target.write_bytes(b'old')
            manager = ModelSyncManager(layout, FakeClient(manifest, direct_bytes={'detect/a.pt': data}))
            status = manager.sync_once_for_test()
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(status.state, 'completed')
            self.assertFalse(Path(str(target) + '.part').exists())

    def test_direct_failure_falls_back_to_proxy(self):
        from system.model_sync.manager import ModelSyncManager
        data = b'proxy-model'
        manifest = ModelManifest('b' * 64, (entry('detect/a.pt', data),))
        client = FakeClient(manifest, proxy_bytes={'detect/a.pt': data}, fail_direct=True)
        with tempfile.TemporaryDirectory() as td:
            layout = get_model_layout(Path(td) / 'res')
            status = ModelSyncManager(layout, client).sync_once_for_test()
            self.assertEqual((layout.detect_sync / 'a.pt').read_bytes(), data)
            self.assertEqual(client.direct_attempts, 1)
            self.assertEqual(client.proxy_attempts, 1)
            self.assertEqual(status.state, 'completed')

    def test_dual_failure_preserves_old_and_skips_all_stale_deletion(self):
        from system.model_sync.manager import ModelSyncManager
        wanted = b'new'
        manifest = ModelManifest('c' * 64, (entry('detect/a.pt', wanted),))
        client = FakeClient(manifest, fail_direct=True, fail_proxy=True)
        with tempfile.TemporaryDirectory() as td:
            layout = get_model_layout(Path(td) / 'res')
            target = layout.detect_sync / 'a.pt'; target.write_bytes(b'old')
            stale = layout.cls_sync / 'stale.pt'; stale.write_bytes(b'stale')
            user = layout.detect_user / 'mine.pt'; user.write_bytes(b'user')
            status = ModelSyncManager(layout, client).sync_once_for_test()
            self.assertEqual(target.read_bytes(), b'old')
            self.assertTrue(stale.exists())
            self.assertTrue(user.exists())
            self.assertEqual(status.state, 'failed')

    def test_success_removes_only_stale_sync_and_retains_missing_tracker(self):
        from system.model_sync.manager import ModelSyncManager
        manifest = ModelManifest('d' * 64, ())
        with tempfile.TemporaryDirectory() as td:
            layout = get_model_layout(Path(td) / 'res')
            stale = layout.detect_sync / 'stale.pt'; stale.write_bytes(b'stale')
            user = layout.detect_user / 'mine.pt'; user.write_bytes(b'user')
            layout.tracker.write_text('old tracker')
            status = ModelSyncManager(layout, FakeClient(manifest)).sync_once_for_test()
            self.assertFalse(stale.exists())
            self.assertTrue(user.exists())
            self.assertEqual(layout.tracker.read_text(), 'old tracker')
            self.assertEqual(status.state, 'completed')


if __name__ == '__main__':
    unittest.main()
