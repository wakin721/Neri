import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from PIL import Image

from system.backend import services
from system.backend.models import ValidationBatchMarkRequest
from system.training.api import privacy_router
from system.training.policy import AGREEMENT_VERSION
from system.training.queue import TrainingQueue


async def asgi_request(app, method, path, body=None, origin=None):
    payload = json.dumps(body).encode() if body is not None else b''
    headers = [(b'content-type', b'application/json'), (b'host', b'localhost')]
    if origin:
        headers.append((b'origin', origin.encode()))
    messages = []
    async def receive():
        return {'type': 'http.request', 'body': payload, 'more_body': False}
    async def send(message):
        messages.append(message)
    await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
               'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
               'query_string': b'', 'root_path': '', 'headers': headers,
               'client': ('127.0.0.1', 5555), 'server': ('127.0.0.1', 721)}, receive, send)
    status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
    content = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    return status, json.loads(content) if content else None


class PrivacyApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.queue = TrainingQueue(Path(self.temp.name) / 'state')
        self.app = FastAPI()
        self.app.include_router(privacy_router(lambda: self.queue))

    def tearDown(self):
        self.queue.stop()
        self.temp.cleanup()

    def request(self, method, body=None, origin=None):
        return asyncio.run(asgi_request(self.app, method, '/api/privacy', body, origin))

    def test_first_use_and_declining_participation(self):
        code, status = self.request('GET')
        self.assertEqual(code, 200)
        self.assertFalse(status['agreement_accepted'])
        code, status = self.request('PUT', {'agreement_version': AGREEMENT_VERSION, 'training_enabled': False})
        self.assertEqual(code, 200)
        self.assertTrue(status['participation_decided'])
        self.assertFalse(status['training_enabled'])

    def test_cannot_grant_consent_from_a_website_or_ambiguous_value(self):
        code, _ = self.request('PUT', {'agreement_version': AGREEMENT_VERSION, 'training_enabled': True}, 'https://untrusted.example')
        self.assertEqual(code, 403)
        code, _ = self.request('PUT', {'agreement_version': AGREEMENT_VERSION, 'training_enabled': 'false'})
        self.assertEqual(code, 422)
        self.assertFalse(self.queue.status()['training_enabled'])

    def test_stale_version_rejected_and_disable_persists(self):
        code, _ = self.request('PUT', {'agreement_version': 'old', 'training_enabled': True})
        self.assertEqual(code, 400)
        self.request('PUT', {'agreement_version': AGREEMENT_VERSION, 'training_enabled': True})
        code, status = self.request('PUT', {'agreement_version': AGREEMENT_VERSION, 'training_enabled': False})
        self.assertEqual(code, 200)
        self.assertFalse(status['training_enabled'])

    def test_saved_annotation_is_enqueued_and_failed_local_save_is_not(self):
        self.queue.set_consent(AGREEMENT_VERSION, True)
        photo = Path(self.temp.name) / 'animal.jpg'
        Image.new('RGB', (40, 30), 'green').save(photo)
        request = ValidationBatchMarkRequest(input_path=self.temp.name, file_paths=[str(photo)], species_name='赤狐')
        with patch('system.training.get_queue', return_value=self.queue):
            results = services.mark_validation_items(request)
        self.assertTrue(results[0].validated)
        self.assertEqual(self.queue.status()['stats']['pending'], 1)
        self.queue.clear_pending()
        with patch('system.training.get_queue', return_value=self.queue), patch.object(services, '_persist_validation_updates', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                services.mark_validation_items(request)
        self.assertEqual(self.queue.status()['stats']['pending'], 0)


if __name__ == '__main__':
    unittest.main()
