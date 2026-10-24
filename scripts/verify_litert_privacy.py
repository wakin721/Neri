"""Verify the privacy pipeline in a clean environment without PyTorch.

Default: synthetic photos and an in-memory transport. --live sends ONLY the
generated synthetic photos to the configured training broker, then deletes them.
An optional reference image is processed locally and is never enqueued/uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from system.training.policy import AGREEMENT_VERSION
from system.training.privacy import MODEL_NAME, PrivacyBlurrer
from system.training.queue import TrainingQueue
from system.training.transport import HttpTransport, UploadCancelled


class MemoryTransport:
    def __init__(self):
        self.files = {}

    def upload(self, job, photo, annotation, is_current):
        if not is_current():
            raise UploadCancelled()
        self.files[job['sample_id']] = photo

    def delete(self, job, is_current):
        self.files.pop(job['sample_id'], None)


class CountingHttpTransport(HttpTransport):
    def __init__(self):
        super().__init__()
        self.put_count = 0

    def _request(self, url, method, body=None, headers=None):
        if method == 'PUT':
            self.put_count += 1
        return super()._request(url, method, body, headers)


class RecordingTransport:
    def __init__(self, inner):
        self.inner = inner
        self.prepared = {}

    def upload(self, job, photo, annotation, is_current):
        package = json.loads(annotation)
        assert package['privacy_processing'] == 'metadata_removed_local_litert_blurring_v1'
        with Image.open(io.BytesIO(photo)) as image:
            assert not image.getexif()
        assert all(label.encode() and (label.strip() or label == '\n')
                   for label in package['yolo_labels'].values())
        self.prepared[job['sample_id']] = (photo, annotation)
        self.inner.upload(job, photo, annotation, is_current)

    def delete(self, job, is_current):
        self.inner.delete(job, is_current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-image', type=Path)
    parser.add_argument('--live', action='store_true', help='upload only synthetic fixtures and delete them')
    args = parser.parse_args()
    forbidden = ('torch', 'torchvision', 'ultralytics', 'tensorflow')
    assert all(importlib.util.find_spec(name) is None for name in forbidden), 'Use a clean runtime environment'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'model': MODEL_NAME, 'forbidden_packages_available': False,
              'transport': 'live' if args.live else 'memory', 'cleanup_errors': []}
    processor = PrivacyBlurrer(args.cache_dir)
    if args.reference_image:
        original = args.reference_image.read_bytes()
        with Image.open(args.reference_image) as source:
            started = time.perf_counter()
            redacted, regions = processor(source.convert('RGB'))
            report['reference_seconds'] = round(time.perf_counter() - started, 3)
        redacted.save(args.output_dir / 'reference-redacted.jpg')
        assert args.reference_image.read_bytes() == original
        report['reference_blurred_regions'] = regions
    inner = CountingHttpTransport() if args.live else MemoryTransport()
    transport = RecordingTransport(inner)
    queue = TrainingQueue(args.output_dir / 'private-state', transport=transport,
                          debounce_seconds=0, privacy_blurrer=processor)
    jobs = []
    try:
        queue.set_consent(AGREEMENT_VERSION, True)  # Isolated TEST state only.
        report['test_device_id'] = queue.status()['device_id']
        hashes = {}
        for index in (1, 2):
            path = args.output_dir / f'synthetic-{index}.jpg'
            image = Image.new('RGB', (320, 200), (30, 90, 70))
            drawing = ImageDraw.Draw(image)
            drawing.rectangle((20, 20, 140, 160), fill=(160, 50, 80))
            drawing.ellipse((160, 30, 290, 170), fill=(20, 140, 170))
            drawing.text((15, 180), f'NERI LITERT SYNTHETIC TEST {index}', fill='white')
            exif = Image.Exif()
            exif[270] = 'Synthetic-only metadata to remove'
            image.save(path, exif=exif)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = {'物种名称': 'NeriLiteRT自测', '检测框': [
                {'species': 'NeriLiteRT自测', 'bbox': [20, 20, 140, 160], 'confidence': 1}
            ]} if index == 1 else {'物种名称': '空'}
            assert queue.enqueue(path, data)
        with queue._connect() as database:
            jobs = [dict(row) for row in database.execute('SELECT * FROM jobs')]
        for _ in jobs:
            assert queue.process_once()
        report['status'] = queue.status()
        report['source_files_unchanged'] = all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
        assert report['source_files_unchanged']
        assert report['status']['stats']['uploaded'] == 2, 'Inspect private test state for failed attempts'
        if args.live:
            report['file_puts'] = inner.put_count
            assert inner.put_count == 6
            for job in jobs:
                job['payload'] = json.loads(job['payload'])
                job.update(device_id=queue.device_id, device_secret=queue._device_secret)
                photo, annotation = transport.prepared[job['sample_id']]
                before = inner.put_count
                inner.upload(job, photo, annotation, lambda: True)
                assert inner.put_count == before
            report['completed_retry_without_resend'] = True
        assert not any(name.split('.')[0] in forbidden for name in sys.modules)
        report['forbidden_packages_loaded'] = False
    except Exception as error:
        report['error'] = type(error).__name__
        raise
    finally:
        deleted = 0
        for job in jobs:
            try:
                transport.delete(job, lambda: True)
                deleted += 1
            except Exception as error:
                report['cleanup_errors'].append(type(error).__name__)
        report['deleted_test_samples'] = deleted
        queue.stop()
        # No capabilities, secrets or reference/source paths in the public report.
        (args.output_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if report['cleanup_errors']:
        raise RuntimeError('Test cleanup needs retry; keep the private-state directory')


if __name__ == '__main__':
    main()
