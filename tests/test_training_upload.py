from __future__ import annotations

import io
import json
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

from PIL import Image

from system.training.media import prepare_sample
from system.training.policy import AGREEMENT_VERSION
from system.training.queue import TrainingQueue
from system.training.transport import UploadCancelled


def save_mpo(path, *, orientation=1):
    exif = Image.Exif()
    exif[270] = 'private camera location'
    exif[274] = orientation
    with Image.new('RGB', (80, 40), 'red') as primary:
        with Image.new('RGB', (30, 50), 'blue') as secondary:
            primary.save(path, format='MPO', save_all=True,
                         append_images=[secondary], exif=exif)


class MemoryCloud:
    def __init__(self):
        self.files = {}
        self.failures = 0
        self.lost_responses = 0
        self.on_upload = None

    def upload(self, job, photo, annotation, is_current):
        if self.on_upload:
            self.on_upload()
        if not is_current():
            raise UploadCancelled()
        if self.failures:
            self.failures -= 1
            raise OSError('offline')
        self.files[job['sample_id']] = json.loads(annotation)
        if self.lost_responses:
            self.lost_responses -= 1
            raise OSError('completion_response_lost')

    def delete(self, job, is_current):
        if not is_current():
            raise UploadCancelled()
        self.files.pop(job['sample_id'], None)


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cloud = MemoryCloud()
        self.now = 1000.0
        self.queue = self.make_queue()

    def tearDown(self):
        self.queue.stop()
        self.temp.cleanup()

    def make_queue(self):
        return TrainingQueue(self.root / 'state', transport=self.cloud,
                             clock=lambda: self.now, debounce_seconds=0)

    def enable(self):
        self.queue.set_consent(AGREEMENT_VERSION, True)

    def photo(self, name='camera/a.jpg'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (60, 40), 'green').save(path)
        return path

    def drain(self):
        for _ in range(30):
            if not self.queue.process_once():
                break

    def restore_legacy_schema(self):
        self.queue.stop()
        with closing(sqlite3.connect(self.queue.db_path)) as db:
            with db:
                db.execute('ALTER TABLE jobs DROP COLUMN remote_possible')

    def test_legacy_schema_concurrent_initialization_preserves_jobs_and_quota(self):
        self.enable()
        for i in range(3):
            self.queue.enqueue(self.photo(f'camera/empty{i}.jpg'), {'物种名称': '空'})
        self.drain()
        self.queue.enqueue(self.photo('camera/pending.jpg'), {'物种名称': '赤狐'})
        before = self.queue.status()
        self.assertEqual(before['stats']['uploaded'], 3)
        self.assertEqual(before['stats']['pending'], 1)
        empty_samples = set(self.cloud.files)
        self.restore_legacy_schema()
        start = Barrier(8)

        def initialize(_):
            start.wait(timeout=10)
            return self.make_queue()

        with ThreadPoolExecutor(8) as pool:
            queues = list(pool.map(initialize, range(8)))
        for queue in queues:
            self.assertEqual(queue.status(), before)
        self.queue = queues[0]
        self.assertFalse(self.queue.enqueue(self.photo('camera/extra.jpg'), {'物种名称': '空'}))
        self.drain()
        self.assertEqual(len(self.cloud.files), 4)
        self.assertTrue(empty_samples.issubset(self.cloud.files))
        self.assertEqual(self.queue.status()['stats']['uploaded'], 4)

    def test_legacy_schema_migration_preserves_lost_response_retraction(self):
        self.enable()
        photo = self.photo()
        self.cloud.lost_responses = 1
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        self.queue.process_once()
        self.assertEqual(len(self.cloud.files), 1)
        self.assertEqual(self.queue.status()['stats']['failed'], 1)
        self.restore_legacy_schema()
        self.queue = self.make_queue()
        self.queue.enqueue(photo, {}, validated=False)
        self.drain()
        self.assertFalse(self.cloud.files)
        self.assertEqual(self.queue.status()['stats']['failed'], 0)
        self.queue.stop()
        self.queue = self.make_queue()
        self.assertFalse(self.queue.process_once())

    def test_default_off_never_accesses_nonexistent_photo(self):
        self.assertFalse(self.queue.enqueue(self.root / 'missing.jpg', {'物种名称': '赤狐'}))
        self.assertFalse(self.queue.process_once())
        self.assertEqual(self.queue.status()['stats']['pending'], 0)
        self.assertFalse(self.cloud.files)

    def test_explicit_matching_agreement_required(self):
        with self.assertRaises(ValueError):
            self.queue.set_consent('old-version', True)
        self.assertFalse(self.queue.status()['training_enabled'])
        self.queue.set_consent(AGREEMENT_VERSION, False)
        status = self.queue.status()
        self.assertTrue(status['agreement_accepted'])
        self.assertTrue(status['participation_decided'])
        self.assertFalse(status['training_enabled'])

    def test_three_empty_photos_per_physical_folder_survive_restart(self):
        self.enable()
        for i in range(5):
            self.queue.enqueue(self.photo(f'camera/{i}.jpg'), {'物种名称': '空'})
        self.drain()
        self.assertEqual(len(self.cloud.files), 3)
        self.queue.stop()
        self.queue = self.make_queue()
        self.assertFalse(self.queue.enqueue(self.photo('camera/new.jpg'), {'物种名称': '空'}))
        self.assertTrue(self.queue.enqueue(self.photo('other/new.jpg'), {'物种名称': '空'}))
        self.drain()
        self.assertEqual(len(self.cloud.files), 4)

    def test_concurrent_empty_reservations_do_not_exceed_three(self):
        self.enable()
        photos = [self.photo(f'camera/{i}.jpg') for i in range(12)]
        with ThreadPoolExecutor(6) as pool:
            accepted = list(pool.map(lambda p: self.queue.enqueue(p, {'物种名称': '空'}), photos))
        self.assertEqual(sum(accepted), 3)
        self.drain()
        self.assertEqual(len(self.cloud.files), 3)

    def test_renames_preserve_folder_quota_and_sample_ownership(self):
        self.enable()
        for i in range(3):
            self.queue.enqueue(self.photo(f'camera/{i}.jpg'), {'物种名称': '空'})
        self.drain()
        (self.root / 'camera').rename(self.root / 'renamed')
        self.assertFalse(self.queue.enqueue(self.photo('renamed/new.jpg'), {'物种名称': '空'}))
        old = self.photo('renamed/animal.jpg')
        self.queue.enqueue(old, {'物种名称': '赤狐'})
        self.drain()
        renamed = old.with_name('new-name.jpg')
        old.rename(renamed)
        self.queue.enqueue(renamed, {'物种名称': '狼'})
        self.drain()
        self.assertEqual(len(self.cloud.files), 4)
        self.assertEqual(sum(x['species'] == ['狼'] for x in self.cloud.files.values()), 1)
        self.queue.enqueue(renamed, {}, validated=False)
        self.drain()
        self.assertEqual(len(self.cloud.files), 3)

    def test_multiple_journal_owners_share_quota_and_only_one_claims(self):
        self.enable()
        queues = [self.make_queue() for _ in range(8)]
        photos = [self.photo(f'camera/{i}.jpg') for i in range(8)]
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda pair: pair[0].enqueue(pair[1], {'物种名称': '空'}), zip(queues, photos)))
        self.assertEqual(sum(results), 3)
        claims = []
        def during_upload():
            other = self.make_queue()
            claims.append(other.process_once())
        self.cloud.on_upload = during_upload
        self.drain()
        self.assertEqual(claims, [False, False, False])
        self.assertEqual(len(self.cloud.files), 3)

    def test_species_are_unlimited_and_pending_corrections_coalesce(self):
        self.enable()
        photo = self.photo()
        self.queue.enqueue(photo, {'物种名称': '赤狐', '备注': 'private camera site'})
        self.queue.enqueue(photo, {'物种名称': '狼，狗', '物种数量': '1,2'})
        self.drain()
        sample = next(iter(self.cloud.files.values()))
        self.assertEqual(sample['species'], ['狼', '狗'])
        self.assertEqual(sample['species_counts'], [1, 2])
        self.assertNotIn('private camera site', json.dumps(sample))
        self.assertNotIn(str(self.root), json.dumps(sample))
        for i in range(6):
            self.queue.enqueue(self.photo(f'camera/species{i}.jpg'), {'物种名称': '赤狐'})
        self.drain()
        self.assertEqual(len(self.cloud.files), 7)

    def test_duplicate_marks_do_not_create_new_uploads(self):
        self.enable()
        photo = self.photo()
        self.queue.enqueue(photo, {'物种名称': '赤狐', '检测时间': 'before'})
        self.drain()
        self.assertFalse(self.queue.enqueue(photo, {'物种名称': '赤狐', '检测时间': 'after'}))
        self.assertFalse(self.queue.process_once())
        self.assertEqual(len(self.cloud.files), 1)

    def test_retry_reuses_sample_and_empty_reservation(self):
        self.enable()
        self.cloud.failures = 1
        self.queue.enqueue(self.photo(), {'物种名称': '空'})
        self.queue.process_once()
        self.assertEqual(self.queue.status()['stats']['failed'], 1)
        self.now += 4000
        self.drain()
        self.assertEqual(len(self.cloud.files), 1)
        self.assertEqual(self.queue.status()['stats']['uploaded'], 1)

    def test_withdrawal_during_upload_prevents_completion_and_clears_waiting(self):
        self.enable()
        self.queue.enqueue(self.photo(), {'物种名称': '赤狐'})
        self.queue.enqueue(self.photo('camera/b.jpg'), {'物种名称': '空'})
        self.cloud.on_upload = lambda: self.queue.set_consent(AGREEMENT_VERSION, False)
        self.queue.process_once()
        self.assertFalse(self.cloud.files)
        self.assertEqual(self.queue.status()['stats']['pending'], 0)
        self.assertFalse(self.queue.process_once())

    def test_lost_completion_response_can_be_unverified_after_restart(self):
        self.enable()
        photo = self.photo()
        self.cloud.lost_responses = 1
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        self.queue.process_once()
        self.assertEqual(len(self.cloud.files), 1)
        self.assertEqual(self.queue.status()['stats']['failed'], 1)
        self.queue.stop()
        self.queue = self.make_queue()
        self.queue.enqueue(photo, {}, validated=False)
        self.drain()
        self.assertFalse(self.cloud.files)

    def test_lost_completion_response_relabel_then_sensitive_removes_sample(self):
        self.enable()
        photo = self.photo()
        self.cloud.lost_responses = 1
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        self.queue.process_once()
        self.assertEqual(len(self.cloud.files), 1)
        self.assertEqual(self.queue.status()['stats']['failed'], 1)
        self.queue.enqueue(photo, {'物种名称': '狼'})
        self.queue.enqueue(photo, {'物种名称': '人'})
        self.drain()
        self.assertFalse(self.cloud.files)

    def test_lost_completion_response_relabel_over_empty_quota_removes_sample(self):
        self.enable()
        for i in range(3):
            self.queue.enqueue(self.photo(f'camera/empty{i}.jpg'), {'物种名称': '空'})
        self.drain()
        empty_samples = set(self.cloud.files)
        self.assertEqual(len(empty_samples), 3)
        photo = self.photo()
        self.cloud.lost_responses = 1
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        self.queue.process_once()
        self.assertEqual(len(self.cloud.files), 4)
        self.assertEqual(self.queue.status()['stats']['failed'], 1)
        self.queue.enqueue(photo, {'物种名称': '空'})
        self.drain()
        self.assertEqual(set(self.cloud.files), empty_samples)

    def test_relabel_replaces_old_sample_and_unverify_removes_it(self):
        self.enable()
        photo = self.photo()
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        self.drain()
        sample_id = next(iter(self.cloud.files))
        self.queue.enqueue(photo, {'物种名称': '狼'})
        self.drain()
        self.assertEqual(list(self.cloud.files), [sample_id])
        self.assertEqual(self.cloud.files[sample_id]['species'], ['狼'])
        self.queue.enqueue(photo, {}, validated=False)
        self.drain()
        self.assertFalse(self.cloud.files)

    def test_unknown_sensitive_and_video_inputs_are_excluded(self):
        self.enable()
        for label in ['Unknown', '未检测', '错误', '人', 'person', '车辆']:
            self.assertFalse(self.queue.enqueue(self.photo(), {'物种名称': label}))
        self.assertFalse(self.queue.enqueue(self.root / 'clip.mp4', {'物种名称': '赤狐'}))
        self.assertFalse(self.queue.enqueue(self.photo(), {'物种名称': '空', '检测框': [{'物种': 'person', '边界框': [0, 0, 1, 1]}]}))

    def test_changed_source_is_not_uploaded_with_stale_annotation(self):
        self.enable()
        photo = self.photo()
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        Image.new('RGB', (120, 80), 'red').save(photo)
        self.drain()
        self.assertFalse(self.cloud.files)

    def test_previously_skipped_mpo_can_be_reconfirmed_after_restart(self):
        self.enable()
        photo = self.photo()
        save_mpo(photo)
        self.queue.enqueue(photo, {'物种名称': '赤狐'})
        # Simulate the persisted result from the old multi-frame exclusion.
        with self.queue._connect() as db:
            db.execute("UPDATE jobs SET state='skipped',error='ValueError',empty_reserved=0")
            sample_id = db.execute('SELECT sample_id FROM jobs').fetchone()[0]
        self.queue.stop()
        self.queue = self.make_queue()
        self.assertTrue(self.queue.enqueue(photo, {'物种名称': '赤狐'}))
        self.drain()
        self.assertEqual(self.queue.status()['stats']['skipped'], 0)
        self.assertEqual(self.queue.status()['stats']['uploaded'], 1)
        self.assertEqual(list(self.cloud.files), [sample_id])
        self.assertEqual(self.cloud.files[sample_id]['image'],
                         {'format': 'jpeg', 'width': 80, 'height': 40})


class TrainingMediaTests(unittest.TestCase):
    def test_mpo_uses_only_primary_image_and_strips_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'camera.jpg'
            for orientation, size in [(1, (80, 40)), (6, (40, 80))]:
                with self.subTest(orientation=orientation):
                    save_mpo(path, orientation=orientation)
                    original = path.read_bytes()
                    photo, annotation = prepare_sample(path, {
                        'sample_id': 'c' * 32, 'revision': 1,
                        'payload': {'species': ['赤狐'], 'model_boxes': [
                            {'species': '赤狐', 'bbox': [8, 4, 40, 20], 'confidence': 0.9}
                        ]},
                    })
                    with Image.open(io.BytesIO(photo)) as image:
                        self.assertEqual(image.format, 'JPEG')
                        self.assertEqual(getattr(image, 'n_frames', 1), 1)
                        self.assertEqual(image.size, size)
                        red, green, blue = image.getpixel((10, 10))
                        self.assertGreater(red, 240)
                        self.assertLess(green, 15)
                        self.assertLess(blue, 15)
                        self.assertFalse(image.getexif())
                        self.assertNotIn('mp', image.info)
                    data = json.loads(annotation)
                    if orientation == 1:
                        self.assertEqual(data['model_predictions'][0]['bbox_xyxy_normalized'],
                                         [0.1, 0.1, 0.5, 0.5])
                    else:
                        self.assertEqual(data['model_predictions'], [])
                    self.assertEqual(path.read_bytes(), original)
                    self.assertNotIn('private camera location', annotation.decode())

    def test_animated_png_is_excluded_even_with_jpg_extension(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'animation.jpg'
            with Image.new('RGB', (40, 30), 'red') as first:
                with Image.new('RGB', (40, 30), 'blue') as second:
                    first.save(path, format='PNG', save_all=True,
                               append_images=[second], duration=100, loop=0)
            with self.assertRaisesRegex(ValueError, 'animated_image_excluded'):
                prepare_sample(path, {'sample_id': 'd' * 32, 'revision': 1,
                                      'payload': {'species': ['赤狐']}})

    def test_reencoding_strips_exif_limits_dimensions_and_preserves_source(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'personal-camera.jpg'
            exif = Image.Exif()
            exif[270] = 'private location'
            Image.new('RGB', (3000, 1500), 'blue').save(path, exif=exif)
            original = path.read_bytes()
            photo, annotation = prepare_sample(path, {
                'sample_id': 'a' * 32, 'revision': 1,
                'payload': {'species': ['赤狐'], 'species_counts': [1], 'model_boxes': [
                    {'species': '赤狐', 'bbox': [300, 150, 900, 600], 'confidence': 0.7}
                ]},
            })
            with Image.open(io.BytesIO(photo)) as image:
                self.assertEqual(image.size, (2048, 1024))
                self.assertFalse(image.getexif())
            data = json.loads(annotation)
            self.assertEqual(data['model_predictions'][0]['bbox_xyxy_normalized'], [0.1, 0.1, 0.3, 0.4])
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn('private', annotation.decode())
            self.assertNotIn(path.name, annotation.decode())

    def test_rotated_exif_image_does_not_publish_ambiguous_boxes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rotated.jpg'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (60, 40)).save(path, exif=exif)
            photo, annotation = prepare_sample(path, {
                'sample_id': 'b' * 32, 'revision': 1,
                'payload': {'species': ['狼'], 'species_counts': [], 'model_boxes': [
                    {'species': '狼', 'bbox': [0, 0, 10, 20], 'confidence': 0.5}
                ]},
            })
            with Image.open(io.BytesIO(photo)) as image:
                self.assertEqual(image.size, (40, 60))
            self.assertEqual(json.loads(annotation)['model_predictions'], [])


if __name__ == '__main__':
    unittest.main()
