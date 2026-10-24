"""Pixel processing and fail-closed model acquisition, with no network/inference."""
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from system.training.privacy import (
    MAX_DETECTIONS, MODEL_NAME, PRIVACY_CLASSES, PrivacyBlurrer, PrivacyProcessingError,
)


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))


def detections(coordinates, classes):
    return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor(coordinates), cls=tensor(classes)))]


class PrivacyBlurringTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.blurrer = PrivacyBlurrer(Path(self.temp.name))
        self.model = Mock()
        self.blurrer._model = self.model
        self.pixels = np.random.default_rng(23).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.pixels)

    def test_people_vehicles_and_screens_are_redacted_without_overlays(self):
        for class_id in (0, 2, 3, 5, 7, 62, 63, 67, 73):
            with self.subTest(class_id=class_id):
                def predict(**kwargs):
                    # The output ROI is modified after prediction; snapshot
                    # the input at call time rather than retaining its reference.
                    np.testing.assert_array_equal(kwargs['source'], self.pixels[:, :, ::-1])
                    return detections([[20, 20, 80, 80]], [class_id])
                self.model.predict.side_effect = predict
                clean, count = self.blurrer(self.image)
                actual = np.asarray(clean)
                self.assertEqual(count, 1)
                self.assertLess(actual[30:70, 30:70].std(), self.pixels[30:70, 30:70].std() / 4)
                np.testing.assert_array_equal(actual[100:, 110:], self.pixels[100:, 110:])
                # Padding also covers the detection's outer border.
                self.assertFalse(np.array_equal(actual[15:20, 25:70], self.pixels[15:20, 25:70]))
                np.testing.assert_array_equal(np.asarray(self.image), self.pixels)
                args = self.model.predict.call_args.kwargs
                self.assertEqual(args['classes'], list(PRIVACY_CLASSES))
                self.assertEqual(args['device'], 'cpu')
                self.assertFalse(args['save'])

    def test_zero_detections_preserves_pixels_and_color(self):
        self.model.predict.return_value = detections([], [])
        image, count = self.blurrer(self.image)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(np.asarray(image), self.pixels)

    def test_boxes_at_image_edges_are_clipped(self):
        self.model.predict.return_value = detections([[-10, -5, 20, 30], [140, 100, 180, 150]], [0, 2])
        clean, count = self.blurrer(self.image)
        self.assertEqual(count, 2)
        self.assertEqual(clean.size, self.image.size)

    def test_incomplete_malformed_or_saturated_detection_never_returns_pixels(self):
        invalid = [
            [], [SimpleNamespace(boxes=None)],
            detections([[0, 0, 10, 10]], []),
            detections([[0, 0, 10, 10]], [16]),
            detections([[0, 0, float('nan'), 10]], [0]),
            detections([[10, 0, 0, 10]], [0]),
            detections([[200, 200, 220, 220]], [0]),
            detections([[0, 0, 10, 10]] * MAX_DETECTIONS, [0] * MAX_DETECTIONS),
        ]
        for result in invalid:
            with self.subTest(result=str(result)[:60]):
                self.model.predict.return_value = result
                with self.assertRaises(PrivacyProcessingError):
                    self.blurrer(self.image)

    def test_runtime_error_is_sanitized_and_model_is_reloaded_on_retry(self):
        self.model.predict.side_effect = ValueError('private/path/photo.jpg')
        with self.assertRaisesRegex(PrivacyProcessingError, '^privacy_processing_unavailable$'):
            self.blurrer(self.image)
        self.assertIsNone(self.blurrer._model)
        replacement = Mock()
        replacement.predict.return_value = detections([], [])
        with patch.object(self.blurrer, '_load_model', return_value=replacement) as load:
            _, count = self.blurrer(self.image)
        load.assert_called_once()
        self.assertEqual(count, 0)

    def test_missing_runtime_is_retryable(self):
        self.blurrer._model = None
        with patch.object(self.blurrer, '_load_model', side_effect=ImportError('ultralytics')):
            with self.assertRaises(PrivacyProcessingError):
                self.blurrer(self.image)

    def test_status_polling_never_loads_runtime_or_fetches_model(self):
        with patch.object(self.blurrer, '_load_model', side_effect=AssertionError('load')), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            status = self.blurrer.status()
        self.assertTrue(status['required'])
        self.assertFalse(status['model_cached'])
        self.assertEqual(status['model'], MODEL_NAME)

    def test_official_model_requires_checksum_and_uses_atomic_cache(self):
        content = b'verified model fixture'
        digest = hashlib.sha256(content).hexdigest()
        with patch('system.training.privacy.MODEL_SHA256', digest), \
                patch('urllib.request.urlopen', return_value=io.BytesIO(content)) as download:
            path = self.blurrer._model_path()
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(self.blurrer._model_path(), path)
        download.assert_called_once()
        self.assertFalse(list(path.parent.glob('*.part')))

    def test_corrupt_cache_and_bad_download_are_never_loaded(self):
        path = Path(self.temp.name) / MODEL_NAME
        path.write_bytes(b'corrupt cache')
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong download')):
            with self.assertRaisesRegex(PrivacyProcessingError, 'checksum_mismatch'):
                self.blurrer._model_path()
        self.assertEqual(path.read_bytes(), b'corrupt cache')
        self.assertFalse(list(path.parent.glob('*.part')))

    def test_wrong_class_schema_rejected_even_if_runtime_loads(self):
        model = Mock(names={0: 'wildlife'})
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda *a, **kw: model)}), \
                patch.object(self.blurrer, '_model_path', return_value=Path('verified.pt')):
            with self.assertRaisesRegex(PrivacyProcessingError, 'classes_mismatch'):
                self.blurrer._load_model()


if __name__ == '__main__':
    unittest.main()
