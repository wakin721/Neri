"""Regression tests for the pinned LiteRT model's raw YOLO head and geometry."""
import unittest

import numpy as np

from system.training.litert_detector import ANCHOR_COUNT, CLASS_COUNT, decode_detections, letterbox


def head(rows):
    result = np.zeros((1, CLASS_COUNT + 4, ANCHOR_COUNT), dtype=np.float32)
    for index, (box, class_id, confidence) in enumerate(rows):
        result[0, :4, index] = box
        result[0, 4 + class_id, index] = confidence
    return result


class LiteRtDecodingTests(unittest.TestCase):
    def test_letterbox_preserves_rgb_and_records_exact_padding(self):
        rgb = np.full((320, 640, 3), (255, 0, 64), dtype=np.uint8)
        tensor, geometry = letterbox(rgb)
        self.assertEqual(tensor.shape, (1, 640, 640, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(geometry, (0, 160, 1, 1))
        np.testing.assert_allclose(tensor[0, 10, 10], [114 / 255] * 3)
        np.testing.assert_allclose(tensor[0, 200, 10], [1, 0, 64 / 255])

    def test_nms_class_filtering_and_source_coordinates(self):
        output = head([
            ((0.5, 0.5, 0.5, 0.25), 0, 0.9),
            ((0.5, 0.5, 0.5, 0.25), 0, 0.8),
            ((0.5, 0.5, 0.5, 0.25), 2, 0.8),
            ((0.5, 0.5, 0.5, 0.25), 16, 0.95),
            ((0.5, 0.5, 0.5, 0.25), 0, 0.14),
        ])
        output[0, 4, 3] = 0.2  # A stronger animal class must win over "person".
        result = decode_detections(output, (0, 160, 0.5, 0.5), (0, 2), 1000)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[:, :4], [[320, 160, 960, 480]] * 2)
        self.assertEqual(set(result[:, 5]), {0, 2})

    def test_odd_letterbox_geometry_is_reversible(self):
        _, geometry = letterbox(np.zeros((333, 1000, 3), np.uint8))
        left, top, gx, gy = geometry
        source = np.array([100, 30, 700, 300], dtype=np.float32)
        model = source * [gx, gy, gx, gy] + [left, top, left, top]
        center = (model[:2] + model[2:]) / 2 / 640
        size = (model[2:] - model[:2]) / 640
        output = head([(tuple(center) + tuple(size), 0, 0.9)])
        result = decode_detections(output, geometry, (0,), 1000)
        np.testing.assert_allclose(result[0, :4], source, atol=0.001)

    def test_empty_detections_are_a_valid_complete_result(self):
        result = decode_detections(head([]), (0, 0, 1, 1), (0,), 1000)
        self.assertEqual(result.shape, (0, 6))

    def test_malformed_or_nonfinite_outputs_fail_closed(self):
        invalid = [np.zeros((1, 6, 300), np.float32), head([]).astype(np.float64)]
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            output = head([])
            output[0, 4, 0] = value
            invalid.append(output)
        invalid.append(head([((0.5, 0.5, -0.1, 0.2), 0, 0.9)]))
        for output in invalid:
            with self.subTest(shape=output.shape), self.assertRaises(ValueError):
                decode_detections(output, (0, 0, 1, 1), (0,), 1000)

    def test_saturated_result_is_never_silently_truncated(self):
        output = head([((0.2, 0.2, 0.1, 0.1), 0, 0.9), ((0.8, 0.8, 0.1, 0.1), 0, 0.9)])
        with self.assertRaisesRegex(ValueError, 'saturated'):
            decode_detections(output, (0, 0, 1, 1), (0,), 2)


if __name__ == '__main__':
    unittest.main()
