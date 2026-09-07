import os
import unittest
from unittest.mock import patch

from system.utils import resource_path


class ModelResourcePathTests(unittest.TestCase):
    def _resolve(self, relative):
        with patch('system.utils.os.path.abspath', return_value=os.path.normpath('/app')):
            return os.path.normpath(resource_path(relative))

    def test_legacy_tracker_resource_alias_points_to_canonical_tracker(self):
        self.assertEqual(
            self._resolve(os.path.join('res', 'model_cls', 'tracker.yaml')),
            os.path.normpath('/app/res/Model/tracker.yaml'),
        )

    def test_legacy_detection_root_alias_points_to_canonical_detect_root(self):
        self.assertEqual(
            self._resolve(os.path.join('res', 'model')),
            os.path.normpath('/app/res/Model/detect'),
        )

    def test_legacy_classification_root_alias_points_to_canonical_cls_root(self):
        self.assertEqual(
            self._resolve(os.path.join('res', 'model_cls')),
            os.path.normpath('/app/res/Model/cls'),
        )


if __name__ == '__main__':
    unittest.main()
