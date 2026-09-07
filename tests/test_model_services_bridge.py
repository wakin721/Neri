import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from system.model_sync.layout import get_model_layout


class ModelServicesBridgeTests(unittest.TestCase):
    def test_lists_user_then_sync_with_source_metadata(self):
        from system.backend import services

        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_user / "bird.pt").write_bytes(b"user")
            (layout.detect_sync / "bird.pt").write_bytes(b"sync")
            with patch.object(services, "get_model_layout", return_value=layout):
                models = services.list_available_models()

            self.assertEqual([model.source for model in models], ["user", "sync"])
            self.assertEqual([model.kind for model in models], ["detect", "detect"])
            self.assertEqual([model.name for model in models], ["bird.pt", "bird.pt"])
            self.assertNotEqual(models[0].path, models[1].path)
            self.assertEqual(services.model_directory(), layout.root / "detect")

    def test_lists_classification_extensions(self):
        from system.backend import services

        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.cls_user / "a.onnx").write_bytes(b"a")
            (layout.cls_sync / "b.engine").write_bytes(b"b")
            with patch.object(services, "get_model_layout", return_value=layout):
                models = services.list_available_classification_models()

            self.assertEqual([model.name for model in models], ["a.onnx", "b.engine"])
            self.assertEqual([model.kind for model in models], ["cls", "cls"])
            self.assertEqual(services.classification_model_directory(), layout.root / "cls")


if __name__ == "__main__":
    unittest.main()
