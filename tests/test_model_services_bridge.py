import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from system.model_sync.layout import get_model_layout
from system.backend.preview_scan import PreviewScanCoordinator


class ModelServicesBridgeTests(unittest.TestCase):
    def test_new_preview_generation_supersedes_previous_scan(self):
        coordinator = PreviewScanCoordinator()

        first = coordinator.begin()
        second = coordinator.begin()

        self.assertFalse(coordinator.is_current(first))
        self.assertTrue(coordinator.is_current(second))

    def test_preview_enumeration_stops_when_cancelled(self):
        from system.backend.services_legacy import _resolve_supported_inputs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(10):
                (root / f"image-{index}.jpg").write_bytes(b"image")

            checks = 0

            def cancelled():
                nonlocal checks
                checks += 1
                return checks >= 4

            with self.assertRaisesRegex(RuntimeError, "用户取消任务"):
                list(_resolve_supported_inputs(root, cancelled=cancelled))

            self.assertLess(checks, 10)

    def test_lists_user_then_sync_with_source_metadata(self):
        from system.backend import model_services

        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_user / "bird.pt").write_bytes(b"user")
            (layout.detect_sync / "bird.pt").write_bytes(b"sync")
            with patch.object(model_services, "get_model_layout", return_value=layout):
                models = model_services.list_available_models()
                self.assertEqual([model.source for model in models], ["user", "sync"])
                self.assertEqual([model.kind for model in models], ["detect", "detect"])
                self.assertEqual([model.name for model in models], ["bird.pt", "bird.pt"])
                self.assertNotEqual(models[0].path, models[1].path)
                self.assertEqual(model_services.model_directory(), layout.root / "detect")

    def test_lists_classification_extensions(self):
        from system.backend import model_services

        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.cls_user / "a.onnx").write_bytes(b"a")
            (layout.cls_sync / "b.engine").write_bytes(b"b")
            with patch.object(model_services, "get_model_layout", return_value=layout):
                models = model_services.list_available_classification_models()
                self.assertEqual([model.name for model in models], ["a.onnx", "b.engine"])
                self.assertEqual([model.kind for model in models], ["cls", "cls"])
                self.assertEqual(model_services.classification_model_directory(), layout.root / "cls")

    def test_backend_api_uses_canonical_model_services_bridge(self):
        from system.backend import main_core, model_services

        self.assertIs(main_core.model_directory, model_services.model_directory)
        self.assertIs(
            main_core.classification_model_directory,
            model_services.classification_model_directory,
        )
        self.assertIs(main_core.list_available_models, model_services.list_available_models)
        self.assertIs(
            main_core.list_available_classification_models,
            model_services.list_available_classification_models,
        )
        self.assertIn(
            "cancelled",
            inspect.signature(main_core.preview_media_items).parameters,
        )


if __name__ == "__main__":
    unittest.main()
