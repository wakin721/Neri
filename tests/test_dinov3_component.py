import tempfile
import unittest
from pathlib import Path

import system.dinov3 as dinov3
from system.model_sync.layout import get_model_layout


class DinoV3ComponentLayoutTests(unittest.TestCase):
    def test_model_layout_exposes_dinov3_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resource_root = Path(tmp).resolve()
            layout = get_model_layout(resource_root)
            self.assertEqual(
                layout.dinov3_root,
                resource_root / "model" / "DINOv3",
            )

    def test_dinov3_package_exposes_component_status_api(self) -> None:
        self.assertTrue(hasattr(dinov3, "dinov3_component_status"))

    def test_existing_incomplete_component_is_installed_but_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            root.mkdir()
            status = dinov3.dinov3_component_status(root=root)
            self.assertTrue(status["installed"])
            self.assertFalse(status["healthy"])


if __name__ == "__main__":
    unittest.main()
