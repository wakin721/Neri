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


if __name__ == "__main__":
    unittest.main()
