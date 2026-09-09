import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
