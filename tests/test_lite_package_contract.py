import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class LitePackageContractTests(unittest.TestCase):
    def test_pruner_removes_pytorch_but_preserves_unrelated_packages(self):
        repository_root = Path(__file__).resolve().parents[1]
        prune_script = repository_root / "scripts" / "prune_lite_python.ps1"
        self.assertTrue(
            prune_script.is_file(),
            "lite packaging must provide scripts/prune_lite_python.ps1",
        )
        if os.name != "nt":
            self.skipTest("PowerShell packaging regression runs on Windows")

        with tempfile.TemporaryDirectory() as tmp:
            toolkit = Path(tmp) / "toolkit"
            site_packages = toolkit / "Lib" / "site-packages"
            site_packages.mkdir(parents=True)

            pytorch_artifacts = (
                "torch",
                "torchvision",
                "torchvision.libs",
                "torchaudio",
                "torchaudio.lib",
                "torchaudio.libs",
                "functorch",
                "torchgen",
                "torch-2.13.0+cu130.dist-info",
                "torchvision-0.24.0+cu130.dist-info",
                "torchaudio-2.13.0+cu130.dist-info",
            )
            for name in pytorch_artifacts:
                artifact = site_packages / name
                artifact.mkdir(parents=True)
                (artifact / "sentinel.txt").write_text("remove", encoding="utf-8")

            preserved = site_packages / "fastapi"
            preserved.mkdir()
            (preserved / "sentinel.txt").write_text("keep", encoding="utf-8")

            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(prune_script),
                    "-ToolkitPath",
                    str(toolkit),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            for name in pytorch_artifacts:
                self.assertFalse((site_packages / name).exists(), name)
            self.assertEqual(
                (preserved / "sentinel.txt").read_text(encoding="utf-8"),
                "keep",
            )

    def test_package_windows_runs_pruner_after_copying_python(self):
        repository_root = Path(__file__).resolve().parents[1]
        package_script = (repository_root / "scripts" / "package_windows.ps1").read_text(
            encoding="utf-8"
        )
        copy_marker = 'Copy-Item "$pythonRoot\\*" $toolkit -Recurse -Force'
        prune_marker = 'prune_lite_python.ps1'
        self.assertIn(copy_marker, package_script)
        self.assertIn(prune_marker, package_script)
        self.assertGreater(
            package_script.index(prune_marker),
            package_script.index(copy_marker),
            "PyTorch pruning must happen after the build Python environment is copied",
        )


if __name__ == "__main__":
    unittest.main()
