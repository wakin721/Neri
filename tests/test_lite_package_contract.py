import os
import subprocess
import sys
import textwrap
import tempfile
import unittest
from pathlib import Path

from scripts.package_litert_resources import copy_resources, verify_resources, CANONICAL_DIRECTORIES


class LitePackageContractTests(unittest.TestCase):
    def test_lite_model_catalog_and_component_api_import_without_torch(self):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
            import importlib.abc, sys, tempfile
            from pathlib import Path
            attempts=[]
            class MissingTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname=='torch' or fullname.startswith('torch.'):
                        attempts.append(fullname)
                        raise ModuleNotFoundError('PyTorch is optional', name='torch')
            sys.meta_path.insert(0,MissingTorch())
            from system.model_sync.catalog import discover_models
            from system.model_sync.layout import get_model_layout
            from system.dinov3.component import dinov3_component_status
            from system.backend.dinov3_environment import dinov3_environment_router
            with tempfile.TemporaryDirectory() as temp:
                layout=get_model_layout(Path(temp)/'res')
                (layout.detect_sync/'fixture.tflite').write_bytes(b'fixture')
                assert any(model.backend=='litert' for model in discover_models(layout,'detect'))
                assert not dinov3_component_status(root=layout.dinov3_root)['installed']
                assert dinov3_environment_router()
            assert attempts==[],attempts
        ''')], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_resources_include_litert_and_matching_sidecars_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            files = {
                'translate.json': '{}', 'demo/help.md': 'help',
                'model/tracker.yaml': 'tracker_type: botsort',
                'model/detect/sync/detector.tflite': 'LiteRT model',
                'model/detect/sync/detector.labels.txt': 'animal',
                'model/detect/sync/detector.LICENSE.md': 'license',
                'model/detect/sync/detector.SOURCE.md': 'source',
                'model/detect/user/another.litert': 'another LiteRT model',
                'model/detect/user/legacy.pt': 'legacy',
                'model/cls/user/classifier.onnx': 'classifier',
                'model/cls/user/experiment.neri.json': '{}',
                'model/cls/user/assets/encoder.pth': 'encoder',
                'model/cls/user/assets/model.neri-model': 'classifier head',
                'model/cls/user/assets/calibration.npz': 'calibration',
                'model/DINOv3/weights.safetensors': 'component',
                'model/detect/sync/legacy.labels.txt': 'stale labels',
                'Model/old.engine': 'uppercase legacy',
                'model_cls/old.pt': 'old directory',
                'demo/downloaded-model.zip': 'bundled archive',
            }
            for relative, contents in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, 'utf-8')
            result = copy_resources(source, output)
            self.assertEqual(result['policy'], 'litert-only')
            self.assertEqual(len(result['models']), 2)
            self.assertTrue((output/'translate.json').is_file())
            self.assertTrue((output/'demo/help.md').is_file())
            self.assertTrue((output/'model/tracker.yaml').is_file())
            self.assertTrue((output/'model/detect/sync/detector.LICENSE.md').is_file())
            self.assertFalse((output/'model/detect/sync/legacy.labels.txt').exists())
            self.assertFalse((output/'model/DINOv3').exists())
            self.assertNotIn('Model', [path.name for path in output.iterdir()])
            self.assertFalse((output/'model/old.engine').exists())
            self.assertFalse((output/'model_cls').exists())
            self.assertFalse((output/'demo/downloaded-model.zip').exists())
            for directory in CANONICAL_DIRECTORIES:
                self.assertTrue((output/'model'/directory).is_dir())
            for relative, contents in files.items():
                self.assertEqual((source/relative).read_text('utf-8'), contents)
            self.assertEqual(verify_resources(output), result)

    def test_model_verifier_rejects_unexpected_weights_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            model = source/'model/detect/sync/lite.tflite'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'LiteRT model')
            copy_resources(source, output)
            extra = output/'model/cls/user/unexpected.pt'
            extra.write_bytes(b'forbidden weights')
            with self.assertRaisesRegex(ValueError, 'Non-LiteRT'):
                verify_resources(output)
            extra.unlink()
            (output/'model/detect/sync/lite.tflite').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_resources(output)

    def test_resource_packager_does_not_overwrite_existing_user_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            source.mkdir()
            sentinel = output/'model/cls/user/user.pt'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'keep user model')
            with self.assertRaises(FileExistsError):
                copy_resources(source, output)
            self.assertEqual(sentinel.read_bytes(), b'keep user model')

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
        self.assertIn('package_litert_resources.py', package_script)
        self.assertNotIn('Copy-Item "res" $package -Recurse -Force', package_script)


if __name__ == "__main__":
    unittest.main()
