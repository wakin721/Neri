import hashlib
import json
import zipfile

import pytest
from threadpoolctl import threadpool_limits

from scripts.package_dinov3_preview import package
from system.dinov3.deployment import publish_model
from system.dinov3.experiment import PREVIEW_DISPLAY_NAME
from system.dinov3.runtime import read_manifest, resolve_manifest_asset
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_experiment import experiment


def prepare(experiment):
    manifest, model, _, calibration = experiment
    with threadpool_limits(limits=1):
        publish_model(model, manifest, existing_manifest=read_manifest(manifest),
                      calibration_data=calibration, display_name=PREVIEW_DISPLAY_NAME)
    return manifest


def test_preview_bundle_is_portable_and_preserves_all_asset_hashes(experiment, tmp_path):
    manifest = prepare(experiment)
    original = read_manifest(manifest)
    output = tmp_path / "Neri1.2preview.zip"
    report = package(manifest, output)
    assert report["passed"] and report["known_species"] == 2
    with zipfile.ZipFile(output) as archive:
        prefix = PREVIEW_DISPLAY_NAME + "/"
        exported = json.loads(archive.read(prefix+PREVIEW_DISPLAY_NAME+".neri.json"))
        assert exported["display_name"] == PREVIEW_DISPLAY_NAME
        for key, field in (("checkpoint", "checkpoint_sha256"),
                           ("encoder_weights", "encoder_sha256"),
                           ("calibration", "calibration_sha256")):
            assert hashlib.sha256(archive.read(prefix+exported[key])).hexdigest() == original[field]
        assert archive.read(prefix+"assets/encoder-code/LICENSE.md")
        assert archive.read(prefix+"assets/encoder-code/dinov3/hub/backbones.py")
        inventory = json.loads(archive.read(prefix+"package_inventory.json"))
        assert len(inventory["files"]) == 6
        assert original["calibration_sha256"] == inventory["calibration_sha256"]
    assert output.with_suffix('.zip.sha256').read_text().split()[0] == report["archive_sha256"]
    assert read_manifest(manifest) == original


def test_preview_packaging_rejects_corrupt_assets_and_existing_outputs(experiment, tmp_path):
    manifest = prepare(experiment)
    output = tmp_path / "preview.zip"
    output.write_bytes(b'keep existing artifact')
    with pytest.raises(FileExistsError):
        package(manifest, output)
    assert output.read_bytes() == b'keep existing artifact'
    calibration = resolve_manifest_asset(manifest, read_manifest(manifest)["calibration"])
    calibration.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='digest'):
        package(manifest, tmp_path/'new.zip')
    assert not (tmp_path/'new.zip').exists()
