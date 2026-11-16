"""Classifier weights are excluded from lite packages while explicit local shard assembly remains available."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_bundled_classifier import FILES, NAME, MANIFEST, SIDECARS, load_index, prepare
from scripts.package_litert_resources import BUILTIN_DETECTOR, INVENTORY, copy_resources, verify_resources


@pytest.fixture
def bundled_source(tmp_path):
    root=tmp_path/'res'
    source=root/'bundled-models/Neri1.2preview-litert-fp32'
    source.mkdir(parents=True)
    values={'encoder':b'FP32 encoder','checkpoint':b'head','calibration':b'calibration'}
    digest=lambda value:hashlib.sha256(value).hexdigest()
    manifest=dict(schema_version=1,backend='dinov3',architecture='dinov3_vitb16',
        encoder_backend='litert',encoder_weights=FILES['encoder'],encoder_litert_sha256=digest(values['encoder']),
        encoder_sha256='a'*64,checkpoint=FILES['checkpoint'],checkpoint_sha256=digest(values['checkpoint']),
        calibration=FILES['calibration'],calibration_sha256=digest(values['calibration']),
        head_type='ridge_within_experiment',inference_recipe='experiment_full',feature_dim=768,
        preprocessing='letterbox224_imagenet',event_aggregation='complete_sequence',requires_detector=True,
        supports_video_fast=False,supports_video_all=False,display_name='builtin')
    assets=[]
    for key,data in values.items():
        part=key+'.part000'
        (source/part).write_bytes(data)
        assets.append(dict(path=FILES[key],size=len(data),sha256=digest(data),
            parts=[dict(path=part,size=len(data),sha256=digest(data))]))
    for name in SIDECARS:
        (source/name).write_text('license or source','utf8')
    index=dict(schema_version=1,name=NAME,manifest=manifest,assets=assets)
    (source/'assets.json').write_text(json.dumps(index),'utf8')
    detector=root/'model/detect/bundled/MDV6-yolov10-c.tflite'
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b'detector')
    return root,source,index


def test_explicit_dev_tool_reassembles_local_shards_while_package_excludes_classifier(bundled_source, tmp_path):
    root, source, index = bundled_source
    assembled_dir = root / 'model/cls/bundled'
    prep_report = prepare(source, assembled_dir)
    assert prep_report['passed']
    assert (assembled_dir / MANIFEST).is_file()
    for item in index['assets']:
        assert (assembled_dir / item['path']).read_bytes() == (source / item['parts'][0]['path']).read_bytes()

    output = tmp_path / 'package'
    report = copy_resources(root, output)
    assert report['passed']
    assert [row['path'] for row in report['models']] == [BUILTIN_DETECTOR]
    assert 'classifier_assets' not in report
    assert not (output / 'bundled-models').exists()
    assert (output / 'model/cls/bundled').is_dir()
    assert list((output / 'model/cls/bundled').rglob('*')) == []
    assert verify_resources(output, require_bundled_detector=True) == report

    from system.model_sync.layout import get_model_layout
    from system.model_sync.catalog import discover_models
    from system.model_sync.manager import ModelSyncManager
    from system.model_sync.protocol import ModelManifest
    from tests.test_model_sync_manager import FakeClient
    layout = get_model_layout(output)
    ModelSyncManager(layout, FakeClient(ModelManifest('a' * 64, ()))).sync_once_for_test()
    assert any(m.source == 'bundled' and m.backend == 'litert' for m in discover_models(layout, 'detect'))
    assert [m for m in discover_models(layout, 'cls') if m.source == 'bundled'] == []


def test_packaging_succeeds_without_repo_shards_or_classifier_assets(tmp_path):
    root = tmp_path / 'res'
    detector = root / BUILTIN_DETECTOR
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b'detector')
    output = tmp_path / 'package'
    report = copy_resources(root, output)
    assert report['passed']
    assert [row['path'] for row in report['models']] == [BUILTIN_DETECTOR]
    assert verify_resources(output, require_bundled_detector=True) == report


@pytest.mark.parametrize('mutation',['missing','corrupt','escape'])
def test_source_chunk_validation_never_trusts_existing_outputs(bundled_source,mutation):
    root,source,index=bundled_source
    destination=root/'model/cls/bundled'
    prepare(source,destination)
    if mutation=='missing':
        (source/'encoder.part000').unlink()
    elif mutation=='corrupt':
        (source/'encoder.part000').write_bytes(b'wrong')
    else:
        index['assets'][0]['parts'][0]['path']='../outside.part'
        (source/'assets.json').write_text(json.dumps(index),'utf8')
    with pytest.raises(ValueError):
        prepare(source,destination)
    assert (destination/MANIFEST).is_file()


@pytest.mark.parametrize(
    'relative_path',
    [
        'model/cls/user/unexpected.npz',
        'model/cls/user/unexpected.neri-model',
        'model/cls/user/unexpected.neri.json',
        'model/cls/user/unexpected.pth',
        'model/cls/bundled/Neri1.2preview-litert.neri.json',
        'model/cls/bundled/Neri1.2preview-litert-assets/encoder.tflite',
        'model/cls/bundled/Neri1.2preview-litert-assets/model.neri-model',
        'model/cls/bundled/Neri1.2preview-litert.LICENSE.md',
        'bundled-models/Neri1.2preview-litert-fp32/encoder.part000',
    ],
)
def test_classifier_and_shard_artifacts_are_rejected_in_package(bundled_source, tmp_path, relative_path):
    root, _, _ = bundled_source
    output = tmp_path / 'package'
    copy_resources(root, output)
    injected = output / relative_path
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_bytes(b'forbidden')
    with pytest.raises(ValueError):
        verify_resources(output)


def test_inventory_rejects_classifier_assets_field(tmp_path):
    root = tmp_path / 'source'
    detector = root / BUILTIN_DETECTOR
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b'detector')
    output = tmp_path / 'package'
    copy_resources(root, output)
    inventory_path = output / INVENTORY
    inventory = json.loads(inventory_path.read_text('utf-8'))
    inventory['classifier_assets'] = [{'path': 'model/cls/bundled/Neri1.2preview-litert.neri.json', 'size': 2, 'sha256': '0' * 64}]
    inventory_path.write_text(json.dumps(inventory), 'utf-8')
    with pytest.raises(ValueError, match='classifier'):
        verify_resources(output)


def test_detector_remains_required_for_distribution(tmp_path):
    root = tmp_path / 'source'
    detector = root / 'model/detect/user/other.tflite'
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b'detector')
    output = tmp_path / 'package'
    copy_resources(root, output)
    with pytest.raises(ValueError, match='Missing bundled LiteRT detector'):
        verify_resources(output, require_bundled_detector=True)


def test_local_classifier_shard_index_valid_when_present():
    repo_root = Path(__file__).resolve().parents[1]
    shard_dir = repo_root / 'res/bundled-models/Neri1.2preview-litert-fp32'
    if not (shard_dir / 'assets.json').is_file():
        pytest.skip('local Neri1.2preview LiteRT shards not present in checkout')
    index = load_index(shard_dir)
    assert index['name'] == NAME
