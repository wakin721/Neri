import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.dinov3.deployment import publish_model
from system.dinov3.encoder import PREPROCESSING, sha256
from system.dinov3.runtime import load_dinov3_model, read_manifest, resolve_manifest_asset


@pytest.fixture
def deployed(tmp_path):
    from system.dinov3.model import DinoV3RidgeWithinModel
    rng = np.random.default_rng(14)
    raw = rng.normal(0, .01, (16, 768))
    raw[:8, 0] += 1
    raw[8:, 1] += 1
    labels = np.asarray(["A"] * 8 + ["B"] * 8)
    cameras = np.asarray([f"c{i//4}" for i in range(16)])
    sequences = np.asarray([f"s{i//2}" for i in range(16)])
    weights = tmp_path / "source.pth"
    weights.write_bytes(b"test-only-encoder")
    code = tmp_path / "source-code"
    (code / "dinov3/hub").mkdir(parents=True)
    (code / "dinov3/hub/backbones.py").write_text("# test-only source", "utf8")
    (code / "LICENSE.md").write_text("test-only source license", "utf8")
    unknown = rng.normal(size=(8, 768))
    with threadpool_limits(limits=1):
        model = DinoV3RidgeWithinModel().fit(raw, labels, cameras, sequences,
            origin=np.zeros(768), transform=np.eye(768),
            metadata=dict(encoder_sha256=sha256(weights), preprocessing=PREPROCESSING))
        model.calibrate(np.r_[raw, unknown], np.r_[labels, ["U"] * 8],
                        ["cal"] * 24, [f"q{i}" for i in range(24)])
        manifest = publish_model(model, tmp_path / "joint.neri.json", encoder_weights=weights, encoder_code=code)
    return manifest, model, raw


def test_deployment_loads_joint_model_and_classifies(deployed):
    manifest, model, raw = deployed
    assert read_manifest(manifest)["display_name"] == "Neri1.1"
    encoder = SimpleNamespace(encode=lambda crops, **kw: raw[:len(crops)])
    with threadpool_limits(limits=1):
        runtime = load_dinov3_model(manifest, encoder_factory=lambda *args, **kwargs: encoder)
    assert runtime.registry is None and runtime.feedback is None
    assert runtime.classifier.backend == "dinov3"
    predictions = runtime.classifier.classify_crops(["a", "b"])
    assert len(predictions) == 2
    assert predictions[0].best_known_species == "A"
    candidate = predictions[0].as_candidate(detection_confidence=.8)
    assert 0 <= candidate["conf"] <= 1
    assert candidate["known_score"] == predictions[0].known_score
    assert candidate["raw_det_conf"] == .8
    runtime.close()
    assert runtime.classifier.encoder is None


def test_rejects_corrupt_checkpoint_before_constructing_encoder(deployed):
    manifest, model, raw = deployed
    payload = read_manifest(manifest)
    resolve_manifest_asset(manifest, payload["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_dinov3_model(manifest, encoder_factory=lambda *a, **k: pytest.fail("do not load encoder"))


def test_publication_failure_keeps_current_manifest(deployed, monkeypatch):
    manifest, model, raw = deployed
    previous = manifest.read_bytes()
    payload = read_manifest(manifest)
    monkeypatch.setattr(model, "save", lambda _: (_ for _ in ()).throw(RuntimeError("disk error")))
    with pytest.raises(RuntimeError, match="disk error"):
        publish_model(model, manifest, existing_manifest=payload)
    assert manifest.read_bytes() == previous
    assert resolve_manifest_asset(manifest, payload["checkpoint"]).is_file()


def test_old_dinov3_manifest_is_not_loaded_as_joint_model(tmp_path):
    path = tmp_path / "legacy.neri.json"
    path.write_text(json.dumps(dict(backend="dinov3", schema_version=2)), "utf8")
    with pytest.raises(ValueError, match="contract"):
        read_manifest(path)


def test_legacy_component_status_explains_migration_without_marking_it_healthy(tmp_path):
    from system.dinov3.component import dinov3_component_status
    root=tmp_path/'DINOv3'
    root.mkdir()
    (root/'multi_prototype.neri.json').write_text(json.dumps({
        'schema_version':1,'backend':'dinov3','architecture':'dinov3_vitb16',
        'event_aggregation':'mean_l2_normalized_crop_embeddings',
        'checkpoint':'multi_prototype.pt',
    }),encoding='utf-8')
    status=dinov3_component_status(root=root)
    assert status['installed'] and not status['healthy']
    assert '旧版' in status['message'] and '迁移' in status['message']
    assert (root/'multi_prototype.neri.json').is_file()


def test_update_command_publishes_new_joint_generation_with_frozen_geometry(deployed, tmp_path):
    from scripts.train_dinov3_ridge_within import train
    from system.dinov3.model import DinoV3RidgeWithinModel
    manifest, before, raw = deployed
    payload = read_manifest(manifest)
    digest = payload["encoder_sha256"]
    supports = tmp_path / "supports.npz"
    np.savez(supports, features=raw[:2], labels=["A"] * 2,
             cameras=["supplement"] * 2, sequences=["extra"] * 2, encoder_sha256=digest)
    cal = tmp_path / "cal.npz"
    unknown = np.random.default_rng(31).normal(size=(8, 768))
    np.savez(cal, features=np.r_[raw, unknown], labels=["A"] * 8 + ["B"] * 8 + ["U"] * 8,
             cameras=["cal"] * 24, sequences=[f"q{i}" for i in range(24)], encoder_sha256=digest)
    with threadpool_limits(limits=1):
        train(SimpleNamespace(command="update", manifest=manifest, supports=supports, calibration=cal))
        following = read_manifest(manifest)
        after = DinoV3RidgeWithinModel.load(resolve_manifest_asset(manifest, following["checkpoint"]))
    assert following["checkpoint"] != payload["checkpoint"]
    assert following["encoder_weights"] == payload["encoder_weights"]
    assert resolve_manifest_asset(manifest, payload["checkpoint"]).is_file()
    np.testing.assert_array_equal(after.origin_, before.origin_)
    np.testing.assert_array_equal(after.transform_, before.transform_)
    assert after.ridge_head.class_counts_[0] == before.ridge_head.class_counts_[0] + 2
    assert after.calibration is not None


def test_publish_model_exposes_fast_video_only_on_new_and_updated_manifests(deployed):
    manifest, model, _ = deployed
    published = read_manifest(manifest)
    assert published["supports_video_fast"] is True
    assert published["supports_video_all"] is False

    legacy_payload = dict(published, supports_video_fast=False, supports_video_all=False)
    manifest.write_text(json.dumps(legacy_payload), encoding="utf-8")
    with threadpool_limits(limits=1):
        publish_model(model, manifest, existing_manifest=read_manifest(manifest))
    republished = read_manifest(manifest)
    assert republished["supports_video_fast"] is True
    assert republished["supports_video_all"] is False


def test_read_manifest_accepts_old_false_and_new_true_fast_video_flags(deployed):
    manifest, _, _ = deployed
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    for fast_flag in (False, True):
        candidate = dict(payload, supports_video_fast=fast_flag, supports_video_all=False)
        manifest.write_text(json.dumps(candidate), encoding="utf-8")
        loaded = read_manifest(manifest)
        assert loaded["supports_video_fast"] is fast_flag
        assert loaded["supports_video_all"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"supports_video_all": True},
        {"supports_video_all": 0},
        {"supports_video_fast": "true"},
        {"supports_video_fast": 1},
        {"supports_video_fast": 0},
        {"_drop": "supports_video_fast"},
        {"_drop": "supports_video_all"},
    ],
)
def test_read_manifest_rejects_invalid_video_capability_flags(deployed, mutation):
    manifest, _, _ = deployed
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    drop_key = mutation.get("_drop")
    if drop_key:
        payload.pop(drop_key, None)
    else:
        payload.update(mutation)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        read_manifest(manifest)
