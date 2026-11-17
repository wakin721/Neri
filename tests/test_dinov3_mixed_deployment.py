from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from system.dinov3.mixed_media import MEDIA_SEQUENCE_VERSION

import scripts.deploy_mixed_media as deploy_mod


EXPLICIT_FILES = (
    "system/metadata_extractor.py",
    "system/backend/services.py",
    "system/backend/services_legacy.py",
    "system/backend/video_frames.py",
    "system/backend/media_time.py",
    "system/dinov3/classifier.py",
    "system/dinov3/experiment.py",
    "system/dinov3/model.py",
    "system/dinov3/processor.py",
    "system/dinov3/evidence.py",
    "system/dinov3/media_cache.py",
    "system/dinov3/mixed_media.py",
    "system/dinov3/video.py",
    "system/dinov3/video_reference.py",
    "scripts/verify_mixed_media_installation.py",
    "docs/dinov3-mixed-media-sequences-20261116.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_file(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _build_fake_roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    install = tmp_path / "install"
    source.mkdir()
    install.mkdir()

    for idx, rel in enumerate(EXPLICIT_FILES):
        _write_file(source / rel, f"# new {rel} v{idx}\n")

    # Unrelated dirty/unmanaged files in source that must never be deployed.
    _write_file(source / "system/backend/main_core_legacy.py", "# dirty source legacy\n")
    _write_file(source / "res/model/cls/sync/new_model.pt", b"new-model-weights")
    _write_file(source / "frontend/build/windows/x64/runner/Release/Neri.exe", b"new-ui")
    _write_file(source / "temp/user_history.sqlite3", b"source-history")

    # Populate installation: 2 unchanged, 11 older, 3 missing (evidence.py, video_reference.py & doc).
    for idx, rel in enumerate(EXPLICIT_FILES):
        if idx < 2:
            _write_file(install / rel, f"# new {rel} v{idx}\n")
        elif rel not in (
            "system/dinov3/evidence.py",
            "system/dinov3/video_reference.py",
            "docs/dinov3-mixed-media-sequences-20261116.md",
        ):
            _write_file(install / rel, f"# old {rel} v{idx}\n")

    _write_file(install / "system/backend/main_core_legacy.py", "# installed legacy\n")
    _write_file(install / "res/model/cls/user/selected.neri.json", '{"model":"user"}')
    _write_file(install / "Neri.exe", b"installed-ui")
    _write_file(install / "temp/user_history.sqlite3", b"installed-history")

    settings = {
        "video_mode": "skip",
        "confidence": 0.35,
        "selected_classification_model": str((install / "res/model/cls/user/selected.neri.json").resolve()),
        "nested": {"中文键": True, "counts": [1, 2, 3]},
    }
    _write_file(install / "temp/settings.json", json.dumps(settings, ensure_ascii=False, indent=4))
    return source, install


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch):
    if deploy_mod.psutil is not None:
        monkeypatch.setattr(deploy_mod.psutil, "process_iter", lambda *a, **kw: [])


def test_plan_is_read_only_computes_changed_only_and_stages_settings_in_runs(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    report_default = tmp_path / "plan_default.json"
    before_snapshot = {
        p.relative_to(install).as_posix(): _sha256(p)
        for p in install.rglob("*")
        if p.is_file()
    }

    rc = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_default),
    ])
    assert rc == 0

    after_snapshot = {
        p.relative_to(install).as_posix(): _sha256(p)
        for p in install.rglob("*")
        if p.is_file()
    }
    assert after_snapshot == before_snapshot
    assert not (install / "repair-backups").exists()

    plan = json.loads(report_default.read_text(encoding="utf-8"))
    assert deploy_mod.EXPLICIT_FILES == EXPLICIT_FILES
    assert plan["mode"] == "plan"
    assert plan["files_to_update"] == 14
    assert len(plan["unchanged_files"]) == 2
    changed_paths = [entry["path"] for entry in plan["changes"]]
    assert set(changed_paths) == set(EXPLICIT_FILES[2:])
    assert "system/backend/main_core_legacy.py" not in changed_paths
    assert "temp/settings.json" not in changed_paths
    for entry in plan["changes"]:
        assert entry["sha256"] == _sha256(source / entry["path"])
        expected_old = _sha256(install / entry["path"]) if (install / entry["path"]).is_file() else None
        assert entry["old_sha256"] == expected_old

    # With --enable-fast-video in plan mode, stage generated settings inside source/runs only.
    report_fast = tmp_path / "plan_fast.json"
    rc = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_fast),
        "--enable-fast-video",
    ])
    assert rc == 0
    assert {p.relative_to(install).as_posix(): _sha256(p) for p in install.rglob("*") if p.is_file()} == before_snapshot
    assert json.loads((install / "temp/settings.json").read_text(encoding="utf-8"))["video_mode"] == "skip"

    fast_plan = json.loads(report_fast.read_text(encoding="utf-8"))
    staged_settings = Path(fast_plan["staged_settings_path"])
    assert staged_settings.is_file()
    assert staged_settings.resolve().is_relative_to((source / "runs").resolve())
    staged_data = json.loads(staged_settings.read_text(encoding="utf-8"))
    assert staged_data["video_mode"] == "fast"
    assert staged_data["nested"] == {"中文键": True, "counts": [1, 2, 3]}
    assert "temp/settings.json" in [entry["path"] for entry in fast_plan["changes"]]


def test_apply_updates_changed_explicit_files_and_preserves_unmanaged_and_settings(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    report_path = tmp_path / "apply_report.json"
    orig_settings_bytes = (install / "temp/settings.json").read_bytes()
    orig_legacy_hash = _sha256(install / "system/backend/main_core_legacy.py")
    orig_model_hash = _sha256(install / "res/model/cls/user/selected.neri.json")
    orig_ui_hash = _sha256(install / "Neri.exe")
    orig_history_hash = _sha256(install / "temp/user_history.sqlite3")

    rc = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_path),
        "--apply",
    ])
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "apply"
    assert report["applied"] is True
    assert report["files_updated"] == 14

    for rel in EXPLICIT_FILES:
        assert _sha256(install / rel) == _sha256(source / rel)

    assert (install / "temp/settings.json").read_bytes() == orig_settings_bytes
    assert _sha256(install / "system/backend/main_core_legacy.py") == orig_legacy_hash
    assert _sha256(install / "res/model/cls/user/selected.neri.json") == orig_model_hash
    assert _sha256(install / "Neri.exe") == orig_ui_hash
    assert _sha256(install / "temp/user_history.sqlite3") == orig_history_hash
    assert not (install / "res/model/cls/sync/new_model.pt").exists()

    backup_dir = Path(report["backup_dir"])
    assert backup_dir.parent == install / "repair-backups"
    assert backup_dir.name.startswith("mixed-media-")
    assert (backup_dir / "inventory.json").is_file()
    # Unchanged explicit files were not backed up as overwrites.
    assert not (backup_dir / "files" / EXPLICIT_FILES[0]).exists()
    assert not (backup_dir / "files" / EXPLICIT_FILES[1]).exists()
    # Newly created files have no backup copy; overwritten ones do.
    assert not (backup_dir / "files/system/dinov3/evidence.py").exists()
    assert not (backup_dir / "files/system/dinov3/video_reference.py").exists()
    assert (backup_dir / "files/system/dinov3/video.py").is_file()
    assert (backup_dir / "files/system/dinov3/processor.py").is_file()


def test_apply_enable_fast_video_changes_only_video_mode_and_backs_up(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    orig_settings = json.loads((install / "temp/settings.json").read_text(encoding="utf-8"))
    report_path = tmp_path / "apply_fast.json"

    rc = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_path),
        "--apply",
        "--enable-fast-video",
    ])
    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["files_updated"] == 15
    assert report["settings_updated"] is True

    new_settings = json.loads((install / "temp/settings.json").read_text(encoding="utf-8"))
    assert new_settings == {**orig_settings, "video_mode": "fast"}

    backup_settings = Path(report["backup_dir"]) / "files/temp/settings.json"
    assert json.loads(backup_settings.read_text(encoding="utf-8")) == orig_settings

    # Second run is a no-op because hashes and video_mode already match.
    report_rerun = tmp_path / "apply_fast_rerun.json"
    rc2 = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_rerun),
        "--apply",
        "--enable-fast-video",
    ])
    assert rc2 == 0
    assert json.loads(report_rerun.read_text(encoding="utf-8"))["files_updated"] == 0


def test_apply_failure_rolls_back_written_only_and_checks_new_file_hash(tmp_path, monkeypatch):
    source, install = _build_fake_roots(tmp_path)
    # Make only 3 changed files: one existing, one new, one existing that will fail.
    for rel in EXPLICIT_FILES:
        if rel not in (
            "system/backend/video_frames.py",
            "system/dinov3/evidence.py",
            "system/dinov3/processor.py",
        ):
            _write_file(install / rel, (source / rel).read_bytes())

    orig_video_frames = _sha256(install / "system/backend/video_frames.py")
    orig_processor = _sha256(install / "system/dinov3/processor.py")
    assert not (install / "system/dinov3/evidence.py").exists()

    real_replace = os.replace
    replace_calls = []

    def flaky_replace(src, dst):
        dst_path = Path(dst)
        replace_calls.append(dst_path.name)
        # Fail on the 3rd target replacement (sorted order: evidence.py, processor.py, video_frames.py)
        if dst_path.name == "processor.py" and ".deploy-tmp-" in Path(src).name:
            raise OSError("simulated atomic replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(deploy_mod.os, "replace", flaky_replace)
    report_path = tmp_path / "failed_apply.json"

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        deploy_mod.apply_deployment(source, install, report_path=report_path)

    # Newly created evidence.py was written first and must be removed on rollback.
    assert not (install / "system/dinov3/evidence.py").exists()
    # Processor.py never replaced and video_frames.py never reached.
    assert _sha256(install / "system/dinov3/processor.py") == orig_processor
    assert _sha256(install / "system/backend/video_frames.py") == orig_video_frames
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["applied"] is False
    assert report["rolled_back"] is True


def test_rollback_skips_new_file_if_hash_changed_since_apply(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    apply_report = tmp_path / "apply.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(apply_report),
        "--apply",
        "--enable-fast-video",
    ]) == 0
    backup_dir = Path(json.loads(apply_report.read_text(encoding="utf-8"))["backup_dir"])

    # Mutate newly created evidence.py so it no longer matches the applied hash.
    evidence = install / "system/dinov3/evidence.py"
    evidence.write_text("# user edited after deploy\n", encoding="utf-8")

    rb_report = tmp_path / "rollback.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(rb_report),
        "--rollback-from", str(backup_dir),
    ]) == 0

    rb = json.loads(rb_report.read_text(encoding="utf-8"))
    assert rb["rolled_back"] is True
    # Restored overwritten files + settings, removed unmodified new doc, preserved modified evidence.py.
    assert not (install / "docs/dinov3-mixed-media-sequences-20261116.md").exists()
    assert evidence.is_file()
    assert "system/dinov3/evidence.py" in rb["skipped_modified_new_files"]
    assert json.loads((install / "temp/settings.json").read_text(encoding="utf-8"))["video_mode"] == "skip"


def test_manual_rollback_validates_installation_root_and_inventory_paths(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    apply_report = tmp_path / "apply.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(apply_report),
        "--apply",
    ]) == 0
    backup_dir = Path(json.loads(apply_report.read_text(encoding="utf-8"))["backup_dir"])

    other_install = tmp_path / "other_install"
    other_install.mkdir()
    with pytest.raises(ValueError, match="installation"):
        deploy_mod.rollback_deployment(other_install, backup_dir)

    # Outside repair-backups directory rejected.
    outside_dir = tmp_path / "outside_backup"
    outside_dir.mkdir()
    with pytest.raises(ValueError, match="repair-backups"):
        deploy_mod.rollback_deployment(install, outside_dir)

    # Traversal path inside inventory.json rejected before mutating files.
    inventory_path = backup_dir / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["changes"][0]["path"] = "../escape.txt"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        deploy_mod.rollback_deployment(install, backup_dir)


def test_missing_source_file_fails_before_touching_installation(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    (source / "system/dinov3/mixed_media.py").unlink()
    with pytest.raises(FileNotFoundError):
        deploy_mod.plan_deployment(source, install)
    with pytest.raises(FileNotFoundError):
        deploy_mod.apply_deployment(source, install)
    assert not (install / "repair-backups").exists()


def test_unsafe_paths_and_symlink_escapes_rejected(tmp_path):
    source, install = _build_fake_roots(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe"):
        deploy_mod.resolve_safe_path(source, "../outside.py")
    with pytest.raises(ValueError, match="Unsafe"):
        deploy_mod.resolve_safe_path(install, str(outside))

    symlink_target = install / "system/dinov3/media_cache.py"
    symlink_target.unlink()
    try:
        symlink_target.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this Windows host")

    with pytest.raises(ValueError, match="symlink|Unsafe"):
        deploy_mod.apply_deployment(source, install)
    assert outside.read_text(encoding="utf-8") == "# outside\n"


def test_active_process_guard_blocks_install_exe_and_ignores_irrelevant_parent(tmp_path, monkeypatch):
    source, install = _build_fake_roots(tmp_path)
    external_python = tmp_path / "external_venv/python.exe"
    _write_file(external_python, b"")
    install_neri = install / "Neri.exe"

    # Standard-library-only fallback when psutil is unavailable.
    monkeypatch.setattr(deploy_mod, "psutil", None)
    assert deploy_mod.active_processes(install) == []
    report_no_psutil = tmp_path / "plan_no_psutil.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_no_psutil),
    ]) == 0

    def fake_iter_irrelevant(*args, **kwargs):
        return [SimpleNamespace(pid=4242, info={"pid": 4242, "name": "python.exe"}, exe=lambda: str(external_python))]

    monkeypatch.setattr(deploy_mod, "psutil", SimpleNamespace(process_iter=fake_iter_irrelevant, Error=Exception))
    report = tmp_path / "ok.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report),
        "--apply",
    ]) == 0

    def fake_iter_active(*args, **kwargs):
        return [SimpleNamespace(pid=5150, info={"pid": 5150, "name": "Neri.exe"}, exe=lambda: str(install_neri))]

    monkeypatch.setattr(deploy_mod, "psutil", SimpleNamespace(process_iter=fake_iter_active, Error=Exception))
    with pytest.raises(RuntimeError, match="active|Close Neri"):
        deploy_mod.apply_deployment(source, install)


def test_verifier_cli_defaults_and_scenario_choices():
    import scripts.verify_mixed_media_installation as verify_mod

    parser = verify_mod.build_parser()
    base_args = [
        "--application-root", ".",
        "--source-image", "sample.jpg",
        "--manifest", "manifest.json",
        "--detector", "det.pt",
        "--runtime", "cpu",
        "--output", "out",
    ]
    parsed_default = parser.parse_args(base_args)
    assert parsed_default.scenario == "own"

    for scenario in ("own", "burst", "none"):
        parsed = parser.parse_args([*base_args, "--scenario", scenario])
        assert parsed.scenario == scenario

    with pytest.raises(SystemExit):
        parser.parse_args([*base_args, "--scenario", "invalid"])


def test_verifier_prepares_disposable_camera_files_for_all_scenarios(tmp_path):
    from PIL import Image
    import scripts.verify_mixed_media_installation as verify_mod
    from system.backend.media_time import video_capture_time
    from system.metadata_extractor import ImageMetadataExtractor

    source_img = tmp_path / "seed.jpg"
    Image.new("RGB", (64, 64), color=(120, 80, 40)).save(source_img)

    own_dir = tmp_path / "own_cam"
    own_photos, own_video = verify_mod.prepare_disposable_camera(own_dir, source_img, "own")
    assert [p.name for p in own_photos] == ["photo.jpg"]
    assert own_video.name == "clip.avi"
    own_meta, own_im = ImageMetadataExtractor.extract_metadata(str(own_photos[0]), own_photos[0].name)
    if own_im is not None:
        own_im.close()
    assert own_meta["拍摄日期"] == "2026-11-16" and own_meta["拍摄时间"] == "08:00:00"
    own_vtime = video_capture_time(own_video)
    assert own_vtime["timestamp_reliable"] is True
    assert own_vtime["date_taken"] == "2026-11-16T08:00:10"

    burst_dir = tmp_path / "burst_cam"
    burst_photos, burst_video = verify_mod.prepare_disposable_camera(burst_dir, source_img, "burst")
    assert [p.name for p in burst_photos] == ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"]
    assert burst_video.name == "IMG_0004.avi"
    for idx, photo in enumerate(burst_photos):
        meta, im = ImageMetadataExtractor.extract_metadata(str(photo), photo.name)
        if im is not None:
            im.close()
        assert meta["拍摄日期"] == "2026-11-16"
        assert meta["拍摄时间"] == f"08:00:0{idx}"
    burst_vtime = video_capture_time(burst_video)
    assert burst_vtime["timestamp_reliable"] is False
    assert burst_vtime["date_taken"] is None

    none_dir = tmp_path / "none_cam"
    none_photos, none_video = verify_mod.prepare_disposable_camera(none_dir, source_img, "none")
    assert [p.name for p in none_photos] == ["photo.jpg"]
    assert none_video.name == "clip.avi"
    none_meta, none_im = ImageMetadataExtractor.extract_metadata(str(none_photos[0]), none_photos[0].name)
    if none_im is not None:
        none_im.close()
    assert none_meta["拍摄日期"] == "2026-11-16" and none_meta["拍摄时间"] == "08:00:00"
    none_vtime = video_capture_time(none_video)
    assert none_vtime["timestamp_reliable"] is False
    assert none_vtime["date_taken"] is None


def _make_fake_job_for_verifier(tmp_path: Path, scenario: str):
    from system.backend.models import DetectionItem, JobState, JobSummary
    from system.dinov3.observation_support import make_dinov3_observation_id

    cam = tmp_path / "camera"
    cam.mkdir(parents=True, exist_ok=True)
    if scenario == "burst":
        photo_paths = [cam / f"IMG_000{i}.jpg" for i in (1, 2, 3)]
        video_path = cam / "IMG_0004.avi"
    else:
        photo_paths = [cam / "photo.jpg"]
        video_path = cam / "clip.avi"

    items = []
    photo_seq_ids = []
    for idx, p in enumerate(photo_paths):
        p.write_bytes(b"jpg")
        obs_id = make_dinov3_observation_id(p.resolve(), 0, [10.0, 10.0, 50.0, 50.0])
        items.append(
            DetectionItem(
                filename=p.name,
                path=str(p.resolve()),
                file_type="jpg",
                date_taken=f"2026-11-16T08:00:0{idx}",
                detection_boxes=[{"bbox": [10.0, 10.0, 50.0, 50.0], "species": "deer", "confidence": 0.9}],
                detection_data={
                    "media_sequence_version": MEDIA_SEQUENCE_VERSION,
                    "sequence_completed": True,
                    "检测框": [
                        {
                            "observation_id": obs_id,
                            "track_id": "cam/seq_00:track_00",
                            "event_id": "cam/seq_00",
                        }
                    ],
                },
            )
        )
        photo_seq_ids.append("cam/seq_00")

    video_path.write_bytes(b"avi")
    video_id_prefix = "video:deadbeef"
    v_obs_0 = make_dinov3_observation_id(video_path.resolve(), 0, [10.0, 10.0, 50.0, 50.0])
    v_obs_1 = make_dinov3_observation_id(video_path.resolve(), 1, [60.0, 10.0, 90.0, 50.0])
    v_obs_2 = make_dinov3_observation_id(video_path.resolve(), 2, [12.0, 10.0, 52.0, 50.0])

    if scenario == "own":
        video_date = "2026-11-16T08:00:10"
        capture_time = {"timestamp_reliable": True, "date_taken": video_date}
        video_ref = {
            "source": "video",
            "reference_time": "2026-11-16T00:00:10+00:00",
            "precision": "exact",
            "burst_group_id": None,
            "group_members": [],
            "group_end_time": None,
        }
    elif scenario == "burst":
        video_date = None
        capture_time = {"timestamp_reliable": False, "date_taken": None}
        video_ref = {
            "source": "burst_group",
            "reference_time": "2026-11-16T00:00:00+00:00",
            "precision": "group_anchor",
            "burst_group_id": "burst-123",
            "group_members": [str(p.resolve()) for p in [*photo_paths, video_path]],
            "group_end_time": "2026-11-16T00:00:02+00:00",
        }
    else:
        video_date = None
        capture_time = {"timestamp_reliable": False, "date_taken": None}
        video_ref = {
            "source": "none",
            "reference_time": None,
            "precision": "unknown",
            "burst_group_id": None,
            "group_members": [],
            "group_end_time": None,
        }

    video_boxes = [
        {
            "observation_id": v_obs_0,
            "frame_index": 0,
            "timestamp": 0.0,
            "track_id": f"{video_id_prefix}:track_00",
            "event_id": video_id_prefix,
        },
        {
            "observation_id": v_obs_1,
            "frame_index": 0,
            "timestamp": 0.0,
            "track_id": f"{video_id_prefix}:track_01",
            "event_id": video_id_prefix,
        },
        {
            "observation_id": v_obs_2,
            "frame_index": 25,
            "timestamp": 1.0,
            "track_id": f"{video_id_prefix}:track_00",
            "event_id": video_id_prefix,
        },
    ]
    items.append(
        DetectionItem(
            filename=video_path.name,
            path=str(video_path.resolve()),
            file_type="avi",
            date_taken=video_date,
            detection_boxes=[{"bbox": [10.0, 10.0, 50.0, 50.0]} for _ in video_boxes],
            detection_data={
                "media_sequence_version": MEDIA_SEQUENCE_VERSION,
                "sequence_completed": True,
                "classification_scope": "video_track",
                "sampled_frames": [0, 25],
                "capture_time": capture_time,
                "video_reference": video_ref,
                "检测框": video_boxes,
            },
        )
    )
    video_seq_ids = [b["track_id"] for b in video_boxes]
    calls = [{"crops": len(photo_seq_ids) + len(video_seq_ids), "sequence_ids": [*photo_seq_ids, *video_seq_ids]}]
    now = "2026-11-16T08:00:00+08:00"
    job = JobSummary(
        id="mixed-smoke",
        state=JobState.COMPLETED,
        input_dir=str(cam),
        created_at=now,
        updated_at=now,
        results=items,
    )
    return job, calls


@pytest.mark.parametrize("scenario", ["own", "burst", "none"])
def test_verifier_validates_v2_policy_for_all_scenarios(tmp_path, scenario):
    import scripts.verify_mixed_media_installation as verify_mod

    job, calls = _make_fake_job_for_verifier(tmp_path, scenario)
    verify_mod.validate_job_results(job, calls, scenario)


def test_verifier_rejects_v2_policy_violations(tmp_path):
    import scripts.verify_mixed_media_installation as verify_mod

    # 1. evidence_weights passed to classifier
    job, calls = _make_fake_job_for_verifier(tmp_path / "c1", "own")
    calls[0]["has_evidence_weights"] = True
    with pytest.raises(AssertionError):
        verify_mod.validate_job_results(job, calls, "own")

    # 2. photo and video share sequence_id
    job, calls = _make_fake_job_for_verifier(tmp_path / "c2", "own")
    calls[0]["sequence_ids"][1] = calls[0]["sequence_ids"][0]
    with pytest.raises(AssertionError):
        verify_mod.validate_job_results(job, calls, "own")

    # 3. fallback video has fabricated date_taken
    job, calls = _make_fake_job_for_verifier(tmp_path / "c3", "burst")
    job.results[-1] = job.results[-1].model_copy(update={"date_taken": "2026-11-16T08:00:00"})
    with pytest.raises(AssertionError):
        verify_mod.validate_job_results(job, calls, "burst")

    # 4. same-frame objects share track_id
    job, calls = _make_fake_job_for_verifier(tmp_path / "c4", "none")
    job.results[-1].detection_data["检测框"][1]["track_id"] = job.results[-1].detection_data["检测框"][0]["track_id"]
    with pytest.raises(AssertionError):
        verify_mod.validate_job_results(job, calls, "none")

    # 5. old media_sequence_version == 1 rejected
    job, calls = _make_fake_job_for_verifier(tmp_path / "c5", "own")
    job.results[-1].detection_data["media_sequence_version"] = 1
    with pytest.raises(AssertionError):
        verify_mod.validate_job_results(job, calls, "own")
