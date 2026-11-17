from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

import scripts.deploy_mixed_media as base_deploy_mod
import scripts.deploy_validation_fixes as deploy_mod
import scripts.flutter_accessibility_hotfix as hotfix_mod

RELEASE_UI_REL = "frontend/build/windows/x64/runner/Release"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_file(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _build_fake_roots(
    tmp_path: Path,
    *,
    backend_files: tuple[str, ...] | None = None,
) -> tuple[Path, Path, tuple[str, ...]]:
    source = tmp_path / "source"
    install = tmp_path / "install"
    source.mkdir()
    install.mkdir()

    active_backend = tuple(deploy_mod.EXPLICIT_FILES if backend_files is None else backend_files)
    for idx, rel in enumerate(active_backend):
        _write_file(source / rel, f"# new {rel} v{idx}\n")

    # Compiled UI artifacts in Release directory.
    release_dir = source / RELEASE_UI_REL
    ui_payloads: dict[str, bytes] = {
        "Neri.exe": b"new-neri-exe-payload",
        "flutter_windows.dll": b"guarded-flutter-engine-bytes",
        "data/app.so": b"new-app-so-payload",
        "data/flutter_assets/AssetManifest.json": b'{"assets": ["ok"]}',
        "plugin_helper.dll": b"new-plugin-dll",
    }
    for rel, payload in ui_payloads.items():
        _write_file(release_dir / rel, payload)

    # Unrelated / dirty / unmanaged files in source that must never be deployed.
    _write_file(source / "system/backend/main_core_legacy.py", "# dirty source legacy\n")
    _write_file(source / "res/model/cls/sync/new_model.pt", b"new-model-weights")
    _write_file(source / "temp/settings.json", '{"video_mode": "overwrite_attempt"}')
    _write_file(source / "temp/user_history.sqlite3", b"source-history")
    _write_file(source / "Neri.exe", b"stray-root-exe-must-not-deploy")

    # Populate installation: first backend file unchanged, last backend file missing, rest older.
    for idx, rel in enumerate(active_backend):
        if idx == 0:
            _write_file(install / rel, f"# new {rel} v{idx}\n")
        elif idx < len(active_backend) - 1:
            _write_file(install / rel, f"# old {rel} v{idx}\n")

    # Populate existing installation UI: some older, plugin_helper.dll new, data/flutter_assets missing.
    _write_file(install / "Neri.exe", b"old-neri-exe")
    _write_file(install / "flutter_windows.dll", b"old-engine-bytes")
    _write_file(install / "data/app.so", b"old-app-so")
    _write_file(install / "plugin_helper.dll", b"new-plugin-dll")

    # Protected / unmanaged installed files.
    _write_file(install / "system/backend/main_core_legacy.py", "# installed legacy\n")
    _write_file(install / "res/model/cls/user/selected.neri.json", '{"model": "user"}')
    _write_file(install / "temp/user_history.sqlite3", b"installed-history")
    settings = {
        "video_mode": "skip",
        "confidence": 0.4,
        "selected_classification_model": str((install / "res/model/cls/user/selected.neri.json").resolve()),
        "nested": {"key": "value"},
    }
    _write_file(install / "temp/settings.json", json.dumps(settings, ensure_ascii=False, indent=2))
    return source, install, active_backend


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch):
    if deploy_mod.psutil is not None:
        monkeypatch.setattr(deploy_mod.psutil, "process_iter", lambda *a, **kw: [])


@pytest.fixture
def _guarded_engine(monkeypatch):
    monkeypatch.setattr(hotfix_mod, "patched_bytes", lambda data: data)


def test_plan_is_read_only_and_includes_backend_plus_mapped_release_ui(tmp_path, _guarded_engine):
    source, install, backend_files = _build_fake_roots(tmp_path)
    report_path = tmp_path / "plan.json"
    before_snapshot = {
        p.relative_to(install).as_posix(): _sha256(p)
        for p in install.rglob("*")
        if p.is_file()
    }

    rc = deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_path),
    ])
    assert rc == 0

    after_snapshot = {
        p.relative_to(install).as_posix(): _sha256(p)
        for p in install.rglob("*")
        if p.is_file()
    }
    assert after_snapshot == before_snapshot
    assert not (install / "repair-backups").exists()

    plan = json.loads(report_path.read_text(encoding="utf-8"))
    assert plan["mode"] == "plan"
    assert plan["engine_hotfix_verified"] is True
    assert plan["engine_sha256"] == _sha256(source / RELEASE_UI_REL / "flutter_windows.dll")

    expected_new_or_changed = {
        *backend_files[1:],
        "Neri.exe",
        "flutter_windows.dll",
        "data/app.so",
        "data/flutter_assets/AssetManifest.json",
    }
    changed_by_path = {entry["path"]: entry for entry in plan["changes"]}
    assert set(changed_by_path) == expected_new_or_changed
    assert set(plan["unchanged_files"]) == {backend_files[0], "plugin_helper.dll"}
    assert "system/backend/preview_fast.py" in deploy_mod.EXPLICIT_FILES
    assert "system/backend/preview_fast.py" in changed_by_path

    assert changed_by_path["Neri.exe"]["source"] == str((source / RELEASE_UI_REL / "Neri.exe").resolve())
    assert changed_by_path["Neri.exe"]["sha256"] != _sha256(source / "Neri.exe")
    assert "system/backend/main_core_legacy.py" not in changed_by_path
    assert "temp/settings.json" not in changed_by_path
    assert "res/model/cls/sync/new_model.pt" not in changed_by_path

    assert plan["protected"]["temp/settings.json"] == _sha256(install / "temp/settings.json")
    assert plan["protected"]["res/model/cls/user/selected.neri.json"] == _sha256(
        install / "res/model/cls/user/selected.neri.json"
    )

    # Explicit --plan produces the same read-only result.
    explicit_plan_path = tmp_path / "explicit_plan.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(explicit_plan_path),
        "--plan",
    ]) == 0
    assert json.loads(explicit_plan_path.read_text(encoding="utf-8"))["changes"] == plan["changes"]


def test_apply_updates_backend_and_ui_creates_backup_and_preserves_protected_metadata(
    tmp_path,
    _guarded_engine,
):
    source, install, backend_files = _build_fake_roots(tmp_path)
    report_path = tmp_path / "apply.json"
    orig_settings_bytes = (install / "temp/settings.json").read_bytes()
    orig_legacy_hash = _sha256(install / "system/backend/main_core_legacy.py")
    orig_model_hash = _sha256(install / "res/model/cls/user/selected.neri.json")
    orig_history_hash = _sha256(install / "temp/user_history.sqlite3")
    orig_exe_hash = _sha256(install / "Neri.exe")

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
    assert report["verified"] is True
    assert report["engine_hotfix_verified"] is True

    for rel in backend_files:
        assert _sha256(install / rel) == _sha256(source / rel)
    for rel in (
        "Neri.exe",
        "flutter_windows.dll",
        "data/app.so",
        "data/flutter_assets/AssetManifest.json",
        "plugin_helper.dll",
    ):
        assert _sha256(install / rel) == _sha256(source / RELEASE_UI_REL / rel)

    assert (install / "temp/settings.json").read_bytes() == orig_settings_bytes
    assert _sha256(install / "system/backend/main_core_legacy.py") == orig_legacy_hash
    assert _sha256(install / "res/model/cls/user/selected.neri.json") == orig_model_hash
    assert _sha256(install / "temp/user_history.sqlite3") == orig_history_hash
    assert not (install / "res/model/cls/sync/new_model.pt").exists()

    backup_dir = Path(report["backup_dir"])
    assert backup_dir.parent == install / "repair-backups"
    assert (backup_dir / "inventory.json").is_file()
    assert _sha256(backup_dir / "files/Neri.exe") == orig_exe_hash
    assert not (backup_dir / "files" / backend_files[0]).exists()
    assert not (backup_dir / "files/data/flutter_assets/AssetManifest.json").exists()

    inventory = json.loads((backup_dir / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["protected"]["temp/settings.json"] == _sha256(install / "temp/settings.json")
    assert inventory["protected"]["res/model/cls/user/selected.neri.json"] == orig_model_hash

    # Idempotent re-apply creates no extra backup when everything matches.
    noop_report_path = tmp_path / "noop_apply.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(noop_report_path),
        "--apply",
    ]) == 0
    noop_report = json.loads(noop_report_path.read_text(encoding="utf-8"))
    assert noop_report["files_updated"] == 0
    assert noop_report["backup_dir"] is None


def _make_unknown_x64_pe_dll() -> bytes:
    # Minimal Windows x64 PE DLL with one section covering RVA 0x3A9ED so patched_bytes reaches hash check.
    pe_offset = 0x80
    buf = bytearray(0x400)
    buf[:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, pe_offset)
    buf[pe_offset:pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<H", buf, pe_offset + 4, 0x8664)  # AMD64
    struct.pack_into("<H", buf, pe_offset + 6, 1)       # 1 section
    struct.pack_into("<H", buf, pe_offset + 20, 0xF0)   # optional header size
    section_offset = pe_offset + 24 + 0xF0
    struct.pack_into("<IIII", buf, section_offset + 8, 0x50000, 0x1000, 0x50000, 0x200)
    return bytes(buf)


def test_engine_guard_rejects_unpatched_and_unknown_engine(tmp_path, monkeypatch):
    source, install, _ = _build_fake_roots(tmp_path)
    before_exe = _sha256(install / "Neri.exe")

    # Known unpatched engine: patched_bytes returns modified bytes -> fail telling caller to apply hotfix.
    monkeypatch.setattr(hotfix_mod, "patched_bytes", lambda data: data + b"-hotfixed")
    with pytest.raises(ValueError, match="hotfix"):
        deploy_mod.plan_deployment(source, install)
    with pytest.raises(ValueError, match="hotfix"):
        deploy_mod.apply_deployment(source, install)
    assert not (install / "repair-backups").exists()
    assert _sha256(install / "Neri.exe") == before_exe

    # Unknown engine hash via real hotfix validator -> fail closed.
    monkeypatch.undo()
    if deploy_mod.psutil is not None:
        monkeypatch.setattr(deploy_mod.psutil, "process_iter", lambda *a, **kw: [])
    with pytest.raises(ValueError):
        deploy_mod.plan_deployment(source, install)
    with pytest.raises(ValueError):
        deploy_mod.apply_deployment(source, install)

    # Minimal valid x64 PE DLL with unknown SHA-256 -> fails closed on unsupported engine hash.
    _write_file(source / RELEASE_UI_REL / "flutter_windows.dll", _make_unknown_x64_pe_dll())
    with pytest.raises(ValueError, match="Unsupported engine hash"):
        deploy_mod.plan_deployment(source, install)
    with pytest.raises(ValueError, match="Unsupported engine hash"):
        deploy_mod.apply_deployment(source, install)
    assert not (install / "repair-backups").exists()
    assert _sha256(install / "Neri.exe") == before_exe


@pytest.mark.parametrize("missing_rel", ("Neri.exe", "data/app.so", "flutter_windows.dll"))
def test_missing_required_ui_artifact_rejected(tmp_path, _guarded_engine, missing_rel):
    source, install, _ = _build_fake_roots(tmp_path)
    (source / RELEASE_UI_REL / missing_rel).unlink()

    with pytest.raises(FileNotFoundError):
        deploy_mod.plan_deployment(source, install)
    with pytest.raises(FileNotFoundError):
        deploy_mod.apply_deployment(source, install)
    assert not (install / "repair-backups").exists()


def test_missing_release_dir_and_missing_backend_file_rejected(tmp_path, _guarded_engine):
    source, install, backend_files = _build_fake_roots(tmp_path)
    missing_backend = source / backend_files[-1]
    missing_backend.unlink()
    with pytest.raises(FileNotFoundError):
        deploy_mod.plan_deployment(source, install)

    _write_file(missing_backend, "# restored\n")
    for path in sorted((source / RELEASE_UI_REL).rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    (source / RELEASE_UI_REL).rmdir()
    with pytest.raises(FileNotFoundError):
        deploy_mod.apply_deployment(source, install)
    assert not (install / "repair-backups").exists()


def test_path_guards_reject_escapes_reserved_ui_collisions_and_symlinks(tmp_path, _guarded_engine):
    source, install, _ = _build_fake_roots(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-secret")

    with pytest.raises(ValueError, match="Unsafe"):
        deploy_mod.resolve_safe_path(source, "../outside.bin")
    with pytest.raises(ValueError, match="Unsafe"):
        deploy_mod.resolve_safe_path(install, str(outside))

    # Reject Release UI files targeting reserved backend/settings/model trees.
    _write_file(source / RELEASE_UI_REL / "temp/settings.json", b"{}")
    with pytest.raises(ValueError, match="reserved"):
        deploy_mod.plan_deployment(source, install)
    (source / RELEASE_UI_REL / "temp/settings.json").unlink()

    # Reject symlinks in Release UI and installation targets.
    release_link = source / RELEASE_UI_REL / "linked_asset.dll"
    try:
        release_link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this Windows host")

    with pytest.raises(ValueError, match="symlink|Unsafe"):
        deploy_mod.plan_deployment(source, install)
    release_link.unlink()

    install_target = install / "Neri.exe"
    install_target.unlink()
    install_target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|Unsafe"):
        deploy_mod.apply_deployment(source, install)
    assert outside.read_bytes() == b"outside-secret"


def test_apply_failure_rolls_back_and_manual_rollback_restores_ui_and_backend(
    tmp_path,
    _guarded_engine,
    monkeypatch,
):
    source, install, backend_files = _build_fake_roots(tmp_path)
    orig_neri_exe = _sha256(install / "Neri.exe")
    orig_app_so = _sha256(install / "data/app.so")
    orig_engine = _sha256(install / "flutter_windows.dll")
    orig_settings = _sha256(install / "temp/settings.json")

    # Simulate mid-transaction failure during UI replacement after earlier writes succeed.
    real_replace = os.replace

    def flaky_replace(src, dst):
        if Path(dst).name == "app.so" and ".deploy-tmp-" in Path(src).name:
            raise OSError("simulated UI atomic replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    failed_report_path = tmp_path / "failed_apply.json"
    with pytest.raises(OSError, match="simulated UI atomic replace failure"):
        deploy_mod.apply_deployment(source, install, report_path=failed_report_path)

    failed_report = json.loads(failed_report_path.read_text(encoding="utf-8"))
    assert failed_report["applied"] is False
    assert failed_report["rolled_back"] is True
    assert _sha256(install / "Neri.exe") == orig_neri_exe
    assert _sha256(install / "data/app.so") == orig_app_so
    assert not (install / backend_files[-1]).exists()

    # Clean apply followed by manual --rollback-from restores both UI and backend.
    monkeypatch.setattr(os, "replace", real_replace)
    apply_report_path = tmp_path / "clean_apply.json"
    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(apply_report_path),
        "--apply",
    ]) == 0
    backup_dir = Path(json.loads(apply_report_path.read_text(encoding="utf-8"))["backup_dir"])

    rollback_report_path = tmp_path / "manual_rollback.json"
    assert deploy_mod.main([
        "--installation-root", str(install),
        "--report", str(rollback_report_path),
        "--rollback-from", str(backup_dir),
    ]) == 0

    rb_report = json.loads(rollback_report_path.read_text(encoding="utf-8"))
    assert rb_report["rolled_back"] is True
    assert _sha256(install / "Neri.exe") == orig_neri_exe
    assert _sha256(install / "data/app.so") == orig_app_so
    assert _sha256(install / "flutter_windows.dll") == orig_engine
    assert _sha256(install / "temp/settings.json") == orig_settings
    assert not (install / backend_files[-1]).exists()
    assert not (install / "data/flutter_assets/AssetManifest.json").exists()

    outside_backup = tmp_path / "outside_backup"
    outside_backup.mkdir()
    with pytest.raises(ValueError, match="repair-backups"):
        deploy_mod.rollback_deployment(install, outside_backup)


@pytest.mark.parametrize(
    "custom_backend",
    [
        ("system/dinov3/detection_categories.py", "docs/validation-correctness-20261117.md"),
        ("system/dinov3/context.py", "system/dinov3/job_cache.py", "system/dino/image_processor.py"),
    ],
)
def test_parameterized_backend_file_list_fixture(
    tmp_path,
    _guarded_engine,
    monkeypatch,
    custom_backend,
):
    original_base_files = base_deploy_mod.EXPLICIT_FILES
    monkeypatch.setattr(deploy_mod, "EXPLICIT_FILES", custom_backend)
    source, install, backend_files = _build_fake_roots(tmp_path, backend_files=custom_backend)
    report_path = tmp_path / "param_apply.json"

    assert deploy_mod.main([
        "--source-root", str(source),
        "--installation-root", str(install),
        "--report", str(report_path),
        "--apply",
    ]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["backend_files_total"] == len(custom_backend)
    for rel in backend_files:
        assert _sha256(install / rel) == _sha256(source / rel)
    assert base_deploy_mod.EXPLICIT_FILES == original_base_files


def test_active_process_guard_and_base_explicit_files_unchanged(tmp_path, _guarded_engine, monkeypatch):
    source, install, _ = _build_fake_roots(tmp_path)
    install_neri = install / "Neri.exe"
    original_base_files = base_deploy_mod.EXPLICIT_FILES

    monkeypatch.setattr(deploy_mod, "psutil", None)
    assert deploy_mod.active_processes(install) == []
    assert deploy_mod.plan_deployment(source, install)["active_processes"] == []

    def fake_iter_active(*args, **kwargs):
        return [SimpleNamespace(pid=777, info={"pid": 777, "name": "Neri.exe"}, exe=lambda: str(install_neri))]

    monkeypatch.setattr(deploy_mod, "psutil", SimpleNamespace(process_iter=fake_iter_active, Error=Exception))
    with pytest.raises(RuntimeError, match="active|Close Neri"):
        deploy_mod.apply_deployment(source, install)
    assert base_deploy_mod.EXPLICIT_FILES == original_base_files
