from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dinov2_encoder_is_offline_only():
    source = (ROOT / "system" / "dinov2" / "encoder.py").read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "trust_remote_code=False" in source
    assert "AutoImageProcessor" not in source


def test_dinov2_verification_workflow_runs_python_and_flutter_suites():
    workflow = (ROOT / ".github" / "workflows" / "dinov2-implementation-verification.yml").read_text(encoding="utf-8")
    assert "DINOv2 implementation verification" in workflow
    assert "test_dinov2" in workflow
    assert "flutter analyze" in workflow
    assert "flutter test" in workflow
    assert "dinov3" not in workflow.lower()


def test_lite_package_keeps_model_root_generic_for_optional_dinov2_component():
    package = (ROOT / "scripts" / "package_windows.ps1").read_text(encoding="utf-8")
    assert 'Copy-Item "res" $package -Recurse -Force' in package
    assert 'res\\model\\cls\\user' in package
    assert 'res\\model\\detect\\user' in package
    assert "DINOv3" not in package
