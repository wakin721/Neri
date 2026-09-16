from pathlib import Path


def _source(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_legacy_dinov3_runtime_tree_is_retired():
    assert not Path("system/dinov3").exists()
    assert not list(Path("system/backend").glob("dinov3_*.py"))


def test_legacy_dinov3_tests_and_workflow_are_retired():
    assert not list(Path("tests").glob("test_dinov3*.py"))
    assert not Path("tests/dinov3_multi_prototype_fixtures.py").exists()
    assert not Path(".github/workflows/dinov3-implementation-verification.yml").exists()


def test_legacy_dinov3_active_artifacts_are_retired():
    for raw in (
        ".github/workflows/dinov2-retire-v3.yml",
        "docs/dinov3-ci-verified.txt",
        "scripts/calibrate_dinov3_multi_dual.py",
        "res/model/cls/user/README_DINOv3.md",
        "res/model/cls/user/dinov3_classifier_merged_reviewed_20260908.neri.json",
    ):
        assert not Path(raw).exists(), raw


def test_active_flutter_source_has_no_dinov3_symbols():
    for path in Path("frontend/lib").rglob("*.dart"):
        assert "dinov3" not in _source(path), path


def test_dinov2_registry_and_examples_have_no_dinov3_dependency():
    for raw in (
        "frontend/lib/src/models/dinov2_registry.dart",
        "system/dinov2/registry_examples.py",
    ):
        path = Path(raw)
        assert "dinov3" not in _source(path), path
