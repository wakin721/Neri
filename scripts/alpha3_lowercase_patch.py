from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOWERCASE_PATH_FILES = (
    Path("README.md"),
    Path("docs/superpowers/specs/2026-09-07-model-sync-design.md"),
    Path("docs/superpowers/plans/2026-09-07-model-sync.md"),
    Path("frontend/lib/src/screens/start_screen.dart"),
    Path("frontend/test/model_selection_test.dart"),
)

for relative in LOWERCASE_PATH_FILES:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    updated = text.replace("res/Model", "res/model")
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print(f"updated local model paths: {relative}")

settings_path = ROOT / "frontend/lib/src/screens/settings_screen.dart"
settings = settings_path.read_text(encoding="utf-8")
old_version = "defaultValue: '3.0.6-alpha2(0f6ac7)'"
new_version = "defaultValue: '3.0.6-alpha3(0f6ac7)'"
if old_version in settings:
    settings = settings.replace(old_version, new_version, 1)
elif new_version not in settings:
    raise SystemExit("settings_screen.dart frontend version anchor not found")
settings_path.write_text(settings, encoding="utf-8")
print("updated settings fallback version to alpha3")
