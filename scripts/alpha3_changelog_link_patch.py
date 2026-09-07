from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
old = "[更新日志](res/demo/README_Update.md)"
new = "[更新日志](CHANGELOG.md)"
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("README changelog link anchor not found")
readme.write_text(text, encoding="utf-8")

settings = ROOT / "frontend/lib/src/screens/settings_screen.dart"
text = settings.read_text(encoding="utf-8")
old = "https://github.com/wakin721/Neri/blob/main/res/demo/README_Update.md"
new = "https://github.com/wakin721/Neri/blob/main/CHANGELOG.md"
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("settings changelog URL anchor not found")
settings.write_text(text, encoding="utf-8")
