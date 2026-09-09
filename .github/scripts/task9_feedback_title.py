from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')
old = """    final predicted = box.predictedSpecies?.trim();
    final title = predicted == null || predicted.isEmpty
        ? '检测框校验'
        : '检测框校验 · $predicted';
"""
new = """    const title = '检测框校验';
"""
if old not in text:
    raise SystemExit('feedback title anchor not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
