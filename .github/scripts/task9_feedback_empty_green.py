from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')
old = """            OutlinedButton(onPressed: null, child: const Text('空 / 误检')),\n"""
new = """            OutlinedButton(\n              onPressed: canSubmit\n                  ? () => unawaited(_submitDinoBoxFeedback(box, 'empty'))\n                  : null,\n              child: const Text('空 / 误检'),\n            ),\n"""
if old not in text:
    raise SystemExit('empty feedback button anchor not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
