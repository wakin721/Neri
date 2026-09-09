from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')
old = """          _recordMarkHistory(<DetectionItem>[\n            updated,\n          ], quickMarkSpeciesName: usedQuickSpecies);\n"""
new = """          _recordMarkHistory(\n            <DetectionItem>[updated],\n            quickMarkSpeciesName: usedQuickSpecies,\n            feedbackOperationId: feedbackOperationId,\n          );\n"""
if old not in text:
    raise SystemExit('file validation history anchor not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
