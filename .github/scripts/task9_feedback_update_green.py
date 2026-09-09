from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

anchor = """  Widget _buildDinoFeedbackPanel(DetectionBox box) {\n"""
helper = """  Future<void> _showDinoBoxSpeciesDialog(DetectionBox box) async {\n    if (_marking) return;\n    final predictedSpecies = box.predictedSpecies?.trim() ?? '';\n    final initialSpecies = predictedSpecies.isNotEmpty\n        ? predictedSpecies\n        : box.species.trim();\n    final draft = await showDialog<_OtherSpeciesDraft>(\n      context: context,\n      builder: (context) {\n        return _OtherSpeciesDialog(\n          initialSpecies: initialSpecies,\n          initialCount: '1',\n          initialType: widget.speciesTypes[initialSpecies] ?? '',\n          initialRemark: '',\n          speciesTypes: widget.speciesTypes,\n          speciesUsageCounts: _speciesUsageCounts(),\n        );\n      },\n    );\n    if (!mounted || draft == null) return;\n    final speciesName = draft.speciesName.trim();\n    if (speciesName.isEmpty) return;\n    await _submitDinoBoxFeedback(\n      box,\n      'update',\n      speciesName: speciesName,\n    );\n  }\n\n"""
if anchor not in text:
    raise SystemExit('feedback panel anchor not found')
text = text.replace(anchor, helper + anchor, 1)

old = """            OutlinedButton(onPressed: null, child: const Text('修改物种')),\n"""
new = """            OutlinedButton(\n              onPressed: canSubmit\n                  ? () => unawaited(_showDinoBoxSpeciesDialog(box))\n                  : null,\n              child: const Text('修改物种'),\n            ),\n"""
if old not in text:
    raise SystemExit('update feedback button anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
