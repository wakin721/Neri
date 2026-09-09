from pathlib import Path

path = Path('frontend/lib/src/main_window.dart')
text = path.read_text(encoding='utf-8')

anchor = """const _lastInputPathKey = 'last_input_path';\n\n"""
addition = """String? resolveDinoV3ValidationModelPath(\n  ModelInfo? model,\n  String? selectedPath,\n) {\n  final path = selectedPath?.trim() ?? '';\n  if (model?.isDinoV3 != true || path.isEmpty) return null;\n  return path;\n}\n\n"""
if anchor not in text:
    raise SystemExit('top-level insertion anchor not found')
text = text.replace(anchor, anchor + addition, 1)

anchor = """  bool _useAugment() {\n"""
helper = """  String? _selectedDinoV3ValidationModelPath() {\n    return resolveDinoV3ValidationModelPath(\n      _selectedClassificationModelInfo(),\n      _selectedClassificationModelPath,\n    );\n  }\n\n"""
if anchor not in text:
    raise SystemExit('selected model helper anchor not found')
text = text.replace(anchor, helper + anchor, 1)

old = """    String? speciesType,\n    String? remark,\n  }) async {\n    return _runValidationBusy(() async {\n"""
new = """    String? speciesType,\n    String? remark,\n    String? feedbackOperationId,\n  }) async {\n    return _runValidationBusy(() async {\n"""
if old not in text:
    raise SystemExit('single validation signature anchor not found')
text = text.replace(old, new, 1)

old = """        speciesType: speciesType,\n        remark: remark,\n      );\n      final merged = _mergeValidationUpdate(item, updated);\n"""
new = """        speciesType: speciesType,\n        remark: remark,\n        classificationModelPath: _selectedDinoV3ValidationModelPath(),\n        feedbackOperationId: feedbackOperationId,\n      );\n      final merged = _mergeValidationUpdate(item, updated);\n"""
if old not in text:
    raise SystemExit('single validation API anchor not found')
text = text.replace(old, new, 1)

old = """    String? speciesType,\n    String? remark,\n  }) async {\n    return _runValidationBusy(() async {\n"""
new = """    String? speciesType,\n    String? remark,\n    String? feedbackOperationId,\n  }) async {\n    return _runValidationBusy(() async {\n"""
if old not in text:
    raise SystemExit('batch validation signature anchor not found')
text = text.replace(old, new, 1)

old = """        speciesType: speciesType,\n        remark: remark,\n      );\n      final fallbackByPath = <String, DetectionItem>{\n"""
new = """        speciesType: speciesType,\n        remark: remark,\n        classificationModelPath: _selectedDinoV3ValidationModelPath(),\n        feedbackOperationId: feedbackOperationId,\n      );\n      final fallbackByPath = <String, DetectionItem>{\n"""
if old not in text:
    raise SystemExit('batch validation API anchor not found')
text = text.replace(old, new, 1)

old = """    final validationScreen = SpeciesValidationScreen(\n      apiClient: widget.apiClient,\n      inputPath: inputPath,\n      items: items,\n"""
new = """    final validationScreen = SpeciesValidationScreen(\n      apiClient: widget.apiClient,\n      inputPath: inputPath,\n      classificationModelPath: _selectedDinoV3ValidationModelPath(),\n      items: items,\n"""
if old not in text:
    raise SystemExit('validation screen wiring anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
