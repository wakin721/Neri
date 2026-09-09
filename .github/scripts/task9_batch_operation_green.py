from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

anchor = """  Future<void> _markBatch(\n"""
helper = """  Future<List<DetectionItem>> _callMarkItems(\n    List<DetectionItem> items,\n    String action, {\n    String? speciesName,\n    String? speciesCount,\n    String? speciesType,\n    String? remark,\n    String? feedbackOperationId,\n  }) {\n    final callback = widget.onMarkItems;\n    if (feedbackOperationId != null &&\n        callback is MarkValidationItemsWithFeedback) {\n      return callback(\n        items,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n        feedbackOperationId: feedbackOperationId,\n      );\n    }\n    return callback(\n      items,\n      action,\n      speciesName: speciesName,\n      speciesCount: speciesCount,\n      speciesType: speciesType,\n      remark: remark,\n    );\n  }\n\n"""
if anchor not in text:
    raise SystemExit('mark batch anchor not found')
text = text.replace(anchor, helper + anchor, 1)

old = """    final visibleBefore = _visibleItems(_currentBuckets());\n    final nextPath = _nextPathAfterBatch(visibleBefore, items);\n    _deferRegroupForItems(items);\n\n    _setMarking(true);\n    try {\n      final updatedItems = await widget.onMarkItems(\n        items,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n      );\n"""
new = """    final visibleBefore = _visibleItems(_currentBuckets());\n    final nextPath = _nextPathAfterBatch(visibleBefore, items);\n    _deferRegroupForItems(items);\n    final feedbackOperationId = _newValidationFeedbackOperationId(action);\n\n    _setMarking(true);\n    try {\n      final updatedItems = await _callMarkItems(\n        items,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n        feedbackOperationId: feedbackOperationId,\n      );\n"""
if old not in text:
    raise SystemExit('batch callback anchor not found')
text = text.replace(old, new, 1)

old = """          _recordMarkHistory(\n            updatedItems,\n            quickMarkSpeciesName: usedQuickSpecies,\n          );\n"""
new = """          _recordMarkHistory(\n            updatedItems,\n            quickMarkSpeciesName: usedQuickSpecies,\n            feedbackOperationId: feedbackOperationId,\n          );\n"""
if old not in text:
    raise SystemExit('batch history anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
