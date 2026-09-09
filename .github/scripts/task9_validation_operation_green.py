from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

anchor = """typedef MarkValidationItems =\n    Future<List<DetectionItem>> Function(\n      List<DetectionItem> items,\n      String action, {\n      String? speciesName,\n      String? speciesCount,\n      String? speciesType,\n      String? remark,\n    });\n\n"""
addition = """typedef MarkValidationItemWithFeedback =\n    Future<DetectionItem> Function(\n      DetectionItem item,\n      String action, {\n      String? speciesName,\n      String? speciesCount,\n      String? speciesType,\n      String? remark,\n      String? feedbackOperationId,\n    });\n\ntypedef MarkValidationItemsWithFeedback =\n    Future<List<DetectionItem>> Function(\n      List<DetectionItem> items,\n      String action, {\n      String? speciesName,\n      String? speciesCount,\n      String? speciesType,\n      String? remark,\n      String? feedbackOperationId,\n    });\n\n"""
if anchor not in text:
    raise SystemExit('validation typedef anchor not found')
text = text.replace(anchor, anchor + addition, 1)

anchor = """  Future<void> _markBatch(\n"""
helpers = """  String? _newValidationFeedbackOperationId(String action) {\n    final classificationModelPath = widget.classificationModelPath?.trim() ?? '';\n    if (action == 'unverified' || classificationModelPath.isEmpty) return null;\n    return _newFeedbackOperationId();\n  }\n\n  Future<DetectionItem> _callMarkItem(\n    DetectionItem item,\n    String action, {\n    String? speciesName,\n    String? speciesCount,\n    String? speciesType,\n    String? remark,\n    String? feedbackOperationId,\n  }) {\n    final callback = widget.onMarkItem;\n    if (feedbackOperationId != null &&\n        callback is MarkValidationItemWithFeedback) {\n      return callback(\n        item,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n        feedbackOperationId: feedbackOperationId,\n      );\n    }\n    return callback(\n      item,\n      action,\n      speciesName: speciesName,\n      speciesCount: speciesCount,\n      speciesType: speciesType,\n      remark: remark,\n    );\n  }\n\n"""
if anchor not in text:
    raise SystemExit('mark batch anchor not found')
text = text.replace(anchor, helpers + anchor, 1)

old = """    _setMarking(true);\n    try {\n      final updated = await widget.onMarkItem(\n        item,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n      );\n"""
new = """    final feedbackOperationId = _newValidationFeedbackOperationId(action);\n    _setMarking(true);\n    try {\n      final updated = await _callMarkItem(\n        item,\n        action,\n        speciesName: speciesName,\n        speciesCount: speciesCount,\n        speciesType: speciesType,\n        remark: remark,\n        feedbackOperationId: feedbackOperationId,\n      );\n"""
if old not in text:
    raise SystemExit('mark selected callback anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
