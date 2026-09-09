from pathlib import Path

path = Path('frontend/lib/src/api_client_core.dart')
text = path.read_text(encoding='utf-8')

old = """    String? speciesType,\n    String? remark,\n  }) async {\n"""
new = """    String? speciesType,\n    String? remark,\n    String? classificationModelPath,\n    String? feedbackOperationId,\n  }) async {\n"""
if old not in text:
    raise SystemExit('single validation signature anchor not found')
text = text.replace(old, new, 1)

old = """        if (speciesType != null) 'species_type': speciesType,\n        if (remark != null) 'remark': remark,\n      }),\n"""
new = """        if (speciesType != null) 'species_type': speciesType,\n        if (remark != null) 'remark': remark,\n        if (classificationModelPath != null &&\n            classificationModelPath.trim().isNotEmpty)\n          'classification_model_path': classificationModelPath.trim(),\n        if (feedbackOperationId != null &&\n            feedbackOperationId.trim().isNotEmpty)\n          'feedback_operation_id': feedbackOperationId.trim(),\n      }),\n"""
if old not in text:
    raise SystemExit('single validation body anchor not found')
text = text.replace(old, new, 1)

old = """    String? speciesType,\n    String? remark,\n  }) async {\n    final response = await _httpClient.post(\n      _uri('/api/validation/mark/batch'),\n"""
new = """    String? speciesType,\n    String? remark,\n    String? classificationModelPath,\n    String? feedbackOperationId,\n  }) async {\n    final response = await _httpClient.post(\n      _uri('/api/validation/mark/batch'),\n"""
if old not in text:
    raise SystemExit('batch validation signature anchor not found')
text = text.replace(old, new, 1)

old = """        if (speciesType != null) 'species_type': speciesType,\n        if (remark != null) 'remark': remark,\n      }),\n    );\n    try {\n"""
new = """        if (speciesType != null) 'species_type': speciesType,\n        if (remark != null) 'remark': remark,\n        if (classificationModelPath != null &&\n            classificationModelPath.trim().isNotEmpty)\n          'classification_model_path': classificationModelPath.trim(),\n        if (feedbackOperationId != null &&\n            feedbackOperationId.trim().isNotEmpty)\n          'feedback_operation_id': feedbackOperationId.trim(),\n      }),\n    );\n    try {\n"""
if old not in text:
    raise SystemExit('batch validation body anchor not found')
text = text.replace(old, new, 1)

old = """            speciesType: speciesType,\n            remark: remark,\n          ),\n"""
new = """            speciesType: speciesType,\n            remark: remark,\n            classificationModelPath: classificationModelPath,\n            feedbackOperationId: feedbackOperationId,\n          ),\n"""
if old not in text:
    raise SystemExit('batch fallback anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
