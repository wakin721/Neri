from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

old = """class _MarkHistoryEntry {\n  _MarkHistoryEntry(\n    Iterable<DetectionItem> items, {\n    Iterable<String>? quickMarkSpecies,\n  }) : items = items.toList(),\n       quickMarkSpecies = List<String>.from(\n         quickMarkSpecies ?? const <String>[],\n       );\n\n  final List<DetectionItem> items;\n  final List<String> quickMarkSpecies;\n}\n"""
new = """class _MarkHistoryEntry {\n  _MarkHistoryEntry(\n    Iterable<DetectionItem> items, {\n    this.feedbackOperationId,\n    Iterable<String>? quickMarkSpecies,\n  }) : items = items.toList(),\n       quickMarkSpecies = List<String>.from(\n         quickMarkSpecies ?? const <String>[],\n       );\n\n  final List<DetectionItem> items;\n  final List<String> quickMarkSpecies;\n  final String? feedbackOperationId;\n}\n"""
if old not in text:
    raise SystemExit('history entry anchor not found')
text = text.replace(old, new, 1)

old = """      await widget.apiClient.markDinoV3BoxFeedback(\n        inputPath: widget.inputPath,\n        filePath: item.path,\n        classificationModelPath: classificationModelPath,\n        observationId: observationId,\n        action: action,\n        speciesName: speciesName,\n        feedbackOperationId: operationId,\n      );\n      if (!mounted) return;\n      _showSnackBar('已记录检测框反馈');\n"""
new = """      final result = await widget.apiClient.markDinoV3BoxFeedback(\n        inputPath: widget.inputPath,\n        filePath: item.path,\n        classificationModelPath: classificationModelPath,\n        observationId: observationId,\n        action: action,\n        speciesName: speciesName,\n        feedbackOperationId: operationId,\n      );\n      if (!mounted) return;\n      final recordedOperationId = result.operationId.trim().isEmpty\n          ? operationId\n          : result.operationId.trim();\n      _recordMarkHistory(\n        <DetectionItem>[result.item],\n        feedbackOperationId: recordedOperationId,\n      );\n      _showSnackBar('已记录检测框反馈');\n"""
if old not in text:
    raise SystemExit('submit feedback anchor not found')
text = text.replace(old, new, 1)

old = """  void _recordMarkHistory(\n    Iterable<DetectionItem> items, {\n    String? quickMarkSpeciesName,\n  }) {\n"""
new = """  void _recordMarkHistory(\n    Iterable<DetectionItem> items, {\n    String? quickMarkSpeciesName,\n    String? feedbackOperationId,\n  }) {\n"""
if old not in text:
    raise SystemExit('record history signature anchor not found')
text = text.replace(old, new, 1)

old = """      _MarkHistoryEntry(\n        itemsByPath.values,\n        quickMarkSpecies: quickMarkSpeciesName == null\n            ? const <String>[]\n            : _splitSpeciesNames(quickMarkSpeciesName),\n      ),\n"""
new = """      _MarkHistoryEntry(\n        itemsByPath.values,\n        feedbackOperationId: feedbackOperationId,\n        quickMarkSpecies: quickMarkSpeciesName == null\n            ? const <String>[]\n            : _splitSpeciesNames(quickMarkSpeciesName),\n      ),\n"""
if old not in text:
    raise SystemExit('record history entry anchor not found')
text = text.replace(old, new, 1)

old = """    final quickMarkSpeciesToUndo = List<String>.from(\n      _markHistory.last.quickMarkSpecies,\n    );\n\n    final visibleBefore = _visibleItems(_currentBuckets());\n"""
new = """    final quickMarkSpeciesToUndo = List<String>.from(\n      _markHistory.last.quickMarkSpecies,\n    );\n    final feedbackOperationId = _markHistory.last.feedbackOperationId?.trim();\n\n    final visibleBefore = _visibleItems(_currentBuckets());\n"""
if old not in text:
    raise SystemExit('undo capture anchor not found')
text = text.replace(old, new, 1)

old = """      final updatedItems = await widget.onMarkItems(targets, 'unverified');\n      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n      if (!mounted) return;\n"""
new = """      final updatedItems = await widget.onMarkItems(targets, 'unverified');\n      if (feedbackOperationId != null && feedbackOperationId.isNotEmpty) {\n        final classificationModelPath =\n            widget.classificationModelPath?.trim() ?? '';\n        if (classificationModelPath.isEmpty) {\n          throw StateError('缺少 DINOv3 分类模型路径，无法撤回学习反馈。');\n        }\n        await widget.apiClient.revertDinoV3Feedback(\n          classificationModelPath: classificationModelPath,\n          feedbackOperationId: feedbackOperationId,\n        );\n      }\n      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n      if (!mounted) return;\n"""
if old not in text:
    raise SystemExit('undo revert anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
