from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

old = """    required this.apiClient,\n    required this.inputPath,\n    required this.items,\n"""
new = """    required this.apiClient,\n    required this.inputPath,\n    this.classificationModelPath,\n    required this.items,\n"""
if old not in text:
    raise SystemExit('constructor anchor not found')
text = text.replace(old, new, 1)

old = """  final NeriApiClient apiClient;\n  final String inputPath;\n  final List<DetectionItem> items;\n"""
new = """  final NeriApiClient apiClient;\n  final String inputPath;\n  final String? classificationModelPath;\n  final List<DetectionItem> items;\n"""
if old not in text:
    raise SystemExit('field anchor not found')
text = text.replace(old, new, 1)

old = """  bool _marking = false;\n  bool _exporting = false;\n  String? _selectedObservationId;\n"""
new = """  bool _marking = false;\n  bool _exporting = false;\n  String? _selectedObservationId;\n  int _feedbackOperationSequence = 0;\n"""
if old not in text:
    raise SystemExit('state anchor not found')
text = text.replace(old, new, 1)

old = """  Widget _buildDinoFeedbackPanel(DetectionBox box) {\n    const title = '检测框校验';\n    return _ValidationPanel(\n"""
new = """  String _newFeedbackOperationId() {\n    _feedbackOperationSequence += 1;\n    return 'feedback-${DateTime.now().microsecondsSinceEpoch}-$_feedbackOperationSequence';\n  }\n\n  Future<void> _submitDinoBoxFeedback(\n    DetectionBox box,\n    String action, {\n    String? speciesName,\n  }) async {\n    if (_marking || widget.items.isEmpty) return;\n    final observationId = box.observationId?.trim() ?? '';\n    final classificationModelPath = widget.classificationModelPath?.trim() ?? '';\n    if (observationId.isEmpty || classificationModelPath.isEmpty) return;\n    final item = widget.items.firstWhere(\n      (candidate) => candidate.path == _selectedPath,\n      orElse: () => widget.items.first,\n    );\n    final operationId = _newFeedbackOperationId();\n\n    _setMarking(true);\n    try {\n      await widget.apiClient.markDinoV3BoxFeedback(\n        inputPath: widget.inputPath,\n        filePath: item.path,\n        classificationModelPath: classificationModelPath,\n        observationId: observationId,\n        action: action,\n        speciesName: speciesName,\n        feedbackOperationId: operationId,\n      );\n      if (!mounted) return;\n      _showSnackBar('已记录检测框反馈');\n    } catch (error) {\n      if (!mounted) return;\n      _showSnackBar('检测框反馈失败：$error');\n    } finally {\n      _setMarking(false);\n    }\n  }\n\n  Widget _buildDinoFeedbackPanel(DetectionBox box) {\n    const title = '检测框校验';\n    final classificationModelPath = widget.classificationModelPath?.trim() ?? '';\n    final observationId = box.observationId?.trim() ?? '';\n    final canSubmit =\n        !_marking && classificationModelPath.isNotEmpty && observationId.isNotEmpty;\n    return _ValidationPanel(\n"""
if old not in text:
    raise SystemExit('feedback-panel anchor not found')
text = text.replace(old, new, 1)

old = """            const SizedBox(width: 12),\n            OutlinedButton(onPressed: null, child: const Text('正确')),\n            const SizedBox(width: 8),\n"""
new = """            const SizedBox(width: 12),\n            OutlinedButton(\n              onPressed: canSubmit\n                  ? () => unawaited(_submitDinoBoxFeedback(box, 'correct'))\n                  : null,\n              child: const Text('正确'),\n            ),\n            const SizedBox(width: 8),\n"""
if old not in text:
    raise SystemExit('correct-button anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
