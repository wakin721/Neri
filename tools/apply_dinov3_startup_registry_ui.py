from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def replace_count(text: str, old: str, new: str, expected: int, *, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


startup_path = Path("frontend/lib/src/dinov3_startup_check.dart")
startup_path.write_text(
    """import 'api_client.dart';

String? dinoV3StartupStatusMessage(DinoV3ComponentStatus status) {
  if (!status.installed) {
    return 'DINOv3 尚未安装，可进入设置安装。';
  }
  if (status.healthy) return null;

  final reason = status.message.trim();
  final lower = reason.toLowerCase();
  final mentionsTorch = lower.contains('pytorch') || lower.contains('torch');
  final indicatesMissing =
      reason.contains('未安装') ||
      lower.contains('not installed') ||
      lower.contains('no module named') ||
      lower.contains('missing') ||
      lower.contains('unavailable');
  if (mentionsTorch && indicatesMissing) {
    return 'DINOv3 已安装，但 PyTorch 未安装，当前不可推理';
  }
  if (reason.isEmpty) return 'DINOv3 安装异常';
  return 'DINOv3 安装异常：$reason';
}

class DinoV3StartupCheck {
  int? _lastGeneration;

  Future<void> run({
    required int generation,
    required NeriApiClient apiClient,
    required void Function(String message) onMessage,
  }) async {
    if (generation <= 0 || _lastGeneration == generation) return;
    _lastGeneration = generation;
    try {
      final status = await apiClient.fetchDinoV3ComponentStatus();
      if (_lastGeneration != generation) return;
      final message = dinoV3StartupStatusMessage(status);
      if (message != null && message.isNotEmpty) onMessage(message);
    } catch (_) {
      // The backend itself is already usable at this point. A transient status
      // endpoint failure must not block startup or generate a misleading alert.
    }
  }
}
""",
    encoding="utf-8",
)


# Wire a single checker instance into NeriApp so its backend-generation memory
# survives MaterialApp rebuilds caused by theme changes.
path = "frontend/lib/main.dart"
text = read(path)
text = replace_once(
    text,
    "import 'src/crash_watchdog.dart';\nimport 'src/models/theme_settings.dart';",
    "import 'src/crash_watchdog.dart';\nimport 'src/dinov3_startup_check.dart';\nimport 'src/models/theme_settings.dart';",
    label="main import",
)
text = replace_once(
    text,
    "  // MainWindow closes this client after shutting down the backend.\n  final _apiClient = NeriApiClient();",
    "  // MainWindow closes this client after shutting down the backend.\n  final _apiClient = NeriApiClient();\n  final _dinoV3StartupCheck = DinoV3StartupCheck();",
    label="main checker field",
)
text = replace_once(
    text,
    "              home: MainWindow(\n                apiClient: _apiClient,\n                themeNotifier: widget.themeNotifier,\n              ),",
    "              home: MainWindow(\n                apiClient: _apiClient,\n                themeNotifier: widget.themeNotifier,\n                dinoV3StartupCheck: _dinoV3StartupCheck,\n              ),",
    label="main checker wiring",
)
write(path, text)


# MainWindow assigns a monotonically increasing generation to each successful
# backend start/restart. The checker is invoked only after _refresh() has loaded
# /api/settings, never from the 2-second silent refresh timer.
path = "frontend/lib/src/main_window.dart"
text = read(path)
text = replace_once(
    text,
    "import 'crash_watchdog.dart';\nimport 'local_maintenance_status.dart';",
    "import 'crash_watchdog.dart';\nimport 'dinov3_startup_check.dart';\nimport 'local_maintenance_status.dart';",
    label="main_window import",
)
text = replace_once(
    text,
    "  const MainWindow({\n    required this.apiClient,\n    required this.themeNotifier,\n    super.key,\n  });",
    "  const MainWindow({\n    required this.apiClient,\n    required this.themeNotifier,\n    this.dinoV3StartupCheck,\n    super.key,\n  });",
    label="main_window constructor",
)
text = replace_once(
    text,
    "  final NeriApiClient apiClient;\n  final ValueNotifier<ThemeSettings> themeNotifier;",
    "  final NeriApiClient apiClient;\n  final ValueNotifier<ThemeSettings> themeNotifier;\n  final DinoV3StartupCheck? dinoV3StartupCheck;",
    label="main_window field",
)
text = replace_once(
    text,
    "  final _maintenanceStatusStore = LocalMaintenanceStatusStore();\n  final _appUpdater = AppUpdater();",
    "  final _maintenanceStatusStore = LocalMaintenanceStatusStore();\n  final _appUpdater = AppUpdater();\n  late final DinoV3StartupCheck _dinoV3StartupCheck =\n      widget.dinoV3StartupCheck ?? DinoV3StartupCheck();",
    label="main_window checker state",
)
text = replace_once(
    text,
    "  int _modelSelectionRevision = 0;\n  int _closeFlowId = 0;",
    "  int _modelSelectionRevision = 0;\n  int _backendStartupGeneration = 0;\n  int _closeFlowId = 0;",
    label="main_window generation field",
)
text = replace_once(
    text,
    "      _backendReady = true;\n      final privacyReady = await _loadPrivacyStatus();",
    "      _backendReady = true;\n      _backendStartupGeneration += 1;\n      final privacyReady = await _loadPrivacyStatus();",
    label="main_window generation increment",
)
text = replace_once(
    text,
    "  Future<void> _refreshInitialPageData() async {\n    await _refresh(includeJobResults: true, finishLoading: false);\n    if (!mounted || _closeFlowBlocksBackendStartup) return;\n    _scheduleStartupUpdateCheck();",
    "  Future<void> _refreshInitialPageData() async {\n    await _refresh(includeJobResults: true, finishLoading: false);\n    if (!mounted || _closeFlowBlocksBackendStartup) return;\n    unawaited(\n      _dinoV3StartupCheck.run(\n        generation: _backendStartupGeneration,\n        apiClient: widget.apiClient,\n        onMessage: _showSnackBar,\n      ),\n    );\n    _scheduleStartupUpdateCheck();",
    label="main_window post settings check",
)
write(path, text)


# Keep registry event metadata needed by the UI while remaining backward
# compatible with old registry rows that have no crop metadata.
path = "frontend/lib/src/models/dinov3_registry.dart"
text = read(path)
text = replace_once(
    text,
    "  bool get isCandidate => status == 'candidate';",
    "  bool get isCandidate => status == 'candidate';\n\n  bool get canDelete => const <String>{\n    'candidate',\n    'provisional',\n    'confirmed',\n    'mature',\n  }.contains(status.toLowerCase());",
    label="registry canDelete",
)
text = replace_once(
    text,
    "  const DinoV3RegistryEvent({\n    required this.eventKey,\n    required this.sourcePath,\n    required this.cameraId,\n    required this.sampleCount,\n    required this.timestampMissing,\n    this.startedAt,\n    this.endedAt,\n  });",
    "  const DinoV3RegistryEvent({\n    required this.eventKey,\n    required this.sourcePath,\n    required this.cameraId,\n    required this.sampleCount,\n    required this.timestampMissing,\n    this.id = 0,\n    this.bbox = const <double>[],\n    this.frameIndex,\n    this.timestampSeconds,\n    this.hasExample = false,\n    this.startedAt,\n    this.endedAt,\n  });",
    label="registry event constructor",
)
text = replace_once(
    text,
    "    return DinoV3RegistryEvent(\n      eventKey: json['event_key']?.toString() ?? '',",
    "    return DinoV3RegistryEvent(\n      id: (json['id'] as num?)?.toInt() ?? 0,\n      eventKey: json['event_key']?.toString() ?? '',",
    label="registry event id parse",
)
text = replace_once(
    text,
    "      timestampMissing: json['timestamp_missing'] == true,\n      startedAt: json['started_at']?.toString(),",
    "      timestampMissing: json['timestamp_missing'] == true,\n      bbox: (json['bbox'] as List<dynamic>? ?? const <dynamic>[])\n          .whereType<num>()\n          .map((value) => value.toDouble())\n          .toList(),\n      frameIndex: (json['frame_index'] as num?)?.toInt(),\n      timestampSeconds: (json['timestamp_seconds'] as num?)?.toDouble(),\n      hasExample: json['has_example'] == true,\n      startedAt: json['started_at']?.toString(),",
    label="registry event crop parse",
)
text = replace_once(
    text,
    "  final String eventKey;\n  final String sourcePath;",
    "  final int id;\n  final String eventKey;\n  final String sourcePath;",
    label="registry event id field",
)
text = replace_once(
    text,
    "  final bool timestampMissing;\n  final String? startedAt;",
    "  final bool timestampMissing;\n  final List<double> bbox;\n  final int? frameIndex;\n  final double? timestampSeconds;\n  final bool hasExample;\n  final String? startedAt;",
    label="registry event crop fields",
)
write(path, text)


# Add the two lightweight registry management calls. Example bytes are served
# from the existing backend and are never written to a new local cache here.
path = "frontend/lib/src/api_client_core.dart"
text = read(path)
text = replace_once(
    text,
    "import 'dart:convert';",
    "import 'dart:convert';\nimport 'dart:typed_data';",
    label="api typed_data import",
)
marker = "  Future<DinoV3RegistryEntry> updateDinoV3RegistryIdentity({"
insert = """  Future<Uint8List> fetchDinoV3RegistryExample(
    String classificationModelPath,
    int registrationId,
    int eventId,
  ) async {
    final uri = _uri(
      '/api/dinov3/registry/$registrationId/events/$eventId/example',
    ).replace(
      queryParameters: {'classification_model_path': classificationModelPath},
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return response.bodyBytes;
  }

  Future<void> deleteDinoV3RegistryEntry(
    String classificationModelPath,
    int registrationId,
  ) async {
    final uri = _uri('/api/dinov3/registry/$registrationId').replace(
      queryParameters: {'classification_model_path': classificationModelPath},
    );
    final response = await _httpClient.delete(uri);
    _ensureSuccess(response);
  }

"""
if marker not in text:
    raise RuntimeError("api registry insertion marker missing")
text = text.replace(marker, insert + marker, 1)
write(path, text)


# Registry dialog: asynchronously load at most three crop examples, expose
# deletion only for local registry lifecycle states, and keep audit feedback
# untouched because the DELETE API only removes registrations/events/prototypes.
path = "frontend/lib/src/widgets/dinov3_registry_dialog.dart"
text = read(path)
text = replace_once(
    text,
    "import 'dart:async';",
    "import 'dart:async';\nimport 'dart:typed_data';",
    label="dialog typed_data import",
)
text = replace_once(
    text,
    "  bool _saving = false;\n  String? _error;",
    "  bool _saving = false;\n  bool _examplesLoading = false;\n  int _examplesRequestId = 0;\n  List<DinoV3RegistryEvent> _events = const <DinoV3RegistryEvent>[];\n  Map<int, Uint8List> _exampleBytes = const <int, Uint8List>{};\n  String? _error;",
    label="dialog example state",
)
old_select = """  void _select(DinoV3RegistryEntry entry, {bool notify = true}) {
    void update() {
      _selected = entry;
      _commonNameController.text = entry.commonName;
      _scientificNameController.text = entry.scientificName;
      _error = null;
    }

    if (notify) {
      setState(update);
    } else {
      update();
    }
  }
"""
new_select = """  void _select(DinoV3RegistryEntry entry, {bool notify = true}) {
    void update() {
      _selected = entry;
      _commonNameController.text = entry.commonName;
      _scientificNameController.text = entry.scientificName;
      _events = const <DinoV3RegistryEvent>[];
      _exampleBytes = const <int, Uint8List>{};
      _error = null;
    }

    if (notify) {
      setState(update);
    } else {
      update();
    }
    unawaited(_loadExamples(entry));
  }
"""
text = replace_once(text, old_select, new_select, label="dialog select")

marker = "  Future<void> _continueValidation() async {"
insert = """  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {
    final requestId = ++_examplesRequestId;
    if (mounted && _selected?.id == entry.id) {
      setState(() => _examplesLoading = true);
    }
    try {
      final events = await widget.apiClient.fetchDinoV3RegistryEvents(
        widget.modelPath,
        entry.id,
      );
      final bytes = <int, Uint8List>{};
      for (final event in events
          .where((event) => event.id > 0 && event.hasExample)
          .take(3)) {
        try {
          final data = await widget.apiClient.fetchDinoV3RegistryExample(
            widget.modelPath,
            entry.id,
            event.id,
          );
          if (data.isNotEmpty) bytes[event.id] = data;
        } catch (_) {
          // One stale source file must not hide the other representative crops.
        }
      }
      if (!mounted ||
          requestId != _examplesRequestId ||
          _selected?.id != entry.id) {
        return;
      }
      setState(() {
        _events = events;
        _exampleBytes = bytes;
        _examplesLoading = false;
      });
    } catch (_) {
      if (!mounted ||
          requestId != _examplesRequestId ||
          _selected?.id != entry.id) {
        return;
      }
      setState(() {
        _events = const <DinoV3RegistryEvent>[];
        _exampleBytes = const <int, Uint8List>{};
        _examplesLoading = false;
      });
    }
  }

  Future<void> _deleteSelected() async {
    final selected = _selected;
    if (selected == null || !selected.canDelete || _saving) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('删除“${selected.displayName}”？'),
        content: const Text(
          '将删除该物种的本地注册记录、事件关联和本地 prototypes。\\n'
          '此操作不可撤销。\\n\\n'
          '历史 human-feedback / audit 数据将保留。',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('确认删除'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      await widget.apiClient.deleteDinoV3RegistryEntry(
        widget.modelPath,
        selected.id,
      );
      if (!mounted) return;
      final remaining = _entries
          .where((entry) => entry.id != selected.id)
          .toList(growable: false);
      ++_examplesRequestId;
      setState(() {
        _entries = remaining;
        _selected = null;
        _events = const <DinoV3RegistryEvent>[];
        _exampleBytes = const <int, Uint8List>{};
        _examplesLoading = false;
        _commonNameController.clear();
        _scientificNameController.clear();
      });
      if (remaining.isNotEmpty) _select(remaining.first);
    } catch (error) {
      if (mounted) setState(() => _error = '删除物种失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Widget _buildExampleGallery() {
    if (_examplesLoading) {
      return const SizedBox(
        height: 104,
        child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
      );
    }
    final examples = _events
        .where((event) => event.hasExample && _exampleBytes.containsKey(event.id))
        .take(3)
        .toList(growable: false);
    if (examples.isEmpty) {
      return SizedBox(
        height: 72,
        child: Align(
          alignment: Alignment.centerLeft,
          child: Text(
            '暂无裁切例图',
            style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant),
          ),
        ),
      );
    }
    return SizedBox(
      height: 112,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: examples.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (context, index) {
          final event = examples[index];
          final bytes = _exampleBytes[event.id]!;
          return ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: SizedBox(
              width: 112,
              height: 112,
              child: Image.memory(
                bytes,
                key: ValueKey('dinov3-registry-example-${event.id}'),
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => const Center(
                  child: Icon(Icons.broken_image_outlined),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

"""
if marker not in text:
    raise RuntimeError("dialog insertion marker missing")
text = text.replace(marker, insert + marker, 1)

text = replace_once(
    text,
    "                const SizedBox(height: 14),\n                Text('注册条件', style: Theme.of(context).textTheme.titleSmall),",
    "                const SizedBox(height: 14),\n                Text('裁切例图', style: Theme.of(context).textTheme.titleSmall),\n                const SizedBox(height: 8),\n                _buildExampleGallery(),\n                const SizedBox(height: 14),\n                Text('注册条件', style: Theme.of(context).textTheme.titleSmall),",
    label="dialog gallery UI",
)
text = replace_once(
    text,
    "          children: [\n            OutlinedButton.icon(\n              onPressed: _saving ? null : _continueValidation,",
    "          children: [\n            if (entry.canDelete) ...[\n              OutlinedButton.icon(\n                onPressed: _saving ? null : _deleteSelected,\n                icon: const Icon(Icons.delete_outline_rounded),\n                label: const Text('删除物种'),\n              ),\n              const SizedBox(width: 8),\n            ],\n            OutlinedButton.icon(\n              onPressed: _saving ? null : _continueValidation,",
    label="dialog delete button",
)
write(path, text)


# Pure title helper keeps the displayed number derived from the current media's
# visible box order and does not persist a second identifier in observation data.
path = "frontend/lib/src/screens/species_validation_screen.dart"
text = read(path)
marker = "const double _validationButtonHeight = 40;"
helper = """String dinoV3FeedbackPanelTitle(
  DetectionBox box,
  List<DetectionBox> visibleBoxes,
) {
  var numberedBoxes = visibleBoxes;
  if (box.frameIndex != null) {
    final sameFrame = visibleBoxes
        .where((candidate) => candidate.frameIndex == box.frameIndex)
        .toList(growable: false);
    if (sameFrame.isNotEmpty) numberedBoxes = sameFrame;
  } else if (box.timestamp != null) {
    final sameTimestamp = visibleBoxes
        .where(
          (candidate) =>
              candidate.timestamp != null &&
              (candidate.timestamp! - box.timestamp!).abs() < 0.000001,
        )
        .toList(growable: false);
    if (sameTimestamp.isNotEmpty) numberedBoxes = sameTimestamp;
  }

  final observationId = box.observationId?.trim() ?? '';
  final boxIndex = numberedBoxes.indexWhere((candidate) {
    final candidateObservationId = candidate.observationId?.trim() ?? '';
    if (observationId.isNotEmpty && candidateObservationId == observationId) {
      return true;
    }
    return identical(candidate, box);
  });
  final boxNumber = boxIndex >= 0 ? boxIndex + 1 : 1;
  final predictedSpecies = box.predictedSpecies?.trim() ?? '';
  final detectedSpecies = box.species.trim();
  final species = predictedSpecies.isNotEmpty
      ? predictedSpecies
      : detectedSpecies.isNotEmpty
      ? detectedSpecies
      : 'Unknown';
  return '#$boxNumber $species · 检测框校验';
}

"""
if marker not in text:
    raise RuntimeError("feedback helper marker missing")
text = text.replace(marker, helper + marker, 1)
text = replace_count(
    text,
    "_buildDinoFeedbackPanel(selectedDinoBox),",
    "_buildDinoFeedbackPanel(selectedDinoBox, visibleBoxes),",
    2,
    label="feedback panel callsites",
)
text = replace_once(
    text,
    "  Widget _buildDinoFeedbackPanel(DetectionBox box) {\n    const title = '检测框校验';",
    "  Widget _buildDinoFeedbackPanel(\n    DetectionBox box,\n    List<DetectionBox> visibleBoxes,\n  ) {\n    final title = dinoV3FeedbackPanelTitle(box, visibleBoxes);",
    label="feedback panel signature",
)
write(path, text)


# Add a lightweight wiring assertion without weakening the behavior tests.
path = "frontend/test/dinov3_startup_registry_features_test.dart"
text = read(path)
text = replace_once(
    text,
    "import 'dart:convert';",
    "import 'dart:convert';\nimport 'dart:io';",
    label="frontend test io import",
)
text = replace_once(
    text,
    "void main() {\n  test('startup check runs once per backend generation', () async {",
    "void main() {\n  test('NeriApp wires the persistent DINOv3 startup checker', () {\n    final source = File('lib/main.dart').readAsStringSync();\n    expect(source, contains('DinoV3StartupCheck()'));\n    expect(source, contains('dinoV3StartupCheck: _dinoV3StartupCheck'));\n  });\n\n  test('startup check runs once per backend generation', () async {",
    label="frontend wiring test",
)
write(path, text)

print("DINOv3 startup/registry UI implementation staged")
