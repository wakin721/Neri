import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart';

import 'api_client.dart';
import 'models/job.dart';
import 'models/settings.dart';
import 'models/theme_settings.dart';
import 'screens/preview_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/species_validation_screen.dart';
import 'screens/start_screen.dart';

const _lastInputPathKey = 'last_input_path';

enum _CloseDialogPhase { confirming, closing, restoring }

class MainWindow extends StatefulWidget {
  const MainWindow({
    required this.apiClient,
    required this.themeNotifier,
    super.key,
  });

  final NeriApiClient apiClient;
  final ValueNotifier<ThemeSettings> themeNotifier;

  @override
  State<MainWindow> createState() => _MainWindowState();
}

class _MainWindowState extends State<MainWindow> with WindowListener {
  static const _pageTitles = <String>['开始界面', '图像预览', '物种校验', '设置'];
  static const _navigationItems = <_NavigationRailEntry>[
    _NavigationRailEntry(
      label: '开始',
      icon: Icons.home_outlined,
      selectedIcon: Icons.home_rounded,
    ),
    _NavigationRailEntry(
      label: '预览',
      icon: Icons.photo_library_outlined,
      selectedIcon: Icons.photo_library_rounded,
    ),
    _NavigationRailEntry(
      label: '校验',
      icon: Icons.fact_check_outlined,
      selectedIcon: Icons.fact_check_rounded,
    ),
    _NavigationRailEntry(
      label: '设置',
      icon: Icons.settings_outlined,
      selectedIcon: Icons.settings_rounded,
    ),
  ];

  static final _versionNumberPattern = RegExp(r'\d+(?:\.\d+)+');

  final _inputController = TextEditingController();
  final Map<String, DetectionItem> _previewMetadataCache =
      <String, DetectionItem>{};
  final Set<String> _previewMetadataLoading = <String>{};

  NeriSettings? _settings;
  List<ProcessingJob> _jobs = const <ProcessingJob>[];
  List<DetectionItem> _previewItems = const <DetectionItem>[];
  Timer? _timer;
  Timer? _previewRefreshTimer;
  Process? _backendProcess;
  Future<void>? _backendShutdownTask;
  Future<void>? _closeBackendShutdownTask;
  int _previewRefreshRequestId = 0;
  int _closeFlowId = 0;
  bool _loading = true;
  bool _backendStarting = false;
  bool _backendReady = false;
  bool _submitting = false;
  bool _previewLoading = false;
  bool _validationBusy = false;
  int _validationBusyRequests = 0;
  final Map<String, int> _pendingStartJobBaselines = <String, int>{};
  final Set<String> _pendingStopJobIds = <String>{};
  bool _useFp16 = false;
  bool _previewShowDetections = true;
  bool _previewDetecting = false;
  bool _isMaximized = false;
  bool _closing = false;
  bool _showClosingOverlay = false;
  bool _closeBackendStopped = false;
  _CloseDialogPhase _closeDialogPhase = _CloseDialogPhase.confirming;
  double _confidence = 0.25;
  double _iou = 0.45;
  double _previewConfidenceThreshold = 0.25;
  String _previewSpeciesFilter = previewAllSpeciesLabel;
  String? _selectedModelPath;
  String? _selectedClassificationModelPath;
  String _videoMode = 'all';
  int _vidStride = 1;
  String? _previewLoadedPath;
  int _selectedIndex = 0;
  int _selectedPreviewIndex = 0;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    unawaited(windowManager.setPreventClose(true));
    _initWindowState();

    try {
      MediaKit.ensureInitialized();
    } catch (_) {}

    _inputController.addListener(_schedulePreviewRefresh);
    _loadLastInputPath();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_startBackendAndInitialRefresh());
    });
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    _timer?.cancel();
    _previewRefreshTimer?.cancel();
    if (_closing) {
      widget.apiClient.close();
    } else {
      unawaited(
        _shutdownBackend()
            .catchError((_) {})
            .whenComplete(widget.apiClient.close),
      );
    }
    _inputController.dispose();
    super.dispose();
  }

  @override
  void onWindowClose() {
    _requestCloseWindow();
  }

  bool get _closeFlowBlocksBackendStartup =>
      _closing && _closeDialogPhase != _CloseDialogPhase.restoring;

  void _requestCloseWindow() {
    if (_closing) return;
    final flowId = ++_closeFlowId;
    _closing = true;
    _timer?.cancel();
    _timer = null;
    _previewRefreshTimer?.cancel();
    _previewRefreshTimer = null;
    _previewRefreshRequestId++;
    if (mounted) {
      setState(() {
        _closeBackendStopped = false;
        _closeDialogPhase = _CloseDialogPhase.confirming;
        _showClosingOverlay = true;
      });
    }
    final shutdownTask = _shutdownBackend();
    _closeBackendShutdownTask = shutdownTask;
    unawaited(
      shutdownTask.catchError((_) {}).whenComplete(() {
        if (!mounted || !_showClosingOverlay || flowId != _closeFlowId) return;
        if (_closeDialogPhase == _CloseDialogPhase.confirming) {
          setState(() => _closeBackendStopped = true);
        }
      }),
    );
  }

  void _confirmCloseWindow() {
    if (!_closing || _closeDialogPhase == _CloseDialogPhase.closing) return;
    final flowId = _closeFlowId;
    if (mounted) {
      setState(() => _closeDialogPhase = _CloseDialogPhase.closing);
    }
    unawaited(_finishCloseWindow(flowId));
  }

  void _cancelCloseWindow() {
    if (!_closing || _closeDialogPhase != _CloseDialogPhase.confirming) return;
    final shutdownTask = _closeBackendShutdownTask;
    final flowId = ++_closeFlowId;
    if (mounted) {
      setState(() => _closeDialogPhase = _CloseDialogPhase.restoring);
    }
    unawaited(_restartBackendAfterCloseCancelled(shutdownTask, flowId));
  }

  Future<void> _finishCloseWindow(int flowId) async {
    if (!mounted || flowId != _closeFlowId) return;
    try {
      await windowManager
          .setPreventClose(false)
          .timeout(const Duration(milliseconds: 200));
    } catch (_) {}
    if (!mounted || flowId != _closeFlowId) return;

    try {
      await windowManager.destroy().timeout(const Duration(milliseconds: 300));
    } catch (_) {}
    if (_closing && flowId == _closeFlowId) {
      exit(0);
    }
  }

  Future<void> _restartBackendAfterCloseCancelled(
    Future<void>? shutdownTask,
    int flowId,
  ) async {
    try {
      if (shutdownTask != null) {
        await shutdownTask.catchError((_) {});
      }
      while (_backendStarting) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
      if (!mounted || flowId != _closeFlowId) return;
      await _startBackendAndInitialRefresh();
    } catch (error) {
      if (mounted && flowId == _closeFlowId) {
        _showSnackBar('重启 Python 后端失败：$error');
      }
    } finally {
      if (mounted && flowId == _closeFlowId) {
        setState(() {
          _closing = false;
          _showClosingOverlay = false;
          _closeBackendStopped = false;
          _closeBackendShutdownTask = null;
          _closeDialogPhase = _CloseDialogPhase.confirming;
        });
      }
    }
  }

  @override
  void onWindowMaximize() {
    if (mounted) setState(() => _isMaximized = true);
  }

  @override
  void onWindowUnmaximize() {
    if (mounted) setState(() => _isMaximized = false);
  }

  Future<void> _initWindowState() async {
    try {
      final isMaximized = await windowManager.isMaximized();
      if (mounted) setState(() => _isMaximized = isMaximized);
    } catch (_) {}
  }

  Future<void> _loadLastInputPath() async {
    final preferences = await SharedPreferences.getInstance();
    final lastPath = preferences.getString(_lastInputPathKey);
    if (!mounted || lastPath == null || lastPath.isEmpty) return;
    setState(() => _inputController.text = lastPath);
    _schedulePreviewRefresh();
  }

  Future<void> _startBackendAndInitialRefresh() async {
    if (_closeFlowBlocksBackendStartup) return;
    if (_backendStarting) return;
    _backendStarting = true;
    if (mounted) setState(() => _loading = true);
    try {
      await _ensureBackendStarted();
      if (_closeFlowBlocksBackendStartup) {
        if (mounted) setState(() => _loading = false);
        return;
      }
      final ready = await _waitForBackendReady();
      if (!mounted) return;
      if (_closeFlowBlocksBackendStartup) {
        setState(() => _loading = false);
        return;
      }
      if (!ready) {
        setState(() => _loading = false);
        _showSnackBar('Python 后端启动超时，请检查 toolkit\\python.exe 和端口 721。');
        return;
      }
      _backendReady = true;
      await _refreshInitialPageData();
      _timer ??= Timer.periodic(
        const Duration(seconds: 2),
        (_) => _refresh(silent: true),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() => _loading = false);
      _showSnackBar('启动 Python 后端失败：$error');
    } finally {
      _backendStarting = false;
    }
  }

  Future<void> _refreshInitialPageData() async {
    await _refresh(includeJobResults: true, finishLoading: false);
    if (!mounted || _closeFlowBlocksBackendStartup) return;
    if (_inputController.text.trim().isEmpty) {
      _stopGlobalLoading();
      return;
    }
    _previewRefreshTimer?.cancel();
    _previewRefreshTimer = null;
    await _refreshPreviewItems(force: true, finishGlobalLoading: true);
  }

  void _stopGlobalLoading() {
    if (!mounted || !_loading) return;
    setState(() => _loading = false);
  }

  Future<void> _ensureBackendStarted() async {
    if (await _backendResponds()) return;
    final projectRoot = _resolveProjectRoot();
    if (projectRoot == null) {
      throw StateError('未找到项目根目录，无法定位 toolkit\\python.exe。');
    }
    final pythonExe = _fileUnder(projectRoot, [
      'toolkit',
      Platform.isWindows ? 'python.exe' : 'python',
    ]);
    if (!pythonExe.existsSync()) {
      throw StateError('未找到 ${pythonExe.path}。');
    }

    final process = await Process.start(
      pythonExe.path,
      const <String>[
        '-m',
        'uvicorn',
        'system.backend.main:app',
        '--app-dir',
        '.',
        '--host',
        '127.0.0.1',
        '--port',
        '721',
      ],
      workingDirectory: projectRoot.path,
      environment: <String, String>{'NERI_PARENT_PID': pid.toString()},
      includeParentEnvironment: true,
      runInShell: false,
    );
    _backendProcess = process;
    process.stdout.transform(utf8.decoder).listen((_) {});
    process.stderr.transform(utf8.decoder).listen((_) {});
    unawaited(
      process.exitCode.then((_) {
        if (identical(_backendProcess, process)) {
          _backendProcess = null;
          _backendReady = false;
        }
      }),
    );
  }

  Future<bool> _waitForBackendReady() async {
    final deadline = DateTime.now().add(const Duration(seconds: 30));
    while (DateTime.now().isBefore(deadline)) {
      if (await _backendResponds()) return true;
      await Future<void>.delayed(const Duration(milliseconds: 500));
    }
    return false;
  }

  Future<bool> _backendResponds() async {
    try {
      return await widget.apiClient.health().timeout(
        const Duration(milliseconds: 800),
      );
    } catch (_) {
      return false;
    }
  }

  Future<void> _shutdownBackend() {
    final currentTask = _backendShutdownTask;
    if (currentTask != null) return currentTask;
    final task = _shutdownBackendNow();
    _backendShutdownTask = task;
    unawaited(
      task
          .whenComplete(() {
            if (identical(_backendShutdownTask, task)) {
              _backendShutdownTask = null;
            }
          })
          .catchError((_) {}),
    );
    return task;
  }

  Future<void> _shutdownBackendNow() async {
    final process = _backendProcess;
    _backendReady = false;
    var shutdownRequested = false;

    if (process == null &&
        !await _backendPortOpen(const Duration(milliseconds: 250))) {
      return;
    }

    try {
      await widget.apiClient.shutdownBackend().timeout(
        const Duration(milliseconds: 800),
      );
      shutdownRequested = true;
    } catch (_) {}

    if (process != null) {
      if (shutdownRequested &&
          await _waitForProcessExit(
            process,
            const Duration(milliseconds: 1500),
          )) {
        if (identical(_backendProcess, process)) {
          _backendProcess = null;
        }
        return;
      }
      await _terminateBackendProcess(process);
      if (identical(_backendProcess, process)) {
        _backendProcess = null;
      }
    } else {
      if (shutdownRequested &&
          await _waitForBackendStopped(const Duration(milliseconds: 1500))) {
        return;
      }
      await _terminateBackendPortOwner();
    }
  }

  Future<bool> _waitForBackendStopped(Duration timeout) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      if (!await _backendResponds()) return true;
      await Future<void>.delayed(const Duration(milliseconds: 250));
    }
    return !await _backendResponds();
  }

  Future<bool> _waitForProcessExit(Process process, Duration timeout) async {
    try {
      await process.exitCode.timeout(timeout);
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> _backendPortOpen(Duration timeout) async {
    Socket? socket;
    try {
      socket = await Socket.connect('127.0.0.1', 721, timeout: timeout);
      return true;
    } catch (_) {
      return false;
    } finally {
      socket?.destroy();
    }
  }

  Future<void> _terminateBackendProcess(Process process) async {
    if (Platform.isWindows) {
      try {
        await _taskkillProcessTree(process.pid);
        return;
      } catch (_) {
        process.kill();
      }
    } else {
      process.kill(ProcessSignal.sigterm);
      try {
        await process.exitCode.timeout(const Duration(seconds: 2));
        return;
      } catch (_) {
        process.kill(ProcessSignal.sigkill);
      }
    }
  }

  Future<void> _terminateBackendPortOwner() async {
    if (!Platform.isWindows) return;
    final ownerPid = await _windowsListeningPortOwner(721);
    if (ownerPid == null || ownerPid == pid) return;
    try {
      await _taskkillProcessTree(ownerPid);
    } catch (_) {}
  }

  Future<void> _taskkillProcessTree(int targetPid) async {
    await Process.run('taskkill', [
      '/PID',
      targetPid.toString(),
      '/T',
      '/F',
    ]).timeout(const Duration(seconds: 3));
  }

  Future<int?> _windowsListeningPortOwner(int port) async {
    try {
      final result = await Process.run('netstat', [
        '-ano',
        '-p',
        'tcp',
      ]).timeout(const Duration(seconds: 2));
      final output = '${result.stdout}\n${result.stderr}';
      final portSuffix = ':$port';
      for (final line in output.split('\n')) {
        final trimmed = line.trim();
        if (!trimmed.contains('LISTENING')) continue;
        final columns = trimmed.split(RegExp(r'\s+'));
        if (columns.length < 5) continue;
        final localAddress = columns[1];
        if (!localAddress.endsWith(portSuffix)) continue;
        return int.tryParse(columns.last);
      }
    } catch (_) {}
    return null;
  }

  Directory? _resolveProjectRoot() {
    final bases = <Directory>[
      Directory.current,
      File(Platform.resolvedExecutable).parent,
    ];
    final visited = <String>{};
    for (final base in bases) {
      var current = base.absolute;
      for (var depth = 0; depth < 8; depth++) {
        final path = current.path;
        if (visited.add(path) &&
            _fileUnder(current, [
              'toolkit',
              Platform.isWindows ? 'python.exe' : 'python',
            ]).existsSync() &&
            _fileUnder(current, [
              'system',
              'backend',
              'main.py',
            ]).existsSync()) {
          return current;
        }
        final parent = current.parent;
        if (parent.path == current.path) break;
        current = parent;
      }
    }
    return null;
  }

  File _fileUnder(Directory root, List<String> parts) {
    return File([root.path, ...parts].join(Platform.pathSeparator));
  }

  Future<void> _saveLastInputPath(String path) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(_lastInputPathKey, path);
  }

  Future<void> _saveThemeSettings(ThemeSettings settings) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setInt('theme_mode', settings.themeMode.index);
    await preferences.setBool('use_dynamic_color', settings.useDynamicColor);
    await preferences.setInt('seed_color', settings.seedColor.toARGB32());
  }

  void _updateTheme(ThemeSettings updated) {
    widget.themeNotifier.value = updated;
    unawaited(_saveThemeSettings(updated));
  }

  void _setValidationBusy(bool value) {
    if (!mounted) return;
    final nextCount = value
        ? _validationBusyRequests + 1
        : (_validationBusyRequests - 1).clamp(0, 1 << 20).toInt();
    final nextBusy = nextCount > 0;
    if (_validationBusyRequests == nextCount && _validationBusy == nextBusy) {
      return;
    }
    setState(() {
      _validationBusyRequests = nextCount;
      _validationBusy = nextBusy;
    });
  }

  Future<T> _runValidationBusy<T>(Future<T> Function() action) async {
    _setValidationBusy(true);
    try {
      return await action();
    } finally {
      _setValidationBusy(false);
    }
  }

  bool get _jobProcessingBusy => _jobs.any((job) => job.isWorkerActive);

  void _syncJobTransitionState(List<ProcessingJob> jobs) {
    if (_pendingStartJobBaselines.isEmpty && _pendingStopJobIds.isEmpty) {
      return;
    }
    final jobsById = {for (final job in jobs) job.id: job};
    var changed = false;

    final beforeStartCount = _pendingStartJobBaselines.length;
    _pendingStartJobBaselines.removeWhere((id, baselineProcessed) {
      final job = jobsById[id];
      return job == null ||
          !job.isWorkerActive ||
          job.processed > baselineProcessed;
    });
    changed = changed || _pendingStartJobBaselines.length != beforeStartCount;

    final beforeStopCount = _pendingStopJobIds.length;
    _pendingStopJobIds.removeWhere((id) {
      final job = jobsById[id];
      return job == null || !job.isWorkerActive;
    });
    changed = changed || _pendingStopJobIds.length != beforeStopCount;

    if (changed && mounted) setState(() {});
  }

  void _showSnackBar(String message) {
    if (!mounted || message.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final messenger = ScaffoldMessenger.maybeOf(context);
      if (messenger == null) return;
      messenger.clearSnackBars();

      final controller = messenger.showSnackBar(
        SnackBar(
          content: Text(message),
          duration: const Duration(seconds: 4),
          action: SnackBarAction(
            label: '关闭',
            onPressed: messenger.hideCurrentSnackBar,
          ),
        ),
      );
      // 强制绕过无障碍模式驻留时间限制
      Future.delayed(const Duration(seconds: 4), controller.close);
    });
  }

  void _schedulePreviewRefresh() {
    _previewRefreshTimer?.cancel();
    if (_closeFlowBlocksBackendStartup) return;
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      _previewRefreshRequestId++;
      if (!mounted) return;
      setState(() {
        _previewItems = const <DetectionItem>[];
        _previewMetadataCache.clear();
        _previewMetadataLoading.clear();
        _previewLoadedPath = null;
        _previewLoading = false;
      });
      return;
    }

    unawaited(_saveLastInputPath(inputPath));
    _previewRefreshTimer = Timer(
      const Duration(milliseconds: 450),
      () => _refreshPreviewItems(),
    );
  }

  Future<void> _refreshPreviewItems({
    bool force = false,
    bool finishGlobalLoading = false,
  }) async {
    if (_closeFlowBlocksBackendStartup) {
      if (finishGlobalLoading) _stopGlobalLoading();
      return;
    }
    if (!_backendReady) {
      if (finishGlobalLoading) _stopGlobalLoading();
      return;
    }
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      if (finishGlobalLoading) _stopGlobalLoading();
      return;
    }
    if (!force && _previewLoadedPath == inputPath && _previewItems.isNotEmpty) {
      if (finishGlobalLoading) _stopGlobalLoading();
      return;
    }

    final requestId = ++_previewRefreshRequestId;
    if (!mounted) return;
    setState(() {
      _previewLoading = true;
      if (_previewLoadedPath != inputPath || force) {
        _previewItems = const <DetectionItem>[];
        _previewMetadataCache.clear();
        _previewMetadataLoading.clear();
        _previewLoadedPath = inputPath;
        _selectedPreviewIndex = 0;
      }
    });

    try {
      final items = await widget.apiClient.fetchPreviewItems(
        inputPath: inputPath,
        includeCached: true,
      );
      if (!mounted ||
          _inputController.text.trim() != inputPath ||
          requestId != _previewRefreshRequestId) {
        if (finishGlobalLoading) _stopGlobalLoading();
        return;
      }
      final sortedItems = _sortMediaItemsForDisplay(items);
      setState(() {
        _previewItems = sortedItems;
        _previewLoadedPath = inputPath;
        _previewLoading = false;
        if (finishGlobalLoading) {
          _loading = false;
        }
        _selectedPreviewIndex = _safePreviewIndex(sortedItems);
      });
      if (sortedItems.isNotEmpty) {
        unawaited(_loadPreviewMetadata(sortedItems[_selectedPreviewIndex]));
      }
    } catch (error) {
      if (!mounted ||
          _inputController.text.trim() != inputPath ||
          requestId != _previewRefreshRequestId) {
        if (finishGlobalLoading) _stopGlobalLoading();
        return;
      }
      _showSnackBar('无法读取输入文件夹预览：$error');
      setState(() {
        _previewLoading = false;
        if (finishGlobalLoading) {
          _loading = false;
        }
      });
    }
  }

  Future<void> _loadPreviewMetadata(DetectionItem item) async {
    if (_closeFlowBlocksBackendStartup) return;
    if (item.path.isEmpty ||
        _previewMetadataCache.containsKey(item.path) ||
        _previewMetadataLoading.contains(item.path)) {
      return;
    }

    _previewMetadataLoading.add(item.path);
    try {
      final fullItem = await widget.apiClient.fetchPreviewItem(
        filePath: item.path,
        inputPath: _inputController.text.trim(),
      );
      if (!mounted || _closeFlowBlocksBackendStartup) return;
      setState(() {
        _previewMetadataCache[item.path] = fullItem;
        _previewItems = _sortMediaItemsForDisplay(
          _previewItems
              .map(
                (previewItem) =>
                    previewItem.path == fullItem.path ? fullItem : previewItem,
              )
              .toList(),
        );
      });
    } catch (_) {
      // Fast preview data is still usable if per-file metadata is unavailable.
    } finally {
      _previewMetadataLoading.remove(item.path);
    }
  }

  List<DetectionItem> _mergePreviewItems(
    List<DetectionItem> currentItems,
    List<DetectionItem> updates,
  ) {
    if (currentItems.isEmpty) return _sortMediaItemsForDisplay(updates);

    final updateByPath = <String, DetectionItem>{
      for (final item in updates)
        if (item.path.isNotEmpty) item.path: item,
    };
    final seen = currentItems.map((item) => item.path).toSet();
    return _sortMediaItemsForDisplay(<DetectionItem>[
      for (final item in currentItems) updateByPath[item.path] ?? item,
      for (final item in updates)
        if (!seen.contains(item.path)) item,
    ]);
  }

  List<DetectionItem> _replacePreviewItems(
    List<DetectionItem> currentItems,
    Iterable<DetectionItem> updates,
  ) {
    final updateByPath = <String, DetectionItem>{
      for (final item in updates)
        if (item.path.isNotEmpty) item.path: item,
    };
    if (updateByPath.isEmpty) return currentItems;

    var appended = false;
    final seen = currentItems.map((item) => item.path).toSet();
    final nextItems = <DetectionItem>[
      for (final item in currentItems) updateByPath[item.path] ?? item,
    ];
    for (final item in updateByPath.values) {
      if (seen.contains(item.path)) continue;
      nextItems.add(item);
      appended = true;
    }
    return appended ? _sortMediaItemsForDisplay(nextItems) : nextItems;
  }

  DetectionItem _mergeValidationUpdate(
    DetectionItem fallback,
    DetectionItem update,
  ) {
    final current = _previewItems.firstWhere(
      (item) => item.path == update.path,
      orElse: () => fallback,
    );
    return current.mergeValidationUpdate(update);
  }

  List<DetectionItem> _sortMediaItemsForDisplay(List<DetectionItem> items) {
    return List<DetectionItem>.from(items)..sort(_compareMediaItemsForDisplay);
  }

  int _compareMediaItemsForDisplay(DetectionItem a, DetectionItem b) {
    final timeA = _mediaSortTimestamp(a);
    final timeB = _mediaSortTimestamp(b);
    if (timeA != null && timeB != null) {
      final timeCompare = timeA.compareTo(timeB);
      if (timeCompare != 0) return timeCompare;
    } else if (timeA != null) {
      return -1;
    } else if (timeB != null) {
      return 1;
    }

    final nameCompare = _naturalCompare(a.filename, b.filename);
    if (nameCompare != 0) return nameCompare;
    return _naturalCompare(a.path, b.path);
  }

  DateTime? _mediaSortTimestamp(DetectionItem item) {
    final dateText = item.dateTaken?.trim();
    if (dateText != null && dateText.isNotEmpty) {
      final timeText = item.detectionData['拍摄时间']?.toString().trim();
      if (dateText.contains(':') || dateText.contains('T')) {
        final dateTime = DateTime.tryParse(dateText);
        if (dateTime != null) return dateTime;
      }
      if (timeText != null && timeText.isNotEmpty) {
        final dateTime = DateTime.tryParse('$dateText $timeText');
        if (dateTime != null) return dateTime;
      }
    }
    final modifiedAt = item.modifiedAt?.trim();
    if (modifiedAt == null || modifiedAt.isEmpty) return null;
    return DateTime.tryParse(modifiedAt);
  }

  int _naturalCompare(String a, String b) {
    final segmentsA = _naturalSegments(a);
    final segmentsB = _naturalSegments(b);
    final length = segmentsA.length < segmentsB.length
        ? segmentsA.length
        : segmentsB.length;
    for (var index = 0; index < length; index++) {
      final partA = segmentsA[index];
      final partB = segmentsB[index];
      final numberA = int.tryParse(partA);
      final numberB = int.tryParse(partB);
      final compare = numberA != null && numberB != null
          ? numberA.compareTo(numberB)
          : partA.toLowerCase().compareTo(partB.toLowerCase());
      if (compare != 0) return compare;
    }
    return segmentsA.length.compareTo(segmentsB.length);
  }

  List<String> _naturalSegments(String value) {
    return RegExp(r'\d+|\D+')
        .allMatches(value)
        .map((match) => match.group(0) ?? '')
        .where((part) => part.isNotEmpty)
        .toList();
  }

  List<DetectionItem> _jobResultsForInputPath(
    List<ProcessingJob> jobs,
    String inputPath,
  ) {
    if (inputPath.isEmpty) return const <DetectionItem>[];

    final normalizedInput = _normalizedLocalPath(inputPath);
    final inputPrefix = normalizedInput.endsWith('/')
        ? normalizedInput
        : '$normalizedInput/';
    final results = <DetectionItem>[];
    final seen = <String>{};
    for (final job in jobs) {
      final normalizedJobInput = _normalizedLocalPath(job.inputDir);
      final jobInputMatches =
          normalizedJobInput == normalizedInput ||
          normalizedJobInput.startsWith(inputPrefix);
      for (final item in job.results) {
        if (!_hasPreviewResultData(item)) continue;
        final normalizedItemPath = _normalizedLocalPath(item.path);
        final itemMatches =
            normalizedItemPath == normalizedInput ||
            normalizedItemPath.startsWith(inputPrefix);
        if ((jobInputMatches || itemMatches) && seen.add(item.path)) {
          results.add(item);
        }
      }
    }
    return results;
  }

  bool _hasPreviewResultData(DetectionItem item) {
    return item.species.isNotEmpty ||
        item.detectionBoxes.isNotEmpty ||
        item.detectionData.isNotEmpty ||
        item.confidence != null ||
        item.validated != null ||
        item.error != null;
  }

  String _normalizedLocalPath(String path) {
    return path.replaceAll('\\', '/').toLowerCase();
  }

  Future<void> _refresh({
    bool silent = false,
    bool reloadSettings = false,
    bool includeJobResults = true,
    bool finishLoading = true,
  }) async {
    if (_closeFlowBlocksBackendStartup) return;
    if (!silent) {
      setState(() => _loading = true);
    }

    try {
      final shouldFetchSettings =
          reloadSettings || !silent || _settings == null;
      final settings = shouldFetchSettings
          ? await widget.apiClient.fetchSettings()
          : _settings;
      final summariesOnly =
          !includeJobResults || (silent && _jobProcessingBusy);
      var jobs = await widget.apiClient.listJobs(
        includeResults: !summariesOnly,
      );
      if (summariesOnly &&
          includeJobResults &&
          jobs.every((job) => !job.isWorkerActive)) {
        jobs = await widget.apiClient.listJobs();
      } else if (summariesOnly) {
        jobs = _mergeJobSummariesWithCachedResults(jobs);
      }
      if (!mounted || _closeFlowBlocksBackendStartup || settings == null) {
        return;
      }
      _syncJobTransitionState(jobs);

      final firstLoad = _settings == null && shouldFetchSettings;

      final settingsChanged = shouldFetchSettings;

      bool jobsChanged = _jobs.length != jobs.length;
      if (!jobsChanged) {
        for (int i = 0; i < jobs.length; i++) {
          if (_jobs[i].id != jobs[i].id ||
              _jobs[i].state != jobs[i].state ||
              _jobs[i].processed != jobs[i].processed ||
              _jobs[i].active != jobs[i].active ||
              _jobs[i].results.length != jobs[i].results.length ||
              _jobs[i].message != jobs[i].message ||
              _jobs[i].error != jobs[i].error ||
              _jobs[i].updatedAt != jobs[i].updatedAt) {
            jobsChanged = true;
            break;
          }
        }
      }

      final previewJobUpdates = jobsChanged && !summariesOnly
          ? _jobResultsForInputPath(jobs, _inputController.text.trim())
          : const <DetectionItem>[];

      if (settingsChanged || jobsChanged) {
        setState(() {
          if (settingsChanged) {
            _settings = settings;
          }
          final modelPaths = settings.availableModels
              .map((model) => model.path)
              .toSet();
          if (_selectedModelPath == null ||
              !modelPaths.contains(_selectedModelPath)) {
            _selectedModelPath =
                settings.selectedModel ??
                (settings.availableModels.isEmpty
                    ? null
                    : settings.availableModels.first.path);
          }
          final classificationModelPaths = settings
              .availableClassificationModels
              .map((model) => model.path)
              .toSet();
          if (_selectedClassificationModelPath == null ||
              (_selectedClassificationModelPath!.isNotEmpty &&
                  !classificationModelPaths.contains(
                    _selectedClassificationModelPath,
                  ))) {
            _selectedClassificationModelPath =
                settings.selectedClassificationModel ?? '';
          }
          if (firstLoad) {
            _confidence = _doubleSetting(settings, 'confidence', _confidence);
            _iou = _doubleSetting(settings, 'iou', _iou);
            _useFp16 =
                settings.gpuAvailable &&
                _boolSetting(settings, 'use_fp16', _useFp16);
            _videoMode =
                _stringSetting(settings, 'video_mode', _videoMode) == 'fast'
                ? 'fast'
                : 'all';
            _vidStride = _intSetting(
              settings,
              'vid_stride',
              _vidStride,
            ).clamp(1, 120).toInt();
          }
          _jobs = jobs;
          if (previewJobUpdates.isNotEmpty) {
            _previewItems = _mergePreviewItems(
              _previewItems,
              previewJobUpdates,
            );
            for (final item in previewJobUpdates) {
              _previewMetadataCache[item.path] = item;
            }
            _selectedPreviewIndex = _safePreviewIndex(_previewItems);
          }
        });
      }

      // 清理加载状态
      if (!silent && finishLoading && _loading) {
        setState(() => _loading = false);
      }
    } catch (error) {
      if (!mounted) return;
      if (_closeFlowBlocksBackendStartup) return;
      _backendReady = false;
      setState(() => _loading = false);
      if (!silent) _showSnackBar('无法连接 Python 后端：$error');
    }
  }

  List<ProcessingJob> _mergeJobSummariesWithCachedResults(
    List<ProcessingJob> summaries,
  ) {
    final cachedById = {for (final job in _jobs) job.id: job};
    return [
      for (final summary in summaries)
        if (summary.results.isEmpty &&
            (cachedById[summary.id]?.results.isNotEmpty ?? false))
          ProcessingJob(
            id: summary.id,
            state: summary.state,
            inputDir: summary.inputDir,
            createdAt: summary.createdAt,
            updatedAt: summary.updatedAt,
            outputDir: summary.outputDir,
            total: summary.total,
            processed: summary.processed,
            message: summary.message,
            results: cachedById[summary.id]!.results,
            error: summary.error,
            active: summary.active,
          )
        else
          summary,
    ];
  }

  Future<void> _saveAdvancedSettings(Map<String, dynamic> settings) async {
    try {
      final saved = await widget.apiClient.saveSettings(settings);
      if (!mounted) return;
      setState(() {
        _settings = saved;
        _selectedModelPath = saved.selectedModel ?? _selectedModelPath;
        _selectedClassificationModelPath =
            saved.selectedClassificationModel ??
            _selectedClassificationModelPath;
        _confidence = _doubleSetting(saved, 'confidence', _confidence);
        _iou = _doubleSetting(saved, 'iou', _iou);
        _useFp16 =
            saved.gpuAvailable && _boolSetting(saved, 'use_fp16', _useFp16);
        _videoMode = _stringSetting(saved, 'video_mode', _videoMode) == 'fast'
            ? 'fast'
            : 'all';
        _vidStride = _intSetting(
          saved,
          'vid_stride',
          _vidStride,
        ).clamp(1, 120).toInt();
      });
      _showSnackBar('设置已保存');
    } catch (error) {
      _showSnackBar('保存设置失败：$error');
      rethrow;
    }
  }

  bool _boolSetting(NeriSettings settings, String key, bool fallback) {
    final value = settings.settings[key];
    if (value is bool) return value;
    if (value is String) return value.toLowerCase() == 'true';
    return fallback;
  }

  double _doubleSetting(NeriSettings settings, String key, double fallback) {
    final value = settings.settings[key];
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? fallback;
    return fallback;
  }

  int _intSetting(NeriSettings settings, String key, int fallback) {
    final value = settings.settings[key];
    if (value is num) return value.toInt();
    if (value is String) return int.tryParse(value) ?? fallback;
    return fallback;
  }

  String _stringSetting(NeriSettings settings, String key, String fallback) {
    final value = settings.settings[key];
    if (value is String && value.trim().isNotEmpty) return value;
    return fallback;
  }

  int _processingBatchSize({bool singleFile = false}) {
    if (singleFile) return 1;
    final settings = _settingsOrEmpty();
    final savedBatchSize = _intSetting(settings, 'batch_size', 16);
    final gpuAvailable = _settings?.gpuAvailable == true;
    return gpuAvailable ? savedBatchSize.clamp(1, 64).toInt() : 1;
  }

  bool _effectiveUseFp16() {
    return _settings?.gpuAvailable == true && _useFp16;
  }

  int _videoStride() {
    return _vidStride.clamp(1, 120).toInt();
  }

  String _effectiveVideoMode() {
    return _videoMode == 'fast' ? 'fast' : 'all';
  }

  bool _useAugment() {
    return _boolSetting(_settingsOrEmpty(), 'use_augment', true);
  }

  bool _useAgnosticNms() {
    return _boolSetting(_settingsOrEmpty(), 'use_agnostic_nms', true);
  }

  String _packageSource() {
    return _stringSetting(_settingsOrEmpty(), 'package_source', 'official');
  }

  String _pytorchVersion() {
    return _stringSetting(_settingsOrEmpty(), 'pytorch_version', '自动检测');
  }

  List<String> _stringListSetting(
    NeriSettings settings,
    String key,
    List<String> fallback,
  ) {
    final value = settings.settings[key];
    if (value is List) {
      return value.map((item) => item.toString()).toList();
    }
    return fallback;
  }

  Map<String, int> _intMapSetting(NeriSettings settings, String key) {
    final value = settings.settings[key];
    if (value is! Map) return const <String, int>{};
    return value.map((key, value) {
      final parsed = value is num
          ? value.toInt()
          : int.tryParse(value.toString()) ?? 0;
      return MapEntry(key.toString(), parsed);
    });
  }

  List<String> _splitSpeciesNames(String speciesName) {
    return speciesName
        .replaceAll('，', ',')
        .split(',')
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty && name != '空' && name != 'Unknown')
        .toList();
  }

  Future<void> _recordQuickMarkUsage(String speciesName) async {
    final current = _settings;
    if (current == null) return;
    final names = _splitSpeciesNames(speciesName);
    if (names.isEmpty) return;

    final nextSettings = Map<String, dynamic>.from(current.settings);
    final counts = Map<String, int>.from(
      _intMapSetting(current, 'quick_mark_usage_counts'),
    );
    final history = _stringListSetting(
      current,
      'quick_mark_recent_history',
      const <String>[],
    ).toList();

    for (final name in names) {
      counts[name] = (counts[name] ?? 0) + 1;
      history.add(name);
    }
    if (history.length > 200) {
      history.removeRange(0, history.length - 200);
    }

    nextSettings['quick_mark_usage_counts'] = counts;
    nextSettings['quick_mark_recent_history'] = history;

    try {
      final saved = await widget.apiClient.saveSettings(nextSettings);
      if (mounted) setState(() => _settings = saved);
    } catch (_) {
      // Usage history is only used for sorting quick-mark buttons; marking
      // itself should not fail if this lightweight preference save is blocked.
    }
  }

  Future<void> _createJob() async {
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      _showSnackBar('请输入红外相机媒体文件夹路径');
      return;
    }

    setState(() => _submitting = true);
    try {
      await _saveLastInputPath(inputPath);
      final createdJob = await widget.apiClient.createJob(
        inputDir: inputPath,
        modelPath: _selectedModelPath,
        classificationModelPath: _selectedClassificationModelPath,
        confidence: _confidence,
        iou: _iou,
        useFp16: _effectiveUseFp16(),
        useAugment: _useAugment(),
        useAgnosticNms: _useAgnosticNms(),
        batchSize: _processingBatchSize(),
        vidStride: _videoStride(),
        videoMode: _effectiveVideoMode(),
        enableDetection: true,
        selectedSpeciesNames: _selectedSpeciesNames(),
      );
      if (mounted && createdJob.id.isNotEmpty) {
        setState(() {
          _pendingStartJobBaselines[createdJob.id] = createdJob.processed;
        });
      }
      await _refresh(silent: true);
      await _refreshPreviewItems(force: true);
      _showSnackBar('任务已提交');
    } catch (error) {
      if (!mounted) return;
      if (await _handleYoloDependencyError(error)) return;
      _showSnackBar('创建任务失败：$error');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _cancelJob(ProcessingJob job) async {
    if (mounted && job.id.isNotEmpty) {
      setState(() => _pendingStopJobIds.add(job.id));
    }
    try {
      await widget.apiClient.cancelJob(job.id);
      await _refresh(silent: true);
      _showSnackBar('已请求停止任务');
    } catch (error) {
      if (!mounted) return;
      setState(() => _pendingStopJobIds.remove(job.id));
      _showSnackBar('停止任务失败：$error');
    }
  }

  Future<void> _resumeJob(ProcessingJob job) async {
    if (mounted && job.id.isNotEmpty) {
      setState(() {
        _pendingStartJobBaselines[job.id] = job.processed;
      });
    }
    try {
      await widget.apiClient.resumeJob(job.id);
      await _refresh(silent: true);
      _showSnackBar('已继续任务');
    } catch (error) {
      if (!mounted) return;
      setState(() => _pendingStartJobBaselines.remove(job.id));
      _showSnackBar('继续任务失败：$error');
    }
  }

  Future<void> _deleteJob(ProcessingJob job) async {
    try {
      await widget.apiClient.deleteJob(job.id);
      await _refresh(silent: true);
      _showSnackBar('任务已删除');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('删除任务失败：$error');
    }
  }

  Future<void> _clearJobs() async {
    try {
      await widget.apiClient.clearJobs();
      await _refresh(silent: true);
      _showSnackBar('已清除所有任务');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('清除任务失败：$error');
    }
  }

  Future<bool> _handleYoloDependencyError(Object error) async {
    if (error is! ApiException || !error.isMissingYoloDependencies) {
      return false;
    }
    final missingText = error.missingDependencies.isEmpty
        ? 'YOLO 处理依赖'
        : error.missingDependencies.join('、');
    final envChoice = _pytorchVersion();
    final packageSource = _packageSource();
    final sourceLabel = switch (packageSource) {
      'aliyun' => '阿里源',
      'tsinghua' => '清华源',
      _ => '官方源',
    };
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('需要安装 YOLO 依赖'),
        content: Text(
          '当前缺少：$missingText。\n\n'
          '安装时会先自动安装适用于“$envChoice”的 PyTorch，然后从$sourceLabel安装 ultralytics。'
          '安装完成后 Python 后端会自动重启。',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('安装依赖'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return true;

    try {
      final response = await widget.apiClient.installYoloDependencies(
        envChoice: envChoice,
        packageSource: packageSource,
      );
      if (!mounted) return true;
      _showSnackBar(response.message);
    } catch (installError) {
      if (!mounted) return true;
      _showSnackBar('启动 YOLO 依赖安装失败：$installError');
    }
    return true;
  }

  Future<void> _detectCurrentPreviewImage(DetectionItem item) async {
    setState(() => _previewDetecting = true);
    try {
      await widget.apiClient.createJob(
        inputDir: item.path,
        modelPath: _selectedModelPath,
        classificationModelPath: _selectedClassificationModelPath,
        confidence: _previewConfidenceThreshold,
        iou: _iou,
        useFp16: _effectiveUseFp16(),
        useAugment: _useAugment(),
        useAgnosticNms: _useAgnosticNms(),
        batchSize: _processingBatchSize(singleFile: true),
        vidStride: _videoStride(),
        videoMode: _effectiveVideoMode(),
        enableDetection: true,
        selectedSpeciesNames: _selectedSpeciesNames(),
      );
      await _refresh(silent: true);
      await _refreshPreviewItems(force: true);
      if (mounted) setState(() => _selectedIndex = 1);
      _showSnackBar('已提交当前图像检测');
    } catch (error) {
      if (!mounted) return;
      if (await _handleYoloDependencyError(error)) return;
      _showSnackBar('检测当前图像失败：$error');
    } finally {
      if (mounted) setState(() => _previewDetecting = false);
    }
  }

  Future<void> _redetectValidationItems(
    List<DetectionItem> items, {
    required double confidence,
  }) async {
    if (items.isEmpty) return;
    setState(() => _submitting = true);
    try {
      final inputPaths = items
          .map((item) => item.path.trim())
          .where((path) => path.isNotEmpty)
          .toSet()
          .toList();
      if (inputPaths.isEmpty) return;
      await widget.apiClient.createJob(
        inputPaths: inputPaths,
        modelPath: _selectedModelPath,
        classificationModelPath: _selectedClassificationModelPath,
        confidence: confidence,
        iou: _iou,
        useFp16: _effectiveUseFp16(),
        useAugment: _useAugment(),
        useAgnosticNms: _useAgnosticNms(),
        batchSize: _processingBatchSize(singleFile: true),
        vidStride: _videoStride(),
        videoMode: _effectiveVideoMode(),
        enableDetection: true,
        selectedSpeciesNames: _selectedSpeciesNames(),
      );
      await _refresh(silent: true);
      await _refreshPreviewItems(force: true);
    } catch (error) {
      if (!mounted) return;
      if (await _handleYoloDependencyError(error)) return;
      _showSnackBar('重新检测失败：$error');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<DetectionItem> _markValidationItem(
    DetectionItem item,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    return _runValidationBusy(() async {
      final inputPath = _inputController.text.trim();
      if (inputPath.isEmpty) {
        throw StateError('请先在开始界面设置输入文件夹。');
      }

      final updated = await widget.apiClient.markValidationItem(
        inputPath: inputPath,
        filePath: item.path,
        action: action,
        speciesName: speciesName,
        speciesCount: speciesCount,
        speciesType: speciesType,
        remark: remark,
      );
      final merged = _mergeValidationUpdate(item, updated);

      if (!mounted) return merged;
      setState(() {
        _cacheMarkedSpeciesTypes(speciesName, speciesType);
        _previewMetadataCache[merged.path] = merged;
        _previewItems = _replacePreviewItems(_previewItems, [merged]);
      });
      return merged;
    });
  }

  Future<List<DetectionItem>> _markValidationItems(
    List<DetectionItem> items,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    return _runValidationBusy(() async {
      final inputPath = _inputController.text.trim();
      if (inputPath.isEmpty) {
        throw StateError('请先在开始界面设置输入文件夹。');
      }

      final rawUpdatedItems = await widget.apiClient.markValidationItems(
        inputPath: inputPath,
        filePaths: items.map((item) => item.path).toList(),
        action: action,
        speciesName: speciesName,
        speciesCount: speciesCount,
        speciesType: speciesType,
        remark: remark,
      );
      final fallbackByPath = <String, DetectionItem>{
        for (final item in items) item.path: item,
      };
      final updatedItems = <DetectionItem>[
        for (final updated in rawUpdatedItems)
          _mergeValidationUpdate(
            fallbackByPath[updated.path] ?? updated,
            updated,
          ),
      ];

      if (!mounted) return updatedItems;
      setState(() {
        _cacheMarkedSpeciesTypes(speciesName, speciesType);
        for (final item in updatedItems) {
          _previewMetadataCache[item.path] = item;
        }
        _previewItems = _replacePreviewItems(_previewItems, updatedItems);
      });
      return updatedItems;
    });
  }

  void _cacheMarkedSpeciesTypes(String? speciesName, String? speciesType) {
    final settings = _settings;
    if (settings == null) return;
    final names = _splitCommaValues(speciesName)
        .where((name) => name != '空' && name != 'Unknown' && name != '未检测')
        .toList();
    final types = _splitCommaValues(speciesType);
    if (names.isEmpty || types.isEmpty) return;

    final nextSpeciesTypes = Map<String, String>.from(settings.speciesTypes);
    var changed = false;
    for (var index = 0; index < names.length; index++) {
      final typeIndex = types.length == 1
          ? 0
          : (index < types.length ? index : types.length - 1);
      final type = types[typeIndex];
      if (type.isEmpty || nextSpeciesTypes[names[index]] == type) continue;
      nextSpeciesTypes[names[index]] = type;
      changed = true;
    }
    if (changed) {
      _settings = settings.copyWith(speciesTypes: nextSpeciesTypes);
    }
  }

  List<String> _splitCommaValues(String? value) {
    return (value ?? '')
        .replaceAll('，', ',')
        .split(',')
        .map((item) => item.trim())
        .where((item) => item.isNotEmpty)
        .toList();
  }

  void _openFileWithSystem(String path) {
    try {
      if (Platform.isWindows) {
        unawaited(Process.run('cmd', ['/c', 'start', '', path]));
      } else if (Platform.isMacOS) {
        unawaited(Process.run('open', [path]));
      } else if (Platform.isLinux) {
        unawaited(Process.run('xdg-open', [path]));
      }
    } catch (error) {
      _showSnackBar('无法打开文件：$error');
    }
  }

  List<String> _selectedSpeciesNames() {
    return _stringListSetting(
      _settingsOrEmpty(),
      'selected_species_names',
      const <String>[],
    ).where((name) => name.trim().isNotEmpty).toList();
  }

  String _compactVersionLabel(String? version) {
    final trimmed = version?.trim() ?? '';
    if (trimmed.isEmpty) return 'Neri';
    return _versionNumberPattern.firstMatch(trimmed)?.group(0) ?? trimmed;
  }

  int _safePreviewIndex(List<DetectionItem> items) {
    if (items.isEmpty) return 0;
    if (_selectedPreviewIndex >= items.length) return items.length - 1;
    if (_selectedPreviewIndex < 0) return 0;
    return _selectedPreviewIndex;
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final titleBarColor = colorScheme.surfaceContainerHigh;
    final showGlobalProgress =
        _loading || _previewDetecting || _submitting || _validationBusy;

    final content = Stack(
      fit: StackFit.expand,
      children: [
        Scaffold(
          appBar: PreferredSize(
            preferredSize: const Size.fromHeight(kToolbarHeight),
            child: DragToMoveArea(
              child: AppBar(
                backgroundColor: titleBarColor,
                surfaceTintColor: Colors.transparent,
                scrolledUnderElevation: 0,
                titleSpacing: 12,
                title: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildTitleLogo(size: 28),
                    const SizedBox(width: 10),
                    Text(_pageTitles[_selectedIndex]),
                  ],
                ),
                actions: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    child: Center(
                      child: Text(_compactVersionLabel(_settings?.appVersion)),
                    ),
                  ),
                  IconButton(
                    tooltip: '刷新',
                    onPressed: _loading || _previewLoading
                        ? null
                        : () async {
                            if (!_backendReady) {
                              await _startBackendAndInitialRefresh();
                            } else {
                              await _refreshInitialPageData();
                            }
                          },
                    icon: const Icon(Icons.refresh_rounded),
                  ),
                  const SizedBox(width: 8),
                  const VerticalDivider(indent: 14, endIndent: 14),
                  const SizedBox(width: 8),
                  IconButton(
                    tooltip: '最小化',
                    icon: const Icon(Icons.remove_rounded, size: 20),
                    onPressed: () => windowManager.minimize(),
                  ),
                  IconButton(
                    tooltip: _isMaximized ? '向下还原' : '最大化',
                    icon: Icon(
                      _isMaximized
                          ? Icons.filter_none_rounded
                          : Icons.crop_square_rounded,
                      size: 18,
                    ),
                    onPressed: () async {
                      if (await windowManager.isMaximized()) {
                        await windowManager.unmaximize();
                      } else {
                        await windowManager.maximize();
                      }
                    },
                  ),
                  Builder(
                    builder: (context) {
                      return IconButton(
                        tooltip: '关闭',
                        icon: const Icon(Icons.close_rounded, size: 20),
                        style: ButtonStyle(
                          overlayColor: WidgetStateProperty.resolveWith((
                            states,
                          ) {
                            if (states.contains(WidgetState.hovered)) {
                              return Colors.red.withValues(alpha: 0.9);
                            }
                            return null;
                          }),
                          foregroundColor: WidgetStateProperty.resolveWith((
                            states,
                          ) {
                            if (states.contains(WidgetState.hovered)) {
                              return Colors.white;
                            }
                            return Theme.of(
                              context,
                            ).colorScheme.onSurfaceVariant;
                          }),
                        ),
                        onPressed: _requestCloseWindow,
                      );
                    },
                  ),
                  const SizedBox(width: 8),
                ],
              ),
            ),
          ),
          body: Stack(
            children: [
              Column(
                children: [
                  // 标题栏下方的全局进度条
                  SizedBox(
                    height: 4, // 固定高度防止页面抖动
                    child: showGlobalProgress
                        ? const ExcludeSemantics(
                            child: LinearProgressIndicator(),
                          )
                        : null,
                  ),
                  // 原有的主体内容区域
                  Expanded(
                    child: Row(
                      children: [
                        _NativeNavigationRail(
                          selectedIndex: _selectedIndex,
                          entries: _navigationItems,
                          onDestinationSelected: (index) {
                            setState(() => _selectedIndex = index);
                            if (index == 1 || index == 2) {
                              unawaited(_refreshPreviewItems());
                            }
                          },
                        ),
                        const VerticalDivider(width: 1),
                        Expanded(
                          child: IndexedStack(
                            index: _selectedIndex,
                            children: [
                              _buildTabWrapper(0, _buildStartScreen()),
                              _buildTabWrapper(1, _buildPreviewPage()),
                              _buildTabWrapper(2, _buildValidationPage()),
                              _buildTabWrapper(3, _buildSettingsScreen()),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        if (_showClosingOverlay)
          _ClosingOverlay(
            phase: _closeDialogPhase,
            backendStopped: _closeBackendStopped,
            onCancel: _cancelCloseWindow,
            onConfirm: _confirmCloseWindow,
          ),
      ],
    );
    return Platform.isWindows ? ExcludeSemantics(child: content) : content;
  }

  Widget _buildTitleLogo({double size = 26}) {
    final logoFile = _resolveLogoFile();
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: Image.file(
          logoFile,
          width: size,
          height: size,
          fit: BoxFit.contain,
          errorBuilder: (context, error, stackTrace) {
            return Icon(
              Icons.camera_outdoor_rounded,
              size: size - 2,
              color: Theme.of(context).colorScheme.primary,
            );
          },
        ),
      ),
    );
  }

  File _resolveLogoFile() {
    final candidates = <File>[File('res/logo.png'), File('../res/logo.png')];
    for (final file in candidates) {
      if (file.existsSync()) return file;
    }
    return candidates.first;
  }

  Widget _buildTabWrapper(int tabIndex, Widget child) {
    final isActive = _selectedIndex == tabIndex;
    return ExcludeSemantics(
      excluding: !isActive,
      child: FocusScope(canRequestFocus: isActive, child: child),
    );
  }

  Widget _buildStartScreen() {
    return StartScreen(
      settings: _settings,
      inputController: _inputController,
      selectedModelPath: _selectedModelPath,
      onModelChanged: (value) => setState(() => _selectedModelPath = value),
      selectedClassificationModelPath: _selectedClassificationModelPath,
      onClassificationModelChanged: (value) =>
          setState(() => _selectedClassificationModelPath = value ?? ''),
      videoMode: _videoMode,
      onVideoModeChanged: (value) => setState(() => _videoMode = value),
      vidStride: _vidStride,
      onVidStrideChanged: (value) => setState(() => _vidStride = value),
      useFp16: _effectiveUseFp16(),
      onUseFp16Changed: (value) => setState(() => _useFp16 = value),
      confidence: _confidence,
      onConfidenceChanged: (value) => setState(() => _confidence = value),
      iou: _iou,
      onIouChanged: (value) => setState(() => _iou = value),
      submitting: _submitting,
      onCreateJob: _createJob,
      onCancelJob: _cancelJob,
      onResumeJob: _resumeJob,
      onDeleteJob: _deleteJob,
      onClearJobs: _clearJobs,
      pendingStartJobIds: _pendingStartJobBaselines.keys.toSet(),
      pendingStopJobIds: _pendingStopJobIds,
      jobs: _jobs,
    );
  }

  Widget _buildSettingsScreen() {
    return SettingsScreen(
      settings: _settings,
      apiClient: widget.apiClient,
      themeNotifier: widget.themeNotifier,
      onUpdateTheme: _updateTheme,
      onSaveSettings: _saveAdvancedSettings,
      onShowMessage: _showSnackBar,
    );
  }

  Widget _buildPreviewPage() {
    final inputPath = _inputController.text.trim();
    final items = inputPath.isEmpty
        ? _sortMediaItemsForDisplay(_jobs.expand((job) => job.results).toList())
        : _previewItems;
    final selectedIndex = _safePreviewIndex(items);
    final selectedItem = items.isEmpty ? null : items[selectedIndex];

    return PreviewScreen(
      inputPath: inputPath,
      items: items,
      selectedIndex: selectedIndex,
      selectedItem: selectedItem,
      speciesTypes: _settings?.speciesTypes ?? const <String, String>{},
      showDetections: _previewShowDetections,
      onShowDetectionsChanged: (value) =>
          setState(() => _previewShowDetections = value),
      selectedSpeciesFilter: _previewSpeciesFilter,
      onSpeciesFilterChanged: (value) =>
          setState(() => _previewSpeciesFilter = value),
      confidenceThreshold: _previewConfidenceThreshold,
      onConfidenceThresholdChanged: (value) =>
          setState(() => _previewConfidenceThreshold = value),
      detecting: _previewDetecting,
      loading: _previewLoading,
      onDetectCurrentImage: _detectCurrentPreviewImage,
      onSelected: (index, item) {
        setState(() => _selectedPreviewIndex = index);
        unawaited(_loadPreviewMetadata(item));
      },
      onLoadMetadata: _loadPreviewMetadata,
      onRefresh: () => _refreshPreviewItems(force: true),
      onOpenExternal: _openFileWithSystem,
    );
  }

  Widget _buildValidationPage() {
    final inputPath = _inputController.text.trim();
    final items = inputPath.isEmpty
        ? _sortMediaItemsForDisplay(_jobs.expand((job) => job.results).toList())
        : _previewItems;

    return SpeciesValidationScreen(
      apiClient: widget.apiClient,
      inputPath: inputPath,
      items: items,
      loading: _previewLoading,
      refreshVersion: _previewRefreshRequestId,
      speciesTypes: _settings?.speciesTypes ?? const <String, String>{},
      autoGroup: _boolSetting(_settingsOrEmpty(), 'auto_group', true),
      collapseGroups: _boolSetting(
        _settingsOrEmpty(),
        'collapse_groups',
        false,
      ),
      autoGroupDetectBurst: _boolSetting(
        _settingsOrEmpty(),
        'auto_group_detect_burst',
        false,
      ),
      autoGroupBurstSize: _intSetting(
        _settingsOrEmpty(),
        'auto_group_burst_size',
        3,
      ),
      autoGroupGapSeconds: _intSetting(
        _settingsOrEmpty(),
        'auto_group_gap_seconds',
        30,
      ),
      autoSortQuickMarks: _boolSetting(_settingsOrEmpty(), 'auto_sort', false),
      undoSteps: _intSetting(
        _settingsOrEmpty(),
        'undo_steps',
        1,
      ).clamp(1, 50).toInt(),
      quickMarkSpecies: _stringListSetting(
        _settingsOrEmpty(),
        'quick_mark_list',
        const <String>[],
      ),
      quickMarkRecentHistory: _stringListSetting(
        _settingsOrEmpty(),
        'quick_mark_recent_history',
        const <String>[],
      ),
      quickMarkUsageCounts: _intMapSetting(
        _settingsOrEmpty(),
        'quick_mark_usage_counts',
      ),
      exportColumns: _stringListSetting(
        _settingsOrEmpty(),
        'export_columns',
        validationExportColumns,
      ),
      onRefresh: () => _refreshPreviewItems(force: true),
      onLoadMetadata: _loadPreviewMetadata,
      onOpenExternal: _openFileWithSystem,
      onMarkItem: _markValidationItem,
      onMarkItems: _markValidationItems,
      onQuickMarkUsed: _recordQuickMarkUsage,
      onRedetectItems: _redetectValidationItems,
    );
  }

  NeriSettings _settingsOrEmpty() {
    return _settings ??
        const NeriSettings(
          appTitle: 'Neri',
          appVersion: '',
          supportedImageExtensions: <String>[],
          supportedVideoExtensions: <String>[],
          modelDirectory: 'res/model',
          classificationModelDirectory: 'res/model_cls',
          availableModels: <ModelInfo>[],
          availableClassificationModels: <ModelInfo>[],
          speciesTypes: <String, String>{},
          settings: <String, dynamic>{},
          gpuAvailable: false,
          missingYoloDependencies: <String>[],
        );
  }
}

class _NavigationRailEntry {
  const _NavigationRailEntry({
    required this.label,
    required this.icon,
    required this.selectedIcon,
  });

  final String label;
  final IconData icon;
  final IconData selectedIcon;
}

class _ClosingOverlay extends StatelessWidget {
  const _ClosingOverlay({
    required this.phase,
    required this.backendStopped,
    required this.onCancel,
    required this.onConfirm,
  });

  final _CloseDialogPhase phase;
  final bool backendStopped;
  final VoidCallback onCancel;
  final VoidCallback onConfirm;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final showActions = phase == _CloseDialogPhase.confirming;
    final showProgress =
        phase != _CloseDialogPhase.confirming || !backendStopped;
    final title = switch (phase) {
      _CloseDialogPhase.confirming => '关闭程序？',
      _CloseDialogPhase.closing => '正在关闭',
      _CloseDialogPhase.restoring => '正在恢复',
    };
    final message = switch (phase) {
      _CloseDialogPhase.confirming =>
        backendStopped
            ? 'Python 后端已关闭。确认后将退出程序，取消会重新启动后端。'
            : '正在先关闭 Python 后端。确认后将立即退出程序，取消会重新启动后端。',
      _CloseDialogPhase.closing => '正在退出程序。若后端仍在收尾，会随主程序退出自动结束。',
      _CloseDialogPhase.restoring => '正在重新启动 Python 后端，请稍候。',
    };

    return Positioned.fill(
      child: Stack(
        children: [
          ModalBarrier(
            dismissible: false,
            color: scheme.scrim.withValues(alpha: 0.36),
          ),
          Center(
            child: Card(
              elevation: 6,
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 20, 24, 18),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          if (showProgress) ...[
                            SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.6,
                                color: scheme.primary,
                              ),
                            ),
                            const SizedBox(width: 14),
                          ],
                          Expanded(
                            child: Text(
                              title,
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Text(
                        message,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                      if (showActions) ...[
                        const SizedBox(height: 18),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            TextButton(
                              onPressed: onCancel,
                              child: const Text('取消'),
                            ),
                            const SizedBox(width: 8),
                            FilledButton(
                              onPressed: onConfirm,
                              child: const Text('关闭'),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Kept for the legacy rail implementation below; the active sidebar uses a
// fixed native NavigationRail to avoid Windows accessibility-tree churn.
class _NavigationToggleButton extends StatelessWidget {
  const _NavigationToggleButton({
    required this.expanded,
    required this.onPressed,
  });

  final bool expanded;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      tooltip: expanded ? '收起侧边栏' : '展开侧边栏',
      onPressed: onPressed,
      icon: Icon(expanded ? Icons.menu_open_rounded : Icons.menu_rounded),
    );
  }
}

class _NativeNavigationRail extends StatelessWidget {
  const _NativeNavigationRail({
    required this.selectedIndex,
    required this.entries,
    required this.onDestinationSelected,
  });

  final int selectedIndex;
  final List<_NavigationRailEntry> entries;
  final ValueChanged<int> onDestinationSelected;

  static const _width = 80.0;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    // 移除 Semantics 和 ExcludeSemantics 包裹
    return NavigationRail(
      backgroundColor: colorScheme.surfaceContainerLowest,
      labelType: NavigationRailLabelType.all,
      minWidth: _width,
      selectedIndex: selectedIndex,
      groupAlignment: -0.85,
      useIndicator: true,
      indicatorColor: colorScheme.secondaryContainer,
      indicatorShape: const StadiumBorder(),
      selectedIconTheme: IconThemeData(
        color: colorScheme.onSecondaryContainer,
        size: 26,
      ),
      unselectedIconTheme: IconThemeData(
        color: colorScheme.onSurfaceVariant,
        size: 26,
      ),
      selectedLabelTextStyle: TextStyle(
        color: colorScheme.onSecondaryContainer,
        fontWeight: FontWeight.w700,
      ),
      unselectedLabelTextStyle: TextStyle(
        color: colorScheme.onSurfaceVariant,
        fontWeight: FontWeight.w600,
      ),
      onDestinationSelected: onDestinationSelected,
      destinations: [
        for (final entry in entries)
          NavigationRailDestination(
            icon: Icon(entry.icon),
            selectedIcon: Icon(entry.selectedIcon),
            label: Text(entry.label),
          ),
      ],
    );
  }
}

// ignore: unused_element
class _ModalNavigationRail extends StatelessWidget {
  const _ModalNavigationRail({
    required this.expanded,
    required this.selectedIndex,
    required this.entries,
    required this.onToggleExpanded,
    required this.onDestinationSelected,
    required this.child,
  });

  final bool expanded;
  final int selectedIndex;
  final List<_NavigationRailEntry> entries;
  final VoidCallback onToggleExpanded;
  final ValueChanged<int> onDestinationSelected;
  final Widget child;

  static const _collapsedWidth = 88.0;
  static const _expandedWidth = 220.0;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Stack(
      children: [
        // 1. 底层页面内容
        Row(
          children: [
            const SizedBox(width: _collapsedWidth),
            Expanded(child: child),
          ],
        ),

        // 2. 遮罩层 (Scrim)
        Positioned.fill(
          left: _collapsedWidth,
          child: IgnorePointer(
            ignoring: !expanded,
            child: AnimatedOpacity(
              opacity: expanded ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 250),
              curve: Curves.easeInOutCubic,
              child: GestureDetector(
                onTap: onToggleExpanded,
                child: ColoredBox(
                  color: colorScheme.scrim.withValues(alpha: 0.32),
                ),
              ),
            ),
          ),
        ),

        // 3. 原生 NavigationRail 侧边栏
        Positioned(
          left: 0,
          top: 0,
          bottom: 0,
          // 💡 关键修改 1：使用 Material 替代 AnimatedPhysicalModel
          // Material 原生支持阴影和形状的隐式动画，能完美实现平滑的右侧圆角过渡
          child: Material(
            animationDuration: const Duration(milliseconds: 250),
            elevation: expanded ? 8.0 : 0.0,
            color: colorScheme.surfaceContainerLowest,
            shadowColor: colorScheme.shadow.withValues(alpha: 0.4),
            // 💡 关键修改 2：展开时赋予右侧圆角，收起时恢复直角
            borderRadius: expanded
                ? const BorderRadius.horizontal(right: Radius.circular(16))
                : BorderRadius.zero,
            clipBehavior: Clip.antiAlias, // 确保点击高亮和水波纹不会溢出圆角边界
            child: _NativeNavigationRailPanel(
              extended: expanded,
              selectedIndex: selectedIndex,
              entries: entries,
              onToggleExpanded: onToggleExpanded,
              onDestinationSelected: onDestinationSelected,
            ),
          ),
        ),

        // 4. 绝对固定的 Menu 按钮
        Positioned(
          left: 0,
          top: 10,
          child: SizedBox(
            width: _collapsedWidth,
            height: 48,
            child: Align(
              alignment: Alignment.center,
              child: _NavigationToggleButton(
                expanded: expanded,
                onPressed: onToggleExpanded,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ignore: unused_element
class _NativeNavigationRailPanel extends StatelessWidget {
  const _NativeNavigationRailPanel({
    required this.extended,
    required this.selectedIndex,
    required this.entries,
    required this.onToggleExpanded,
    required this.onDestinationSelected,
  });

  final bool extended;
  final int selectedIndex;
  final List<_NavigationRailEntry> entries;
  final VoidCallback onToggleExpanded;
  final ValueChanged<int> onDestinationSelected;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return NavigationRail(
      backgroundColor: Colors.transparent,
      extended: extended,
      minWidth: _ModalNavigationRail._collapsedWidth,
      minExtendedWidth: _ModalNavigationRail._expandedWidth,
      selectedIndex: selectedIndex,
      // 💡 关键修改 3：移除了动态改变的 labelType！
      // 保持默认设置，不要强制指定 labelType。
      // 这会让 Flutter 内部的 Tween 动画自然生效：折叠时无文字，展开时文字平滑淡入，且 Indicator 平滑拉伸。
      groupAlignment: -0.95,
      useIndicator: true,
      indicatorColor: colorScheme.secondaryContainer,
      indicatorShape: const StadiumBorder(),
      selectedIconTheme: IconThemeData(
        color: colorScheme.onSecondaryContainer,
        size: 26,
      ),
      unselectedIconTheme: IconThemeData(
        color: colorScheme.onSurfaceVariant,
        size: 26,
      ),
      selectedLabelTextStyle: TextStyle(
        color: colorScheme.onSecondaryContainer,
        fontWeight: FontWeight.w700,
      ),
      unselectedLabelTextStyle: TextStyle(
        color: colorScheme.onSurfaceVariant,
        fontWeight: FontWeight.w600,
      ),
      leading: const Padding(
        padding: EdgeInsets.only(top: 10, bottom: 18),
        child: SizedBox(
          width: _ModalNavigationRail._collapsedWidth,
          height: 48,
        ),
      ),
      onDestinationSelected: onDestinationSelected,
      destinations: [
        for (final entry in entries)
          NavigationRailDestination(
            icon: Icon(entry.icon),
            selectedIcon: Icon(entry.selectedIcon),
            label: Text(entry.label),
          ),
      ],
    );
  }
}
