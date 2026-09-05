import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:media_kit/media_kit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart';

import 'api_client.dart';
import 'app_updater.dart';
import 'crash_reporter.dart';
import 'crash_watchdog.dart';
import 'local_maintenance_status.dart';
import 'models/close_behavior.dart';
import 'models/job.dart';
import 'models/settings.dart';
import 'models/theme_settings.dart';
import 'models/video_processing_mode.dart';
import 'privacy/privacy_gate_overlay.dart';
import 'privacy/privacy_status.dart';
import 'screens/preview_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/species_validation_screen.dart';
import 'screens/start_screen.dart';
import 'utils/job_result_refresh.dart';
import 'utils/local_detection_items.dart';

const _lastInputPathKey = 'last_input_path';

enum _CloseDialogPhase { choosing, shuttingDown, restoring }

enum _CloseAction { hideToTray, exit }

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
  static const _windowsShellChannel = MethodChannel('neri/windows_shell');
  static const _supportedMediaExtensions = <String>{
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.tiff',
    '.webp',
    '.mp4',
    '.avi',
    '.mov',
    '.mkv',
    '.wmv',
    '.flv',
    '.webm',
    '.m4v',
    '.ts',
  };
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
  static const _backendOutputTailMaxCharacters = 12000;

  final _inputController = TextEditingController();
  final Map<String, DetectionItem> _previewMetadataCache =
      <String, DetectionItem>{};
  final Set<String> _previewMetadataLoading = <String>{};
  final Set<int> _expectedBackendExitPids = <int>{};
  final _maintenanceStatusStore = LocalMaintenanceStatusStore();
  final _appUpdater = AppUpdater();

  NeriSettings? _settings;
  List<ProcessingJob> _jobs = const <ProcessingJob>[];
  List<DetectionItem> _previewItems = const <DetectionItem>[];
  Timer? _timer;
  Timer? _previewRefreshTimer;
  Timer? _inputDirectoryChangeTimer;
  Timer? _modelSelectionSaveTimer;
  StreamSubscription<FileSystemEvent>? _inputDirectoryWatcher;
  Process? _backendProcess;
  Future<void>? _backendShutdownTask;
  Future<void>? _closeBackendShutdownTask;
  int _previewRefreshRequestId = 0;
  int _previewContentVersion = 0;
  int _modelSelectionRevision = 0;
  int _closeFlowId = 0;
  int _closeBehaviorRevision = 0;
  int _lastCloseActionRevision = 0;
  bool _loading = true;
  bool _backendStarting = false;
  bool _backendReady = false;
  bool _submitting = false;
  bool _previewLoading = false;
  bool _settingsSaving = false;
  bool _modelSelectionSaveInProgress = false;
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
  bool _closeSelectionSaving = false;
  bool _closeDontAskAgain = false;
  bool _checkingForAppUpdate = false;
  bool _startupUpdateCheckScheduled = false;
  bool _installingDownloadedAppUpdate = false;
  bool _softwareUpdateCardDismissed = false;
  _CloseDialogPhase _closeDialogPhase = _CloseDialogPhase.choosing;
  _CloseAction _selectedCloseAction = _CloseAction.exit;
  String _closeBehavior = closeBehaviorAsk;
  _SoftwareUpdateProgress? _softwareUpdateProgress;
  _AvailableSoftwareUpdate? _availableSoftwareUpdate;
  DownloadedAppUpdate? _downloadedAppUpdate;
  Directory? _downloadedAppUpdateInstallDirectory;
  DateTime? _lastSoftwareUpdateProgressPaint;
  double _confidence = 0.25;
  double _iou = 0.45;
  final Map<String, double> _previewConfidenceSettings = <String, double>{
    previewAllSpeciesLabel: 0.25,
  };
  String _previewSpeciesFilter = previewAllSpeciesLabel;
  double get _previewConfidenceThreshold =>
      _previewConfidenceSettings[_previewSpeciesFilter] ??
      _previewConfidenceSettings[previewAllSpeciesLabel] ??
      0.25;
  String? _selectedModelPath;
  String? _selectedClassificationModelPath;
  MaintenanceStatus? _startupMaintenanceStatus;
  String _backendOutputTail = '';
  String _videoMode = defaultVideoProcessingMode;
  int _imageSize = 1920;
  int _vidStride = defaultVideoSampleCount;
  int? _autoGroupInferredBurstSize;
  String? _previewLoadedPath;
  String? _watchedInputDirectory;
  int _selectedIndex = 0;
  int _selectedPreviewIndex = 0;
  String? _lastWindowsShellStatusSignature;
  bool _duplicateLaunchDialogVisible = false;
  PrivacyStatus? _privacyStatus;
  bool _privacyLoading = true;
  String? _privacyLoadError;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    unawaited(windowManager.setPreventClose(true));
    _initWindowState();
    if (Platform.isWindows) {
      _windowsShellChannel.setMethodCallHandler(_handleWindowsShellCall);
    }

    try {
      MediaKit.ensureInitialized();
    } catch (_) {}

    _inputController.addListener(() {
      _schedulePreviewRefresh();
      _updateInputDirectoryWatcher();
      unawaited(_updateWindowsShellStatus());
    });
    _loadLastInputPath();
    _loadCloseBehavior();
    _loadLastCloseAction();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_updateWindowsShellStatus());
      unawaited(_startBackendAndInitialRefresh());
    });
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    if (Platform.isWindows) {
      _windowsShellChannel.setMethodCallHandler(null);
    }
    _timer?.cancel();
    _previewRefreshTimer?.cancel();
    _inputDirectoryChangeTimer?.cancel();
    _modelSelectionSaveTimer?.cancel();
    final inputDirectoryWatcher = _inputDirectoryWatcher;
    if (inputDirectoryWatcher != null) {
      unawaited(inputDirectoryWatcher.cancel());
    }
    _appUpdater.close();
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

  ProcessingJob? get _windowsShellJob {
    for (final job in _jobs) {
      if (job.isWorkerActive) return job;
    }
    for (final job in _jobs) {
      if (job.canResume) return job;
    }
    return null;
  }

  Future<dynamic> _handleWindowsShellCall(MethodCall call) async {
    if (call.method == 'duplicateLaunch') {
      await _showAndFocusWindow();
      await _showDuplicateLaunchDialog();
      return null;
    }
    if (call.method != 'trayAction') return null;
    switch (call.arguments) {
      case 'toggleProcessing':
        if (_privacyStatus?.isConsentComplete != true) {
          await _showAndFocusWindow();
          return null;
        }
        final job = _windowsShellJob;
        if (job == null) {
          if (_canCreateJobFromTray) await _createJob();
        } else if (job.isWorkerActive) {
          if (!_pendingStopJobIds.contains(job.id)) {
            await _cancelJob(job);
          }
        } else if (job.canResume &&
            !_pendingStartJobBaselines.containsKey(job.id)) {
          await _resumeJob(job);
        }
      case 'openSettings':
        if (mounted) setState(() => _selectedIndex = 3);
        await _showAndFocusWindow();
      case 'exit':
        await _showAndFocusWindow();
        if (mounted) _requestCloseWindow(forceExit: true);
    }
    return null;
  }

  Future<void> _showDuplicateLaunchDialog() async {
    if (!mounted || _duplicateLaunchDialogVisible) return;
    _duplicateLaunchDialogVisible = true;
    try {
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          icon: const Icon(Icons.info_outline_rounded),
          title: const Text('Neri 已经打开'),
          content: const Text('已切换到正在运行的 Neri 窗口，无需重复启动。'),
          actions: [
            FilledButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('知道了'),
            ),
          ],
        ),
      );
    } finally {
      _duplicateLaunchDialogVisible = false;
    }
  }

  Future<void> _showAndFocusWindow() async {
    try {
      await windowManager.show();
      if (await windowManager.isMinimized()) {
        await windowManager.restore();
      }
      await windowManager.focus();
    } catch (_) {}
  }

  Future<void> _updateWindowsShellStatus() async {
    if (!Platform.isWindows || !mounted) return;
    final job = _windowsShellJob;
    final isRunning = job?.isWorkerActive ?? false;
    final isFailed = job?.state == 'failed';
    final hasProgress = job != null && job.total > 0;
    final progress = hasProgress
        ? job.progress.clamp(0.0, 1.0).toDouble()
        : 0.0;
    final percent = (progress * 100).round();

    String tooltip = 'Neri';
    if (job != null) {
      final counts = hasProgress
          ? ' ($percent% · ${job.processed}/${job.total})'
          : '';
      if (isRunning) {
        tooltip = hasProgress ? 'Neri - 任务进度$counts' : 'Neri - 任务准备中';
      } else if (isFailed) {
        tooltip = 'Neri - 任务失败$counts';
      } else {
        tooltip = 'Neri - 任务已暂停$counts';
      }
    }

    final privacyReady = _privacyStatus?.isConsentComplete == true;
    final taskActionEnabled =
        privacyReady &&
        (job == null
            ? _canCreateJobFromTray
            : isRunning
            ? !_pendingStopJobIds.contains(job.id)
            : job.canResume && !_pendingStartJobBaselines.containsKey(job.id));
    final progressState = job == null
        ? 'none'
        : isRunning
        ? (hasProgress ? 'normal' : 'indeterminate')
        : isFailed
        ? 'error'
        : 'paused';
    final status = <String, Object>{
      'tooltip': tooltip,
      'taskActionEnabled': taskActionEnabled,
      'taskIsRunning': isRunning,
      'progressState': progressState,
      'progress': progress,
    };
    final signature = jsonEncode(status);
    if (_lastWindowsShellStatusSignature == signature) return;
    try {
      await _windowsShellChannel.invokeMethod<void>('updateStatus', status);
      _lastWindowsShellStatusSignature = signature;
    } on MissingPluginException {
      // Non-Windows test runners do not register the native shell channel.
    } on PlatformException {
      // A transient native-shell failure should not interrupt processing.
    }
  }

  bool get _canCreateJobFromTray =>
      _privacyStatus?.isConsentComplete == true &&
      _backendReady &&
      _settings != null &&
      !_submitting &&
      _inputController.text.trim().isNotEmpty;

  @override
  void onWindowClose() {
    _requestCloseWindow();
  }

  bool get _closeFlowBlocksBackendStartup =>
      _closing && _closeDialogPhase != _CloseDialogPhase.restoring;

  String get _configuredCloseBehavior => normalizeCloseBehavior(_closeBehavior);

  void _requestCloseWindow({bool forceExit = false}) {
    if (_closing || _showClosingOverlay) return;
    final behavior = forceExit ? closeBehaviorExit : _configuredCloseBehavior;
    if (behavior == closeBehaviorHideToTray) {
      unawaited(_hideWindowToTray());
      return;
    }
    if (behavior == closeBehaviorExit) {
      _beginBackendShutdown();
      return;
    }

    ++_closeFlowId;
    if (!mounted) return;
    setState(() {
      _closeDialogPhase = _CloseDialogPhase.choosing;
      _closeDontAskAgain = false;
      _closeSelectionSaving = false;
      _showClosingOverlay = true;
    });
  }

  void _dismissCloseWindowPrompt() {
    if (_closeDialogPhase != _CloseDialogPhase.choosing ||
        _closeSelectionSaving) {
      return;
    }
    ++_closeFlowId;
    setState(() {
      _showClosingOverlay = false;
      _closeDontAskAgain = false;
      _closeSelectionSaving = false;
    });
  }

  Future<void> _hideWindowToTray() async {
    try {
      await windowManager.hide();
    } catch (error) {
      if (mounted) _showSnackBar('隐藏到任务托盘失败：$error');
    }
  }

  Future<void> _confirmCloseWindow() async {
    if (_closeDialogPhase != _CloseDialogPhase.choosing ||
        _closeSelectionSaving) {
      return;
    }
    final flowId = _closeFlowId;
    final selectedAction = _selectedCloseAction;
    final dontAskAgain = _closeDontAskAgain;
    setState(() => _closeSelectionSaving = true);
    try {
      await _saveLastCloseAction(selectedAction);
    } catch (_) {
      // A preference persistence failure must not block the selected action.
    }
    if (!mounted ||
        flowId != _closeFlowId ||
        _closeDialogPhase != _CloseDialogPhase.choosing) {
      return;
    }
    if (dontAskAgain) {
      final behavior = selectedAction == _CloseAction.hideToTray
          ? closeBehaviorHideToTray
          : closeBehaviorExit;
      try {
        await _updateCloseBehavior(behavior);
      } catch (_) {
        // A preference persistence failure must not block the selected action.
      }
      if (!mounted ||
          flowId != _closeFlowId ||
          _closeDialogPhase != _CloseDialogPhase.choosing) {
        return;
      }
    }

    if (selectedAction == _CloseAction.hideToTray) {
      setState(() {
        _showClosingOverlay = false;
        _closeSelectionSaving = false;
        _closeDontAskAgain = false;
      });
      await _hideWindowToTray();
      return;
    }
    _beginBackendShutdown();
  }

  void _beginBackendShutdown() {
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
        _closeDialogPhase = _CloseDialogPhase.shuttingDown;
        _closeSelectionSaving = false;
        _showClosingOverlay = true;
      });
    }
    final shutdownTask = _shutdownBackend();
    _closeBackendShutdownTask = shutdownTask;
    unawaited(
      shutdownTask
          .catchError((_) {})
          .whenComplete(() => _finishCloseWindow(flowId)),
    );
  }

  void _cancelCloseWindow() {
    if (!_closing || _closeDialogPhase != _CloseDialogPhase.shuttingDown) {
      return;
    }
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

    CrashWatchdog.markNormalExit();
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
          _closeSelectionSaving = false;
          _closeDontAskAgain = false;
          _closeBackendShutdownTask = null;
          _closeDialogPhase = _CloseDialogPhase.choosing;
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

  Future<void> _loadCloseBehavior() async {
    final revision = _closeBehaviorRevision;
    final preferences = await SharedPreferences.getInstance();
    final behavior = normalizeCloseBehavior(
      preferences.getString(closeBehaviorSettingKey),
    );
    if (!mounted || revision != _closeBehaviorRevision) return;
    setState(() => _closeBehavior = behavior);
  }

  Future<void> _loadLastCloseAction() async {
    final revision = _lastCloseActionRevision;
    final preferences = await SharedPreferences.getInstance();
    final savedAction = normalizeLastCloseAction(
      preferences.getString(lastCloseActionSettingKey),
    );
    if (!mounted || revision != _lastCloseActionRevision) return;
    setState(() {
      _selectedCloseAction = savedAction == closeBehaviorExit
          ? _CloseAction.exit
          : _CloseAction.hideToTray;
    });
  }

  Future<void> _updateLastCloseAction(_CloseAction action) async {
    if (_selectedCloseAction == action) return;
    final previous = _selectedCloseAction;
    final revision = ++_lastCloseActionRevision;
    if (mounted) setState(() => _selectedCloseAction = action);
    try {
      await _saveLastCloseAction(action);
      if (mounted && revision != _lastCloseActionRevision) {
        await _saveLastCloseAction(_selectedCloseAction);
      }
    } catch (error) {
      if (mounted && revision == _lastCloseActionRevision) {
        setState(() => _selectedCloseAction = previous);
        _showSnackBar('保存上次关闭选择失败：$error');
      }
    }
  }

  Future<void> _saveLastCloseAction(_CloseAction action) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(
      lastCloseActionSettingKey,
      action == _CloseAction.exit ? closeBehaviorExit : closeBehaviorHideToTray,
    );
  }

  Future<void> _updateCloseBehavior(String behavior) async {
    final normalized = normalizeCloseBehavior(behavior);
    final previous = _closeBehavior;
    final revision = ++_closeBehaviorRevision;
    if (mounted && _closeBehavior != normalized) {
      setState(() => _closeBehavior = normalized);
    }
    try {
      final preferences = await SharedPreferences.getInstance();
      await preferences.setString(closeBehaviorSettingKey, normalized);
    } catch (_) {
      if (mounted && revision == _closeBehaviorRevision) {
        setState(() => _closeBehavior = previous);
      }
      rethrow;
    }
  }

  Future<void> _updateCloseBehaviorFromSettings(String behavior) async {
    try {
      await _updateCloseBehavior(behavior);
    } catch (error) {
      if (mounted) _showSnackBar('保存关闭方式失败：$error');
    }
  }

  Future<void> _startBackendAndInitialRefresh() async {
    if (_closeFlowBlocksBackendStartup) return;
    if (_backendStarting) return;
    _backendStarting = true;
    if (mounted) {
      setState(() {
        _loading = true;
        if (_privacyStatus?.isConsentComplete != true) {
          _privacyLoading = true;
          _privacyLoadError = null;
        }
      });
    }
    try {
      await _waitForActiveEnvironmentMaintenance();
      if (!mounted || _closeFlowBlocksBackendStartup) return;
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
        setState(() {
          _loading = false;
          _privacyLoading = false;
          _privacyLoadError = 'Python 后端启动超时，请检查 toolkit\\python.exe 和端口 721。';
        });
        return;
      }
      _backendReady = true;
      final privacyReady = await _loadPrivacyStatus();
      if (!mounted || _closeFlowBlocksBackendStartup) return;
      if (!privacyReady) {
        setState(() => _loading = false);
        return;
      }
      await _refreshInitialPageData();
      _timer ??= Timer.periodic(
        const Duration(seconds: 2),
        (_) => _refresh(silent: true),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _privacyLoading = false;
        if (_privacyStatus?.isConsentComplete != true) {
          _privacyStatus = null;
          _privacyLoadError = '启动或连接本地服务失败：$error';
        }
      });
      if (_privacyStatus?.isConsentComplete == true) {
        _showSnackBar('启动 Python 后端失败：$error');
      }
    } finally {
      _backendStarting = false;
    }
  }

  Future<bool> _loadPrivacyStatus() async {
    if (mounted) {
      setState(() {
        _privacyLoading = true;
        _privacyLoadError = null;
      });
    }
    try {
      final status = await widget.apiClient.fetchPrivacyStatus();
      if (!mounted) return false;
      setState(() {
        _privacyStatus = status;
        _privacyLoading = false;
      });
      return status.isConsentComplete;
    } catch (error) {
      if (!mounted) return false;
      setState(() {
        _privacyStatus = null;
        _privacyLoading = false;
        _privacyLoadError = error.toString();
      });
      return false;
    }
  }

  Future<void> _completePrivacyConsent(PrivacyStatus status) async {
    if (!status.isConsentComplete || !mounted) return;
    setState(() {
      _privacyStatus = status;
      _privacyLoading = false;
      _privacyLoadError = null;
      _loading = true;
    });
    await _refreshInitialPageData();
    if (!mounted || _closeFlowBlocksBackendStartup) return;
    _timer ??= Timer.periodic(
      const Duration(seconds: 2),
      (_) => _refresh(silent: true),
    );
  }

  Future<void> _waitForActiveEnvironmentMaintenance() async {
    var status = await _maintenanceStatusStore.tryRead();
    if (status == null || !_maintenanceStatusStore.isActive(status)) return;

    var missingWorkerChecks = 0;
    while (mounted && !_closeFlowBlocksBackendStartup) {
      if (!_maintenanceStatusStore.isActive(status!)) break;
      setState(() => _startupMaintenanceStatus = status);

      if (await _maintenanceStatusStore.isWorkerRunning(status)) {
        missingWorkerChecks = 0;
      } else {
        missingWorkerChecks++;
        if (missingWorkerChecks >= 3) {
          await _maintenanceStatusStore.markInterrupted(status);
          break;
        }
      }

      await Future<void>.delayed(const Duration(seconds: 1));
      status =
          await _maintenanceStatusStore.tryRead(path: status.statusPath) ??
          status;
    }

    if (mounted) setState(() => _startupMaintenanceStatus = null);
  }

  Future<void> _refreshInitialPageData() async {
    await _refresh(includeJobResults: true, finishLoading: false);
    if (!mounted || _closeFlowBlocksBackendStartup) return;
    _scheduleStartupUpdateCheck();
    if (_inputController.text.trim().isEmpty) {
      _stopGlobalLoading();
      return;
    }
    _previewRefreshTimer?.cancel();
    _previewRefreshTimer = null;
    await _refreshPreviewItems(force: true, finishGlobalLoading: true);
  }

  void _scheduleStartupUpdateCheck() {
    if (_startupUpdateCheckScheduled ||
        !_appUpdater.isSupported ||
        _settings == null) {
      return;
    }
    _startupUpdateCheckScheduled = true;
    unawaited(
      Future<void>.delayed(const Duration(milliseconds: 700), () async {
        if (!mounted || _closing) return;
        await _checkForSoftwareUpdate(manual: false);
      }),
    );
  }

  Future<void> _checkForSoftwareUpdate({
    required bool manual,
    String? channel,
    String? downloadSource,
  }) async {
    if (_checkingForAppUpdate || _closing) {
      if (manual) _showSnackBar('正在检查或安装更新，请稍候。');
      return;
    }
    if (_downloadedAppUpdate != null) {
      if (manual && mounted) {
        setState(() => _softwareUpdateCardDismissed = false);
        _showSnackBar('更新已下载，可从右下角选择重启更新。');
      }
      return;
    }
    if (_availableSoftwareUpdate != null) {
      if (manual && mounted) {
        setState(() => _softwareUpdateCardDismissed = false);
        _showSnackBar('已发现新版本，请在右下角选择是否下载更新。');
      }
      return;
    }
    final settings = _settings;
    if (settings == null) {
      if (manual) _showSnackBar('版本信息尚未加载，请稍后重试。');
      return;
    }
    final installDirectory = _resolveProjectRoot();
    if (installDirectory == null) {
      if (manual) _showSnackBar('未找到程序目录，无法执行自动更新。');
      return;
    }
    final selectedChannel =
        channel ?? _stringSetting(settings, 'update_channel', 'Preview');
    final selectedDownloadSource =
        downloadSource ?? _stringSetting(settings, 'update_source', 'auto');
    final downloadSourceDetail = switch (selectedDownloadSource) {
      'domestic' => '将使用 Neri 国内源检查并下载更新。',
      'github' => '将使用 GitHub 官方源检查并下载更新。',
      _ => '正在选择最合适的更新源',
    };

    _checkingForAppUpdate = true;
    if (manual && mounted) {
      setState(() {
        _softwareUpdateCardDismissed = false;
        _softwareUpdateProgress = _SoftwareUpdateProgress(
          message: '正在检查最新版本…',
          detail: downloadSourceDetail,
        );
      });
    }

    try {
      final useMainlandChinaSource = await widget.apiClient
          .shouldUseMainlandUpdateSource(selectedDownloadSource);
      final release = await _appUpdater.checkForUpdate(
        currentVersion: settings.appVersion,
        channel: selectedChannel,
        useMainlandChinaSource: useMainlandChinaSource,
      );
      if (!mounted || _closing) return;
      setState(() => _softwareUpdateProgress = null);
      if (release == null) {
        if (manual) _showSnackBar('当前已是所选通道的最新版本。');
        return;
      }
      setState(() {
        _softwareUpdateCardDismissed = false;
        _availableSoftwareUpdate = _AvailableSoftwareUpdate(
          release: release,
          installDirectory: installDirectory,
        );
      });
    } catch (error) {
      if (!mounted || _closing) return;
      setState(() {
        _softwareUpdateProgress = null;
      });
      if (manual) {
        _showSnackBar('检查更新失败：$error');
      }
    } finally {
      _checkingForAppUpdate = false;
    }
  }

  void _dismissAvailableSoftwareUpdate() {
    if (!mounted) return;
    setState(() => _availableSoftwareUpdate = null);
  }

  void _startAvailableSoftwareUpdateDownload() {
    final available = _availableSoftwareUpdate;
    if (available == null || _checkingForAppUpdate || _closing) return;
    _checkingForAppUpdate = true;
    setState(() {
      _availableSoftwareUpdate = null;
      _softwareUpdateCardDismissed = false;
    });
    _showSnackBar('更新下载已开始，完成后会显示“重启并更新”。');
    unawaited(
      _downloadAppUpdate(
            available.release,
            installDirectory: available.installDirectory,
          )
          .catchError((Object error, StackTrace stackTrace) {
            if (mounted && !_closing) {
              setState(() => _softwareUpdateProgress = null);
              _showSnackBar('下载更新失败：$error');
            }
          })
          .whenComplete(() => _checkingForAppUpdate = false),
    );
  }

  Future<void> _downloadAppUpdate(
    AppUpdateRelease release, {
    required Directory installDirectory,
  }) async {
    _lastSoftwareUpdateProgressPaint = null;
    if (mounted) {
      setState(() {
        _softwareUpdateProgress = const _SoftwareUpdateProgress(
          message: '正在准备下载更新…',
          detail: '更新包将保存到系统临时文件夹。',
          isDownloading: true,
        );
      });
    }

    try {
      final downloaded = await _appUpdater.downloadUpdate(
        release,
        onProgress: _handleSoftwareUpdateDownloadProgress,
      );
      if (!mounted || _closing) {
        return;
      }
      setState(() {
        _downloadedAppUpdate = downloaded;
        _downloadedAppUpdateInstallDirectory = installDirectory;
        _softwareUpdateProgress = null;
        _softwareUpdateCardDismissed = false;
      });
      _showSnackBar('更新下载完成，可从右下角选择“重启并更新”。');
    } catch (error) {
      // Keep completed and partial files so a later attempt can reuse them.
      if (mounted && !_closing) {
        setState(() {
          _softwareUpdateProgress = null;
        });
        _showSnackBar('下载更新失败：$error');
      }
    }
  }

  void _dismissSoftwareUpdateCard() {
    if (!mounted) return;
    setState(() => _softwareUpdateCardDismissed = true);
  }

  Future<void> _installDownloadedAppUpdate() async {
    final downloaded = _downloadedAppUpdate;
    final installDirectory = _downloadedAppUpdateInstallDirectory;
    if (downloaded == null ||
        installDirectory == null ||
        _installingDownloadedAppUpdate ||
        _closing) {
      return;
    }
    _installingDownloadedAppUpdate = true;
    if (mounted) {
      setState(() {
        _softwareUpdateCardDismissed = false;
        _softwareUpdateProgress = const _SoftwareUpdateProgress(
          message: '正在启动更新脚本…',
          detail: '确认安装程序已启动后，Neri 将自动关闭。',
          fraction: 1,
        );
      });
    }
    try {
      await _appUpdater.launchInstaller(
        update: downloaded,
        installDirectory: installDirectory,
        restartExecutable: _restartExecutable(installDirectory),
      );
      if (!mounted) return;
      setState(() {
        _softwareUpdateProgress = const _SoftwareUpdateProgress(
          message: '正在关闭程序并安装更新…',
          detail: '请勿手动关闭更新脚本。安装完成后 Neri 会自动重新启动。',
          fraction: 1,
        );
      });
      await _exitForSoftwareUpdate();
    } catch (error) {
      if (mounted && !_closing) {
        setState(() => _softwareUpdateProgress = null);
        _showSnackBar('启动更新安装失败：$error');
      }
    } finally {
      _installingDownloadedAppUpdate = false;
    }
  }

  void _handleSoftwareUpdateDownloadProgress(UpdateDownloadProgress progress) {
    if (!mounted || _closing) return;
    final now = DateTime.now();
    final isComplete =
        progress.totalBytes != null &&
        progress.receivedBytes >= progress.totalBytes!;
    if (!isComplete &&
        _lastSoftwareUpdateProgressPaint != null &&
        now.difference(_lastSoftwareUpdateProgressPaint!).inMilliseconds < 80) {
      return;
    }
    _lastSoftwareUpdateProgressPaint = now;
    final total = progress.totalBytes;
    final detail = total == null
        ? '已下载 ${_formatByteCount(progress.receivedBytes)}'
        : '${_formatByteCount(progress.receivedBytes)} / ${_formatByteCount(total)}';
    setState(() {
      _softwareUpdateProgress = _SoftwareUpdateProgress(
        message: '正在从${progress.sourceLabel}下载更新…',
        detail: detail,
        fraction: progress.fraction,
        isDownloading: true,
      );
    });
  }

  File _restartExecutable(Directory installDirectory) {
    final launcher = _fileUnder(installDirectory, ['Neri.exe']);
    return launcher.existsSync()
        ? launcher
        : File(Platform.resolvedExecutable).absolute;
  }

  Future<void> _exitForSoftwareUpdate() async {
    _closing = true;
    _timer?.cancel();
    _timer = null;
    _previewRefreshTimer?.cancel();
    _previewRefreshTimer = null;
    _backendReady = false;
    try {
      await _shutdownBackend().timeout(const Duration(seconds: 10));
    } catch (_) {}
    try {
      await windowManager
          .setPreventClose(false)
          .timeout(const Duration(milliseconds: 300));
    } catch (_) {}
    CrashWatchdog.markNormalExit();
    try {
      await windowManager.destroy().timeout(const Duration(milliseconds: 500));
    } catch (_) {}
    exit(0);
  }

  String _formatByteCount(int bytes) {
    if (bytes < 1024) return '$bytes B';
    const units = <String>['KB', 'MB', 'GB', 'TB'];
    var value = bytes / 1024;
    var unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }
    return '${value.toStringAsFixed(value >= 100 ? 0 : 1)} ${units[unitIndex]}';
  }

  Future<void> _restartBackendAndRefresh() async {
    if (_closeFlowBlocksBackendStartup || _backendStarting) return;
    _timer?.cancel();
    _timer = null;
    _previewRefreshTimer?.cancel();
    _previewRefreshTimer = null;
    if (mounted) {
      setState(() {
        _loading = true;
        _backendReady = false;
      });
    }
    await _forceShutdownBackend();
    if (!mounted || _closeFlowBlocksBackendStartup) return;
    await _startBackendAndInitialRefresh();
    if (mounted && _backendReady) {
      _showSnackBar('Python 后端已重启');
    }
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

    final backendLogFile = CrashReporter.createBackendLogFile();
    _backendOutputTail = '';
    backendLogFile.writeAsStringSync(
      'Command: ${pythonExe.path} -m uvicorn system.backend.main:app --app-dir . --host 127.0.0.1 --port 721\n'
      'Working directory: ${projectRoot.path}\n\n',
      encoding: utf8,
      mode: FileMode.append,
      flush: true,
    );

    late final Process process;
    try {
      process = await Process.start(
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
    } catch (_) {
      CrashReporter.deleteBackendLogFile(backendLogFile);
      rethrow;
    }
    _backendProcess = process;
    _bindBackendProcessLogging(process, backendLogFile);
  }

  void _bindBackendProcessLogging(Process process, File logFile) {
    void writeOutput(String streamName, String data) {
      final formatted = _formatBackendOutput(streamName, data);
      _appendBackendOutputTail(formatted);
      try {
        logFile.writeAsStringSync(
          formatted,
          encoding: utf8,
          mode: FileMode.append,
        );
      } catch (_) {}
    }

    final stdoutDone = process.stdout
        .transform(utf8.decoder)
        .listen((data) => writeOutput('stdout', data))
        .asFuture<void>()
        .catchError((_) {});
    final stderrDone = process.stderr
        .transform(utf8.decoder)
        .listen((data) => writeOutput('stderr', data))
        .asFuture<void>()
        .catchError((_) {});

    unawaited(
      process.exitCode
          .then((exitCode) async {
            try {
              await Future.wait([
                stdoutDone,
                stderrDone,
              ]).timeout(const Duration(seconds: 2));
            } catch (_) {}
            final expectedExit = _expectedBackendExitPids.remove(process.pid);
            final exitLine =
                '\n[${DateTime.now().toIso8601String()}] Backend process exited with code $exitCode\n';
            _appendBackendOutputTail(exitLine);
            try {
              logFile.writeAsStringSync(
                exitLine,
                encoding: utf8,
                mode: FileMode.append,
                flush: true,
              );
            } catch (_) {}

            if (identical(_backendProcess, process)) {
              _backendProcess = null;
              _backendReady = false;
            }
            if (!expectedExit && exitCode != 0) {
              if (mounted) {
                setState(() => _loading = false);
              }
              CrashReporter.recordBackendCrash(
                exitCode: exitCode,
                logPath: logFile.path,
                outputTail: _backendOutputTail,
              );
            }
            CrashReporter.deleteBackendLogFile(logFile);
          })
          .catchError((_) {}),
    );
  }

  String _formatBackendOutput(String streamName, String data) {
    if (data.isEmpty) return '';
    final timestamp = DateTime.now().toIso8601String();
    final normalized = data.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
    final buffer = StringBuffer();
    final lines = normalized.split('\n');
    for (var index = 0; index < lines.length; index++) {
      final line = lines[index];
      if (line.isEmpty && index == lines.length - 1) continue;
      buffer.writeln('[$timestamp] [$streamName] $line');
    }
    return buffer.toString();
  }

  void _appendBackendOutputTail(String output) {
    if (output.isEmpty) return;
    final combined = '$_backendOutputTail$output';
    if (combined.length <= _backendOutputTailMaxCharacters) {
      _backendOutputTail = combined;
      return;
    }
    _backendOutputTail = combined.substring(
      combined.length - _backendOutputTailMaxCharacters,
    );
  }

  void _markBackendExitExpected(Process? process) {
    if (process == null) return;
    _expectedBackendExitPids.add(process.pid);
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
    _markBackendExitExpected(process);
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

  Future<void> _forceShutdownBackend() async {
    _backendReady = false;
    final process = _backendProcess;
    _markBackendExitExpected(process);
    if (process != null) {
      await _terminateBackendProcess(process);
      if (identical(_backendProcess, process)) {
        _backendProcess = null;
      }
    }
    await _terminateBackendPortOwner();
    await _waitForBackendStopped(const Duration(milliseconds: 1500));
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

  Future<void> _refreshVisibleResultsPage() async {
    await _refresh(silent: true);
    if (!mounted || (_selectedIndex != 1 && _selectedIndex != 2)) return;
    // Re-scan the input directory whenever a results page is opened. Reusing
    // the list cached for the same path would keep files that were deleted
    // outside Neri visible in both preview and validation.
    await _refreshPreviewItems(force: true);
  }

  void _updateInputDirectoryWatcher() {
    final inputPath = _inputController.text.trim();
    if (_watchedInputDirectory == inputPath && _inputDirectoryWatcher != null) {
      return;
    }

    _inputDirectoryChangeTimer?.cancel();
    _inputDirectoryChangeTimer = null;
    final previousWatcher = _inputDirectoryWatcher;
    _inputDirectoryWatcher = null;
    _watchedInputDirectory = null;
    if (previousWatcher != null) {
      unawaited(previousWatcher.cancel());
    }

    if (inputPath.isEmpty) return;
    final directory = Directory(inputPath);
    if (!directory.existsSync()) return;

    try {
      _inputDirectoryWatcher = directory
          .watch(
            events:
                FileSystemEvent.create |
                FileSystemEvent.delete |
                FileSystemEvent.move,
            recursive: true,
          )
          .listen(_handleInputDirectoryEvent, onError: (_) {});
      _watchedInputDirectory = inputPath;
    } on FileSystemException {
      // Re-scanning when a results page is opened remains the fallback for
      // file systems that do not support recursive directory watching.
    }
  }

  void _handleInputDirectoryEvent(FileSystemEvent event) {
    final destination = event is FileSystemMoveEvent ? event.destination : null;
    final affectsMedia =
        event.isDirectory ||
        _isSupportedMediaPath(event.path) ||
        (destination != null && _isSupportedMediaPath(destination));
    if (!affectsMedia) return;

    _inputDirectoryChangeTimer?.cancel();
    _inputDirectoryChangeTimer = Timer(const Duration(milliseconds: 350), () {
      if (!mounted || _inputController.text.trim() != _watchedInputDirectory) {
        return;
      }
      unawaited(_refreshPreviewItems(force: true));
    });
  }

  bool _isSupportedMediaPath(String path) {
    final normalizedPath = path.toLowerCase();
    return _supportedMediaExtensions.any(normalizedPath.endsWith);
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
      if (shouldClearPreviewItemsBeforeRefresh(
        loadedPath: _previewLoadedPath,
        inputPath: inputPath,
      )) {
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
        _previewContentVersion++;
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
      var fetchedCompleteJobResults = shouldFetchCompleteJobResults(
        includeJobResults: includeJobResults,
        silent: silent,
        jobProcessingBusy: _jobProcessingBusy,
        resultsPageVisible: _selectedIndex == 1 || _selectedIndex == 2,
      );
      var jobs = await widget.apiClient.listJobs(
        includeResults: fetchedCompleteJobResults,
      );
      final jobFinishedSinceLastRefresh = _hasJobFinishedSinceLastRefresh(jobs);
      if (!fetchedCompleteJobResults &&
          includeJobResults &&
          (jobFinishedSinceLastRefresh ||
              jobs.every((job) => !job.isWorkerActive))) {
        jobs = await widget.apiClient.listJobs();
        fetchedCompleteJobResults = true;
      } else if (!fetchedCompleteJobResults) {
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

      var previewJobUpdates = jobsChanged && fetchedCompleteJobResults
          ? _jobResultsForInputPath(jobs, _inputController.text.trim())
          : const <DetectionItem>[];
      if (previewJobUpdates.isNotEmpty) {
        previewJobUpdates = await existingLocalDetectionItems(
          previewJobUpdates,
        );
        if (!mounted || _closeFlowBlocksBackendStartup) return;
      }

      if (settingsChanged || jobsChanged) {
        setState(() {
          if (settingsChanged) {
            _settings = settings;
          }
          final modelPaths = settings.availableModels
              .map((model) => model.path)
              .toSet();
          if (_selectedModelPath == null ||
              (_selectedModelPath!.isNotEmpty &&
                  !modelPaths.contains(_selectedModelPath))) {
            _selectedModelPath = _savedModelSelection(
              settings,
              'selected_model',
              settings.selectedModel ?? '',
              modelPaths,
            );
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
            _selectedClassificationModelPath = _savedModelSelection(
              settings,
              'selected_classification_model',
              settings.selectedClassificationModel ?? '',
              classificationModelPaths,
            );
          }
          if (firstLoad) {
            _confidence = _doubleSetting(settings, 'confidence', _confidence);
            _iou = _doubleSetting(settings, 'iou', _iou);
            _imageSize = _intSetting(
              settings,
              'imgsz',
              _imageSize,
            ).clamp(320, 4096).toInt();
            _useFp16 =
                settings.gpuAvailable &&
                _boolSetting(settings, 'use_fp16', _useFp16);
            _videoMode = normalizeVideoProcessingMode(
              _stringSetting(settings, 'video_mode', _videoMode),
            );
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
      if (settingsChanged || jobsChanged) {
        unawaited(_updateWindowsShellStatus());
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
            elapsedSeconds: summary.elapsedSeconds,
            speed: summary.speed,
            remainingSeconds: summary.remainingSeconds,
            message: summary.message,
            results: cachedById[summary.id]!.results,
            error: summary.error,
            active: summary.active,
          )
        else
          summary,
    ];
  }

  bool _hasJobFinishedSinceLastRefresh(List<ProcessingJob> jobs) {
    final previousById = {for (final job in _jobs) job.id: job};
    return jobs.any((job) {
      final previous = previousById[job.id];
      return previous != null && previous.isWorkerActive && !job.isWorkerActive;
    });
  }

  Future<void> _saveAdvancedSettings(Map<String, dynamic> settings) async {
    if (mounted) setState(() => _settingsSaving = true);
    try {
      final saved = await widget.apiClient.saveSettings(settings);
      if (!mounted) return;
      setState(() {
        _settings = saved;
        _selectedModelPath = _savedModelSelection(
          saved,
          'selected_model',
          saved.selectedModel ?? '',
          saved.availableModels.map((model) => model.path).toSet(),
        );
        _selectedClassificationModelPath = _savedModelSelection(
          saved,
          'selected_classification_model',
          saved.selectedClassificationModel ?? '',
          saved.availableClassificationModels
              .map((model) => model.path)
              .toSet(),
        );
        _confidence = _doubleSetting(saved, 'confidence', _confidence);
        _iou = _doubleSetting(saved, 'iou', _iou);
        _imageSize = _intSetting(
          saved,
          'imgsz',
          _imageSize,
        ).clamp(320, 4096).toInt();
        _useFp16 =
            saved.gpuAvailable && _boolSetting(saved, 'use_fp16', _useFp16);
        _videoMode = normalizeVideoProcessingMode(
          _stringSetting(saved, 'video_mode', _videoMode),
        );
        _vidStride = _intSetting(
          saved,
          'vid_stride',
          _vidStride,
        ).clamp(1, 120).toInt();
      });
      _showSnackBar('设置已保存');
    } catch (error) {
      if (mounted) _showSnackBar('保存设置失败：$error');
      rethrow;
    } finally {
      if (mounted) setState(() => _settingsSaving = false);
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

  String _savedModelSelection(
    NeriSettings settings,
    String key,
    String fallback,
    Set<String> availablePaths,
  ) {
    final value = settings.settings[key];
    if (value is String) {
      final cleaned = value.trim();
      if (cleaned.isEmpty || availablePaths.contains(cleaned)) return cleaned;
    }
    return availablePaths.contains(fallback) ? fallback : '';
  }

  int _processingBatchSize({bool singleFile = false}) {
    if (singleFile) return 1;
    final settings = _settingsOrEmpty();
    final savedBatchSize = _intSetting(settings, 'batch_size', 16);
    final gpuAvailable = _settings?.gpuAvailable == true;
    return gpuAvailable ? savedBatchSize.clamp(1, 64).toInt() : 1;
  }

  int _processingThreadCount({bool singleFile = false}) {
    if (singleFile || _settings?.gpuAvailable != true) return 1;
    final settings = _settingsOrEmpty();
    return _intSetting(settings, 'thread_count', 4).clamp(1, 8).toInt();
  }

  int _imageSizeSetting() {
    return _intSetting(
      _settingsOrEmpty(),
      'imgsz',
      _imageSize,
    ).clamp(320, 4096).toInt();
  }

  bool _effectiveUseFp16() {
    return _settings?.gpuAvailable == true && _useFp16;
  }

  int _videoStride() {
    return _vidStride.clamp(1, 120).toInt();
  }

  String _effectiveVideoMode() {
    return normalizeVideoProcessingMode(_videoMode);
  }

  bool _useAugment() {
    return _boolSetting(_settingsOrEmpty(), 'use_augment', true);
  }

  bool _useAgnosticNms() {
    return _boolSetting(_settingsOrEmpty(), 'use_agnostic_nms', true);
  }

  String _confidencePriority() {
    return _stringSetting(
              _settingsOrEmpty(),
              'confidence_priority',
              'classification',
            ) ==
            'detection'
        ? 'detection'
        : 'classification';
  }

  String _packageSource() {
    return _stringSetting(_settingsOrEmpty(), 'package_source', 'auto');
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
    try {
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

      if (mounted) {
        setState(() => _settings = current.copyWith(settings: nextSettings));
      }
      final saved = await widget.apiClient.saveSettings(nextSettings);
      if (mounted) setState(() => _settings = saved);
    } catch (_) {
      // Quick-mark preferences should never make the mark action fail.
    }
  }

  Future<void> _revertQuickMarkUsage(String speciesName) async {
    try {
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
        final count = counts[name] ?? 0;
        if (count <= 1) {
          counts.remove(name);
        } else {
          counts[name] = count - 1;
        }
        for (var index = history.length - 1; index >= 0; index--) {
          if (history[index].trim() == name) {
            history.removeAt(index);
            break;
          }
        }
      }

      nextSettings['quick_mark_usage_counts'] = counts;
      nextSettings['quick_mark_recent_history'] = history;

      if (mounted) {
        setState(() => _settings = current.copyWith(settings: nextSettings));
      }
      final saved = await widget.apiClient.saveSettings(nextSettings);
      if (mounted) setState(() => _settings = saved);
    } catch (_) {
      // Undoing mark history must not make the undo action fail.
    }
  }

  Future<bool> _ensureDetectionModelSelected() async {
    final hasDetectionModel = _selectedModelPath?.trim().isNotEmpty == true;
    final hasClassificationModel =
        _selectedClassificationModelPath?.trim().isNotEmpty == true;
    if (hasDetectionModel || hasClassificationModel) return true;
    if (!mounted) return false;

    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('请选择检测模型'),
        content: const Text('探测模型和分类模型不能同时设为“不使用”。请至少选择一个模型后再开始检测。'),
        actions: [
          FilledButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('知道了'),
          ),
        ],
      ),
    );
    return false;
  }

  Future<void> _createJob() async {
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      _showSnackBar('请输入红外相机媒体文件夹路径');
      return;
    }
    if (!await _ensureDetectionModelSelected()) return;

    setState(() => _submitting = true);
    unawaited(_updateWindowsShellStatus());
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
        confidencePriority: _confidencePriority(),
        batchSize: _processingBatchSize(),
        threadCount: _processingThreadCount(),
        imageSize: _imageSizeSetting(),
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
      if (mounted) {
        setState(() => _submitting = false);
        unawaited(_updateWindowsShellStatus());
      }
    }
  }

  Future<void> _cancelJob(ProcessingJob job) async {
    if (mounted && job.id.isNotEmpty) {
      setState(() => _pendingStopJobIds.add(job.id));
      unawaited(_updateWindowsShellStatus());
    }
    try {
      await widget.apiClient.cancelJob(job.id);
      await _refresh(silent: true);
      _showSnackBar('已请求停止任务');
    } catch (error) {
      if (!mounted) return;
      setState(() => _pendingStopJobIds.remove(job.id));
      unawaited(_updateWindowsShellStatus());
      _showSnackBar('停止任务失败：$error');
    }
  }

  Future<void> _resumeJob(ProcessingJob job) async {
    if (mounted && job.id.isNotEmpty) {
      setState(() {
        _pendingStartJobBaselines[job.id] = job.processed;
      });
      unawaited(_updateWindowsShellStatus());
    }
    try {
      await widget.apiClient.resumeJob(job.id);
      await _refresh(silent: true);
      _showSnackBar('已继续任务');
    } catch (error) {
      if (!mounted) return;
      setState(() => _pendingStartJobBaselines.remove(job.id));
      unawaited(_updateWindowsShellStatus());
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
      'nju' => '南京大学源',
      _ => '官方源',
    };
    final installIntelDriver = await _resolveIntelDriverInstall(envChoice);
    if (installIntelDriver == null || !mounted) return true;

    final driverStep = installIntelDriver
        ? '安装时会先从 Intel 官网下载并运行显卡驱动安装程序，再使用$sourceLabel及对应硬件版本的 PyTorch wheel 源安装适用于“$envChoice”的 PyTorch，然后从$sourceLabel安装 ultralytics。'
        : '安装时会使用$sourceLabel及对应硬件版本的 PyTorch wheel 源安装适用于“$envChoice”的 PyTorch，然后从$sourceLabel安装 ultralytics。';
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('需要安装 YOLO 依赖'),
        content: Text(
          '当前缺少：$missingText。\n\n'
          '$driverStep'
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
        installIntelDriver: installIntelDriver,
      );
      if (!mounted) return true;
      _showSnackBar(response.message);
    } catch (installError) {
      if (!mounted) return true;
      _showSnackBar('启动 YOLO 依赖安装失败：$installError');
    }
    return true;
  }

  Future<bool?> _resolveIntelDriverInstall(String envChoice) async {
    try {
      final plan = await widget.apiClient.fetchPytorchInstallPlan(envChoice);
      if (!mounted) return null;
      if (!plan.needsIntelDriver) return false;

      final deviceText = plan.intelDeviceName.isEmpty
          ? 'Intel GPU'
          : plan.intelDeviceName;
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('需要安装 Intel 显卡驱动'),
          content: Text(
            '将安装 Intel XPU 版本 PyTorch，但当前检测到 $deviceText 尚未安装可用的 Intel 官方显卡驱动。\n\n'
            '是否先从 Intel 官网下载并运行显卡驱动安装程序？安装过程中可能出现 Windows 权限确认，完成后可能需要重启。',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('取消'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('下载并安装驱动'),
            ),
          ],
        ),
      );
      if (!mounted || confirmed != true) return null;
      return true;
    } catch (error) {
      if (mounted) _showSnackBar('检测 Intel 显卡驱动失败：$error');
      return null;
    }
  }

  Future<void> _detectCurrentPreviewImage(DetectionItem item) async {
    if (!await _ensureDetectionModelSelected()) return;
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
        confidencePriority: _confidencePriority(),
        batchSize: _processingBatchSize(singleFile: true),
        threadCount: _processingThreadCount(singleFile: true),
        imageSize: _imageSizeSetting(),
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
    if (!await _ensureDetectionModelSelected()) return;
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
        confidencePriority: _confidencePriority(),
        batchSize: _processingBatchSize(singleFile: inputPaths.length == 1),
        threadCount: _processingThreadCount(singleFile: inputPaths.length == 1),
        imageSize: _imageSizeSetting(),
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

  List<String> _normalizeFavoritePhotoPaths(List<String> paths) {
    final seen = <String>{};
    final normalized = <String>[];
    for (final path in paths) {
      final trimmed = path.trim();
      if (trimmed.isEmpty || !seen.add(trimmed)) continue;
      normalized.add(trimmed);
    }
    return normalized;
  }

  String _favoritePhotoExportMode(NeriSettings settings) {
    final value = _stringSetting(
      settings,
      'favorite_photo_export_mode',
      favoritePhotoExportAsk,
    );
    return switch (value) {
      favoritePhotoExportAlways => favoritePhotoExportAlways,
      favoritePhotoExportNever => favoritePhotoExportNever,
      _ => favoritePhotoExportAsk,
    };
  }

  String _emptyPhotoDeleteMode(NeriSettings settings) {
    final value = _stringSetting(
      settings,
      'empty_photo_delete_mode',
      emptyPhotoDeleteAsk,
    );
    return switch (value) {
      emptyPhotoDeleteAlways => emptyPhotoDeleteAlways,
      emptyPhotoDeleteNever => emptyPhotoDeleteNever,
      _ => emptyPhotoDeleteAsk,
    };
  }

  Future<void> _saveSilentSettingsPatch(Map<String, dynamic> patch) async {
    final current = _settings;
    if (current == null) {
      throw StateError('设置尚未加载，请稍后重试');
    }
    final nextSettings = Map<String, dynamic>.from(current.settings)
      ..addAll(patch);
    if (mounted) {
      setState(() => _settings = current.copyWith(settings: nextSettings));
    }
    final saved = await widget.apiClient.saveSettings(nextSettings);
    if (mounted) setState(() => _settings = saved);
  }

  Future<void> _updateFavoritePhotoPaths(List<String> paths) {
    return _saveSilentSettingsPatch({
      'favorite_photo_paths': _normalizeFavoritePhotoPaths(paths),
    });
  }

  Future<void> _updateFavoritePhotoExportMode(String mode) {
    final normalizedMode = switch (mode) {
      favoritePhotoExportAlways => favoritePhotoExportAlways,
      favoritePhotoExportNever => favoritePhotoExportNever,
      _ => favoritePhotoExportAsk,
    };
    return _saveSilentSettingsPatch({
      'favorite_photo_export_mode': normalizedMode,
    });
  }

  Future<void> _updateEmptyPhotoDeleteMode(String mode) {
    final normalizedMode = switch (mode) {
      emptyPhotoDeleteAlways => emptyPhotoDeleteAlways,
      emptyPhotoDeleteNever => emptyPhotoDeleteNever,
      _ => emptyPhotoDeleteAsk,
    };
    return _saveSilentSettingsPatch({
      'empty_photo_delete_mode': normalizedMode,
    });
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
    final versionNumber = _versionNumberPattern.firstMatch(trimmed)?.group(0);
    return versionNumber == null ? trimmed : 'Neri $versionNumber';
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
        _loading ||
        _previewDetecting ||
        _submitting ||
        _settingsSaving ||
        _validationBusy;

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
                    tooltip: '重启 Python 后端并刷新',
                    onPressed: _loading || _previewLoading
                        ? null
                        : _restartBackendAndRefresh,
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
                              unawaited(_refreshVisibleResultsPage());
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
        if (_startupMaintenanceStatus case final status?)
          _StartupMaintenanceOverlay(status: status),
        if (_softwareUpdateProgress case final progress?)
          if (!_softwareUpdateCardDismissed)
            _SoftwareUpdateProgressCard(
              progress: progress,
              onClose: _dismissSoftwareUpdateCard,
            ),
        if (_availableSoftwareUpdate case final availableUpdate?)
          if (!_softwareUpdateCardDismissed &&
              _softwareUpdateProgress == null &&
              _downloadedAppUpdate == null)
            _AvailableSoftwareUpdateCard(
              release: availableUpdate.release,
              onLater: _dismissAvailableSoftwareUpdate,
              onDownload: _startAvailableSoftwareUpdateDownload,
            ),
        if (_downloadedAppUpdate case final downloadedUpdate?)
          if (!_softwareUpdateCardDismissed && _softwareUpdateProgress == null)
            _DownloadedSoftwareUpdateCard(
              release: downloadedUpdate.release,
              onClose: _dismissSoftwareUpdateCard,
              onInstall: () => unawaited(_installDownloadedAppUpdate()),
            ),
        if (_privacyStatus?.isConsentComplete != true)
          PrivacyGateOverlay(
            status: _privacyStatus,
            loading: _privacyLoading,
            loadError: _privacyLoadError,
            onRetry: () => unawaited(_startBackendAndInitialRefresh()),
            onSave: (trainingEnabled) => widget.apiClient.savePrivacyStatus(
              trainingEnabled: trainingEnabled,
            ),
            onSaved: (status) => unawaited(_completePrivacyConsent(status)),
            onCloseApp: () => _requestCloseWindow(forceExit: true),
          ),
        if (_showClosingOverlay)
          _ClosingOverlay(
            phase: _closeDialogPhase,
            selectedAction: _selectedCloseAction,
            dontAskAgain: _closeDontAskAgain,
            selectionSaving: _closeSelectionSaving,
            onActionChanged: (action) =>
                unawaited(_updateLastCloseAction(action)),
            onDontAskAgainChanged: (value) =>
                setState(() => _closeDontAskAgain = value),
            onDismiss: _dismissCloseWindowPrompt,
            onCancel: _cancelCloseWindow,
            onConfirm: () => unawaited(_confirmCloseWindow()),
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

  void _updateStartModelSelection(
    String? value, {
    required bool classification,
  }) {
    final normalizedValue = value ?? '';
    setState(() {
      if (classification) {
        _selectedClassificationModelPath = normalizedValue;
      } else {
        _selectedModelPath = normalizedValue;
      }
      final current = _settings;
      if (current != null) {
        final nextSettings = Map<String, dynamic>.from(current.settings)
          ..['selected_model'] = _selectedModelPath ?? ''
          ..['selected_classification_model'] =
              _selectedClassificationModelPath ?? '';
        _settings = current.copyWith(
          settings: nextSettings,
          selectedModel: _selectedModelPath ?? '',
          selectedClassificationModel: _selectedClassificationModelPath ?? '',
        );
      }
      _modelSelectionRevision += 1;
    });
    _scheduleModelSelectionSave();
  }

  void _scheduleModelSelectionSave() {
    _modelSelectionSaveTimer?.cancel();
    _modelSelectionSaveTimer = Timer(
      const Duration(milliseconds: 300),
      () => unawaited(_persistModelSelections()),
    );
  }

  Future<void> _persistModelSelections() async {
    _modelSelectionSaveTimer?.cancel();
    _modelSelectionSaveTimer = null;
    if (_modelSelectionSaveInProgress) return;
    final current = _settings;
    if (current == null) return;

    final saveRevision = _modelSelectionRevision;
    final detectionModel = _selectedModelPath ?? '';
    final classificationModel = _selectedClassificationModelPath ?? '';
    final nextSettings = Map<String, dynamic>.from(current.settings)
      ..['selected_model'] = detectionModel
      ..['selected_classification_model'] = classificationModel;
    _modelSelectionSaveInProgress = true;
    try {
      final saved = await widget.apiClient.saveSettings(nextSettings);
      if (!mounted) return;
      setState(() {
        if (saveRevision == _modelSelectionRevision) {
          _settings = saved;
          return;
        }
        final latestSettings = Map<String, dynamic>.from(saved.settings)
          ..['selected_model'] = _selectedModelPath ?? ''
          ..['selected_classification_model'] =
              _selectedClassificationModelPath ?? '';
        _settings = saved.copyWith(
          settings: latestSettings,
          selectedModel: _selectedModelPath ?? '',
          selectedClassificationModel: _selectedClassificationModelPath ?? '',
        );
      });
    } catch (error) {
      if (mounted && saveRevision == _modelSelectionRevision) {
        _showSnackBar('保存模型选择失败：$error');
      }
    } finally {
      _modelSelectionSaveInProgress = false;
      if (mounted && saveRevision != _modelSelectionRevision) {
        _scheduleModelSelectionSave();
      }
    }
  }

  Widget _buildStartScreen() {
    return StartScreen(
      settings: _settings,
      inputController: _inputController,
      selectedModelPath: _selectedModelPath,
      onModelChanged: (value) =>
          _updateStartModelSelection(value, classification: false),
      selectedClassificationModelPath: _selectedClassificationModelPath,
      onClassificationModelChanged: (value) =>
          _updateStartModelSelection(value, classification: true),
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
      isActive: _selectedIndex == 3,
      settings: _settings,
      autoGroupInferredBurstSize: _autoGroupInferredBurstSize,
      apiClient: widget.apiClient,
      themeNotifier: widget.themeNotifier,
      onUpdateTheme: _updateTheme,
      closeBehavior: _closeBehavior,
      onCloseBehaviorChanged: (behavior) =>
          unawaited(_updateCloseBehaviorFromSettings(behavior)),
      onSaveSettings: _saveAdvancedSettings,
      onCheckForUpdates:
          ({required String channel, required String downloadSource}) =>
              _checkForSoftwareUpdate(
                manual: true,
                channel: channel,
                downloadSource: downloadSource,
              ),
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
      useCombinedConfidence:
          _selectedModelPath?.trim().isNotEmpty == true &&
          _selectedClassificationModelPath?.trim().isNotEmpty == true,
      showDetections: _previewShowDetections,
      onShowDetectionsChanged: (value) =>
          setState(() => _previewShowDetections = value),
      selectedSpeciesFilter: _previewSpeciesFilter,
      onSpeciesFilterChanged: (value) =>
          setState(() => _previewSpeciesFilter = value),
      confidenceThreshold: _previewConfidenceThreshold,
      onConfidenceThresholdChanged: (value) => setState(
        () => _previewConfidenceSettings[_previewSpeciesFilter] = value,
      ),
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
    final settings = _settingsOrEmpty();
    final quickMarkSpecies = _stringListSetting(
      settings,
      'quick_mark_list',
      const <String>[],
    );

    return SpeciesValidationScreen(
      apiClient: widget.apiClient,
      inputPath: inputPath,
      items: items,
      loading: _previewLoading,
      refreshVersion: _previewContentVersion,
      speciesTypes: _settings?.speciesTypes ?? const <String, String>{},
      useCombinedConfidence:
          _selectedModelPath?.trim().isNotEmpty == true &&
          _selectedClassificationModelPath?.trim().isNotEmpty == true,
      minFrameRatio: _doubleSetting(
        settings,
        'min_frame_ratio',
        0.0,
      ).clamp(0.0, 1.0).toDouble(),
      autoGroup: _boolSetting(settings, 'auto_group', true),
      collapseGroups: _boolSetting(settings, 'collapse_groups', true),
      autoGroupDetectBurst: _boolSetting(
        settings,
        'auto_group_detect_burst',
        true,
      ),
      autoGroupBurstSize: _intSetting(settings, 'auto_group_burst_size', 3),
      autoGroupGapSeconds: _intSetting(settings, 'auto_group_gap_seconds', 30),
      autoSortQuickMarks: _boolSetting(settings, 'auto_sort', false),
      undoSteps: _intSetting(
        settings,
        'undo_steps',
        200,
      ).clamp(10, 200).toInt(),
      quickMarkSpecies: quickMarkSpecies,
      quickMarkRecentHistory: _stringListSetting(
        settings,
        'quick_mark_recent_history',
        const <String>[],
      ),
      quickMarkUsageCounts: _intMapSetting(settings, 'quick_mark_usage_counts'),
      quantityButtons: _stringListSetting(
        settings,
        'quantity_buttons',
        validationQuantityButtons,
      ),
      exportColumns: _stringListSetting(
        settings,
        'export_columns',
        validationExportColumns,
      ),
      favoritePhotoPaths: _stringListSetting(
        settings,
        'favorite_photo_paths',
        const <String>[],
      ),
      favoritePhotoExportMode: _favoritePhotoExportMode(settings),
      emptyPhotoDeleteMode: _emptyPhotoDeleteMode(settings),
      onRefresh: () => _refreshPreviewItems(force: true),
      onLoadMetadata: _loadPreviewMetadata,
      onOpenExternal: _openFileWithSystem,
      onMarkItem: _markValidationItem,
      onMarkItems: _markValidationItems,
      onQuickMarkUsed: _recordQuickMarkUsage,
      onQuickMarkReverted: _revertQuickMarkUsage,
      onRedetectItems: _redetectValidationItems,
      onFavoritePhotoPathsChanged: _updateFavoritePhotoPaths,
      onFavoritePhotoExportModeChanged: _updateFavoritePhotoExportMode,
      onEmptyPhotoDeleteModeChanged: _updateEmptyPhotoDeleteMode,
      onAutoGroupInferredBurstSizeChanged:
          _handleAutoGroupInferredBurstSizeChanged,
    );
  }

  void _handleAutoGroupInferredBurstSizeChanged(int? value) {
    if (_autoGroupInferredBurstSize == value) return;
    if (!mounted) return;
    setState(() => _autoGroupInferredBurstSize = value);
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

class _StartupMaintenanceOverlay extends StatelessWidget {
  const _StartupMaintenanceOverlay({required this.status});

  final MaintenanceStatus status;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Positioned(
      left: 0,
      right: 0,
      top: kToolbarHeight + 4,
      bottom: 0,
      child: ColoredBox(
        color: scheme.scrim.withValues(alpha: 0.36),
        child: Center(
          child: Card(
            elevation: 6,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 460),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox.square(
                      dimension: 24,
                      child: CircularProgressIndicator(
                        value: status.progress / 100,
                        strokeWidth: 2.8,
                        color: scheme.primary,
                        backgroundColor: scheme.surfaceContainerHighest,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '环境维护中',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            status.message.isEmpty
                                ? '正在继续之前的依赖安装，请稍候。'
                                : status.message,
                            style: Theme.of(context).textTheme.bodyMedium
                                ?.copyWith(color: scheme.onSurfaceVariant),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _AvailableSoftwareUpdate {
  const _AvailableSoftwareUpdate({
    required this.release,
    required this.installDirectory,
  });

  final AppUpdateRelease release;
  final Directory installDirectory;
}

class _SoftwareUpdateProgress {
  const _SoftwareUpdateProgress({
    required this.message,
    required this.detail,
    this.fraction,
    this.isDownloading = false,
  });

  final String message;
  final String detail;
  final double? fraction;
  final bool isDownloading;
}

class _SoftwareUpdateProgressCard extends StatelessWidget {
  const _SoftwareUpdateProgressCard({
    required this.progress,
    required this.onClose,
  });

  final _SoftwareUpdateProgress progress;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Positioned(
      right: 20,
      bottom: 20,
      child: Card(
        elevation: 6,
        child: SizedBox(
          width: 360,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(18, 16, 14, 14),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.download_rounded, color: scheme.primary),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        progress.isDownloading ? '下载更新' : '软件更新',
                        style: theme.textTheme.titleSmall,
                      ),
                    ),
                    IconButton(
                      onPressed: onClose,
                      tooltip: '关闭进度卡',
                      icon: const Icon(Icons.close_rounded, size: 19),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(value: progress.fraction),
                const SizedBox(height: 10),
                Text(
                  progress.message,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 3),
                Text(
                  progress.detail,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AvailableSoftwareUpdateCard extends StatelessWidget {
  const _AvailableSoftwareUpdateCard({
    required this.release,
    required this.onLater,
    required this.onDownload,
  });

  final AppUpdateRelease release;
  final VoidCallback onLater;
  final VoidCallback onDownload;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final notes = release.notes.trim();
    return Positioned(
      right: 20,
      bottom: 20,
      child: Card(
        elevation: 7,
        child: SizedBox(
          width: 400,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 16, 14),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.system_update_alt_rounded,
                      color: scheme.primary,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '发现新版本 ${release.tag}',
                            style: theme.textTheme.titleSmall,
                          ),
                          const SizedBox(height: 3),
                          Text(
                            release.prerelease ? '预览版更新' : '稳定版更新',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: scheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      onPressed: onLater,
                      tooltip: '关闭',
                      icon: const Icon(Icons.close_rounded, size: 19),
                    ),
                  ],
                ),
                if (notes.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(
                    notes,
                    maxLines: 5,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                ],
                const SizedBox(height: 14),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton(onPressed: onLater, child: const Text('稍后')),
                    const SizedBox(width: 8),
                    FilledButton.icon(
                      onPressed: onDownload,
                      icon: const Icon(Icons.download_rounded),
                      label: const Text('下载更新'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _DownloadedSoftwareUpdateCard extends StatelessWidget {
  const _DownloadedSoftwareUpdateCard({
    required this.release,
    required this.onClose,
    required this.onInstall,
  });

  final AppUpdateRelease release;
  final VoidCallback onClose;
  final VoidCallback onInstall;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Positioned(
      right: 20,
      bottom: 20,
      child: Card(
        elevation: 7,
        child: SizedBox(
          width: 400,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 16, 14),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.download_done_rounded, color: scheme.primary),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '更新已下载 ${release.tag}',
                            style: theme.textTheme.titleSmall,
                          ),
                          const SizedBox(height: 3),
                          Text(
                            release.displayName,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: scheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      onPressed: onClose,
                      tooltip: '关闭',
                      icon: const Icon(Icons.close_rounded, size: 19),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  '更新包已下载完成，可以稍后安装或立即重启 Neri 完成更新。',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 14),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton(onPressed: onClose, child: const Text('稍后')),
                    const SizedBox(width: 8),
                    FilledButton.icon(
                      onPressed: onInstall,
                      icon: const Icon(Icons.restart_alt_rounded),
                      label: const Text('重启并更新'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ClosingOverlay extends StatelessWidget {
  const _ClosingOverlay({
    required this.phase,
    required this.selectedAction,
    required this.dontAskAgain,
    required this.selectionSaving,
    required this.onActionChanged,
    required this.onDontAskAgainChanged,
    required this.onDismiss,
    required this.onCancel,
    required this.onConfirm,
  });

  final _CloseDialogPhase phase;
  final _CloseAction selectedAction;
  final bool dontAskAgain;
  final bool selectionSaving;
  final ValueChanged<_CloseAction> onActionChanged;
  final ValueChanged<bool> onDontAskAgainChanged;
  final VoidCallback onDismiss;
  final VoidCallback onCancel;
  final VoidCallback onConfirm;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final title = switch (phase) {
      _CloseDialogPhase.choosing => '关闭主窗口？',
      _CloseDialogPhase.shuttingDown => '正在退出主程序',
      _CloseDialogPhase.restoring => '正在恢复',
    };
    final message = switch (phase) {
      _CloseDialogPhase.choosing => '请选择关闭主窗口后的操作。',
      _CloseDialogPhase.shuttingDown => '正在关闭 Python 后端，关闭完成后将自动退出主程序。',
      _CloseDialogPhase.restoring => '正在重新启动 Python 后端，请稍候。',
    };

    return Positioned.fill(
      child: Stack(
        children: [
          ModalBarrier(
            dismissible:
                phase == _CloseDialogPhase.choosing && !selectionSaving,
            onDismiss: phase == _CloseDialogPhase.choosing && !selectionSaving
                ? onDismiss
                : null,
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
                          if (phase != _CloseDialogPhase.choosing) ...[
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
                          if (phase == _CloseDialogPhase.choosing)
                            IconButton(
                              tooltip: '关闭',
                              onPressed: selectionSaving ? null : onDismiss,
                              visualDensity: VisualDensity.compact,
                              icon: const Icon(Icons.close_rounded, size: 20),
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
                      if (phase == _CloseDialogPhase.choosing) ...[
                        const SizedBox(height: 12),
                        RadioGroup<_CloseAction>(
                          groupValue: selectedAction,
                          onChanged: (value) {
                            if (!selectionSaving && value != null) {
                              onActionChanged(value);
                            }
                          },
                          child: Column(
                            children: [
                              RadioListTile<_CloseAction>(
                                contentPadding: EdgeInsets.zero,
                                value: _CloseAction.hideToTray,
                                enabled: !selectionSaving,
                                title: const Text('隐藏到任务托盘'),
                                subtitle: const Text('主程序和 Python 后端继续在后台运行'),
                              ),
                              RadioListTile<_CloseAction>(
                                contentPadding: EdgeInsets.zero,
                                value: _CloseAction.exit,
                                enabled: !selectionSaving,
                                title: const Text('退出主程序'),
                                subtitle: const Text('关闭 Python 后端并退出 Neri'),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 8),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Expanded(
                              child: CheckboxListTile(
                                dense: true,
                                contentPadding: EdgeInsets.zero,
                                controlAffinity:
                                    ListTileControlAffinity.leading,
                                value: dontAskAgain,
                                onChanged: selectionSaving
                                    ? null
                                    : (value) =>
                                          onDontAskAgainChanged(value ?? false),
                                title: const Text('不再提醒'),
                                subtitle: const Text('以后自动执行本次选择'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            FilledButton(
                              onPressed: selectionSaving ? null : onConfirm,
                              child: selectionSaving
                                  ? const SizedBox.square(
                                      dimension: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Text('确认'),
                            ),
                          ],
                        ),
                      ] else if (phase == _CloseDialogPhase.shuttingDown) ...[
                        const SizedBox(height: 18),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            TextButton(
                              onPressed: onCancel,
                              child: const Text('取消'),
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
