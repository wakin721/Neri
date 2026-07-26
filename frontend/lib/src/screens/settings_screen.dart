import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api_client.dart';
import '../app_updater.dart';
import '../crash_reporter.dart';
import '../crash_watchdog.dart';
import '../local_debug_logs.dart';
import '../local_maintenance_status.dart';
import '../models/close_behavior.dart';
import '../models/settings.dart';
import '../models/theme_settings.dart';
import '../utils/quick_mark_sort.dart';
import '../widgets/app_menu_style.dart';
import '../widgets/section_card.dart';
import '../widgets/workspace_split_metrics.dart';

const _defaultExportColumns = <String>[
  '文件名',
  '格式',
  '拍摄日期',
  '拍摄时间',
  '工作天数',
  '物种名称',
  '学名',
  '目名',
  '目拉丁名',
  '科名',
  '科拉丁名',
  '属名',
  '属拉丁名',
  '物种类型',
  '物种数量',
  '最低置信度',
  '独立探测首只',
  '备注',
];

const _defaultQuickMarks = <String>[
  '灰雁',
  '大白鹭',
  '绿头鸭',
  '赤嘴潜鸭',
  '赤麻鸭',
  '青头潜鸭',
  '马',
  '黑翅长脚鹬',
  '未知鸟',
];

const _defaultQuantityButtons = <String>[
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  '10',
  '25',
  '50',
];

const _releaseNotesUrl =
    'https://github.com/wakin721/Neri/blob/main/res/demo/README_Update.md';
const _officialWebsiteUrl = 'https://myneri.top/';
const _feedbackUrl = 'https://github.com/wakin721/Neri/issues';
const _sourceCodeUrl = 'https://github.com/wakin721/Neri';
const _frontendVersion = String.fromEnvironment(
  'NERI_FRONTEND_VERSION',
  defaultValue: '3.0.5-beta6(4cfd72)',
);
const _debugModeKey = 'debug_mode';
const _debugTapThreshold = 5;
const _debugTapResetDuration = Duration(seconds: 3);
const _autoSaveDelay = Duration(milliseconds: 800);
const _xpuEnabled = false;
const _favoritePhotoExportAsk = 'ask';
const _favoritePhotoExportAlways = 'export';
const _favoritePhotoExportNever = 'skip';
const _detectionConfidencePriority = 'detection';
const _classificationConfidencePriority = 'classification';
const _updateMirrorsKey = 'update_mirrors';
const _legacyUpdateMirrorUrlKey = 'update_mirror_url';
const _legacyUpdateCustomMirrorsKey = 'update_custom_mirrors';

typedef SoftwareUpdateCheckCallback =
    Future<void> Function({
      required String channel,
      required String mirror,
      required List<String> mirrorTemplates,
    });

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    required this.settings,
    required this.autoGroupInferredBurstSize,
    required this.apiClient,
    required this.themeNotifier,
    required this.onUpdateTheme,
    required this.closeBehavior,
    required this.onCloseBehaviorChanged,
    required this.onSaveSettings,
    required this.onCheckForUpdates,
    required this.onShowMessage,
    super.key,
  });

  final NeriSettings? settings;
  final int? autoGroupInferredBurstSize;
  final NeriApiClient apiClient;
  final ValueNotifier<ThemeSettings> themeNotifier;
  final ValueChanged<ThemeSettings> onUpdateTheme;
  final String closeBehavior;
  final ValueChanged<String> onCloseBehaviorChanged;
  final Future<void> Function(Map<String, dynamic> settings) onSaveSettings;
  final SoftwareUpdateCheckCallback onCheckForUpdates;
  final ValueChanged<String> onShowMessage;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _packageController = TextEditingController();
  final _maintenanceStatusStore = LocalMaintenanceStatusStore();
  Map<String, dynamic> _draft = <String, dynamic>{};
  int _sectionIndex = 0;
  bool _saving = false;
  bool _dirty = false;
  bool _resettingDraft = false;
  bool _loadingModelClasses = false;
  String? _maintenancePreparationOperation;
  bool _installingPytorch = false;
  bool _reinstallingPackage = false;
  bool _debugModeSaving = false;
  bool _clearingCache = false;
  bool _checkingForUpdates = false;
  int _draftRevision = 0;
  Timer? _autoSaveTimer;
  Timer? _maintenanceTimer;
  String? _maintenanceOperation;
  String? _maintenanceMessage;
  double? _maintenanceProgress;
  String? _maintenanceStatusPath;
  String? _modelClassesPath;
  List<ModelClassInfo> _modelClassOptions = const <ModelClassInfo>[];

  @override
  void initState() {
    super.initState();
    _packageController.addListener(_handlePackageChanged);
    _resetDraft();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _loadModelClassesForSelection();
      unawaited(_resumeMaintenanceWatchIfActive());
    });
  }

  @override
  void didUpdateWidget(covariant SettingsScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.settings != widget.settings) {
      if (!_dirty && !_saving) {
        _resetDraft();
        _loadModelClassesForSelection();
      } else {
        _syncQuickMarkDraftFromSettings();
      }
    }
  }

  @override
  void dispose() {
    _autoSaveTimer?.cancel();
    _maintenanceTimer?.cancel();
    _packageController.removeListener(_handlePackageChanged);
    _packageController.dispose();
    super.dispose();
  }

  void _resetDraft() {
    final settings = widget.settings;
    final saved = Map<String, dynamic>.from(
      settings?.settings ?? const <String, dynamic>{},
    );
    final savedQuickMarks = _stringListSetting(
      saved,
      'quick_mark_list',
      _defaultQuickMarks,
    );
    final dependenciesReady =
        settings == null || settings.missingYoloDependencies.isEmpty;
    final updateMirrors = saved.containsKey(_updateMirrorsKey)
        ? _normalizedUpdateMirrors(
            _stringListSetting(saved, _updateMirrorsKey, const <String>[]),
          )
        : _normalizedUpdateMirrors(<String>[
            _normalizedUpdateMirror(
              _stringSetting(
                saved,
                _legacyUpdateMirrorUrlKey,
                defaultGithubMirrorTemplate,
              ),
            ),
            defaultGithubMirrorTemplate,
            ..._stringListSetting(
              saved,
              _legacyUpdateCustomMirrorsKey,
              const <String>[],
            ),
          ]);
    _draft = <String, dynamic>{
      ...saved,
      'selected_model': _stringSetting(
        saved,
        'selected_model',
        settings?.selectedModel ?? '',
      ),
      'selected_classification_model': _stringSetting(
        saved,
        'selected_classification_model',
        settings?.selectedClassificationModel ?? '',
      ),
      'package_source': _stringSetting(saved, 'package_source', 'auto'),
      'pytorch_version': _normalizedPytorchVersion(
        _stringSetting(saved, 'pytorch_version', '自动检测'),
      ),
      'confidence': _doubleSetting(saved, 'confidence', 0.25),
      'iou': _doubleSetting(saved, 'iou', 0.30),
      'imgsz': _intSetting(saved, 'imgsz', 1920),
      'batch_size': _intSetting(saved, 'batch_size', 16),
      'thread_count': _intSetting(saved, 'thread_count', 4),
      'use_fp16': !dependenciesReady
          ? _boolSetting(saved, 'use_fp16', false)
          : settings?.gpuAvailable == true &&
                _boolSetting(saved, 'use_fp16', settings?.gpuAvailable == true),
      'use_augment': _boolSetting(saved, 'use_augment', false),
      'use_agnostic_nms': _boolSetting(saved, 'use_agnostic_nms', true),
      'confidence_priority':
          _stringSetting(
                saved,
                'confidence_priority',
                _classificationConfidencePriority,
              ) ==
              _detectionConfidencePriority
          ? _detectionConfidencePriority
          : _classificationConfidencePriority,
      'video_mode': _stringSetting(saved, 'video_mode', 'all'),
      'vid_stride': _intSetting(saved, 'vid_stride', 1),
      'min_frame_ratio': _doubleSetting(saved, 'min_frame_ratio', 0.0),
      'auto_group': _boolSetting(saved, 'auto_group', true),
      'collapse_groups': _boolSetting(saved, 'collapse_groups', true),
      'auto_group_detect_burst': _boolSetting(
        saved,
        'auto_group_detect_burst',
        true,
      ),
      'auto_group_burst_size': _intSetting(saved, 'auto_group_burst_size', 3),
      'auto_group_gap_seconds': _intSetting(
        saved,
        'auto_group_gap_seconds',
        30,
      ),
      'undo_steps': _undoStepsSetting(saved),
      'auto_sort': _boolSetting(saved, 'auto_sort', false),
      'quick_mark_list': savedQuickMarks,
      'quantity_buttons': _stringListSetting(
        saved,
        'quantity_buttons',
        _defaultQuantityButtons,
      ),
      'export_columns': _stringListSetting(
        saved,
        'export_columns',
        _defaultExportColumns,
      ),
      'favorite_photo_paths': _stringListSetting(
        saved,
        'favorite_photo_paths',
        const <String>[],
      ),
      'favorite_photo_export_mode': _favoritePhotoExportModeSetting(saved),
      'selected_species_names': _stringListSetting(
        saved,
        'selected_species_names',
        const <String>[],
      ),
      'update_channel': _stringSetting(saved, 'update_channel', 'Preview'),
      'update_mirror': _stringSetting(saved, 'update_mirror', 'KKGitHub'),
      _updateMirrorsKey: updateMirrors,
      _debugModeKey: _boolSetting(saved, _debugModeKey, false),
    };
    _resettingDraft = true;
    _packageController.text = _stringSetting(saved, 'package', '');
    _resettingDraft = false;
    _dirty = false;
    _autoSaveTimer?.cancel();
  }

  void _handlePackageChanged() {
    if (_resettingDraft) return;
    _draft['package'] = _packageController.text.trim();
    _markDirty();
  }

  void _syncQuickMarkDraftFromSettings() {
    final settings = widget.settings;
    if (settings == null) return;
    final saved = settings.settings;
    _draft['quick_mark_list'] = _stringListSetting(
      saved,
      'quick_mark_list',
      _defaultQuickMarks,
    );
    _draft['quantity_buttons'] = _stringListSetting(
      saved,
      'quantity_buttons',
      _defaultQuantityButtons,
    );
    _draft['quick_mark_recent_history'] = _stringListSetting(
      saved,
      'quick_mark_recent_history',
      const <String>[],
    );
    _draft['quick_mark_usage_counts'] = _intMapSetting(
      saved,
      'quick_mark_usage_counts',
    );
    if (saved.containsKey('auto_sort')) {
      _draft['auto_sort'] = _boolSetting(
        saved,
        'auto_sort',
        _bool('auto_sort'),
      );
    }
  }

  List<String> get _missingYoloDependencies =>
      widget.settings?.missingYoloDependencies ?? const <String>[];

  bool get _detectionDependenciesReady => _missingYoloDependencies.isEmpty;

  String get _missingYoloDependenciesLabel {
    final missing = _missingYoloDependencies;
    if (missing.isEmpty) return '';
    return missing.join('、');
  }

  void _markDirty() {
    if (!mounted) return;
    setState(_markDraftChanged);
    _scheduleAutoSave();
  }

  void _markDraftChanged() {
    _dirty = true;
    _draftRevision += 1;
  }

  void _scheduleAutoSave() {
    _autoSaveTimer?.cancel();
    if (!_dirty || !mounted) return;
    _autoSaveTimer = Timer(_autoSaveDelay, () => unawaited(_save()));
  }

  Future<void> _loadModelClassesForSelection() async {
    if (!_detectionDependenciesReady) {
      if (!mounted) return;
      setState(() {
        _modelClassesPath = null;
        _modelClassOptions = const <ModelClassInfo>[];
        _loadingModelClasses = false;
      });
      return;
    }

    final path = _modelClassesSelectionPath();
    if (path == null || path.isEmpty) {
      if (!mounted) return;
      setState(() {
        _modelClassesPath = null;
        _modelClassOptions = const <ModelClassInfo>[];
        _loadingModelClasses = false;
      });
      return;
    }
    if (_modelClassesPath == path && _modelClassOptions.isNotEmpty) return;

    setState(() {
      _modelClassesPath = path;
      _loadingModelClasses = true;
    });

    try {
      final classes = await widget.apiClient.fetchModelClasses(path);
      if (!mounted || _modelClassesSelectionPath() != path) return;
      setState(() {
        _modelClassOptions = classes;
        _loadingModelClasses = false;
      });
    } catch (error) {
      if (!mounted || _modelClassesSelectionPath() != path) return;
      setState(() {
        _modelClassOptions = const <ModelClassInfo>[];
        _loadingModelClasses = false;
      });
      widget.onShowMessage('读取模型物种列表失败：$error');
    }
  }

  Future<bool?> _resolveIntelDriverInstall(String envChoice) async {
    try {
      final plan = await widget.apiClient.fetchPytorchInstallPlan(envChoice);
      if (!mounted) return null;
      if (!plan.needsIntelDriver) return false;

      final deviceText = plan.intelDeviceName.isEmpty
          ? 'Intel GPU'
          : plan.intelDeviceName;
      final confirmed = await _confirmMaintenance(
        title: '需要安装 Intel 显卡驱动',
        message:
            '将安装 Intel XPU 版本 PyTorch，但当前检测到 $deviceText 尚未安装可用的 Intel 官方显卡驱动。\n\n是否先从 Intel 官网下载并运行显卡驱动安装程序？安装过程中可能出现 Windows 权限确认，完成后可能需要重启。',
        confirmLabel: '下载并安装驱动',
      );
      if (!mounted || confirmed != true) return null;
      return true;
    } catch (error) {
      widget.onShowMessage('检测 Intel 显卡驱动失败：$error');
      return null;
    }
  }

  Future<void> _installPytorch() async {
    if (_maintenanceBusy) return;
    _startMaintenancePreparation(
      operation: 'install_pytorch',
      message: '正在检查环境维护状态...',
    );

    final envChoice = _normalizedPytorchVersion(
      _string('pytorch_version', '自动检测'),
    );
    late final PackageSourceResolution? packageSource;
    late final bool? installIntelDriver;
    try {
      if (await _resumeMaintenanceWatchIfActive(announce: true)) return;
      _updateMaintenancePreparation('正在检测 Python 包安装源和运行环境...');
      final checks = await Future.wait<Object?>([
        _resolvePackageSource(),
        _resolveIntelDriverInstall(envChoice),
      ]);
      packageSource = checks[0] as PackageSourceResolution?;
      installIntelDriver = checks[1] as bool?;
    } finally {
      _finishMaintenancePreparation();
    }

    if (packageSource == null || !mounted) return;
    if (installIntelDriver == null || !mounted) return;

    final installStep = installIntelDriver
        ? '将先从 Intel 官网下载并运行显卡驱动安装程序，再使用${packageSource.label}重新安装适用于$envChoice的 PyTorch。'
        : '将使用${packageSource.label}重新安装适用于$envChoice的 PyTorch。';
    final confirmed = await _confirmPythonInstallation(
      title: '安装 PyTorch',
      message: installStep,
    );
    if (confirmed != true) return;

    final installUltralytics = _missingYoloDependencies.contains('ultralytics')
        ? await _confirmInstallUltralyticsWithPytorch(packageSource.label)
        : false;
    if (installUltralytics == null || !mounted) return;

    setState(() {
      _installingPytorch = true;
      _maintenanceOperation = installUltralytics
          ? 'install_yolo_dependencies'
          : 'install_pytorch';
      _maintenanceMessage = installUltralytics
          ? '正在启动 PyTorch 和 ultralytics 安装...'
          : '正在启动 PyTorch 安装...';
    });
    try {
      final response = installUltralytics
          ? await widget.apiClient.installYoloDependencies(
              envChoice: envChoice,
              packageSource: packageSource.source,
              installIntelDriver: installIntelDriver,
            )
          : await widget.apiClient.installPytorch(
              envChoice,
              packageSource: packageSource.source,
              installIntelDriver: installIntelDriver,
            );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? (installUltralytics
                  ? 'install_yolo_dependencies'
                  : 'install_pytorch')
            : response.operation,
        message: response.message,
        progress: response.progress,
        statusPath: response.statusPath,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      await _handleMaintenanceStartFailure(error, '启动 PyTorch 安装失败');
    }
  }

  Future<bool?> _confirmInstallUltralyticsWithPytorch(String sourceLabel) {
    return showDialog<bool?>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('同时安装 ultralytics？'),
        content: Text(
          '当前未安装 ultralytics。它是 YOLO 检测需要的运行库。\n\n是否在安装 PyTorch 后继续从$sourceLabel安装 ultralytics？',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(null),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('仅安装 PyTorch'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('同时安装'),
          ),
        ],
      ),
    );
  }

  Future<void> _installYoloDependencies() async {
    if (_maintenanceBusy) return;
    _startMaintenancePreparation(
      operation: 'install_yolo_dependencies',
      message: '正在检查环境维护状态...',
    );

    const envChoice = '自动检测';
    late final PackageSourceResolution? packageSource;
    late final bool? installIntelDriver;
    try {
      if (await _resumeMaintenanceWatchIfActive(announce: true)) return;
      _updateMaintenancePreparation('正在检测 Python 包安装源和运行环境...');
      final checks = await Future.wait<Object?>([
        _resolvePackageSource(),
        _resolveIntelDriverInstall(envChoice),
      ]);
      packageSource = checks[0] as PackageSourceResolution?;
      installIntelDriver = checks[1] as bool?;
    } finally {
      _finishMaintenancePreparation();
    }

    if (packageSource == null || !mounted) return;
    if (installIntelDriver == null || !mounted) return;

    final driverStep = installIntelDriver
        ? '安装时会先从 Intel 官网下载并运行显卡驱动安装程序，再使用${packageSource.label}及对应硬件版本的 PyTorch wheel 源自动安装适用的 PyTorch，然后从${packageSource.label}安装 ultralytics。'
        : '安装时会使用${packageSource.label}及对应硬件版本的 PyTorch wheel 源自动安装适用的 PyTorch，然后从${packageSource.label}安装 ultralytics。';
    final missingText = _missingYoloDependenciesLabel.isEmpty
        ? 'YOLO 处理依赖'
        : _missingYoloDependenciesLabel;
    final confirmed = await _confirmPythonInstallation(
      title: '安装 YOLO 检测库',
      message: '当前缺少：$missingText。\n\n$driverStep',
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _installingPytorch = true;
      _maintenanceOperation = 'install_yolo_dependencies';
      _maintenanceMessage = '正在启动 YOLO 依赖安装...';
    });
    try {
      final response = await widget.apiClient.installYoloDependencies(
        envChoice: envChoice,
        packageSource: packageSource.source,
        installIntelDriver: installIntelDriver,
      );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'install_yolo_dependencies'
            : response.operation,
        message: response.message,
        progress: response.progress,
        statusPath: response.statusPath,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      await _handleMaintenanceStartFailure(error, '启动 YOLO 依赖安装失败');
    }
  }

  Future<PackageSourceResolution?> _resolvePackageSource() async {
    try {
      return await widget.apiClient.resolvePackageSource(
        _string('package_source', 'auto'),
      );
    } catch (error) {
      widget.onShowMessage('检测 Python 包安装源失败：$error');
      return null;
    }
  }

  String _normalizedPytorchVersion(String value) {
    if (!_xpuEnabled && value.toUpperCase().contains('XPU')) {
      return '自动检测';
    }
    return value;
  }

  void _startMaintenancePreparation({
    required String operation,
    required String message,
  }) {
    setState(() {
      _maintenancePreparationOperation = operation;
      _maintenanceMessage = message;
      _maintenanceProgress = null;
    });
  }

  void _updateMaintenancePreparation(String message) {
    if (!mounted || _maintenancePreparationOperation == null) return;
    setState(() => _maintenanceMessage = message);
  }

  void _finishMaintenancePreparation() {
    if (!mounted || _maintenancePreparationOperation == null) return;
    setState(() {
      _maintenancePreparationOperation = null;
      if (!_maintenanceInProgress) {
        _maintenanceMessage = null;
        _maintenanceProgress = null;
      }
    });
  }

  Future<void> _reinstallPythonPackage() async {
    if (_maintenanceBusy) return;
    final packageSpec = _packageController.text.trim();
    if (packageSpec.isEmpty) {
      widget.onShowMessage('请输入要重新安装的 Python 包名。');
      return;
    }

    _startMaintenancePreparation(
      operation: 'reinstall_package',
      message: '正在检查环境维护状态...',
    );
    PackageSourceResolution? packageSource;
    try {
      if (await _resumeMaintenanceWatchIfActive(announce: true)) return;
      _updateMaintenancePreparation('正在检测 Python 包安装源...');
      packageSource = await _resolvePackageSource();
    } finally {
      _finishMaintenancePreparation();
    }

    if (packageSource == null || !mounted) return;
    final confirmed = await _confirmPythonInstallation(
      title: '安装 Python 库',
      message:
          '将使用${packageSource.label}通过 toolkit\\python.exe 强制重新安装 $packageSpec。',
    );
    if (confirmed != true) return;

    setState(() {
      _reinstallingPackage = true;
      _maintenanceOperation = 'reinstall_package';
      _maintenanceMessage = '正在启动 Python 包重新安装...';
    });
    try {
      final response = await widget.apiClient.reinstallPythonPackage(
        packageSpec,
        packageSource.source,
      );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'reinstall_package'
            : response.operation,
        message: response.message,
        progress: response.progress,
        statusPath: response.statusPath,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      await _handleMaintenanceStartFailure(error, '启动 Python 包重新安装失败');
    }
  }

  Future<void> _handleMaintenanceStartFailure(
    Object error,
    String prefix,
  ) async {
    if (await _resumeMaintenanceWatchIfActive(announce: true)) return;
    if (!mounted) return;
    setState(() {
      _installingPytorch = false;
      _reinstallingPackage = false;
      _maintenanceOperation = null;
      _maintenanceMessage = null;
      _maintenanceProgress = null;
      _maintenanceStatusPath = null;
    });
    widget.onShowMessage('$prefix：$error');
  }

  Future<bool> _resumeMaintenanceWatchIfActive({bool announce = false}) async {
    MaintenanceStatus? status;
    try {
      status = await widget.apiClient.fetchMaintenanceStatus().timeout(
        const Duration(seconds: 1),
      );
    } catch (_) {
      status = await _maintenanceStatusStore.tryRead(
        path: _maintenanceStatusPath,
      );
    }
    if (!mounted ||
        status == null ||
        !_maintenanceStatusStore.isActive(status)) {
      return false;
    }

    final operation = status.operation ?? '';
    if (!const {
      'install_pytorch',
      'install_yolo_dependencies',
      'reinstall_package',
    }.contains(operation)) {
      return false;
    }
    _startMaintenanceWatch(
      operation: operation,
      message: status.message,
      progress: status.progress,
      statusPath: status.statusPath ?? _maintenanceStatusStore.statusFile.path,
    );
    if (announce) widget.onShowMessage('已恢复正在执行的环境维护任务。');
    return true;
  }

  void _startMaintenanceWatch({
    required String operation,
    required String message,
    required int progress,
    String? statusPath,
  }) {
    _maintenanceTimer?.cancel();
    setState(() {
      _maintenanceOperation = operation;
      _maintenanceMessage = message.isEmpty ? '正在安装...' : message;
      _maintenanceProgress = progress.clamp(0, 100) / 100;
      _maintenanceStatusPath = statusPath;
      _installingPytorch =
          operation == 'install_pytorch' ||
          operation == 'install_yolo_dependencies';
      _reinstallingPackage = operation == 'reinstall_package';
    });
    _maintenanceTimer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => unawaited(_pollMaintenanceStatus()),
    );
    unawaited(_pollMaintenanceStatus());
  }

  Future<void> _pollMaintenanceStatus() async {
    if (!_maintenanceInProgress) return;
    try {
      late final MaintenanceStatus status;
      try {
        status = await widget.apiClient.fetchMaintenanceStatus();
      } catch (_) {
        status = await _readLocalMaintenanceStatus();
      }
      if (!mounted || !_maintenanceInProgress) return;
      final operation = status.operation ?? _maintenanceOperation;
      final message = status.message.isEmpty
          ? (_maintenanceMessage ?? '正在安装...')
          : status.message;
      if (_maintenanceStatusStore.isActive(status)) {
        setState(() {
          _maintenanceOperation = operation;
          _maintenanceMessage = message;
          _maintenanceProgress = status.progress / 100;
          _maintenanceStatusPath = status.statusPath ?? _maintenanceStatusPath;
          _installingPytorch =
              operation == 'install_pytorch' ||
              operation == 'install_yolo_dependencies';
          _reinstallingPackage = operation == 'reinstall_package';
        });
        return;
      }

      _finishMaintenanceWatch(message);
    } catch (_) {
      if (!mounted || !_maintenanceInProgress) return;
      setState(() {
        _maintenanceMessage = '正在安装，暂时无法读取最新进度；Python 后端完成后会自动重启...';
      });
    }
  }

  Future<MaintenanceStatus> _readLocalMaintenanceStatus() async {
    return _maintenanceStatusStore.read(path: _maintenanceStatusPath);
  }

  bool get _maintenanceInProgress => _installingPytorch || _reinstallingPackage;
  bool get _maintenanceBusy =>
      _maintenancePreparationOperation != null || _maintenanceInProgress;

  void _finishMaintenanceWatch(String message) {
    _maintenanceTimer?.cancel();
    _maintenanceTimer = null;
    setState(() {
      _installingPytorch = false;
      _reinstallingPackage = false;
      _maintenanceOperation = null;
      _maintenanceMessage = null;
      _maintenanceProgress = null;
      _maintenanceStatusPath = null;
    });
    if (message.isNotEmpty) widget.onShowMessage(message);
  }

  Future<bool?> _confirmMaintenance({
    required String title,
    required String message,
    required String confirmLabel,
  }) {
    return showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text(confirmLabel),
          ),
        ],
      ),
    );
  }

  Future<bool?> _confirmPythonInstallation({
    required String title,
    required String message,
  }) {
    return _confirmMaintenance(
      title: title,
      message:
          '$message\n\n'
          '安装完成后 Python 后端会自动重启，期间界面可能短暂显示后端离线。',
      confirmLabel: '开始安装',
    );
  }

  void _openExternalUrl(String url) {
    try {
      if (Platform.isWindows) {
        unawaited(Process.run('cmd', ['/c', 'start', '', url]));
      } else if (Platform.isMacOS) {
        unawaited(Process.run('open', [url]));
      } else if (Platform.isLinux) {
        unawaited(Process.run('xdg-open', [url]));
      } else {
        widget.onShowMessage(url);
      }
    } catch (_) {
      widget.onShowMessage(url);
    }
  }

  Future<void> _setDebugMode(bool enabled) async {
    final previous = _bool(_debugModeKey);
    if (_debugModeSaving || previous == enabled) {
      if (previous && enabled) widget.onShowMessage('调试模式已开启');
      return;
    }

    setState(() {
      _debugModeSaving = true;
      _draft[_debugModeKey] = enabled;
    });

    try {
      await widget.onSaveSettings(<String, dynamic>{_debugModeKey: enabled});
      if (!mounted) return;
      setState(() => _debugModeSaving = false);
      widget.onShowMessage(enabled ? '调试模式已开启' : '调试模式已关闭');
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _draft[_debugModeKey] = previous;
        _debugModeSaving = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final debugModeEnabled = _bool(_debugModeKey);
    final selectedIndex = _sectionIndex
        .clamp(0, debugModeEnabled ? 6 : 5)
        .toInt();

    return LayoutBuilder(
      builder: (context, constraints) {
        final drawerWidth = previewDistanceToLeadingDivider(
          constraints.maxWidth,
        );

        return Row(
          children: [
            SizedBox(
              width: drawerWidth,
              child: NavigationDrawer(
                selectedIndex: selectedIndex,
                onDestinationSelected: (index) {
                  setState(() => _sectionIndex = index);
                },
                children: [
                  const Padding(
                    padding: EdgeInsets.fromLTRB(28, 20, 16, 12),
                    child: Text('设置', style: TextStyle(fontSize: 18)),
                  ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.tune_outlined),
                    selectedIcon: Icon(Icons.tune_rounded),
                    label: Text('检测设置'),
                  ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.construction_outlined),
                    selectedIcon: Icon(Icons.construction_rounded),
                    label: Text('环境维护'),
                  ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.fact_check_outlined),
                    selectedIcon: Icon(Icons.fact_check_rounded),
                    label: Text('基础设置'),
                  ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.palette_outlined),
                    selectedIcon: Icon(Icons.palette_rounded),
                    label: Text('外观主题'),
                  ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.system_update_alt_rounded),
                    selectedIcon: Icon(Icons.system_update_alt_rounded),
                    label: Text('软件更新'),
                  ),
                  if (debugModeEnabled)
                    const NavigationDrawerDestination(
                      icon: Icon(Icons.bug_report_outlined),
                      selectedIcon: Icon(Icons.bug_report_rounded),
                      label: Text('调试模式'),
                    ),
                  const NavigationDrawerDestination(
                    icon: Icon(Icons.info_outline_rounded),
                    selectedIcon: Icon(Icons.info_rounded),
                    label: Text('关于'),
                  ),
                ],
              ),
            ),
            const VerticalDivider(width: 1),
            Expanded(
              child: ListView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(16),
                children: [_buildSelectedSection()],
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildSelectedSection() {
    final debugModeEnabled = _bool(_debugModeKey);
    if (_sectionIndex == 5 && debugModeEnabled) {
      return _buildDebugSettings();
    }

    return switch (_sectionIndex) {
      0 => _buildDetectionSettings(),
      1 => _buildEnvironmentMaintenance(),
      2 => _buildBasicSettings(),
      3 => _buildAppearanceSettings(),
      4 => _buildUpdateSettingsSection(),
      _ => _buildProjectSettings(),
    };
  }

  Widget _buildDetectionSettings() {
    final settings = widget.settings;
    final selectedModel = _validModelValue(
      _string('selected_model'),
      settings?.availableModels ?? const <ModelInfo>[],
      allowEmpty: true,
    );
    final selectedClassificationModel = _validModelValue(
      _string('selected_classification_model'),
      settings?.availableClassificationModels ?? const <ModelInfo>[],
      allowEmpty: true,
    );
    final videoMode = _string('video_mode', 'all');
    final strideLabel = videoMode == 'all' ? '帧间隔' : '快速识别帧数';
    final detectionEnabled = _detectionDependenciesReady;
    final combinedModelsEnabled =
        selectedModel?.isNotEmpty == true &&
        selectedClassificationModel?.isNotEmpty == true;
    final confidencePriority =
        _string('confidence_priority', _classificationConfidencePriority) ==
            _detectionConfidencePriority
        ? _detectionConfidencePriority
        : _classificationConfidencePriority;

    return SectionCard(
      title: '检测设置',
      subtitle: '模型管理、识别物种、阈值、推理加速和视频检测',
      icon: Icons.tune_rounded,
      child: Column(
        children: [
          if (!detectionEnabled) _buildDetectionDependencyNotice(),
          if (_maintenanceBusy) _buildMaintenanceProgress(),
          _SettingsPanel(
            title: '探测模型',
            subtitle: '选择目标探测模型，或设为不使用',
            icon: Icons.build_rounded,
            child: _SettingsMenuButton<String>(
              value: selectedModel,
              placeholder: '未发现探测模型',
              enabled: detectionEnabled,
              options: [
                const _SettingsOption<String>(value: '', label: '不使用'),
                ...(settings?.availableModels ?? const <ModelInfo>[]).map(
                  (model) => _SettingsOption<String>(
                    value: model.path,
                    label: model.name,
                  ),
                ),
              ],
              onChanged: (value) {
                _set('selected_model', value);
                if (selectedClassificationModel?.isEmpty ?? true) {
                  _set('selected_species_names', <String>[]);
                  _loadModelClassesForSelection();
                }
              },
            ),
          ),
          _SettingsPanel(
            title: '分类模型',
            subtitle: '用于整图识别或探测后的二次分类，也可设为不使用',
            icon: Icons.account_tree_rounded,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _SettingsMenuButton<String>(
                  value: selectedClassificationModel,
                  enabled: detectionEnabled,
                  options: [
                    const _SettingsOption<String>(value: '', label: '不使用'),
                    ...(settings?.availableClassificationModels ??
                            const <ModelInfo>[])
                        .map(
                          (model) => _SettingsOption<String>(
                            value: model.path,
                            label: model.name,
                          ),
                        ),
                  ],
                  onChanged: (value) {
                    _set('selected_classification_model', value);
                    _set('selected_species_names', <String>[]);
                    _loadModelClassesForSelection();
                  },
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
          if (combinedModelsEnabled)
            _SettingsPanel(
              title: '组合置信度策略',
              subtitle: '设置探测模型与分类模型共同使用时的综合置信度权重',
              icon: Icons.balance_rounded,
              child: _SettingsMenuButton<String>(
                value: confidencePriority,
                enabled: detectionEnabled,
                options: const [
                  _SettingsOption<String>(
                    value: _detectionConfidencePriority,
                    label: '探测置信度优先',
                  ),
                  _SettingsOption<String>(
                    value: _classificationConfidencePriority,
                    label: '分类置信度优先',
                  ),
                ],
                onChanged: (value) => _set('confidence_priority', value),
              ),
            ),
          _SettingsPanel(
            title: '识别物种设置',
            subtitle: '不选择时默认识别全部物种',
            icon: Icons.pets_rounded,
            child: _buildSpeciesSelector(enabled: detectionEnabled),
          ),
          _SettingsPanel(
            title: '检测阈值设置',
            subtitle: '控制目标检测的置信度和 IOU 阈值',
            icon: Icons.speed_rounded,
            child: Column(
              children: [
                _LabeledSlider(
                  label: '置信度',
                  value: _double('confidence'),
                  min: 0.05,
                  max: 0.95,
                  divisions: 90,
                  enabled: detectionEnabled,
                  onChanged: (value) => _set('confidence', value),
                ),
                _LabeledSlider(
                  label: 'IOU',
                  value: _double('iou'),
                  min: 0.10,
                  max: 0.90,
                  divisions: 80,
                  enabled: detectionEnabled,
                  onChanged: (value) => _set('iou', value),
                ),
                _LabeledSlider(
                  label: '输入尺寸',
                  value: _int('imgsz', 1920).toDouble(),
                  min: 640,
                  max: 2560,
                  divisions: 12,
                  valueLabel: _int('imgsz', 1920).toString(),
                  enabled: detectionEnabled,
                  onChanged: (value) => _set('imgsz', value.round()),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '模型加速选项',
            subtitle: '批处理数控制单次推理规模，线程数控制完整视频文件并发',
            icon: Icons.bolt_rounded,
            child: Column(
              children: [
                _LabeledSlider(
                  label: '批处理数',
                  value: _int('batch_size').toDouble(),
                  min: 1,
                  max: 32,
                  divisions: 31,
                  valueLabel: _int('batch_size').toString(),
                  enabled: detectionEnabled,
                  onChanged: (value) => _set('batch_size', value.round()),
                ),
                _LabeledSlider(
                  label: '线程数',
                  value: _int('thread_count', 4).toDouble(),
                  min: 1,
                  max: 8,
                  divisions: 7,
                  valueLabel: _int('thread_count', 4).toString(),
                  enabled: detectionEnabled,
                  onChanged: (value) => _set('thread_count', value.round()),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value:
                      detectionEnabled &&
                      _bool('use_fp16') &&
                      settings?.gpuAvailable == true,
                  onChanged: detectionEnabled && settings?.gpuAvailable == true
                      ? (value) => _set('use_fp16', value)
                      : null,
                  title: const Text('使用 FP16 加速'),
                  subtitle: Text(
                    settings?.gpuAvailable == true
                        ? '需要支持半精度推理的 GPU'
                        : '当前为 CPU 模式，已禁用 FP16',
                  ),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '高级检测选项',
            subtitle: 'TTA 可能提高准确率但会降低速度，NMS 控制跨类别框合并',
            icon: Icons.auto_awesome_rounded,
            child: Column(
              children: [
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: _bool('use_augment'),
                  onChanged: detectionEnabled
                      ? (value) => _set('use_augment', value)
                      : null,
                  title: const Text('使用 TTA 数据增强'),
                  subtitle: const Text('推理时使用测试时增强，通常更慢但可能更稳。'),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: _bool('use_agnostic_nms'),
                  onChanged: detectionEnabled
                      ? (value) => _set('use_agnostic_nms', value)
                      : null,
                  title: const Text('使用跨类别 NMS'),
                  subtitle: const Text('合并不同类别间重叠较高的检测框。'),
                ),
              ],
            ),
          ),
          ..._buildVideoSettingPanels(
            videoMode,
            strideLabel,
            enabled: detectionEnabled,
          ),
        ],
      ),
    );
  }

  Widget _buildDetectionDependencyNotice() {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.errorContainer.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: scheme.error.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.report_problem_rounded, color: scheme.error),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              '检测设置需要先安装 PyTorch、torchvision 和 ultralytics。当前缺少：$_missingYoloDependenciesLabel。',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: scheme.onErrorContainer,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const SizedBox(width: 12),
          FilledButton(
            onPressed: _maintenanceBusy ? null : _installYoloDependencies,
            child:
                _maintenancePreparationOperation ==
                        'install_yolo_dependencies' ||
                    _installingPytorch
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('安装依赖'),
          ),
        ],
      ),
    );
  }

  Widget _buildSpeciesSelector({bool enabled = true}) {
    final speciesNames = _modelSpeciesNames();
    final selected = _stringList('selected_species_names');
    final summary = _loadingModelClasses
        ? '读取中...'
        : _speciesSelectionSummary(selected, speciesNames);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        if (_loadingModelClasses)
          const Padding(
            padding: EdgeInsets.only(bottom: 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                SizedBox(width: 8),
              ],
            ),
          ),
        if (!_loadingModelClasses)
          _SummaryDialogButton(
            summary: summary,
            onPressed: enabled
                ? () => _showSpeciesSelectionDialog(speciesNames)
                : null,
          ),
      ],
    );
  }

  List<String> _modelSpeciesNames() {
    final seen = <String>{};
    return [
      for (final species in _modelClassOptions)
        if (species.label.isNotEmpty && seen.add(species.label)) species.label,
    ];
  }

  String _selectionSummary(
    List<String> selected, {
    required String emptyLabel,
  }) {
    final cleaned = selected.where((item) => item.trim().isNotEmpty).toList();
    if (cleaned.isEmpty) return emptyLabel;
    final preview = cleaned.take(3).join(', ');
    return cleaned.length > 3 ? '$preview, ...' : preview;
  }

  String _speciesSelectionSummary(List<String> selected, List<String> options) {
    final optionSet = options.toSet();
    final cleaned = selected
        .where((item) => item.trim().isNotEmpty)
        .where((item) => optionSet.isEmpty || optionSet.contains(item))
        .toSet();
    if (cleaned.isEmpty ||
        (options.isNotEmpty && cleaned.length >= options.length)) {
      return '全部识别';
    }
    return _selectionSummary(cleaned.toList(), emptyLabel: '全部识别');
  }

  Future<void> _showSpeciesSelectionDialog(List<String> speciesNames) async {
    final initialSelection = _stringList('selected_species_names').toSet();
    final result = await showDialog<List<String>>(
      context: context,
      builder: (context) {
        final selected = initialSelection.toSet();
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final hasOptions = speciesNames.isNotEmpty;
            return AlertDialog(
              title: const Text('选择识别物种'),
              content: SizedBox(
                width: 460,
                height: hasOptions ? 480 : 120,
                child: hasOptions
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const _MutedText('不勾选时默认识别全部模型类别。'),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            children: [
                              TextButton.icon(
                                onPressed: () => setDialogState(() {
                                  selected
                                    ..clear()
                                    ..addAll(speciesNames);
                                }),
                                icon: const Icon(Icons.select_all_rounded),
                                label: const Text('全选'),
                              ),
                              TextButton.icon(
                                onPressed: selected.isEmpty
                                    ? null
                                    : () => setDialogState(() {
                                        selected.clear();
                                      }),
                                icon: const Icon(Icons.deselect_rounded),
                                label: const Text('清空'),
                              ),
                            ],
                          ),
                          const Divider(height: 20),
                          Expanded(
                            child: ListView.builder(
                              itemCount: speciesNames.length,
                              itemBuilder: (context, index) {
                                final name = speciesNames[index];
                                return CheckboxListTile(
                                  dense: true,
                                  value: selected.contains(name),
                                  title: Text(name),
                                  onChanged: (value) {
                                    setDialogState(() {
                                      if (value == true) {
                                        selected.add(name);
                                      } else {
                                        selected.remove(name);
                                      }
                                    });
                                  },
                                );
                              },
                            ),
                          ),
                        ],
                      )
                    : const Center(child: Text('当前模型暂未读取到可选物种，处理时会识别全部模型类别。')),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: hasOptions
                      ? () {
                          final ordered = speciesNames
                              .where(selected.contains)
                              .toList();
                          Navigator.of(context).pop(
                            ordered.length == speciesNames.length
                                ? <String>[]
                                : ordered,
                          );
                        }
                      : () => Navigator.of(context).pop(<String>[]),
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result != null) {
      _set('selected_species_names', result);
    }
  }

  Widget _buildVideoSettings() {
    final videoMode = _string('video_mode', 'all');
    final strideLabel = videoMode == 'all' ? '帧间隔' : '快速识别帧数';

    return SectionCard(
      title: '视频检测设置',
      subtitle: '视频处理模式、跳帧和检测过滤',
      icon: Icons.movie_filter_rounded,
      child: Column(
        children: _buildVideoSettingPanels(
          videoMode,
          strideLabel,
          enabled: _detectionDependenciesReady,
        ),
      ),
    );
  }

  List<Widget> _buildVideoSettingPanels(
    String videoMode,
    String strideLabel, {
    bool enabled = true,
  }) {
    return [
      _SettingsPanel(
        title: '视频处理模式',
        subtitle: '全部识别会按帧间隔处理，快速识别只抽取关键帧',
        icon: Icons.play_circle_outline_rounded,
        child: SegmentedButton<String>(
          segments: const [
            ButtonSegment(
              value: 'all',
              icon: Icon(Icons.video_collection_rounded),
              label: Text('全部识别'),
            ),
            ButtonSegment(
              value: 'fast',
              icon: Icon(Icons.flash_on_rounded),
              label: Text('快速识别'),
            ),
          ],
          selected: {videoMode},
          onSelectionChanged: enabled
              ? (selection) {
                  if (selection.isEmpty) return;
                  _set('video_mode', selection.first);
                }
              : null,
        ),
      ),
      _SettingsPanel(
        title: '跳帧设置',
        subtitle: videoMode == 'all' ? '全部识别模式下表示帧间隔' : '快速识别模式下表示抽取检测的帧数',
        icon: Icons.skip_next_rounded,
        child: _LabeledSlider(
          label: strideLabel,
          value: _int('vid_stride').toDouble(),
          min: 1,
          max: 30,
          divisions: 29,
          valueLabel: _int('vid_stride').toString(),
          enabled: enabled,
          onChanged: (value) => _set('vid_stride', value.round()),
        ),
      ),
      _SettingsPanel(
        title: '检测过滤',
        subtitle: '设置检测到的最低帧数比例',
        icon: Icons.filter_alt_rounded,
        showDivider: false,
        child: _LabeledSlider(
          label: '最低帧数比例',
          value: _double('min_frame_ratio'),
          min: 0,
          max: 0.30,
          divisions: 30,
          valueLabel: '${(_double('min_frame_ratio') * 100).round()}%',
          enabled: enabled,
          onChanged: (value) => _set('min_frame_ratio', value),
        ),
      ),
    ];
  }

  Widget _buildMaintenanceProgress() {
    if (!_maintenanceBusy) return const SizedBox.shrink();
    final scheme = Theme.of(context).colorScheme;
    final message = _maintenanceMessage ?? '正在安装...';
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 4, 0, 12),
      child: Row(
        children: [
          SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(
              value: _maintenanceProgress,
              strokeWidth: 2.6,
              color: scheme.primary,
              backgroundColor: scheme.surfaceContainerHighest,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: scheme.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEnvironmentMaintenance() {
    return SectionCard(
      title: '环境维护',
      subtitle: 'PyTorch 与单个 Python 包维护入口',
      icon: Icons.construction_rounded,
      child: Column(
        children: [
          if (_maintenanceBusy) _buildMaintenanceProgress(),
          _SettingsPanel(
            title: 'Python 包安装源',
            subtitle: '用于 PyTorch 依赖、ultralytics 和单个 Python 包安装',
            icon: Icons.travel_explore_rounded,
            child: _SettingsMenuButton<String>(
              value: _string('package_source', 'auto'),
              enabled: !_maintenanceBusy,
              options: const [
                _SettingsOption<String>(value: 'auto', label: '自动选择源'),
                _SettingsOption<String>(value: 'official', label: '官方源'),
                _SettingsOption<String>(value: 'aliyun', label: '阿里源'),
                _SettingsOption<String>(value: 'tsinghua', label: '清华源'),
                _SettingsOption<String>(value: 'nju', label: '南京大学源'),
              ],
              onChanged: (value) => _set('package_source', value),
            ),
          ),
          _SettingsPanel(
            title: '安装 PyTorch',
            subtitle: '选择安装环境后执行安装或修复',
            icon: Icons.download_rounded,
            child: Row(
              children: [
                Expanded(
                  child: _SettingsMenuButton<String>(
                    value: _string('pytorch_version', '自动检测'),
                    enabled: !_maintenanceBusy,
                    options: const [
                      _SettingsOption<String>(value: '自动检测', label: '自动检测'),
                      _SettingsOption<String>(
                        value: 'CUDA 13.2',
                        label: 'CUDA 13.2',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 13.0',
                        label: 'CUDA 13.0',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.8',
                        label: 'CUDA 12.8',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.6',
                        label: 'CUDA 12.6',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.4',
                        label: 'CUDA 12.4',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.1',
                        label: 'CUDA 12.1',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 11.8',
                        label: 'CUDA 11.8',
                      ),
                      if (_xpuEnabled)
                        _SettingsOption<String>(
                          value: 'Intel XPU',
                          label: 'Intel XPU',
                        ),
                      _SettingsOption<String>(
                        value: 'CPU Only',
                        label: 'CPU Only',
                      ),
                    ],
                    onChanged: (value) => _set('pytorch_version', value),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _maintenanceBusy ? null : _installPytorch,
                  child:
                      _maintenancePreparationOperation == 'install_pytorch' ||
                          _installingPytorch
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('安装'),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '重新安装单个 Python 包',
            subtitle: '使用 toolkit\\python.exe 强制重新安装指定包',
            icon: Icons.extension_rounded,
            showDivider: false,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _packageController,
                    decoration: const InputDecoration(
                      labelText: '包名',
                      hintText: '例如 numpy 或 numpy==1.26.4',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) => _draft['package'] = value.trim(),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _maintenanceBusy ? null : _reinstallPythonPackage,
                  child:
                      _maintenancePreparationOperation == 'reinstall_package' ||
                          _reinstallingPackage
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('重新安装'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBasicSettings() {
    final undoSteps = _undoStepsSetting(_draft);
    return SectionCard(
      title: '基础设置',
      subtitle: '窗口关闭、校验辅助、缓存清理、快速标记和导出字段',
      icon: Icons.fact_check_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '物种校验界面自动分组',
            subtitle: '开启后，将根据视频文件自动对连续拍摄的照片和视频进行分组，并在文件列表中按所在组校验最多的物种统一归类显示。',
            icon: Icons.auto_awesome_motion_rounded,
            child: Align(
              alignment: Alignment.centerRight,
              child: Switch(
                value: _bool('auto_group'),
                onChanged: (value) => _set('auto_group', value),
              ),
            ),
          ),
          if (_bool('auto_group')) ...[
            _SettingsPanel(
              title: '折叠分组',
              subtitle:
                  '开启后，物种校验界面的分组默认折叠，点击标记会直接标记该分组内的所有照片和视频；在文件列表中右键可展开或折叠分组。',
              icon: Icons.folder_copy_rounded,
              child: Align(
                alignment: Alignment.centerRight,
                child: Switch(
                  value: _bool('collapse_groups'),
                  onChanged: (value) => _set('collapse_groups', value),
                ),
              ),
            ),
            _SettingsPanel(
              title: '自动分组规则',
              subtitle: '综合拍摄时间和连拍照片张数判断事件边界；下一项是配套视频时会优先并入当前组。',
              icon: Icons.timelapse_rounded,
              child: _SummaryDialogButton(
                summary: _autoGroupRuleSummary(),
                onPressed: _showAutoGroupRulesDialog,
              ),
            ),
          ],
          _SettingsPanel(
            title: '可撤回记录步数',
            subtitle: '最多保留多少次校验操作可供逐步撤回；批量标记会作为一次操作整体记录。',
            icon: Icons.undo_rounded,
            child: _LabeledSlider(
              label: '记录步数',
              value: undoSteps.toDouble(),
              min: 10,
              max: 200,
              divisions: 19,
              valueLabel: '$undoSteps 步',
              onChanged: (value) =>
                  _set('undo_steps', _normalizeUndoSteps(value.round())),
            ),
          ),
          _buildClearCachePanel(),
          _buildQuickMarkEditor(),
          Builder(
            builder: (context) => Padding(
              padding: const EdgeInsets.only(top: 16, bottom: 12),
              child: Divider(
                height: 1,
                color: Theme.of(context).colorScheme.outlineVariant,
              ),
            ),
          ),
          _buildExportColumns(showDivider: true),
          _buildFavoritePhotoExportMode(showDivider: true),
          _buildCloseBehaviorSettings(),
        ],
      ),
    );
  }

  Widget _buildCloseBehaviorSettings({bool showDivider = false}) {
    return _SettingsPanel(
      title: '关闭主窗口时',
      subtitle: '选择每次询问、隐藏到任务托盘或直接退出主程序',
      icon: Icons.close_rounded,
      showDivider: showDivider,
      child: _SettingsMenuButton<String>(
        value: normalizeCloseBehavior(widget.closeBehavior),
        options: const [
          _SettingsOption<String>(value: closeBehaviorAsk, label: '每次询问'),
          _SettingsOption<String>(
            value: closeBehaviorHideToTray,
            label: '隐藏到任务托盘',
          ),
          _SettingsOption<String>(value: closeBehaviorExit, label: '退出主程序'),
        ],
        onChanged: widget.onCloseBehaviorChanged,
      ),
    );
  }

  String _autoGroupRuleSummary() {
    if (!_bool('auto_group')) return '已关闭';
    final burst = _int('auto_group_burst_size').clamp(1, 20).toInt();
    final gap = _int('auto_group_gap_seconds').clamp(1, 3600).toInt();
    final burstText = _bool('auto_group_detect_burst')
        ? '自动识别 · 当前 ${_autoGroupDisplayBurstSize(burst)} 张'
        : '$burst 张';
    return '$burstText · $gap 秒';
  }

  int _autoGroupDisplayBurstSize(int fallback) {
    final inferred = widget.autoGroupInferredBurstSize;
    if (inferred == null) return fallback;
    return inferred.clamp(1, 20).toInt();
  }

  Future<void> _showAutoGroupRulesDialog() async {
    var detectBurst = _bool('auto_group_detect_burst');
    var burstSize = _int('auto_group_burst_size').clamp(1, 20).toInt();
    var gapSeconds = _int('auto_group_gap_seconds').clamp(1, 3600).toInt();

    final result = await showDialog<Map<String, dynamic>>(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final currentBurstSize = _autoGroupDisplayBurstSize(burstSize);
            return AlertDialog(
              title: const Text('自动分组规则'),
              content: SizedBox(
                width: 460,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SwitchListTile(
                      contentPadding: EdgeInsets.zero,
                      value: detectBurst,
                      onChanged: (value) =>
                          setDialogState(() => detectBurst = value),
                      title: const Text('自动识别连拍张数'),
                      subtitle: Text(
                        '开启后禁用手动连拍张数；当前分组连拍张数为 $currentBurstSize 张',
                      ),
                    ),
                    const SizedBox(height: 8),
                    _LabeledSlider(
                      label: '连拍张数',
                      value: (detectBurst ? currentBurstSize : burstSize)
                          .toDouble(),
                      min: 1,
                      max: 20,
                      divisions: 19,
                      valueLabel:
                          '${detectBurst ? currentBurstSize : burstSize} 张',
                      enabled: !detectBurst,
                      onChanged: (value) =>
                          setDialogState(() => burstSize = value.round()),
                    ),
                    _LabeledSlider(
                      label: '时间间隔',
                      value: gapSeconds.toDouble(),
                      min: 5,
                      max: 120,
                      divisions: 23,
                      valueLabel: '$gapSeconds 秒',
                      onChanged: (value) =>
                          setDialogState(() => gapSeconds = value.round()),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(context).pop({
                    'auto_group_detect_burst': detectBurst,
                    'auto_group_burst_size': burstSize,
                    'auto_group_gap_seconds': gapSeconds,
                  }),
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result == null) return;
    setState(() {
      _draft['auto_group_detect_burst'] = result['auto_group_detect_burst'];
      _draft['auto_group_burst_size'] = result['auto_group_burst_size'];
      _draft['auto_group_gap_seconds'] = result['auto_group_gap_seconds'];
      _markDraftChanged();
    });
    _scheduleAutoSave();
  }

  Widget _buildClearCachePanel() {
    return _SettingsPanel(
      title: '清理缓存',
      subtitle: '可清除软件日志、运行状态和崩溃标记等可再生缓存，不会删除设置和物种数据库。',
      icon: Icons.cleaning_services_rounded,
      child: Align(
        alignment: Alignment.centerRight,
        child: FilledButton.tonalIcon(
          onPressed: _clearingCache ? null : _showClearCacheDialog,
          icon: _clearingCache
              ? const SizedBox.square(
                  dimension: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.delete_sweep_rounded),
          label: Text(_clearingCache ? '清理中' : '选择清理'),
        ),
      ),
    );
  }

  Future<void> _showClearCacheDialog() async {
    var clearLogs = true;
    var clearSoftwareCache = true;
    final selection = await showDialog<({bool logs, bool softwareCache})>(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final canSubmit = clearLogs || clearSoftwareCache;
            return AlertDialog(
              title: const Text('清理缓存'),
              content: SizedBox(
                width: 440,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CheckboxListTile(
                      contentPadding: EdgeInsets.zero,
                      value: clearLogs,
                      onChanged: (value) =>
                          setDialogState(() => clearLogs = value ?? false),
                      title: const Text('清除日志'),
                      subtitle: const Text('清除 logs 目录中的软件日志。'),
                    ),
                    CheckboxListTile(
                      contentPadding: EdgeInsets.zero,
                      value: clearSoftwareCache,
                      onChanged: (value) => setDialogState(
                        () => clearSoftwareCache = value ?? false,
                      ),
                      title: const Text('清除软件缓存'),
                      subtitle: const Text('清除运行状态、崩溃标记和 Python 缓存。'),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: canSubmit
                      ? () => Navigator.of(context).pop((
                          logs: clearLogs,
                          softwareCache: clearSoftwareCache,
                        ))
                      : null,
                  child: const Text('清理'),
                ),
              ],
            );
          },
        );
      },
    );
    if (selection == null || !mounted) return;

    setState(() => _clearingCache = true);
    try {
      final result = await widget.apiClient.clearCache(
        clearLogs: selection.logs,
        clearSoftwareCache: selection.softwareCache,
      );
      if (!mounted) return;
      widget.onShowMessage(_clearCacheMessage(result));
    } catch (error) {
      if (!mounted) return;
      widget.onShowMessage('清理缓存失败：$error');
    } finally {
      if (mounted) setState(() => _clearingCache = false);
    }
  }

  String _clearCacheMessage(ClearCacheResult result) {
    final parts = <String>[
      '已清理 ${result.clearedFiles} 个文件',
      '${result.clearedDirectories} 个目录',
      _formatBytes(result.reclaimedBytes),
    ];
    if (result.skipped.isNotEmpty) {
      parts.add('${result.skipped.length} 项未能清理');
    }
    return parts.join(' · ');
  }

  Widget _buildQuickMarkSettings() {
    return SectionCard(
      title: '快速标记设置',
      subtitle: '物种按钮、数量按钮和自动排序',
      icon: Icons.bookmark_add_rounded,
      child: _buildQuickMarkEditor(),
    );
  }

  Widget _buildExportSettings() {
    return SectionCard(
      title: '导出设置',
      subtitle: '自定义导出表格中的列和收藏照片同步策略',
      icon: Icons.table_chart_rounded,
      child: Column(
        children: [
          _buildExportColumns(showDivider: true),
          _buildFavoritePhotoExportMode(),
        ],
      ),
    );
  }

  Widget _buildAppearanceSettings() {
    return SectionCard(
      title: '外观与主题',
      subtitle: '主题模式、动态颜色和调色板',
      icon: Icons.palette_rounded,
      child: _AppearanceControls(
        themeNotifier: widget.themeNotifier,
        onUpdateTheme: widget.onUpdateTheme,
      ),
    );
  }

  Widget _buildUpdateSettingsSection() {
    return SectionCard(
      title: '软件更新',
      subtitle: '选择更新通道和镜像源',
      icon: Icons.system_update_alt_rounded,
      child: _buildUpdateSettings(),
    );
  }

  Widget _buildDebugSettings() {
    return SectionCard(
      title: '调试模式',
      subtitle: '当前环境、依赖、已安装库和软件日志',
      icon: Icons.bug_report_rounded,
      child: _DebugInfoPanel(
        settings: widget.settings,
        apiClient: widget.apiClient,
        debugModeSaving: _debugModeSaving,
        onDebugModeChanged: _setDebugMode,
        onShowMessage: widget.onShowMessage,
      ),
    );
  }

  Widget _buildProjectSettings() {
    final debugModeEnabled = _bool(_debugModeKey);
    return SectionCard(
      title: '关于',
      subtitle: '版本、更新说明、官网和项目入口',
      icon: Icons.info_rounded,
      child: _AboutInfo(
        settings: widget.settings,
        debugModeEnabled: debugModeEnabled,
        debugModeSaving: _debugModeSaving,
        onDebugModeChanged: _setDebugMode,
        onOpenUrl: _openExternalUrl,
        onShowMessage: widget.onShowMessage,
      ),
    );
  }

  Widget _buildQuickMarkEditor() {
    final configuredMarks = _stringList('quick_mark_list');
    final displayedMarks = _bool('auto_sort')
        ? _autoSortedConfiguredQuickMarks(configuredMarks)
        : configuredMarks;
    final quantityButtons = _quantityButtons();
    return Column(
      children: [
        _SettingsPanel(
          title: '物种按钮',
          subtitle: '设置校验界面右侧快速标记列显示的物种按钮',
          icon: Icons.bookmarks_rounded,
          child: _SummaryDialogButton(
            summary: _quickMarksSummary(displayedMarks),
            onPressed: _showQuickMarkButtonsDialog,
          ),
        ),
        _SettingsPanel(
          title: '数量按钮',
          subtitle: '选择校验界面右侧数量列显示哪些数字快捷按钮',
          icon: Icons.format_list_numbered_rounded,
          showDivider: false,
          child: _SummaryDialogButton(
            summary: _quantityButtonsSummary(quantityButtons),
            onPressed: _showQuantityButtonsDialog,
          ),
        ),
      ],
    );
  }

  String _quickMarksSummary(List<String> values) {
    final cleaned = values.where((value) => value.trim().isNotEmpty).toList();
    if (cleaned.isEmpty) return '使用默认物种';
    if (cleaned.length <= 3) return cleaned.join('、');
    return '${cleaned.take(2).join('、')} 等 ${cleaned.length} 个';
  }

  List<String> _autoSortedConfiguredQuickMarks(Iterable<String> values) {
    final configured = values
        .map((value) => value.trim())
        .where((value) => value.isNotEmpty)
        .toList();
    if (configured.isEmpty) return configured;

    final sorted = autoSortedQuickMarks(
      configuredMarks: configured,
      usageCounts: _intMapSetting(_draft, 'quick_mark_usage_counts'),
      recentHistory: _stringList('quick_mark_recent_history'),
    );
    final remaining = configured.toList();
    final result = <String>[];
    for (final species in sorted) {
      final index = remaining.indexOf(species);
      if (index < 0) continue;
      result.add(remaining.removeAt(index));
    }
    result.addAll(remaining);
    return result;
  }

  Future<void> _showQuickMarkButtonsDialog() async {
    final markControllers = _stringList(
      'quick_mark_list',
    ).map((value) => TextEditingController(text: value)).toList();
    final removedMarkControllers = <TextEditingController>[];
    final newMarkController = TextEditingController();
    Map<String, dynamic>? result;

    try {
      result = await showDialog<Map<String, dynamic>>(
        context: context,
        builder: (context) {
          var autoSort = _bool('auto_sort');
          return StatefulBuilder(
            builder: (context, setDialogState) {
              List<TextEditingController> displayedControllers() {
                if (!autoSort) return markControllers;
                final sorted = autoSortedQuickMarks(
                  configuredMarks: markControllers.map(
                    (controller) => controller.text.trim(),
                  ),
                  usageCounts: _intMapSetting(
                    _draft,
                    'quick_mark_usage_counts',
                  ),
                  recentHistory: _stringList('quick_mark_recent_history'),
                );
                final remaining = markControllers.toList();
                final result = <TextEditingController>[];
                for (final species in sorted) {
                  final index = remaining.indexWhere(
                    (controller) => controller.text.trim() == species,
                  );
                  if (index < 0) continue;
                  result.add(remaining.removeAt(index));
                }
                result.addAll(remaining);
                return result;
              }

              void addQuickMark() {
                final value = newMarkController.text.trim();
                if (value.isEmpty) return;
                setDialogState(() {
                  markControllers.add(TextEditingController(text: value));
                  newMarkController.clear();
                });
              }

              return AlertDialog(
                title: const Text('物种按钮'),
                content: SizedBox(
                  width: 460,
                  height: 480,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const _MutedText('设置校验界面右侧快速标记列显示的物种按钮。拖拽可调整按钮顺序。'),
                      const SizedBox(height: 8),
                      SwitchListTile(
                        value: autoSort,
                        contentPadding: EdgeInsets.zero,
                        title: const Text('自动排序'),
                        subtitle: const Text('开启后下方列表按校验页当前自动排序展示'),
                        onChanged: (value) {
                          setDialogState(() => autoSort = value);
                        },
                      ),
                      const Divider(height: 20),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: newMarkController,
                              decoration: const InputDecoration(
                                labelText: '新增快速标记',
                                isDense: true,
                                border: OutlineInputBorder(),
                              ),
                              onSubmitted: (_) => addQuickMark(),
                            ),
                          ),
                          const SizedBox(width: 10),
                          FilledButton.icon(
                            onPressed: addQuickMark,
                            icon: const Icon(Icons.add_rounded),
                            label: const Text('新增'),
                          ),
                        ],
                      ),
                      const Divider(height: 20),
                      Expanded(
                        child: markControllers.isEmpty
                            ? const Center(child: Text('保存空列表时会使用默认物种按钮。'))
                            : Builder(
                                builder: (context) {
                                  final displayItems = displayedControllers();
                                  return ReorderableListView.builder(
                                    buildDefaultDragHandles: false,
                                    itemCount: displayItems.length,
                                    onReorderItem: (oldIndex, newIndex) {
                                      if (autoSort) return;
                                      setDialogState(() {
                                        final item = markControllers.removeAt(
                                          oldIndex,
                                        );
                                        markControllers.insert(newIndex, item);
                                      });
                                    },
                                    itemBuilder: (context, index) {
                                      final controller = displayItems[index];
                                      return Padding(
                                        key: ObjectKey(controller),
                                        padding: const EdgeInsets.only(
                                          bottom: 8,
                                        ),
                                        child: Row(
                                          children: [
                                            SizedBox(
                                              width: 36,
                                              child: Text('${index + 1}'),
                                            ),
                                            Expanded(
                                              child: TextField(
                                                controller: controller,
                                                decoration:
                                                    const InputDecoration(
                                                      isDense: true,
                                                      border:
                                                          OutlineInputBorder(),
                                                    ),
                                                onChanged: (_) {
                                                  if (autoSort) {
                                                    setDialogState(() {});
                                                  }
                                                },
                                              ),
                                            ),
                                            if (autoSort)
                                              IconButton(
                                                tooltip: '自动排序开启时由使用记录决定顺序',
                                                onPressed: null,
                                                icon: const Icon(
                                                  Icons.drag_handle_rounded,
                                                ),
                                              )
                                            else
                                              ReorderableDragStartListener(
                                                index: index,
                                                child: IconButton(
                                                  tooltip: '拖拽排序',
                                                  onPressed: () {},
                                                  icon: const Icon(
                                                    Icons.drag_handle_rounded,
                                                  ),
                                                ),
                                              ),
                                            IconButton(
                                              tooltip: '删除',
                                              onPressed: () {
                                                setDialogState(() {
                                                  final removedIndex =
                                                      markControllers.indexOf(
                                                        controller,
                                                      );
                                                  if (removedIndex < 0) return;
                                                  final removed =
                                                      markControllers.removeAt(
                                                        removedIndex,
                                                      );
                                                  removedMarkControllers.add(
                                                    removed,
                                                  );
                                                });
                                              },
                                              icon: const Icon(
                                                Icons.delete_outline_rounded,
                                              ),
                                            ),
                                          ],
                                        ),
                                      );
                                    },
                                  );
                                },
                              ),
                      ),
                    ],
                  ),
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('取消'),
                  ),
                  FilledButton(
                    onPressed: () {
                      final next = markControllers
                          .map((controller) => controller.text.trim())
                          .where((value) => value.isNotEmpty)
                          .toList(growable: false);
                      Navigator.of(
                        context,
                      ).pop({'auto_sort': autoSort, 'quick_mark_list': next});
                    },
                    child: const Text('保存'),
                  ),
                ],
              );
            },
          );
        },
      );
    } finally {
      newMarkController.dispose();
      for (final controller in markControllers) {
        controller.dispose();
      }
      for (final controller in removedMarkControllers) {
        controller.dispose();
      }
    }

    final savedResult = result;
    if (savedResult != null) {
      setState(() {
        _draft['auto_sort'] = savedResult['auto_sort'];
        _draft['quick_mark_list'] = savedResult['quick_mark_list'];
        _markDraftChanged();
      });
      _scheduleAutoSave();
    }
  }

  List<String> _quantityButtons() {
    final values = _normalizeQuantityButtons(_stringList('quantity_buttons'));
    return values.isEmpty ? List<String>.from(_defaultQuantityButtons) : values;
  }

  List<String> _quantityButtonCandidates() {
    return _normalizeQuantityButtons([
      ..._quantityButtons(),
      ..._defaultQuantityButtons,
    ]);
  }

  List<String> _normalizeQuantityButtons(Iterable<String> values) {
    final result = <String>[];
    final seen = <String>{};
    for (final value in values) {
      final parsed = int.tryParse(value.trim());
      if (parsed == null || parsed <= 0) continue;
      final label = parsed.toString();
      if (seen.add(label)) result.add(label);
    }
    return result;
  }

  String _quantityButtonsSummary(List<String> values) {
    if (values.isEmpty) return '使用默认数量';
    if (values.length <= 5) return values.join('、');
    return '${values.take(4).join('、')} 等 ${values.length} 个';
  }

  Future<void> _showQuantityButtonsDialog() async {
    final initialSelection = _quantityButtons().toSet();
    final ordered = _quantityButtonCandidates();
    final quantityController = TextEditingController(
      text: initialSelection.isEmpty ? '1' : initialSelection.first,
    );
    List<String>? result;
    try {
      result = await showDialog<List<String>>(
        context: context,
        builder: (context) {
          final selected = initialSelection.toSet();
          final items = ordered.toList();

          return StatefulBuilder(
            builder: (context, setDialogState) {
              void setControllerValue(int value) {
                final text = math.max(1, value).toString();
                quantityController.value = TextEditingValue(
                  text: text,
                  selection: TextSelection.collapsed(offset: text.length),
                );
              }

              void adjustQuantity(int delta) {
                final current =
                    int.tryParse(quantityController.text.trim()) ?? 1;
                setDialogState(() => setControllerValue(current + delta));
              }

              void setQuantityButton() {
                final parsed = int.tryParse(quantityController.text.trim());
                if (parsed == null || parsed <= 0) return;
                final quantity = parsed.toString();
                setDialogState(() {
                  if (!items.contains(quantity)) {
                    items.insert(0, quantity);
                  }
                  selected.add(quantity);
                  setControllerValue(parsed);
                });
              }

              Widget buildQuantityControls() {
                final input = TextField(
                  controller: quantityController,
                  keyboardType: TextInputType.number,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  decoration: const InputDecoration(
                    labelText: '数量按钮',
                    isDense: true,
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => setQuantityButton(),
                );
                final setButton = FilledButton.icon(
                  onPressed: setQuantityButton,
                  icon: const Icon(Icons.check_rounded),
                  label: const Text('设置'),
                );

                return LayoutBuilder(
                  builder: (context, constraints) {
                    final compact = constraints.maxWidth < 400;
                    final stepper = Row(
                      children: [
                        IconButton.outlined(
                          tooltip: '减少',
                          onPressed: () => adjustQuantity(-1),
                          icon: const Icon(Icons.remove_rounded),
                        ),
                        const SizedBox(width: 8),
                        Expanded(child: input),
                        const SizedBox(width: 8),
                        IconButton.outlined(
                          tooltip: '增加',
                          onPressed: () => adjustQuantity(1),
                          icon: const Icon(Icons.add_rounded),
                        ),
                      ],
                    );

                    if (compact) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          stepper,
                          const SizedBox(height: 10),
                          setButton,
                        ],
                      );
                    }

                    return Row(
                      children: [
                        Expanded(child: stepper),
                        const SizedBox(width: 10),
                        setButton,
                      ],
                    );
                  },
                );
              }

              return AlertDialog(
                title: const Text('数量按钮'),
                content: SizedBox(
                  width: 460,
                  height: 480,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const _MutedText('勾选校验界面右侧数量列要显示的数字按钮。拖拽可调整按钮顺序。'),
                      const SizedBox(height: 8),
                      buildQuantityControls(),
                      const Divider(height: 20),
                      Expanded(
                        child: ReorderableListView.builder(
                          buildDefaultDragHandles: false,
                          itemCount: items.length,
                          onReorderItem: (oldIndex, newIndex) {
                            setDialogState(() {
                              final item = items.removeAt(oldIndex);
                              items.insert(newIndex, item);
                            });
                          },
                          itemBuilder: (context, index) {
                            final quantity = items[index];
                            final checked = selected.contains(quantity);
                            return CheckboxListTile(
                              key: ValueKey('quantity-button-$quantity'),
                              dense: true,
                              value: checked,
                              title: Text(quantity),
                              contentPadding: EdgeInsets.zero,
                              controlAffinity: ListTileControlAffinity.leading,
                              secondary: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  ReorderableDragStartListener(
                                    index: index,
                                    child: IconButton(
                                      tooltip: '拖拽排序',
                                      onPressed: () {},
                                      icon: const Icon(
                                        Icons.drag_handle_rounded,
                                      ),
                                    ),
                                  ),
                                  IconButton(
                                    tooltip: '删除',
                                    onPressed: () {
                                      setDialogState(() {
                                        items.removeAt(index);
                                        selected.remove(quantity);
                                      });
                                    },
                                    icon: const Icon(
                                      Icons.delete_outline_rounded,
                                    ),
                                  ),
                                ],
                              ),
                              onChanged: (value) {
                                setDialogState(() {
                                  if (value == true) {
                                    selected.add(quantity);
                                  } else {
                                    selected.remove(quantity);
                                  }
                                  quantityController.text = quantity;
                                });
                              },
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('取消'),
                  ),
                  FilledButton(
                    onPressed: selected.isEmpty
                        ? null
                        : () {
                            final next = items
                                .where(selected.contains)
                                .toList(growable: false);
                            Navigator.of(context).pop(next);
                          },
                    child: const Text('保存'),
                  ),
                ],
              );
            },
          );
        },
      );
    } finally {
      quantityController.dispose();
    }

    if (result != null) {
      _set('quantity_buttons', result);
    }
  }

  Widget _buildExportColumns({bool showDivider = false}) {
    final selected = _stringList('export_columns');
    final summary = selected.isEmpty
        ? '使用默认列'
        : selected.length == _defaultExportColumns.length
        ? '全部列'
        : _selectionSummary(selected, emptyLabel: '使用默认列');

    return _SettingsPanel(
      title: '导出表格列',
      subtitle: '选择校验数据导出时包含的字段',
      icon: Icons.view_column_rounded,
      showDivider: showDivider,
      child: _SummaryDialogButton(
        summary: summary,
        onPressed: _showExportColumnsDialog,
      ),
    );
  }

  Widget _buildFavoritePhotoExportMode({bool showDivider = false}) {
    return _SettingsPanel(
      title: '同步导出收藏照片',
      subtitle: '导出校验表格时，选择是否把收藏照片复制到单独文件夹。',
      icon: Icons.star_rounded,
      showDivider: showDivider,
      child: _SettingsMenuButton<String>(
        value: _favoritePhotoExportMode(),
        options: const [
          _SettingsOption<String>(
            value: _favoritePhotoExportAlways,
            label: '导出',
          ),
          _SettingsOption<String>(
            value: _favoritePhotoExportNever,
            label: '不导出',
          ),
          _SettingsOption<String>(
            value: _favoritePhotoExportAsk,
            label: '每次询问',
          ),
        ],
        onChanged: (value) => _set('favorite_photo_export_mode', value),
      ),
    );
  }

  Future<void> _showExportColumnsDialog() async {
    final initialSelection = _stringList('export_columns').toSet();
    final result = await showDialog<List<String>>(
      context: context,
      builder: (context) {
        final selected = initialSelection.toSet();
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('选择导出表格列'),
              content: SizedBox(
                width: 460,
                height: 480,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      children: [
                        TextButton.icon(
                          onPressed: () => setDialogState(() {
                            selected
                              ..clear()
                              ..addAll(_defaultExportColumns);
                          }),
                          icon: const Icon(Icons.select_all_rounded),
                          label: const Text('全选'),
                        ),
                        TextButton.icon(
                          onPressed: selected.isEmpty
                              ? null
                              : () => setDialogState(() {
                                  selected.clear();
                                }),
                          icon: const Icon(Icons.deselect_rounded),
                          label: const Text('清空'),
                        ),
                      ],
                    ),
                    const Divider(height: 20),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _defaultExportColumns.length,
                        itemBuilder: (context, index) {
                          final column = _defaultExportColumns[index];
                          return CheckboxListTile(
                            dense: true,
                            value: selected.contains(column),
                            title: Text(column),
                            onChanged: (value) {
                              setDialogState(() {
                                if (value == true) {
                                  selected.add(column);
                                } else {
                                  selected.remove(column);
                                }
                              });
                            },
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: () {
                    final ordered = _defaultExportColumns
                        .where(selected.contains)
                        .toList();
                    Navigator.of(context).pop(ordered);
                  },
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result != null) {
      _set('export_columns', result);
    }
  }

  String _normalizedUpdateMirror(String value) {
    try {
      return normalizeGithubMirrorTemplate(value);
    } on FormatException {
      return defaultGithubMirrorTemplate;
    }
  }

  List<String> _normalizedUpdateMirrors(Iterable<String> values) {
    final mirrors = <String>[];
    for (final value in values) {
      try {
        final normalized = normalizeGithubMirrorTemplate(value);
        if (!mirrors.contains(normalized)) {
          mirrors.add(normalized);
        }
      } on FormatException {
        // Ignore malformed entries saved by older or manually edited settings.
      }
    }
    return mirrors;
  }

  List<String> _updateMirrors() {
    return _normalizedUpdateMirrors(_stringList(_updateMirrorsKey));
  }

  String _updateSourceSummary() {
    if (_string('update_mirror', 'KKGitHub') == 'Official') {
      return 'GitHub 官方源';
    }
    final mirrorCount = _updateMirrors().length;
    return mirrorCount == 0 ? '国内源 · 仅使用官方回退' : '国内源 · $mirrorCount 个镜像';
  }

  Future<String?> _promptForUpdateMirror(BuildContext ownerContext) async {
    final controller = TextEditingController();
    String? validationMessage;
    final result = await showDialog<String>(
      context: ownerContext,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            void submit() {
              try {
                final normalized = normalizeGithubMirrorTemplate(
                  controller.text,
                );
                final officialProbe = Uri.parse(
                  'https://github.com/wakin721/Neri/releases/download/v1/Neri.zip',
                );
                final resolved = resolveGithubMirrorUri(
                  officialProbe,
                  normalized,
                );
                if (resolved == officialProbe) {
                  throw const FormatException('该地址仍指向 GitHub 官方源');
                }
                Navigator.of(dialogContext).pop(normalized);
              } on FormatException catch (error) {
                setDialogState(() => validationMessage = error.message);
              }
            }

            return AlertDialog(
              title: const Text('添加 GitHub 镜像站'),
              content: SizedBox(
                width: 520,
                child: TextField(
                  controller: controller,
                  autofocus: true,
                  maxLength: 500,
                  keyboardType: TextInputType.url,
                  textInputAction: TextInputAction.done,
                  onSubmitted: (_) => submit(),
                  decoration: InputDecoration(
                    labelText: '镜像地址或模板',
                    hintText: 'https://mirror.example.com',
                    helperText:
                        '站点根地址会替换 github.com；代理前缀可写成 https://example.com/{url}，也支持 {path}。',
                    helperMaxLines: 3,
                    errorText: validationMessage,
                  ),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(dialogContext).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(onPressed: submit, child: const Text('添加')),
              ],
            );
          },
        );
      },
    );
    controller.dispose();
    return result;
  }

  Future<void> _showUpdateSourcesDialog() async {
    var source = _string('update_mirror', 'KKGitHub') == 'Official'
        ? 'Official'
        : 'KKGitHub';
    final mirrors = _updateMirrors().toList(growable: true);
    final result = await showDialog<({String source, List<String> mirrors})>(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            Future<void> addMirror() async {
              final mirror = await _promptForUpdateMirror(dialogContext);
              if (mirror == null || mirrors.contains(mirror)) return;
              setDialogState(() => mirrors.add(mirror));
            }

            Widget sourceButton({
              required String value,
              required IconData icon,
              required String label,
            }) {
              final selected = source == value;
              return TextButton.icon(
                onPressed: () => setDialogState(() => source = value),
                icon: Icon(selected ? Icons.check_circle_rounded : icon),
                label: Text(label),
              );
            }

            return AlertDialog(
              title: const Text('选择更新镜像源'),
              content: SizedBox(
                width: 460,
                height: 480,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      children: [
                        sourceButton(
                          value: 'Official',
                          icon: Icons.public_rounded,
                          label: 'GitHub 官方源',
                        ),
                        sourceButton(
                          value: 'KKGitHub',
                          icon: Icons.dns_rounded,
                          label: '国内镜像源',
                        ),
                        if (source == 'KKGitHub')
                          TextButton.icon(
                            onPressed: addMirror,
                            icon: const Icon(Icons.add_rounded),
                            label: const Text('添加'),
                          ),
                        if (source == 'KKGitHub')
                          TextButton.icon(
                            onPressed: mirrors.isEmpty
                                ? null
                                : () => setDialogState(mirrors.clear),
                            icon: const Icon(Icons.deselect_rounded),
                            label: const Text('清空'),
                          ),
                      ],
                    ),
                    const Divider(height: 20),
                    if (source == 'Official')
                      const Expanded(
                        child: Center(
                          child: _MutedText('更新文件将直接从 GitHub 官方地址下载。'),
                        ),
                      )
                    else ...[
                      const _MutedText('按从上到下的顺序逐个尝试；全部失败后使用 GitHub 官方源。'),
                      const SizedBox(height: 8),
                      Expanded(
                        child: mirrors.isEmpty
                            ? const Center(
                                child: _MutedText('当前没有国内镜像，将直接使用 GitHub 官方源。'),
                              )
                            : ReorderableListView.builder(
                                buildDefaultDragHandles: false,
                                itemCount: mirrors.length,
                                onReorderItem: (oldIndex, newIndex) {
                                  setDialogState(() {
                                    final mirror = mirrors.removeAt(oldIndex);
                                    mirrors.insert(newIndex, mirror);
                                  });
                                },
                                itemBuilder: (context, index) {
                                  final mirror = mirrors[index];
                                  return ListTile(
                                    key: ValueKey('update-mirror-$mirror'),
                                    dense: true,
                                    leading: SizedBox(
                                      width: 28,
                                      child: Text(
                                        '${index + 1}',
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                    title: Text(
                                      mirror == defaultGithubMirrorTemplate
                                          ? 'KKGitHub'
                                          : mirror,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                    subtitle:
                                        mirror == defaultGithubMirrorTemplate
                                        ? Text(mirror)
                                        : null,
                                    trailing: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        IconButton(
                                          tooltip: '删除镜像',
                                          onPressed: () {
                                            setDialogState(
                                              () => mirrors.removeAt(index),
                                            );
                                          },
                                          icon: const Icon(
                                            Icons.delete_outline_rounded,
                                          ),
                                        ),
                                        ReorderableDragStartListener(
                                          index: index,
                                          child: const Padding(
                                            padding: EdgeInsets.all(12),
                                            child: Icon(
                                              Icons.drag_handle_rounded,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                      ),
                    ],
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(dialogContext).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(
                    dialogContext,
                  ).pop((source: source, mirrors: List<String>.from(mirrors))),
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );
    if (!mounted || result == null) return;

    setState(() {
      _draft['update_mirror'] = result.source;
      _draft[_updateMirrorsKey] = result.mirrors;
      _markDraftChanged();
    });
    _scheduleAutoSave();
  }

  Widget _buildUpdateSettings() {
    return Column(
      children: [
        _SoftwareVersionInfo(
          settings: widget.settings,
          onReleaseNotes: () => _openExternalUrl(_releaseNotesUrl),
        ),
        const SizedBox(height: 12),
        _SettingsPanel(
          title: '更新通道',
          subtitle: '选择稳定版或预览版更新来源',
          icon: Icons.commit_rounded,
          child: _SettingsMenuButton<String>(
            value: _string('update_channel', 'Preview'),
            options: const [
              _SettingsOption<String>(value: 'Release', label: '稳定版'),
              _SettingsOption<String>(value: 'Preview', label: '预览版'),
            ],
            onChanged: (value) => _set('update_channel', value),
          ),
        ),
        _SettingsPanel(
          title: '镜像源',
          subtitle: '管理国内镜像的增删与下载顺序',
          icon: Icons.public_rounded,
          child: _SummaryDialogButton(
            summary: _updateSourceSummary(),
            onPressed: _showUpdateSourcesDialog,
          ),
        ),
        _SettingsPanel(
          title: '检查更新',
          subtitle: '立即扫描所选通道的最新 Windows 版本',
          icon: Icons.system_update_alt_rounded,
          showDivider: false,
          child: Align(
            alignment: Alignment.centerRight,
            child: FilledButton(
              onPressed: _checkingForUpdates ? null : _checkForUpdates,
              child: _checkingForUpdates
                  ? const SizedBox.square(
                      dimension: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('检查更新'),
            ),
          ),
        ),
      ],
    );
  }

  Future<void> _checkForUpdates() async {
    if (_checkingForUpdates) return;
    setState(() => _checkingForUpdates = true);
    try {
      await _save();
      if (!mounted) return;
      await widget.onCheckForUpdates(
        channel: _string('update_channel', 'Preview'),
        mirror: _string('update_mirror', 'KKGitHub'),
        mirrorTemplates: _updateMirrors(),
      );
    } finally {
      if (mounted) setState(() => _checkingForUpdates = false);
    }
  }

  Future<void> _save() async {
    _autoSaveTimer?.cancel();
    if (_saving || !_dirty) return;
    final saveRevision = _draftRevision;
    setState(() {
      _saving = true;
    });
    var saved = false;
    try {
      _draft['package'] = _packageController.text.trim();
      if (_detectionDependenciesReady &&
          widget.settings?.gpuAvailable != true) {
        _draft['use_fp16'] = false;
      }
      await widget.onSaveSettings(Map<String, dynamic>.from(_draft));
      saved = true;
    } catch (_) {
      saved = false;
    } finally {
      if (mounted) {
        setState(() {
          _saving = false;
          if (saved && _draftRevision == saveRevision) {
            _dirty = false;
          }
        });
        if (_draftRevision != saveRevision) {
          _scheduleAutoSave();
        }
      }
    }
  }

  void _set(String key, dynamic value) {
    setState(() {
      _draft[key] = value;
      _markDraftChanged();
    });
    _scheduleAutoSave();
  }

  bool _bool(String key, [bool fallback = false]) {
    return _boolSetting(_draft, key, fallback);
  }

  double _double(String key, [double fallback = 0]) {
    return _doubleSetting(_draft, key, fallback);
  }

  int _int(String key, [int fallback = 0]) {
    return _intSetting(_draft, key, fallback);
  }

  int _undoStepsSetting(Map<String, dynamic> values) {
    return _normalizeUndoSteps(_intSetting(values, 'undo_steps', 200));
  }

  int _normalizeUndoSteps(int value) {
    final clamped = value.clamp(10, 200).toInt();
    return ((clamped / 10).round() * 10).clamp(10, 200).toInt();
  }

  String _string(String key, [String fallback = '']) {
    return _stringSetting(_draft, key, fallback);
  }

  List<String> _stringList(String key) {
    return _stringListSetting(_draft, key, const <String>[]);
  }

  String _favoritePhotoExportMode() {
    return _favoritePhotoExportModeSetting(_draft);
  }

  String _favoritePhotoExportModeSetting(Map<String, dynamic> values) {
    final value = _stringSetting(
      values,
      'favorite_photo_export_mode',
      _favoritePhotoExportAsk,
    );
    return switch (value) {
      _favoritePhotoExportAlways => _favoritePhotoExportAlways,
      _favoritePhotoExportNever => _favoritePhotoExportNever,
      _ => _favoritePhotoExportAsk,
    };
  }

  String? _validModelValue(
    String value,
    List<ModelInfo> models, {
    bool allowEmpty = false,
  }) {
    if (allowEmpty && value.isEmpty) return '';
    if (models.any((model) => model.path == value)) return value;
    if (allowEmpty) return '';
    return models.isEmpty ? null : models.first.path;
  }

  String? _modelClassesSelectionPath() {
    final classificationModel = _validModelValue(
      _string('selected_classification_model'),
      widget.settings?.availableClassificationModels ?? const <ModelInfo>[],
      allowEmpty: true,
    );
    if (classificationModel != null && classificationModel.isNotEmpty) {
      return classificationModel;
    }
    final detectionModel = _validModelValue(
      _string('selected_model'),
      widget.settings?.availableModels ?? const <ModelInfo>[],
      allowEmpty: true,
    );
    return detectionModel == null || detectionModel.isEmpty
        ? null
        : detectionModel;
  }
}

class _SettingsOption<T> {
  const _SettingsOption({required this.value, required this.label});

  final T value;
  final String label;
}

class _SettingsMenuButton<T> extends StatelessWidget {
  const _SettingsMenuButton({
    required this.value,
    required this.options,
    required this.onChanged,
    this.placeholder = '请选择',
    this.minMenuWidth = 180,
    this.maxMenuWidth = 360,
    this.enabled = true,
  });

  final T? value;
  final List<_SettingsOption<T>> options;
  final ValueChanged<T> onChanged;
  final String placeholder;
  final double minMenuWidth;
  final double maxMenuWidth;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final selectedOption = _selectedOption;
    final label = selectedOption?.label ?? placeholder;
    final canOpen = enabled && options.isNotEmpty;

    return Align(
      alignment: Alignment.centerRight,
      child: MenuAnchor(
        style: appMenuStyle(context, minWidth: minMenuWidth),
        menuChildren: [
          for (final option in options)
            MenuItemButton(
              style: appMenuItemStyle(
                context,
                selected: option.value == value,
                minWidth: minMenuWidth,
              ),
              leadingIcon: option.value == value
                  ? const Icon(Icons.check_rounded)
                  : const SizedBox(width: 24),
              onPressed: () => onChanged(option.value),
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  minWidth: minMenuWidth,
                  maxWidth: maxMenuWidth,
                ),
                child: Text(option.label, overflow: TextOverflow.ellipsis),
              ),
            ),
        ],
        builder: (context, controller, child) {
          return TextButton(
            onPressed: canOpen
                ? () {
                    if (controller.isOpen) {
                      controller.close();
                    } else {
                      controller.open();
                    }
                  }
                : null,
            style: TextButton.styleFrom(
              foregroundColor: scheme.primary,
              disabledForegroundColor: scheme.onSurfaceVariant,
              minimumSize: Size.zero,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
              alignment: Alignment.centerRight,
              textStyle: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 320),
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.end,
              ),
            ),
          );
        },
      ),
    );
  }

  _SettingsOption<T>? get _selectedOption {
    for (final option in options) {
      if (option.value == value) return option;
    }
    return null;
  }
}

class _SummaryDialogButton extends StatelessWidget {
  const _SummaryDialogButton({required this.summary, required this.onPressed});

  final String summary;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return TextButton(
      onPressed: onPressed,
      style: TextButton.styleFrom(
        foregroundColor: scheme.primary,
        disabledForegroundColor: scheme.onSurfaceVariant,
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
        alignment: Alignment.centerRight,
        textStyle: Theme.of(
          context,
        ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 320),
        child: Text(
          summary,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.end,
        ),
      ),
    );
  }
}

class _SettingsPanel extends StatelessWidget {
  const _SettingsPanel({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.child,
    this.showDivider = true,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Widget child;
  final bool showDivider;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Column(
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              final label = Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(icon, size: 20, color: scheme.primary),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          title,
                          style: textTheme.titleSmall?.copyWith(
                            color: scheme.primary,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          subtitle,
                          style: textTheme.bodySmall?.copyWith(
                            color: scheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              );

              if (constraints.maxWidth < 720) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [label, const SizedBox(height: 12), child],
                );
              }

              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 4, child: label),
                  const SizedBox(width: 28),
                  Expanded(
                    flex: 5,
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: child,
                    ),
                  ),
                ],
              );
            },
          ),
          if (showDivider)
            Padding(
              padding: const EdgeInsets.only(top: 16),
              child: Divider(height: 1, color: scheme.outlineVariant),
            ),
        ],
      ),
    );
  }
}

class _LabeledSlider extends StatelessWidget {
  const _LabeledSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
    this.valueLabel,
    this.enabled = true,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;
  final String? valueLabel;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final display = valueLabel ?? value.toStringAsFixed(2);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 110, child: Text(label)),
          Expanded(
            child: Slider(
              value: value.clamp(min, max).toDouble(),
              min: min,
              max: max,
              divisions: divisions,
              label: display,
              onChanged: enabled ? onChanged : null,
            ),
          ),
          SizedBox(width: 56, child: Text(display, textAlign: TextAlign.end)),
        ],
      ),
    );
  }
}

class _AppearanceControls extends StatelessWidget {
  const _AppearanceControls({
    required this.themeNotifier,
    required this.onUpdateTheme,
  });

  final ValueNotifier<ThemeSettings> themeNotifier;
  final ValueChanged<ThemeSettings> onUpdateTheme;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeSettings>(
      valueListenable: themeNotifier,
      builder: (context, settings, _) {
        final paletteDisabled = settings.useDynamicColor;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _SettingsPanel(
              title: '主题模式',
              subtitle: '选择浅色、深色或跟随系统',
              icon: Icons.brightness_6_rounded,
              child: SegmentedButton<ThemeMode>(
                segments: const [
                  ButtonSegment(
                    value: ThemeMode.light,
                    icon: Icon(Icons.light_mode_rounded),
                    label: Text('浅色'),
                  ),
                  ButtonSegment(
                    value: ThemeMode.dark,
                    icon: Icon(Icons.dark_mode_rounded),
                    label: Text('深色'),
                  ),
                  ButtonSegment(
                    value: ThemeMode.system,
                    icon: Icon(Icons.brightness_auto_rounded),
                    label: Text('自动'),
                  ),
                ],
                selected: {settings.themeMode},
                onSelectionChanged: (selection) {
                  if (selection.isEmpty) return;
                  onUpdateTheme(settings.copyWith(themeMode: selection.first));
                },
              ),
            ),
            _SettingsPanel(
              title: '使用系统动态颜色',
              subtitle: '开启后调色板由系统主题接管',
              icon: Icons.color_lens_rounded,
              child: Align(
                alignment: Alignment.centerRight,
                child: Switch(
                  value: settings.useDynamicColor,
                  onChanged: (value) =>
                      onUpdateTheme(settings.copyWith(useDynamicColor: value)),
                ),
              ),
            ),
            _SettingsPanel(
              title: paletteDisabled ? '调色板已由系统接管' : '调色板',
              subtitle: '选择应用主色调',
              icon: Icons.palette_rounded,
              showDivider: false,
              child: Wrap(
                alignment: WrapAlignment.end,
                spacing: 12,
                runSpacing: 12,
                children: kSeedColorOptions.map((option) {
                  final selected =
                      !paletteDisabled &&
                      settings.seedColor.toARGB32() == option.color.toARGB32();
                  return Tooltip(
                    message: option.label,
                    child: _ColorSwatch(
                      color: option.color,
                      selected: selected,
                      disabled: paletteDisabled,
                      onTap: paletteDisabled
                          ? null
                          : () => onUpdateTheme(
                              settings.copyWith(seedColor: option.color),
                            ),
                    ),
                  );
                }).toList(),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _SoftwareVersionInfo extends StatelessWidget {
  const _SoftwareVersionInfo({
    required this.settings,
    required this.onReleaseNotes,
  });

  final NeriSettings? settings;
  final VoidCallback onReleaseNotes;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final version = settings?.appVersion.isNotEmpty == true
        ? settings!.appVersion
        : '未读取到版本信息';

    return Card.outlined(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(Icons.verified_rounded, color: scheme.primary),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('当前版本', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 4),
                  Text(
                    version,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
            TextButton.icon(
              onPressed: onReleaseNotes,
              icon: const Icon(Icons.article_outlined),
              label: const Text('更新说明'),
            ),
          ],
        ),
      ),
    );
  }
}

class _DebugInfoPanel extends StatefulWidget {
  const _DebugInfoPanel({
    required this.settings,
    required this.apiClient,
    required this.debugModeSaving,
    required this.onDebugModeChanged,
    required this.onShowMessage,
  });

  final NeriSettings? settings;
  final NeriApiClient apiClient;
  final bool debugModeSaving;
  final ValueChanged<bool> onDebugModeChanged;
  final ValueChanged<String> onShowMessage;

  @override
  State<_DebugInfoPanel> createState() => _DebugInfoPanelState();
}

class _DebugInfoPanelState extends State<_DebugInfoPanel> {
  late Future<RuntimeDiagnostics> _runtimeFuture;
  bool _simulatingBackendCrash = false;

  @override
  void initState() {
    super.initState();
    _runtimeFuture = widget.apiClient.fetchRuntimeDiagnostics();
  }

  void _refreshRuntime() {
    setState(() {
      _runtimeFuture = widget.apiClient.fetchRuntimeDiagnostics();
    });
  }

  Future<void> _copyDebugInfo() async {
    RuntimeDiagnostics? runtime;
    try {
      runtime = await _runtimeFuture;
    } catch (_) {}
    await Clipboard.setData(
      ClipboardData(
        text: _debugInfoText(
          settings: widget.settings,
          apiClient: widget.apiClient,
          debugModeEnabled: true,
          runtime: runtime,
        ),
      ),
    );
    widget.onShowMessage('调试信息已复制');
  }

  void _showInstalledPackages(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => _InstalledPackagesDialog(
        apiClient: widget.apiClient,
        onShowMessage: widget.onShowMessage,
      ),
    );
  }

  void _showSoftwareLogs(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => _SoftwareLogsDialog(onShowMessage: widget.onShowMessage),
    );
  }

  Future<void> _simulateFrontendCrash() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('模拟前端崩溃'),
        content: const Text('前端进程会以非零退出码结束，程序将记录日志并在重启后显示崩溃原因。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('模拟崩溃'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    final watchdogStarted = await CrashWatchdog.ensureStarted();
    if (!mounted) return;
    if (!watchdogStarted) {
      widget.onShowMessage('无法启动前端崩溃监听器，已取消模拟崩溃。请查看软件日志。');
      return;
    }
    await Future<void>.delayed(const Duration(milliseconds: 250));
    if (!mounted) return;
    CrashReporter.simulateFrontendCrash();
  }

  Future<void> _simulateBackendCrash() async {
    if (_simulatingBackendCrash) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('模拟后端崩溃'),
        content: const Text('Python 后端会以非零退出码结束，程序将记录日志并显示崩溃原因。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('模拟崩溃'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() => _simulatingBackendCrash = true);
    final previousReport = CrashReporter.latestReport.value;
    try {
      await widget.apiClient.simulateBackendCrash();
      if (!mounted) return;
      widget.onShowMessage('已触发后端崩溃模拟');
      await Future<void>.delayed(const Duration(milliseconds: 2500));
      if (!mounted) return;
      if (identical(CrashReporter.latestReport.value, previousReport)) {
        CrashReporter.recordBackendCrash(
          exitCode: 86,
          logPath: 'logs/${CrashReporter.crashLogFileName}',
          outputTail: '调试模式模拟 Python 后端崩溃。',
        );
      }
    } catch (error) {
      if (!mounted) return;
      widget.onShowMessage('触发后端崩溃模拟失败：$error');
    } finally {
      if (mounted) {
        setState(() => _simulatingBackendCrash = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final missing =
        widget.settings?.missingYoloDependencies ?? const <String>[];
    final dependenciesLabel = missing.isEmpty ? '完整' : missing.join('、');

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          value: true,
          onChanged: widget.debugModeSaving ? null : widget.onDebugModeChanged,
          secondary: const Icon(Icons.bug_report_rounded),
          title: const Text('调试模式'),
          subtitle: const Text('已开启'),
        ),
        Divider(height: 1, color: scheme.outlineVariant),
        const SizedBox(height: 12),
        _DebugFactRow(label: '后端地址', value: widget.apiClient.baseUrl),
        _DebugFactRow(
          label: '运行平台',
          value:
              '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
        ),
        _DebugFactRow(label: '前端版本', value: _frontendVersion),
        FutureBuilder<RuntimeDiagnostics>(
          future: _runtimeFuture,
          builder: (context, snapshot) {
            final fallback = widget.settings?.appVersion ?? '';
            final value = switch (snapshot.connectionState) {
              ConnectionState.done when snapshot.hasData =>
                snapshot.data!.backendVersion.isNotEmpty
                    ? snapshot.data!.backendVersion
                    : fallback,
              ConnectionState.done when snapshot.hasError =>
                fallback.isEmpty ? '读取失败：${snapshot.error}' : fallback,
              _ => fallback.isEmpty ? '正在读取后端版本...' : fallback,
            };
            return _DebugFactRow(label: '后端版本', value: value);
          },
        ),
        FutureBuilder<RuntimeDiagnostics>(
          future: _runtimeFuture,
          builder: (context, snapshot) {
            final value = switch (snapshot.connectionState) {
              ConnectionState.done when snapshot.hasData => _runtimeGpuStatus(
                snapshot.data!,
              ),
              ConnectionState.done when snapshot.hasError =>
                '读取失败：${snapshot.error}',
              _ => '正在读取显卡与 PyTorch 信息...',
            };
            return _DebugFactRow(
              label: 'GPU 状态',
              value: value,
              trailing: IconButton(
                tooltip: '刷新',
                onPressed: _refreshRuntime,
                icon: const Icon(Icons.refresh_rounded),
              ),
            );
          },
        ),
        _DebugFactRow(label: 'YOLO 依赖', value: dependenciesLabel),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            FilledButton.tonalIcon(
              onPressed: _copyDebugInfo,
              icon: const Icon(Icons.content_copy_rounded),
              label: const Text('复制调试信息'),
            ),
            OutlinedButton.icon(
              onPressed: () => _showInstalledPackages(context),
              icon: const Icon(Icons.inventory_2_outlined),
              label: const Text('已安装库'),
            ),
            OutlinedButton.icon(
              onPressed: () => _showSoftwareLogs(context),
              icon: const Icon(Icons.receipt_long_outlined),
              label: const Text('软件日志'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Divider(height: 1, color: scheme.outlineVariant),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            OutlinedButton.icon(
              onPressed: _simulateFrontendCrash,
              icon: const Icon(Icons.warning_amber_rounded),
              label: const Text('模拟前端崩溃'),
            ),
            OutlinedButton.icon(
              onPressed: _simulatingBackendCrash ? null : _simulateBackendCrash,
              icon: _simulatingBackendCrash
                  ? const SizedBox.square(
                      dimension: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.dangerous_outlined),
              label: const Text('模拟后端崩溃'),
            ),
          ],
        ),
      ],
    );
  }
}

class _DebugFactRow extends StatelessWidget {
  const _DebugFactRow({
    required this.label,
    required this.value,
    this.trailing,
  });

  final String label;
  final String value;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 96,
            child: Text(
              label,
              style: textTheme.bodyMedium?.copyWith(
                color: scheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: SelectableText(
              value,
              style: textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          if (trailing != null) ...[const SizedBox(width: 8), trailing!],
        ],
      ),
    );
  }
}

class _InstalledPackagesDialog extends StatefulWidget {
  const _InstalledPackagesDialog({
    required this.apiClient,
    required this.onShowMessage,
  });

  final NeriApiClient apiClient;
  final ValueChanged<String> onShowMessage;

  @override
  State<_InstalledPackagesDialog> createState() =>
      _InstalledPackagesDialogState();
}

class _InstalledPackagesDialogState extends State<_InstalledPackagesDialog> {
  late final Future<List<InstalledPackageInfo>> _packagesFuture;

  @override
  void initState() {
    super.initState();
    _packagesFuture = widget.apiClient.fetchInstalledPackages();
  }

  Future<void> _copyPackages(List<InstalledPackageInfo> packages) async {
    final text = packages
        .map((package) => '${package.name}==${package.version}')
        .join('\n');
    await Clipboard.setData(ClipboardData(text: text));
    widget.onShowMessage('已安装库列表已复制');
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('已安装库'),
      content: SizedBox(
        width: 560,
        height: 520,
        child: FutureBuilder<List<InstalledPackageInfo>>(
          future: _packagesFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return _DebugDialogMessage(
                icon: Icons.error_outline_rounded,
                message: '读取已安装库失败：${snapshot.error}',
              );
            }

            final packages = snapshot.data ?? const <InstalledPackageInfo>[];
            if (packages.isEmpty) {
              return const _DebugDialogMessage(
                icon: Icons.inventory_2_outlined,
                message: '未读取到已安装库。',
              );
            }

            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    Expanded(child: Text('${packages.length} 个库')),
                    TextButton.icon(
                      onPressed: () => _copyPackages(packages),
                      icon: const Icon(Icons.copy_rounded),
                      label: const Text('复制全部'),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: ListView.separated(
                    itemCount: packages.length,
                    separatorBuilder: (_, _) => const Divider(height: 1),
                    itemBuilder: (context, index) {
                      final package = packages[index];
                      return ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        title: Text(
                          package.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        trailing: SelectableText(package.version),
                      );
                    },
                  ),
                ),
              ],
            );
          },
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('关闭'),
        ),
      ],
    );
  }
}

class _SoftwareLogsDialog extends StatefulWidget {
  const _SoftwareLogsDialog({required this.onShowMessage});

  final ValueChanged<String> onShowMessage;

  @override
  State<_SoftwareLogsDialog> createState() => _SoftwareLogsDialogState();
}

class _SoftwareLogsDialogState extends State<_SoftwareLogsDialog> {
  static const _logPreviewMaxBytes = 32000;

  final _logReader = LocalDebugLogReader();
  late final Future<List<DebugLogInfo>> _logsFuture;
  DebugLogInfo? _selectedLog;
  Future<DebugLogContent>? _contentFuture;

  @override
  void initState() {
    super.initState();
    _logsFuture = _logReader.listLogs();
  }

  void _selectLog(DebugLogInfo log) {
    setState(() {
      _selectedLog = log;
      _contentFuture = _logReader.readLog(log, maxBytes: _logPreviewMaxBytes);
    });
  }

  void _previewSelectedLog() {
    final log = _selectedLog;
    if (log == null) return;
    setState(() {
      _contentFuture = _logReader.readLog(log, maxBytes: _logPreviewMaxBytes);
    });
  }

  Future<void> _copyLog(DebugLogContent log) async {
    await Clipboard.setData(ClipboardData(text: log.content));
    widget.onShowMessage('日志内容已复制');
  }

  Future<void> _openLogExternally(DebugLogInfo log) async {
    try {
      if (Platform.isWindows) {
        await Process.run('cmd', ['/c', 'start', '', log.path]);
      } else if (Platform.isMacOS) {
        await Process.run('open', [log.path]);
      } else if (Platform.isLinux) {
        await Process.run('xdg-open', [log.path]);
      } else {
        await Clipboard.setData(ClipboardData(text: log.path));
        widget.onShowMessage('日志路径已复制');
        return;
      }
      widget.onShowMessage('已交给系统打开日志');
    } catch (error) {
      await Clipboard.setData(ClipboardData(text: log.path));
      widget.onShowMessage('打开日志失败，已复制路径：$error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('软件日志'),
      content: SizedBox(
        width: 820,
        height: 560,
        child: FutureBuilder<List<DebugLogInfo>>(
          future: _logsFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return _DebugDialogMessage(
                icon: Icons.error_outline_rounded,
                message: '读取软件日志失败：${snapshot.error}',
              );
            }

            final logs = snapshot.data ?? const <DebugLogInfo>[];
            if (logs.isEmpty) {
              return const _DebugDialogMessage(
                icon: Icons.receipt_long_outlined,
                message: '未找到软件日志。',
              );
            }
            if (_selectedLog == null) {
              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted && _selectedLog == null) {
                  _selectLog(logs.first);
                }
              });
            }

            return Row(
              children: [
                SizedBox(
                  width: 260,
                  child: ListView.separated(
                    itemCount: logs.length,
                    separatorBuilder: (_, _) => const Divider(height: 1),
                    itemBuilder: (context, index) {
                      final log = logs[index];
                      return ListTile(
                        dense: true,
                        selected: log.path == _selectedLog?.path,
                        contentPadding: EdgeInsets.zero,
                        title: Text(
                          log.name,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        subtitle: Text(_formatBytes(log.sizeBytes)),
                        onTap: () => _selectLog(log),
                      );
                    },
                  ),
                ),
                const VerticalDivider(width: 24),
                Expanded(child: _buildSelectedLogContent()),
              ],
            );
          },
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('关闭'),
        ),
      ],
    );
  }

  Widget _buildSelectedLogContent() {
    final selectedLog = _selectedLog;
    if (selectedLog == null) {
      return const _DebugDialogMessage(
        icon: Icons.receipt_long_outlined,
        message: '选择一个日志查看内容。',
      );
    }

    final future = _contentFuture;
    if (future == null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _SelectedLogHeader(
            log: selectedLog,
            onPreview: _previewSelectedLog,
            onOpenExternal: () => _openLogExternally(selectedLog),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: _DebugDialogMessage(
              icon: Icons.article_outlined,
              message: '正在准备日志预览，也可以用系统打开完整日志。',
            ),
          ),
        ],
      );
    }

    return FutureBuilder<DebugLogContent>(
      future: future,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return _DebugDialogMessage(
            icon: Icons.error_outline_rounded,
            message: '读取日志内容失败：${snapshot.error}',
          );
        }

        final log = snapshot.data;
        if (log == null) {
          return const _DebugDialogMessage(
            icon: Icons.receipt_long_outlined,
            message: '日志内容为空。',
          );
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _SelectedLogHeader(
              log: log,
              onPreview: _previewSelectedLog,
              onOpenExternal: () => _openLogExternally(log),
              onCopy: () => _copyLog(log),
            ),
            if (log.truncated)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  '内容过长，仅显示末尾部分。',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Theme.of(context).colorScheme.error,
                  ),
                ),
              ),
            const SizedBox(height: 8),
            Expanded(
              child: Scrollbar(
                child: SingleChildScrollView(
                  child: SelectableText(
                    log.content.isEmpty ? '日志为空。' : log.content,
                    style: const TextStyle(
                      fontFamily: 'Consolas',
                      fontSize: 12,
                    ),
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _SelectedLogHeader extends StatelessWidget {
  const _SelectedLogHeader({
    required this.log,
    required this.onPreview,
    required this.onOpenExternal,
    this.onCopy,
  });

  final DebugLogInfo log;
  final VoidCallback onPreview;
  final VoidCallback onOpenExternal;
  final VoidCallback? onCopy;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          log.name,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: textTheme.titleSmall,
        ),
        const SizedBox(height: 4),
        Text(
          '${_formatBytes(log.sizeBytes)}'
          '${log.modifiedAt == null ? '' : ' · ${_formatLogTime(log.modifiedAt)}'}',
          style: textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            FilledButton.tonalIcon(
              onPressed: onPreview,
              icon: const Icon(Icons.segment_rounded),
              label: const Text('预览尾部'),
            ),
            OutlinedButton.icon(
              onPressed: onOpenExternal,
              icon: const Icon(Icons.open_in_new_rounded),
              label: const Text('用系统打开'),
            ),
            if (onCopy != null)
              TextButton.icon(
                onPressed: onCopy,
                icon: const Icon(Icons.copy_rounded),
                label: const Text('复制预览'),
              ),
          ],
        ),
      ],
    );
  }
}

class _DebugDialogMessage extends StatelessWidget {
  const _DebugDialogMessage({required this.icon, required this.message});

  final IconData icon;
  final String message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 32, color: scheme.onSurfaceVariant),
          const SizedBox(height: 12),
          Text(message, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

class _AboutInfo extends StatefulWidget {
  const _AboutInfo({
    required this.settings,
    required this.debugModeEnabled,
    required this.debugModeSaving,
    required this.onDebugModeChanged,
    required this.onOpenUrl,
    required this.onShowMessage,
  });

  final NeriSettings? settings;
  final bool debugModeEnabled;
  final bool debugModeSaving;
  final ValueChanged<bool> onDebugModeChanged;
  final ValueChanged<String> onOpenUrl;
  final ValueChanged<String> onShowMessage;

  @override
  State<_AboutInfo> createState() => _AboutInfoState();
}

class _AboutInfoState extends State<_AboutInfo> {
  int _versionTapCount = 0;
  Timer? _versionTapResetTimer;

  @override
  void didUpdateWidget(covariant _AboutInfo oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.debugModeEnabled != widget.debugModeEnabled) {
      _versionTapResetTimer?.cancel();
      _versionTapCount = 0;
    }
  }

  @override
  void dispose() {
    _versionTapResetTimer?.cancel();
    super.dispose();
  }

  void _handleVersionTap() {
    if (widget.debugModeSaving) {
      widget.onShowMessage('调试模式正在保存');
      return;
    }
    if (widget.debugModeEnabled) {
      widget.onShowMessage('调试模式已开启');
      return;
    }

    final nextCount = _versionTapCount + 1;
    final remaining = _debugTapThreshold - nextCount;
    _versionTapResetTimer?.cancel();

    if (remaining <= 0) {
      setState(() => _versionTapCount = 0);
      widget.onDebugModeChanged(true);
      return;
    }

    setState(() => _versionTapCount = nextCount);
    if (remaining <= 2) {
      widget.onShowMessage('再点击 $remaining 次开启调试模式');
    }
    _versionTapResetTimer = Timer(_debugTapResetDuration, () {
      if (!mounted) return;
      setState(() => _versionTapCount = 0);
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;
    final settings = widget.settings;
    final version = settings?.appVersion.isNotEmpty == true
        ? settings!.appVersion
        : '未读取到版本信息';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 4, 4, 12),
          child: Row(
            children: [
              _AboutLogo(color: scheme.primary),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Neri',
                      style: textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'NERI Enables Rapid Identification',
                      style: textTheme.bodyMedium?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        Card.outlined(
          margin: EdgeInsets.zero,
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.tag_rounded),
                title: const Text('版本号'),
                subtitle: Text(version),
                trailing: widget.debugModeEnabled
                    ? Tooltip(
                        message: '调试模式已开启',
                        child: Icon(
                          Icons.bug_report_rounded,
                          color: scheme.primary,
                        ),
                      )
                    : null,
                onTap: _handleVersionTap,
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.article_outlined),
                title: const Text('更新说明'),
                subtitle: const Text('查看版本更新记录'),
                trailing: const Icon(Icons.open_in_new_rounded),
                onTap: () => widget.onOpenUrl(_releaseNotesUrl),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.language_rounded),
                title: const Text('官网'),
                subtitle: const Text('访问 Neri 官方网站'),
                trailing: const Icon(Icons.open_in_new_rounded),
                onTap: () => widget.onOpenUrl(_officialWebsiteUrl),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.feedback_outlined),
                title: const Text('反馈建议'),
                subtitle: const Text('提交问题或功能建议'),
                trailing: const Icon(Icons.open_in_new_rounded),
                onTap: () => widget.onOpenUrl(_feedbackUrl),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.code_rounded),
                title: const Text('源代码'),
                subtitle: const Text('查看 GitHub 项目仓库'),
                trailing: const Icon(Icons.open_in_new_rounded),
                onTap: () => widget.onOpenUrl(_sourceCodeUrl),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _AboutLogo extends StatelessWidget {
  const _AboutLogo({required this.color});

  final Color color;

  @override
  Widget build(BuildContext context) {
    final logoFile = _resolveLogoFile();
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.file(
        logoFile,
        width: 56,
        height: 56,
        fit: BoxFit.contain,
        errorBuilder: (context, error, stackTrace) {
          return Icon(Icons.camera_outdoor_rounded, size: 48, color: color);
        },
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
}

class _MutedText extends StatelessWidget {
  const _MutedText(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(
        text,
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
          color: Theme.of(context).colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }
}

class _ColorSwatch extends StatelessWidget {
  const _ColorSwatch({
    required this.color,
    required this.selected,
    required this.disabled,
    required this.onTap,
  });

  final Color color;
  final bool selected;
  final bool disabled;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = ColorScheme.fromSeed(
      seedColor: color,
      brightness: theme.brightness,
    );
    final tileColor = selected
        ? Color.alphaBlend(
            scheme.primary.withValues(alpha: 0.10),
            theme.colorScheme.surface,
          )
        : theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5);
    final checkColor =
        ThemeData.estimateBrightnessForColor(scheme.primary) == Brightness.dark
        ? Colors.white
        : theme.colorScheme.onSurface;

    return GestureDetector(
      onTap: onTap,
      child: TweenAnimationBuilder<double>(
        tween: Tween<double>(end: selected ? 1 : 0),
        duration: const Duration(milliseconds: 260),
        curve: Curves.easeOutBack,
        builder: (context, selectedAmount, _) {
          final selectedProgress = selectedAmount.clamp(0.0, 1.0).toDouble();
          final ringAlpha = (0.12 + selectedProgress * 0.18)
              .clamp(0.0, 1.0)
              .toDouble();
          final palette = Transform.scale(
            scale: 1 + selectedProgress * 0.08,
            child: SizedBox(
              width: 46,
              height: 46,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  CustomPaint(
                    size: const Size.square(46),
                    painter: _PaletteCirclePainter(scheme),
                  ),
                  Opacity(
                    opacity: selectedProgress,
                    child: Transform.scale(
                      scale: 0.72 + selectedProgress * 0.28,
                      child: Container(
                        width: 29,
                        height: 29,
                        decoration: BoxDecoration(
                          color: Color.alphaBlend(
                            scheme.primary.withValues(alpha: 0.08),
                            theme.colorScheme.surface,
                          ),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          Icons.check_rounded,
                          color: checkColor,
                          size: 20,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );

          return AnimatedOpacity(
            duration: const Duration(milliseconds: 200),
            opacity: disabled ? 0.35 : 1.0,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              curve: Curves.easeOutCubic,
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: tileColor,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: scheme.primary.withValues(
                    alpha: selectedProgress == 0 ? 0.0 : ringAlpha,
                  ),
                  width: 1 + selectedProgress * 2,
                ),
                boxShadow: [
                  if (selectedProgress > 0)
                    BoxShadow(
                      color: scheme.primary.withValues(
                        alpha: 0.10 * selectedProgress,
                      ),
                      blurRadius: 12 * selectedProgress,
                      spreadRadius: 1 * selectedProgress,
                    ),
                ],
              ),
              alignment: Alignment.center,
              child: palette,
            ),
          );
        },
      ),
    );
  }
}

class _PaletteCirclePainter extends CustomPainter {
  const _PaletteCirclePainter(this.scheme);

  final ColorScheme scheme;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final paint = Paint()..isAntiAlias = true;

    canvas.drawArc(
      rect,
      math.pi,
      math.pi,
      true,
      paint..color = scheme.primaryContainer,
    );
    canvas.drawArc(
      rect,
      math.pi / 2,
      math.pi / 2,
      true,
      paint..color = scheme.tertiaryContainer,
    );
    canvas.drawArc(rect, 0, math.pi / 2, true, paint..color = scheme.primary);
  }

  @override
  bool shouldRepaint(covariant _PaletteCirclePainter oldDelegate) {
    return oldDelegate.scheme != scheme;
  }
}

String _debugInfoText({
  required NeriSettings? settings,
  required NeriApiClient apiClient,
  required bool debugModeEnabled,
  RuntimeDiagnostics? runtime,
}) {
  final missing = settings?.missingYoloDependencies ?? const <String>[];
  final backendVersion = runtime?.backendVersion.isNotEmpty == true
      ? runtime!.backendVersion
      : settings?.appVersion ?? '';
  return <String>[
    'Neri debug information',
    'Frontend version: $_frontendVersion',
    'Backend version: $backendVersion',
    'Backend URL: ${apiClient.baseUrl}',
    'Debug mode: $debugModeEnabled',
    'Platform: ${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
    'GPU status: ${runtime == null ? settings?.gpuAvailable ?? false : _runtimeGpuStatus(runtime)}',
    'Missing YOLO dependencies: ${missing.isEmpty ? 'none' : missing.join(', ')}',
    'Model directory: ${settings?.modelDirectory ?? ''}',
    'Classification model directory: ${settings?.classificationModelDirectory ?? ''}',
    'Detection model count: ${settings?.availableModels.length ?? 0}',
    'Classification model count: ${settings?.availableClassificationModels.length ?? 0}',
  ].join('\n');
}

String _runtimeGpuStatus(RuntimeDiagnostics runtime) {
  final pytorchVersion = runtime.pytorchInstalled
      ? 'PyTorch ${runtime.pytorchVersion?.isNotEmpty == true ? runtime.pytorchVersion : '未知版本'}'
      : 'PyTorch 未安装';
  final cudaVersion = runtime.pytorchCudaVersion?.isNotEmpty == true
      ? 'CUDA ${runtime.pytorchCudaVersion}'
      : null;
  final deviceNames = runtime.gpuDevices.isNotEmpty
      ? runtime.gpuDevices
      : runtime.hardwareGpus;
  final devices = deviceNames.isEmpty ? '未读取到显卡信息' : deviceNames.join('、');
  final status = runtime.gpuAvailable ? '可用' : '不可用';
  final parts = <String>[status, devices, pytorchVersion];
  if (cudaVersion != null) parts.add(cudaVersion);
  if (runtime.error?.isNotEmpty == true) {
    parts.add('诊断信息：${runtime.error}');
  }
  return parts.join(' · ');
}

String _formatBytes(int bytes) {
  if (bytes < 1024) return '$bytes B';
  const units = <String>['KB', 'MB', 'GB'];
  var value = bytes / 1024;
  var index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return '${value.toStringAsFixed(value >= 10 ? 1 : 2)} ${units[index]}';
}

String _formatLogTime(String? value) {
  if (value == null || value.isEmpty) return '';
  try {
    return DateTime.parse(value).toLocal().toString().split('.').first;
  } catch (_) {
    return value;
  }
}

bool _boolSetting(Map<String, dynamic> values, String key, bool fallback) {
  final value = values[key];
  if (value is bool) return value;
  if (value is String) return value.toLowerCase() == 'true';
  return fallback;
}

double _doubleSetting(
  Map<String, dynamic> values,
  String key,
  double fallback,
) {
  final value = values[key];
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value) ?? fallback;
  return fallback;
}

int _intSetting(Map<String, dynamic> values, String key, int fallback) {
  final value = values[key];
  if (value is int) return value;
  if (value is num) return value.round();
  if (value is String) return int.tryParse(value) ?? fallback;
  return fallback;
}

String _stringSetting(
  Map<String, dynamic> values,
  String key,
  String fallback,
) {
  final value = values[key];
  if (value is String) return value;
  return fallback;
}

List<String> _stringListSetting(
  Map<String, dynamic> values,
  String key,
  List<String> fallback,
) {
  final value = values[key];
  if (value is List) {
    return value.map((item) => item.toString()).toList();
  }
  return List<String>.from(fallback);
}

Map<String, int> _intMapSetting(Map<String, dynamic> values, String key) {
  final value = values[key];
  if (value is! Map) return const <String, int>{};
  return value.map((key, value) {
    final parsed = switch (value) {
      int number => number,
      num number => number.round(),
      String text => int.tryParse(text) ?? 0,
      _ => 0,
    };
    return MapEntry(key.toString(), parsed);
  });
}
