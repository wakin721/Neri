import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart';

import 'api_client.dart';
import 'models/job.dart';
import 'models/settings.dart';
import 'models/theme_settings.dart';
import 'screens/about_screen.dart';
import 'screens/preview_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/species_validation_screen.dart';
import 'screens/start_screen.dart';

const _lastInputPathKey = 'last_input_path';

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
  static const _pageTitles = <String>['开始界面', '图像预览', '物种校验', '设置', '关于'];

  final _inputController = TextEditingController();
  final Map<String, DetectionItem> _previewMetadataCache =
      <String, DetectionItem>{};
  final Set<String> _previewMetadataLoading = <String>{};

  NeriSettings? _settings;
  List<ProcessingJob> _jobs = const <ProcessingJob>[];
  List<DetectionItem> _previewItems = const <DetectionItem>[];
  Timer? _timer;
  Timer? _previewRefreshTimer;
  bool _loading = true;
  bool _submitting = false;
  bool _previewLoading = false;
  bool _enableDetection = false;
  bool _useFp16 = false;
  bool _previewShowDetections = true;
  bool _previewDetecting = false;
  bool _isMaximized = false;
  double _confidence = 0.25;
  double _iou = 0.45;
  double _previewConfidenceThreshold = 0.25;
  String _previewSpeciesFilter = previewAllSpeciesLabel;
  String? _selectedModelPath;
  String? _previewLoadedPath;
  int _selectedIndex = 0;
  int _selectedPreviewIndex = 0;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _initWindowState();

    try {
      MediaKit.ensureInitialized();
    } catch (_) {}

    _inputController.addListener(_schedulePreviewRefresh);
    _loadLastInputPath();
    _refresh();
    _timer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => _refresh(silent: true),
    );
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    _timer?.cancel();
    _previewRefreshTimer?.cancel();
    _inputController.dispose();
    widget.apiClient.close();
    super.dispose();
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

  void _showSnackBar(String message) {
    if (!mounted || message.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final messenger = ScaffoldMessenger.maybeOf(context);
      messenger?.hideCurrentSnackBar();
      messenger?.showSnackBar(SnackBar(content: Text(message)));
    });
  }

  void _schedulePreviewRefresh() {
    _previewRefreshTimer?.cancel();
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      if (!mounted) return;
      setState(() {
        _previewItems = const <DetectionItem>[];
        _previewMetadataCache.clear();
        _previewMetadataLoading.clear();
        _previewLoadedPath = null;
      });
      return;
    }

    unawaited(_saveLastInputPath(inputPath));
    _previewRefreshTimer = Timer(
      const Duration(milliseconds: 450),
      () => _refreshPreviewItems(),
    );
  }

  Future<void> _refreshPreviewItems({bool force = false}) async {
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) return;
    if (!force && _previewLoadedPath == inputPath && _previewItems.isNotEmpty) {
      return;
    }

    if (!mounted) return;
    setState(() => _previewLoading = true);

    try {
      final items = await widget.apiClient.fetchPreviewItems(
        inputPath: inputPath,
      );
      if (!mounted || _inputController.text.trim() != inputPath) return;
      setState(() {
        _previewItems = items;
        _previewMetadataCache.clear();
        _previewMetadataLoading.clear();
        _previewLoadedPath = inputPath;
        _selectedPreviewIndex = _safePreviewIndex(items);
      });
      if (items.isNotEmpty) {
        unawaited(_loadPreviewMetadata(items[_selectedPreviewIndex]));
      }
    } catch (error) {
      if (!mounted || _inputController.text.trim() != inputPath) return;
      _showSnackBar('无法读取输入文件夹预览：$error');
    } finally {
      if (mounted && _inputController.text.trim() == inputPath) {
        setState(() => _previewLoading = false);
      }
    }
  }

  Future<void> _loadPreviewMetadata(DetectionItem item) async {
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
      if (!mounted) return;
      setState(() => _previewMetadataCache[item.path] = fullItem);
    } catch (_) {
      // Fast preview data is still usable if per-file metadata is unavailable.
    } finally {
      _previewMetadataLoading.remove(item.path);
    }
  }

  DetectionItem _resolvedPreviewItem(DetectionItem item) {
    return _previewMetadataCache[item.path] ?? item;
  }

  Future<void> _refresh({bool silent = false}) async {
    if (!silent) {
      setState(() => _loading = true);
    }

    final firstLoad = _settings == null;
    try {
      final settings = await widget.apiClient.fetchSettings();
      final jobs = await widget.apiClient.listJobs();
      if (!mounted) return;
      setState(() {
        _settings = settings;
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
        if (firstLoad) {
          _confidence = _doubleSetting(settings, 'confidence', _confidence);
          _iou = _doubleSetting(settings, 'iou', _iou);
          _useFp16 =
              settings.gpuAvailable &&
              _boolSetting(settings, 'use_fp16', _useFp16);
        }
        _jobs = jobs;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _loading = false);
      if (!silent) _showSnackBar('无法连接 Python 后端：$error');
    }
  }

  Future<void> _saveAdvancedSettings(Map<String, dynamic> settings) async {
    try {
      final saved = await widget.apiClient.saveSettings(settings);
      if (!mounted) return;
      setState(() {
        _settings = saved;
        _selectedModelPath = saved.selectedModel ?? _selectedModelPath;
        _confidence = _doubleSetting(saved, 'confidence', _confidence);
        _iou = _doubleSetting(saved, 'iou', _iou);
        _useFp16 =
            saved.gpuAvailable && _boolSetting(saved, 'use_fp16', _useFp16);
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

  Future<void> _createJob() async {
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      _showSnackBar('请输入红外相机媒体文件夹路径');
      return;
    }

    setState(() => _submitting = true);
    try {
      await _saveLastInputPath(inputPath);
      await widget.apiClient.createJob(
        inputDir: inputPath,
        modelPath: _selectedModelPath,
        confidence: _confidence,
        iou: _iou,
        useFp16: _useFp16,
        enableDetection: _enableDetection,
      );
      await _refresh(silent: true);
      await _refreshPreviewItems(force: true);
      _showSnackBar('任务已提交');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('创建任务失败：$error');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _detectCurrentPreviewImage(DetectionItem item) async {
    setState(() => _previewDetecting = true);
    try {
      await widget.apiClient.createJob(
        inputDir: item.path,
        modelPath: _selectedModelPath,
        confidence: _previewConfidenceThreshold,
        iou: _iou,
        useFp16: _useFp16,
        enableDetection: true,
      );
      await _refresh(silent: true);
      await _refreshPreviewItems(force: true);
      if (mounted) setState(() => _selectedIndex = 1);
      _showSnackBar('已提交当前图像检测');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('检测当前图像失败：$error');
    } finally {
      if (mounted) setState(() => _previewDetecting = false);
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

    if (!mounted) return updated;
    setState(() {
      _previewMetadataCache[updated.path] = updated;
      _previewItems = _previewItems
          .map(
            (previewItem) =>
                previewItem.path == updated.path ? updated : previewItem,
          )
          .toList();
    });
    return updated;
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

  int _safePreviewIndex(List<DetectionItem> items) {
    if (items.isEmpty) return 0;
    if (_selectedPreviewIndex >= items.length) return items.length - 1;
    if (_selectedPreviewIndex < 0) return 0;
    return _selectedPreviewIndex;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(kToolbarHeight),
        child: DragToMoveArea(
          child: AppBar(
            title: Text(_pageTitles[_selectedIndex]),
            actions: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: Center(child: Text(_settings?.appTitle ?? 'Neri')),
              ),
              IconButton(
                tooltip: '刷新',
                onPressed: _loading
                    ? null
                    : () async {
                        await _refresh();
                        await _refreshPreviewItems(force: true);
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
                      overlayColor: WidgetStateProperty.resolveWith((states) {
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
                        return Theme.of(context).colorScheme.onSurfaceVariant;
                      }),
                    ),
                    onPressed: () => windowManager.close(),
                  );
                },
              ),
              const SizedBox(width: 8),
            ],
          ),
        ),
      ),
      body: Column(
        children: [
          // 标题栏下方的全局进度条
          // 当全局加载、预览加载或提交任务时显示
          if (_loading || _previewLoading || _submitting)
            const LinearProgressIndicator()
          else
            const SizedBox(height: 4), // 占位高度为 4，与进度条默认高度一致，防止显示/隐藏时页面抖动
            
          // 原有的主体内容区域
          Expanded(
            child: Row(
              children: [
                NavigationRail(
                  selectedIndex: _selectedIndex,
                  onDestinationSelected: (index) {
                    setState(() => _selectedIndex = index);
                    if (index == 1 || index == 2) {
                      unawaited(_refreshPreviewItems());
                    }
                  },
                  labelType: NavigationRailLabelType.all,
                  leading: Padding(
                    padding: const EdgeInsets.only(top: 8, bottom: 16),
                    child: Icon(
                      Icons.camera_outdoor_rounded,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                  destinations: const [
                    NavigationRailDestination(
                      icon: Icon(Icons.home_outlined),
                      selectedIcon: Icon(Icons.home_rounded),
                      label: Text('开始'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.photo_library_outlined),
                      selectedIcon: Icon(Icons.photo_library_rounded),
                      label: Text('预览'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.fact_check_outlined),
                      selectedIcon: Icon(Icons.fact_check_rounded),
                      label: Text('校验'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.settings_outlined),
                      selectedIcon: Icon(Icons.settings_rounded),
                      label: Text('设置'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.info_outline_rounded),
                      selectedIcon: Icon(Icons.info_rounded),
                      label: Text('关于'),
                    ),
                  ],
                ),
                const VerticalDivider(width: 1),
                Expanded(
                  // 使用 IndexedStack 保持所有页面的状态（包括滚动位置和局部过滤选项）
                  child: IndexedStack(
                    index: _selectedIndex,
                    children: [
                      _buildStartScreen(),
                      _buildPreviewPage(),
                      _buildValidationPage(),
                      _buildSettingsScreen(),
                      AboutScreen(settings: _settings),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

Widget _buildStartScreen() {
    return StartScreen(
      settings: _settings,
      inputController: _inputController,
      selectedModelPath: _selectedModelPath,
      onModelChanged: (value) => setState(() => _selectedModelPath = value),
      enableDetection: _enableDetection,
      onEnableDetectionChanged: (value) =>
          setState(() => _enableDetection = value),
      useFp16: _useFp16,
      onUseFp16Changed: (value) => setState(() => _useFp16 = value),
      confidence: _confidence,
      onConfidenceChanged: (value) => setState(() => _confidence = value),
      iou: _iou,
      onIouChanged: (value) => setState(() => _iou = value),
      submitting: _submitting,
      onCreateJob: _createJob,
      jobs: _jobs,
    );
  }

  Widget _buildSettingsScreen() {
    return SettingsScreen(
      settings: _settings,
      themeNotifier: widget.themeNotifier,
      onUpdateTheme: _updateTheme,
      onSaveSettings: _saveAdvancedSettings,
      onShowMessage: _showSnackBar,
    );
  }

  Widget _buildPreviewPage() {
    final inputPath = _inputController.text.trim();
    final jobResults = _jobs.expand((job) => job.results).toList();
    final rawItems = inputPath.isEmpty ? jobResults : _previewItems;
    final items = rawItems.map(_resolvedPreviewItem).toList();
    final selectedIndex = _safePreviewIndex(items);
    final selectedItem = items.isEmpty ? null : items[selectedIndex];

    return PreviewScreen(
      inputPath: inputPath,
      items: items,
      selectedIndex: selectedIndex,
      selectedItem: selectedItem,
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
      onDetectCurrentImage: _detectCurrentPreviewImage,
      onSelected: (index, item) {
        setState(() => _selectedPreviewIndex = index);
        unawaited(_loadPreviewMetadata(item));
      },
      onLoadMetadata: _loadPreviewMetadata,
      onOpenExternal: _openFileWithSystem,
    );
  }

  Widget _buildValidationPage() {
    final inputPath = _inputController.text.trim();
    final jobResults = _jobs.expand((job) => job.results).toList();
    final sourceItems = inputPath.isEmpty ? jobResults : _previewItems;
    final items = sourceItems.map(_resolvedPreviewItem).toList();

    return SpeciesValidationScreen(
      apiClient: widget.apiClient,
      inputPath: inputPath,
      items: items,
      speciesTypes: _settings?.speciesTypes ?? const <String, String>{},
      autoGroup: _boolSetting(_settingsOrEmpty(), 'auto_group', true),
      quickMarkSpecies: _stringListSetting(
        _settingsOrEmpty(),
        'quick_mark_list',
        const <String>[],
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
        );
  }
}
