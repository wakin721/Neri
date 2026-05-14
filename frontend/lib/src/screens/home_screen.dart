import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:window_manager/window_manager.dart';

import '../api_client.dart';
import '../models/job.dart';
import '../models/settings.dart';
import '../models/theme_settings.dart';
import '../widgets/section_card.dart';

const _defaultModelDirectory = 'res/model';
const _lastInputPathKey = 'last_input_path';
const _allSpeciesLabel = '全部物种';

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    required this.apiClient,
    required this.themeNotifier,
    super.key,
  });

  final NeriApiClient apiClient;
  final ValueNotifier<ThemeSettings> themeNotifier;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WindowListener {
  final _inputController = TextEditingController();

  NeriSettings? _settings;
  List<ProcessingJob> _jobs = const <ProcessingJob>[];
  List<DetectionItem> _previewItems = const <DetectionItem>[];
  final Map<String, DetectionItem> _previewMetadataCache = <String, DetectionItem>{};
  final Set<String> _previewMetadataLoading = <String>{};
  Timer? _timer;
  Timer? _previewRefreshTimer;
  bool _loading = true;
  bool _submitting = false;
  bool _previewLoading = false;
  bool _enableDetection = false;
  bool _useFp16 = false;
  bool _previewShowDetections = true;
  bool _previewDetecting = false;
  double _confidence = 0.25;
  double _iou = 0.45;
  double _previewConfidenceThreshold = 0.25;
  String? _error;
  String _previewSpeciesFilter = _allSpeciesLabel;
  String? _selectedModelPath;
  String? _previewLoadedPath;
  String? _previewError;
  int _selectedIndex = 0;
  int _selectedPreviewIndex = 0;
  bool _isMaximized = false;

  static const _pageTitles = <String>[
    '开始界面',
    '图像预览',
    '物种校验',
    '设置',
    '关于',
  ];

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _initWindowState();

    // 初始化万能解码器内核
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
    await preferences.setInt('seed_color', settings.seedColor.value);
  }

  void _updateTheme(ThemeSettings updated) {
    widget.themeNotifier.value = updated;
    _saveThemeSettings(updated);
  }

  Future<void> _initWindowState() async {
    try {
      bool isMaximized = await windowManager.isMaximized();
      if (mounted) setState(() => _isMaximized = isMaximized);
    } catch (_) {}
  }

  @override
  void onWindowMaximize() {
    if (mounted) setState(() => _isMaximized = true);
  }

  @override
  void onWindowUnmaximize() {
    if (mounted) setState(() => _isMaximized = false);
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
        _previewError = null;
      });
      return;
    }

    // 当路径有效时，自动保存，防止未点击“开始处理”就关闭应用导致路径丢失
    _saveLastInputPath(inputPath);

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
    setState(() {
      _previewLoading = true;
      _previewError = null;
    });

    try {
      final items = await widget.apiClient.fetchPreviewItems(inputPath: inputPath);
      if (!mounted || _inputController.text.trim() != inputPath) return;
      setState(() {
        _previewItems = items;
        _previewMetadataCache.clear();
        _previewMetadataLoading.clear();
        _previewLoadedPath = inputPath;
        _selectedPreviewIndex = _selectedPreviewIndex >= items.length
            ? (items.isEmpty ? 0 : items.length - 1)
            : _selectedPreviewIndex;
      });
      if (items.isNotEmpty) {
        unawaited(_loadPreviewMetadata(items[_selectedPreviewIndex]));
      }
    } catch (error) {
      if (!mounted || _inputController.text.trim() != inputPath) return;
      setState(() => _previewError = '无法读取输入文件夹预览：$error');
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
      // Fast preview data is still usable if per-file metadata cannot be read.
    } finally {
      _previewMetadataLoading.remove(item.path);
    }
  }

  DetectionItem _resolvedPreviewItem(DetectionItem item) {
    return _previewMetadataCache[item.path] ?? item;
  }

  Future<void> _refresh({bool silent = false}) async {
    if (!silent) {
      setState(() {
        _loading = true;
        _error = null;
      });
    }
    try {
      final settings = await widget.apiClient.fetchSettings();
      final jobs = await widget.apiClient.listJobs();
      if (!mounted) return;
      setState(() {
        _settings = settings;
        final modelPaths = settings.availableModels.map((model) => model.path).toSet();
        if (_selectedModelPath == null || !modelPaths.contains(_selectedModelPath)) {
          _selectedModelPath = settings.selectedModel ??
              (settings.availableModels.isEmpty ? null : settings.availableModels.first.path);
        }
        _jobs = jobs;
        _loading = false;
        _error = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = '无法连接 Python 后端：$error';
      });
    }
  }

  Future<void> _createJob() async {
    final inputPath = _inputController.text.trim();
    if (inputPath.isEmpty) {
      setState(() => _error = '请输入红外相机媒体文件夹路径');
      return;
    }
    setState(() {
      _submitting = true;
      _error = null;
    });
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
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = '创建任务失败：$error');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _detectCurrentPreviewImage(DetectionItem item) async {
    setState(() {
      _previewDetecting = true;
      _error = null;
    });
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
      if (mounted) {
        setState(() {
          _selectedIndex = 1;
          // 移除 _selectedPreviewIndex = 0; 以保留当前选中的图像位置
        });
      }
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = '检测当前图像失败：$error');
    } finally {
      if (mounted) setState(() => _previewDetecting = false);
    }
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
                  _isMaximized ? Icons.filter_none_rounded : Icons.crop_square_rounded,
                  size: 18,
                ),
                onPressed: () async {
                  if (await windowManager.isMaximized()) {
                    windowManager.unmaximize();
                  } else {
                    windowManager.maximize();
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
                          return Colors.red.withOpacity(0.9);
                        }
                        return null;
                      }),
                      foregroundColor: WidgetStateProperty.resolveWith((states) {
                        if (states.contains(WidgetState.hovered)) {
                          return Colors.white;
                        }
                        return Theme.of(context).colorScheme.onSurfaceVariant;
                      }),
                    ),
                    onPressed: () => windowManager.close(),
                  );
                }
              ),
              const SizedBox(width: 8),
            ],
          ),
        ),
      ),
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (index) {
              setState(() => _selectedIndex = index);
              if (index == 1) _refreshPreviewItems();
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
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : RefreshIndicator(
                    onRefresh: _refresh,
                    child: _buildSelectedPage(),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildSelectedPage() {
    return switch (_selectedIndex) {
      0 => _buildStartPage(),
      1 => _buildPreviewPage(),
      2 => _buildValidationPage(),
      3 => _buildSettingsPage(),
      _ => _buildAboutPage(),
    };
  }

  Widget _buildPageList(List<Widget> children) {
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      children: [
        if (_error != null) _buildErrorBanner(context),
        ...children,
      ],
    );
  }

  Widget _buildStartPage() {
    return _buildPageList([
      _buildHero(Theme.of(context).colorScheme),
      _buildCreateJobCard(),
      _buildJobsCard(),
    ]);
  }

  Widget _buildPreviewPage() {
    final inputPath = _inputController.text.trim();
    final jobResults = _jobs.expand((job) => job.results).toList();
    final results = inputPath.isEmpty ? jobResults : _previewItems;
    final selectedIndex = results.isEmpty
        ? 0
        : _selectedPreviewIndex >= results.length
            ? results.length - 1
            : _selectedPreviewIndex;
    final selectedItem = results.isEmpty ? null : _resolvedPreviewItem(results[selectedIndex]);

    // 使用 Padding + Column 替代原先的 ListView，消除页面整体的上下滚动
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_error != null) _buildErrorBanner(context),
          // 使用 Visibility 保持控件占用固定高度空间，防止加载时整个界面上下跳动
          Visibility(
            visible: _previewLoading,
            maintainSize: true,
            maintainAnimation: true,
            maintainState: true,
            child: const Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                LinearProgressIndicator(),
                SizedBox(height: 12),
              ],
            ),
          ),
          if (_previewError != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text(
                _previewError!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          if (results.isEmpty)
            Text(inputPath.isEmpty ? '请先在开始界面设置输入文件夹。' : '该输入文件夹中暂无可预览图像。')
          else
            // 使用 Expanded 让工作区自动撑满剩余所有的屏幕高度
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return _buildResponsivePreviewWorkspace(
                    constraints.maxWidth,
                    results,
                    selectedIndex,
                    selectedItem!,
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildResponsivePreviewWorkspace(
    double availableWidth,
    List<DetectionItem> results,
    int selectedIndex,
    DetectionItem selectedItem,
  ) {
    // 移除原有的高度计算逻辑，交由外部的 Expanded 自动控制填充剩余空间
    final useHorizontalLayout = availableWidth >= 900;

    if (useHorizontalLayout) {
      // 缩小文件列表宽度，使图像预览区域增加
      final listWidth = (availableWidth * 0.20).clamp(200.0, 300.0).toDouble();
      return Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            width: listWidth,
            child: _buildPreviewFileList(results, selectedIndex),
          ),
          const VerticalDivider(width: 24),
          Expanded(child: _buildPreviewDetail(selectedItem)),
        ],
      );
    }

    // 屏幕较窄时的上下布局
    final listHeight = availableWidth < 600 ? 220.0 : 280.0;
    return Column(
      children: [
        SizedBox(
          height: listHeight,
          child: _buildPreviewFileList(results, selectedIndex),
        ),
        const SizedBox(height: 16),
        // 下方的详细信息通过 Expanded 填满
        Expanded(
          child: _buildPreviewDetail(selectedItem),
        ),
      ],
    );
  }

  Widget _buildPreviewFileList(List<DetectionItem> results, int selectedIndex) {
    return Card.filled(
      clipBehavior: Clip.antiAlias,
      child: ListView.separated(
        itemCount: results.length,
        separatorBuilder: (_, __) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final item = results[index];
          final selected = index == selectedIndex;
          return ListTile(
            selected: selected,
            leading: Icon(_previewFileIcon(item)),
            title: Text(
              item.filename,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            subtitle: Text(
              item.species.isEmpty ? item.fileType : item.species.join('、'),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            trailing: item.error == null
                ? null
                : const Icon(Icons.error_outline_rounded),
            onTap: () {
              setState(() => _selectedPreviewIndex = index);
              unawaited(_loadPreviewMetadata(item));
            },
          );
        },
      ),
    );
  }

  Widget _buildPreviewDetail(DetectionItem item) {
    final filterApplies = item.detectionBoxes.any((box) => box.species == _previewSpeciesFilter) ||
        item.species.contains(_previewSpeciesFilter);
    final selectedSpecies = filterApplies ? _previewSpeciesFilter : _allSpeciesLabel;
    final visibleBoxes = _filteredPreviewBoxes(item, selectedSpecies: selectedSpecies);

    final speciesOptions = <String>{
      _allSpeciesLabel,
      ...item.species,
      ...item.detectionBoxes.map((box) => box.species),
    }.where((species) => species.isNotEmpty).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 使用 Expanded 包裹图像区域，使其填满所有剩余的高度空间
        Expanded(
          child: _buildPreviewMedia(item, visibleBoxes),
        ),
        const SizedBox(height: 12),
        _buildImageInfoCard(item, visibleBoxes),
        const SizedBox(height: 12),
        _buildPreviewDetectionControls(item, selectedSpecies, speciesOptions),
      ],
    );
  }

  Widget _buildPreviewMedia(DetectionItem item, List<DetectionBox> visibleBoxes) {
    final isImage = _isPreviewImage(item);
    const videoTypes = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'};
    final isVideo = videoTypes.contains(item.fileType.toLowerCase());

    return Card.filled(
      clipBehavior: Clip.antiAlias,
      child: Container(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        alignment: Alignment.center,
        // 取消了双击和右键跳转手势，直接返回构建内容
        child: _buildMediaContent(
          item, 
          isImage, 
          isVideo, 
          visibleBoxes,
          () => _openFileWithSystem(item.path),
        ),
      ),
    );
  }

  Widget _buildMediaContent(
    DetectionItem item, 
    bool isImage, 
    bool isVideo, 
    List<DetectionBox> visibleBoxes,
    VoidCallback onOpenExternal,
  ) {
    if (isImage) {
      return Stack(
        fit: StackFit.expand,
        children: [
          Image.file(
            File(item.path),
            fit: BoxFit.contain,
            errorBuilder: (context, error, stackTrace) {
              return _buildPlaceholder(Icons.broken_image_rounded, '无法读取图片：${item.filename}');
            },
          ),
          if (_previewShowDetections && visibleBoxes.isNotEmpty)
            Positioned.fill(
              child: IgnorePointer(
                child: CustomPaint(
                  painter: _DetectionBoxPainter(
                    boxes: visibleBoxes,
                    imageSize: Size(
                      (item.width ?? 0).toDouble(),
                      (item.height ?? 0).toDouble(),
                    ),
                    colorScheme: Theme.of(context).colorScheme,
                  ),
                ),
              ),
            ),
          // 图片右上角的系统应用跳转按钮
          Positioned(
            top: 12,
            right: 12,
            child: IconButton.filledTonal(
              icon: const Icon(Icons.open_in_new_rounded),
              tooltip: '使用系统应用打开',
              onPressed: onOpenExternal,
            ),
          ),
        ],
      );
    } else if (isVideo) {
      return _PreviewVideoPlayer(
        path: item.path,
        onOpenExternal: onOpenExternal,
      );
    } else {
      return Stack(
        fit: StackFit.expand,
        children: [
          _buildPlaceholder(Icons.insert_drive_file_rounded, '不支持直接预览的文件格式'),
          Positioned(
            top: 12,
            right: 12,
            child: IconButton.filledTonal(
              icon: const Icon(Icons.open_in_new_rounded),
              tooltip: '使用系统应用打开',
              onPressed: onOpenExternal,
            ),
          ),
        ],
      );
    }
  }

  Widget _buildPlaceholder(IconData icon, String message) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, size: 56),
        const SizedBox(height: 12),
        Text(message, textAlign: TextAlign.center),
      ],
    );
  }

  void _openFileWithSystem(String path) {
    try {
      if (Platform.isWindows) {
        Process.run('cmd', ['/c', 'start', '', path]);
      } else if (Platform.isMacOS) {
        Process.run('open', [path]);
      } else if (Platform.isLinux) {
        Process.run('xdg-open', [path]);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('无法打开文件: $e')),
      );
    }
  }

  Widget _buildImageInfoCard(DetectionItem item, List<DetectionBox> visibleBoxes) {
    String detectionResultStr = '暂无';
    String boxCountStr = '0';
    String minConfStr = '未知';

    final rawSpecies = item.detectionData['物种名称'];
    final rawConf = item.detectionData['最低置信度'];

    // 将统计与筛选状态实时融合至卡片中
    if (item.error != null) {
      detectionResultStr = '错误：${item.error}';
    } else if (item.detectionBoxes.isNotEmpty) {
      if (visibleBoxes.isEmpty) {
        detectionResultStr = '无结果';
      } else {
        final counts = <String, int>{};
        final confidences = <double>[];
        for (final box in visibleBoxes) {
          counts[box.species] = (counts[box.species] ?? 0) + 1;
          if (box.confidence != null) confidences.add(box.confidence!);
        }
        detectionResultStr = counts.entries.map((e) => '${e.key}×${e.value}').join('，');
        boxCountStr = visibleBoxes.length.toString();
        if (confidences.isNotEmpty) {
          minConfStr = confidences.reduce((a, b) => a < b ? a : b).toStringAsFixed(2);
        }
      }
    } else if (rawSpecies != null && rawSpecies.toString().trim().isNotEmpty) {
      detectionResultStr = rawSpecies.toString();
      if (rawConf != null && rawConf.toString().trim().isNotEmpty) {
        minConfStr = rawConf.toString();
      }
    } else if (item.species.isNotEmpty) {
      detectionResultStr = item.species.join('、');
      if (item.confidence != null) {
        minConfStr = item.confidence!.toStringAsFixed(2);
      }
    }

    return Card.outlined(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildInfoLine([
              _buildInlineInfo('文件名', item.filename, maxWidth: 220),
              _buildInlineInfo('类型', item.fileType, maxWidth: 80),
              _buildInlineInfo('尺寸', _formatImageSize(item), maxWidth: 120),
              _buildInlineInfo('大小', _formatBytes(item.sizeBytes), maxWidth: 100),
              _buildInlineInfo('拍摄时间', item.dateTaken ?? '未知', maxWidth: 160),
            ]),
            _buildInfoLine([
              _buildInlineInfo('检测结果', detectionResultStr, maxWidth: 220),
              _buildInlineInfo('检测框', boxCountStr, maxWidth: 80),
              _buildInlineInfo('最低置信度', minConfStr, maxWidth: 120),
              _buildInlineInfo('检测时间', _detectionTimeLabel(item), maxWidth: 160),
            ]),
            if (item.error != null) ...[
              const SizedBox(height: 4),
              Text(
                '错误：${item.error}',
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInfoLine(List<Widget> children) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Wrap(
        spacing: 16,
        runSpacing: 4,
        children: children,
      ),
    );
  }

  Widget _buildInlineInfo(String label, String value, {double maxWidth = 200}) {
    return ConstrainedBox(
      constraints: BoxConstraints(maxWidth: maxWidth),
      child: RichText(
        text: TextSpan(
          style: Theme.of(context).textTheme.bodyMedium,
          children: [
            TextSpan(
              text: '$label：',
              style: Theme.of(context).textTheme.labelMedium,
            ),
            TextSpan(text: value),
          ],
        ),
        overflow: TextOverflow.ellipsis,
        maxLines: 1,
      ),
    );
  }

  String _formatImageSize(DetectionItem item) {
    if (item.width == null || item.height == null) return '未知';
    return '${item.width}×${item.height}';
  }

  String _formatBytes(int? bytes) {
    if (bytes == null) return '未知';
    if (bytes < 1024) return '${bytes}B';
    final kb = bytes / 1024;
    if (kb < 1024) return '${kb.toStringAsFixed(1)}KB';
    final mb = kb / 1024;
    if (mb < 1024) return '${mb.toStringAsFixed(1)}MB';
    return '${(mb / 1024).toStringAsFixed(1)}GB';
  }

  String _detectionTimeLabel(DetectionItem item) {
    final raw = item.detectionData['检测时间'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString();
    }
    return '暂无';
  }

  Widget _buildPreviewDetectionControls(DetectionItem item, String selectedSpecies, List<String> speciesOptions) {
    return Card.outlined(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isWide = constraints.maxWidth >= 600;

            final toggle = Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('显示结果'),
                Switch(
                  value: _previewShowDetections,
                  onChanged: (value) => setState(() => _previewShowDetections = value),
                ),
              ],
            );

            final speciesDropdown = Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('物种'),
                const SizedBox(width: 8),
                SizedBox(
                  width: 150, // 增加了下拉栏的宽度，从 110 改为 150
                  child: DropdownButtonFormField<String>(
                    value: selectedSpecies,
                    isExpanded: true,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 0),
                    ),
                    items: speciesOptions.map((s) => DropdownMenuItem(value: s, child: Text(s, overflow: TextOverflow.ellipsis, maxLines: 1))).toList(),
                    onChanged: (value) {
                      if (value != null) setState(() => _previewSpeciesFilter = value);
                    },
                  ),
                ),
              ],
            );

            final sliderRow = Row(
              children: [
                const Text('置信度'),
                Expanded(
                  child: Slider(
                    value: _previewConfidenceThreshold,
                    divisions: 100,
                    label: _previewConfidenceThreshold.toStringAsFixed(2),
                    onChanged: (value) => setState(() => _previewConfidenceThreshold = value),
                  ),
                ),
                SizedBox(
                  width: 32,
                  child: Text(_previewConfidenceThreshold.toStringAsFixed(2)),
                ),
              ],
            );

            final detectBtn = FilledButton.icon(
              onPressed: _previewDetecting || !_isPreviewImage(item)
                  ? null
                  : () => _detectCurrentPreviewImage(item),
              icon: _previewDetecting
                  // 统一进度条和原图标尺寸（18px），防止按钮宽/高轻微抖动
                  ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.center_focus_strong_rounded, size: 18),
              label: Text(_previewDetecting ? '检测中...' : '检测当前图像'),
            );

            if (isWide) {
              // 宽度充足时单行平铺
              return Row(
                children: [
                  toggle,
                  const SizedBox(width: 16),
                  speciesDropdown,
                  const SizedBox(width: 16),
                  Expanded(child: sliderRow),
                  const SizedBox(width: 16),
                  detectBtn,
                ],
              );
            } else {
              // 宽度较小时分为两行保证不会溢出
              return Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [toggle, speciesDropdown],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(child: sliderRow),
                      const SizedBox(width: 12),
                      detectBtn,
                    ],
                  ),
                ],
              );
            }
          },
        ),
      ),
    );
  }

  List<DetectionBox> _filteredPreviewBoxes(
    DetectionItem item, {
    String? selectedSpecies,
  }) {
    final species = selectedSpecies ?? _previewSpeciesFilter;
    return item.detectionBoxes.where((box) {
      final matchesSpecies = species == _allSpeciesLabel || box.species == species;
      final confidence = box.confidence;
      final matchesConfidence =
          confidence == null || confidence >= _previewConfidenceThreshold;
      return matchesSpecies && matchesConfidence && box.bbox.length >= 4;
    }).toList();
  }

  IconData _previewFileIcon(DetectionItem item) {
    return _isPreviewImage(item) ? Icons.image_rounded : Icons.movie_rounded;
  }

  bool _isPreviewImage(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }

// --- 在这之后添加一个新的 Widget，用于处理视频播放控制 ---
// 建议加在 _buildValidationPage() 上面，或者类的末尾

  Widget _buildValidationPage() {
    final detectedItems = _jobs
        .expand((job) => job.results)
        .where((item) => item.species.isNotEmpty || item.error != null)
        .toList();
    return _buildPageList([
      SectionCard(
        title: '物种校验界面',
        subtitle: '集中查看模型输出，后续可扩展为人工校验与修正流程。',
        icon: Icons.fact_check_rounded,
        child: detectedItems.isEmpty
            ? const Text('暂无待校验物种。启用 YOLO 识别并完成任务后，检测结果会显示在这里。')
            : Column(
                children: detectedItems.map((item) {
                  return ListTile(
                    leading: Icon(
                      item.error == null
                          ? Icons.pets_rounded
                          : Icons.error_outline_rounded,
                    ),
                    title: Text(item.filename),
                    subtitle: Text(item.error ?? item.species.join('、')),
                    trailing: item.confidence == null
                        ? null
                        : Text(item.confidence!.toStringAsFixed(2)),
                  );
                }).toList(),
              ),
      ),
    ]);
  }

  Widget _buildSettingsPage() {
    return _buildPageList([
      _buildAppearanceCard(),
      _buildSettingsCard(),
    ]);
  }

  Widget _buildAppearanceCard() {
    return ValueListenableBuilder<ThemeSettings>(
      valueListenable: widget.themeNotifier,
      builder: (context, settings, _) {
        final colorScheme = Theme.of(context).colorScheme;
        final paletteDisabled = settings.useDynamicColor;

        return SectionCard(
          title: '外观与主题',
          icon: Icons.palette_outlined,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Row 1: Theme mode ──────────────────────────────────────
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    SizedBox(
                      width: 88,
                      child: Text(
                        '主题',
                        style: Theme.of(context).textTheme.labelLarge,
                      ),
                    ),
                    const SizedBox(width: 8),
                    SegmentedButton<ThemeMode>(
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
                        if (selection.isNotEmpty) {
                          _updateTheme(
                            settings.copyWith(themeMode: selection.first),
                          );
                        }
                      },
                    ),
                  ],
                ),
              ),

              const Divider(height: 24),

              // ── Row 2: Dynamic color ───────────────────────────────────
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                secondary: const Icon(Icons.auto_awesome_rounded),
                title: const Text('动态颜色'),
                subtitle: const Text('跟随系统强调色（仅在 Android 12+ 上生效）'),
                value: settings.useDynamicColor,
                onChanged: (value) =>
                    _updateTheme(settings.copyWith(useDynamicColor: value)),
              ),

              const Divider(height: 24),

              // ── Row 3: Color palette ───────────────────────────────────
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  children: [
                    const Icon(Icons.colorize_rounded, size: 20),
                    const SizedBox(width: 8),
                    Text(
                      '调色板',
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                            color: paletteDisabled
                                ? colorScheme.onSurface.withOpacity(0.38)
                                : null,
                          ),
                    ),
                    if (paletteDisabled) ...[
                      const SizedBox(width: 8),
                      Text(
                        '（已由系统接管）',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: colorScheme.onSurface.withOpacity(0.38),
                            ),
                      ),
                    ],
                  ],
                ),
              ),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: kSeedColorOptions.map((option) {
                  final selected = !paletteDisabled &&
                      settings.seedColor.value == option.color.value;
                  return Tooltip(
                    message: option.label,
                    child: _ColorSwatch(
                      color: option.color,
                      label: option.label,
                      selected: selected,
                      disabled: paletteDisabled,
                      onTap: paletteDisabled
                          ? null
                          : () => _updateTheme(
                                settings.copyWith(seedColor: option.color),
                              ),
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildAboutPage() {
    return _buildPageList([
      SectionCard(
        title: '关于 Neri',
        subtitle: _settings == null ? null : _settings!.appVersion,
        icon: Icons.info_rounded,
        child: const Text(
          'Neri 是红外相机图像智能处理工具。当前 Flutter Material 3 客户端通过 Python 后端复用项目已有的 EXIF 提取、批量处理和 YOLO 识别能力。',
        ),
      ),
    ]);
  }

  Widget _buildHero(ColorScheme colorScheme) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(28),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.camera_outdoor_rounded,
            size: 48,
            color: colorScheme.onPrimaryContainer,
          ),
          const SizedBox(height: 12),
          Text(
            '红外相机影像智能处理',
            style: Theme.of(context)
                .textTheme
                .headlineSmall
                ?.copyWith(color: colorScheme.onPrimaryContainer),
          ),
          const SizedBox(height: 8),
          Text(
            '使用 Flutter Material 3 前端连接 Python 后端，完成批量索引、EXIF 提取、YOLO 识别与结果导出。',
            style: Theme.of(context)
                .textTheme
                .bodyLarge
                ?.copyWith(color: colorScheme.onPrimaryContainer),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: MaterialBanner(
        content: Text(_error!),
        leading: const Icon(Icons.warning_amber_rounded),
        actions: [
          TextButton(
            onPressed: () => setState(() => _error = null),
            child: const Text('关闭'),
          ),
        ],
      ),
    );
  }

  Widget _buildModelSelector() {
    final settings = _settings;
    final models = settings?.availableModels ?? const <ModelInfo>[];
    final selectedValue = models.any((model) => model.path == _selectedModelPath)
        ? _selectedModelPath
        : (models.isEmpty ? null : models.first.path);

    if (models.isEmpty) {
      return InputDecorator(
        decoration: InputDecoration(
          labelText: '模型文件',
          helperText: '未在 ${settings?.modelDirectory ?? _defaultModelDirectory} 中找到 .pt 模型',
          prefixIcon: const Icon(Icons.folder_off_rounded),
          border: const OutlineInputBorder(),
        ),
        child: const Text('暂无可用模型'),
      );
    }

    return DropdownButtonFormField<String>(
      value: selectedValue,
      decoration: InputDecoration(
        labelText: '模型文件',
        helperText: '扫描 ${settings?.modelDirectory ?? _defaultModelDirectory} 下的 .pt 文件',
        prefixIcon: const Icon(Icons.memory_rounded),
        border: const OutlineInputBorder(),
      ),
      items: models.map((model) {
        final sizeLabel = model.sizeBytes == null
            ? ''
            : ' · ${(model.sizeBytes! / (1024 * 1024)).toStringAsFixed(1)} MB';
        return DropdownMenuItem<String>(
          value: model.path,
          child: Text('${model.name}$sizeLabel'),
        );
      }).toList(),
      onChanged: (value) => setState(() => _selectedModelPath = value),
    );
  }

  Widget _buildCreateJobCard() {
    return SectionCard(
      title: '新建处理任务',
      subtitle: '输入本机路径，由 Python 后端读取和处理文件。',
      icon: Icons.playlist_add_rounded,
      child: Column(
        children: [
          TextField(
            controller: _inputController,
            decoration: const InputDecoration(
              labelText: '输入文件夹',
              hintText: '/path/to/camera-trap-folder',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          _buildModelSelector(),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _enableDetection,
            onChanged: (value) => setState(() => _enableDetection = value),
            title: const Text('启用 YOLO 识别'),
            subtitle: const Text('关闭时仅进行快速文件索引和 EXIF 元数据提取。'),
          ),
          SwitchListTile(
            value: _useFp16,
            onChanged: _enableDetection ? (value) => setState(() => _useFp16 = value) : null,
            title: const Text('使用 FP16 加速'),
          ),
          _buildSlider(
            '置信度阈值',
            _confidence,
            (value) => setState(() => _confidence = value),
          ),
          _buildSlider(
            'IOU 阈值',
            _iou,
            (value) => setState(() => _iou = value),
          ),
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: _submitting ? null : _createJob,
            icon: _submitting
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.play_arrow_rounded),
            label: Text(_submitting ? '提交中...' : '开始处理'),
          ),
        ],
      ),
    );
  }

  Widget _buildSlider(
    String label,
    double value,
    ValueChanged<double> onChanged,
  ) {
    return Row(
      children: [
        SizedBox(width: 96, child: Text(label)),
        Expanded(
          child: Slider(
            value: value,
            divisions: 100,
            label: value.toStringAsFixed(2),
            onChanged: onChanged,
          ),
        ),
        Text(value.toStringAsFixed(2)),
      ],
    );
  }

  Widget _buildJobsCard() {
    return SectionCard(
      title: '任务进度',
      subtitle: _jobs.isEmpty ? '暂无任务' : '自动每 2 秒刷新一次',
      icon: Icons.insights_rounded,
      child: _jobs.isEmpty
          ? const Text('创建任务后，处理进度和识别结果会显示在这里。')
          : Column(children: _jobs.map(_buildJobTile).toList()),
    );
  }

  Widget _buildJobTile(ProcessingJob job) {
    return ExpansionTile(
      leading: Icon(_jobIcon(job.state)),
      title: Text(job.message.isEmpty ? job.inputDir : job.message),
      subtitle: LinearProgressIndicator(
        value: job.total == 0 && job.isActive ? null : job.progress,
      ),
      trailing: Chip(label: Text('${job.processed}/${job.total}')),
      children: [
        if (job.error != null)
          ListTile(
            leading: const Icon(Icons.error_outline_rounded),
            title: Text(job.error!),
          ),
        ...job.results.take(12).map(
              (item) => ListTile(
                title: Text(item.filename),
                subtitle: Text(
                  item.species.isEmpty
                      ? (item.dateTaken ?? item.path)
                      : item.species.join('、'),
                ),
                trailing: item.confidence == null
                    ? null
                    : Text(item.confidence!.toStringAsFixed(2)),
              ),
            ),
        if (job.results.length > 12)
          ListTile(
            title: Text(
              '还有 ${job.results.length - 12} 条结果，可在导出的 CSV 中查看。',
            ),
          ),
      ],
    );
  }

  IconData _jobIcon(String state) {
    return switch (state) {
      'completed' => Icons.check_circle_rounded,
      'failed' => Icons.error_rounded,
      'running' => Icons.sync_rounded,
      _ => Icons.schedule_rounded,
    };
  }

  Widget _buildSettingsCard() {
    final settings = _settings;
    return SectionCard(
      title: '项目配置',
      subtitle: settings == null ? null : '版本 ${settings.appVersion}',
      icon: Icons.tune_rounded,
      child: settings == null
          ? const Text('未读取到后端配置。')
          : Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ...settings.supportedImageExtensions.map(
                  (extension) => Chip(label: Text(extension)),
                ),
                ...settings.supportedVideoExtensions.map(
                  (extension) => Chip(label: Text(extension)),
                ),
              ],
            ),
    );
  }
}

// ── Appearance helpers ─────────────────────────────────────────────────────────

class _ColorSwatch extends StatelessWidget {
  const _ColorSwatch({
    required this.color,
    required this.label,
    required this.selected,
    required this.disabled,
    required this.onTap,
  });

  final Color color;
  final String label;
  final bool selected;
  final bool disabled;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    // 基于种子颜色即时生成配套调色板，用于展示四个扇区的颜色
    final scheme = ColorScheme.fromSeed(
      seedColor: color,
      brightness: Theme.of(context).brightness,
    );

    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 200),
        opacity: disabled ? 0.35 : 1.0,
        child: Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.5),
            borderRadius: BorderRadius.circular(16),
          ),
          alignment: Alignment.center,
          child: SizedBox(
            width: 46,
            height: 46,
            child: ClipOval(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Column(
                    children: [
                      // 上半圆
                      Expanded(
                        child: Container(color: scheme.primaryContainer),
                      ),
                      // 下半圆
                      Expanded(
                        child: Row(
                          children: [
                            // 左下象限
                            Expanded(child: Container(color: scheme.tertiaryContainer)),
                            // 右下象限
                            Expanded(child: Container(color: scheme.primary)),
                          ],
                        ),
                      ),
                    ],
                  ),
                  if (selected)
                    Center(
                      child: Icon(
                        Icons.check_rounded,
                        // 确保对勾在不同亮度的种子色下清晰可见
                        color: ThemeData.estimateBrightnessForColor(color) == Brightness.dark
                            ? Colors.white
                            : Colors.black,
                        size: 24,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ── Detection box painter ──────────────────────────────────────────────────────

class _DetectionBoxPainter extends CustomPainter {
  _DetectionBoxPainter({
    required this.boxes,
    required this.imageSize,
    required this.colorScheme,
  });

  final List<DetectionBox> boxes;
  final Size imageSize;
  final ColorScheme colorScheme;

  @override
  void paint(Canvas canvas, Size size) {
    if (boxes.isEmpty || size.isEmpty) return;

    final sourceSize = _resolveSourceSize();
    if (sourceSize.isEmpty) return;

    final imageRect = _containRect(size, sourceSize);
    final strokeWidth = math.max(2.0, math.min(imageRect.width, imageRect.height) * 0.004);

    for (final box in boxes) {
      final rect = _boxRect(box, sourceSize, imageRect);
      if (rect.isEmpty) continue;

      final color = _speciesColor(box.species);
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round
        ..color = color;
      canvas.drawRect(rect, paint);

      final confidence = box.confidence == null ? '' : ' ${(box.confidence! * 100).toStringAsFixed(0)}%';
      final label = '${box.species}$confidence';
      _drawLabel(canvas, imageRect, rect, label, color);
    }
  }

  Size _resolveSourceSize() {
    if (imageSize.width > 0 && imageSize.height > 0) return imageSize;

    var width = 1.0;
    var height = 1.0;
    for (final box in boxes) {
      if (box.bbox.length < 4) continue;
      width = math.max(width, math.max(box.bbox[0], box.bbox[2]));
      height = math.max(height, math.max(box.bbox[1], box.bbox[3]));
    }
    return Size(width, height);
  }

  Rect _containRect(Size canvasSize, Size sourceSize) {
    final sourceAspect = sourceSize.width / sourceSize.height;
    final canvasAspect = canvasSize.width / canvasSize.height;
    double width;
    double height;
    if (canvasAspect > sourceAspect) {
      height = canvasSize.height;
      width = height * sourceAspect;
    } else {
      width = canvasSize.width;
      height = width / sourceAspect;
    }
    return Rect.fromLTWH(
      (canvasSize.width - width) / 2,
      (canvasSize.height - height) / 2,
      width,
      height,
    );
  }

  Rect _boxRect(DetectionBox box, Size sourceSize, Rect imageRect) {
    if (box.bbox.length < 4) return Rect.zero;
    var x1 = box.bbox[0];
    var y1 = box.bbox[1];
    var x2 = box.bbox[2];
    var y2 = box.bbox[3];
    final normalized = [x1, y1, x2, y2].every((value) => value >= 0 && value <= 1) &&
        (x2 > 0 || y2 > 0);
    if (normalized) {
      x1 *= sourceSize.width;
      x2 *= sourceSize.width;
      y1 *= sourceSize.height;
      y2 *= sourceSize.height;
    }

    final left = imageRect.left + (math.min(x1, x2) / sourceSize.width) * imageRect.width;
    final right = imageRect.left + (math.max(x1, x2) / sourceSize.width) * imageRect.width;
    final top = imageRect.top + (math.min(y1, y2) / sourceSize.height) * imageRect.height;
    final bottom = imageRect.top + (math.max(y1, y2) / sourceSize.height) * imageRect.height;
    return Rect.fromLTRB(left, top, right, bottom).intersect(imageRect);
  }

  void _drawLabel(Canvas canvas, Rect imageRect, Rect boxRect, String label, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
      maxLines: 1,
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: math.max(80, imageRect.width - 16));

    final labelWidth = textPainter.width + 12;
    final labelHeight = textPainter.height + 6;
    var left = boxRect.left;
    var top = boxRect.top - labelHeight;
    if (top < imageRect.top) top = boxRect.top;
    if (left + labelWidth > imageRect.right) left = imageRect.right - labelWidth;
    left = math.max(imageRect.left, left);

    final background = RRect.fromRectAndRadius(
      Rect.fromLTWH(left, top, labelWidth, labelHeight),
      const Radius.circular(4),
    );
    canvas.drawRRect(background, Paint()..color = color.withOpacity(0.92));
    textPainter.paint(canvas, Offset(left + 6, top + 3));
  }

  Color _speciesColor(String species) {
    const palette = <Color>[
      Color(0xff2563eb),
      Color(0xffdc2626),
      Color(0xff059669),
      Color(0xffd97706),
      Color(0xff7c3aed),
      Color(0xff0891b2),
      Color(0xffbe123c),
      Color(0xff4d7c0f),
    ];
    final hash = species.codeUnits.fold<int>(0, (value, unit) => value + unit);
    return palette[hash % palette.length];
  }

  @override
  bool shouldRepaint(covariant _DetectionBoxPainter oldDelegate) {
    return oldDelegate.boxes != boxes ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.colorScheme != colorScheme;
  }
}

// ── In-App Video Player ────────────────────────────────────────────────────────

class _PreviewVideoPlayer extends StatefulWidget {
  const _PreviewVideoPlayer({
    required this.path,
    required this.onOpenExternal,
  });
  
  final String path;
  final VoidCallback onOpenExternal;

  @override
  State<_PreviewVideoPlayer> createState() => _PreviewVideoPlayerState();
}

class _PreviewVideoPlayerState extends State<_PreviewVideoPlayer> {
  late final Player _player;
  late final VideoController _controller;
  bool _isError = false;

  @override
  void initState() {
    super.initState();
    _player = Player();
    _controller = VideoController(_player);
    _initPlayer(widget.path);
  }

  Future<void> _initPlayer(String path) async {
    setState(() => _isError = false);
    try {
      await _player.open(Media(path));
      await _player.setPlaylistMode(PlaylistMode.loop);
      await _player.play();
    } catch (e) {
      if (mounted) setState(() => _isError = true);
    }
  }

  @override
  void didUpdateWidget(_PreviewVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.path != widget.path) {
      _initPlayer(widget.path);
    }
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  String _formatDuration(Duration d) {
    final minutes = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (d.inHours > 0) {
      final hours = d.inHours.toString();
      return '$hours:$minutes:$seconds';
    }
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    if (_isError) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline_rounded, size: 56),
          const SizedBox(height: 12),
          const Text('媒体加载失败\n(文件可能已损坏或格式完全不受支持)', textAlign: TextAlign.center),
          const SizedBox(height: 16),
          FilledButton.tonalIcon(
            onPressed: widget.onOpenExternal,
            icon: const Icon(Icons.open_in_new_rounded),
            label: const Text('使用系统默认播放器打开'),
          ),
        ],
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        Video(
          controller: _controller,
          controls: NoVideoControls, // 完全禁用自带的控制条
        ),
        
        // 单击播放/暂停层 (铺满全屏)
        Positioned.fill(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: () {
              _player.state.playing ? _player.pause() : _player.play();
            },
            child: StreamBuilder<bool>(
              stream: _player.stream.playing,
              builder: (context, snapshot) {
                final isPlaying = snapshot.data ?? _player.state.playing;
                return AnimatedOpacity(
                  opacity: isPlaying ? 0.0 : 1.0,
                  duration: const Duration(milliseconds: 200),
                  child: Container(
                    color: Colors.black26,
                    alignment: Alignment.center,
                    child: const Icon(
                      Icons.play_circle_outline_rounded,
                      size: 64,
                      color: Colors.white,
                    ),
                  ),
                );
              },
            ),
          ),
        ),

        // MD3 风格进度条层
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: StreamBuilder<Duration>(
            stream: _player.stream.position,
            builder: (context, positionSnapshot) {
              return StreamBuilder<Duration>(
                stream: _player.stream.duration,
                builder: (context, durationSnapshot) {
                  final position = positionSnapshot.data ?? _player.state.position;
                  final duration = durationSnapshot.data ?? _player.state.duration;

                  double max = duration.inMilliseconds.toDouble();
                  double value = position.inMilliseconds.toDouble();
                  if (max <= 0) max = 1;
                  if (value < 0) value = 0;
                  if (value > max) value = max;

                  return Container(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter,
                        end: Alignment.topCenter,
                        colors: [Colors.black87, Colors.transparent],
                      ),
                    ),
                    padding: const EdgeInsets.only(left: 16, right: 16, bottom: 8, top: 32),
                    child: Row(
                      children: [
                        Text(
                          _formatDuration(position),
                          style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500),
                        ),
                        Expanded(
                          child: SliderTheme(
                            data: SliderTheme.of(context).copyWith(
                              trackHeight: 4,
                              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
                              overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
                              activeTrackColor: Theme.of(context).colorScheme.primary,
                              inactiveTrackColor: Colors.white30,
                              thumbColor: Theme.of(context).colorScheme.primary,
                              overlayColor: Theme.of(context).colorScheme.primary.withOpacity(0.2),
                            ),
                            child: Slider(
                              value: value,
                              max: max,
                              onChanged: (v) {
                                _player.seek(Duration(milliseconds: v.toInt()));
                              },
                            ),
                          ),
                        ),
                        Text(
                          _formatDuration(duration),
                          style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500),
                        ),
                      ],
                    ),
                  );
                },
              );
            },
          ),
        ),

        // 右上角系统应用跳转按钮
        Positioned(
          top: 12,
          right: 12,
          child: IconButton.filledTonal(
            icon: const Icon(Icons.open_in_new_rounded),
            tooltip: '使用系统应用打开',
            onPressed: widget.onOpenExternal,
          ),
        ),
      ],
    );
  }
}