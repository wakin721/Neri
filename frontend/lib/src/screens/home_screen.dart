import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api_client.dart';
import '../models/job.dart';
import '../models/settings.dart';
import '../widgets/section_card.dart';

const _defaultModelDirectory = 'res/model';
const _lastInputPathKey = 'last_input_path';
const _allSpeciesLabel = '全部物种';

class HomeScreen extends StatefulWidget {
  const HomeScreen({required this.apiClient, super.key});

  final NeriApiClient apiClient;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
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
          _selectedPreviewIndex = 0;
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
      appBar: AppBar(
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
        ],
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

    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      children: [
        if (_error != null) _buildErrorBanner(context),
        if (_previewLoading) const LinearProgressIndicator(),
        if (_previewLoading) const SizedBox(height: 12),
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
          LayoutBuilder(
            builder: (context, constraints) {
              return _buildResponsivePreviewWorkspace(
                constraints.maxWidth,
                results,
                selectedIndex,
                selectedItem!,
              );
            },
          ),
      ],
    );
  }

  Widget _buildResponsivePreviewWorkspace(
    double availableWidth,
    List<DetectionItem> results,
    int selectedIndex,
    DetectionItem selectedItem,
  ) {
    final windowHeight = MediaQuery.of(context).size.height;
    final workspaceHeight = (windowHeight - 220).clamp(560.0, 900.0).toDouble();
    final useHorizontalLayout = availableWidth >= 900;

    if (useHorizontalLayout) {
      final listWidth = (availableWidth * 0.28).clamp(280.0, 380.0).toDouble();
      return SizedBox(
        height: workspaceHeight,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              width: listWidth,
              child: _buildPreviewFileList(results, selectedIndex),
            ),
            const VerticalDivider(width: 24),
            Expanded(child: _buildPreviewDetail(selectedItem)),
          ],
        ),
      );
    }

    final listHeight = availableWidth < 600 ? 220.0 : 280.0;
    final detailHeight = (workspaceHeight - listHeight - 16).clamp(560.0, 760.0).toDouble();
    return Column(
      children: [
        SizedBox(
          height: listHeight,
          child: _buildPreviewFileList(results, selectedIndex),
        ),
        const SizedBox(height: 16),
        SizedBox(
          height: detailHeight,
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(flex: 5, child: _buildPreviewImage(item)),
        const SizedBox(height: 16),
        Expanded(
          flex: 4,
          child: ListView(
            children: [
              _buildImageInfoCard(item),
              const SizedBox(height: 12),
              _buildPreviewDetectionControls(item),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPreviewImage(DetectionItem item) {
    final isImage = _isPreviewImage(item);
    final filterApplies = item.detectionBoxes.any((box) => box.species == _previewSpeciesFilter) ||
        item.species.contains(_previewSpeciesFilter);
    final selectedSpecies = filterApplies ? _previewSpeciesFilter : _allSpeciesLabel;
    final visibleBoxes = _filteredPreviewBoxes(item, selectedSpecies: selectedSpecies);
    return Card.filled(
      clipBehavior: Clip.antiAlias,
      child: Container(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        alignment: Alignment.center,
        child: isImage
            ? Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(
                    File(item.path),
                    fit: BoxFit.contain,
                    errorBuilder: (context, error, stackTrace) {
                      return _buildImagePlaceholder('无法读取图片：${item.filename}');
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
                ],
              )
            : _buildImagePlaceholder('当前文件不是可直接预览的图片'),
      ),
    );
  }

  Widget _buildImagePlaceholder(String message) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(Icons.broken_image_rounded, size: 56),
        const SizedBox(height: 12),
        Text(message, textAlign: TextAlign.center),
      ],
    );
  }

  Widget _buildImageInfoCard(DetectionItem item) {
    return Card.outlined(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildInfoLine([
              _buildInlineInfo('文件名', item.filename),
              _buildInlineInfo('文件类型', item.fileType),
            ]),
            _buildInfoLine([
              _buildInlineInfo('拍摄时间', item.dateTaken ?? '未读取'),
              _buildInlineInfo('图像尺寸', _formatImageSize(item)),
              _buildInlineInfo('文件大小', _formatBytes(item.sizeBytes)),
            ]),
            _buildInfoLine([
              _buildInlineInfo('检测结果', _detectionResultLabel(item)),
              _buildInlineInfo('最低置信度', _confidenceLabel(item)),
              _buildInlineInfo('检测时间', _detectionTimeLabel(item)),
            ]),
            if (item.error != null) ...[
              const SizedBox(height: 8),
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
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Wrap(
        spacing: 24,
        runSpacing: 6,
        children: children,
      ),
    );
  }

  Widget _buildInlineInfo(String label, String value) {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 150, maxWidth: 360),
      child: RichText(
        text: TextSpan(
          style: Theme.of(context).textTheme.bodyMedium,
          children: [
            TextSpan(
              text: '$label：',
              style: Theme.of(context).textTheme.labelLarge,
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
    return '${item.width} × ${item.height}';
  }

  String _formatBytes(int? bytes) {
    if (bytes == null) return '未知';
    if (bytes < 1024) return '$bytes B';
    final kb = bytes / 1024;
    if (kb < 1024) return '${kb.toStringAsFixed(1)} KB';
    final mb = kb / 1024;
    if (mb < 1024) return '${mb.toStringAsFixed(1)} MB';
    return '${(mb / 1024).toStringAsFixed(1)} GB';
  }

  String _detectionResultLabel(DetectionItem item) {
    final raw = item.detectionData['物种名称'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString();
    }
    if (item.species.isNotEmpty) return item.species.join('、');
    return '暂无';
  }

  String _confidenceLabel(DetectionItem item) {
    final raw = item.detectionData['最低置信度'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString();
    }
    return item.confidence == null ? '未知' : item.confidence!.toStringAsFixed(3);
  }

  String _detectionTimeLabel(DetectionItem item) {
    final raw = item.detectionData['检测时间'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString();
    }
    return '暂无';
  }

  Widget _buildPreviewDetectionControls(DetectionItem item) {
    final speciesOptions = <String>{
      _allSpeciesLabel,
      ...item.species,
      ...item.detectionBoxes.map((box) => box.species),
    }.where((species) => species.isNotEmpty).toList();
    final selectedSpecies = speciesOptions.contains(_previewSpeciesFilter)
        ? _previewSpeciesFilter
        : _allSpeciesLabel;
    final visibleBoxes = _filteredPreviewBoxes(item, selectedSpecies: selectedSpecies);
    final visibleSpecies = item.species.where((species) {
      final matchesSpecies =
          selectedSpecies == _allSpeciesLabel || species == selectedSpecies;
      final matchesConfidence = item.confidence == null ||
          item.confidence! >= _previewConfidenceThreshold;
      return matchesSpecies && matchesConfidence;
    }).toList();

    return Card.outlined(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('显示检测结果'),
              subtitle: Text(
                _previewShowDetections ? '正在显示当前图像的检测摘要。' : '检测结果已隐藏。',
              ),
              value: _previewShowDetections,
              onChanged: (value) {
                setState(() => _previewShowDetections = value);
              },
            ),
            const SizedBox(height: 8),
            LayoutBuilder(
              builder: (context, constraints) {
                final speciesSelector = DropdownButtonFormField<String>(
                  value: selectedSpecies,
                  decoration: const InputDecoration(
                    labelText: '置信度物种选择',
                    border: OutlineInputBorder(),
                  ),
                  items: speciesOptions.map((species) {
                    return DropdownMenuItem(value: species, child: Text(species));
                  }).toList(),
                  onChanged: (value) {
                    if (value != null) {
                      setState(() => _previewSpeciesFilter = value);
                    }
                  },
                );
                final confidenceSlider = _buildSlider(
                  '置信度',
                  _previewConfidenceThreshold,
                  (value) => setState(() => _previewConfidenceThreshold = value),
                );

                if (constraints.maxWidth >= 720) {
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Expanded(child: speciesSelector),
                      const SizedBox(width: 16),
                      Expanded(child: confidenceSlider),
                    ],
                  );
                }

                return Column(
                  children: [
                    speciesSelector,
                    const SizedBox(height: 12),
                    confidenceSlider,
                  ],
                );
              },
            ),
            const SizedBox(height: 8),
            if (_previewShowDetections)
              _buildDetectionSummary(item, visibleSpecies, visibleBoxes),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.icon(
                onPressed: _previewDetecting || !_isPreviewImage(item)
                    ? null
                    : () => _detectCurrentPreviewImage(item),
                icon: _previewDetecting
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.center_focus_strong_rounded),
                label: Text(_previewDetecting ? '检测中...' : '检测当前图像'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectionSummary(
    DetectionItem item,
    List<String> visibleSpecies,
    List<DetectionBox> visibleBoxes,
  ) {
    if (item.error != null) {
      return Text('检测错误：${item.error}');
    }
    if (item.detectionBoxes.isNotEmpty) {
      if (visibleBoxes.isEmpty) {
        return const Text('当前筛选条件下没有检测框。');
      }
      final counts = <String, int>{};
      final confidences = <double>[];
      for (final box in visibleBoxes) {
        counts[box.species] = (counts[box.species] ?? 0) + 1;
        final confidence = box.confidence;
        if (confidence != null) confidences.add(confidence);
      }
      final confidenceLabel = confidences.isEmpty
          ? '未知'
          : confidences.reduce((a, b) => a < b ? a : b).toStringAsFixed(2);
      return Wrap(
        spacing: 8,
        runSpacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          ...counts.entries.map(
            (entry) => Chip(label: Text('${entry.key} × ${entry.value}')),
          ),
          Chip(label: Text('检测框 ${visibleBoxes.length}')),
          Chip(label: Text('最低置信度 $confidenceLabel')),
        ],
      );
    }
    if (item.species.isEmpty) {
      return const Text('暂无检测结果。可点击“检测当前图像”运行识别。');
    }
    if (visibleSpecies.isEmpty) {
      return const Text('当前筛选条件下没有检测结果。');
    }
    final confidence = item.confidence == null
        ? '未知'
        : item.confidence!.toStringAsFixed(2);
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        ...visibleSpecies.map((species) => Chip(label: Text(species))),
        Chip(label: Text('置信度 $confidence')),
      ],
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
    return _buildPageList([_buildSettingsCard()]);
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
