import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models/job.dart';
import '../widgets/detection_media_viewer.dart';
import '../widgets/selectable_list_card.dart';

import 'package:lpinyin/lpinyin.dart';

typedef MarkValidationItem =
    Future<DetectionItem> Function(
      DetectionItem item,
      String action, {
      String? speciesName,
      String? speciesCount,
      String? speciesType,
      String? remark,
    });

const validationExportColumns = <String>[
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

class SpeciesValidationScreen extends StatefulWidget {
  const SpeciesValidationScreen({
    required this.apiClient,
    required this.inputPath,
    required this.items,
    required this.speciesTypes,
    required this.autoGroup,
    required this.quickMarkSpecies,
    required this.exportColumns,
    required this.onRefresh,
    required this.onLoadMetadata,
    required this.onOpenExternal,
    required this.onMarkItem,
    super.key,
  });

  final NeriApiClient apiClient;
  final String inputPath;
  final List<DetectionItem> items;
  final Map<String, String> speciesTypes;
  final bool autoGroup;
  final List<String> quickMarkSpecies;
  final List<String> exportColumns;
  final Future<void> Function() onRefresh;
  final Future<void> Function(DetectionItem item) onLoadMetadata;
  final void Function(String path) onOpenExternal;
  final MarkValidationItem onMarkItem;

  @override
  State<SpeciesValidationScreen> createState() =>
      _SpeciesValidationScreenState();
}

enum _ValidationListFocus { species, photos }

class _SpeciesValidationScreenState extends State<SpeciesValidationScreen> {
  static const _globalSpecies = 'global';
  static const _quickSpecies = <String>[
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
  static const _quantityOptions = <String>[
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '25', '50',
  ];

  // 使用静态变量持久化保存跨界面的状态
  static String? _savedSelectedPath;
  static String? _savedSelectedBucketKey;
  static String _savedSelectedQuantity = '1';
  static String _savedSelectedSpeciesFilter = _globalSpecies;
  static String _savedExportFormat = 'csv';
  static double _savedConfidence = 0.25;
  static bool _savedShowDetections = true;
  static _ValidationListFocus? _savedActiveList;

  String? get _selectedPath => _savedSelectedPath;
  set _selectedPath(String? value) => _savedSelectedPath = value;

  String? get _selectedBucketKey => _savedSelectedBucketKey;
  set _selectedBucketKey(String? value) => _savedSelectedBucketKey = value;

  String get _selectedQuantity => _savedSelectedQuantity;
  set _selectedQuantity(String value) => _savedSelectedQuantity = value;

  String get _selectedSpeciesFilter => _savedSelectedSpeciesFilter;
  set _selectedSpeciesFilter(String value) => _savedSelectedSpeciesFilter = value;

  String get _exportFormat => _savedExportFormat;
  set _exportFormat(String value) => _savedExportFormat = value;

  double get _confidence => _savedConfidence;
  set _confidence(double value) => _savedConfidence = value;

  bool get _showDetections => _savedShowDetections;
  set _showDetections(bool value) => _savedShowDetections = value;

  _ValidationListFocus? get _activeList => _savedActiveList;
  set _activeList(_ValidationListFocus? value) => _savedActiveList = value;

  // 临时状态（标记中、导出中不需要跨界面保存）
  bool _marking = false;
  bool _exporting = false;
  DetectionItem? _lastMarkedItem;

  @override
  void didUpdateWidget(covariant SpeciesValidationScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.items.isEmpty) {
      _selectedPath = null;
      _selectedBucketKey = null;
      return;
    }
    if (_selectedPath == null ||
        !widget.items.any((item) => item.path == _selectedPath)) {
      _selectedPath = widget.items.first.path;
      unawaited(widget.onLoadMetadata(widget.items.first));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.inputPath.isEmpty) {
      return const Center(child: Text('请先在开始界面设置输入文件夹。'));
    }

    final buckets = _buildBuckets(widget.items);
    if (buckets.isEmpty) {
      _selectedBucketKey = null;
    } else if (_selectedBucketKey == null ||
        !buckets.any((bucket) => bucket.key == _selectedBucketKey)) {
      _selectedBucketKey = buckets.first.key;
    }
    final visibleItems = _visibleItems(buckets);
    final selectedItem = _selectedItem(visibleItems);

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: widget.items.isEmpty
                ? _EmptyValidationState(onRefresh: widget.onRefresh)
                : LayoutBuilder(
                    builder: (context, constraints) {
                      if (constraints.maxWidth < 980) {
                        return _buildNarrowLayout(
                          buckets,
                          visibleItems,
                          selectedItem,
                        );
                      }
                      return _buildWideLayout(
                        constraints.maxWidth,
                        buckets,
                        visibleItems,
                        selectedItem,
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildWideLayout(
    double availableWidth,
    List<_SpeciesBucket> buckets,
    List<DetectionItem> visibleItems,
    DetectionItem selectedItem,
  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    // 将 220.0 修改为 200.0，与预览界面保持完全一致
    final listWidth = (availableWidth * 0.20).clamp(200.0, 300.0).toDouble();
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          width: listWidth,
          child: _buildLeftLists(buckets, visibleItems),
        ),
        const VerticalDivider(width: 24),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(child: _buildImagePanel(selectedItem, visibleBoxes)),
              const SizedBox(height: 10),
              _buildSummaryPanel(selectedItem, visibleBoxes),
              const SizedBox(height: 10),
              _buildFilterPanel(selectedItem, visibleBoxes),
            ],
          ),
        ),
        const SizedBox(width: 16),
        SizedBox(width: 190, child: _buildRightActions(selectedItem)),
      ],
    );
  }

  Widget _buildNarrowLayout(
    List<_SpeciesBucket> buckets,
    List<DetectionItem> visibleItems,
    DetectionItem selectedItem,
  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    return ListView(
      children: [
        SizedBox(
          height: 330,
          child: _buildImagePanel(selectedItem, visibleBoxes),
        ),
        const SizedBox(height: 10),
        SizedBox(height: 260, child: _buildLeftLists(buckets, visibleItems)),
        const SizedBox(height: 10),
        SizedBox(height: 260, child: _buildRightActions(selectedItem)),
        const SizedBox(height: 10),
        _buildSummaryPanel(selectedItem, visibleBoxes),
        const SizedBox(height: 10),
        _buildFilterPanel(selectedItem, visibleBoxes),
      ],
    );
  }

  Widget _buildLeftLists(
    List<_SpeciesBucket> buckets,
    List<DetectionItem> visibleItems,
  ) {
    final selectedBucketIndex = buckets.indexWhere(
      (bucket) => bucket.key == _selectedBucketKey,
    );
    final selectedPhotoIndex = visibleItems.indexWhere(
      (item) => item.path == _selectedPath,
    );
    final speciesRatio = switch (_activeList) {
      _ValidationListFocus.species => 0.66,
      _ValidationListFocus.photos => 0.34,
      null => 0.5,
    };

    return LayoutBuilder(
      builder: (context, constraints) {
        const gap = 10.0;
        final usableHeight = (constraints.maxHeight - gap).clamp(
          0.0,
          double.infinity,
        );
        return TweenAnimationBuilder<double>(
          tween: Tween<double>(end: speciesRatio),
          duration: const Duration(milliseconds: 240),
          curve: Curves.easeOutCubic,
          builder: (context, ratio, _) {
            return Column(
              children: [
                SizedBox(
                  height: usableHeight * ratio,
                  child: SelectableListCard<_SpeciesBucket>(
                    items: buckets,
                    selectedIndex: selectedBucketIndex,
                    padding: const EdgeInsets.symmetric(vertical: 6),
                    // 使用 Icon 区分已校验和未校验状态
                    leadingBuilder: (bucket) => Icon(
                      bucket.isValidated
                          ? Icons.check_circle_rounded
                          : Icons.pending_actions_rounded,
                      color: bucket.isValidated ? Colors.green : Colors.orange,
                    ),
                    titleBuilder: (bucket) => bucket.species,
                    subtitleBuilder: (bucket) => '${bucket.items.length} 张照片',
                    // 取消物种栏后面的数字
                    trailingBuilder: (bucket) => null,
                    onSelected: (index, bucket) {
                      setState(() {
                        _activeList = _ValidationListFocus.species;
                        _selectedBucketKey = bucket.key;
                        _selectedPath = bucket.items.first.path;
                      });
                      unawaited(widget.onLoadMetadata(bucket.items.first));
                    },
                  ),
                ),
                const SizedBox(height: gap),
                SizedBox(
                  height: usableHeight * (1 - ratio),
                  child: SelectableListCard<DetectionItem>(
                    key: ValueKey(_selectedBucketKey),
                    items: visibleItems,
                    selectedIndex: selectedPhotoIndex,
                    padding: const EdgeInsets.symmetric(vertical: 6),
                    leadingBuilder: (item) => Icon(_fileIcon(item)),
                    titleBuilder: (item) => item.filename,
                    subtitleBuilder: (item) {
                      final manual = item.detectionData['物种名称']?.toString().trim();
                      final models = item.detectionBoxes
                          .map((b) => b.species)
                          .where((s) => s.isNotEmpty && s != 'Unknown')
                          .toSet();
                      
                      final parts = <String>[];
                      if (manual != null && manual.isNotEmpty && manual != '未知鸟') {
                        parts.add(manual);
                      } else if (item.species.isNotEmpty) {
                        parts.addAll(item.species);
                      }
                      
                      final modelStrs = models.where((m) => !parts.contains(m)).toList();
                      if (modelStrs.isNotEmpty) {
                        if (parts.isNotEmpty) {
                          parts.add('[模型: ${modelStrs.join('、')}]');
                        } else {
                          parts.addAll(modelStrs);
                        }
                      }
                      
                      return parts.isEmpty ? item.fileType : parts.join(' ');
                    },
                    trailingBuilder: (item) => item.error == null
                        ? null
                        : const Icon(Icons.error_outline_rounded),
                    onSelected: (index, item) {
                      setState(() => _activeList = _ValidationListFocus.photos);
                      _selectItem(item);
                    },
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }

  Widget _buildImagePanel(DetectionItem item, List<DetectionBox> visibleBoxes) {
    // 增加 Card.filled 包装，与 preview_screen.dart 中的图像圆角保持一致
    return Card.filled(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: DetectionMediaViewer(
        item: item,
        visibleBoxes: visibleBoxes,
        showDetections: _showDetections,
        onOpenExternal: () => widget.onOpenExternal(item.path),
      ),
    );
  }

  Widget _buildRightActions(DetectionItem item) {
    final quickSpecies = _quickMarkSpecies(item);
    return Column(
      children: [
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: _ValidationPanel(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8, 10, 8, 8),
                    child: Column(
                      children: [
                        _ActionButton(
                          label: '正确',
                          onPressed: _marking
                              ? null
                              : () => _markSelected('correct'),
                        ),
                        const SizedBox(height: 6),
                        _ActionButton(
                          label: '空',
                          onPressed: _marking
                              ? null
                              : () => _markSelected('empty'),
                        ),
                        const SizedBox(height: 6),
                        const Divider(height: 12), // 添加了空按钮下方的分割线
                        Expanded(
                          child: ListView.separated(
                            itemCount: quickSpecies.length,
                            separatorBuilder: (_, __) =>
                                const SizedBox(height: 6),
                            itemBuilder: (context, index) {
                              final species = quickSpecies[index];
                              return _ActionButton(
                                label: species,
                                onPressed: _marking
                                    ? null
                                    : () => _markSelected(
                                        'update',
                                        speciesName: species,
                                        speciesCount: _selectedQuantity,
                                      ),
                              );
                            },
                          ),
                        ),
                        const Divider(height: 14),
                        _ActionButton(
                          label: '其他',
                          onPressed: _marking ? null : _markOtherSpecies,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 78,
                child: _ValidationPanel(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8, 10, 8, 8),
                    child: Column(
                      children: [
                        Expanded(
                          child: ListView.separated(
                            itemCount: _quantityOptions.length,
                            separatorBuilder: (_, __) =>
                                const SizedBox(height: 6),
                            itemBuilder: (context, index) {
                              final quantity = _quantityOptions[index];
                              return _QuantityButton(
                                label: quantity,
                                selected: quantity == _selectedQuantity,
                                onPressed: () => _markQuantity(quantity),
                              );
                            },
                          ),
                        ),
                        const Divider(height: 14),
                        Tooltip(
                          message: '撤回本次校验',
                          child: IconButton.filledTonal(
                            onPressed: _lastMarkedItem == null || _marking
                                ? null
                                : () => _markSelected(
                                    'unverified',
                                    itemOverride: _lastMarkedItem,
                                  ),
                            icon: const Icon(Icons.undo_rounded),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        _buildExportSplitButton(),
      ],
    );
  }

  // 独立出来的物种信息面板
  Widget _buildSummaryPanel(DetectionItem item, List<DetectionBox> visibleBoxes) {
    final summary = _summaryFor(item, visibleBoxes);
    final colorScheme = Theme.of(context).colorScheme;

    return _ValidationPanel(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: colorScheme.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.analytics_rounded,
                color: colorScheme.onPrimaryContainer,
                size: 20,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                '物种: ${summary.species}  |  数量: ${summary.count}  |  类型: ${summary.type}  |  置信度: ${summary.confidence}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 独立出来的过滤面板 (开关、物种选择、置信度)
  // 独立出来的过滤面板 (开关、物种选择、置信度)
  Widget _buildFilterPanel(DetectionItem item, List<DetectionBox> visibleBoxes) {
    final speciesOptions = <String>{
      _globalSpecies,
      ...item.species,
      ...item.detectionBoxes.map((box) => box.species),
    }.where((value) => value.isNotEmpty).toList();
    
    final selectedSpecies = speciesOptions.contains(_selectedSpeciesFilter)
        ? _selectedSpeciesFilter
        : _globalSpecies;

    return _ValidationPanel(
      child: Padding(
        // 与 PreviewScreen 保持相同的 Padding 来统一高度体验
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            const Text('显示结果', style: TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(width: 4),
            Switch(
              value: _showDetections,
              onChanged: (value) => setState(() => _showDetections = value),
            ),
            const SizedBox(width: 16),
            const Text('置信度', style: TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(width: 4),
            SizedBox(
              width: 150, // 与预览界面相同的 150 宽度
              child: DropdownButtonFormField<String>(
                key: ValueKey(
                  'validation-species-$selectedSpecies-${speciesOptions.join('|')}',
                ),
                initialValue: selectedSpecies,
                isExpanded: true,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 8, // 与预览界面内边距严格一致
                    vertical: 0,
                  ),
                ),
                items: speciesOptions.map((species) {
                  return DropdownMenuItem(
                    value: species,
                    child: Text(
                      species == _globalSpecies ? '全局设置' : species,
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                    ),
                  );
                }).toList(),
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _selectedSpeciesFilter = value);
                  }
                },
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Slider(
                value: _confidence,
                min: 0.05,
                max: 0.95,
                divisions: 90,
                label: _confidence.toStringAsFixed(2),
                onChanged: (value) => setState(() => _confidence = value),
              ),
            ),
            SizedBox(
              width: 36,
              child: Text(_confidence.toStringAsFixed(2)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExportSplitButton() {
    final label = _exportFormat == 'excel' ? '导出 Excel' : '导出 CSV';
    return MenuAnchor(
      menuChildren: [
        MenuItemButton(
          leadingIcon: _exportFormat == 'csv'
              ? const Icon(Icons.check_rounded)
              : const SizedBox(width: 24),
          onPressed: _exporting
              ? null
              : () {
                  setState(() => _exportFormat = 'csv');
                  unawaited(_exportData());
                },
          child: const Text('导出 CSV'),
        ),
        MenuItemButton(
          leadingIcon: _exportFormat == 'excel'
              ? const Icon(Icons.check_rounded)
              : const SizedBox(width: 24),
          onPressed: _exporting
              ? null
              : () {
                  setState(() => _exportFormat = 'excel');
                  unawaited(_exportData());
                },
          child: const Text('导出 Excel'),
        ),
      ],
      builder: (context, controller, child) {
        final isOpen = controller.isOpen;
        final colorScheme = Theme.of(context).colorScheme;
        
        return SizedBox(
          height: 64, // 虽然去掉了 FilterPanel 强制 64px 的限制，但保留按钮本身的 64px 高度仍然能保持很好的视觉重量和点击区
          width: double.infinity,
          child: Stack(
            alignment: Alignment.centerRight,
            children: [
              // 左侧主按钮
              Positioned(
                left: 0,
                top: 0,
                bottom: 0,
                right: 50, // 48(右侧闭合宽度) + 2(缝隙)，保证左侧固定不动
                child: FilledButton.icon(
                  onPressed: _exporting ? null : _exportData,
                  icon: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    transitionBuilder: (child, animation) => 
                        FadeTransition(
                          opacity: animation,
                          child: ScaleTransition(scale: animation, child: child),
                        ),
                    child: _exporting
                        ? const SizedBox(
                            key: ValueKey('loading'),
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                                strokeWidth: 2.5, color: Colors.white),
                          )
                        : const Icon(Icons.ios_share_rounded, key: ValueKey('icon')),
                  ),
                  label: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    transitionBuilder: (child, animation) => 
                        FadeTransition(opacity: animation, child: child),
                    child: Text(
                      label,
                      key: ValueKey(label),
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  style: FilledButton.styleFrom(
                    shape: const RoundedRectangleBorder(
                      borderRadius: BorderRadius.horizontal(
                        left: Radius.circular(32),
                        right: Radius.circular(4), // 极小圆角以匹配拆分效果
                      ),
                    ),
                  ),
                ),
              ),
              
              // 右侧下拉按钮 (悬浮/展开时变成圆形并重叠覆盖在左侧按钮上)
              AnimatedPositioned(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeOutCubic,
                right: 0,
                top: 0,
                bottom: 0,
                width: isOpen ? 48 : 48, // 展开时变为正方形(配合borderRadius变成正圆)，闭合时较窄
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeOutCubic,
                  decoration: BoxDecoration(
                    color: _exporting 
                        ? colorScheme.onSurface.withOpacity(0.12)
                        : colorScheme.primary,
                    borderRadius: BorderRadius.horizontal(
                      left: Radius.circular(isOpen ? 32 : 4), // 展开时左侧变圆
                      right: const Radius.circular(32),
                    ),
                    boxShadow: isOpen && !_exporting
                        ? [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.15),
                              blurRadius: 10,
                              offset: const Offset(-2, 0), // 添加一点向左的阴影增强层叠立体感
                            )
                          ]
                        : [],
                  ),
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      borderRadius: BorderRadius.horizontal(
                        left: Radius.circular(isOpen ? 32 : 4),
                        right: const Radius.circular(32),
                      ),
                      onTap: _exporting
                          ? null
                          : () => isOpen ? controller.close() : controller.open(),
                      child: Center(
                        child: AnimatedRotation(
                          turns: isOpen ? 0.5 : 0, // 箭头反转动画 180度
                          duration: const Duration(milliseconds: 300),
                          curve: Curves.easeOutCubic,
                          child: Icon(
                            Icons.keyboard_arrow_down_rounded, // 默认向下
                            color: _exporting 
                                ? colorScheme.onSurface.withOpacity(0.38)
                                : colorScheme.onPrimary,
                            size: isOpen ? 28 : 24, // 展开时图标稍微放大一点点
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _selectItem(DetectionItem item) {
    setState(() => _selectedPath = item.path);
    unawaited(widget.onLoadMetadata(item));
  }

  IconData _fileIcon(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase())
        ? Icons.image_rounded
        : Icons.movie_rounded;
  }

  DetectionItem _selectedItem(List<DetectionItem> visibleItems) {
    if (visibleItems.isEmpty) return widget.items.first;
    return visibleItems.firstWhere(
      (item) => item.path == _selectedPath,
      orElse: () => visibleItems.first,
    );
  }

  List<DetectionItem> _visibleItems(List<_SpeciesBucket> buckets) {
    if (_selectedBucketKey == null) return widget.items;
    return buckets
        .firstWhere(
          (bucket) => bucket.key == _selectedBucketKey,
          orElse: () =>
              buckets.isEmpty ? _SpeciesBucket.empty() : buckets.first,
        )
        .items;
  }

  List<_SpeciesBucket> _buildBuckets(List<DetectionItem> items) {
    final map = <String, _SpeciesBucket>{};
    for (final item in items) {
      final species = _primarySpecies(item);
      final isValidated = item.validated == true || item.detectionData['最低置信度'] == '人工校验';
      final key = '${isValidated ? 1 : 0}::$species';
      map.putIfAbsent(
        key,
        () => _SpeciesBucket(
          key: key,
          isValidated: isValidated,
          species: species,
          items: <DetectionItem>[],
        ),
      );
      map[key]!.items.add(item);
    }
    final buckets = map.values.toList()
      ..sort((a, b) {
        if (a.isValidated != b.isValidated) {
          return a.isValidated ? 1 : -1;
        }
        return a.species.compareTo(b.species);
      });
    return buckets;
  }

  List<DetectionBox> _filteredBoxes(DetectionItem item) {
    return item.detectionBoxes.where((box) {
      final matchesSpecies =
          _selectedSpeciesFilter == _globalSpecies ||
          box.species == _selectedSpeciesFilter;
      final confidence = box.confidence;
      final matchesConfidence = confidence == null || confidence >= _confidence;
      return matchesSpecies && matchesConfidence && box.bbox.length >= 4;
    }).toList();
  }

  String _primarySpecies(DetectionItem item) {
    final raw = item.detectionData['物种名称'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString().trim();
    }
    if (item.species.isNotEmpty) return item.species.join(',');
    if (item.error != null) return '错误';
    return '未知鸟';
  }

  List<String> _quickMarkSpecies(DetectionItem item) {
    final configuredQuickSpecies = widget.quickMarkSpecies.isEmpty
        ? _quickSpecies
        : widget.quickMarkSpecies;
    final values = <String>{
      ...item.species,
      ...item.detectionBoxes.map((box) => box.species),
      ...configuredQuickSpecies,
    };
    values.removeWhere(
      (value) => value.trim().isEmpty || value == 'Unknown' || value == '空',
    );
    return values.take(12).toList();
  }

  _DetectionSummary _summaryFor(
    DetectionItem item,
    List<DetectionBox> visibleBoxes,
  ) {
    var species = _primarySpecies(item);
    var count = item.detectionData['物种数量']?.toString() ?? '';
    var confidence = item.detectionData['最低置信度']?.toString() ?? '';

    if (visibleBoxes.isNotEmpty && item.detectionData['最低置信度'] != '人工校验') {
      final counts = <String, int>{};
      final confidences = <double>[];
      for (final box in visibleBoxes) {
        counts[box.species] = (counts[box.species] ?? 0) + 1;
        if (box.confidence != null) confidences.add(box.confidence!);
      }
      species = counts.keys.join(',');
      count = counts.values.join(',');
      if (confidences.isNotEmpty) {
        confidence = confidences
            .reduce((a, b) => a < b ? a : b)
            .toStringAsFixed(2);
      }
    }

    if (count.trim().isEmpty) {
      count = species == '空'
          ? '空'
          : (visibleBoxes.isEmpty ? '未知' : visibleBoxes.length.toString());
    }
    if (confidence.trim().isEmpty) {
      confidence = item.confidence?.toStringAsFixed(2) ?? 'N/A';
    }

    return _DetectionSummary(
      species: species,
      count: count,
      type: _typeLabel(species, item),
      confidence: confidence,
    );
  }

  String _typeLabel(String species, [DetectionItem? item]) {
    final manualType = item?.detectionData['物种类型']?.toString().trim();
    if (manualType != null && manualType.isNotEmpty) return manualType;
    if (species == '空' || species == '未知鸟') return '空';
    final matchedTypes = species
        .split(',')
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty)
        .map((name) => widget.speciesTypes[name] ?? '')
        .where((speciesType) => speciesType.isNotEmpty)
        .toSet()
        .toList();
    if (matchedTypes.isNotEmpty) return matchedTypes.join(',');
    if (species.contains('人') || species.contains('牧民')) return '人员';
    return '待补全';
  }

  void _markQuantity(String quantity) {
    setState(() => _selectedQuantity = quantity);
    final item = widget.items.firstWhere(
      (candidate) => candidate.path == _selectedPath,
      orElse: () => widget.items.first,
    );
    final species = _primarySpecies(item);
    if (species != '未知鸟' && species != '错误') {
      unawaited(
        _markSelected('update', speciesName: species, speciesCount: quantity),
      );
    }
  }

  Future<void> _markOtherSpecies() async {
    final item = widget.items.firstWhere(
      (candidate) => candidate.path == _selectedPath,
      orElse: () => widget.items.first,
    );
    
    final summary = _summaryFor(item, _filteredBoxes(item));
    
    final initialSpecies = item.detectionData['物种名称']?.toString() ?? (summary.species == '未知鸟' ? '' : summary.species);
    final initialCount = item.detectionData['物种数量']?.toString() ?? summary.count;
    final initialType = item.detectionData['物种类型']?.toString() ?? (summary.type == '待补全' ? '' : summary.type);
    final initialRemark = item.detectionData['备注']?.toString() ?? '';

    final draft = await showDialog<_OtherSpeciesDraft>(
      context: context,
      builder: (context) {
        return _OtherSpeciesDialog(
          initialSpecies: initialSpecies,
          initialCount: initialCount,
          initialType: initialType,
          initialRemark: initialRemark,
          speciesTypes: widget.speciesTypes,
        );
      },
    );
    
    if (draft == null || draft.speciesName.isEmpty || !mounted) return;
    await _markSelected(
      'update',
      speciesName: draft.speciesName,
      speciesCount: draft.speciesCount.isEmpty
          ? _selectedQuantity
          : draft.speciesCount,
      speciesType: draft.speciesType,
      remark: draft.remark,
    );
  }

  Future<void> _markSelected(
    String action, {
    DetectionItem? itemOverride,
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    if (widget.items.isEmpty) return;
    final item =
        itemOverride ??
        widget.items.firstWhere(
          (candidate) => candidate.path == _selectedPath,
          orElse: () => widget.items.first,
        );
    final visibleItems = _visibleItems(_buildBuckets(widget.items));
    final currentIndex = visibleItems.indexWhere(
      (candidate) => candidate.path == item.path,
    );
    final nextPath = currentIndex >= 0 && currentIndex + 1 < visibleItems.length
        ? visibleItems[currentIndex + 1].path
        : null;

    setState(() => _marking = true);
    try {
      final updated = await widget.onMarkItem(
        item,
        action,
        speciesName: speciesName,
        speciesCount: speciesCount,
        speciesType: speciesType,
        remark: remark,
      );
      if (!mounted) return;
      setState(() {
        _lastMarkedItem = action == 'unverified' ? null : updated;
        _selectedPath = nextPath ?? updated.path;
      });
      final message = action == 'unverified'
          ? '已撤回校验标记'
          : '已标记 ${updated.filename}';
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(message)));
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('标记失败：$error')));
    } finally {
      if (mounted) setState(() => _marking = false);
    }
  }

  Future<void> _exportData() async {
    setState(() => _exporting = true);
    try {
      final confidenceSettings = _selectedSpeciesFilter == _globalSpecies
          ? {'global': _confidence}
          : {'global': 0.25, _selectedSpeciesFilter: _confidence};
      final result = await widget.apiClient.exportValidationData(
        inputPath: widget.inputPath,
        fileFormat: _exportFormat,
        columnsToExport: widget.exportColumns.isEmpty
            ? validationExportColumns
            : widget.exportColumns,
        confidenceSettings: confidenceSettings,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '已导出 ${result.exportedCount} 条记录到 ${result.outputPath}',
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('导出失败：$error')));
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }
}

class _ValidationPanel extends StatelessWidget {
  const _ValidationPanel({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    // 将此处重构为 Card.outlined，使其拥有与预览界面（PreviewScreen）完全相同的 Material 3 圆角和外边框
    return Card.outlined(
      margin: EdgeInsets.zero,
      child: child,
    );
  }
}

class _ActionButton extends StatelessWidget {
  const _ActionButton({required this.label, required this.onPressed});

  final String label;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 36,
      child: FilledButton.tonal(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        child: Text(label, maxLines: 1, overflow: TextOverflow.ellipsis),
      ),
    );
  }
}

class _QuantityButton extends StatelessWidget {
  const _QuantityButton({
    required this.label,
    required this.selected,
    required this.onPressed,
  });

  final String label;
  final bool selected;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 30,
      child: FilledButton.tonal(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          padding: EdgeInsets.zero,
          backgroundColor: selected
              ? Theme.of(context).colorScheme.primaryContainer
              : null,
          foregroundColor: selected
              ? Theme.of(context).colorScheme.onPrimaryContainer
              : null,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(13),
          ),
        ),
        child: Text(label),
      ),
    );
  }
}

class _EmptyValidationState extends StatelessWidget {
  const _EmptyValidationState({required this.onRefresh});

  final Future<void> Function() onRefresh;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.fact_check_outlined, size: 56),
          const SizedBox(height: 12),
          const Text('暂无待校验图像。'),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: onRefresh,
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('重新读取'),
          ),
        ],
      ),
    );
  }
}

class _SpeciesBucket {
  _SpeciesBucket({
    required this.key,
    required this.isValidated,
    required this.species,
    required this.items,
  });

  factory _SpeciesBucket.empty() {
    return _SpeciesBucket(
      key: '',
      isValidated: false,
      species: '',
      items: const <DetectionItem>[],
    );
  }

  final String key;
  final bool isValidated;
  final String species;
  final List<DetectionItem> items;
}

class _DetectionSummary {
  const _DetectionSummary({
    required this.species,
    required this.count,
    required this.type,
    required this.confidence,
  });

  final String species;
  final String count;
  final String type;
  final String confidence;
}

class _OtherSpeciesDraft {
  const _OtherSpeciesDraft({
    required this.speciesName,
    required this.speciesCount,
    required this.speciesType,
    required this.remark,
  });

  final String speciesName;
  final String speciesCount;
  final String speciesType;
  final String remark;
}

class _OtherSpeciesDialog extends StatefulWidget {
  const _OtherSpeciesDialog({
    required this.initialSpecies,
    required this.initialCount,
    required this.initialType,
    required this.initialRemark,
    required this.speciesTypes,
  });

  final String initialSpecies;
  final String initialCount;
  final String initialType;
  final String initialRemark;
  final Map<String, String> speciesTypes;

  @override
  State<_OtherSpeciesDialog> createState() => _OtherSpeciesDialogState();
}

class _OtherSpeciesDialogState extends State<_OtherSpeciesDialog> {
  late final TextEditingController speciesController;
  late final TextEditingController countController;
  late final TextEditingController typeController;
  late final TextEditingController remarkController;

  List<String> _suggestions = [];

  @override
  void initState() {
    super.initState();
    speciesController = TextEditingController(text: widget.initialSpecies);
    countController = TextEditingController(text: widget.initialCount);
    typeController = TextEditingController(text: widget.initialType);
    remarkController = TextEditingController(text: widget.initialRemark);

    speciesController.addListener(_onSpeciesChanged);
  }

  @override
  void dispose() {
    speciesController.removeListener(_onSpeciesChanged);
    speciesController.dispose();
    countController.dispose();
    typeController.dispose();
    remarkController.dispose();
    super.dispose();
  }

  List<String> _getMatches(String input) {
    final trimInput = input.trim();
    if (trimInput.isEmpty) return [];

    final matches = <String>[];
    final isPinyin = RegExp(r'^[a-zA-Z]+$').hasMatch(trimInput);
    final lowerInput = trimInput.toLowerCase();

    for (final species in widget.speciesTypes.keys) {
      if (isPinyin) {
        final initials = PinyinHelper.getShortPinyin(species).toLowerCase();
        if (initials.startsWith(lowerInput)) {
          matches.add(species);
        }
      } else {
        if (species.contains(trimInput)) {
          matches.add(species);
        }
      }
    }
    return matches;
  }

  void _onSpeciesChanged() {
    final text = speciesController.text;
    if (text.isEmpty) {
      setState(() => _suggestions = []);
      typeController.text = '';
      countController.text = '';
      return;
    }

    final parts = text.split(RegExp(r'[,，]'));
    final currentWord = parts.last;
    
    setState(() {
      _suggestions = _getMatches(currentWord);
    });

    if (text.endsWith(',') || text.endsWith('，')) {
      bool changed = false;
      final resolvedParts = <String>[];
      
      for (int i = 0; i < parts.length - 1; i++) {
        final part = parts[i].trim();
        final partMatches = _getMatches(part);
        if (RegExp(r'^[a-zA-Z]+$').hasMatch(part) && partMatches.isNotEmpty) {
          resolvedParts.add(partMatches.first);
          changed = true;
        } else {
          resolvedParts.add(part); 
        }
      }

      if (changed) {
        final newText = '${resolvedParts.join(', ')}, ';
        if (newText.replaceAll(' ', '') != text.replaceAll(' ', '')) {
          speciesController.value = TextEditingValue(
            text: newText,
            selection: TextSelection.collapsed(offset: newText.length),
          );
          return; 
        }
      }
    }

    final activeSpecies = <String>[];
    for (final part in parts) {
      final p = part.trim();
      if (p.isEmpty) continue;
      
      if (!RegExp(r'^[a-zA-Z]+$').hasMatch(p)) {
        activeSpecies.add(p);
      } else {
        final pMatches = _getMatches(p);
        if (pMatches.isNotEmpty) {
          activeSpecies.add(pMatches.first);
        } else {
          // 核心修复1：如果是未识别的纯英文字母（如 animal），也要保留，否则数量会变成0
          activeSpecies.add(p);
        }
      }
    }

    final validEntities = activeSpecies.where((e) => e != '空' && e != '未知鸟').toList();
    
    // 核心修复2：动态填充物种类型，支持 1对1 逗号分隔
    if (validEntities.isNotEmpty) {
      final resolvedTypes = <String>[];
      for (final s in validEntities) {
        final type = widget.speciesTypes[s];
        resolvedTypes.add((type != null && type.isNotEmpty) ? type : '待补全');
      }
      final newTypeStr = resolvedTypes.join(',');
      if (typeController.text != newTypeStr) {
         typeController.text = newTypeStr;
      }
    } else {
      typeController.text = activeSpecies.isNotEmpty ? '待补全' : '';
    }

    // 核心修复3：动态填充数量，保留现有数值，新增物种默认补 1，以逗号分隔
    if (validEntities.isNotEmpty) {
      final currentCounts = countController.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
      final newCounts = <String>[];
      
      for (int i = 0; i < validEntities.length; i++) {
        if (i < currentCounts.length) {
           final parsed = int.tryParse(currentCounts[i]);
           if (parsed != null && parsed > 0) {
               newCounts.add(currentCounts[i]); // 保留原有的数字（比如 6）
           } else {
               newCounts.add('1');
           }
        } else {
           newCounts.add('1'); // 新增的物种默认填 1
        }
      }
      
      final newCountStr = newCounts.join(',');
      if (countController.text != newCountStr) {
        countController.text = newCountStr;
      }
    } else {
      if (activeSpecies.contains('空') || activeSpecies.contains('未知鸟')) {
         countController.text = activeSpecies.first;
      } else {
         countController.text = '';
      }
    }
  }

  void _onSuggestionTapped(String species) {
    final text = speciesController.text;
    final parts = text.split(RegExp(r'[,，]'));
    
    parts[parts.length - 1] = species;
    
    final newText = '${parts.join(', ')}, ';
    speciesController.value = TextEditingValue(
      text: newText,
      selection: TextSelection.collapsed(offset: newText.length),
    );
  }

  void _submit() {
    final text = speciesController.text;
    final parts = text.split(RegExp(r'[,，]'));
    final finalSpeciesList = <String>[];

    for (final part in parts) {
      final p = part.trim();
      if (p.isEmpty) continue;
      if (RegExp(r'^[a-zA-Z]+$').hasMatch(p)) {
        final matches = _getMatches(p);
        if (matches.isNotEmpty) {
           finalSpeciesList.add(matches.first);
        } else {
           finalSpeciesList.add(p);
        }
      } else {
        finalSpeciesList.add(p);
      }
    }

    Navigator.of(context).pop(
      _OtherSpeciesDraft(
        speciesName: finalSpeciesList.join(','),
        speciesCount: countController.text.trim(),
        speciesType: typeController.text.trim(),
        remark: remarkController.text.trim(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('其他标注'),
      content: SizedBox(
        width: 380,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: speciesController,
              autofocus: true,
              decoration: const InputDecoration(
                labelText: '物种名称 (拼音首字母，逗号分隔)',
                hintText: '如: cmy(赤麻鸭), bl(白鹭)',
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _submit(),
            ),
            
            AnimatedSize(
              duration: const Duration(milliseconds: 200),
              child: _suggestions.isEmpty
                  ? const SizedBox.shrink()
                  : Padding(
                      padding: const EdgeInsets.only(top: 10.0, bottom: 4.0),
                      child: Wrap(
                        spacing: 8.0,
                        runSpacing: 8.0, // 核心修复4：修改 runSpacing 为正数，解决按钮重叠问题
                        children: _suggestions.take(8).map((species) {
                          return ActionChip(
                            label: Text(species, style: const TextStyle(fontSize: 13)),
                            visualDensity: VisualDensity.compact,
                            onPressed: () => _onSuggestionTapped(species),
                          );
                        }).toList(),
                      ),
                    ),
            ),
            const SizedBox(height: 12),
            
            TextField(
              controller: countController,
              decoration: const InputDecoration(
                labelText: '物种数量',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: typeController,
              decoration: const InputDecoration(
                labelText: '物种类型',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: remarkController,
              minLines: 2,
              maxLines: 3,
              decoration: const InputDecoration(
                labelText: '备注',
                border: OutlineInputBorder(),
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
          onPressed: _submit,
          child: const Text('确定'),
        ),
      ],
    );
  }
}