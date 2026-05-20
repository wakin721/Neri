import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api_client.dart';
import '../models/job.dart';
import '../widgets/app_menu_style.dart';
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

typedef MarkValidationItems =
    Future<List<DetectionItem>> Function(
      List<DetectionItem> items,
      String action, {
      String? speciesName,
      String? speciesCount,
      String? speciesType,
      String? remark,
    });

typedef RedetectValidationItems =
    Future<void> Function(
      List<DetectionItem> items, {
      required double confidence,
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

const double _validationButtonHeight = 40;

class SpeciesValidationScreen extends StatefulWidget {
  const SpeciesValidationScreen({
    required this.apiClient,
    required this.inputPath,
    required this.items,
    required this.loading,
    required this.refreshVersion,
    required this.speciesTypes,
    required this.autoGroup,
    required this.collapseGroups,
    required this.autoGroupDetectBurst,
    required this.autoGroupBurstSize,
    required this.autoGroupGapSeconds,
    required this.autoSortQuickMarks,
    required this.undoSteps,
    required this.quickMarkSpecies,
    required this.quickMarkRecentHistory,
    required this.quickMarkUsageCounts,
    required this.exportColumns,
    required this.onRefresh,
    required this.onLoadMetadata,
    required this.onOpenExternal,
    required this.onMarkItem,
    required this.onMarkItems,
    required this.onQuickMarkUsed,
    required this.onRedetectItems,
    super.key,
  });

  final NeriApiClient apiClient;
  final String inputPath;
  final List<DetectionItem> items;
  final bool loading;
  final int refreshVersion;
  final Map<String, String> speciesTypes;
  final bool autoGroup;
  final bool collapseGroups;
  final bool autoGroupDetectBurst;
  final int autoGroupBurstSize;
  final int autoGroupGapSeconds;
  final bool autoSortQuickMarks;
  final int undoSteps;
  final List<String> quickMarkSpecies;
  final List<String> quickMarkRecentHistory;
  final Map<String, int> quickMarkUsageCounts;
  final List<String> exportColumns;
  final Future<void> Function() onRefresh;
  final Future<void> Function(DetectionItem item) onLoadMetadata;
  final void Function(String path) onOpenExternal;
  final MarkValidationItem onMarkItem;
  final MarkValidationItems onMarkItems;
  final Future<void> Function(String speciesName) onQuickMarkUsed;
  final RedetectValidationItems onRedetectItems;

  @override
  State<SpeciesValidationScreen> createState() =>
      _SpeciesValidationScreenState();
}

enum _ValidationListFocus { species, photos }

class _MarkHistoryEntry {
  _MarkHistoryEntry(Iterable<DetectionItem> items) : items = items.toList();

  final List<DetectionItem> items;
}

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

  // 使用静态变量持久化保存跨界面的状态
  static String? _savedSelectedPath;
  static String? _savedSelectedBucketKey;
  static String _savedSelectedQuantity = '1';
  static bool _savedShowQuantitySelection = false;
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

  bool get _showQuantitySelection => _savedShowQuantitySelection;
  set _showQuantitySelection(bool value) => _savedShowQuantitySelection = value;

  String get _selectedSpeciesFilter => _savedSelectedSpeciesFilter;
  set _selectedSpeciesFilter(String value) =>
      _savedSelectedSpeciesFilter = value;

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
  final List<_MarkHistoryEntry> _markHistory = <_MarkHistoryEntry>[];
  String? _pendingSpeciesName;
  String? _pendingQuantity;
  String? _pendingItemPath;
  final List<String> _sessionQuickMarkHistory = <String>[];
  final Set<String> _selectedPaths = <String>{};
  String? _selectionAnchorPath;
  String? _selectedGroupSignature;
  bool? _bucketCacheAutoGroup;
  bool? _bucketCacheAutoGroupDetectBurst;
  int? _bucketCacheBurstSize;
  int? _bucketCacheGapSeconds;
  String? _bucketCachePathSignature;
  String? _bucketCacheGroupingSignature;
  int? _bucketCacheRefreshVersion;
  List<DetectionItem>? _bucketCacheItemsIdentity;
  List<_SpeciesBucket> _bucketCache = const <_SpeciesBucket>[];
  Map<String, DetectionItem> _bucketCacheItemByPath =
      const <String, DetectionItem>{};
  Map<String, _ValidationGroupIndexEntry> _bucketCacheGroupByPath =
      const <String, _ValidationGroupIndexEntry>{};
  Map<String, _ValidationMediaGroup> _bucketCacheGroupBySignature =
      const <String, _ValidationMediaGroup>{};
  Map<String, int> _bucketCacheGroupIndexBySignature = const <String, int>{};
  int _bucketCacheGroupCount = 0;
  final Map<String, _MediaTimestampCacheEntry> _mediaTimestampCache =
      <String, _MediaTimestampCacheEntry>{};
  final Map<String, String> _groupSpeciesLabelCache = <String, String>{};
  final Set<String> _deferredRegroupGroupSignatures = <String>{};
  final Set<String> _expandedGroupSignatures = <String>{};

  bool get _useCollapsedGroups => widget.autoGroup && widget.collapseGroups;
  int get _undoHistoryLimit => widget.undoSteps.clamp(1, 50).toInt();
  String get _undoTooltip => '撤回上一步校验';

  @override
  void didUpdateWidget(covariant SpeciesValidationScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.inputPath != widget.inputPath) {
      _markHistory.clear();
    }
    if (oldWidget.autoGroup != widget.autoGroup ||
        oldWidget.collapseGroups != widget.collapseGroups ||
        oldWidget.autoGroupDetectBurst != widget.autoGroupDetectBurst ||
        oldWidget.autoGroupBurstSize != widget.autoGroupBurstSize ||
        oldWidget.autoGroupGapSeconds != widget.autoGroupGapSeconds) {
      _expandedGroupSignatures.clear();
      _selectedGroupSignature = null;
    }
    if (widget.items.isEmpty) {
      _selectedPath = null;
      _selectedBucketKey = null;
      _selectedPaths.clear();
      _markHistory.clear();
      _selectionAnchorPath = null;
      _selectedGroupSignature = null;
      _bucketCacheAutoGroup = null;
      _bucketCacheAutoGroupDetectBurst = null;
      _bucketCacheBurstSize = null;
      _bucketCacheGapSeconds = null;
      _bucketCachePathSignature = null;
      _bucketCacheGroupingSignature = null;
      _bucketCacheRefreshVersion = null;
      _bucketCacheItemsIdentity = null;
      _bucketCache = const <_SpeciesBucket>[];
      _bucketCacheItemByPath = const <String, DetectionItem>{};
      _bucketCacheGroupByPath = const <String, _ValidationGroupIndexEntry>{};
      _bucketCacheGroupBySignature = const <String, _ValidationMediaGroup>{};
      _bucketCacheGroupIndexBySignature = const <String, int>{};
      _bucketCacheGroupCount = 0;
      _mediaTimestampCache.clear();
      _groupSpeciesLabelCache.clear();
      _deferredRegroupGroupSignatures.clear();
      _expandedGroupSignatures.clear();
      return;
    }
    final validPaths = widget.items.map((item) => item.path).toSet();
    _selectedPaths.removeWhere((path) => !validPaths.contains(path));
    _discardMarkHistoryForPaths(
      _markHistory
          .expand((entry) => entry.items)
          .map((item) => item.path)
          .where((path) => !validPaths.contains(path))
          .toSet(),
    );
    if (_selectionAnchorPath != null &&
        !validPaths.contains(_selectionAnchorPath)) {
      _selectionAnchorPath = null;
    }
    if (_selectedPath == null ||
        !widget.items.any((item) => item.path == _selectedPath)) {
      _selectedPath = widget.items.first.path;
      _selectedPaths
        ..clear()
        ..add(widget.items.first.path);
      _selectionAnchorPath = widget.items.first.path;
      _setSelectedGroupFromPath(widget.items.first.path);
      _resetPendingMark();
      unawaited(widget.onLoadMetadata(widget.items.first));
    } else if (_selectedPaths.isEmpty && _selectedPath != null) {
      _selectedPaths.add(_selectedPath!);
      _selectionAnchorPath = _selectedPath;
      _setSelectedGroupFromPath(_selectedPath);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.inputPath.isEmpty) {
      return const Center(child: Text('请先在开始界面设置输入文件夹。'));
    }
    if (widget.items.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: _EmptyValidationState(
          loading: widget.loading,
          onRefresh: widget.onRefresh,
        ),
      );
    }

    final buckets = _currentBuckets();
    if (buckets.isEmpty) {
      _selectedBucketKey = null;
    } else if (_selectedBucketKey == null ||
        !buckets.any((bucket) => bucket.key == _selectedBucketKey)) {
      _selectedBucketKey = buckets.first.key;
    }
    final visibleRows = _visibleRows(buckets);
    if (visibleRows.isNotEmpty && !_hasVisibleSelection(visibleRows)) {
      _applyRowSelection(visibleRows.first);
      unawaited(widget.onLoadMetadata(visibleRows.first.item));
    } else if (_selectedPath != null && _selectedPaths.isEmpty) {
      _selectedPaths.add(_selectedPath!);
      _selectionAnchorPath = _selectedPath;
      _setSelectedGroupFromPath(_selectedPath, buckets: buckets);
    }
    final selectedItem = _selectedItemFromRows(visibleRows);

    return Theme(
      data: Theme.of(context).copyWith(
        menuTheme: MenuThemeData(
          style: MenuStyle(
            shape: WidgetStatePropertyAll(
              RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16), // 设置为你想要的大圆角，推荐 12-16
              ),
            ),
          ),
        ),
      ),
      child: Focus(
        autofocus: true,
        onKeyEvent: (node, event) => _handleKeyEvent(event),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: widget.items.isEmpty
                    ? _EmptyValidationState(
                        loading: widget.loading,
                        onRefresh: widget.onRefresh,
                      )
                    : LayoutBuilder(
                        builder: (context, constraints) {
                          if (constraints.maxWidth < 980) {
                            return _buildNarrowLayout(
                              buckets,
                              visibleRows,
                              selectedItem,
                            );
                          }
                          return _buildWideLayout(
                            constraints.maxWidth,
                            buckets,
                            visibleRows,
                            selectedItem,
                          );
                        },
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  KeyEventResult _handleKeyEvent(KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
      _moveSelectedPhoto(-1);
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
      _moveSelectedPhoto(1);
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  void _moveSelectedPhoto(int delta) {
    final visibleItems = _visibleItems(_currentBuckets());
    if (visibleItems.isEmpty) return;
    final currentIndex = visibleItems.indexWhere(
      (item) => item.path == _selectedPath,
    );
    final nextIndex = (currentIndex < 0 ? 0 : currentIndex + delta)
        .clamp(0, visibleItems.length - 1)
        .toInt();
    final nextItem = visibleItems[nextIndex];
    if (nextItem.path == _selectedPath) return;
    setState(() {
      _activeList = _ValidationListFocus.photos;
      _selectedPath = nextItem.path;
      _selectedPaths
        ..clear()
        ..add(nextItem.path);
      _selectionAnchorPath = nextItem.path;
      _setSelectedGroupFromItem(nextItem);
      _resetPendingMark();
    });
    unawaited(widget.onLoadMetadata(nextItem));
  }

  Widget _buildWideLayout(
    double availableWidth,
    List<_SpeciesBucket> buckets,
    List<_ValidationFileRow> visibleRows,
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
          child: _buildLeftLists(buckets, visibleRows),
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
    List<_ValidationFileRow> visibleRows,
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
        SizedBox(height: 260, child: _buildLeftLists(buckets, visibleRows)),
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
    List<_ValidationFileRow> visibleRows,
  ) {
    final colorScheme = Theme.of(context).colorScheme;
    final selectedBucketIndex = buckets.indexWhere(
      (bucket) => bucket.key == _selectedBucketKey,
    );
    final selectedPhotoIndex = _selectedPhotoRowIndex(visibleRows);
    final speciesRatio = switch (_activeList) {
      _ValidationListFocus.species => 0.66,
      _ValidationListFocus.photos => 0.34,
      null => 0.5,
    };

    // 添加这一行，解析出用于底部交互的 visibleItems
    final visibleItems = _itemsFromRows(visibleRows);

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
                    subtitleBuilder: (bucket) => widget.autoGroup
                        ? '${bucket.groups.length} 组 / ${bucket.items.length} 个文件'
                        : '${bucket.items.length} 个文件',
                    // 取消物种栏后面的数字
                    trailingBuilder: (bucket) {
                      if (bucket.key != _selectedBucketKey) return null;
                      return Icon(
                        Icons.check_circle_rounded,
                        color: colorScheme.primary,
                        size: 20,
                      );
                    },
                    onSelected: (index, bucket) {
                      setState(() {
                        _activeList = _ValidationListFocus.species;
                        _selectedBucketKey = bucket.key;
                        _selectedPath = bucket.items.first.path;
                        _selectedPaths
                          ..clear()
                          ..add(bucket.items.first.path);
                        _selectionAnchorPath = bucket.items.first.path;
                        _setSelectedGroupFromPath(
                          bucket.items.first.path,
                          buckets: buckets,
                        );
                        _resetPendingMark();
                      });
                      unawaited(widget.onLoadMetadata(bucket.items.first));
                    },
                  ),
                ),
                const SizedBox(height: gap),
                SizedBox(
                  height: usableHeight * (1 - ratio),
                  child: SelectableListCard<_ValidationFileRow>(
                    key: ValueKey(_selectedBucketKey),
                    items: visibleRows,
                    selectedIndex: selectedPhotoIndex,
                    isSelected: (_, row) => _isRowSelected(row),
                    padding: const EdgeInsets.symmetric(vertical: 6),
                    leadingBuilder: (row) {
                      final selected = _isRowSelected(row);
                      return Icon(
                        row.isGroupHeader
                            ? Icons.folder_copy_rounded
                            : _fileIcon(row.item),
                        color: selected ? colorScheme.primary : null,
                      );
                    },
                    titleBuilder: (row) {
                      if (row.isGroupHeader) {
                        return _groupHeaderTitle(row);
                      }
                      return row.item.filename;
                    },
                    subtitleBuilder: (row) {
                      if (row.isGroupHeader) {
                        return _groupHeaderSubtitle(row);
                      }
                      return _fileListSubtitle(row);
                    },
                    tileColorBuilder: (index, row) {
                      if (!widget.autoGroup) return null;
                      final groupIndex = row.groupIndex;
                      if (groupIndex.isEven) return null;
                      return colorScheme.surfaceContainerHighest.withValues(
                        alpha: 0.42,
                      );
                    },
                    // 修改：重构冗余声明并修复错乱的变量(expandedGroup等)
                    trailingBuilder: (row) {
                      final selected = _isRowSelected(row);
                      final hasError = row.isGroupHeader
                          ? row.group.items.any(
                              (candidate) => candidate.error != null,
                            )
                          : row.item.error != null;
                      final showDownArrow =
                          row.isGroupHeader && !row.groupExpanded && !selected;
                      final showUpArrow =
                          row.isGroupHeader && row.groupExpanded;
                      final showSelectedCheck =
                          selected && !(row.isGroupHeader && row.groupExpanded);

                      Widget groupToggleButton({
                        required IconData icon,
                        required Color color,
                        required String tooltip,
                      }) {
                        return IconButton(
                          tooltip: tooltip,
                          visualDensity: VisualDensity.compact,
                          constraints: const BoxConstraints.tightFor(
                            width: 32,
                            height: 32,
                          ),
                          padding: EdgeInsets.zero,
                          onPressed: () => _toggleGroupExpansionForRow(row),
                          icon: Icon(icon, size: 20, color: color),
                        );
                      }

                      if (showUpArrow) {
                        return groupToggleButton(
                          icon: Icons.expand_less_rounded,
                          color: colorScheme.onSurfaceVariant,
                          tooltip: '折叠分组',
                        );
                      }

                      if (!selected &&
                          !hasError &&
                          !showDownArrow &&
                          !showSelectedCheck) {
                        return null;
                      }
                      return Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          if (hasError)
                            Icon(
                              Icons.error_outline_rounded,
                              color: colorScheme.error,
                              size: 20,
                            ),
                          if (showDownArrow) ...[
                            if (hasError) const SizedBox(width: 6),
                            groupToggleButton(
                              icon: Icons.expand_more_rounded,
                              color: colorScheme.onSurfaceVariant,
                              tooltip: '展开分组',
                            ),
                          ],
                          if (showSelectedCheck) ...[
                            if (hasError || row.isGroupHeader)
                              const SizedBox(width: 6),
                            row.isGroupHeader
                                ? groupToggleButton(
                                    icon: Icons.check_circle_rounded,
                                    color: colorScheme.primary,
                                    tooltip: row.groupExpanded
                                        ? '折叠分组'
                                        : '展开分组',
                                  )
                                : Icon(
                                    Icons.check_circle_rounded,
                                    color: colorScheme.primary,
                                    size: 20,
                                  ),
                          ],
                        ],
                      );
                    },
                    onSelected: (index, row) {
                      setState(() => _activeList = _ValidationListFocus.photos);
                      _selectItemWithKeyboard(row.item, visibleItems, row: row);
                    },
                    onMenuOpening: (index, row) {
                      _prepareBatchContextMenu(row);
                    },
                    menuChildrenBuilder: (context, index, row) {
                      return _buildBatchMenuChildren(row, visibleItems);
                    },
                    onLongPress: (index, row) {
                      _toggleItemSelection(row.item, row: row);
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
                                selected:
                                    _pendingItemPath == item.path &&
                                    _pendingSpeciesName == species,
                                onPressed: _marking
                                    ? null
                                    : () => _markQuickSpecies(species),
                              );
                            },
                          ),
                        ),
                        const Divider(height: 14),
                        _ActionButton(
                          label: '其他',
                          height: _validationButtonHeight,
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
                                selected:
                                    _showQuantitySelection &&
                                    quantity == _selectedQuantity,
                                onPressed: _marking
                                    ? null
                                    : () => _markQuantity(quantity),
                              );
                            },
                          ),
                        ),
                        const Divider(height: 14),
                        Tooltip(
                          message: _undoTooltip,
                          child: SizedBox(
                            width: _validationButtonHeight,
                            height: _validationButtonHeight,
                            child: IconButton.filledTonal(
                              onPressed: _markHistory.isEmpty || _marking
                                  ? null
                                  : _undoRecentMarks,
                              icon: const Icon(Icons.undo_rounded),
                              padding: EdgeInsets.zero,
                              constraints: const BoxConstraints.tightFor(
                                width: _validationButtonHeight,
                                height: _validationButtonHeight,
                              ),
                            ),
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
        _buildExportControls(),
      ],
    );
  }

  // 独立出来的物种信息面板
  Widget _buildSummaryPanel(
    DetectionItem item,
    List<DetectionBox> visibleBoxes,
  ) {
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
  Widget _buildFilterPanel(
    DetectionItem item,
    List<DetectionBox> visibleBoxes,
  ) {
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
            const SizedBox(width: 8),
            _ValidationMenuSelect(
              width: 150,
              label: '物种',
              value: selectedSpecies,
              options: speciesOptions,
              labelBuilder: (species) =>
                  species == _globalSpecies ? '全局设置' : species,
              onChanged: (value) =>
                  setState(() => _selectedSpeciesFilter = value),
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
            SizedBox(width: 36, child: Text(_confidence.toStringAsFixed(2))),
          ],
        ),
      ),
    );
  }

  Widget _buildExportControls() {
    final label = _exportFormat == 'excel' ? '导出 Excel' : '导出 CSV';
    return _ValidationPanel(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SegmentedButton<String>(
              segments: const [
                ButtonSegment<String>(value: 'csv', label: Text('CSV')),
                ButtonSegment<String>(value: 'excel', label: Text('Excel')),
              ],
              selected: {_exportFormat},
              onSelectionChanged: _exporting
                  ? null
                  : (selection) {
                      if (selection.isEmpty) return;
                      setState(() => _exportFormat = selection.first);
                    },
            ),
            const SizedBox(height: 8),
            FilledButton.icon(
              onPressed: _exporting ? null : _exportData,
              style: FilledButton.styleFrom(
                minimumSize: const Size(0, _validationButtonHeight),
              ),
              icon: _exporting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.ios_share_rounded),
              label: Text(label),
            ),
          ],
        ),
      ),
    );
  }

  void _selectItem(DetectionItem item) {
    setState(() {
      _selectedPath = item.path;
      _activeList = _ValidationListFocus.photos;
      _selectedPaths
        ..clear()
        ..add(item.path);
      _selectionAnchorPath = item.path;
      _setSelectedGroupFromItem(item);
      _resetPendingMark();
    });
    unawaited(widget.onLoadMetadata(item));
  }

  void _selectItemWithKeyboard(
    DetectionItem item,
    List<DetectionItem> visibleItems, {
    _ValidationFileRow? row,
  }) {
    final targetItem = row != null && row.isGroupHeader ? row.item : item;
    final pressedKeys = HardwareKeyboard.instance.logicalKeysPressed;
    final rangeSelection =
        pressedKeys.contains(LogicalKeyboardKey.shiftLeft) ||
        pressedKeys.contains(LogicalKeyboardKey.shiftRight);
    final toggleSelection =
        pressedKeys.contains(LogicalKeyboardKey.controlLeft) ||
        pressedKeys.contains(LogicalKeyboardKey.controlRight) ||
        pressedKeys.contains(LogicalKeyboardKey.metaLeft) ||
        pressedKeys.contains(LogicalKeyboardKey.metaRight);

    setState(() {
      _selectedPath = targetItem.path;
      _activeList = _ValidationListFocus.photos;
      if (rangeSelection && _selectionAnchorPath != null) {
        _selectedPaths
          ..clear()
          ..addAll(_rangeSelectionPaths(visibleItems, targetItem.path));
      } else if (toggleSelection) {
        if (_selectedPaths.contains(targetItem.path) &&
            _selectedPaths.length > 1) {
          _selectedPaths.remove(targetItem.path);
        } else {
          _selectedPaths.add(targetItem.path);
        }
        _selectionAnchorPath = targetItem.path;
      } else {
        _selectedPaths
          ..clear()
          ..add(targetItem.path);
        _selectionAnchorPath = targetItem.path;
      }
      if (row == null) {
        _setSelectedGroupFromItem(targetItem);
      } else {
        _setSelectedGroupFromRow(row);
      }
      _resetPendingMark();
    });
    unawaited(widget.onLoadMetadata(targetItem));
  }

  void _toggleItemSelection(DetectionItem item, {_ValidationFileRow? row}) {
    final targetItem = row != null && row.isGroupHeader ? row.item : item;
    setState(() {
      _selectedPath = targetItem.path;
      _activeList = _ValidationListFocus.photos;
      if (_selectedPaths.contains(targetItem.path) &&
          _selectedPaths.length > 1) {
        _selectedPaths.remove(targetItem.path);
      } else {
        _selectedPaths.add(targetItem.path);
      }
      _selectionAnchorPath = targetItem.path;
      if (row == null) {
        _setSelectedGroupFromItem(targetItem);
      } else {
        _setSelectedGroupFromRow(row);
      }
      _resetPendingMark();
    });
    unawaited(widget.onLoadMetadata(targetItem));
  }

  Iterable<String> _rangeSelectionPaths(
    List<DetectionItem> visibleItems,
    String targetPath,
  ) sync* {
    final anchorIndex = visibleItems.indexWhere(
      (item) => item.path == _selectionAnchorPath,
    );
    final targetIndex = visibleItems.indexWhere(
      (item) => item.path == targetPath,
    );
    if (anchorIndex < 0 || targetIndex < 0) {
      yield targetPath;
      return;
    }
    final start = anchorIndex < targetIndex ? anchorIndex : targetIndex;
    final end = anchorIndex < targetIndex ? targetIndex : anchorIndex;
    for (var index = start; index <= end; index++) {
      yield visibleItems[index].path;
    }
  }

  List<DetectionItem> _selectedItemsIn(List<DetectionItem> visibleItems) {
    final selected = visibleItems
        .where((item) => _selectedPaths.contains(item.path))
        .toList();
    if (selected.isNotEmpty) return selected;
    final fallback = _selectedItem(visibleItems);
    return <DetectionItem>[fallback];
  }

  List<DetectionItem> _groupAwareSelectedItems(
    _ValidationFileRow row,
    List<DetectionItem> visibleItems,
  ) {
    if (row.isGroupHeader && row.groupExpanded) {
      return row.group.items;
    }

    final baseItems = _selectedPaths.contains(row.item.path)
        ? _selectedItemsIn(visibleItems)
        : <DetectionItem>[row.item];
    if (row.isGroupHeader || _useCollapsedGroups) {
      return _expandCollapsedGroupTargets(baseItems);
    }
    return baseItems;
  }

  void _prepareBatchContextMenu(_ValidationFileRow row) {
    if (row.isGroupHeader) {
      final groupHasSelection = row.group.items.any(
        (item) => _selectedPaths.contains(item.path),
      );
      final activeItemInGroup =
          _selectedPath != null &&
          row.group.items.any((item) => item.path == _selectedPath);
      DetectionItem? itemToLoad;
      setState(() {
        _activeList = _ValidationListFocus.photos;
        _setSelectedGroupFromRow(row);
        if (!groupHasSelection) {
          _selectedPaths
            ..clear()
            ..add(row.item.path);
          _selectionAnchorPath = row.item.path;
        }
        if (!activeItemInGroup) {
          _selectedPath = row.item.path;
          itemToLoad = row.item;
        }
        _resetPendingMark();
      });
      if (itemToLoad != null) {
        unawaited(widget.onLoadMetadata(itemToLoad!));
      }
      return;
    }

    if (!_selectedPaths.contains(row.item.path)) {
      _selectItem(row.item);
      return;
    }

    setState(() {
      _activeList = _ValidationListFocus.photos;
      _setSelectedGroupFromRow(row);
    });
  }

  List<Widget> _buildBatchMenuChildren(
    _ValidationFileRow row,
    List<DetectionItem> visibleItems,
  ) {
    final selectedItems = _groupAwareSelectedItems(row, visibleItems);
    final groupContext = _groupContextForItem(row.item);
    _ValidationGroupContext? groupToggleContext;
    if (_useCollapsedGroups &&
        groupContext != null &&
        groupContext.group.items.length > 1) {
      groupToggleContext = groupContext;
    }
    final count = selectedItems.length;
    return [
      if (groupToggleContext != null) ...[
        MenuItemButton(
          leadingIcon: Icon(
            groupToggleContext.expanded
                ? Icons.expand_less_rounded
                : Icons.expand_more_rounded,
          ),
          onPressed: () => _toggleGroupExpansion(groupToggleContext!),
          child: Text(groupToggleContext.expanded ? '折叠分组' : '展开分组'),
        ),
        const Padding(
          padding: EdgeInsets.symmetric(vertical: 4.0),
          child: Divider(height: 1, indent: 12, endIndent: 12),
        ),
      ],
      MenuItemButton(
        leadingIcon: const Icon(Icons.check_circle_rounded),
        onPressed: _marking
            ? null
            : () => unawaited(_markBatch(selectedItems, 'correct')),
        child: _BatchMenuItem(label: '批量标记为正确', count: count),
      ),
      MenuItemButton(
        leadingIcon: const Icon(Icons.edit_note_rounded),
        onPressed: _marking
            ? null
            : () => unawaited(_markOtherSpecies(itemsOverride: selectedItems)),
        child: _BatchMenuItem(label: '批量标记为其他', count: count),
      ),
      MenuItemButton(
        leadingIcon: const Icon(Icons.hide_image_rounded),
        onPressed: _marking
            ? null
            : () => unawaited(_markBatch(selectedItems, 'empty')),
        child: _BatchMenuItem(label: '批量标记为空', count: count),
      ),
      const Padding(
        padding: EdgeInsets.symmetric(vertical: 4.0),
        child: Divider(height: 1, indent: 12, endIndent: 12),
      ),
      MenuItemButton(
        leadingIcon: const Icon(Icons.auto_fix_high_rounded),
        onPressed: _marking
            ? null
            : () => unawaited(_redetectBatch(selectedItems)),
        child: _BatchMenuItem(label: '批量重新检测', count: count),
      ),
    ];
  }

  IconData _fileIcon(DetectionItem item) {
    return _isImage(item) ? Icons.image_rounded : Icons.movie_rounded;
  }

  bool _isImage(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }

  DetectionItem _selectedItem(List<DetectionItem> visibleItems) {
    if (visibleItems.isEmpty) {
      if (widget.items.isEmpty) {
        throw StateError('No validation item is available.');
      }
      return widget.items.first;
    }
    return visibleItems.firstWhere(
      (item) => item.path == _selectedPath,
      orElse: () => visibleItems.first,
    );
  }

  List<DetectionItem> _visibleItems(List<_SpeciesBucket> buckets) {
    if (_selectedBucketKey == null) return widget.items;
    final bucket = _selectedBucketFrom(buckets);
    if (!_useCollapsedGroups) return bucket.items;

    final visibleItems = <DetectionItem>[];
    for (final group in bucket.groups) {
      if (group.items.isEmpty) continue;
      final signature = _groupSignature(group);
      if (_expandedGroupSignatures.contains(signature)) {
        visibleItems.addAll(group.items);
      } else {
        visibleItems.add(_preferredGroupItem(group));
      }
    }
    return visibleItems;
  }

  _SpeciesBucket _selectedBucketFrom(List<_SpeciesBucket> buckets) {
    return buckets.firstWhere(
      (bucket) => bucket.key == _selectedBucketKey,
      orElse: () => buckets.isEmpty ? _SpeciesBucket.empty() : buckets.first,
    );
  }

  _ValidationGroupIndexEntry? _groupIndexEntryForPath(
    String? path, {
    List<_SpeciesBucket>? buckets,
  }) {
    if (path == null) return null;
    if (buckets == null || identical(buckets, _bucketCache)) {
      return _bucketCacheGroupByPath[path];
    }
    for (final bucket in buckets) {
      for (var index = 0; index < bucket.groups.length; index++) {
        final group = bucket.groups[index];
        final signature = _groupSignature(group);
        for (final item in group.items) {
          if (item.path == path) {
            return _ValidationGroupIndexEntry(
              index: _globalGroupIndexForSignature(
                signature,
                fallback: index,
              ),
              group: group,
              signature: signature,
            );
          }
        }
      }
    }
    return null;
  }

  _ValidationGroupContext? _groupContextForItem(
    DetectionItem item, {
    List<_SpeciesBucket>? buckets,
  }) {
    if (!widget.autoGroup) return null;
    final entry = _groupIndexEntryForPath(
      item.path,
      buckets: buckets ?? _currentBuckets(),
    );
    if (entry == null) return null;
    return _ValidationGroupContext(
      index: entry.index,
      group: entry.group,
      signature: entry.signature,
      expanded: _expandedGroupSignatures.contains(entry.signature),
    );
  }

  bool _isCollapsedGroupRow(
    DetectionItem item,
    _ValidationGroupContext? groupContext,
  ) {
    return _useCollapsedGroups &&
        groupContext != null &&
        !groupContext.expanded &&
        groupContext.group.items.length > 1 &&
        groupContext.group.items.any(
          (candidate) => candidate.path == item.path,
        );
  }

  List<DetectionItem> _expandCollapsedGroupTargets(
    Iterable<DetectionItem> items,
  ) {
    if (!_useCollapsedGroups) return items.toList();
    final selectedByPath = <String, DetectionItem>{};
    for (final item in items) {
      final groupContext = _groupContextForItem(item);
      final useWholeGroup = _isCollapsedGroupRow(item, groupContext);
      final targets = useWholeGroup && groupContext != null
          ? groupContext.group.items
          : <DetectionItem>[item];
      for (final target in targets) {
        selectedByPath[target.path] = target;
      }
    }
    return selectedByPath.values.toList();
  }

  List<DetectionItem> _selectedActionTargets(DetectionItem item) {
    final visibleItems = _visibleItems(_currentBuckets());
    final baseItems = _selectedPaths.contains(item.path)
        ? _selectedItemsIn(visibleItems)
        : <DetectionItem>[item];
    return _expandCollapsedGroupTargets(baseItems);
  }

  String? _groupSignatureForPath(
    String? path, {
    List<_SpeciesBucket>? buckets,
  }) {
    if (!_useCollapsedGroups || path == null) return null;
    return _groupIndexEntryForPath(
      path,
      buckets: buckets ?? _currentBuckets(),
    )?.signature;
  }

  void _setSelectedGroupFromPath(
    String? path, {
    List<_SpeciesBucket>? buckets,
  }) {
    _selectedGroupSignature = _groupSignatureForPath(path, buckets: buckets);
  }

  void _setSelectedGroupFromItem(DetectionItem item) {
    _setSelectedGroupFromPath(item.path);
  }

  void _setSelectedGroupFromRow(_ValidationFileRow row) {
    _selectedGroupSignature = _useCollapsedGroups ? row.group.signature : null;
  }

  void _toggleGroupExpansionForRow(_ValidationFileRow row) {
    if (!row.isGroupHeader) return;
    _toggleGroupExpansion(
      _ValidationGroupContext(
        index: row.groupIndex,
        group: row.group,
        signature: row.group.signature,
        expanded: row.groupExpanded,
      ),
    );
  }

  void _toggleGroupExpansion(_ValidationGroupContext groupContext) {
    DetectionItem? nextItemToLoad;
    final groupPaths = groupContext.group.items
        .map((item) => item.path)
        .toSet();
    final representative = _preferredGroupItem(groupContext.group);

    setState(() {
      if (groupContext.expanded) {
        _expandedGroupSignatures.remove(groupContext.signature);
        if (_selectedPath != null && groupPaths.contains(_selectedPath)) {
          _selectedPath = representative.path;
          _selectedPaths
            ..clear()
            ..add(representative.path);
          _selectionAnchorPath = representative.path;
          _selectedGroupSignature = groupContext.signature;
          nextItemToLoad = representative;
        }
      } else {
        _expandedGroupSignatures.add(groupContext.signature);
        _selectedPath = representative.path;
        _selectedGroupSignature = groupContext.signature;
        _selectedPaths
          ..clear()
          ..add(representative.path);
        _selectionAnchorPath = representative.path;
        nextItemToLoad = representative;
      }
      _activeList = _ValidationListFocus.photos;
      _resetPendingMark();
    });

    if (nextItemToLoad != null) {
      unawaited(widget.onLoadMetadata(nextItemToLoad!));
    }
  }

  List<_ValidationMediaGroup> _buildValidationGroups(List<DetectionItem> items) {
    return widget.autoGroup
        ? _buildAutoGroups(items)
        : [
            for (final item in _sortValidationItems(items))
              _ValidationMediaGroup(<DetectionItem>[item]),
          ];
  }

  List<_SpeciesBucket> _buildBuckets(List<_ValidationMediaGroup> groups) {
    final map = <String, _SpeciesBucket>{};

    for (final group in groups) {
      final species = widget.autoGroup
          ? _groupSpeciesLabel(group)
          : _primarySpecies(group.items.first);
      final isValidated = group.items.every(_isItemValidated);
      final key = '${isValidated ? 1 : 0}::$species';
      map.putIfAbsent(
        key,
        () => _SpeciesBucket(
          key: key,
          isValidated: isValidated,
          species: species,
          items: <DetectionItem>[],
          groups: <_ValidationMediaGroup>[],
        ),
      );
      map[key]!
        ..items.addAll(group.items)
        ..groups.add(group);
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

  List<_SpeciesBucket> _currentBuckets() {
    final hasSameGroupingSettings =
        _bucketCacheAutoGroup == widget.autoGroup &&
        _bucketCacheAutoGroupDetectBurst == widget.autoGroupDetectBurst &&
        _bucketCacheBurstSize == widget.autoGroupBurstSize &&
        _bucketCacheGapSeconds == widget.autoGroupGapSeconds &&
        _bucketCacheRefreshVersion == widget.refreshVersion;

    if (hasSameGroupingSettings &&
        identical(_bucketCacheItemsIdentity, widget.items)) {
      if (!_hasCompletedDeferredGroup()) {
        return _bucketCache;
      }
    }

    final pathSignature = _itemsPathSignature(widget.items);
    final groupingSignature = _itemsGroupingSignature(widget.items);
    final canReuseCache =
        _bucketCachePathSignature == pathSignature &&
        _bucketCacheGroupingSignature == groupingSignature &&
        hasSameGroupingSettings;

    if (canReuseCache) {
      _updateCachedBucketItems();
      if (!_hasCompletedDeferredGroup()) {
        return _bucketCache;
      }
    }

    final canDeferRegroup =
        _deferredRegroupGroupSignatures.isNotEmpty &&
        _bucketCachePathSignature == pathSignature &&
        hasSameGroupingSettings;
    if (canDeferRegroup) {
      _bucketCacheGroupingSignature = groupingSignature;
      _updateCachedBucketItems();
      if (!_hasCompletedDeferredGroup()) {
        return _bucketCache;
      }
    }

    return _rebuildBucketCache(pathSignature, groupingSignature);
  }

  List<_SpeciesBucket> _rebuildBucketCache(
    String pathSignature,
    String groupingSignature,
  ) {
    _bucketCacheAutoGroup = widget.autoGroup;
    _bucketCacheAutoGroupDetectBurst = widget.autoGroupDetectBurst;
    _bucketCacheBurstSize = widget.autoGroupBurstSize;
    _bucketCacheGapSeconds = widget.autoGroupGapSeconds;
    _bucketCachePathSignature = pathSignature;
    _bucketCacheGroupingSignature = groupingSignature;
    _bucketCacheRefreshVersion = widget.refreshVersion;
    _bucketCacheItemsIdentity = widget.items;
    _groupSpeciesLabelCache.clear();
    final groups = _buildValidationGroups(widget.items);
    _rebuildGlobalGroupIndexes(groups);
    _bucketCache = _buildBuckets(groups);
    _rebuildBucketLookups();
    _deferredRegroupGroupSignatures.clear();
    _trimExpandedGroupSignatures();
    return _bucketCache;
  }

  void _rebuildGlobalGroupIndexes(List<_ValidationMediaGroup> groups) {
    _bucketCacheGroupCount = groups.length;
    _bucketCacheGroupIndexBySignature = <String, int>{
      for (var index = 0; index < groups.length; index++)
        _groupSignature(groups[index]): index,
    };
  }

  int _globalGroupIndexForSignature(String signature, {int fallback = 0}) {
    return _bucketCacheGroupIndexBySignature[signature] ?? fallback;
  }

  void _trimExpandedGroupSignatures() {
    final currentSignatures = <String>{};
    for (final bucket in _bucketCache) {
      for (final group in bucket.groups) {
        currentSignatures.add(_groupSignature(group));
      }
    }
    if (_expandedGroupSignatures.isNotEmpty) {
      _expandedGroupSignatures.removeWhere(
        (signature) => !currentSignatures.contains(signature),
      );
    }
    if (_selectedGroupSignature != null &&
        !currentSignatures.contains(_selectedGroupSignature)) {
      _selectedGroupSignature = null;
    }
  }

  String _itemsPathSignature(List<DetectionItem> items) {
    return items.map((item) => item.path).join('\u001f');
  }

  String _itemsGroupingSignature(List<DetectionItem> items) {
    return items
        .map(
          (item) => [
            item.path,
            item.dateTaken ?? '',
            item.detectionData['拍摄时间']?.toString() ?? '',
            item.modifiedAt ?? '',
            item.fileType,
            _primarySpecies(item),
            _isItemValidated(item),
            item.error ?? '',
          ].join('\u001d'),
        )
        .join('\u001f');
  }

  String _groupSignature(_ValidationMediaGroup group) {
    return group.signature;
  }

  void _updateCachedBucketItems() {
    _bucketCacheItemsIdentity = widget.items;
    final latestByPath = <String, DetectionItem>{
      for (final item in widget.items) item.path: item,
    };
    _bucketCacheItemByPath = latestByPath;
    if (_bucketCache.isEmpty) return;
    for (final bucket in _bucketCache) {
      for (var index = 0; index < bucket.items.length; index++) {
        bucket.items[index] =
            latestByPath[bucket.items[index].path] ?? bucket.items[index];
      }
      for (final group in bucket.groups) {
        for (var index = 0; index < group.items.length; index++) {
          group.items[index] =
              latestByPath[group.items[index].path] ?? group.items[index];
        }
      }
    }
    _rebuildBucketLookups();
  }

  void _rebuildBucketLookups() {
    _bucketCacheItemByPath = <String, DetectionItem>{
      for (final item in widget.items) item.path: item,
    };
    _mediaTimestampCache.removeWhere(
      (path, _) => !_bucketCacheItemByPath.containsKey(path),
    );
    final groupByPath = <String, _ValidationGroupIndexEntry>{};
    final groupBySignature = <String, _ValidationMediaGroup>{};
    for (final bucket in _bucketCache) {
      for (var index = 0; index < bucket.groups.length; index++) {
        final group = bucket.groups[index];
        final signature = _groupSignature(group);
        groupBySignature[signature] = group;
        final entry = _ValidationGroupIndexEntry(
          index: _globalGroupIndexForSignature(signature, fallback: index),
          group: group,
          signature: signature,
        );
        for (final item in group.items) {
          groupByPath[item.path] = entry;
        }
      }
    }
    _bucketCacheGroupByPath = groupByPath;
    _bucketCacheGroupBySignature = groupBySignature;
  }

  void _deferRegroupForItems(Iterable<DetectionItem> items) {
    if (!widget.autoGroup) return;
    final targetPaths = items.map((item) => item.path).toSet();
    if (targetPaths.isEmpty) return;
    _currentBuckets();
    for (final path in targetPaths) {
      final entry = _bucketCacheGroupByPath[path];
      if (entry != null) {
        _deferredRegroupGroupSignatures.add(entry.signature);
      }
    }
  }

  bool _hasCompletedDeferredGroup() {
    if (_deferredRegroupGroupSignatures.isEmpty) return false;
    for (final signature in _deferredRegroupGroupSignatures) {
      final group = _bucketCacheGroupBySignature[signature];
      if (group == null || _groupComplete(group)) {
        return true;
      }
    }
    return false;
  }

  bool _groupComplete(_ValidationMediaGroup group) {
    return group.items.isNotEmpty && group.items.every(_isItemValidated);
  }

  List<_ValidationMediaGroup> _buildAutoGroups(List<DetectionItem> items) {
    final orderedItems = _sortValidationItems(items);
    final groups = <_ValidationMediaGroup>[];
    var current = <DetectionItem>[];
    final gapThreshold = Duration(
      seconds: widget.autoGroupGapSeconds.clamp(1, 3600).toInt(),
    );
    final burstSize = widget.autoGroupDetectBurst
        ? _inferAutoGroupBurstSize(orderedItems, gapThreshold)
        : widget.autoGroupBurstSize.clamp(1, 20).toInt();

    void flushCurrent() {
      if (current.isEmpty) return;
      final groupItems = List<DetectionItem>.from(current);
      groups.add(_ValidationMediaGroup(groupItems));
      current = <DetectionItem>[];
    }

    for (final item in orderedItems) {
      if (_shouldStartNewAutoGroup(
        current,
        item,
        burstSize: burstSize,
        gapThreshold: gapThreshold,
      )) {
        flushCurrent();
      }
      current.add(item);
      if (_isVideo(item)) {
        flushCurrent();
      }
    }

    flushCurrent();

    return groups;
  }

  int _inferAutoGroupBurstSize(
    List<DetectionItem> orderedItems,
    Duration gapThreshold,
  ) {
    final counts = <int, int>{};
    var runLength = 0;
    DetectionItem? previous;

    void finishRun() {
      if (runLength <= 0) return;
      final clamped = runLength.clamp(1, 20).toInt();
      counts[clamped] = (counts[clamped] ?? 0) + 1;
      runLength = 0;
    }

    for (final item in orderedItems) {
      final previousItem = previous;
      final startsNewRun =
          previousItem != null &&
          (_isVideo(previousItem) ||
              _isVideo(item) ||
              ((_mediaGap(previousItem, item) ?? Duration.zero) >=
                  gapThreshold));
      if (startsNewRun) finishRun();
      if (_isImage(item)) {
        runLength += 1;
      } else {
        finishRun();
      }
      previous = item;
    }
    finishRun();

    if (counts.isEmpty) return widget.autoGroupBurstSize.clamp(1, 20).toInt();
    final entries = counts.entries.toList()
      ..sort((a, b) {
        final countCompare = b.value.compareTo(a.value);
        if (countCompare != 0) return countCompare;
        return b.key.compareTo(a.key);
      });
    return entries.first.key;
  }

  bool _shouldStartNewAutoGroup(
    List<DetectionItem> current,
    DetectionItem next, {
    required int burstSize,
    required Duration gapThreshold,
  }) {
    if (current.isEmpty) return false;
    if (_isVideo(current.last)) return true;

    final nextIsVideo = _isVideo(next);
    if (nextIsVideo) {
      final gap = _mediaGap(current.last, next);
      return gap != null && gap >= gapThreshold;
    }

    final gap = _mediaGap(current.last, next);
    if (gap != null) return gap >= gapThreshold;

    final photoCount = current.where(_isImage).length;
    return photoCount >= burstSize && _isImage(next);
  }

  Duration? _mediaGap(DetectionItem previous, DetectionItem next) {
    final previousTime = _mediaSortTimestamp(previous);
    final nextTime = _mediaSortTimestamp(next);
    if (previousTime == null || nextTime == null) return null;
    final gap = nextTime.difference(previousTime).abs();
    return gap;
  }

  String _primarySpeciesForGroup(_ValidationMediaGroup group) {
    final orderedValidSpecies = <String>[];
    final seenValidSpecies = <String>{};
    final fallbackVotes = <String, _SpeciesVote>{};

    for (final item in group.items) {
      final speciesNames = _splitGroupSpeciesNames(_primarySpecies(item));
      final validSpeciesNames = speciesNames
          .where((species) => !_isFallbackSpecies(species))
          .toList();

      if (validSpeciesNames.isNotEmpty) {
        for (final species in validSpeciesNames) {
          if (seenValidSpecies.add(species)) {
            orderedValidSpecies.add(species);
          }
        }
        continue;
      }

      final fallbackNames = speciesNames.isEmpty
          ? <String>['未检测']
          : speciesNames;
      for (final species in fallbackNames) {
        final vote = fallbackVotes.putIfAbsent(species, _SpeciesVote.new);
        vote.count += 1;
        if (_isItemValidated(item)) vote.validatedCount += 1;
      }
    }

    if (orderedValidSpecies.isNotEmpty) return orderedValidSpecies.join('、');
    if (fallbackVotes.isNotEmpty) return _topVotedSpecies(fallbackVotes);
    return '未检测';
  }

  String _groupSpeciesLabel(_ValidationMediaGroup group) {
    final signature = _groupSignature(group);
    return _groupSpeciesLabelCache.putIfAbsent(
      signature,
      () => _primarySpeciesForGroup(group),
    );
  }

  String _topVotedSpecies(Map<String, _SpeciesVote> votes) {
    final entries = votes.entries.toList()
      ..sort((a, b) {
        final countCompare = b.value.count.compareTo(a.value.count);
        if (countCompare != 0) return countCompare;
        final validatedCompare = b.value.validatedCount.compareTo(
          a.value.validatedCount,
        );
        if (validatedCompare != 0) return validatedCompare;
        return a.key.compareTo(b.key);
      });
    return entries.first.key;
  }

  bool _isFallbackSpecies(String species) {
    final normalized = species.trim();
    if (normalized.isEmpty || normalized == '空') return true;
    if (normalized.toLowerCase() == 'empty') return true;
    return const {'Unknown', '未检测', '错误', '需人工校验'}.contains(species);
  }

  List<String> _splitGroupSpeciesNames(String speciesName) {
    return speciesName
        .replaceAll('，', ',')
        .replaceAll('、', ',')
        .split(',')
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty)
        .toList();
  }

  bool _hasValidSpecies(DetectionItem item) {
    return _splitGroupSpeciesNames(
      _primarySpecies(item),
    ).any((species) => !_isFallbackSpecies(species));
  }

  DetectionItem _preferredGroupItem(_ValidationMediaGroup group) {
    DetectionItem? firstImage;
    DetectionItem? firstValid;
    for (final item in group.items) {
      final isImage = _isImage(item);
      final hasValidSpecies = _hasValidSpecies(item);
      if (isImage && hasValidSpecies) return item;
      firstImage ??= isImage ? item : null;
      firstValid ??= hasValidSpecies ? item : null;
    }
    return firstValid ?? firstImage ?? group.items.first;
  }

  bool _isItemValidated(DetectionItem item) {
    return item.validated == true || item.detectionData['最低置信度'] == '人工校验';
  }

  String _groupHeaderTitle(_ValidationFileRow row) {
    final species = _groupSpeciesLabel(row.group);
    return '$species · ${row.group.items.length} 个文件';
  }

  String _groupHeaderSubtitle(_ValidationFileRow row) {
    final state = row.groupExpanded ? '已展开' : '已折叠';
    return '${_groupOrdinalLabel(row)} · $state';
  }

  String _fileListSubtitle(_ValidationFileRow row) {
    final label = _finalResultLabel(row.item);
    if (!widget.autoGroup) return label;
    if (_useCollapsedGroups && row.group.items.length > 1) {
      final state = row.groupExpanded ? '已展开' : '已折叠';
      final groupSize = row.group.items.length;
      return '${_groupOrdinalLabel(row)} · $state $groupSize 个文件 · $label';
    }
    return '${_groupOrdinalLabel(row)} · $label';
  }

  String _groupOrdinalLabel(_ValidationFileRow row) {
    final total = _bucketCacheGroupCount;
    final current = row.groupIndex + 1;
    if (total <= 0) return '第 $current 组';
    return '第 $current/$total 组';
  }

  List<DetectionItem> _sortValidationItems(List<DetectionItem> items) {
    return List<DetectionItem>.from(items)..sort(_compareValidationItems);
  }

  int _compareValidationItems(DetectionItem a, DetectionItem b) {
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
    final dateText = item.dateTaken?.trim() ?? '';
    final timeText = item.detectionData['拍摄时间']?.toString().trim() ?? '';
    final modifiedAt = item.modifiedAt?.trim() ?? '';
    final cacheKey = '$dateText\u001d$timeText\u001d$modifiedAt';
    final cached = _mediaTimestampCache[item.path];
    if (cached != null && cached.cacheKey == cacheKey) {
      return cached.timestamp;
    }

    DateTime? timestamp;
    if (dateText.isNotEmpty) {
      if (dateText.contains(':') || dateText.contains('T')) {
        final dateTime = DateTime.tryParse(dateText);
        if (dateTime != null) timestamp = dateTime;
      }
      if (timestamp == null && timeText.isNotEmpty) {
        final dateTime = DateTime.tryParse('$dateText $timeText');
        if (dateTime != null) timestamp = dateTime;
      }
    }
    if (timestamp == null && modifiedAt.isNotEmpty) {
      timestamp = DateTime.tryParse(modifiedAt);
    }
    _mediaTimestampCache[item.path] = _MediaTimestampCacheEntry(
      cacheKey: cacheKey,
      timestamp: timestamp,
    );
    return timestamp;
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

  String _finalResultLabel(DetectionItem item) {
    final finalSpecies = item.detectionData['物种名称']?.toString().trim();
    if (finalSpecies != null && finalSpecies.isNotEmpty) {
      return finalSpecies;
    }

    final species = item.species
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty && name != 'Unknown')
        .toList();
    if (species.isNotEmpty) return species.join('、');

    final boxSpecies = item.detectionBoxes
        .map((box) => box.species.trim())
        .where((name) => name.isNotEmpty && name != 'Unknown')
        .toSet()
        .toList();
    return boxSpecies.isEmpty ? item.fileType : boxSpecies.join('、');
  }

  String _primarySpecies(DetectionItem item) {
    final raw = item.detectionData['物种名称'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString().trim();
    }
    if (item.species.isNotEmpty) return item.species.join(',');
    final boxSpecies = item.detectionBoxes
        .map((box) => box.species.trim())
        .where((name) => name.isNotEmpty && name != 'Unknown')
        .toSet()
        .toList();
    if (boxSpecies.isNotEmpty) return boxSpecies.join(',');
    if (item.error != null) return '错误';
    return '未检测';
  }

  List<String> _quickMarkSpecies(DetectionItem _) {
    final configuredQuickSpecies = widget.quickMarkSpecies.isEmpty
        ? _quickSpecies
        : widget.quickMarkSpecies;
    final ordered = widget.autoSortQuickMarks
        ? _autoSortedQuickMarkSpecies(configuredQuickSpecies)
        : configuredQuickSpecies;
    return _uniqueSpecies(ordered).take(12).toList();
  }

  List<String> _autoSortedQuickMarkSpecies(List<String> fallbackSpecies) {
    final history = <String>[
      ...widget.quickMarkRecentHistory,
      ..._sessionQuickMarkHistory,
    ];
    final latestIndex = <String, int>{};
    final recentCounts = <String, int>{};
    for (var index = 0; index < history.length; index++) {
      final species = history[index].trim();
      if (species.isEmpty || species == 'Unknown' || species == '空') continue;
      latestIndex[species] = index;
      recentCounts[species] = (recentCounts[species] ?? 0) + 1;
    }

    final candidates = <String>{
      ...fallbackSpecies,
      ...widget.quickMarkUsageCounts.keys,
      ...recentCounts.keys,
    }.where((species) => species.trim().isNotEmpty).toList();

    candidates.sort((a, b) {
      final recentCompare = (recentCounts[b] ?? 0).compareTo(
        recentCounts[a] ?? 0,
      );
      if (recentCompare != 0) return recentCompare;
      final usageCompare = (widget.quickMarkUsageCounts[b] ?? 0).compareTo(
        widget.quickMarkUsageCounts[a] ?? 0,
      );
      if (usageCompare != 0) return usageCompare;
      final latestCompare = (latestIndex[b] ?? -1).compareTo(
        latestIndex[a] ?? -1,
      );
      if (latestCompare != 0) return latestCompare;
      final aFallbackIndex = fallbackSpecies.indexOf(a);
      final bFallbackIndex = fallbackSpecies.indexOf(b);
      if (aFallbackIndex >= 0 && bFallbackIndex >= 0) {
        return aFallbackIndex.compareTo(bFallbackIndex);
      }
      if (aFallbackIndex >= 0) return -1;
      if (bFallbackIndex >= 0) return 1;
      return a.compareTo(b);
    });

    return candidates;
  }

  Map<String, int> _speciesUsageCounts() {
    final counts = Map<String, int>.from(widget.quickMarkUsageCounts);
    for (final species in _sessionQuickMarkHistory) {
      final cleaned = species.trim();
      if (cleaned.isEmpty || cleaned == 'Unknown' || cleaned == '空') {
        continue;
      }
      counts[cleaned] = (counts[cleaned] ?? 0) + 1;
    }
    return counts;
  }

  List<String> _uniqueSpecies(Iterable<String> values) {
    final seen = <String>{};
    final result = <String>[];
    for (final value in values) {
      final species = value.trim();
      if (species.isEmpty ||
          species == 'Unknown' ||
          species == '空' ||
          seen.contains(species)) {
        continue;
      }
      seen.add(species);
      result.add(species);
    }
    return result;
  }

  _DetectionSummary _summaryFor(
    DetectionItem item,
    List<DetectionBox> visibleBoxes,
  ) {
    var species = _primarySpecies(item);
    var count = item.detectionData['物种数量']?.toString() ?? '';
    var confidence = item.detectionData['最低置信度']?.toString() ?? '';

    if (visibleBoxes.isNotEmpty && item.detectionData['最低置信度'] != '人工校验') {
      final counts = _isVideo(item)
          ? _trackCountsBySpecies(visibleBoxes)
          : _boxCountsBySpecies(visibleBoxes);
      final confidences = <double>[];
      for (final box in visibleBoxes) {
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
    if (species == '空' || species == '未知鸟' || species == '未检测') return '空';
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

  String _otherDialogCount(
    DetectionItem item,
    List<DetectionBox> visibleBoxes,
    _DetectionSummary summary,
  ) {
    if (visibleBoxes.isEmpty) {
      return item.detectionData['物种数量']?.toString() ?? summary.count;
    }
    final counts = _isVideo(item)
        ? _trackCountsBySpecies(visibleBoxes)
        : _boxCountsBySpecies(visibleBoxes);
    if (counts.isEmpty) return summary.count;
    return counts.values.join(',');
  }

  bool _isVideo(DetectionItem item) {
    const videoTypes = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'};
    return videoTypes.contains(item.fileType.toLowerCase());
  }

  Map<String, int> _boxCountsBySpecies(List<DetectionBox> boxes) {
    final counts = <String, int>{};
    for (final box in boxes) {
      counts[box.species] = (counts[box.species] ?? 0) + 1;
    }
    return counts;
  }

  Map<String, int> _trackCountsBySpecies(List<DetectionBox> boxes) {
    final trackVotes = <String, Map<String, int>>{};
    final fallbackCounts = <String, int>{};
    for (final box in boxes) {
      final trackId = box.trackId?.trim();
      if (trackId == null || trackId.isEmpty) {
        fallbackCounts[box.species] = (fallbackCounts[box.species] ?? 0) + 1;
        continue;
      }
      final votes = trackVotes.putIfAbsent(trackId, () => <String, int>{});
      votes[box.species] = (votes[box.species] ?? 0) + 1;
    }

    final counts = <String, int>{};
    for (final votes in trackVotes.values) {
      final species = votes.entries
          .reduce((a, b) => a.value >= b.value ? a : b)
          .key;
      counts[species] = (counts[species] ?? 0) + 1;
    }
    for (final entry in fallbackCounts.entries) {
      counts[entry.key] = (counts[entry.key] ?? 0) + entry.value;
    }
    return counts;
  }

  void _markQuickSpecies(String species) {
    setState(() {
      if (_pendingItemPath != _selectedPath) {
        _pendingQuantity = null;
      }
      _pendingItemPath = _selectedPath;
      _pendingSpeciesName = species;
    });
    unawaited(_submitPendingMarkIfReady());
  }

  void _markQuantity(String quantity) {
    setState(() {
      if (_pendingItemPath != _selectedPath) {
        _pendingSpeciesName = null;
      }
      _selectedQuantity = quantity;
      _showQuantitySelection = true;
      _pendingItemPath = _selectedPath;
      _pendingQuantity = quantity;
    });
    unawaited(_submitPendingMarkIfReady());
  }

  Future<void> _submitPendingMarkIfReady() async {
    if (_marking ||
        _pendingItemPath == null ||
        _pendingItemPath != _selectedPath ||
        _pendingSpeciesName == null ||
        _pendingQuantity == null) {
      return;
    }
    await _markSelected(
      'update',
      speciesName: _pendingSpeciesName,
      speciesCount: _pendingQuantity,
    );
  }

  void _resetPendingMark() {
    _pendingSpeciesName = null;
    _pendingQuantity = null;
    _pendingItemPath = null;
    _showQuantitySelection = false;
  }

  Future<void> _markOtherSpecies({List<DetectionItem>? itemsOverride}) async {
    final batchItems = itemsOverride;
    if ((batchItems == null || batchItems.isEmpty) && widget.items.isEmpty) {
      return;
    }
    final item = batchItems?.isNotEmpty == true
        ? batchItems!.first
        : widget.items.firstWhere(
            (candidate) => candidate.path == _selectedPath,
            orElse: () => widget.items.first,
          );

    final visibleBoxes = _filteredBoxes(item);
    final summary = _summaryFor(item, visibleBoxes);

    final initialSpecies =
        item.detectionData['物种名称']?.toString() ??
        (summary.species == '未知鸟' || summary.species == '未检测'
            ? ''
            : summary.species);
    final initialCount = _otherDialogCount(item, visibleBoxes, summary);
    final initialType =
        item.detectionData['物种类型']?.toString() ??
        (summary.type == '待补全' ? '' : summary.type);
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
          speciesUsageCounts: _speciesUsageCounts(),
        );
      },
    );

    if (draft == null || draft.speciesName.isEmpty || !mounted) return;
    final speciesCount = draft.speciesCount.isEmpty
        ? _selectedQuantity
        : draft.speciesCount;
    if (batchItems == null) {
      await _markSelected(
        'update',
        speciesName: draft.speciesName,
        speciesCount: speciesCount,
        speciesType: draft.speciesType,
        remark: draft.remark,
      );
    } else {
      await _markBatch(
        batchItems,
        'update',
        speciesName: draft.speciesName,
        speciesCount: speciesCount,
        speciesType: draft.speciesType,
        remark: draft.remark,
      );
    }
  }

  Future<void> _markBatch(
    List<DetectionItem> items,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    if (items.isEmpty || _marking) return;
    final visibleBefore = _visibleItems(_currentBuckets());
    final nextPath = _nextPathAfterBatch(visibleBefore, items);
    _deferRegroupForItems(items);

    _setMarking(true);
    try {
      final updatedItems = await widget.onMarkItems(
        items,
        action,
        speciesName: speciesName,
        speciesCount: speciesCount,
        speciesType: speciesType,
        remark: remark,
      );
      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;
      if (!mounted) return;
      final usedQuickSpecies =
          action == 'update' &&
              speciesName != null &&
              speciesName.trim().isNotEmpty
          ? speciesName
          : null;
      setState(() {
        if (action == 'unverified') {
          _discardMarkHistoryForPaths(items.map((item) => item.path).toSet());
        } else {
          _recordMarkHistory(updatedItems);
        }
        final nextSelection = nextPath ?? lastUpdated?.path;
        _selectedPaths.clear();
        if (nextSelection != null) {
          _selectedPath = nextSelection;
          _selectedPaths.add(nextSelection);
          _selectionAnchorPath = nextSelection;
          _setSelectedGroupFromPath(nextSelection);
        } else {
          _selectedGroupSignature = null;
        }
        _resetPendingMark();
        if (usedQuickSpecies != null) {
          _sessionQuickMarkHistory.addAll(_splitSpeciesNames(usedQuickSpecies));
          if (_sessionQuickMarkHistory.length > 200) {
            _sessionQuickMarkHistory.removeRange(
              0,
              _sessionQuickMarkHistory.length - 200,
            );
          }
        }
      });
      if (usedQuickSpecies != null) {
        unawaited(widget.onQuickMarkUsed(usedQuickSpecies));
      }
      _showSnackBar('已批量处理 ${items.length} 个文件');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('批量标记失败：$error');
    } finally {
      _setMarking(false);
    }
  }

  void _recordMarkHistory(Iterable<DetectionItem> items) {
    final itemsByPath = <String, DetectionItem>{};
    for (final item in items) {
      itemsByPath[item.path] = item;
    }
    if (itemsByPath.isEmpty) return;

    _discardMarkHistoryForPaths(itemsByPath.keys.toSet());
    _markHistory.add(_MarkHistoryEntry(itemsByPath.values));
    if (_markHistory.length > _undoHistoryLimit) {
      _markHistory.removeRange(
        0,
        _markHistory.length - _undoHistoryLimit,
      );
    }
  }

  void _discardMarkHistoryForPaths(Set<String> paths) {
    if (paths.isEmpty || _markHistory.isEmpty) return;
    for (final entry in _markHistory) {
      entry.items.removeWhere((item) => paths.contains(item.path));
    }
    _markHistory.removeWhere((entry) => entry.items.isEmpty);
  }

  String? _nextPathAfterBatch(
    List<DetectionItem> visibleItems,
    List<DetectionItem> markedItems,
  ) {
    if (visibleItems.isEmpty) return null;
    final markedPaths = markedItems.map((item) => item.path).toSet();
    var firstMarkedIndex = visibleItems.length;
    for (var index = 0; index < visibleItems.length; index++) {
      if (markedPaths.contains(visibleItems[index].path)) {
        firstMarkedIndex = index;
        break;
      }
    }
    if (firstMarkedIndex == visibleItems.length) {
      return visibleItems.first.path;
    }
    for (var index = firstMarkedIndex; index < visibleItems.length; index++) {
      if (!markedPaths.contains(visibleItems[index].path)) {
        return visibleItems[index].path;
      }
    }
    for (var index = firstMarkedIndex - 1; index >= 0; index--) {
      if (!markedPaths.contains(visibleItems[index].path)) {
        return visibleItems[index].path;
      }
    }
    return null;
  }

  List<DetectionItem> _recentUndoTargets() {
    if (_markHistory.isEmpty) return const <DetectionItem>[];
    final targetsByPath = <String, DetectionItem>{};
    for (final item in _markHistory.last.items) {
      targetsByPath[item.path] = item;
    }
    return targetsByPath.values.toList();
  }

  Future<void> _undoRecentMarks() async {
    if (_marking || _markHistory.isEmpty) return;
    final targets = _recentUndoTargets();
    if (targets.isEmpty) return;

    final visibleBefore = _visibleItems(_currentBuckets());
    final nextPath = _nextPathAfterBatch(visibleBefore, targets);
    _deferRegroupForItems(targets);

    _setMarking(true);
    try {
      final updatedItems = await widget.onMarkItems(targets, 'unverified');
      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;
      if (!mounted) return;
      final targetPaths = targets.map((item) => item.path).toSet();
      setState(() {
        _discardMarkHistoryForPaths(targetPaths);
        final nextSelection = nextPath ?? lastUpdated?.path;
        _selectedPaths.clear();
        if (nextSelection != null) {
          _selectedPath = nextSelection;
          _selectedPaths.add(nextSelection);
          _selectionAnchorPath = nextSelection;
          _setSelectedGroupFromPath(nextSelection);
        } else {
          _selectedGroupSignature = null;
        }
        _resetPendingMark();
      });
      final count = updatedItems.isEmpty ? targets.length : updatedItems.length;
      _showSnackBar(count == 1 ? '已撤回校验标记' : '已撤回 $count 个校验标记');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('撤回校验标记失败：$error');
    } finally {
      _setMarking(false);
    }
  }

  Future<void> _redetectBatch(List<DetectionItem> items) async {
    if (items.isEmpty || _marking) return;
    _setMarking(true);
    try {
      await widget.onRedetectItems(items, confidence: _confidence);
      if (!mounted) return;
      _showSnackBar('已提交 ${items.length} 个文件重新检测');
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('重新检测失败：$error');
    } finally {
      _setMarking(false);
    }
  }

  Future<void> _markSelected(
    String action, {
    DetectionItem? itemOverride,
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
    bool nextPhotoOnly = false,
  }) async {
    if (widget.items.isEmpty) return;
    final item =
        itemOverride ??
        widget.items.firstWhere(
          (candidate) => candidate.path == _selectedPath,
          orElse: () => widget.items.first,
        );
    if (itemOverride == null) {
      final actionTargets = _selectedActionTargets(item);
      if (actionTargets.length > 1 ||
          (actionTargets.isNotEmpty && actionTargets.first.path != item.path)) {
        await _markBatch(
          actionTargets,
          action,
          speciesName: speciesName,
          speciesCount: speciesCount,
          speciesType: speciesType,
          remark: remark,
        );
        return;
      }
    }
    final visibleItems = _visibleItems(_currentBuckets());
    final currentIndex = visibleItems.indexWhere(
      (candidate) => candidate.path == item.path,
    );
    final nextPath = _nextPathAfterItem(
      visibleItems,
      currentIndex,
      photosOnly: nextPhotoOnly,
    );
    _deferRegroupForItems(<DetectionItem>[item]);

    _setMarking(true);
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
      final usedQuickSpecies =
          action == 'update' &&
              speciesName != null &&
              speciesName.trim().isNotEmpty
          ? speciesName
          : null;
      final targetItem = nextPath == null
          ? updated
          : visibleItems.firstWhere(
              (candidate) => candidate.path == nextPath,
              orElse: () => updated,
            );
      setState(() {
        if (action == 'unverified') {
          _discardMarkHistoryForPaths({item.path});
        } else {
          _recordMarkHistory(<DetectionItem>[updated]);
        }
        final targetPath = targetItem.path;
        _selectedPath = targetPath;
        _selectedPaths
          ..clear()
          ..add(targetPath);
        _selectionAnchorPath = targetPath;
        _setSelectedGroupFromPath(targetPath);
        _resetPendingMark();
        if (usedQuickSpecies != null) {
          _sessionQuickMarkHistory.addAll(_splitSpeciesNames(usedQuickSpecies));
          if (_sessionQuickMarkHistory.length > 200) {
            _sessionQuickMarkHistory.removeRange(
              0,
              _sessionQuickMarkHistory.length - 200,
            );
          }
        }
      });
      if (usedQuickSpecies != null) {
        unawaited(widget.onQuickMarkUsed(usedQuickSpecies));
      }
      unawaited(widget.onLoadMetadata(targetItem));
      final message = action == 'unverified'
          ? '已撤回校验标记'
          : '已标记 ${updated.filename}';
      _showSnackBar(message);
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('标记失败：$error');
    } finally {
      _setMarking(false);
    }
  }

  void _setMarking(bool value) {
    if (!mounted || _marking == value) return;
    setState(() => _marking = value);
  }

  String? _nextPathAfterItem(
    List<DetectionItem> visibleItems,
    int currentIndex, {
    required bool photosOnly,
  }) {
    if (currentIndex < 0) return null;
    for (var index = currentIndex + 1; index < visibleItems.length; index++) {
      final candidate = visibleItems[index];
      if (!photosOnly || _isImage(candidate)) return candidate.path;
    }
    return null;
  }

  List<String> _splitSpeciesNames(String speciesName) {
    return speciesName
        .replaceAll('，', ',')
        .split(',')
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty && name != '空' && name != 'Unknown')
        .toList();
  }

  void _showSnackBar(
    String message, {
    String actionLabel = '关闭',
    VoidCallback? onAction,
  }) {
    if (!mounted || message.isEmpty) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();

    final controller = messenger.showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 4), // 强制补齐 4 秒
        action: SnackBarAction(
          label: actionLabel,
          onPressed: onAction ?? messenger.hideCurrentSnackBar,
        ),
      ),
    );
    // 强制绕过无障碍模式的驻留时间
    Future.delayed(const Duration(seconds: 4), controller.close);
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
      _showSnackBar(
        '已导出 ${result.exportedCount} 条记录到 ${result.outputPath}',
        actionLabel: '打开',
        onAction: () => widget.onOpenExternal(result.outputPath),
      );
    } catch (error) {
      if (!mounted) return;
      _showSnackBar('导出失败：$error');
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  List<_ValidationFileRow> _visibleRows(List<_SpeciesBucket> buckets) {
    if (_selectedBucketKey == null) return const [];
    final bucket = _selectedBucketFrom(buckets);
    final rows = <_ValidationFileRow>[];

    for (var index = 0; index < bucket.groups.length; index++) {
      final group = bucket.groups[index];
      if (group.items.isEmpty) continue;

      final signature = _groupSignature(group);
      final globalGroupIndex = _globalGroupIndexForSignature(
        signature,
        fallback: index,
      );
      final expanded = _expandedGroupSignatures.contains(signature);
      final isGroup = _useCollapsedGroups && group.items.length > 1;

      if (isGroup) {
        final representative = _preferredGroupItem(group);
        rows.add(
          _ValidationFileRow(
            item: representative,
            isGroupHeader: true,
            groupExpanded: expanded,
            group: group,
            groupIndex: globalGroupIndex,
          ),
        );

        if (expanded) {
          for (final item in group.items) {
            rows.add(
              _ValidationFileRow(
                item: item,
                isGroupHeader: false,
                groupExpanded: true,
                group: group,
                groupIndex: globalGroupIndex,
              ),
            );
          }
        }
      } else {
        for (final item in group.items) {
          rows.add(
            _ValidationFileRow(
              item: item,
              isGroupHeader: false,
              groupExpanded: false,
              group: group,
              groupIndex: globalGroupIndex,
            ),
          );
        }
      }
    }
    return rows;
  }

  List<DetectionItem> _itemsFromRows(List<_ValidationFileRow> rows) {
    final items = <DetectionItem>[];
    for (final row in rows) {
      if (row.isGroupHeader) {
        if (!row.groupExpanded) {
          items.add(row.item);
        }
      } else {
        items.add(row.item);
      }
    }
    return items;
  }

  int _selectedPhotoRowIndex(List<_ValidationFileRow> rows) {
    final selectedPath = _selectedPath;
    if (selectedPath != null) {
      final itemIndex = rows.indexWhere(
        (row) => !row.isGroupHeader && row.item.path == selectedPath,
      );
      if (itemIndex >= 0) return itemIndex;

      final selectedGroupSignature = _groupSignatureForPath(selectedPath);
      final headerIndex = rows.indexWhere(
        (row) =>
            row.isGroupHeader && row.group.signature == selectedGroupSignature,
      );
      if (headerIndex >= 0) return headerIndex;
    }
    return rows.indexWhere(_isRowSelected);
  }

  bool _hasVisibleSelection(List<_ValidationFileRow> rows) {
    return rows.any(_isRowSelected);
  }

  void _applyRowSelection(_ValidationFileRow row) {
    _selectedPath = row.item.path;
    _selectedPaths
      ..clear()
      ..add(row.item.path);
    _selectionAnchorPath = row.item.path;
    _setSelectedGroupFromRow(row);
  }

  DetectionItem _selectedItemFromRows(List<_ValidationFileRow> rows) {
    if (rows.isEmpty) {
      if (widget.items.isEmpty) {
        throw StateError('No validation item is available.');
      }
      return widget.items.first;
    }
    if (_selectedPath != null) {
      final selectedItem = _itemByPath(_selectedPath!);
      if (selectedItem != null) return selectedItem;
    }
    for (final row in rows) {
      if (!row.isGroupHeader && _isRowSelected(row)) return row.item;
    }
    for (final row in rows) {
      if (_isRowSelected(row)) return row.item;
    }
    return rows.first.item;
  }

  DetectionItem? _itemByPath(String path) {
    return _bucketCacheItemByPath[path];
  }

  bool _isRowSelected(_ValidationFileRow row) {
    if (row.isGroupHeader) {
      final groupHasActivePath =
          _selectedPath != null &&
          _groupSignatureForPath(_selectedPath) == row.group.signature;
      final groupHasSelectedPath = _selectedPaths.any(
        (path) => _groupSignatureForPath(path) == row.group.signature,
      );
      if (!row.groupExpanded) {
        return groupHasActivePath || groupHasSelectedPath;
      }
      return _selectedGroupSignature == row.group.signature ||
          groupHasActivePath ||
          groupHasSelectedPath;
    }
    return row.item.path == _selectedPath ||
        _selectedPaths.contains(row.item.path);
  }
}

class _ValidationMenuSelect extends StatelessWidget {
  const _ValidationMenuSelect({
    required this.label,
    required this.value,
    required this.options,
    required this.labelBuilder,
    required this.onChanged,
    required this.width,
  });

  final String label;
  final String value;
  final List<String> options;
  final String Function(String value) labelBuilder;
  final ValueChanged<String> onChanged;
  final double width;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      child: DropdownMenu<String>(
        initialSelection: value,
        expandedInsets: EdgeInsets.zero,
        label: Text(label),
        menuStyle: appDropdownMenuStyle(context, minWidth: width),
        dropdownMenuEntries: [
          for (final option in options)
            DropdownMenuEntry<String>(
              value: option,
              label: labelBuilder(option),
            ),
        ],
        onSelected: (value) {
          if (value != null) onChanged(value);
        },
      ),
    );
  }
}

class _ValidationPanel extends StatelessWidget {
  const _ValidationPanel({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Card.outlined(margin: EdgeInsets.zero, child: child);
  }
}

class _BatchMenuItem extends StatelessWidget {
  const _BatchMenuItem({required this.label, required this.count});

  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label),
        const SizedBox(width: 10),
        Text(
          '$count',
          style: TextStyle(color: Theme.of(context).colorScheme.primary),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  const _ActionButton({
    required this.label,
    required this.onPressed,
    this.height = 36,
    this.selected = false,
  });

  final String label;
  final VoidCallback? onPressed;
  final double height;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return SizedBox(
      width: double.infinity,
      height: height,
      child: FilledButton.tonal(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          backgroundColor: selected ? colorScheme.primaryContainer : null,
          foregroundColor: selected ? colorScheme.onPrimaryContainer : null,
          side: selected
              ? BorderSide(color: colorScheme.primary, width: 1.5)
              : null,
          textStyle: TextStyle(
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
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
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return SizedBox(
      width: double.infinity,
      height: 30,
      child: FilledButton.tonal(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          padding: EdgeInsets.zero,
          backgroundColor: selected ? colorScheme.primaryContainer : null,
          foregroundColor: selected ? colorScheme.onPrimaryContainer : null,
          side: selected
              ? BorderSide(color: colorScheme.primary, width: 1.5)
              : null,
          textStyle: TextStyle(
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
          ),
        ),
        child: Text(label),
      ),
    );
  }
}

class _EmptyValidationState extends StatelessWidget {
  const _EmptyValidationState({
    required this.loading,
    required this.onRefresh,
  });

  final bool loading;
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
            onPressed: loading ? null : onRefresh,
            icon: loading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh_rounded),
            label: const Text('重新获取'),
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
    required this.groups,
  });

  factory _SpeciesBucket.empty() {
    return _SpeciesBucket(
      key: '',
      isValidated: false,
      species: '',
      items: const <DetectionItem>[],
      groups: const <_ValidationMediaGroup>[],
    );
  }

  final String key;
  final bool isValidated;
  final String species;
  final List<DetectionItem> items;
  final List<_ValidationMediaGroup> groups;
}

class _ValidationMediaGroup {
  _ValidationMediaGroup(this.items)
    : signature = items.map((item) => item.path).join('\u001e');

  final List<DetectionItem> items;
  final String signature;
}

// 新增 _ValidationFileRow 类
class _ValidationFileRow {
  const _ValidationFileRow({
    required this.item,
    required this.isGroupHeader,
    required this.groupExpanded,
    required this.group,
    required this.groupIndex,
  });

  final DetectionItem item;
  final bool isGroupHeader;
  final bool groupExpanded;
  final _ValidationMediaGroup group;
  final int groupIndex;
}

class _ValidationGroupIndexEntry {
  const _ValidationGroupIndexEntry({
    required this.index,
    required this.group,
    required this.signature,
  });

  final int index;
  final _ValidationMediaGroup group;
  final String signature;
}

class _ValidationGroupContext {
  const _ValidationGroupContext({
    required this.index,
    required this.group,
    required this.signature,
    required this.expanded,
  });

  final int index;
  final _ValidationMediaGroup group;
  final String signature;
  final bool expanded;
}

class _SpeciesVote {
  int count = 0;
  int validatedCount = 0;
}

class _MediaTimestampCacheEntry {
  const _MediaTimestampCacheEntry({
    required this.cacheKey,
    required this.timestamp,
  });

  final String cacheKey;
  final DateTime? timestamp;
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
    required this.speciesUsageCounts,
  });

  final String initialSpecies;
  final String initialCount;
  final String initialType;
  final String initialRemark;
  final Map<String, String> speciesTypes;
  final Map<String, int> speciesUsageCounts;

  @override
  State<_OtherSpeciesDialog> createState() => _OtherSpeciesDialogState();
}

class _OtherSpeciesDialogState extends State<_OtherSpeciesDialog> {
  late final TextEditingController speciesController;
  late final TextEditingController countController;
  late final TextEditingController typeController;
  late final TextEditingController remarkController;

  late final FocusNode _speciesFocusNode;

  List<String> _suggestions = [];

  @override
  void initState() {
    super.initState();
    speciesController = TextEditingController(text: widget.initialSpecies);
    countController = TextEditingController(text: widget.initialCount);
    typeController = TextEditingController(text: widget.initialType);
    remarkController = TextEditingController(text: widget.initialRemark);

    _speciesFocusNode = FocusNode();

    speciesController.addListener(_onSpeciesChanged);
  }

  @override
  void dispose() {
    speciesController.removeListener(_onSpeciesChanged);
    speciesController.dispose();
    countController.dispose();
    typeController.dispose();
    remarkController.dispose();
    _speciesFocusNode.dispose();
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
    final originalOrder = <String, int>{};
    var index = 0;
    for (final species in widget.speciesTypes.keys) {
      originalOrder[species] = index++;
    }
    matches.sort((a, b) {
      final usageCompare = (widget.speciesUsageCounts[b] ?? 0).compareTo(
        widget.speciesUsageCounts[a] ?? 0,
      );
      if (usageCompare != 0) return usageCompare;
      return (originalOrder[a] ?? 0).compareTo(originalOrder[b] ?? 0);
    });
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
          activeSpecies.add(p);
        }
      }
    }

    final validEntities = activeSpecies
        .where((e) => e != '空' && e != '未知鸟')
        .toList();

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

    if (validEntities.isNotEmpty) {
      final currentCounts = countController.text
          .split(',')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();
      final newCounts = <String>[];

      for (int i = 0; i < validEntities.length; i++) {
        if (i < currentCounts.length) {
          final parsed = int.tryParse(currentCounts[i]);
          if (parsed != null && parsed > 0) {
            newCounts.add(currentCounts[i]);
          } else {
            newCounts.add('1');
          }
        } else {
          newCounts.add('1');
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

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;

    if (event.logicalKey == LogicalKeyboardKey.enter ||
        event.logicalKey == LogicalKeyboardKey.numpadEnter) {
      if (HardwareKeyboard.instance.logicalKeysPressed.contains(
            LogicalKeyboardKey.shiftLeft,
          ) ||
          HardwareKeyboard.instance.logicalKeysPressed.contains(
            LogicalKeyboardKey.shiftRight,
          )) {
        return KeyEventResult.ignored;
      }

      if (_speciesFocusNode.hasFocus) {
        final text = speciesController.text;
        final parts = text.split(RegExp(r'[,，]'));
        final currentWord = parts.last.trim();

        if (currentWord.isNotEmpty && _suggestions.isNotEmpty) {
          _onSuggestionTapped(_suggestions.first);
          _speciesFocusNode.requestFocus();
        } else {
          _submit();
        }
        return KeyEventResult.handled;
      } else {
        _submit();
        return KeyEventResult.handled;
      }
    }
    return KeyEventResult.ignored;
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      onKeyEvent: _handleKeyEvent,
      child: AlertDialog(
        title: const Text('其他标注'),
        content: SizedBox(
          width: 380,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextField(
                controller: speciesController,
                focusNode: _speciesFocusNode,
                autofocus: true,
                decoration: const InputDecoration(
                  labelText: '物种名称 (拼音首字母，逗号分隔)',
                  hintText: '如: cmy(赤麻鸭), bl(白鹭)',
                  border: OutlineInputBorder(),
                ),
              ),

              AnimatedSize(
                duration: const Duration(milliseconds: 200),
                child: _suggestions.isEmpty
                    ? const SizedBox.shrink()
                    : Padding(
                        padding: const EdgeInsets.only(top: 10.0, bottom: 4.0),
                        child: Wrap(
                          spacing: 8.0,
                          runSpacing: 8.0,
                          children: _suggestions.take(8).map((species) {
                            return ActionChip(
                              label: Text(
                                species,
                                style: const TextStyle(fontSize: 13),
                              ),
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
          FilledButton(onPressed: _submit, child: const Text('确定')),
        ],
      ),
    );
  }
}
