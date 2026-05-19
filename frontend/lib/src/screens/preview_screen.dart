import 'dart:async';

import 'package:flutter/material.dart';

import '../models/job.dart';
import '../widgets/app_menu_style.dart';
import '../widgets/detection_media_viewer.dart';
import '../widgets/selectable_list_card.dart';

const previewAllSpeciesLabel = '全局设置';

class PreviewScreen extends StatelessWidget {
  const PreviewScreen({
    required this.inputPath,
    required this.items,
    required this.selectedIndex,
    required this.selectedItem,
    required this.speciesTypes,
    required this.showDetections,
    required this.onShowDetectionsChanged,
    required this.selectedSpeciesFilter,
    required this.onSpeciesFilterChanged,
    required this.confidenceThreshold,
    required this.onConfidenceThresholdChanged,
    required this.detecting,
    required this.loading,
    required this.onDetectCurrentImage,
    required this.onSelected,
    required this.onLoadMetadata,
    required this.onRefresh,
    required this.onOpenExternal,
    super.key,
  });

  final String inputPath;
  final List<DetectionItem> items;
  final int selectedIndex;
  final DetectionItem? selectedItem;
  final Map<String, String> speciesTypes;
  final bool showDetections;
  final ValueChanged<bool> onShowDetectionsChanged;
  final String selectedSpeciesFilter;
  final ValueChanged<String> onSpeciesFilterChanged;
  final double confidenceThreshold;
  final ValueChanged<double> onConfidenceThresholdChanged;
  final bool detecting;
  final bool loading;
  final ValueChanged<DetectionItem> onDetectCurrentImage;
  final void Function(int index, DetectionItem item) onSelected;
  final Future<void> Function(DetectionItem item) onLoadMetadata;
  final Future<void> Function() onRefresh;
  final void Function(String path) onOpenExternal;

  @override
  Widget build(BuildContext context) {
    if (items.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: _EmptyPreviewState(
          inputPath: inputPath,
          loading: loading,
          onRefresh: onRefresh,
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (items.isEmpty)
            Text(inputPath.isEmpty ? '请先在开始界面设置输入文件夹。' : '该输入文件夹中暂无可预览图像。')
          else
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return _buildWorkspace(context, constraints.maxWidth);
                },
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildWorkspace(BuildContext context, double availableWidth) {
    final safeIndex = selectedIndex.clamp(0, items.length - 1).toInt();
    final item = selectedItem ?? items[safeIndex];
    final useHorizontalLayout = availableWidth >= 900;

    if (useHorizontalLayout) {
      final listWidth = (availableWidth * 0.20).clamp(200.0, 300.0).toDouble();
      return Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(width: listWidth, child: _buildFileList(context)),
          const VerticalDivider(width: 24),
          Expanded(child: _buildPreviewDetail(context, item)),
        ],
      );
    }

    final listHeight = availableWidth < 600 ? 220.0 : 280.0;
    return Column(
      children: [
        SizedBox(height: listHeight, child: _buildFileList(context)),
        const SizedBox(height: 16),
        Expanded(child: _buildPreviewDetail(context, item)),
      ],
    );
  }

  Widget _buildFileList(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final selectedPath =
        selectedItem?.path ??
        (items.isEmpty
            ? null
            : items[selectedIndex.clamp(0, items.length - 1).toInt()].path);
    return SelectableListCard<DetectionItem>(
      items: items,
      selectedIndex: selectedIndex,
      leadingBuilder: (item) => Icon(_previewFileIcon(item)),
      titleBuilder: (item) => item.filename,
      subtitleBuilder: _finalResultLabel,
      trailingBuilder: (item) {
        final selected = item.path == selectedPath;
        if (!selected && item.error == null) return null;
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (item.error != null)
              Icon(
                Icons.error_outline_rounded,
                color: colorScheme.error,
                size: 20,
              ),
            if (selected) ...[
              if (item.error != null) const SizedBox(width: 6),
              Icon(
                Icons.check_circle_rounded,
                color: colorScheme.primary,
                size: 20,
              ),
            ],
          ],
        );
      },
      onSelected: (index, item) {
        onSelected(index, item);
        unawaited(onLoadMetadata(item));
      },
    );
  }

  Widget _buildPreviewDetail(BuildContext context, DetectionItem item) {
    final filterApplies =
        item.detectionBoxes.any(
          (box) => box.species == selectedSpeciesFilter,
        ) ||
        item.species.contains(selectedSpeciesFilter);
    final effectiveSpecies = filterApplies
        ? selectedSpeciesFilter
        : previewAllSpeciesLabel;
    final visibleBoxes = _filteredPreviewBoxes(item, effectiveSpecies);
    final speciesOptions = <String>{
      previewAllSpeciesLabel,
      ...item.species,
      ...item.detectionBoxes.map((box) => box.species),
    }.where((species) => species.isNotEmpty).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: Card.filled(
            margin: EdgeInsets.zero,
            clipBehavior: Clip.antiAlias,
            child: DetectionMediaViewer(
              item: item,
              visibleBoxes: visibleBoxes,
              showDetections: showDetections,
              onOpenExternal: () => onOpenExternal(item.path),
            ),
          ),
        ),
        const SizedBox(height: 10), // 与校验界面保持 10 的间距
        _ImageInfoCard(
          item: item,
          visibleBoxes: visibleBoxes,
          speciesTypes: speciesTypes,
        ),
        const SizedBox(height: 10), // 与校验界面保持 10 的间距
        _PreviewDetectionControls(
          item: item,
          selectedSpecies: effectiveSpecies,
          speciesOptions: speciesOptions,
          showDetections: showDetections,
          onShowDetectionsChanged: onShowDetectionsChanged,
          confidenceThreshold: confidenceThreshold,
          onConfidenceThresholdChanged: onConfidenceThresholdChanged,
          detecting: detecting,
          onSpeciesFilterChanged: onSpeciesFilterChanged,
          onDetectCurrentImage: onDetectCurrentImage,
        ),
      ],
    );
  }

  List<DetectionBox> _filteredPreviewBoxes(
    DetectionItem item,
    String selectedSpecies,
  ) {
    return item.detectionBoxes.where((box) {
      final matchesSpecies =
          selectedSpecies == previewAllSpeciesLabel ||
          box.species == selectedSpecies;
      final confidence = box.confidence;
      final matchesConfidence =
          confidence == null || confidence >= confidenceThreshold;
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

  IconData _previewFileIcon(DetectionItem item) {
    return _isPreviewImage(item) ? Icons.image_rounded : Icons.movie_rounded;
  }

  bool _isPreviewImage(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }
}

class _EmptyPreviewState extends StatelessWidget {
  const _EmptyPreviewState({
    required this.inputPath,
    required this.loading,
    required this.onRefresh,
  });

  final String inputPath;
  final bool loading;
  final Future<void> Function() onRefresh;

  @override
  Widget build(BuildContext context) {
    final hasInputPath = inputPath.trim().isNotEmpty;
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.image_search_rounded, size: 56),
          const SizedBox(height: 12),
          Text(hasInputPath ? '暂无预览图像。' : '请先在开始界面设置输入文件夹。'),
          if (hasInputPath) ...[
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
        ],
      ),
    );
  }
}

class _ImageInfoCard extends StatelessWidget {
  const _ImageInfoCard({
    required this.item,
    required this.visibleBoxes,
    required this.speciesTypes,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final Map<String, String> speciesTypes;

  @override
  Widget build(BuildContext context) {
    final summary = _summaryFor(item, visibleBoxes);
    final colorScheme = Theme.of(context).colorScheme;

    return Card.outlined(
      margin: EdgeInsets.zero,
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

  String _primarySpecies(DetectionItem item) {
    final raw = item.detectionData['物种名称'];
    if (raw != null && raw.toString().trim().isNotEmpty) {
      return raw.toString().trim();
    }
    if (item.species.isNotEmpty) return item.species.join(',');
    if (item.error != null) return '错误';
    return '未检测';
  }

  String _typeLabel(String species, DetectionItem item) {
    final manualType = item.detectionData['物种类型']?.toString().trim();
    if (manualType != null && manualType.isNotEmpty) return manualType;
    if (species == '空' || species == '未知鸟' || species == '未检测') return '空';
    final matchedTypes = species
        .split(',')
        .map((name) => name.trim())
        .where((name) => name.isNotEmpty)
        .map((name) => speciesTypes[name] ?? '')
        .where((speciesType) => speciesType.isNotEmpty)
        .toSet()
        .toList();
    if (matchedTypes.isNotEmpty) return matchedTypes.join(',');
    if (species.contains('人') || species.contains('牧民')) return '人员';
    return '待补全';
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
    for (var index = 0; index < boxes.length; index++) {
      final box = boxes[index];
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

class _PreviewDetectionControls extends StatelessWidget {
  const _PreviewDetectionControls({
    required this.item,
    required this.selectedSpecies,
    required this.speciesOptions,
    required this.showDetections,
    required this.onShowDetectionsChanged,
    required this.confidenceThreshold,
    required this.onConfidenceThresholdChanged,
    required this.detecting,
    required this.onSpeciesFilterChanged,
    required this.onDetectCurrentImage,
  });

  final DetectionItem item;
  final String selectedSpecies;
  final List<String> speciesOptions;
  final bool showDetections;
  final ValueChanged<bool> onShowDetectionsChanged;
  final double confidenceThreshold;
  final ValueChanged<double> onConfidenceThresholdChanged;
  final bool detecting;
  final ValueChanged<String> onSpeciesFilterChanged;
  final ValueChanged<DetectionItem> onDetectCurrentImage;

  @override
  Widget build(BuildContext context) {
    return Card.outlined(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            const Text('显示结果', style: TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(width: 4),
            Switch(value: showDetections, onChanged: onShowDetectionsChanged),
            const SizedBox(width: 16),
            const Text('置信度', style: TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(width: 8),
            SizedBox(
              width: 150,
              child: DropdownMenu<String>(
                initialSelection: selectedSpecies,
                expandedInsets: EdgeInsets.zero,
                menuStyle: appDropdownMenuStyle(context, minWidth: 150),
                label: const Text('物种'),
                dropdownMenuEntries: [
                  for (final species in speciesOptions)
                    DropdownMenuEntry<String>(value: species, label: species),
                ],
                onSelected: (value) {
                  if (value != null) onSpeciesFilterChanged(value);
                },
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Slider(
                value: confidenceThreshold,
                min: 0.05, // 修改了最小值、最大值和分段数量以和校验界面严格统一
                max: 0.95,
                divisions: 90,
                label: confidenceThreshold.toStringAsFixed(2),
                onChanged: onConfidenceThresholdChanged,
              ),
            ),
            SizedBox(
              width: 36,
              child: Text(confidenceThreshold.toStringAsFixed(2)),
            ),
            const SizedBox(width: 16),
            FilledButton.icon(
              onPressed: detecting || !_isPreviewImage(item)
                  ? null
                  : () => onDetectCurrentImage(item),
              icon: detecting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.center_focus_strong_rounded, size: 18),
              label: Text(detecting ? '检测中...' : '检测当前图像'),
            ),
          ],
        ),
      ),
    );
  }

  bool _isPreviewImage(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }
}
