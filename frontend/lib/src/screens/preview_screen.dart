import 'dart:async';

import 'package:flutter/material.dart';

import '../models/job.dart';
import '../widgets/detection_media_viewer.dart';
import '../widgets/selectable_list_card.dart';

const previewAllSpeciesLabel = '全局设置';

class PreviewScreen extends StatelessWidget {
  const PreviewScreen({
    required this.inputPath,
    required this.items,
    required this.selectedIndex,
    required this.selectedItem,
    required this.showDetections,
    required this.onShowDetectionsChanged,
    required this.selectedSpeciesFilter,
    required this.onSpeciesFilterChanged,
    required this.confidenceThreshold,
    required this.onConfidenceThresholdChanged,
    required this.detecting,
    required this.onDetectCurrentImage,
    required this.onSelected,
    required this.onLoadMetadata,
    required this.onOpenExternal,
    super.key,
  });

  final String inputPath;
  final List<DetectionItem> items;
  final int selectedIndex;
  final DetectionItem? selectedItem;
  final bool showDetections;
  final ValueChanged<bool> onShowDetectionsChanged;
  final String selectedSpeciesFilter;
  final ValueChanged<String> onSpeciesFilterChanged;
  final double confidenceThreshold;
  final ValueChanged<double> onConfidenceThresholdChanged;
  final bool detecting;
  final ValueChanged<DetectionItem> onDetectCurrentImage;
  final void Function(int index, DetectionItem item) onSelected;
  final Future<void> Function(DetectionItem item) onLoadMetadata;
  final void Function(String path) onOpenExternal;

  @override
  Widget build(BuildContext context) {
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
          SizedBox(width: listWidth, child: _buildFileList()),
          const VerticalDivider(width: 24),
          Expanded(child: _buildPreviewDetail(context, item)),
        ],
      );
    }

    final listHeight = availableWidth < 600 ? 220.0 : 280.0;
    return Column(
      children: [
        SizedBox(height: listHeight, child: _buildFileList()),
        const SizedBox(height: 16),
        Expanded(child: _buildPreviewDetail(context, item)),
      ],
    );
  }

  Widget _buildFileList() {
    return SelectableListCard<DetectionItem>(
      items: items,
      selectedIndex: selectedIndex,
      leadingBuilder: (item) => Icon(_previewFileIcon(item)),
      titleBuilder: (item) => item.filename,
      subtitleBuilder: (item) =>
          item.species.isEmpty ? item.fileType : item.species.join('、'),
      trailingBuilder: (item) =>
          item.error == null ? null : const Icon(Icons.error_outline_rounded),
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
        _ImageInfoCard(item: item, visibleBoxes: visibleBoxes),
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

  IconData _previewFileIcon(DetectionItem item) {
    return _isPreviewImage(item) ? Icons.image_rounded : Icons.movie_rounded;
  }

  bool _isPreviewImage(DetectionItem item) {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }
}

class _ImageInfoCard extends StatelessWidget {
  const _ImageInfoCard({required this.item, required this.visibleBoxes});

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;

  @override
  Widget build(BuildContext context) {
    String detectionResultStr = '暂无';
    String boxCountStr = '0';
    String minConfStr = '未知';

    final rawSpecies = item.detectionData['物种名称'];
    final rawConf = item.detectionData['最低置信度'];

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
        detectionResultStr = counts.entries
            .map((entry) => '${entry.key}×${entry.value}')
            .join('，');
        boxCountStr = visibleBoxes.length.toString();
        if (confidences.isNotEmpty) {
          minConfStr = confidences
              .reduce((a, b) => a < b ? a : b)
              .toStringAsFixed(2);
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
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12), // 修改为与校验界面的 Summary 相同的 12
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _InfoLine(
              children: [
                _InlineInfo(label: '文件名', value: item.filename, maxWidth: 220),
                _InlineInfo(label: '类型', value: item.fileType, maxWidth: 80),
                _InlineInfo(
                  label: '尺寸',
                  value: _formatImageSize(item),
                  maxWidth: 120,
                ),
                _InlineInfo(
                  label: '大小',
                  value: _formatBytes(item.sizeBytes),
                  maxWidth: 100,
                ),
                _InlineInfo(
                  label: '拍摄时间',
                  value: item.dateTaken ?? '未知',
                  maxWidth: 160,
                ),
              ],
            ),
            _InfoLine(
              children: [
                _InlineInfo(
                  label: '检测结果',
                  value: detectionResultStr,
                  maxWidth: 220,
                ),
                _InlineInfo(label: '检测框', value: boxCountStr, maxWidth: 80),
                _InlineInfo(label: '最低置信度', value: minConfStr, maxWidth: 120),
                _InlineInfo(
                  label: '检测时间',
                  value: _detectionTimeLabel(item),
                  maxWidth: 160,
                ),
              ],
            ),
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
}

class _InfoLine extends StatelessWidget {
  const _InfoLine({required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Wrap(spacing: 16, runSpacing: 4, children: children),
    );
  }
}

class _InlineInfo extends StatelessWidget {
  const _InlineInfo({
    required this.label,
    required this.value,
    this.maxWidth = 200,
  });

  final String label;
  final String value;
  final double maxWidth;

  @override
  Widget build(BuildContext context) {
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
            Switch(
              value: showDetections,
              onChanged: onShowDetectionsChanged,
            ),
            const SizedBox(width: 16),
            const Text('置信度', style: TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(width: 4),
            SizedBox(
              width: 150,
              child: DropdownButtonFormField<String>(
                key: ValueKey(
                  'preview-species-$selectedSpecies-${speciesOptions.join('|')}',
                ),
                initialValue: selectedSpecies,
                isExpanded: true,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 0,
                  ),
                ),
                items: speciesOptions.map((species) {
                  return DropdownMenuItem(
                    value: species,
                    child: Text(
                      species,
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                    ),
                  );
                }).toList(),
                onChanged: (value) {
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
