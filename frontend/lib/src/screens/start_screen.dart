import 'package:flutter/material.dart';

import '../models/job.dart';
import '../models/settings.dart';
import '../widgets/app_menu_style.dart';
import '../widgets/section_card.dart';

const _defaultModelDirectory = 'res/model';

class StartScreen extends StatelessWidget {
  const StartScreen({
    required this.settings,
    required this.inputController,
    required this.selectedModelPath,
    required this.onModelChanged,
    required this.enableDetection,
    required this.onEnableDetectionChanged,
    required this.useFp16,
    required this.onUseFp16Changed,
    required this.confidence,
    required this.onConfidenceChanged,
    required this.iou,
    required this.onIouChanged,
    required this.submitting,
    required this.onCreateJob,
    required this.jobs,
    super.key,
  });

  final NeriSettings? settings;
  final TextEditingController inputController;
  final String? selectedModelPath;
  final ValueChanged<String?> onModelChanged;
  final bool enableDetection;
  final ValueChanged<bool> onEnableDetectionChanged;
  final bool useFp16;
  final ValueChanged<bool> onUseFp16Changed;
  final double confidence;
  final ValueChanged<double> onConfidenceChanged;
  final double iou;
  final ValueChanged<double> onIouChanged;
  final bool submitting;
  final VoidCallback onCreateJob;
  final List<ProcessingJob> jobs;

  @override
  Widget build(BuildContext context) {
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      children: [
        _HeroPanel(colorScheme: Theme.of(context).colorScheme),
        _CreateJobCard(
          settings: settings,
          inputController: inputController,
          selectedModelPath: selectedModelPath,
          onModelChanged: onModelChanged,
          enableDetection: enableDetection,
          onEnableDetectionChanged: onEnableDetectionChanged,
          useFp16: useFp16,
          onUseFp16Changed: onUseFp16Changed,
          confidence: confidence,
          onConfidenceChanged: onConfidenceChanged,
          iou: iou,
          onIouChanged: onIouChanged,
          submitting: submitting,
          onCreateJob: onCreateJob,
        ),
        _JobsCard(jobs: jobs),
      ],
    );
  }
}

class _HeroPanel extends StatelessWidget {
  const _HeroPanel({required this.colorScheme});

  final ColorScheme colorScheme;

  @override
  Widget build(BuildContext context) {
    return Card.filled(
      margin: const EdgeInsets.only(bottom: 16),
      color: colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(24),
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
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                color: colorScheme.onPrimaryContainer,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '使用 Flutter Material 3 前端连接 Python 后端，完成批量索引、EXIF 提取、YOLO 识别与结果导出。',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: colorScheme.onPrimaryContainer,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CreateJobCard extends StatelessWidget {
  const _CreateJobCard({
    required this.settings,
    required this.inputController,
    required this.selectedModelPath,
    required this.onModelChanged,
    required this.enableDetection,
    required this.onEnableDetectionChanged,
    required this.useFp16,
    required this.onUseFp16Changed,
    required this.confidence,
    required this.onConfidenceChanged,
    required this.iou,
    required this.onIouChanged,
    required this.submitting,
    required this.onCreateJob,
  });

  final NeriSettings? settings;
  final TextEditingController inputController;
  final String? selectedModelPath;
  final ValueChanged<String?> onModelChanged;
  final bool enableDetection;
  final ValueChanged<bool> onEnableDetectionChanged;
  final bool useFp16;
  final ValueChanged<bool> onUseFp16Changed;
  final double confidence;
  final ValueChanged<double> onConfidenceChanged;
  final double iou;
  final ValueChanged<double> onIouChanged;
  final bool submitting;
  final VoidCallback onCreateJob;

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: '新建处理任务',
      subtitle: '输入本机路径，由 Python 后端读取和处理文件。',
      icon: Icons.playlist_add_rounded,
      child: Column(
        children: [
          TextField(
            controller: inputController,
            decoration: const InputDecoration(
              labelText: '输入文件夹',
              hintText: '/path/to/camera-trap-folder',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          _ModelSelector(
            settings: settings,
            selectedModelPath: selectedModelPath,
            onChanged: onModelChanged,
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: enableDetection,
            onChanged: onEnableDetectionChanged,
            title: const Text('启用 YOLO 识别'),
            subtitle: const Text('关闭时仅进行快速文件索引和 EXIF 元数据提取。'),
          ),
          SwitchListTile(
            value: useFp16,
            onChanged: enableDetection ? onUseFp16Changed : null,
            title: const Text('使用 FP16 加速'),
          ),
          _ProcessingSlider(
            label: '置信度阈值',
            value: confidence,
            onChanged: onConfidenceChanged,
          ),
          _ProcessingSlider(
            label: 'IOU 阈值',
            value: iou,
            onChanged: onIouChanged,
          ),
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: submitting ? null : onCreateJob,
            icon: submitting
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.play_arrow_rounded),
            label: Text(submitting ? '提交中...' : '开始处理'),
          ),
        ],
      ),
    );
  }
}

class _ModelSelector extends StatelessWidget {
  const _ModelSelector({
    required this.settings,
    required this.selectedModelPath,
    required this.onChanged,
  });

  final NeriSettings? settings;
  final String? selectedModelPath;
  final ValueChanged<String?> onChanged;

  @override
  Widget build(BuildContext context) {
    final models = settings?.availableModels ?? const <ModelInfo>[];
    final selectedValue = models.any((model) => model.path == selectedModelPath)
        ? selectedModelPath
        : (models.isEmpty ? null : models.first.path);

    if (models.isEmpty) {
      return InputDecorator(
        decoration: InputDecoration(
          labelText: '模型文件',
          helperText:
              '未在 ${settings?.modelDirectory ?? _defaultModelDirectory} 中找到 .pt 模型',
          prefixIcon: const Icon(Icons.folder_off_rounded),
          border: const OutlineInputBorder(),
        ),
        child: const Text('暂无可用模型'),
      );
    }

    return DropdownMenu<String>(
      initialSelection: selectedValue,
      expandedInsets: EdgeInsets.zero,
      menuStyle: appDropdownMenuStyle(context),
      label: const Text('模型文件'),
      helperText:
          '扫描 ${settings?.modelDirectory ?? _defaultModelDirectory} 下的 .pt 文件',
      leadingIcon: const Icon(Icons.memory_rounded),
      dropdownMenuEntries: [
        for (final model in models)
          DropdownMenuEntry<String>(value: model.path, label: model.name),
      ],
      onSelected: onChanged,
    );
  }
}

class _ProcessingSlider extends StatelessWidget {
  const _ProcessingSlider({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
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
}

class _JobsCard extends StatelessWidget {
  const _JobsCard({required this.jobs});

  final List<ProcessingJob> jobs;

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: '任务进度',
      subtitle: jobs.isEmpty ? '暂无任务' : '自动每 2 秒刷新一次',
      icon: Icons.insights_rounded,
      child: jobs.isEmpty
          ? const Text('创建任务后，处理进度和识别结果会显示在这里。')
          : Column(children: jobs.map(_buildJobTile).toList()),
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
        ...job.results
            .take(12)
            .map(
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
            title: Text('还有 ${job.results.length - 12} 条结果，可在导出的 CSV 中查看。'),
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
}
