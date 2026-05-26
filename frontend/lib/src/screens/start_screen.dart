import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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
    required this.selectedClassificationModelPath,
    required this.onClassificationModelChanged,
    required this.videoMode,
    required this.onVideoModeChanged,
    required this.vidStride,
    required this.onVidStrideChanged,
    required this.useFp16,
    required this.onUseFp16Changed,
    required this.confidence,
    required this.onConfidenceChanged,
    required this.iou,
    required this.onIouChanged,
    required this.submitting,
    required this.onCreateJob,
    required this.onCancelJob,
    required this.onResumeJob,
    required this.onDeleteJob,
    required this.onClearJobs,
    required this.pendingStartJobIds,
    required this.pendingStopJobIds,
    required this.jobs,
    super.key,
  });

  final NeriSettings? settings;
  final TextEditingController inputController;
  final String? selectedModelPath;
  final ValueChanged<String?> onModelChanged;
  final String? selectedClassificationModelPath;
  final ValueChanged<String?> onClassificationModelChanged;
  final String videoMode;
  final ValueChanged<String> onVideoModeChanged;
  final int vidStride;
  final ValueChanged<int> onVidStrideChanged;
  final bool useFp16;
  final ValueChanged<bool> onUseFp16Changed;
  final double confidence;
  final ValueChanged<double> onConfidenceChanged;
  final double iou;
  final ValueChanged<double> onIouChanged;
  final bool submitting;
  final VoidCallback onCreateJob;
  final ValueChanged<ProcessingJob> onCancelJob;
  final ValueChanged<ProcessingJob> onResumeJob;
  final ValueChanged<ProcessingJob> onDeleteJob;
  final VoidCallback onClearJobs;
  final Set<String> pendingStartJobIds;
  final Set<String> pendingStopJobIds;
  final List<ProcessingJob> jobs;

  @override
  Widget build(BuildContext context) {
    final detectionSettingsEnabled =
        settings == null || settings!.missingYoloDependencies.isEmpty;
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      children: [
        _HeroPanel(colorScheme: Theme.of(context).colorScheme),
        _CreateJobCard(
          settings: settings,
          detectionSettingsEnabled: detectionSettingsEnabled,
          inputController: inputController,
          selectedModelPath: selectedModelPath,
          onModelChanged: onModelChanged,
          selectedClassificationModelPath: selectedClassificationModelPath,
          onClassificationModelChanged: onClassificationModelChanged,
          videoMode: videoMode,
          onVideoModeChanged: onVideoModeChanged,
          vidStride: vidStride,
          onVidStrideChanged: onVidStrideChanged,
          useFp16: useFp16,
          onUseFp16Changed: onUseFp16Changed,
          confidence: confidence,
          onConfidenceChanged: onConfidenceChanged,
          iou: iou,
          onIouChanged: onIouChanged,
          submitting: submitting,
          onCreateJob: onCreateJob,
        ),
        _JobsCard(
          jobs: jobs,
          onCancelJob: onCancelJob,
          onResumeJob: onResumeJob,
          onDeleteJob: onDeleteJob,
          onClearJobs: onClearJobs,
          pendingStartJobIds: pendingStartJobIds,
          pendingStopJobIds: pendingStopJobIds,
        ),
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
              'Neri: NERI Enables Rapid Identification',
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
    required this.detectionSettingsEnabled,
    required this.inputController,
    required this.selectedModelPath,
    required this.onModelChanged,
    required this.selectedClassificationModelPath,
    required this.onClassificationModelChanged,
    required this.videoMode,
    required this.onVideoModeChanged,
    required this.vidStride,
    required this.onVidStrideChanged,
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
  final bool detectionSettingsEnabled;
  final TextEditingController inputController;
  final String? selectedModelPath;
  final ValueChanged<String?> onModelChanged;
  final String? selectedClassificationModelPath;
  final ValueChanged<String?> onClassificationModelChanged;
  final String videoMode;
  final ValueChanged<String> onVideoModeChanged;
  final int vidStride;
  final ValueChanged<int> onVidStrideChanged;
  final bool useFp16;
  final ValueChanged<bool> onUseFp16Changed;
  final double confidence;
  final ValueChanged<double> onConfidenceChanged;
  final double iou;
  final ValueChanged<double> onIouChanged;
  final bool submitting;
  final VoidCallback onCreateJob;

  static const _dialogsChannel = MethodChannel('neri/dialogs');

  Future<void> _selectInputFolder(BuildContext context) async {
    try {
      final selected = await _dialogsChannel.invokeMethod<String>(
        'chooseDirectory',
        <String, String>{'initialDirectory': inputController.text.trim()},
      );
      if (selected == null || selected.isEmpty) return;
      inputController
        ..text = selected
        ..selection = TextSelection.collapsed(offset: selected.length);
    } on PlatformException catch (error) {
      if (!context.mounted) return;
      final message = error.message?.isNotEmpty == true
          ? error.message!
          : '无法打开目录选择器';
      ScaffoldMessenger.maybeOf(
        context,
      )?.showSnackBar(SnackBar(content: Text(message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final fp16Enabled =
        detectionSettingsEnabled && settings?.gpuAvailable == true;
    return SectionCard(
      title: '新建处理任务',
      subtitle: '输入本机路径，由 Python 后端读取和处理文件。',
      icon: Icons.playlist_add_rounded,
      child: Column(
        children: [
          TextField(
            controller: inputController,
            decoration: InputDecoration(
              labelText: '输入文件夹',
              hintText: '/path/to/camera-trap-folder',
              border: const OutlineInputBorder(),
              suffixIcon: IconButton(
                tooltip: '选择文件夹',
                onPressed: () => _selectInputFolder(context),
                icon: const Icon(Icons.folder_open_rounded),
              ),
            ),
          ),
          const SizedBox(height: 12),
          _StartOptionGrid(
            settings: settings,
            enabled: detectionSettingsEnabled,
            selectedModelPath: selectedModelPath,
            onChanged: onModelChanged,
            selectedClassificationModelPath: selectedClassificationModelPath,
            onClassificationModelChanged: onClassificationModelChanged,
            videoMode: videoMode,
            onVideoModeChanged: onVideoModeChanged,
            vidStride: vidStride,
            onVidStrideChanged: onVidStrideChanged,
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: fp16Enabled && useFp16,
            onChanged: fp16Enabled ? onUseFp16Changed : null,
            title: const Text('使用 FP16 加速'),
            subtitle: Text(
              fp16Enabled
                  ? '需要支持半精度推理的 GPU 或 Intel XPU'
                  : '未检测到 GPU / Intel XPU，已禁用 FP16',
            ),
          ),
          _ProcessingSlider(
            label: '置信度阈值',
            value: confidence,
            onChanged: detectionSettingsEnabled ? onConfidenceChanged : null,
          ),
          _ProcessingSlider(
            label: 'IOU 阈值',
            value: iou,
            onChanged: detectionSettingsEnabled ? onIouChanged : null,
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

class _StartOptionGrid extends StatelessWidget {
  const _StartOptionGrid({
    required this.settings,
    required this.enabled,
    required this.selectedModelPath,
    required this.onChanged,
    required this.selectedClassificationModelPath,
    required this.onClassificationModelChanged,
    required this.videoMode,
    required this.onVideoModeChanged,
    required this.vidStride,
    required this.onVidStrideChanged,
  });

  final NeriSettings? settings;
  final bool enabled;
  final String? selectedModelPath;
  final ValueChanged<String?> onChanged;
  final String? selectedClassificationModelPath;
  final ValueChanged<String?> onClassificationModelChanged;
  final String videoMode;
  final ValueChanged<String> onVideoModeChanged;
  final int vidStride;
  final ValueChanged<int> onVidStrideChanged;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth >= 1040
            ? 4
            : constraints.maxWidth >= 620
            ? 2
            : 1;
        const spacing = 12.0;
        final itemWidth =
            (constraints.maxWidth - spacing * (columns - 1)) / columns;
        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: [
            SizedBox(
              width: itemWidth,
              child: _ModelSelector(
                settings: settings,
                enabled: enabled,
                selectedModelPath: selectedModelPath,
                onChanged: onChanged,
              ),
            ),
            SizedBox(
              width: itemWidth,
              child: _ClassificationModelSelector(
                settings: settings,
                enabled: enabled,
                selectedClassificationModelPath:
                    selectedClassificationModelPath,
                onChanged: onClassificationModelChanged,
              ),
            ),
            SizedBox(
              width: itemWidth,
              child: _VideoModeSelector(
                enabled: enabled,
                videoMode: videoMode,
                onChanged: onVideoModeChanged,
              ),
            ),
            SizedBox(
              width: itemWidth,
              child: _VideoStrideSelector(
                enabled: enabled,
                vidStride: vidStride,
                onChanged: onVidStrideChanged,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _ModelSelector extends StatelessWidget {
  const _ModelSelector({
    required this.settings,
    required this.enabled,
    required this.selectedModelPath,
    required this.onChanged,
  });

  final NeriSettings? settings;
  final bool enabled;
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
          helperText: enabled
              ? '未在 ${settings?.modelDirectory ?? _defaultModelDirectory} 中找到 .pt 模型'
              : '检测依赖未安装，安装完成后可选择模型',
          prefixIcon: const Icon(Icons.folder_off_rounded),
          border: const OutlineInputBorder(),
          enabled: enabled,
        ),
        child: const Text('暂无可用模型'),
      );
    }

    return DropdownMenu<String>(
      initialSelection: selectedValue,
      expandedInsets: EdgeInsets.zero,
      enabled: enabled,
      menuStyle: appDropdownMenuStyle(context),
      label: const Text('模型文件'),
      helperText: '扫描 ${settings?.modelDirectory ?? _defaultModelDirectory}',
      leadingIcon: const Icon(Icons.memory_rounded),
      dropdownMenuEntries: [
        for (final model in models)
          DropdownMenuEntry<String>(value: model.path, label: model.name),
      ],
      onSelected: enabled ? onChanged : null,
    );
  }
}

class _ClassificationModelSelector extends StatelessWidget {
  const _ClassificationModelSelector({
    required this.settings,
    required this.enabled,
    required this.selectedClassificationModelPath,
    required this.onChanged,
  });

  final NeriSettings? settings;
  final bool enabled;
  final String? selectedClassificationModelPath;
  final ValueChanged<String?> onChanged;

  @override
  Widget build(BuildContext context) {
    final models =
        settings?.availableClassificationModels ?? const <ModelInfo>[];
    final selectedValue =
        selectedClassificationModelPath == null ||
            selectedClassificationModelPath!.isEmpty
        ? ''
        : (models.any((model) => model.path == selectedClassificationModelPath)
              ? selectedClassificationModelPath
              : '');

    return DropdownMenu<String>(
      initialSelection: selectedValue,
      expandedInsets: EdgeInsets.zero,
      enabled: enabled,
      menuStyle: appDropdownMenuStyle(context),
      label: const Text('分类模型'),
      helperText:
          '扫描 ${settings?.classificationModelDirectory ?? 'res/model_cls'}',
      leadingIcon: const Icon(Icons.account_tree_rounded),
      dropdownMenuEntries: [
        const DropdownMenuEntry<String>(value: '', label: '不使用'),
        for (final model in models)
          DropdownMenuEntry<String>(value: model.path, label: model.name),
      ],
      onSelected: enabled ? onChanged : null,
    );
  }
}

class _VideoModeSelector extends StatelessWidget {
  const _VideoModeSelector({
    required this.enabled,
    required this.videoMode,
    required this.onChanged,
  });

  final bool enabled;
  final String videoMode;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    final selectedValue = videoMode == 'fast' ? 'fast' : 'all';
    return DropdownMenu<String>(
      initialSelection: selectedValue,
      expandedInsets: EdgeInsets.zero,
      enabled: enabled,
      menuStyle: appDropdownMenuStyle(context),
      label: const Text('视频处理模式'),
      helperText: selectedValue == 'fast' ? '抽帧批量识别' : '按帧间隔追踪',
      leadingIcon: const Icon(Icons.video_collection_rounded),
      dropdownMenuEntries: const [
        DropdownMenuEntry<String>(value: 'all', label: '全部识别'),
        DropdownMenuEntry<String>(value: 'fast', label: '快速识别'),
      ],
      onSelected: enabled
          ? (value) {
              if (value != null) onChanged(value);
            }
          : null,
    );
  }
}

class _VideoStrideSelector extends StatelessWidget {
  const _VideoStrideSelector({
    required this.enabled,
    required this.vidStride,
    required this.onChanged,
  });

  final bool enabled;
  final int vidStride;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final selectedValue = vidStride.clamp(1, 30).toInt();
    return DropdownMenu<int>(
      initialSelection: selectedValue,
      expandedInsets: EdgeInsets.zero,
      enabled: enabled,
      menuStyle: appDropdownMenuStyle(context),
      label: const Text('视频跳帧'),
      helperText: '全部识别为帧间隔，快速识别为抽帧数',
      leadingIcon: const Icon(Icons.skip_next_rounded),
      dropdownMenuEntries: [
        for (var value = 1; value <= 30; value++)
          DropdownMenuEntry<int>(value: value, label: value.toString()),
      ],
      onSelected: enabled
          ? (value) {
              if (value != null) onChanged(value);
            }
          : null,
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
  final ValueChanged<double>? onChanged;

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
  const _JobsCard({
    required this.jobs,
    required this.onCancelJob,
    required this.onResumeJob,
    required this.onDeleteJob,
    required this.onClearJobs,
    required this.pendingStartJobIds,
    required this.pendingStopJobIds,
  });

  final List<ProcessingJob> jobs;
  final ValueChanged<ProcessingJob> onCancelJob;
  final ValueChanged<ProcessingJob> onResumeJob;
  final ValueChanged<ProcessingJob> onDeleteJob;
  final VoidCallback onClearJobs;
  final Set<String> pendingStartJobIds;
  final Set<String> pendingStopJobIds;

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: '任务进度',
      subtitle: jobs.isEmpty ? '暂无任务' : '自动每 2 秒刷新一次',
      icon: Icons.insights_rounded,
      child: jobs.isEmpty
          ? const Text('创建任务后，处理进度和识别结果会显示在这里。')
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                ...jobs.map((job) => _buildJobTile(context, job)),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  onPressed: () => _confirmClearJobs(context),
                  icon: const Icon(Icons.delete_sweep_outlined),
                  label: const Text('清除所有任务'),
                ),
              ],
            ),
    );
  }

  Widget _buildJobTile(BuildContext context, ProcessingJob job) {
    final runtimeText = _jobRuntimeText(job);
    return ExpansionTile(
      leading: Icon(_jobIcon(job.state)),
      title: Text(job.inputDir, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (job.message.isNotEmpty) ...[
            Text(job.message, maxLines: 1, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 8),
          ],
          LinearProgressIndicator(value: _jobProgressValue(job)),
          if (runtimeText.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              runtimeText,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ],
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Chip(label: Text('${job.processed}/${job.total}')),
          if (job.isActive) ...[
            const SizedBox(width: 8),
            IconButton(
              tooltip: '停止任务',
              onPressed: () => onCancelJob(job),
              icon: const Icon(Icons.stop_circle_outlined),
            ),
          ] else if (job.canResume) ...[
            const SizedBox(width: 8),
            IconButton(
              tooltip: '继续任务',
              onPressed: () => onResumeJob(job),
              icon: const Icon(Icons.play_circle_outline_rounded),
            ),
          ],
          const SizedBox(width: 8),
          IconButton(
            tooltip: '删除任务',
            onPressed: () => _confirmDeleteJob(context, job),
            icon: const Icon(Icons.delete_outline_rounded),
          ),
        ],
      ),
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

  double? _jobProgressValue(ProcessingJob job) {
    if (pendingStartJobIds.contains(job.id) ||
        pendingStopJobIds.contains(job.id)) {
      return null;
    }
    if (job.total == 0 && job.isActive) return null;
    return job.progress;
  }

  String _jobRuntimeText(ProcessingJob job) {
    final parts = <String>[];
    if (job.speed > 0) {
      parts.add('${job.speed.toStringAsFixed(job.speed >= 10 ? 1 : 2)} 个/秒');
    }
    if (job.elapsedSeconds > 0) {
      parts.add('已用 ${_formatDuration(job.elapsedSeconds)}');
    }
    final remaining = job.remainingSeconds;
    if (job.isActive &&
        remaining != null &&
        remaining.isFinite &&
        remaining >= 0) {
      parts.add('剩余 ${_formatDuration(remaining)}');
    }
    return parts.join(' · ');
  }

  String _formatDuration(double seconds) {
    final totalSeconds = seconds.round();
    if (totalSeconds < 60) return '${totalSeconds}s';
    final minutes = totalSeconds ~/ 60;
    final restSeconds = totalSeconds % 60;
    if (minutes < 60) return '${minutes}m${restSeconds}s';
    final hours = minutes ~/ 60;
    final restMinutes = minutes % 60;
    return '${hours}h${restMinutes}m';
  }

  IconData _jobIcon(String state) {
    return switch (state) {
      'completed' => Icons.check_circle_rounded,
      'failed' => Icons.error_rounded,
      'running' => Icons.sync_rounded,
      'cancelled' => Icons.stop_circle_rounded,
      _ => Icons.schedule_rounded,
    };
  }

  Future<void> _confirmDeleteJob(
    BuildContext context,
    ProcessingJob job,
  ) async {
    final inputPath = job.inputDir.trim().isEmpty ? '未知路径' : job.inputDir;
    final status = job.message.trim();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除任务？'),
        content: Text(
          status.isEmpty
              ? '将从任务进度中删除：\n$inputPath'
              : '将从任务进度中删除：\n$inputPath\n\n当前状态：$status',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('删除'),
          ),
        ],
      ),
    );
    if (confirmed == true) onDeleteJob(job);
  }

  Future<void> _confirmClearJobs(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('清除所有任务？'),
        content: const Text('所有任务进度都会从列表中移除，正在运行的任务也会被停止。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('清除'),
          ),
        ],
      ),
    );
    if (confirmed == true) onClearJobs();
  }
}
