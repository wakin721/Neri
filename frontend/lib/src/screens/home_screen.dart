import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models/job.dart';
import '../models/settings.dart';
import '../widgets/section_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({required this.apiClient, super.key});

  final NeriApiClient apiClient;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _inputController = TextEditingController();
  final _outputController = TextEditingController();
  final _modelController = TextEditingController();

  NeriSettings? _settings;
  List<ProcessingJob> _jobs = const <ProcessingJob>[];
  Timer? _timer;
  bool _loading = true;
  bool _submitting = false;
  bool _enableDetection = false;
  bool _useFp16 = false;
  double _confidence = 0.25;
  double _iou = 0.45;
  String? _error;
  int _selectedIndex = 0;

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
    _refresh();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _refresh(silent: true));
  }

  @override
  void dispose() {
    _timer?.cancel();
    _inputController.dispose();
    _outputController.dispose();
    _modelController.dispose();
    widget.apiClient.close();
    super.dispose();
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
    if (_inputController.text.trim().isEmpty) {
      setState(() => _error = '请输入红外相机媒体文件夹路径');
      return;
    }
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      await widget.apiClient.createJob(
        inputDir: _inputController.text.trim(),
        outputDir: _outputController.text.trim(),
        modelPath: _modelController.text.trim(),
        confidence: _confidence,
        iou: _iou,
        useFp16: _useFp16,
        enableDetection: _enableDetection,
      );
      await _refresh(silent: true);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = '创建任务失败：$error');
    } finally {
      if (mounted) setState(() => _submitting = false);
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
            onPressed: _loading ? null : _refresh,
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
    final results = _jobs.expand((job) => job.results).toList();
    return _buildPageList([
      SectionCard(
        title: '图像预览',
        subtitle: results.isEmpty
            ? '任务完成后会展示最近的媒体文件结果。'
            : '展示最近 ${results.length} 条处理结果。',
        icon: Icons.photo_library_rounded,
        child: results.isEmpty
            ? const Text('暂无可预览图像。请先在开始界面提交处理任务。')
            : Column(
                children: results.take(24).map((item) {
                  return ListTile(
                    leading: const Icon(Icons.image_rounded),
                    title: Text(item.filename),
                    subtitle: Text(item.path),
                    trailing: item.width == null || item.height == null
                        ? Chip(label: Text(item.fileType))
                        : Chip(label: Text('${item.width}×${item.height}')),
                  );
                }).toList(),
              ),
      ),
    ]);
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
          TextField(
            controller: _outputController,
            decoration: const InputDecoration(
              labelText: '导出文件夹（可选）',
              hintText: '/path/to/export',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _modelController,
            enabled: _enableDetection,
            decoration: const InputDecoration(
              labelText: '模型路径（可选）',
              hintText: '默认使用 res/model/11s_1225.pt',
              border: OutlineInputBorder(),
            ),
          ),
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
