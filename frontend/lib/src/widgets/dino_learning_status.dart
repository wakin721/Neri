import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';

class DinoLearningStatus extends StatefulWidget {
  const DinoLearningStatus({
    required this.apiClient,
    required this.modelPath,
    this.active = true,
    this.onLearned,
    super.key,
  });

  final NeriApiClient apiClient;
  final String modelPath;
  final bool active;
  final VoidCallback? onLearned;

  @override
  State<DinoLearningStatus> createState() => _DinoLearningStatusState();
}

class _DinoLearningStatusState extends State<DinoLearningStatus> {
  Map<String, dynamic>? _status;
  String? _error;
  bool _loading = false;
  bool _working = false;
  int _request = 0;
  Timer? _timer;
  String? _lastCompletion;

  @override
  void initState() {
    super.initState();
    _activate();
  }

  void _activate() {
    _timer?.cancel();
    if (!widget.active) return;
    unawaited(_refresh());
    _timer = Timer.periodic(const Duration(seconds: 3), (_) {
      if (!_loading && !_working) unawaited(_refresh());
    });
  }

  @override
  void didUpdateWidget(covariant DinoLearningStatus oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.modelPath != widget.modelPath ||
        oldWidget.active != widget.active ||
        oldWidget.apiClient != widget.apiClient) {
      _request++;
      _loading = false;
      _status = null;
      _error = null;
      if (oldWidget.modelPath != widget.modelPath) _lastCompletion = null;
      _activate();
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _request++;
    super.dispose();
  }

  Future<void> _refresh() async {
    final request = ++_request;
    setState(() => _loading = true);
    try {
      final status = await widget.apiClient.dinoLearning(
        'status',
        widget.modelPath,
      );
      if (!mounted || request != _request) return;
      setState(() {
        _status = status;
        _error = null;
      });
      final completion = status['last_update']?.toString();
      if (status['state'] == 'ready' &&
          completion != null &&
          completion != _lastCompletion) {
        _lastCompletion = completion;
        widget.onLearned?.call();
      }
    } catch (error) {
      if (mounted && request == _request) setState(() => _error = '$error');
    } finally {
      if (mounted && request == _request) setState(() => _loading = false);
    }
  }

  Future<void> _run(String action, {String? calibrationPath}) async {
    if (_working) return;
    final model = widget.modelPath;
    setState(() => _working = true);
    try {
      await widget.apiClient.dinoLearning(
        action,
        model,
        calibrationPath: calibrationPath,
      );
      if (!mounted || widget.modelPath != model) return;
      await _refresh();
    } catch (error) {
      if (mounted && widget.modelPath == model) {
        setState(() => _error = '$error');
      }
    } finally {
      if (mounted) setState(() => _working = false);
    }
  }

  Future<void> _importCalibration() async {
    var path = '';
    final chosen = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('导入校准数据'),
        content: SizedBox(
          width: 480,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('旧模型需要与原编码器匹配的完整校准 NPZ。导入后会自动继续学习已保存的人工校验。'),
              const SizedBox(height: 12),
              TextField(
                autofocus: true,
                onChanged: (value) => path = value,
                decoration: const InputDecoration(labelText: '校准文件完整路径'),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, path.trim()),
            child: const Text('导入'),
          ),
        ],
      ),
    );
    if (chosen != null && chosen.isNotEmpty && mounted) {
      await _run('calibration', calibrationPath: chosen);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = _status?['state']?.toString() ?? 'idle';
    final title = switch (state) {
      'queued' => '等待学习',
      'running' => '正在学习',
      'ready' => '学习已完成',
      'blocked' => '学习暂不可用',
      'failed' => '学习失败',
      _ => '自动学习已就绪',
    };
    final pending = (_status?['pending_count'] as num?)?.toInt() ?? 0;
    final learned = (_status?['learned_count'] as num?)?.toInt() ?? 0;
    final classes = (_status?['class_count'] as num?)?.toInt() ?? 0;
    final rejection = _status?['rejection'] as Map<String, dynamic>? ?? {};
    final experiment = rejection['gate_mode'] == 'sequence';
    String threshold(String key) =>
        (rejection[key] as num?)?.toStringAsFixed(3) ?? '未记录';
    final species =
        _status?['learned_species_counts'] as Map<String, dynamic>? ?? {};
    final last = DateTime.tryParse(
      _status?['last_update']?.toString() ?? '',
    )?.toLocal();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _status == null && _loading ? '正在读取学习状态…' : title,
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: 6),
        const Text('人工确认或修改物种后在后台自动学习，完成后用于下一次识别。'),
        if (_status != null) ...[
          Text('当前模型：$classes 个物种 · 已学习 $learned 个独立样本 · 等待处理 $pending 个检测框'),
          if (species.isNotEmpty)
            Text(species.entries.map((e) => '${e.key}：${e.value}').join('，')),
          if (last != null) Text('最近完成：${last.toString().split('.').first}'),
          Text(_status?['message']?.toString() ?? ''),
          const Divider(),
          const Text(
            'DINO 拒识门限（只读）',
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
          Text(
            experiment
                ? '实验事件门限 ${threshold('sequence_threshold')}'
                : '单框门限 ${threshold('box_threshold')}  ·  序列门限 ${threshold('sequence_threshold')}',
          ),
          Text(
            experiment
                ? '事件分数达到门限才接纳。门限由模型自动校准，不能手动拖动。'
                : '分数越低越容易拒识；两级都通过才接纳。门限由模型自动校准，不能手动拖动。',
          ),
        ],
        if (_error != null)
          Text(
            _error!,
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
        if (state == 'running' || state == 'queued')
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: LinearProgressIndicator(),
          ),
        Wrap(
          spacing: 8,
          children: [
            TextButton(
              onPressed: _loading || _working ? null : _refresh,
              child: const Text('刷新学习状态'),
            ),
            if (_status?['calibration_available'] == false)
              TextButton(
                onPressed: _working ? null : _importCalibration,
                child: const Text('导入校准数据'),
              ),
            if (state == 'failed' ||
                (state == 'blocked' &&
                    _status?['calibration_available'] == true))
              TextButton(
                onPressed: _working ? null : () => _run('retry'),
                child: const Text('重试自动学习'),
              ),
          ],
        ),
      ],
    );
  }
}
