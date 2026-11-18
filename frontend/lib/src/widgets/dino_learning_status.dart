import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import 'dino_species_learning_dialog.dart';
import 'summary_dialog_button.dart';

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
  DialogRoute<void>? _detailsRoute;
  ValueNotifier<SpeciesLearningSnapshot>? _details;

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
      _closeDetails();
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
    _closeDetails();
    super.dispose();
  }

  void _closeDetails() {
    final route = _detailsRoute;
    if (route == null) return;
    _details = null;
    _detailsRoute = null;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (route.isActive) route.navigator?.removeRoute(route);
    });
  }

  Future<void> _showDetails() async {
    if (_detailsRoute != null) return;
    final data = ValueNotifier<SpeciesLearningSnapshot>((
      status: null,
      error: null,
      busy: true,
    ));
    final route = DialogRoute<void>(
      context: context,
      builder: (_) => DinoSpeciesLearningDialog(
        data: data,
        onRefresh: _refresh,
        onRetry: () => _run('retry'),
        onImportCalibration: _importCalibration,
      ),
    );
    _detailsRoute = route;
    _details = data;
    final closed = Navigator.of(context, rootNavigator: true).push(route);
    unawaited(_refresh());
    try {
      await closed;
    } finally {
      if (identical(_detailsRoute, route)) {
        _detailsRoute = null;
        _details = null;
      }
      unawaited(route.completed.whenComplete(data.dispose));
    }
  }

  void _publishDialog({Map<String, dynamic>? status}) {
    final details = _details;
    if (details == null) return;
    details.value = (
      status: status ?? details.value.status,
      error: _error,
      busy: _loading || _working,
    );
  }

  Future<void> _refresh() async {
    final request = ++_request;
    final details = _details;
    setState(() => _loading = true);
    _publishDialog();
    try {
      final status = await widget.apiClient.dinoLearning(
        'status',
        widget.modelPath,
        includeSpecies: details != null,
      );
      if (!mounted || request != _request) return;
      setState(() {
        _status = status;
        _error = null;
      });
      if (details != null && identical(details, _details)) {
        _publishDialog(status: status);
      }
      final completion = status['last_update']?.toString();
      if (status['state'] == 'ready' &&
          completion != null &&
          completion != _lastCompletion) {
        _lastCompletion = completion;
        widget.onLearned?.call();
      }
    } catch (error) {
      if (mounted && request == _request) {
        setState(() => _error = '$error');
      }
    } finally {
      if (mounted && request == _request) {
        setState(() => _loading = false);
        _publishDialog();
      }
    }
  }

  Future<void> _run(String action, {String? calibrationPath}) async {
    if (_working) return;
    final model = widget.modelPath;
    setState(() => _working = true);
    _publishDialog();
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
      if (mounted) {
        setState(() => _working = false);
        _publishDialog();
      }
    }
  }

  Future<void> _importCalibration() async {
    final model = widget.modelPath;
    var path = '';
    final chosen = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        scrollable: true,
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
    if (mounted &&
        widget.active &&
        widget.modelPath == model &&
        chosen != null &&
        chosen.isNotEmpty) {
      await _run('calibration', calibrationPath: chosen);
    }
  }

  @override
  Widget build(BuildContext context) {
    final label = _error != null
        ? '状态暂不可用'
        : _status == null
        ? (_loading ? '正在读取…' : '查看学习状态')
        : learningStateLabel(_status!);
    return Tooltip(
      message: '查看学习明细',
      child: SummaryDialogButton(
        summary: label,
        onPressed: widget.active ? _showDetails : null,
      ),
    );
  }
}
