import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../model_sync_controller.dart';
import '../models/model_sync_status.dart';

class ModelSyncSettingsCard extends StatefulWidget {
  const ModelSyncSettingsCard({
    required this.apiClient,
    required this.onCatalogChanged,
    this.controller,
    this.enabled = true,
    this.pollInterval = const Duration(milliseconds: 500),
    super.key,
  });

  final NeriApiClient apiClient;
  final Future<void> Function() onCatalogChanged;
  final ModelSyncController? controller;
  final bool enabled;
  final Duration pollInterval;

  @override
  State<ModelSyncSettingsCard> createState() => _ModelSyncSettingsCardState();
}

class _ModelSyncSettingsCardState extends State<ModelSyncSettingsCard> {
  late ModelSyncController _controller;
  bool _loading = true;
  bool _running = false;
  String? _requestError;

  @override
  void initState() {
    super.initState();
    _attachController();
    _loading = widget.enabled && _controller.status == null;
    if (widget.enabled) {
      unawaited(_loadStatus());
    }
  }

  @override
  void didUpdateWidget(ModelSyncSettingsCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    final controllerChanged = oldWidget.controller != widget.controller;
    final internalConfigurationChanged =
        widget.controller == null &&
        oldWidget.controller == null &&
        (oldWidget.apiClient != widget.apiClient ||
            oldWidget.pollInterval != widget.pollInterval ||
            oldWidget.onCatalogChanged != widget.onCatalogChanged);
    final rebindController =
        controllerChanged || internalConfigurationChanged;

    if (rebindController) {
      _detachController(dispose: oldWidget.controller == null);
      _attachController();
      _running = false;
      _requestError = null;
    }

    if (rebindController || oldWidget.enabled != widget.enabled) {
      _loading = widget.enabled && _controller.status == null;
      if (widget.enabled) {
        unawaited(_loadStatus());
      }
    }
  }

  void _attachController() {
    _controller =
        widget.controller ??
        ModelSyncController(
          widget.apiClient,
          pollInterval: widget.pollInterval,
          onCatalogChanged: widget.onCatalogChanged,
        );
    _controller.addListener(_handleStatusChanged);
  }

  void _detachController({required bool dispose}) {
    _controller.removeListener(_handleStatusChanged);
    if (dispose) _controller.dispose();
  }

  void _handleStatusChanged() {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _requestError = null;
    });
  }

  Future<void> _loadStatus() async {
    if (!widget.enabled) return;
    try {
      await _controller.refreshStatus();
    } catch (error) {
      if (!mounted || !widget.enabled) return;
      setState(() {
        _loading = false;
        _requestError = error.toString();
      });
    }
  }

  Future<void> _runNow() async {
    if (!widget.enabled ||
        _running ||
        _controller.status?.isActive == true) {
      return;
    }
    setState(() {
      _running = true;
      _requestError = null;
    });
    try {
      await _controller.runNow();
    } catch (error) {
      if (!mounted || !widget.enabled) return;
      setState(() => _requestError = error.toString());
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  void dispose() {
    _detachController(dispose: widget.controller == null);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final status = _controller.status;
    final active = status?.isActive == true;
    final error = status?.error?.trim();
    final requestError = _requestError?.trim();
    final progress = status?.progress;

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.cloud_sync_rounded),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '模型同步',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        _statusLabel(
                          status,
                          loading: _loading,
                          enabled: widget.enabled,
                        ),
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                FilledButton.tonalIcon(
                  onPressed: widget.enabled && !active && !_running
                      ? _runNow
                      : null,
                  icon: active || _running
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.sync_rounded),
                  label: Text(
                    active || _running
                        ? '同步中…'
                        : status?.state == 'failed'
                        ? '重试'
                        : '立即同步',
                  ),
                ),
              ],
            ),
            if (status?.currentFile?.isNotEmpty == true) ...[
              const SizedBox(height: 12),
              Text('当前文件：${status!.currentFile}'),
            ],
            if (active) ...[
              const SizedBox(height: 10),
              LinearProgressIndicator(value: progress),
              const SizedBox(height: 6),
              Text(_progressLabel(status!)),
            ],
            if (status?.cloudDetectCount != null ||
                status?.cloudClsCount != null) ...[
              const SizedBox(height: 10),
              Text(
                '云端模型：探测 ${status?.cloudDetectCount ?? 0} · '
                '分类 ${status?.cloudClsCount ?? 0}',
              ),
            ],
            if (status?.lastSuccessfulSync?.isNotEmpty == true) ...[
              const SizedBox(height: 6),
              Text('上次成功：${status!.lastSuccessfulSync}'),
            ],
            if (error != null && error.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text('同步错误：$error'),
            ],
            if (requestError != null && requestError.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text('同步请求失败：$requestError'),
            ],
            const SizedBox(height: 8),
            Text(
              '同步失败不会影响已安装的本地模型；用户模型目录不会被自动同步修改。',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  String _statusLabel(
    ModelSyncStatus? status, {
    required bool loading,
    required bool enabled,
  }) {
    if (!enabled) return '等待本地服务';
    if (loading && status == null) return '正在读取同步状态…';
    return switch (status?.state) {
      'checking' => '正在同步：检查 NeriCloud…',
      'downloading' => '正在同步：下载模型…',
      'completed' => '同步完成',
      'failed' => '同步失败',
      _ => '等待同步',
    };
  }

  String _progressLabel(ModelSyncStatus status) {
    final totalBytes = status.totalBytes;
    if (totalBytes != null && totalBytes > 0) {
      final percent = (status.receivedBytes * 100 / totalBytes).clamp(0, 100);
      return '下载进度 ${percent.toStringAsFixed(0)}%';
    }
    if (status.totalFiles > 0) {
      return '文件 ${status.completedFiles} / ${status.totalFiles}';
    }
    return '正在下载';
  }
}
