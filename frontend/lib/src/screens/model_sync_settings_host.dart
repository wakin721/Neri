import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../model_sync_controller.dart';
import '../models/model_sync_status.dart';
import 'model_sync_settings_card.dart';

class ModelSyncSettingsHost extends StatefulWidget {
  const ModelSyncSettingsHost({
    required this.apiClient,
    required this.enabled,
    required this.onCatalogChanged,
    required this.child,
    this.pollInterval = const Duration(milliseconds: 500),
    super.key,
  });

  final NeriApiClient apiClient;
  final bool enabled;
  final Future<void> Function() onCatalogChanged;
  final Widget child;
  final Duration pollInterval;

  @override
  State<ModelSyncSettingsHost> createState() => _ModelSyncSettingsHostState();
}

class _ModelSyncSettingsHostState extends State<ModelSyncSettingsHost> {
  late ModelSyncController _controller;
  OverlayEntry? _messageEntry;
  String? _dismissedStatusKey;
  String? _retryError;
  bool _retrying = false;

  @override
  void initState() {
    super.initState();
    _createController();
  }

  @override
  void didUpdateWidget(covariant ModelSyncSettingsHost oldWidget) {
    super.didUpdateWidget(oldWidget);
    final configurationChanged =
        oldWidget.apiClient != widget.apiClient ||
        oldWidget.pollInterval != widget.pollInterval ||
        oldWidget.onCatalogChanged != widget.onCatalogChanged;
    if (configurationChanged) {
      _removeMessageEntry();
      _controller.removeListener(_handleStatusChanged);
      _controller.dispose();
      _dismissedStatusKey = null;
      _retryError = null;
      _retrying = false;
      _createController();
      return;
    }
    if (oldWidget.enabled != widget.enabled) {
      _syncMessageEntry();
    }
  }

  void _createController() {
    _controller = ModelSyncController(
      widget.apiClient,
      pollInterval: widget.pollInterval,
      onCatalogChanged: widget.onCatalogChanged,
    )..addListener(_handleStatusChanged);
  }

  void _handleStatusChanged() {
    _retryError = null;
    _syncMessageEntry();
  }

  String _statusKey(ModelSyncStatus status) {
    final runId = status.runId?.trim();
    if (runId != null && runId.isNotEmpty) return runId;
    return <String>[
      status.state,
      status.manifestId ?? '',
      status.lastSuccessfulSync ?? '',
      status.error ?? '',
    ].join('|');
  }

  bool _shouldShow(ModelSyncStatus status) {
    if (!widget.enabled) return false;
    return status.isActive ||
        status.state == 'completed' ||
        status.state == 'failed';
  }

  void _syncMessageEntry() {
    if (!mounted) return;
    final status = _controller.status;
    if (status == null ||
        !_shouldShow(status) ||
        _dismissedStatusKey == _statusKey(status)) {
      _removeMessageEntry();
      return;
    }

    if (_messageEntry == null) {
      _messageEntry = OverlayEntry(builder: _buildMessageEntry);
      Overlay.of(context, rootOverlay: true).insert(_messageEntry!);
    } else {
      _messageEntry!.markNeedsBuild();
    }
  }

  Widget _buildMessageEntry(BuildContext context) {
    final status = _controller.status;
    if (status == null ||
        !_shouldShow(status) ||
        _dismissedStatusKey == _statusKey(status)) {
      return const SizedBox.shrink();
    }
    return _ModelSyncMessageCard(
      status: status,
      retryError: _retryError,
      retrying: _retrying,
      onClose: _dismissMessage,
      onRetry: status.state == 'failed' ? () => unawaited(_retrySync()) : null,
    );
  }

  void _dismissMessage() {
    final status = _controller.status;
    if (status != null) {
      _dismissedStatusKey = _statusKey(status);
    }
    _removeMessageEntry();
  }

  Future<void> _retrySync() async {
    if (_retrying || _controller.status?.isActive == true) return;
    _dismissedStatusKey = null;
    _retryError = null;
    _retrying = true;
    _messageEntry?.markNeedsBuild();
    try {
      await _controller.runNow();
    } catch (error) {
      _retryError = error.toString();
    } finally {
      _retrying = false;
      _syncMessageEntry();
    }
  }

  void _removeMessageEntry() {
    final entry = _messageEntry;
    _messageEntry = null;
    entry?.remove();
  }

  @override
  void dispose() {
    _removeMessageEntry();
    _controller.removeListener(_handleStatusChanged);
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: ModelSyncSettingsCard(
            apiClient: widget.apiClient,
            controller: _controller,
            enabled: widget.enabled,
            pollInterval: widget.pollInterval,
            onCatalogChanged: widget.onCatalogChanged,
          ),
        ),
        const SizedBox(height: 12),
        Expanded(child: widget.child),
      ],
    );
  }
}

class _ModelSyncMessageCard extends StatelessWidget {
  const _ModelSyncMessageCard({
    required this.status,
    required this.retrying,
    required this.onClose,
    required this.onRetry,
    this.retryError,
  });

  final ModelSyncStatus status;
  final String? retryError;
  final bool retrying;
  final VoidCallback onClose;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final active = status.isActive;
    final currentFile = status.currentFile?.trim() ?? '';
    final error = status.error?.trim() ?? '';
    final retryFailure = retryError?.trim() ?? '';

    return Positioned(
      right: 20,
      bottom: 280,
      child: Card(
        key: const Key('model-sync-message-card'),
        elevation: 7,
        child: SizedBox(
          width: 400,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 16, 14),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(_statusIcon(), color: scheme.primary),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            status.state == 'completed'
                                ? '模型同步完成'
                                : '模型同步',
                            style: theme.textTheme.titleSmall,
                          ),
                          const SizedBox(height: 3),
                          Text(
                            _statusLabel(),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: scheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      onPressed: onClose,
                      tooltip: '关闭',
                      icon: const Icon(Icons.close_rounded, size: 19),
                    ),
                  ],
                ),
                if (active) ...[
                  const SizedBox(height: 12),
                  LinearProgressIndicator(value: status.progress),
                  const SizedBox(height: 8),
                  Text(
                    _progressLabel(),
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                ],
                if (currentFile.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text(
                    '当前文件：$currentFile',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                if (status.state == 'completed') ...[
                  const SizedBox(height: 10),
                  Text(
                    '云端模型：探测 ${status.cloudDetectCount ?? 0} · '
                    '分类 ${status.cloudClsCount ?? 0}',
                  ),
                  if (status.lastSuccessfulSync?.trim().isNotEmpty == true) ...[
                    const SizedBox(height: 5),
                    Text(
                      '上次成功：${status.lastSuccessfulSync}',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ],
                if (status.state == 'failed' && error.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text('同步错误：$error'),
                ],
                if (retryFailure.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Text(
                    '重试失败：$retryFailure',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: scheme.error,
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                Text(
                  '用户模型不会被自动同步修改。',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                if (onRetry != null) ...[
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonalIcon(
                      onPressed: retrying ? null : onRetry,
                      icon: retrying
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.refresh_rounded),
                      label: Text(retrying ? '重试中…' : '重试'),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  IconData _statusIcon() {
    return switch (status.state) {
      'completed' => Icons.cloud_done_rounded,
      'failed' => Icons.cloud_off_rounded,
      _ => Icons.cloud_sync_rounded,
    };
  }

  String _statusLabel() {
    return switch (status.state) {
      'checking' => '正在检查 NeriCloud…',
      'downloading' => '正在下载模型…',
      'completed' => '同步完成',
      'failed' => '同步失败',
      _ => '模型同步',
    };
  }

  String _progressLabel() {
    final totalBytes = status.totalBytes;
    if (totalBytes != null && totalBytes > 0) {
      final percent = (status.receivedBytes * 100 / totalBytes).clamp(0, 100);
      return '下载进度 ${percent.toStringAsFixed(0)}%';
    }
    if (status.totalFiles > 0) {
      return '文件 ${status.completedFiles} / ${status.totalFiles}';
    }
    return status.state == 'checking' ? '正在检查模型清单' : '正在下载';
  }
}
