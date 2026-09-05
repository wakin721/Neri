import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api_client.dart';
import 'privacy_status.dart';
import 'training_upload_diagnostics.dart';

class TrainingUploadDebugDialog extends StatefulWidget {
  const TrainingUploadDebugDialog({
    required this.apiClient,
    required this.onShowMessage,
    super.key,
  });

  final NeriApiClient apiClient;
  final ValueChanged<String> onShowMessage;

  @override
  State<TrainingUploadDebugDialog> createState() =>
      _TrainingUploadDebugDialogState();
}

class _TrainingUploadDebugDialogState extends State<TrainingUploadDebugDialog> {
  TrainingUploadDiagnostics? _details;
  Timer? _timer;
  bool _refreshing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    unawaited(_refresh());
    _timer = Timer.periodic(
      privacyStatusRefreshInterval,
      (_) => unawaited(_refresh()),
    );
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    if (!mounted || _refreshing) return;
    _refreshing = true;
    try {
      final details = await widget.apiClient.fetchTrainingUploadDiagnostics();
      if (!mounted) return;
      setState(() {
        _details = details;
        _error = null;
      });
    } catch (_) {
      if (mounted) {
        setState(
          () => _error = _details == null
              ? '上传设置读取失败，将自动重试。'
              : '暂时无法更新，以下为上次读取的内容，将自动重试。',
        );
      }
    } finally {
      _refreshing = false;
    }
  }

  Future<void> _copy() async {
    final details = _details;
    if (details == null) return;
    await Clipboard.setData(ClipboardData(text: details.copyText));
    if (mounted) widget.onShowMessage('上传设置已复制');
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final details = _details;
    return AlertDialog(
      title: const Text('上传详细设置'),
      scrollable: true,
      content: SizedBox(
        width: 620,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('当前上传参数与状态 · 每 2 秒自动更新'),
            if (_error case final error?)
              Padding(
                padding: const EdgeInsets.only(top: 12),
                child: Text(
                  error,
                  style: TextStyle(color: theme.colorScheme.error),
                ),
              ),
            if (details == null && _error == null)
              const Padding(
                padding: EdgeInsets.all(24),
                child: Center(child: CircularProgressIndicator()),
              ),
            if (details != null)
              for (final entry in details.facts.entries)
                Padding(
                  padding: const EdgeInsets.only(top: 14),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        entry.key,
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: theme.colorScheme.primary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      SelectableText(entry.value),
                    ],
                  ),
                ),
          ],
        ),
      ),
      actions: [
        TextButton.icon(
          onPressed: details == null ? null : _copy,
          icon: const Icon(Icons.content_copy_rounded),
          label: const Text('复制上传设置'),
        ),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('关闭'),
        ),
      ],
    );
  }
}
