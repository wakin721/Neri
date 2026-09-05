import 'dart:async';

import 'package:flutter/material.dart';

import '../api_client.dart';
import 'privacy_consent_dialog.dart';
import 'privacy_status.dart';

const privacyEnableButtonKey = Key('privacy-enable-button');
const privacyDisableButtonKey = Key('privacy-disable-button');

class PrivacySettingsCard extends StatefulWidget {
  const PrivacySettingsCard({
    required this.apiClient,
    this.onStatusChanged,
    this.isActive = true,
    super.key,
  });

  final NeriApiClient apiClient;
  final ValueChanged<PrivacyStatus>? onStatusChanged;
  final bool isActive;

  @override
  State<PrivacySettingsCard> createState() => _PrivacySettingsCardState();
}

class _PrivacySettingsCardState extends State<PrivacySettingsCard> {
  PrivacyStatus? _status;
  bool _loading = true;
  bool _mutating = false;
  bool _refreshing = false;
  int _requestGeneration = 0;
  Timer? _refreshTimer;
  String? _error;

  @override
  void initState() {
    super.initState();
    _startRefreshing();
  }

  void _startRefreshing() {
    if (!widget.isActive) return;
    unawaited(_refresh());
    _refreshTimer = Timer.periodic(
      privacyStatusRefreshInterval,
      (_) => unawaited(_refresh()),
    );
  }

  @override
  void didUpdateWidget(covariant PrivacySettingsCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.isActive == widget.isActive) return;
    _refreshTimer?.cancel();
    _requestGeneration++;
    _startRefreshing();
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    if (!mounted || !widget.isActive || _refreshing || _mutating) return;
    _refreshing = true;
    final generation = ++_requestGeneration;
    try {
      final status = await widget.apiClient.fetchPrivacyStatus();
      if (generation == _requestGeneration) _setStatus(status);
    } catch (error) {
      if (mounted && generation == _requestGeneration) {
        setState(() => _error = '读取隐私状态失败，将自动重试：$error');
      }
    } finally {
      _refreshing = false;
      if (mounted) setState(() => _loading = false);
    }
  }

  void _setStatus(PrivacyStatus status) {
    if (!mounted) return;
    _requestGeneration++;
    setState(() {
      _status = status;
      _error = null;
    });
    widget.onStatusChanged?.call(status);
  }

  Future<void> _mutate(Future<PrivacyStatus> Function() operation) async {
    if (_mutating) return;
    _requestGeneration++;
    setState(() {
      _mutating = true;
      _error = null;
    });
    try {
      _setStatus(await operation());
    } catch (error) {
      if (mounted) setState(() => _error = '更新隐私设置失败：$error');
    } finally {
      if (mounted) setState(() => _mutating = false);
    }
  }

  Future<void> _openConsent() async {
    final status = await showDialog<PrivacyStatus>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) => PrivacyConsentDialog(
        showCloseApp: false,
        onCancel: () => Navigator.of(dialogContext).pop(),
        onCloseApp: () => Navigator.of(dialogContext).pop(),
        onSave: (enabled) =>
            widget.apiClient.savePrivacyStatus(trainingEnabled: enabled),
        onSaved: (saved) => Navigator.of(dialogContext).pop(saved),
      ),
    );
    if (status != null) _setStatus(status);
  }

  Future<void> _clearQueue() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('清除待处理任务？'),
        content: const Text('将丢弃尚未上传和等待重试的任务；已完成的记录与空照片配额会保留。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('清除'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await _mutate(widget.apiClient.clearPrivacyQueue);
    }
  }

  Future<void> _showAgreement() => showPrivacyAgreementDialog(context);

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final status = _status;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(Icons.privacy_tip_rounded, color: scheme.primary),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '隐私与数据',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const Text('用户协议、模型改进计划和待处理提交'),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            if (_loading && status == null)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(20),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (status == null)
              _PrivacyError(message: _error ?? '隐私状态不可用，将自动重试')
            else ...[
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: status.trainingEnabled
                      ? scheme.primaryContainer.withValues(alpha: 0.5)
                      : scheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Icon(
                      status.trainingEnabled
                          ? Icons.cloud_upload_rounded
                          : Icons.cloud_off_rounded,
                      color: status.trainingEnabled
                          ? scheme.primary
                          : scheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        status.trainingEnabled ? '模型改进计划已启用' : '模型改进计划未启用',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '协议版本 ${status.agreementVersion.isEmpty ? '未知' : status.agreementVersion} · ${status.hasCurrentAgreement ? '已同意' : '需重新确认'}',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 14,
                runSpacing: 8,
                children: [
                  Text('待处理 ${status.stats.pending}'),
                  Text('上传中 ${status.stats.uploading}'),
                  Text('已上传 ${status.stats.uploaded}'),
                  Text('失败 ${status.stats.failed}'),
                  Text('已跳过 ${status.stats.skipped}'),
                ],
              ),
              if (_error case final error?) ...[
                const SizedBox(height: 12),
                _PrivacyError(message: error),
              ],
              const SizedBox(height: 16),
              Wrap(
                spacing: 10,
                runSpacing: 8,
                children: [
                  OutlinedButton.icon(
                    onPressed: _mutating ? null : _showAgreement,
                    icon: const Icon(Icons.description_outlined),
                    label: const Text('查看完整协议'),
                  ),
                  if (status.trainingEnabled)
                    FilledButton.tonalIcon(
                      key: privacyDisableButtonKey,
                      onPressed: _mutating
                          ? null
                          : () => _mutate(
                              () => widget.apiClient.savePrivacyStatus(
                                trainingEnabled: false,
                              ),
                            ),
                      icon: const Icon(Icons.pause_circle_outline_rounded),
                      label: const Text('停止参加'),
                    )
                  else
                    FilledButton.icon(
                      key: privacyEnableButtonKey,
                      onPressed: _mutating ? null : _openConsent,
                      icon: const Icon(Icons.volunteer_activism_rounded),
                      label: const Text('了解并选择参加'),
                    ),
                  OutlinedButton.icon(
                    onPressed: _mutating || status.stats.waiting == 0
                        ? null
                        : _clearQueue,
                    icon: const Icon(Icons.delete_sweep_outlined),
                    label: const Text('清除待处理任务'),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                '停止参加会在下一次传输前取消等待中或进行中的任务。远程未完成上传创建满 20 分钟后进入清理；已完成照片和 JSON 自修订创建起最长保存 365 天，故障导致的到期清理会在服务恢复后重试。也可联系 wakin721@outlook.com 申请提前删除。',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _PrivacyError extends StatelessWidget {
  const _PrivacyError({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.errorContainer,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline_rounded, color: scheme.onErrorContainer),
          const SizedBox(width: 10),
          Expanded(child: Text(message)),
        ],
      ),
    );
  }
}
