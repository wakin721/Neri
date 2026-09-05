import 'package:flutter/material.dart';

import 'privacy_consent_dialog.dart';
import 'privacy_status.dart';

const privacyRetryButtonKey = Key('privacy-retry-button');
const privacyLoadErrorCloseButtonKey = Key('privacy-load-error-close-button');

class PrivacyGateOverlay extends StatelessWidget {
  const PrivacyGateOverlay({
    required this.status,
    required this.loading,
    required this.onRetry,
    required this.onSave,
    required this.onSaved,
    required this.onCloseApp,
    this.loadError,
    super.key,
  });

  final PrivacyStatus? status;
  final bool loading;
  final String? loadError;
  final VoidCallback onRetry;
  final PrivacySaveCallback onSave;
  final ValueChanged<PrivacyStatus> onSaved;
  final VoidCallback onCloseApp;

  @override
  Widget build(BuildContext context) {
    if (status?.isConsentComplete == true) return const SizedBox.shrink();
    if (status != null) {
      return PrivacyConsentDialog(
        onSave: onSave,
        onSaved: onSaved,
        onCloseApp: onCloseApp,
      );
    }

    final scheme = Theme.of(context).colorScheme;
    return BlockSemantics(
      child: Material(
        color: Colors.black54,
        child: SafeArea(
          minimum: const EdgeInsets.all(16),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 520),
              child: Card(
                elevation: 12,
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (loading) ...[
                        const CircularProgressIndicator(),
                        const SizedBox(height: 18),
                        Text(
                          '正在读取隐私设置…',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ] else ...[
                        Icon(
                          Icons.cloud_off_rounded,
                          size: 42,
                          color: scheme.error,
                        ),
                        const SizedBox(height: 14),
                        Text(
                          '无法读取隐私设置',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 10),
                        Text(
                          '为避免在未完成选择时进入工作区，请重试连接本地服务。\n${loadError ?? '本地服务暂时不可用。'}',
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 20),
                        Wrap(
                          alignment: WrapAlignment.center,
                          spacing: 12,
                          runSpacing: 8,
                          children: [
                            OutlinedButton(
                              key: privacyLoadErrorCloseButtonKey,
                              onPressed: onCloseApp,
                              child: const Text('关闭应用'),
                            ),
                            FilledButton.icon(
                              key: privacyRetryButtonKey,
                              onPressed: onRetry,
                              icon: const Icon(Icons.refresh_rounded),
                              label: const Text('重试'),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
