import 'package:flutter/material.dart';

import '../api_client.dart';
import 'model_sync_settings_card.dart';

class ModelSyncSettingsHost extends StatelessWidget {
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
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: ModelSyncSettingsCard(
            apiClient: apiClient,
            enabled: enabled,
            pollInterval: pollInterval,
            onCatalogChanged: onCatalogChanged,
          ),
        ),
        const SizedBox(height: 12),
        Expanded(child: child),
      ],
    );
  }
}
