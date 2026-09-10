import 'api_client.dart';

String? dinoV3StartupStatusMessage(DinoV3ComponentStatus status) {
  if (!status.installed) {
    return 'DINOv3 尚未安装，可进入设置安装。';
  }
  if (status.healthy) return null;

  final reason = status.message.trim();
  final lower = reason.toLowerCase();
  final mentionsTorch = lower.contains('pytorch') || lower.contains('torch');
  final indicatesMissing =
      reason.contains('未安装') ||
      lower.contains('not installed') ||
      lower.contains('no module named') ||
      lower.contains('missing') ||
      lower.contains('unavailable');
  if (mentionsTorch && indicatesMissing) {
    return 'DINOv3 已安装，但 PyTorch 未安装，当前不可推理';
  }
  if (reason.isEmpty) return 'DINOv3 安装异常';
  return 'DINOv3 安装异常：$reason';
}

class DinoV3StartupCheck {
  int? _lastGeneration;

  Future<void> run({
    required int generation,
    required NeriApiClient apiClient,
    required void Function(String message) onMessage,
  }) async {
    if (generation <= 0 || _lastGeneration == generation) return;
    _lastGeneration = generation;
    try {
      final status = await apiClient.fetchDinoV3ComponentStatus();
      if (_lastGeneration != generation) return;
      final message = dinoV3StartupStatusMessage(status);
      if (message != null && message.isNotEmpty) onMessage(message);
    } catch (_) {
      // The backend itself is already usable at this point. A transient status
      // endpoint failure must not block startup or generate a misleading alert.
    }
  }
}
