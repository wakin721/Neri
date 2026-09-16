import 'api_client.dart';

String? dinoV2StartupStatusMessage(DinoV2ComponentStatus status) {
  if (!status.installed) {
    return 'DINOv2 尚未安装，可进入设置安装。';
  }
  if (status.healthy) return null;

  final reason = status.message.trim();
  final lower = reason.toLowerCase();
  final mentionsRuntime =
      lower.contains('pytorch') ||
      lower.contains('torch') ||
      lower.contains('transformers') ||
      lower.contains('safetensors');
  final indicatesMissing =
      reason.contains('未安装') ||
      lower.contains('not installed') ||
      lower.contains('no module named') ||
      lower.contains('missing') ||
      lower.contains('unavailable');
  if (mentionsRuntime && indicatesMissing) {
    return 'DINOv2 已安装，但推理依赖不完整，当前不可推理';
  }
  if (reason.isEmpty) return 'DINOv2 安装异常';
  return 'DINOv2 安装异常：$reason';
}

class DinoV2StartupCheck {
  int? _lastGeneration;

  Future<void> run({
    required int generation,
    required NeriApiClient apiClient,
    required void Function(String message) onMessage,
  }) async {
    if (generation <= 0 || _lastGeneration == generation) return;
    _lastGeneration = generation;
    try {
      final status = await apiClient.fetchDinoV2ComponentStatus();
      if (_lastGeneration != generation) return;
      final message = dinoV2StartupStatusMessage(status);
      if (message != null && message.isNotEmpty) onMessage(message);
    } catch (_) {
      // A transient optional-component status failure must not block startup.
    }
  }
}
