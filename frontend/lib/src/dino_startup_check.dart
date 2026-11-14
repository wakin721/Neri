import 'api_client.dart';

String? dinoStartupStatusMessage(DinoComponentStatus status) {
  if (!status.installed) {
    return 'Neri1.1 尚未安装，可进入设置安装。';
  }
  if (status.healthy) return null;

  final reason = status.message.trim();
  final lower = reason.toLowerCase();
  final mentionsRuntime =
      lower.contains('pytorch') ||
      lower.contains('torch') ||
      lower.contains('numpy') ||
      lower.contains('opencv') ||
      lower.contains('pillow');
  final indicatesMissing =
      reason.contains('未安装') ||
      lower.contains('not installed') ||
      lower.contains('no module named') ||
      lower.contains('missing') ||
      lower.contains('unavailable');
  if (mentionsRuntime && indicatesMissing) {
    return 'Neri1.1 已安装，但推理依赖不完整，当前不可推理';
  }
  if (reason.isEmpty) return 'Neri1.1 安装异常';
  return 'Neri1.1 安装异常：$reason';
}

class DinoStartupCheck {
  int? _lastGeneration;

  Future<void> run({
    required int generation,
    required NeriApiClient apiClient,
    required void Function(String message) onMessage,
  }) async {
    if (generation <= 0 || _lastGeneration == generation) return;
    _lastGeneration = generation;
    try {
      final status = await apiClient.fetchDinoComponentStatus();
      if (_lastGeneration != generation) return;
      final message = dinoStartupStatusMessage(status);
      if (message != null && message.isNotEmpty) onMessage(message);
    } catch (_) {
      // A transient optional-component status failure must not block startup.
    }
  }
}
