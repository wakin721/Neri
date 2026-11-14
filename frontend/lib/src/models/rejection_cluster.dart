import 'job.dart';

class RejectionCluster {
  const RejectionCluster({
    required this.id,
    required this.number,
    required this.modelFingerprint,
    required this.eventCount,
    required this.minClusterEvents,
    required this.densityVerified,
    this.isNoiseFlag = false,
    this.status = '',
    this.runId = '',
  });

  factory RejectionCluster.fromJson(Map<String, dynamic> json) {
    final status = json['status']?.toString().trim() ?? '';
    final isNoiseFlag = json['is_noise'] == true;
    final densityVerified = json['density_verified'] == true && !isNoiseFlag;
    return RejectionCluster(
      id: json['cluster_id']?.toString() ?? '',
      number: (json['candidate_number'] as num?)?.toInt() ?? 0,
      modelFingerprint: json['model_fingerprint']?.toString() ?? '',
      eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
      minClusterEvents: (json['min_cluster_events'] as num?)?.toInt() ?? 5,
      densityVerified: densityVerified,
      isNoiseFlag: isNoiseFlag,
      status: status,
      runId: json['run_id']?.toString() ?? '',
    );
  }

  final String id;
  final int number;
  final String modelFingerprint;
  final int eventCount;
  final int minClusterEvents;
  final bool densityVerified;
  final bool isNoiseFlag;
  final String status;
  final String runId;

  bool get isFailed => status == 'failed';
  bool get isClustered => !isFailed && !isNoise && number > 0 && id.isNotEmpty;
  bool get isNoise {
    if (isFailed) return false;
    if (status == 'noise') return true;
    if (status == 'clustered') {
      return isNoiseFlag || !densityVerified || number <= 0;
    }
    return isNoiseFlag || eventCount < minClusterEvents || !densityVerified;
  }
}

class ClusterLearningResult {
  ClusterLearningResult.fromJson(Map<String, dynamic> json)
    : operationId = json['operation_id'] as String,
      featureEvents = (json['feature_events'] as num).toInt(),
      learning = json['learning'] as Map<String, dynamic>,
      items = (json['items'] as List)
          .map((item) => DetectionItem.fromJson(item as Map<String, dynamic>))
          .toList();

  final String operationId;
  final int featureEvents;
  final Map<String, dynamic> learning;
  final List<DetectionItem> items;
}
