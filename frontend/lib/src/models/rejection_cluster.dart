import 'job.dart';

class RejectionCluster {
  const RejectionCluster({
    required this.id,
    required this.number,
    required this.modelFingerprint,
    required this.eventCount,
    required this.minClusterEvents,
  });

  factory RejectionCluster.fromJson(Map<String, dynamic> json) =>
      RejectionCluster(
        id: json['cluster_id'] as String,
        number: (json['candidate_number'] as num).toInt(),
        modelFingerprint: json['model_fingerprint'] as String,
        eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
        minClusterEvents: (json['min_cluster_events'] as num?)?.toInt() ?? 4,
      );

  final String id;
  final int number;
  final String modelFingerprint;
  final int eventCount;
  final int minClusterEvents;
  bool get isNoise => eventCount < minClusterEvents;
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
