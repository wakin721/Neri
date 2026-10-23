import 'job.dart';

class RejectionCluster {
  const RejectionCluster({
    required this.id,
    required this.number,
    required this.modelFingerprint,
  });

  factory RejectionCluster.fromJson(Map<String, dynamic> json) =>
      RejectionCluster(
        id: json['cluster_id'] as String,
        number: (json['candidate_number'] as num).toInt(),
        modelFingerprint: json['model_fingerprint'] as String,
      );

  final String id;
  final int number;
  final String modelFingerprint;
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
