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
