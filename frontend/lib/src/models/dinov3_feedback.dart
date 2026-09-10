import 'job.dart';

class DinoV3BoxFeedbackResult {
  const DinoV3BoxFeedbackResult({
    required this.item,
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoV3BoxFeedbackResult.fromJson(Map<String, dynamic> json) {
    return DinoV3BoxFeedbackResult(
      item: DetectionItem.fromJson(json['item'] as Map<String, dynamic>),
      operationId: json['operation_id'] as String? ?? '',
      affectedSpecies:
          (json['affected_species'] as List<dynamic>? ?? const <dynamic>[])
              .map((value) => value.toString())
              .toList(),
    );
  }

  final DetectionItem item;
  final String operationId;
  final List<String> affectedSpecies;
}

class DinoV3FeedbackRevertResult {
  const DinoV3FeedbackRevertResult({
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoV3FeedbackRevertResult.fromJson(Map<String, dynamic> json) {
    return DinoV3FeedbackRevertResult(
      operationId: json['operation_id'] as String? ?? '',
      affectedSpecies:
          (json['affected_species'] as List<dynamic>? ?? const <dynamic>[])
              .map((value) => value.toString())
              .toList(),
    );
  }

  final String operationId;
  final List<String> affectedSpecies;
}
