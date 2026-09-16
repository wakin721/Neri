import 'job.dart';

class DinoV2BoxFeedbackResult {
  const DinoV2BoxFeedbackResult({
    required this.item,
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoV2BoxFeedbackResult.fromJson(Map<String, dynamic> json) {
    return DinoV2BoxFeedbackResult(
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

class DinoV2FeedbackRevertResult {
  const DinoV2FeedbackRevertResult({
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoV2FeedbackRevertResult.fromJson(Map<String, dynamic> json) {
    return DinoV2FeedbackRevertResult(
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
