import 'job.dart';

class DinoBoxFeedbackResult {
  const DinoBoxFeedbackResult({
    required this.item,
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoBoxFeedbackResult.fromJson(Map<String, dynamic> json) {
    return DinoBoxFeedbackResult(
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

class DinoFeedbackRevertResult {
  const DinoFeedbackRevertResult({
    required this.operationId,
    required this.affectedSpecies,
  });

  factory DinoFeedbackRevertResult.fromJson(Map<String, dynamic> json) {
    return DinoFeedbackRevertResult(
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
