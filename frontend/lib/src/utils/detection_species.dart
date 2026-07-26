import '../models/job.dart';

List<String> detectionSpeciesOptions(
  DetectionItem item, {
  required String globalOption,
}) {
  final seen = <String>{};
  final options = <String>[];

  void add(Object? rawName) {
    final name = rawName?.toString().trim() ?? '';
    if (name.isEmpty || name == 'Unknown' || name == '空' || !seen.add(name)) {
      return;
    }
    options.add(name);
  }

  add(globalOption);
  for (final species in item.species) {
    add(species);
  }
  for (final box in item.detectionBoxes) {
    add(box.species);
    for (final candidate in box.candidates) {
      add(candidate['name']);
    }
  }

  final classificationCandidates = item.detectionData['分类候选项'];
  if (classificationCandidates is List) {
    for (final candidate in classificationCandidates) {
      if (candidate is Map) add(candidate['name']);
    }
  }

  return options;
}

double candidateConfidence(Map<dynamic, dynamic> candidate) {
  final rawConfidence = candidate['conf'] ?? candidate['confidence'];
  if (rawConfidence is num) return rawConfidence.toDouble();
  return double.tryParse(rawConfidence?.toString() ?? '') ?? 0;
}
