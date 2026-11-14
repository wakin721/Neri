import '../models/job.dart';

String detectionBoxLabel(DetectionBox box, int index) {
  final track = box.trackId?.trim() ?? '';
  final number = RegExp(r'^\d{1,6}$').hasMatch(track) ? track : '${index + 1}';
  final confidence = box.confidence?.toStringAsFixed(2) ?? 'N/A';
  return '#$number ${box.species} $confidence';
}
