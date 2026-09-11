import 'models/job.dart';

class DinoValidationBoxSelection {
  const DinoValidationBoxSelection({
    required this.filePath,
    required this.observationId,
    required this.learnableObservationCount,
  });

  final String filePath;
  final String observationId;
  final int learnableObservationCount;
}

DinoValidationBoxSelection? _currentDinoValidationBoxSelection;

String _dinoValidationPathKey(String value) =>
    value.trim().replaceAll('\\', '/').toLowerCase();

Set<String> _dinoValidationObservationIds(DetectionItem item) => item.detectionBoxes
    .map((box) => box.observationId?.trim() ?? '')
    .where((id) => id.isNotEmpty)
    .toSet();

String? dinoValidationSelectedFeedbackObservationId(
  DetectionItem item, {
  required String? selectedObservationId,
  required bool learningRequested,
}) {
  if (!learningRequested) return null;
  final observationIds = _dinoValidationObservationIds(item);
  if (observationIds.length <= 1) return null;
  final selected = selectedObservationId?.trim() ?? '';
  return selected.isNotEmpty && observationIds.contains(selected)
      ? selected
      : null;
}

void recordDinoValidationBoxSelection(
  DetectionItem item,
  DetectionBox? selectedBox,
) {
  final selected = selectedBox?.observationId?.trim() ?? '';
  if (selected.isEmpty) {
    if (_currentDinoValidationBoxSelection != null &&
        _dinoValidationPathKey(_currentDinoValidationBoxSelection!.filePath) ==
            _dinoValidationPathKey(item.path)) {
      _currentDinoValidationBoxSelection = null;
    }
    return;
  }
  final observationIds = _dinoValidationObservationIds(item);
  if (!observationIds.contains(selected)) return;
  _currentDinoValidationBoxSelection = DinoValidationBoxSelection(
    filePath: item.path,
    observationId: selected,
    learnableObservationCount: observationIds.length,
  );
}

DinoValidationBoxSelection? dinoValidationBoxSelectionFor(String filePath) {
  final selection = _currentDinoValidationBoxSelection;
  if (selection == null ||
      _dinoValidationPathKey(selection.filePath) !=
          _dinoValidationPathKey(filePath)) {
    return null;
  }
  return selection;
}
