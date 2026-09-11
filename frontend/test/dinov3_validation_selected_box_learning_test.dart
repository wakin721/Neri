import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  DetectionItem itemWithBoxes(List<DetectionBox> boxes) {
    return DetectionItem(
      filename: 'camera.jpg',
      path: 'C:/camera/camera.jpg',
      fileType: 'jpg',
      detectionBoxes: boxes,
    );
  }

  test('selected box is used when file-level DINO learning would be ambiguous', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.4, 0.4],
        observationId: 'obs-1',
      ),
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.5, 0.5, 0.9, 0.9],
        observationId: 'obs-2',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: 'obs-2',
        learningRequested: true,
      ),
      'obs-2',
    );
  });

  test('multi-box file without an explicit selected box remains skipped', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.4, 0.4],
        observationId: 'obs-1',
      ),
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.5, 0.5, 0.9, 0.9],
        observationId: 'obs-2',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: null,
        learningRequested: true,
      ),
      isNull,
    );
  });

  test('single-box files stay on the existing automatic feedback path', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.9, 0.9],
        observationId: 'obs-1',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: 'obs-1',
        learningRequested: true,
      ),
      isNull,
    );
  });
}
