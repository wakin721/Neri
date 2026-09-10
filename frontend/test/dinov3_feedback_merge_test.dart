import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  test('validation effective items prefer server-authoritative feedback item', () {
    const path = 'C:/camera/frame.png';
    final current = DetectionItem(
      filename: 'frame.png',
      path: path,
      fileType: 'png',
      species: const <String>['盘羊'],
      detectionBoxes: const <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-merge',
          predictedSpecies: '盘羊',
          feedbackStatus: 'unverified',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );
    final authoritative = DetectionItem(
      filename: 'frame.png',
      path: path,
      fileType: 'png',
      species: const <String>['家牛'],
      detectionBoxes: const <DetectionBox>[
        DetectionBox(
          species: '家牛',
          bbox: <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-merge',
          predictedSpecies: '盘羊',
          feedbackStatus: 'corrected',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '家牛'},
    );

    final effective = dinoValidationEffectiveItems(
      <DetectionItem>[current],
      <String, DetectionItem>{path: authoritative},
    );

    expect(effective, hasLength(1));
    expect(effective.single.species, <String>['家牛']);
    expect(effective.single.detectionBoxes.single.species, '家牛');
    expect(effective.single.detectionBoxes.single.feedbackStatus, 'corrected');
  });
}
