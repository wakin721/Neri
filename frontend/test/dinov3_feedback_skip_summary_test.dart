import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  test('multi-box batch reports skipped DINO learning ratio', () {
    final items = <DetectionItem>[
      DetectionItem(
        filename: 'a.jpg',
        path: 'C:/camera/a.jpg',
        fileType: 'jpg',
        detectionBoxes: const <DetectionBox>[
          DetectionBox(
            species: '盘羊',
            bbox: <double>[0.1, 0.1, 0.4, 0.4],
            observationId: 'obs-a-1',
          ),
          DetectionBox(
            species: '盘羊',
            bbox: <double>[0.5, 0.5, 0.9, 0.9],
            observationId: 'obs-a-2',
          ),
        ],
      ),
      DetectionItem(
        filename: 'b.jpg',
        path: 'C:/camera/b.jpg',
        fileType: 'jpg',
        detectionBoxes: const <DetectionBox>[
          DetectionBox(
            species: '盘羊',
            bbox: <double>[0.1, 0.1, 0.9, 0.9],
            observationId: 'obs-b-1',
          ),
        ],
      ),
    ];

    expect(
      dinoValidationLearningSkipSummary(items, learningRequested: true),
      'DINOv3 学习已跳过 1/2 个文件：每个文件需恰好 1 个可学习检测框',
    );
  });
}
