import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/detection_box_label.dart';
import 'package:neri_flutter/src/utils/dino_rejection.dart';
import 'package:neri_flutter/src/widgets/dino_rejection_scores.dart';

void main() {
  const person = DetectionBox(
    species: '人员', confidence: .84, bbox: [1, 2, 3, 4], observationId: 'dinov3:identity',
    candidates: [{
      'name': '人员', 'conf': .84, 'raw_det_conf': .84, 'accepted': true,
      'source': 'detector_category', 'classification_source': 'detector_category',
      'confidence_kind': 'detector_confidence', 'gate_mode': 'detector',
    }],
  );
  test('detector person category is labelled as detector evidence', () {
    expect(detectionBoxLabel(person, 0), '#1 人员 探测0.84');
    expect(dinoRejectionInfo(person).finalLabel, '探测保留 · 人员');
  });
  testWidgets('retained category does not display fabricated DINO gate comparisons', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: Scaffold(body: DinoRejectionScores(boxes: [person]))));
    expect(find.textContaining('探测保留 · 人员'), findsOneWidget);
    expect(find.text('探测证据 0.840'), findsOneWidget);
    expect(find.textContaining('单框 —'), findsNothing);
    expect(find.textContaining('序列 —'), findsNothing);
  });
}
