import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/widgets/sequence_pending_notice.dart';

DetectionItem interruptedPhoto({
  bool rejected = false,
  bool completed = false,
  bool? validated,
  bool manual = false,
}) => DetectionItem.fromJson({
  'filename': 'ECSP0070.JPG',
  'path': 'camera/ECSP0070.JPG',
  'file_type': 'jpg',
  'validated': validated,
  'detection_data': {
    if (completed) 'sequence_completed': true,
    if (manual) '最低置信度': '人工校验',
    '检测框': [
      if (rejected)
        {
          '物种': '拒识/Unknown',
          '置信度': null,
          '候选项': <dynamic>[],
          'source': 'dinov3_ridge_within',
        }
      else
        {
          '物种': '家马',
          '候选项': [
            {
              'name': '家马',
              'sequence_pending': true,
              'source': 'dinov3_ridge_within',
            },
          ],
        },
    ],
  },
});

void main() {
  test(
    'both accepted and rejected legacy interrupted photos need aggregation',
    () {
      expect(interruptedPhoto().isSequencePending, isTrue);
      expect(interruptedPhoto(rejected: true).isSequencePending, isTrue);
      expect(
        const DetectionItem(
          filename: 'frame.jpg',
          path: 'camera/frame.jpg',
          fileType: 'jpg',
          detectionBoxes: [
            DetectionBox(
              species: '家马',
              bbox: [0, 0, 1, 1],
              candidates: [
                {'name': '家马', 'sequence_pending': true},
              ],
            ),
          ],
        ).isSequencePending,
        isTrue,
      );
    },
  );

  test(
    'completed aggregation and human confirmation supersede old markers',
    () {
      expect(interruptedPhoto(completed: true).isSequencePending, isFalse);
      expect(
        interruptedPhoto(rejected: true, completed: true).isSequencePending,
        isFalse,
      );
      expect(interruptedPhoto(validated: true).isSequencePending, isFalse);
      expect(interruptedPhoto(manual: true).isSequencePending, isFalse);
      expect(
        const DetectionItem(
          filename: 'frame.jpg',
          path: 'camera/frame.jpg',
          fileType: 'jpg',
          species: ['家马'],
        ).isSequencePending,
        isFalse,
      );
    },
  );

  testWidgets('notice disappears when refreshed results finish aggregation', (
    tester,
  ) async {
    Future<void> show(DetectionItem item) => tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: SequencePendingNotice(item: item)),
      ),
    );
    await show(interruptedPhoto(rejected: true));
    expect(find.byTooltip(SequencePendingNotice.message), findsOneWidget);
    expect(find.textContaining('临时结果 · 序列待汇总'), findsNothing);
    await tester.longPress(find.byIcon(Icons.info_outline_rounded));
    await tester.pump(const Duration(milliseconds: 200));
    expect(find.text(SequencePendingNotice.message), findsOneWidget);

    final refreshed = interruptedPhoto(
      rejected: true,
    ).mergeValidationUpdate(interruptedPhoto(completed: true));
    await show(refreshed);
    await tester.pumpAndSettle();
    expect(find.byTooltip(SequencePendingNotice.message), findsNothing);
    expect(find.textContaining('序列待汇总'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
