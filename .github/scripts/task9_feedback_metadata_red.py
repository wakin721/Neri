from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
anchor = "\n}\n"
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('confidence-filtered DINO box preserves feedback metadata', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final box = DetectionBox(
      species: '盘羊',
      bbox: const <double>[0.1, 0.1, 0.9, 0.9],
      confidence: 0.9,
      observationId: 'obs-filtered',
      registryId: 7,
      predictedSpecies: '盘羊',
      feedbackStatus: 'unverified',
      candidates: const <Map<String, dynamic>>[
        <String, dynamic>{'name': '盘羊', 'confidence': 0.9},
      ],
    );
    final item = DetectionItem(
      filename: 'frame.png',
      path: imageFile.path,
      fileType: 'png',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: <DetectionBox>[box],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );

    await tester.pumpWidget(
      buildFeedbackScreen(
        client: apiClient,
        item: item,
        classificationModelPath: 'model.neri.json',
      ),
    );
    await tester.pumpAndSettle();

    final viewer = find.byType(DetectionMediaViewer);
    await tester.tapAt(tester.getCenter(viewer));
    await tester.pumpAndSettle();

    expect(find.text('检测框校验'), findsOneWidget);
    final correctFinder = find.widgetWithText(OutlinedButton, '正确');
    expect(correctFinder, findsOneWidget);
    expect(tester.widget<OutlinedButton>(correctFinder).onPressed, isNotNull);
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
