from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
anchor = "\n}\n"
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('update DINO box feedback posts explicit corrected species', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    Map<String, dynamic>? feedbackBody;
    apiClient.close();
    apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dinov3/feedback/box') {
          feedbackBody = jsonDecode(request.body) as Map<String, dynamic>;
          return http.Response(
            jsonEncode(<String, dynamic>{
              'item': <String, dynamic>{
                'filename': 'frame.png',
                'path': imageFile.path,
                'file_type': 'png',
                'width': 10,
                'height': 10,
                'species': <String>['家牛'],
              },
              'operation_id': feedbackBody!['feedback_operation_id'],
              'affected_species': <String>['盘羊', '家牛'],
            }),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );

    final box = DetectionBox(
      species: '盘羊',
      bbox: const <double>[0.1, 0.1, 0.9, 0.9],
      observationId: 'obs-update',
      predictedSpecies: '盘羊',
      feedbackStatus: 'unverified',
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

    final updateFinder = find.widgetWithText(OutlinedButton, '修改物种');
    expect(updateFinder, findsOneWidget);
    final updateButton = tester.widget<OutlinedButton>(updateFinder);
    expect(updateButton.onPressed, isNotNull);
    await tester.tap(updateFinder);
    await tester.pumpAndSettle();

    expect(find.text('其他标注'), findsOneWidget);
    final speciesField = find.byWidgetPredicate(
      (widget) =>
          widget is TextField &&
          widget.decoration?.labelText == '物种名称 (拼音首字母，逗号分隔)',
    );
    expect(speciesField, findsOneWidget);
    await tester.enterText(speciesField, '家牛');
    await tester.pump();
    await tester.tap(find.widgetWithText(FilledButton, '确定'));
    await tester.pumpAndSettle();

    expect(feedbackBody, isNotNull);
    expect(feedbackBody!['observation_id'], 'obs-update');
    expect(feedbackBody!['action'], 'update');
    expect(feedbackBody!['species_name'], '家牛');
    expect(feedbackBody!['classification_model_path'], 'model.neri.json');
    expect(feedbackBody!['feedback_operation_id']?.toString(), isNotEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
