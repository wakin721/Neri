from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
anchor = "\n}\n"
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('ignore DINO box feedback posts unverified action', (tester) async {
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
                'species': <String>['盘羊'],
              },
              'operation_id': feedbackBody!['feedback_operation_id'],
              'affected_species': <String>[],
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
      observationId: 'obs-ignore',
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

    final ignoreFinder = find.widgetWithText(OutlinedButton, '不参与学习');
    expect(ignoreFinder, findsOneWidget);
    final ignoreButton = tester.widget<OutlinedButton>(ignoreFinder);
    expect(ignoreButton.onPressed, isNotNull);
    await tester.tap(ignoreFinder);
    await tester.pumpAndSettle();

    expect(feedbackBody, isNotNull);
    expect(feedbackBody!['observation_id'], 'obs-ignore');
    expect(feedbackBody!['action'], 'unverified');
    expect(feedbackBody!['classification_model_path'], 'model.neri.json');
    expect(feedbackBody!['feedback_operation_id']?.toString(), isNotEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
