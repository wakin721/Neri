from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')

tear_down = """  tearDown(() async {\n    apiClient.close();\n    if (await tempDir.exists()) {\n      await tempDir.delete(recursive: true);\n    }\n  });\n\n"""
helper = r'''  Widget buildFeedbackScreen({
    required NeriApiClient client,
    required DetectionItem item,
    String? classificationModelPath,
  }) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: client,
          inputPath: tempDir.path,
          classificationModelPath: classificationModelPath,
          items: <DetectionItem>[item],
          loading: false,
          refreshVersion: 0,
          speciesTypes: const <String, String>{'盘羊': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: false,
          collapseGroups: false,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: 3,
          autoGroupGapSeconds: 1800,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: const <String>['盘羊'],
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
          favoritePhotoExportMode: favoritePhotoExportAsk,
          emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
          onRefresh: () async {},
          onLoadMetadata: (_) async {},
          onOpenExternal: (_) {},
          onMarkItem:
              (
                candidate,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => candidate,
          onMarkItems:
              (
                candidates,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => candidates,
          onQuickMarkUsed: (_) async {},
          onQuickMarkReverted: (_) async {},
          onRedetectItems: (_, {required confidence}) async {},
          onFavoritePhotoPathsChanged: (_) async {},
          onFavoritePhotoExportModeChanged: (_) async {},
          onEmptyPhotoDeleteModeChanged: (_) async {},
          onAutoGroupInferredBurstSizeChanged: (_) {},
        ),
      ),
    );
  }

'''
if 'Widget buildFeedbackScreen({' not in text:
    if tear_down not in text:
        raise SystemExit('tearDown anchor not found')
    text = text.replace(tear_down, tear_down + helper, 1)

anchor = "\n}\n"
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('empty DINO box feedback posts empty action', (tester) async {
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
                'species': <String>['空'],
              },
              'operation_id': feedbackBody!['feedback_operation_id'],
              'affected_species': <String>['盘羊'],
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
      observationId: 'obs-empty',
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

    final emptyFinder = find.widgetWithText(OutlinedButton, '空 / 误检');
    expect(emptyFinder, findsOneWidget);
    final emptyButton = tester.widget<OutlinedButton>(emptyFinder);
    expect(emptyButton.onPressed, isNotNull);
    await tester.tap(emptyFinder);
    await tester.pumpAndSettle();

    expect(feedbackBody, isNotNull);
    expect(feedbackBody!['observation_id'], 'obs-empty');
    expect(feedbackBody!['action'], 'empty');
    expect(feedbackBody!['classification_model_path'], 'model.neri.json');
    expect(feedbackBody!['feedback_operation_id']?.toString(), isNotEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
