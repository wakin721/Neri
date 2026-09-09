from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
anchor = "\n}\n"
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('undo reverts DINO feedback with the stored operation id', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    late Map<String, dynamic> feedbackBody;
    Map<String, dynamic>? revertBody;
    final calls = <String>[];
    apiClient.close();
    apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dinov3/feedback/box') {
          calls.add('feedback');
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
              'operation_id': feedbackBody['feedback_operation_id'],
              'affected_species': <String>['盘羊'],
            }),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.url.path == '/api/dinov3/feedback/revert') {
          calls.add('revert');
          revertBody = jsonDecode(request.body) as Map<String, dynamic>;
          return http.Response(
            jsonEncode(<String, dynamic>{
              'operation_id': revertBody!['feedback_operation_id'],
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
      observationId: 'obs-1',
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
      MaterialApp(
        home: Scaffold(
          body: SpeciesValidationScreen(
            apiClient: apiClient,
            inputPath: tempDir.path,
            classificationModelPath: 'model.neri.json',
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
                }) async {
                  if (action == 'unverified') calls.add('ecology');
                  return candidates;
                },
            onQuickMarkUsed: (_) async {},
            onQuickMarkReverted: (_) async {},
            onRedetectItems: (_, {required confidence}) async {},
            onFavoritePhotoPathsChanged: (_) async {},
            onFavoritePhotoExportModeChanged: (_) async {},
            onEmptyPhotoDeleteModeChanged: (_) async {},
            onAutoGroupInferredBurstSizeChanged: (_) {},
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    final viewer = find.byType(DetectionMediaViewer);
    await tester.tapAt(tester.getCenter(viewer));
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(OutlinedButton, '正确'));
    await tester.pumpAndSettle();

    final operationId = feedbackBody['feedback_operation_id']?.toString() ?? '';
    expect(operationId, isNotEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();

    final undoFinder = find.widgetWithIcon(IconButton, Icons.undo_rounded);
    expect(undoFinder, findsOneWidget);
    final undoButton = tester.widget<IconButton>(undoFinder);
    expect(undoButton.onPressed, isNotNull);
    await tester.tap(undoFinder);
    await tester.pumpAndSettle();

    expect(calls, <String>['feedback', 'ecology', 'revert']);
    expect(revertBody, isNotNull);
    expect(revertBody!['classification_model_path'], 'model.neri.json');
    expect(revertBody!['feedback_operation_id'], operationId);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
