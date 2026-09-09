from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
anchor = '\n}\n'
if not text.endswith(anchor):
    raise SystemExit('test file closing anchor not found')

test = r'''

  testWidgets('collapsed DINO batch forwards one generated feedback operation id', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    String? operationId;
    var batchCount = 0;
    final secondPath = '${tempDir.path}/frame-2.png';
    final first = DetectionItem(
      filename: 'frame.png',
      path: imageFile.path,
      fileType: 'png',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: const <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-batch-1',
          predictedSpecies: '盘羊',
          feedbackStatus: 'unverified',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );
    final second = DetectionItem(
      filename: 'frame-2.png',
      path: secondPath,
      fileType: 'png',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: const <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-batch-2',
          predictedSpecies: '盘羊',
          feedbackStatus: 'unverified',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SpeciesValidationScreen(
            apiClient: apiClient,
            inputPath: tempDir.path,
            classificationModelPath: 'model.neri.json',
            items: <DetectionItem>[first, second],
            loading: false,
            refreshVersion: 0,
            speciesTypes: const <String, String>{'盘羊': '兽类'},
            useCombinedConfidence: false,
            minFrameRatio: 0,
            autoGroup: true,
            collapseGroups: true,
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
                }) async {
                  throw StateError('batch path must not call onMarkItem');
                },
            onMarkItems:
                (
                  candidates,
                  action, {
                  speciesName,
                  speciesCount,
                  speciesType,
                  remark,
                  feedbackOperationId,
                }) async {
                  batchCount = candidates.length;
                  operationId = feedbackOperationId;
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

    expect(find.textContaining('2 个文件'), findsWidgets);
    await tester.tap(find.text('正确'));
    await tester.pumpAndSettle();

    expect(batchCount, 2);
    expect(operationId, isNotNull);
    expect(operationId, isNotEmpty);
    await tester.pump(const Duration(seconds: 4));
    await tester.pumpAndSettle();
  });
'''

text = text[:-len(anchor)] + test + anchor
path.write_text(text, encoding='utf-8')
