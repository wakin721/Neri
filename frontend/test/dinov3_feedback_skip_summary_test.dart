import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  testWidgets('collapsed multi-box batch reports skipped DINO learning', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final tempDir = await Directory.systemTemp.createTemp('neri_dinov3_skip_');
    addTearDown(() async {
      if (await tempDir.exists()) await tempDir.delete(recursive: true);
    });
    final firstPath = '${tempDir.path}/a.jpg';
    final secondPath = '${tempDir.path}/b.jpg';
    final imageBytes = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
    );
    await File(firstPath).writeAsBytes(imageBytes);
    await File(secondPath).writeAsBytes(imageBytes);

    final client = NeriApiClient(
      httpClient: MockClient(
        (_) async => http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        ),
      ),
    );
    addTearDown(client.close);

    final first = DetectionItem(
      filename: 'a.jpg',
      path: firstPath,
      fileType: 'jpg',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: const <double>[0.1, 0.1, 0.4, 0.4],
          observationId: 'obs-a-1',
        ),
        DetectionBox(
          species: '盘羊',
          bbox: const <double>[0.5, 0.5, 0.9, 0.9],
          observationId: 'obs-a-2',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );
    final second = DetectionItem(
      filename: 'b.jpg',
      path: secondPath,
      fileType: 'jpg',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: const <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-b-1',
        ),
      ],
      detectionData: const <String, dynamic>{'物种名称': '盘羊'},
    );

    var batchCalls = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SpeciesValidationScreen(
            apiClient: client,
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
            onMarkItem: (
              item,
              action, {
              speciesName,
              speciesCount,
              speciesType,
              remark,
            }) async => item,
            onMarkItems: (
              items,
              action, {
              speciesName,
              speciesCount,
              speciesType,
              remark,
              feedbackOperationId,
            }) async {
              batchCalls += 1;
              expect(feedbackOperationId, isNotNull);
              expect(feedbackOperationId, isNotEmpty);
              return items;
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
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.textContaining('2 个文件'), findsWidgets);
    await tester.tap(find.text('正确'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    expect(batchCalls, 1);
    expect(
      find.textContaining('DINOv3 学习已跳过 1/2 个文件'),
      findsOneWidget,
    );

    await tester.pump(const Duration(seconds: 5));
    await tester.pump();
  });
}
