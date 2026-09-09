import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

void main() {
  late Directory tempDir;
  late File imageFile;
  late NeriApiClient apiClient;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('neri_dinov3_feedback_ui_');
    imageFile = File('${tempDir.path}/frame.png');
    await imageFile.writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
    apiClient = NeriApiClient(
      httpClient: MockClient(
        (_) async => http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        ),
      ),
    );
  });

  tearDown(() async {
    apiClient.close();
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  testWidgets('selected DINO box shows correct/update/empty/ignore controls', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

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
      ),
    );
    await tester.pumpAndSettle();

    final viewer = find.byType(DetectionMediaViewer);
    expect(viewer, findsOneWidget);
    await tester.tapAt(tester.getCenter(viewer));
    await tester.pumpAndSettle();

    expect(find.text('检测框校验'), findsOneWidget);
    expect(find.text('正确'), findsWidgets);
    expect(find.text('修改物种'), findsOneWidget);
    expect(find.text('空 / 误检'), findsOneWidget);
    expect(find.text('不参与学习'), findsOneWidget);
  });
}
