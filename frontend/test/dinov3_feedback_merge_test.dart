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
  testWidgets('box feedback merges the server-authoritative DetectionItem', (
    tester,
  ) async {
    // ignore: avoid_print
    print('merge-test: start');
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final tempDir = await Directory.systemTemp.createTemp('neri_dinov3_merge_');
    final imageFile = File('${tempDir.path}/frame.png');
    await imageFile.writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
    addTearDown(() async {
      if (await tempDir.exists()) await tempDir.delete(recursive: true);
    });
    // ignore: avoid_print
    print('merge-test: fixture-ready');

    late Map<String, dynamic> feedbackBody;
    final client = NeriApiClient(
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
                'detection_boxes': <Map<String, dynamic>>[
                  <String, dynamic>{
                    'species': '家牛',
                    'bbox': <double>[0.1, 0.1, 0.9, 0.9],
                    'observation_id': 'obs-merge',
                    'predicted_species': '盘羊',
                    'feedback_status': 'corrected',
                  },
                ],
                'detection_data': <String, dynamic>{'物种名称': '家牛'},
              },
              'operation_id': feedbackBody['feedback_operation_id'],
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
    addTearDown(client.close);

    final item = DetectionItem(
      filename: 'frame.png',
      path: imageFile.path,
      fileType: 'png',
      width: 10,
      height: 10,
      species: const <String>['盘羊'],
      detectionBoxes: const <DetectionBox>[
        DetectionBox(
          species: '盘羊',
          bbox: <double>[0.1, 0.1, 0.9, 0.9],
          observationId: 'obs-merge',
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
            apiClient: client,
            inputPath: tempDir.path,
            classificationModelPath: 'model.neri.json',
            items: <DetectionItem>[item],
            loading: false,
            refreshVersion: 0,
            speciesTypes: const <String, String>{
              '盘羊': '兽类',
              '家牛': '兽类',
            },
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
            onMarkItem: (
              candidate,
              action, {
              speciesName,
              speciesCount,
              speciesType,
              remark,
            }) async => candidate,
            onMarkItems: (
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
    // ignore: avoid_print
    print('merge-test: pump-widget-done');
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));
    // ignore: avoid_print
    print('merge-test: initial-pumps-done');

    final viewerFinder = find.byType(DetectionMediaViewer);
    expect(viewerFinder, findsOneWidget);
    await tester.tapAt(tester.getCenter(viewerFinder));
    // ignore: avoid_print
    print('merge-test: viewer-tapped');
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));
    // ignore: avoid_print
    print('merge-test: selection-pumps-done');
    await tester.tap(find.widgetWithText(OutlinedButton, '正确'));
    // ignore: avoid_print
    print('merge-test: correct-tapped');
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));
    // ignore: avoid_print
    print('merge-test: feedback-pumps-done');

    final viewer = tester.widget<DetectionMediaViewer>(viewerFinder);
    expect(viewer.visibleBoxes, hasLength(1));
    expect(viewer.visibleBoxes.single.species, '家牛');
    expect(viewer.visibleBoxes.single.feedbackStatus, 'corrected');
    // ignore: avoid_print
    print('merge-test: assertions-done');

    await tester.pump(const Duration(seconds: 5));
    await tester.pump();
    // ignore: avoid_print
    print('merge-test: end');
  });
}
