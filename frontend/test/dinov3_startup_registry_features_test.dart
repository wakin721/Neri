import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

void main() {
  test('NeriApp wires a DINOv3 startup status check', () {
    final source = File('lib/main.dart').readAsStringSync();
    expect(source, contains('DinoV3StartupCheck('));
  });

  testWidgets('registry dialog shows cropped examples and can delete a species', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var deleted = false;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response(
            '[{"id":9,"event_key":"e1","source_path":"/a.jpg","camera_id":"cam-a","timestamp_missing":false,"sample_count":1,"has_example":true}]',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/example')) {
          return http.Response.bytes(
            base64Decode(
              'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
            ),
            200,
            headers: const {'content-type': 'image/png'},
          );
        }
        if (request.method == 'DELETE' && request.url.path == '/api/dinov3/registry/1') {
          deleted = true;
          return http.Response('{"deleted":true,"registration_id":1}', 200);
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    const entry = DinoV3RegistryEntry(
      id: 1,
      candidateNumber: 1,
      status: 'candidate',
      displayName: '未知物种 #1',
      commonName: '',
      scientificName: '',
      eventCount: 5,
      cameraCount: 2,
      prototypeCount: 1,
      clusterPurity: 1,
      embeddingConsistency: 1,
      conditions: <String, bool>{},
      canRegister: false,
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: const <DinoV3RegistryEntry>[entry],
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('裁切例图'), findsOneWidget);
    expect(find.byKey(const ValueKey('dinov3-registry-example-9')), findsOneWidget);
    expect(find.text('删除物种'), findsOneWidget);

    await tester.tap(find.text('删除物种'));
    await tester.pumpAndSettle();
    expect(find.textContaining('删除“未知物种 #1”'), findsOneWidget);
    await tester.tap(find.text('确认删除'));
    await tester.pumpAndSettle();

    expect(deleted, isTrue);
    expect(find.text('暂无候选或已注册物种'), findsOneWidget);
  });

  testWidgets('selected DINO box title includes its display number and species', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final tempDir = await Directory.systemTemp.createTemp('neri_dinov3_number_');
    addTearDown(() async {
      if (await tempDir.exists()) await tempDir.delete(recursive: true);
    });
    final imageFile = File('${tempDir.path}/frame.png');
    await imageFile.writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
    final client = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
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
          bbox: <double>[0, 0, 10, 10],
          observationId: 'obs-1',
          predictedSpecies: '盘羊',
          feedbackStatus: 'unverified',
        ),
      ],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SpeciesValidationScreen(
            apiClient: client,
            inputPath: tempDir.path,
            classificationModelPath: 'head.neri.json',
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
    await tester.pumpAndSettle();

    final viewer = find.byType(DetectionMediaViewer);
    await tester.tapAt(tester.getCenter(viewer));
    await tester.pumpAndSettle();

    expect(find.text('#1 盘羊 · 检测框校验'), findsOneWidget);
  });
}
