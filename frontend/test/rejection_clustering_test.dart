import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/rejection_cluster.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/utils/dino_rejection.dart';
import 'package:neri_flutter/src/utils/detection_box_label.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

Map<String, dynamic> cluster(int number, {String status = 'clustered'}) => {
  'cluster_id': 'run:$number',
  'candidate_number': number,
  'run_id': 'run',
  'model_fingerprint': 'encoder:checkpoint',
  'event_count': 5,
  'min_cluster_events': 5,
  'density_verified': status == 'clustered',
  'is_noise': status == 'noise',
  'status': status,
};

DetectionBox box(String id, int number, {String status = 'clustered'}) =>
    DetectionBox.fromJson({
      'species': 'Unknown',
      'bbox': [0.1, 0.1, 0.7, 0.7],
      'confidence': .3,
      'observation_id': 'dinov3:$id',
      'rejection_cluster': number < 0
          ? <String, dynamic>{}
          : cluster(number, status: status),
      'candidates': [
        {'name': 'Unknown', 'accepted': false, 'source': 'dinov3_ridge_within'},
      ],
    });

void main() {
  test(
    'cluster overlay names and colors are stable and preserve manual labels',
    () {
      final first = box('first', 1);
      final second = box('second', 2);
      final sameCluster = box('another-folder', 1);
      expect(detectionBoxLabel(first, 0), '#1 簇1 0.30');
      expect(detectionBoxLabel(second, 1), '#2 簇2 0.30');
      expect(detectionBoxColor(first), isNot(detectionBoxColor(second)));
      expect(detectionBoxColor(first), detectionBoxColor(sameCluster));
      expect(first.species, 'Unknown');
      final human = DetectionBox(
        species: '豹猫',
        bbox: first.bbox,
        feedbackStatus: 'confirmed',
        candidates: first.candidates,
        rejectionCluster: first.rejectionCluster,
      );
      expect(detectionBoxDisplayName(human), '豹猫');
      final accepted = DetectionBox(
        species: '岩羊',
        bbox: first.bbox,
        candidates: const [
          {'source': 'dinov3_ridge_within', 'accepted': true},
        ],
        rejectionCluster: first.rejectionCluster,
      );
      expect(detectionBoxDisplayName(accepted), '岩羊');
      for (final status in ['noise', 'failed']) {
        expect(
          detectionBoxDisplayName(box(status, 0, status: status)),
          'Unknown',
        );
      }
    },
  );

  test(
    'overlay hydrates saved raw metadata and drops stale pending clusters',
    () {
      final original = box('raw', -1);
      final item = DetectionItem(
        filename: 'a.jpg',
        path: 'a.jpg',
        fileType: 'jpg',
        detectionBoxes: [original],
        detectionData: {
          'sequence_completed': true,
          '检测框': [
            {
              'observation_id': original.observationId,
              'rejection_cluster': cluster(2),
            },
          ],
        },
      );
      final hydrated = detectionBoxWithDisplayMetadata(item, original, 0);
      expect(detectionBoxDisplayName(hydrated), '簇2');
      expect(original.rejectionCluster, isEmpty);
      final pending = DetectionItem(
        filename: 'a.jpg',
        path: 'a.jpg',
        fileType: 'jpg',
        detectionBoxes: [hydrated],
        detectionData: const {'sequence_pending': true},
      );
      expect(
        detectionBoxDisplayName(
          detectionBoxWithDisplayMetadata(pending, hydrated, 0),
        ),
        'Unknown',
      );
    },
  );

  test(
    'persisted metadata survives parsing and final-label reconstruction',
    () {
      final saved = cluster(2);
      final parsed = DetectionBox.fromJson({
        'species': '豹猫',
        'bbox': [0, 0, 1, 1],
        'rejection_cluster': saved,
        'candidates': [
          {'accepted': false, 'source': 'dinov3_ridge_within'},
        ],
      });
      expect(dinoFinalBox(parsed).species, 'Unknown');
      expect(dinoFinalBox(parsed).rejectionCluster, saved);
      expect(
        RejectionCluster.fromJson(cluster(0, status: 'failed')).isFailed,
        isTrue,
      );
      expect(
        RejectionCluster.fromJson(cluster(0, status: 'failed')).isNoise,
        isFalse,
      );
      expect(
        RejectionCluster.fromJson(cluster(0, status: 'noise')).isNoise,
        isTrue,
      );
    },
  );

  late Directory directory;
  late NeriApiClient api;
  late List<String> requests;
  setUp(() async {
    directory = await Directory.systemTemp.createTemp('neri_cluster_test_');
    requests = [];
    api = NeriApiClient(
      httpClient: MockClient((request) async {
        requests.add(request.url.path);
        return http.Response(
          '{}',
          200,
          headers: {'content-type': 'application/json'},
        );
      }),
    );
  });
  tearDown(() async {
    api.close();
    await directory.delete(recursive: true);
  });

  Future<DetectionItem> photo(
    String name,
    List<DetectionBox> boxes, {
    bool pending = false,
    String species = 'Unknown',
    bool validated = false,
  }) async {
    final file = File('${directory.path}/$name.png');
    file.writeAsBytesSync(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
    return DetectionItem(
      filename: '$name.png',
      path: file.path,
      fileType: 'png',
      width: 10,
      height: 10,
      dateTaken: '2026-11-14T12:00:00Z',
      species: [species],
      validated: validated,
      detectionBoxes: boxes,
      detectionData: {
        '物种名称': species,
        'sequence_completed': !pending,
        'sequence_pending': pending,
      },
    );
  }

  Future<void> show(
    WidgetTester tester,
    List<DetectionItem> items, {
    bool autoGroup = false,
    int refresh = 0,
  }) async {
    tester.view.physicalSize = const Size(1600, 1200);
    tester.view.devicePixelRatio = 1;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SpeciesValidationScreen(
            apiClient: api,
            classificationModelPath: 'model.neri.json',
            inputPath: directory.path,
            rootInputPath: directory.path,
            rootItems: items,
            exportFormatMode: 'ask',
            items: items,
            loading: false,
            refreshVersion: refresh,
            speciesTypes: const {},
            useCombinedConfidence: false,
            minFrameRatio: 0,
            autoGroup: autoGroup,
            collapseGroups: autoGroup,
            autoGroupDetectBurst: false,
            autoGroupBurstSize: 4,
            autoGroupGapSeconds: 1800,
            autoSortQuickMarks: false,
            undoSteps: 20,
            quickMarkSpecies: const [],
            quickMarkRecentHistory: const [],
            quickMarkUsageCounts: const {},
            quantityButtons: const ['1'],
            exportColumns: const [],
            favoritePhotoPaths: const [],
            favoritePhotoExportMode: favoritePhotoExportAsk,
            emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
            onRefresh: () async {},
            onLoadMetadata: (_) async {},
            onOpenExternal: (_) {},
            onMarkItem:
                (
                  item,
                  action, {
                  speciesName,
                  speciesCount,
                  speciesType,
                  remark,
                }) async => item,
            onMarkItems:
                (
                  items,
                  action, {
                  speciesName,
                  speciesCount,
                  speciesType,
                  remark,
                }) async => items,
            onQuickMarkUsed: (_) async {},
            onQuickMarkReverted: (_) async {},
            onRedetectItems: (_, {required confidence}) async {},
            onFavoritePhotoPathsChanged: (_) async {},
            onFavoritePhotoExportModeChanged: (_) async {},
            onExportFormatModeChanged: (_) async {},
            onEmptyPhotoDeleteModeChanged: (_) async {},
            onAutoGroupInferredBurstSizeChanged: (_) {},
          ),
        ),
      ),
    );
    for (var i = 0; i < 5; i++) {
      await tester.pump(const Duration(milliseconds: 50));
    }
  }

  testWidgets(
    'without automatic grouping each photo remains in its target cluster buckets',
    (tester) async {
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final a = await photo('a', [box('a1', 1)]);
      final b = await photo('b', [box('b2', 2)]);
      final c = await photo('c', [box('c1', 1), box('c2', 2)]);
      await show(tester, [a, b, c]);
      expect(find.text('拒识（簇1）'), findsOneWidget);
      expect(find.text('拒识（簇2）'), findsOneWidget);
      expect(find.byTooltip('确认簇物种并学习'), findsNothing);
      await tester.tap(find.text('拒识（簇1）'));
      await tester.pump();
      expect(find.text('b.png'), findsNothing);
      expect(find.text('a.png'), findsWidgets);
      expect(find.text('c.png'), findsWidgets);
      await tester.tap(find.text('c.png').first);
      await tester.pump();
      var viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.item.path, c.path);
      expect(viewer.selectedObservationId, 'dinov3:c1');
      await tester.tap(find.text('拒识（簇2）'));
      await tester.pump();
      expect(find.text('a.png'), findsNothing);
      await tester.tap(find.text('c.png').first);
      await tester.pump();
      viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.selectedObservationId, 'dinov3:c2');
      expect(
        requests.where(
          (path) =>
              path.contains('rejection-clusters') ||
              path.endsWith('/feedback/cluster'),
        ),
        isEmpty,
      );
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets('correcting one target removes only its old cluster membership', (
    tester,
  ) async {
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final a = await photo('a', [box('a', 1)]);
    final c = await photo('c', [box('c1', 1), box('c2', 2)]);
    await show(tester, [a, c]);
    final corrected = await photo('c', [
      DetectionBox(
        species: '豹猫',
        bbox: c.detectionBoxes.first.bbox,
        observationId: 'dinov3:c1',
        feedbackStatus: 'confirmed',
        rejectionCluster: c.detectionBoxes.first.rejectionCluster,
        candidates: c.detectionBoxes.first.candidates,
      ),
      c.detectionBoxes.last,
    ]);
    await show(tester, [a, corrected]);
    await tester.tap(find.text('拒识（簇1）'));
    await tester.pump();
    expect(find.text('c.png'), findsNothing);
    await tester.tap(find.text('拒识（簇2）'));
    await tester.pump();
    expect(find.text('c.png'), findsWidgets);
    expect(
      tester
          .widget<DetectionMediaViewer>(find.byType(DetectionMediaViewer))
          .selectedObservationId,
      'dinov3:c2',
    );
    expect(find.text('豹猫'), findsWidgets);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets(
    'automatic grouping keeps entire bursts in one combined cluster bucket',
    (tester) async {
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final a = await photo('a', [box('a10', 10)]);
      final b = await photo('b', [box('b2', 2)]);
      final c = await photo('c', [box('c10', 10), box('c2', 2), box('c2b', 2)]);
      await show(tester, [a, b, c], autoGroup: true);
      final list = find.byKey(const ValueKey('validation-species-list'));
      final titles = tester
          .widgetList<ListTile>(
            find.descendant(of: list, matching: find.byType(ListTile)),
          )
          .map((tile) => (tile.title! as Text).data)
          .toList();
      expect(titles, ['簇2、簇10']);
      expect(find.text('1 组 / 3 个文件'), findsOneWidget);
      expect(find.byTooltip('确认簇物种并学习'), findsNothing);
      await tester.tap(find.byTooltip('展开分组'));
      await tester.pump();
      for (final name in ['a.png', 'b.png', 'c.png']) {
        expect(find.text(name), findsWidgets);
      }
      await tester.tap(find.text('c.png').first);
      await tester.pump();
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes.length, 3);
      expect(viewer.visibleBoxes.map(detectionBoxDisplayName).toList(), [
        '簇10',
        '簇2',
        '簇2',
      ]);
      expect(
        viewer.visibleBoxes.every((box) => box.species == 'Unknown'),
        isTrue,
      );
      expect(
        requests.where(
          (path) =>
              path.contains('rejection-clusters') ||
              path.endsWith('/feedback/cluster'),
        ),
        isEmpty,
      );
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'large automatic grouping keeps combined buckets during preparation',
    (tester) async {
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final items = <DetectionItem>[];
      for (var i = 0; i < 520; i++) {
        items.add(
          await photo('frame${i.toString().padLeft(4, '0')}', [
            box('frame$i', i % 2 + 1),
          ]),
        );
      }
      await show(tester, items, autoGroup: true);
      final list = find.byKey(const ValueKey('validation-species-list'));
      for (var i = 0; i < 100; i++) {
        if (find
            .descendant(of: list, matching: find.text('簇1、簇2'))
            .evaluate()
            .isNotEmpty) {
          break;
        }
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump(const Duration(milliseconds: 20));
      }
      final tiles = tester
          .widgetList<ListTile>(
            find.descendant(of: list, matching: find.byType(ListTile)),
          )
          .toList();
      expect(tiles.map((tile) => (tile.title! as Text).data), ['簇1、簇2']);
      expect((tiles.single.subtitle! as Text).data, '130 组 / 520 个文件');
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets('combined groups update when one target is manually corrected', (
    tester,
  ) async {
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final a = await photo('a', [box('a', 1)]);
    final c = await photo('c', [box('c1', 1), box('c2', 2)]);
    await show(tester, [a, c], autoGroup: true);
    expect(find.text('簇1、簇2'), findsWidgets);
    final corrected = await photo('c', [
      DetectionBox(
        species: '豹猫',
        bbox: c.detectionBoxes.first.bbox,
        observationId: 'dinov3:c1',
        feedbackStatus: 'confirmed',
        candidates: c.detectionBoxes.first.candidates,
        rejectionCluster: c.detectionBoxes.first.rejectionCluster,
      ),
      c.detectionBoxes.last,
    ]);
    await show(tester, [a, corrected], autoGroup: true);
    final list = find.byKey(const ValueKey('validation-species-list'));
    final tiles = tester
        .widgetList<ListTile>(
          find.descendant(of: list, matching: find.byType(ListTile)),
        )
        .toList();
    expect(tiles.map((tile) => (tile.title! as Text).data), ['豹猫、簇1、簇2']);
    expect(find.text('1 组 / 2 个文件'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets(
    'noise, failure, pending and historical unclustered states stay distinct',
    (tester) async {
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      await show(tester, [
        await photo('noise', [box('noise', 0, status: 'noise')]),
        await photo('failure', [box('failure', 0, status: 'failed')]),
        await photo('pending', [box('pending', -1)], pending: true),
        await photo('old', [box('old', -1)]),
      ]);
      for (final label in ['拒识（未成簇）', '拒识（聚类失败）', '拒识（待聚类）', '拒识（未聚类）']) {
        expect(find.text(label), findsOneWidget);
      }
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'metadata-only refresh updates membership and sorts cluster numbers numerically',
    (tester) async {
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final a = await photo('a', [box('a', 10)]);
      final b = await photo('b', [box('b', 2)]);
      final noise = await photo('noise', [box('noise', 0, status: 'noise')]);
      final empty = await photo('empty', [], species: '空');
      final confirmed = await photo(
        'confirmed',
        [],
        species: '豹猫',
        validated: true,
      );
      await show(tester, [confirmed, noise, empty, a, b]);
      expect(
        tester.getTopLeft(find.text('拒识（簇2）')).dy,
        lessThan(tester.getTopLeft(find.text('拒识（簇10）')).dy),
      );
      final speciesList = find.byKey(const ValueKey('validation-species-list'));
      expect(
        tester
            .widgetList<ListTile>(
              find.descendant(of: speciesList, matching: find.byType(ListTile)),
            )
            .map((tile) => (tile.title! as Text).data)
            .toList(),
        ['拒识（簇2）', '拒识（簇10）', '拒识（未成簇）', '空', '豹猫'],
      );
      await show(tester, [
        await photo('a', [box('a', 3)]),
        b,
        noise,
        empty,
        confirmed,
      ]);
      expect(find.text('拒识（簇10）'), findsNothing);
      expect(find.text('拒识（簇3）'), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );
}
