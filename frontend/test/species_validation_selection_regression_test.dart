import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/widgets/dino_feature_scatter.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

import 'support/fake_validation_video_backend.dart';

void main() {
  late Directory tempDir;
  late NeriApiClient apiClient;

  setUp(() async {
    installFakeValidationVideoBackend();
    tempDir = await Directory.systemTemp.createTemp(
      'neri_selection_regression_',
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

  DetectionItem createPhoto(
    String name, {
    Directory? dir,
    String? dateTaken = '2026-11-10T12:00:00Z',
    String? modifiedAt = '2026-11-10T12:00:00Z',
    String species = '豹猫',
    List<DetectionBox> boxes = const [],
    bool? validated,
  }) {
    final targetDir = dir ?? tempDir;
    final file = File('${targetDir.path}/$name');
    final speciesList = species
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    return DetectionItem(
      filename: name,
      path: file.path,
      fileType: name.toLowerCase().endsWith('.mov') ? 'mov' : 'jpg',
      dateTaken: dateTaken,
      modifiedAt: modifiedAt,
      width: 10,
      height: 10,
      species: speciesList,
      confidence: 0.95,
      detectionBoxes: boxes,
      detectionData: {'物种名称': species, if (validated == true) '最低置信度': '人工校验'},
      validated: validated,
    );
  }

  DetectionBox clusterBox(
    String id,
    int clusterNumber, {
    String species = '拒识/Unknown',
  }) {
    return DetectionBox(
      species: species,
      bbox: const [0.1, 0.1, 0.4, 0.4],
      observationId: 'dinov3:$id',
      rejectionCluster: {
        'cluster_id': 'fp:$clusterNumber',
        'candidate_number': clusterNumber,
        'model_fingerprint': 'fp',
        'event_count': 8,
        'min_cluster_events': 5,
        'is_noise': false,
        'density_verified': true,
      },
      candidates: const [
        {'name': '豹猫', 'conf': 0.2},
      ],
    );
  }

  void configureView(WidgetTester tester) {
    tester.view.physicalSize = const Size(1600, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(() async {
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump();
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
    });
  }

  Future<void> settleFrames(WidgetTester tester) async {
    await tester.pump();
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump(const Duration(milliseconds: 200));
  }

  Future<void> waitUntilPrepared(WidgetTester tester) async {
    for (var attempt = 0; attempt < 500; attempt++) {
      final blocked = find.byWidgetPredicate(
        (widget) =>
            widget is AbsorbPointer &&
            widget.key == const ValueKey('validation-directory-content') &&
            widget.absorbing,
      );
      if (blocked.evaluate().isEmpty) break;
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
      await tester.pump(const Duration(milliseconds: 16));
    }
  }

  ListTile selectedBucketTile(WidgetTester tester) {
    final speciesList = find.byKey(const ValueKey('validation-species-list'));
    final tiles = tester.widgetList<ListTile>(
      find.descendant(of: speciesList, matching: find.byType(ListTile)),
    );
    return tiles.firstWhere((tile) => tile.selected);
  }

  String selectedBucketTitle(WidgetTester tester) =>
      (selectedBucketTile(tester).title! as Text).data!;

  bool isSelectedBucketValidated(WidgetTester tester) {
    final tile = selectedBucketTile(tester);
    final tooltip = tile.leading! as Tooltip;
    final icon = tooltip.child! as Icon;
    return icon.icon == Icons.check_circle_rounded;
  }

  String activeViewerPath(WidgetTester tester) => tester
      .widget<DetectionMediaViewer>(find.byType(DetectionMediaViewer))
      .item
      .path;

  Future<void> tapBucket(WidgetTester tester, String title) async {
    final speciesList = find.byKey(const ValueKey('validation-species-list'));
    await tester.tap(
      find.descendant(of: speciesList, matching: find.text(title)).first,
    );
    await settleFrames(tester);
  }

  testWidgets(
    'clicking video hydrates UTC details without splitting burst or changing species bucket',
    (tester) async {
      configureView(tester);
      final first = createPhoto(
        'cat.jpg',
        dateTaken: '2025-11-22T10:00:00',
        species: '豹猫',
      );
      final burst = [
        for (var i = 1133; i <= 1135; i++)
          createPhoto(
            'IMAG$i.JPG',
            dateTaken: null,
            modifiedAt: '2025-11-22T16:32:20',
            species: '环颈雉',
          ),
      ];
      final video = createPhoto(
        'IMAG1136.MOV',
        dateTaken: null,
        modifiedAt: '2025-11-22T16:32:32',
        species: '环颈雉',
      );
      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [first, ...burst, video],
          hydrateMetadata: true,
          onLoadMetadataAsync: (item, update) async {
            if (item.path != video.path) return;
            await Future<void>.delayed(const Duration(milliseconds: 20));
            update(
              DetectionItem(
                filename: item.filename,
                path: item.path,
                fileType: item.fileType,
                dateTaken: '2025-11-23T00:32:22+08:00',
                // Simulate an older response arriving after the current results.
                species: const ['家羊'],
                detectionData: const {'物种名称': '家羊'},
                width: 1920,
                height: 1080,
              ),
            );
          },
        ),
      );
      await settleFrames(tester);
      await tapBucket(tester, '环颈雉');
      await tester.tap(find.byTooltip('展开分组'));
      await settleFrames(tester);
      expect(find.textContaining('第 2/2 组'), findsWidgets);
      await tester.tap(find.text('IMAG1136.MOV').first);
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 50)),
      );
      await settleFrames(tester);
      expect(selectedBucketTitle(tester), '环颈雉');
      expect(activeViewerPath(tester), video.path);
      expect(find.textContaining('第 2/2 组'), findsWidgets);
      expect(find.textContaining('第 3/3 组'), findsNothing);
      for (final item in burst) {
        expect(find.text(item.filename), findsOneWidget);
      }
      expect(find.byTooltip('折叠分组'), findsOneWidget);
      expect(
        (selectedBucketTile(tester).subtitle! as Text).data,
        '1 组 / 4 个文件',
      );
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'mixed known species and noise group deduplicates individual names',
    (tester) async {
      configureView(tester);
      final person = const DetectionBox(
        species: '人员',
        bbox: [0, 0, 1, 1],
        confidence: 0.9,
      );
      final sheep = const DetectionBox(
        species: '家羊',
        bbox: [1, 1, 2, 2],
        confidence: 0.9,
      );
      final noise = DetectionBox(
        species: '拒识/Unknown',
        bbox: const [2, 2, 3, 3],
        observationId: 'dinov3:noise',
        rejectionCluster: const {
          'cluster_id': 'fp:noise',
          'is_noise': true,
          'density_verified': false,
        },
      );
      await tester.pumpWidget(
        _RegressionHarness(
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          classificationModelPath: 'test.neri.json',
          burstSize: 3,
          initialItems: [
            createPhoto('1.jpg', species: '人员', boxes: [person]),
            createPhoto('2.jpg', species: '人员,家羊', boxes: [person, sheep]),
            createPhoto('3.jpg', species: '拒识/Unknown', boxes: [noise]),
            createPhoto('4.jpg', species: '人员', boxes: [person]),
            createPhoto('5.jpg', species: '家羊', boxes: [sheep]),
            createPhoto('6.jpg', species: '拒识/Unknown', boxes: [noise]),
          ],
        ),
      );
      await settleFrames(tester);
      expect(selectedBucketTitle(tester), '人员、家羊、噪声');
      expect(
        (selectedBucketTile(tester).subtitle! as Text).data,
        '2 组 / 6 个文件',
      );
      expect(find.textContaining('人员、人员'), findsNothing);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'non-first bucket regroup keeps selected bucket, expansion, and item when burst size changes',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boar0 = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boar1 = createPhoto(
        'boar-1.jpg',
        dateTaken: '2026-11-10T12:00:02Z',
        species: '野猪',
      );
      final boar2 = createPhoto(
        'boar-2.jpg',
        dateTaken: '2026-11-10T12:00:04Z',
        species: '野猪',
      );
      final boar3 = createPhoto(
        'boar-3.jpg',
        dateTaken: '2026-11-10T12:00:06Z',
        species: '野猪',
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boar0, boar1, boar2, boar3],
          burstSize: 4,
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      expect(selectedBucketTitle(tester), '野猪');

      await tester.tap(find.byTooltip('展开分组'));
      await settleFrames(tester);
      await tester.tap(find.text('boar-2.jpg').first);
      await settleFrames(tester);
      expect(activeViewerPath(tester), boar2.path);

      key.currentState!.setBurstSize(2);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boar2.path);
      expect(find.byTooltip('折叠分组'), findsWidgets);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'non-first bucket cluster key change keeps selected item in updated cluster bucket',
    (tester) async {
      configureView(tester);
      final cat = createPhoto('cat-0.jpg', species: '豹猫');
      final rejA = createPhoto(
        'rej-a.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('a', 10)],
      );
      final rejB = createPhoto(
        'rej-b.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('b', 2)],
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, rejA, rejB],
          classificationModelPath: 'model.neri.json',
          autoGroup: false,
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '拒识（簇10）');
      expect(selectedBucketTitle(tester), '拒识（簇10）');
      expect(activeViewerPath(tester), rejA.path);

      final updatedRejA = createPhoto(
        'rej-a.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('a', 3)],
      );
      key.currentState!.updateItems([cat, updatedRejA, rejB]);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '拒识（簇3）');
      expect(activeViewerPath(tester), rejA.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'explicit multi-cluster bucket selection survives metadata regroup',
    (tester) async {
      configureView(tester);
      final shared = createPhoto(
        'shared.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('s1', 1), clusterBox('s2', 2)],
      );
      final other = createPhoto(
        'other.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('o1', 1)],
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [other, shared],
          classificationModelPath: 'model.neri.json',
          autoGroup: false,
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '拒识（簇2）');
      expect(selectedBucketTitle(tester), '拒识（簇2）');
      expect(activeViewerPath(tester), shared.path);

      final updatedShared = createPhoto(
        'shared.jpg',
        dateTaken: '2026-11-10T12:05:00Z',
        species: '拒识/Unknown',
        boxes: [clusterBox('s1', 1), clusterBox('s2', 2)],
      );
      key.currentState!.updateItems([other, updatedShared]);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '拒识（簇2）');
      expect(activeViewerPath(tester), shared.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'validating last remaining group follows updated item to validated bucket',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boar0 = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boar1 = createPhoto(
        'boar-1.jpg',
        dateTaken: '2026-11-10T12:00:02Z',
        species: '野猪',
      );
      final egret = createPhoto(
        'egret-0.jpg',
        dateTaken: '2026-11-10T14:00:00Z',
        species: '大白鹭',
        validated: true,
      );

      await tester.pumpWidget(
        _RegressionHarness(
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boar0, boar1, egret],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      expect(selectedBucketTitle(tester), '野猪');
      expect(isSelectedBucketValidated(tester), isFalse);
      expect(activeViewerPath(tester), boar0.path);

      await tester.tap(find.text('正确').first);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(isSelectedBucketValidated(tester), isTrue);
      expect(activeViewerPath(tester), anyOf(boar0.path, boar1.path));
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'validating last remaining group with species change follows updated item to new species bucket',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boar = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );

      await tester.pumpWidget(
        _RegressionHarness(
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boar],
          quickMarkSpecies: const ['赤麻鸭'],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boar.path);

      await tester.tap(find.text('赤麻鸭').first);
      await settleFrames(tester);
      await tester.tap(find.text('1').last);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '赤麻鸭');
      expect(isSelectedBucketValidated(tester), isTrue);
      expect(activeViewerPath(tester), boar.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'intentional next-item navigation stays in non-first bucket when another group remains',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boarA = createPhoto(
        'boar-a.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boarB = createPhoto(
        'boar-b.jpg',
        dateTaken: '2026-11-10T14:00:00Z',
        species: '野猪',
      );

      await tester.pumpWidget(
        _RegressionHarness(
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boarA, boarB],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      expect(activeViewerPath(tester), boarA.path);

      await tester.tap(find.text('正确').first);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(isSelectedBucketValidated(tester), isFalse);
      expect(activeViewerPath(tester), boarB.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'detector_category box details dialog skips explain API request and shows detector confidence',
    (tester) async {
      configureView(tester);
      final requests = <Uri>[];
      final guardApi = NeriApiClient(
        httpClient: MockClient((request) async {
          requests.add(request.url);
          return http.Response(
            '{}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }),
      );
      addTearDown(guardApi.close);

      final human = createPhoto(
        'human-0.jpg',
        species: '人',
        boxes: const [
          DetectionBox(
            species: '人',
            confidence: 0.912,
            bbox: [0.2, 0.2, 0.6, 0.8],
            observationId: 'obs-human-1',
            candidates: [
              {
                'name': '人',
                'conf': 0.912,
                'raw_det_conf': 0.912,
                'classification_source': 'detector_category',
                'source': 'detector_category',
                'confidence_kind': 'detector_confidence',
                'gate_mode': 'detector',
                'accepted': true,
              },
            ],
          ),
        ],
      );

      await tester.pumpWidget(
        _RegressionHarness(
          apiClient: guardApi,
          initialInputPath: tempDir.path,
          initialItems: [human],
          classificationModelPath: 'model.neri.json',
        ),
      );
      await settleFrames(tester);

      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      viewer.onDetectionBoxSelected?.call(human.detectionBoxes.first);
      await settleFrames(tester);

      final dialog = find.byKey(const ValueKey('dino-detection-box-dialog'));
      expect(dialog, findsOneWidget);
      expect(find.byType(DinoFeatureExplanationPanel), findsNothing);
      expect(
        find.descendant(of: dialog, matching: find.textContaining('探测器类别直接保留')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: dialog, matching: find.textContaining('0.912')),
        findsWidgets,
      );
      expect(requests.where((uri) => uri.path.endsWith('/explain')), isEmpty);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'removing last item in non-first bucket prefers same species or nearest meaningful bucket',
    (tester) async {
      configureView(tester);
      final first = createPhoto(
        'antelope-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '斑羚',
      );
      final boarUnverified = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final third = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T14:00:00Z',
        species: '豹猫',
      );
      final boarValidated = createPhoto(
        'boar-1.jpg',
        dateTaken: '2026-11-10T16:00:00Z',
        species: '野猪',
        validated: true,
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [first, boarUnverified, third, boarValidated],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      expect(selectedBucketTitle(tester), '野猪');
      expect(isSelectedBucketValidated(tester), isFalse);

      // Remove unverified boar-0 while validated boar-1 remains: chooses same species.
      key.currentState!.updateItems([first, third, boarValidated]);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(isSelectedBucketValidated(tester), isTrue);
      expect(activeViewerPath(tester), boarValidated.path);

      // Remove validated boar-1 as well: chooses nearest remaining bucket rather than first blindly.
      key.currentState!.updateItems([first, third]);
      await settleFrames(tester);
      expect(selectedBucketTitle(tester), '豹猫');
      expect(activeViewerPath(tester), third.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'async metadata timestamp update in non-first bucket keeps bucket, selection, and expansion',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boar0 = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boar1 = createPhoto(
        'boar-1.jpg',
        dateTaken: '2026-11-10T12:00:02Z',
        species: '野猪',
      );
      final boar2 = createPhoto(
        'boar-2.jpg',
        dateTaken: '2026-11-10T12:00:04Z',
        species: '野猪',
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boar0, boar1, boar2],
          onLoadMetadataAsync: (item, update) async {
            if (item.path == boar1.path) {
              await Future<void>.delayed(const Duration(milliseconds: 20));
              update(
                DetectionItem(
                  filename: item.filename,
                  path: item.path,
                  fileType: item.fileType,
                  dateTaken: '2026-11-10T12:50:00Z',
                  modifiedAt: item.modifiedAt,
                  width: item.width,
                  height: item.height,
                  species: item.species,
                  confidence: item.confidence,
                  detectionBoxes: item.detectionBoxes,
                  detectionData: item.detectionData,
                  validated: item.validated,
                ),
              );
            }
          },
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      await tester.tap(find.byTooltip('展开分组'));
      await settleFrames(tester);
      await tester.tap(find.text('boar-1.jpg').first);
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 50)),
      );
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boar1.path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'async metadata cluster update in large list preserves selection through preparation',
    (tester) async {
      configureView(tester);
      final items = <DetectionItem>[
        for (var i = 0; i < 518; i++)
          createPhoto(
            'cat-$i.jpg',
            dateTaken:
                '2026-11-10T10:00:${(i % 60).toString().padLeft(2, '0')}Z',
            species: '豹猫',
          ),
        createPhoto(
          'target-a.jpg',
          species: '拒识/Unknown',
          boxes: const [
            DetectionBox(
              species: '拒识/Unknown',
              bbox: [0.1, 0.1, 0.4, 0.4],
              observationId: 'dinov3:target-a',
            ),
          ],
        ),
        createPhoto(
          'target-b.jpg',
          species: '拒识/Unknown',
          boxes: [clusterBox('target-b', 2)],
        ),
      ];

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: items,
          classificationModelPath: 'model.neri.json',
          autoGroup: false,
        ),
      );
      await waitUntilPrepared(tester);
      await settleFrames(tester);

      await tapBucket(tester, '拒识（未聚类）');
      expect(selectedBucketTitle(tester), '拒识（未聚类）');
      expect(activeViewerPath(tester), items[518].path);

      final updatedItems = List<DetectionItem>.from(items);
      updatedItems[518] = createPhoto(
        'target-a.jpg',
        species: '拒识/Unknown',
        boxes: [clusterBox('target-a', 5)],
      );
      key.currentState!.updateItems(updatedItems);
      await waitUntilPrepared(tester);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '拒识（簇5）');
      expect(activeViewerPath(tester), items[518].path);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'refresh preserves current selected bucket, file, and expanded group',
    (tester) async {
      configureView(tester);
      final cat = createPhoto(
        'cat-0.jpg',
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boar0 = createPhoto(
        'boar-0.jpg',
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boar1 = createPhoto(
        'boar-1.jpg',
        dateTaken: '2026-11-10T12:00:02Z',
        species: '野猪',
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: tempDir.path,
          initialItems: [cat, boar0, boar1],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      await tester.tap(find.byTooltip('展开分组'));
      await settleFrames(tester);
      await tester.tap(find.text('boar-1.jpg').first);
      await settleFrames(tester);
      expect(activeViewerPath(tester), boar1.path);

      key.currentState!.refreshWith([
        createPhoto(
          'cat-0.jpg',
          dateTaken: '2026-11-10T10:00:00Z',
          species: '豹猫',
        ),
        createPhoto(
          'boar-0.jpg',
          dateTaken: '2026-11-10T12:00:00Z',
          species: '野猪',
        ),
        createPhoto(
          'boar-1.jpg',
          dateTaken: '2026-11-10T12:00:02Z',
          species: '野猪',
        ),
      ]);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boar1.path);
      expect(find.byTooltip('折叠分组'), findsOneWidget);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'transient empty items during load preserves selection for same inputPath and resets when inputPath changes',
    (tester) async {
      configureView(tester);
      final dirA = Directory('${tempDir.path}/dirA')..createSync();
      final dirB = Directory('${tempDir.path}/dirB')..createSync();
      final catA = createPhoto(
        'cat-a.jpg',
        dir: dirA,
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boarA0 = createPhoto(
        'boar-a0.jpg',
        dir: dirA,
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );
      final boarA1 = createPhoto(
        'boar-a1.jpg',
        dir: dirA,
        dateTaken: '2026-11-10T12:00:02Z',
        species: '野猪',
      );
      final catB = createPhoto(
        'cat-b.jpg',
        dir: dirB,
        dateTaken: '2026-11-10T10:00:00Z',
        species: '豹猫',
      );
      final boarB = createPhoto(
        'boar-b.jpg',
        dir: dirB,
        dateTaken: '2026-11-10T12:00:00Z',
        species: '野猪',
      );

      final key = GlobalKey<_RegressionHarnessState>();
      await tester.pumpWidget(
        _RegressionHarness(
          key: key,
          apiClient: apiClient,
          initialInputPath: dirA.path,
          initialItems: [catA, boarA0, boarA1],
        ),
      );
      await settleFrames(tester);

      await tapBucket(tester, '野猪');
      await tester.tap(find.byTooltip('展开分组'));
      await settleFrames(tester);
      await tester.tap(find.text('boar-a1.jpg').first);
      await settleFrames(tester);
      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boarA1.path);

      // Transient empty items while reloading the same inputPath.
      key.currentState!.setLoadingItems(loading: true, items: const []);
      await settleFrames(tester);

      key.currentState!.setLoadingItems(
        loading: false,
        items: [catA, boarA0, boarA1],
      );
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '野猪');
      expect(activeViewerPath(tester), boarA1.path);
      expect(find.byTooltip('折叠分组'), findsOneWidget);

      // Switching inputPath resets selection to the new folder's default.
      key.currentState!.switchInput(dirB.path, [catB, boarB]);
      await settleFrames(tester);

      expect(selectedBucketTitle(tester), '豹猫');
      expect(activeViewerPath(tester), catB.path);
      expect(tester.takeException(), isNull);
    },
  );
}

class _RegressionHarness extends StatefulWidget {
  const _RegressionHarness({
    super.key,
    required this.apiClient,
    required this.initialInputPath,
    required this.initialItems,
    this.classificationModelPath,
    this.autoGroup = true,
    this.burstSize = 4,
    this.quickMarkSpecies = const <String>['豹猫'],
    this.onLoadMetadataAsync,
    this.hydrateMetadata = false,
  });

  final NeriApiClient apiClient;
  final String initialInputPath;
  final List<DetectionItem> initialItems;
  final String? classificationModelPath;
  final bool autoGroup;
  final int burstSize;
  final List<String> quickMarkSpecies;
  final bool hydrateMetadata;
  final Future<void> Function(
    DetectionItem item,
    void Function(DetectionItem updated) applyUpdate,
  )?
  onLoadMetadataAsync;

  @override
  State<_RegressionHarness> createState() => _RegressionHarnessState();
}

class _RegressionHarnessState extends State<_RegressionHarness> {
  late String _inputPath = widget.initialInputPath;
  late List<DetectionItem> _items = widget.initialItems;
  late int _burstSize = widget.burstSize;
  bool _loading = false;
  int _refreshVersion = 0;

  void setBurstSize(int value) => setState(() => _burstSize = value);

  void updateItems(List<DetectionItem> next) => setState(() => _items = next);

  void refreshWith(List<DetectionItem> next) => setState(() {
    _refreshVersion++;
    _items = next;
  });

  void setLoadingItems({
    required bool loading,
    required List<DetectionItem> items,
  }) => setState(() {
    _loading = loading;
    _items = items;
  });

  void switchInput(String inputPath, List<DetectionItem> items) => setState(() {
    _inputPath = inputPath;
    _items = items;
    _loading = false;
  });

  DetectionItem _markCopy(
    DetectionItem item,
    String action, {
    String? speciesName,
  }) {
    final nextSpecies =
        action == 'update' && speciesName != null && speciesName.isNotEmpty
        ? [speciesName]
        : item.species;
    final nextData = Map<String, dynamic>.from(item.detectionData)
      ..['物种名称'] = nextSpecies.join(',')
      ..['最低置信度'] = '人工校验';
    final updated = item.mergeValidationUpdate(
      DetectionItem(
        filename: item.filename,
        path: item.path,
        fileType: item.fileType,
        dateTaken: item.dateTaken,
        modifiedAt: item.modifiedAt,
        width: item.width,
        height: item.height,
        species: nextSpecies,
        confidence: item.confidence,
        detectionBoxes: item.detectionBoxes,
        detectionData: nextData,
        validated: action == 'unverified' ? null : true,
      ),
    );
    return updated;
  }

  Future<DetectionItem> _markItem(
    DetectionItem item,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final updated = _markCopy(item, action, speciesName: speciesName);
    setState(() {
      _items = [
        for (final current in _items)
          if (current.path == updated.path) updated else current,
      ];
    });
    return updated;
  }

  Future<List<DetectionItem>> _markItems(
    List<DetectionItem> items,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final updatedByPath = {
      for (final item in items)
        item.path: _markCopy(item, action, speciesName: speciesName),
    };
    setState(() {
      _items = [
        for (final current in _items) updatedByPath[current.path] ?? current,
      ];
    });
    return [for (final item in items) updatedByPath[item.path]!];
  }

  Future<void> _handleLoadMetadata(DetectionItem item) async {
    final callback = widget.onLoadMetadataAsync;
    if (callback == null) return;
    await callback(item, (updated) {
      if (!mounted) return;
      setState(() {
        _items = [
          for (final current in _items)
            if (current.path == updated.path)
              widget.hydrateMetadata
                  ? current.mergePreviewMetadata(updated)
                  : updated
            else
              current,
        ];
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: widget.apiClient,
          classificationModelPath: widget.classificationModelPath,
          inputPath: _inputPath,
          rootInputPath: _inputPath,
          rootItems: _items,
          exportFormatMode: 'ask',
          items: _items,
          loading: _loading,
          refreshVersion: _refreshVersion,
          speciesTypes: const <String, String>{'豹猫': '兽类', '野猪': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: widget.autoGroup,
          collapseGroups: true,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: _burstSize,
          autoGroupGapSeconds: 1800,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: widget.quickMarkSpecies,
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
          favoritePhotoExportMode: favoritePhotoExportAsk,
          emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
          onRefresh: () async {},
          onLoadMetadata: _handleLoadMetadata,
          onOpenExternal: (_) {},
          onMarkItem: _markItem,
          onMarkItems: _markItems,
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
    );
  }
}
