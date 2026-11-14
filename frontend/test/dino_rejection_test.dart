import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/utils/dino_rejection.dart';
import 'package:neri_flutter/src/utils/detection_box_label.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';
import 'package:neri_flutter/src/widgets/dino_rejection_scores.dart';

void main() {
  testWidgets('frame evidence and shared experiment gate stay distinct', (
    tester,
  ) async {
    const box = DetectionBox(
      species: '岩羊',
      confidence: 0.426,
      bbox: [0, 0, 1, 1],
      candidates: [
        {
          'source': 'dinov3_ridge_within',
          'gate_mode': 'sequence',
          'inference_recipe': 'experiment_full',
          'confidence_kind': 'sigmoid_frame_class_margin',
          'frame_classification_confidence': 0.426,
          'accepted': true,
          'sequence_known_score': 3.0,
          'sequence_threshold': 1.0,
          'sequence_accepted': true,
        },
      ],
    );
    expect(dinoConfidenceLabel(box), '分类证据');
    expect(detectionBoxLabel(box, 0), '#1 岩羊 分类0.43');
    expect(dinoFinalBox(box).species, '岩羊');
    expect(dinoRejectionInfo(box).finalRejected, isFalse);
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: DinoRejectionScores(boxes: [box])),
      ),
    );
    expect(find.text('分类证据 0.426'), findsOneWidget);
    expect(find.text('实验事件 3.000 ≥ 1.000'), findsOneWidget);
  });

  test(
    'old experiment event confidences are not labelled as class evidence',
    () {
      const box = DetectionBox(
        species: '岩羊',
        confidence: 0.88,
        bbox: [0, 0, 1, 1],
        candidates: [
          {
            'source': 'dinov3_ridge_within',
            'gate_mode': 'sequence',
            'confidence_kind': 'sigmoid_gate_margin',
          },
        ],
      );
      expect(dinoConfidenceLabel(box), '事件接纳');
      expect(detectionBoxLabel(box, 1), '#2 岩羊 事件0.88');
      expect(dinoRejectionInfo(box).classificationConfidence, isNull);
    },
  );
  testWidgets('experiment shows its event gate without a synthetic box gate', (
    tester,
  ) async {
    const box = DetectionBox(
      species: '岩羊',
      bbox: [0, 0, 1, 1],
      candidates: [
        {
          'source': 'dinov3_ridge_within',
          'gate_mode': 'sequence',
          'inference_recipe': 'experiment_full',
          'accepted': true,
          'sequence_known_score': 0.8,
          'sequence_threshold': 0.2,
          'sequence_accepted': true,
        },
      ],
    );
    expect(dinoRejectionInfo(box).isExperimentSingleGate, isTrue);
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: DinoRejectionScores(boxes: [box])),
      ),
    );
    expect(find.text('实验事件 0.800 ≥ 0.200'), findsOneWidget);
    expect(find.textContaining('单框 —'), findsNothing);
    expect(find.textContaining('旧版门控'), findsNothing);
  });
  late Directory directory;
  late String imagePath;

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('neri_dino_rejection_');
    imagePath = '${directory.path}/bird.png';
    await File(imagePath).writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
  });

  tearDown(() async {
    final temporaryRoot = Directory.systemTemp.resolveSymbolicLinksSync();
    final target = directory.resolveSymbolicLinksSync();
    expect(
      target.startsWith(
        '$temporaryRoot${Platform.pathSeparator}neri_dino_rejection_',
      ),
      isTrue,
    );
    // Windows image decoding can briefly retain a file after unmounting.
    for (var attempt = 0; ; attempt++) {
      try {
        await directory.delete(recursive: true);
        return;
      } on FileSystemException catch (error) {
        if (error.osError?.errorCode != 32 || attempt >= 5) rethrow;
        await Future<void>.delayed(const Duration(milliseconds: 50));
      }
    }
  });

  test(
    'dinoFinalBox preserves image_label override even when model rejected',
    () {
      const box = DetectionBox(
        species: '岩松鼠',
        confidence: 0.12,
        bbox: [0.1, 0.1, 0.6, 0.6],
        feedbackStatus: 'image_label',
        candidates: [
          {
            'name': 'Unknown',
            'conf': 0.12,
            'accepted': false,
            'box_accepted': false,
            'sequence_accepted': false,
            'box_known_score': -1.400,
            'box_threshold': -0.993,
            'sequence_known_score': -1.200,
            'sequence_threshold': -0.351,
          },
        ],
      );

      expect(dinoFinalBox(box).species, '岩松鼠');
      final info = dinoRejectionInfo(box);
      expect(info.isImageLabeled, isTrue);
      expect(info.finalRejected, isFalse);
      expect(info.finalLabel, '整图标注 · 岩松鼠');
    },
  );

  test(
    'legacy scores are supported and sigmoid confidences are never used as gate scores',
    () {
      const legacySingleGateBox = DetectionBox(
        species: 'Unknown',
        confidence: 0.28,
        bbox: [0.1, 0.1, 0.5, 0.5],
        observationId: 'dinov3:legacy-1',
        candidates: [
          {
            'name': 'Unknown',
            'conf': 0.28,
            'known_score': -1.250,
            'score_threshold': -0.450,
            'accepted': false,
          },
        ],
      );
      final legacyInfo = dinoRejectionInfo(legacySingleGateBox);
      expect(legacyInfo.isLegacySingleGate, isTrue);
      expect(legacyInfo.legacyScore, -1.250);
      expect(legacyInfo.legacyThreshold, -0.450);
      expect(legacyInfo.legacyPassed, isFalse);
      expect(legacyInfo.finalRejected, isTrue);

      const sigmoidOnlyBox = DetectionBox(
        species: '岩松鼠',
        confidence: 0.785,
        bbox: [0.1, 0.1, 0.5, 0.5],
        observationId: 'dinov3:sigmoid-only',
        candidates: [
          {
            'name': '岩松鼠',
            'conf': 0.785,
            'box_confidence': 0.890,
            'sequence_confidence': 0.770,
            'within_score': 0.420,
          },
        ],
      );
      final sigmoidInfo = dinoRejectionInfo(sigmoidOnlyBox);
      expect(sigmoidInfo.boxScore, isNull);
      expect(sigmoidInfo.sequenceScore, isNull);
      expect(sigmoidInfo.legacyScore, isNull);
    },
  );

  test(
    'detects inconsistency when comparisons contradict saved accepted flag',
    () {
      const inconsistentBox = DetectionBox(
        species: 'Unknown',
        confidence: 0.41,
        bbox: [0.1, 0.1, 0.6, 0.6],
        candidates: [
          {
            'name': 'Unknown',
            'conf': 0.41,
            'accepted': false,
            'box_known_score': 0.091,
            'box_threshold': -0.993,
            'sequence_known_score': -0.092,
            'sequence_threshold': -0.351,
            'source': 'dinov3_ridge_within',
          },
        ],
      );

      expect(dinoFinalBox(inconsistentBox).species, 'Unknown');
      final info = dinoRejectionInfo(inconsistentBox);
      expect(info.hasInconsistentRecord, isTrue);
      expect(info.finalRejected, isTrue);
    },
  );

  testWidgets(
    'DinoRejectionScores explains comparisons, legacy fallback, inconsistency, and fits narrow width',
    (tester) async {
      tester.view.physicalSize = const Size(360, 800);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: Padding(
                padding: EdgeInsets.all(12),
                child: DinoRejectionScores(
                  boxes: [
                    DetectionBox(
                      species: '岩松鼠',
                      confidence: 0.80,
                      bbox: [0.1, 0.1, 0.4, 0.4],
                      candidates: [
                        {
                          'name': '岩松鼠',
                          'conf': 0.80,
                          'accepted': true,
                          'box_accepted': true,
                          'sequence_accepted': true,
                          'box_known_score': 0.091,
                          'box_threshold': -0.993,
                          'sequence_known_score': -0.092,
                          'sequence_threshold': -0.351,
                          'source': 'dinov3_ridge_within',
                        },
                      ],
                    ),
                    DetectionBox(
                      species: 'Unknown',
                      confidence: 0.31,
                      bbox: [0.2, 0.2, 0.5, 0.5],
                      observationId: 'dinov3:legacy',
                      candidates: [
                        {
                          'name': 'Unknown',
                          'conf': 0.31,
                          'known_score': -0.800,
                          'score_threshold': -0.350,
                          'accepted': false,
                        },
                      ],
                    ),
                    DetectionBox(
                      species: 'Unknown',
                      confidence: 0.72,
                      bbox: [0.3, 0.3, 0.6, 0.6],
                      candidates: [
                        {
                          'name': 'Unknown',
                          'conf': 0.72,
                          'accepted': false,
                          'box_known_score': 0.250,
                          'box_threshold': -0.500,
                          'sequence_known_score': 0.110,
                          'sequence_threshold': -0.300,
                          'source': 'dinov3_ridge_within',
                        },
                      ],
                    ),
                    DetectionBox(
                      species: '赤狐',
                      confidence: 0.90,
                      bbox: [0.4, 0.4, 0.7, 0.7],
                      observationId: 'dinov3:sigmoid-only',
                      candidates: [
                        {
                          'name': '赤狐',
                          'conf': 0.90,
                          'box_confidence': 0.90,
                          'sequence_confidence': 0.90,
                        },
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(tester.takeException(), isNull);
      expect(
        tester
            .getSize(find.byKey(const ValueKey('dino-rejection-single-line')))
            .height,
        lessThanOrEqualTo(40),
      );
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Tooltip && widget.message?.contains('本次识别时记录') == true,
        ),
        findsOneWidget,
      );
      expect(find.text('#1 最终接纳 · 岩松鼠'), findsOneWidget);
      expect(find.byTooltip('单框 0.091 ≥ 门限 -0.993（通过）'), findsOneWidget);
      expect(find.byTooltip('序列 -0.092 ≥ 门限 -0.351（通过）'), findsOneWidget);
      expect(find.byTooltip('旧版门控 -0.800 < 门限 -0.350（未通过）'), findsOneWidget);
      expect(find.byTooltip('记录不一致，建议重新识别'), findsOneWidget);
      expect(find.text('0.900'), findsNothing);
      expect(find.byTooltip('单框 未记录'), findsOneWidget);
      expect(find.byTooltip('序列 未记录'), findsOneWidget);
    },
  );

  testWidgets(
    'never emits false math operator when saved accepted conflicts with numeric comparison, and avoids threshold fallback for missing box_threshold',
    (tester) async {
      const boxWithConflict = DetectionBox(
        species: 'Unknown',
        confidence: 0.41,
        bbox: [0.1, 0.1, 0.5, 0.5],
        candidates: [
          {'name': 'other', 'accepted': true},
          {
            'name': '岩松鼠',
            'box_known_score': 0.091,
            'box_threshold': -0.993,
            'box_accepted': false,
            'sequence_known_score': -0.092,
            'sequence_threshold': -0.351,
            'sequence_accepted': true,
            'accepted': false,
            'source': 'dinov3_ridge_within',
          },
        ],
      );
      const missingBoxThreshold = DetectionBox(
        species: '岩松鼠',
        confidence: 0.80,
        bbox: [0.2, 0.2, 0.6, 0.6],
        candidates: [
          {
            'name': '岩松鼠',
            'box_known_score': 0.150,
            'sequence_known_score': 0.250,
            'threshold': -0.351,
            'source': 'dinov3_ridge_within',
          },
        ],
      );

      final infoMissing = dinoRejectionInfo(missingBoxThreshold);
      expect(infoMissing.boxScore, 0.150);
      expect(infoMissing.boxThreshold, isNull);
      expect(infoMissing.sequenceThreshold, -0.351);

      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: DinoRejectionScores(
              boxes: [boxWithConflict, missingBoxThreshold],
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('0.091 <'), findsNothing);
      expect(
        find.byTooltip('单框 0.091 ≥ 门限 -0.993（记录未通过，与比较不一致）'),
        findsOneWidget,
      );
      expect(find.byTooltip('记录不一致，建议重新识别'), findsOneWidget);
      expect(find.byTooltip('单框 0.150 / 门限 未记录'), findsOneWidget);
    },
  );

  test(
    'findRawDinoBox matches by observation_id or bbox after invalid box removal',
    () {
      final rawBoxes = <Map<String, dynamic>>[
        {'物种': '无效框', '边界框': <double>[], 'box_known_score': -9.999},
        {
          '物种': '岩松鼠',
          '边界框': [0.2, 0.2, 0.6, 0.6],
          'observation_id': 'dinov3:obs-2',
          'box_known_score': 0.444,
          'box_threshold': -0.993,
        },
      ];
      const shiftedBox = DetectionBox(
        species: '岩松鼠',
        bbox: [0.2, 0.2, 0.6, 0.6],
        observationId: 'dinov3:obs-2',
      );
      const unmatchedBox = DetectionBox(
        species: '赤狐',
        bbox: [0.7, 0.7, 0.9, 0.9],
      );

      final matched = findRawDinoBox(shiftedBox, 0, rawBoxes);
      expect(matched?['box_known_score'], 0.444);
      expect(findRawDinoBox(unmatchedBox, 0, rawBoxes), isNull);
    },
  );

  testWidgets(
    'DINO preview ignores stale species filter, shows all boxes including Unknown, and preserves manual count',
    (tester) async {
      tester.view.physicalSize = const Size(480, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final item = DetectionItem(
        filename: 'bird.png',
        path: imagePath,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['岩松鼠'],
        confidence: 0.5,
        detectionData: const {
          '物种名称': '岩松鼠',
          '物种数量': '4',
          '最低置信度': '人工校验',
          'image_annotation': {'species': '岩松鼠', 'count': '4'},
          'sequence_completed': true,
        },
        detectionBoxes: const [
          DetectionBox(
            species: '岩松鼠',
            confidence: 0.8,
            bbox: [0.1, 0.1, 0.4, 0.4],
            feedbackStatus: 'image_label',
            candidates: [
              {
                'name': 'Unknown',
                'conf': 0.2,
                'accepted': false,
                'box_known_score': -1.2,
                'box_threshold': -0.993,
                'sequence_known_score': -1.1,
                'sequence_threshold': -0.351,
                'source': 'dinov3_ridge_within',
              },
            ],
          ),
          DetectionBox(
            species: 'Unknown',
            confidence: 0.2,
            bbox: [0.5, 0.5, 0.9, 0.9],
            candidates: [
              {
                'name': 'Unknown',
                'conf': 0.2,
                'accepted': false,
                'box_known_score': -1.4,
                'box_threshold': -0.993,
                'sequence_known_score': -1.3,
                'sequence_threshold': -0.351,
                'source': 'dinov3_ridge_within',
              },
            ],
          ),
        ],
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PreviewScreen(
              inputPath: directory.path,
              items: [item],
              selectedIndex: 0,
              selectedItem: item,
              speciesTypes: const {},
              useCombinedConfidence: false,
              showDetections: true,
              onShowDetectionsChanged: (_) {},
              selectedSpeciesFilter: '岩松鼠',
              onSpeciesFilterChanged: (_) {},
              confidenceThreshold: 0.95,
              onConfidenceThresholdChanged: (_) {},
              detecting: false,
              loading: false,
              onDetectCurrentImage: (_) {},
              onSelected: (_, _) {},
              onLoadMetadata: (_) async {},
              onRefresh: () async {},
              onOpenExternal: (_) {},
            ),
          ),
        ),
      );
      for (var i = 0; i < 60 && find.byType(Image).evaluate().isEmpty; i++) {
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump();
      }
      await tester.pumpAndSettle();

      expect(tester.takeException(), isNull);
      expect(find.text('显示物种'), findsNothing);
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes.map((b) => b.species).toList(), [
        '岩松鼠',
        'Unknown',
      ]);
      expect(find.textContaining('数量: 4'), findsOneWidget);
      expect(find.textContaining('整图标注 · 岩松鼠'), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'switching from YOLO species filter to DINO item in validation shows all boxes without dropdown',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final api = NeriApiClient(
        httpClient: MockClient(
          (_) async => http.Response(
            '{}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          ),
        ),
      );
      addTearDown(api.close);

      final yoloFile = File('${directory.path}/a_yolo.png');
      yoloFile.writeAsBytesSync(File(imagePath).readAsBytesSync());
      final dinoFile = File('${directory.path}/b_dino.png');
      dinoFile.writeAsBytesSync(File(imagePath).readAsBytesSync());

      final yoloItem = DetectionItem(
        filename: 'a_yolo.png',
        path: yoloFile.path,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['岩松鼠'],
        confidence: 0.9,
        detectionBoxes: const [
          DetectionBox(
            species: '岩松鼠',
            confidence: 0.9,
            bbox: [0.1, 0.1, 0.5, 0.5],
            candidates: [
              {'name': '岩松鼠', 'conf': 0.9},
            ],
          ),
        ],
      );
      final dinoItem = DetectionItem(
        filename: 'b_dino.png',
        path: dinoFile.path,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['岩松鼠', 'Unknown'],
        confidence: 0.5,
        detectionData: const {'sequence_completed': true},
        detectionBoxes: const [
          DetectionBox(
            species: '岩松鼠',
            confidence: 0.8,
            bbox: [0.1, 0.1, 0.4, 0.4],
            candidates: [
              {
                'name': '岩松鼠',
                'conf': 0.8,
                'accepted': true,
                'box_known_score': 0.5,
                'box_threshold': -0.993,
                'sequence_known_score': 0.4,
                'sequence_threshold': -0.351,
                'source': 'dinov3_ridge_within',
              },
            ],
          ),
          DetectionBox(
            species: 'Unknown',
            confidence: 0.2,
            bbox: [0.5, 0.5, 0.9, 0.9],
            candidates: [
              {
                'name': 'Unknown',
                'conf': 0.2,
                'accepted': false,
                'box_known_score': -1.4,
                'box_threshold': -0.993,
                'sequence_known_score': -1.2,
                'sequence_threshold': -0.351,
                'source': 'dinov3_ridge_within',
              },
            ],
          ),
        ],
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SpeciesValidationScreen(
              apiClient: api,
              inputPath: directory.path,
              rootInputPath: directory.path,
              rootItems: [yoloItem, dinoItem],
              exportFormatMode: 'ask',
              items: [yoloItem, dinoItem],
              loading: false,
              refreshVersion: 0,
              speciesTypes: const {},
              useCombinedConfidence: false,
              minFrameRatio: 0,
              autoGroup: false,
              collapseGroups: false,
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
              onExportFormatModeChanged: (_) async {},
              onEmptyPhotoDeleteModeChanged: (_) async {},
              onAutoGroupInferredBurstSizeChanged: (_) {},
            ),
          ),
        ),
      );
      for (var i = 0; i < 60 && find.byType(Image).evaluate().isEmpty; i++) {
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump();
      }
      expect(find.byType(Image), findsOneWidget);
      await _waitForDecodedImage(tester);

      // In YOLO item, select '岩松鼠' in the species dropdown.
      await tester.tap(find.byType(DropdownMenu<String>));
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(find.text('岩松鼠').last);
      await tester.pump(const Duration(milliseconds: 100));

      // Switch to DINO item.
      // Rejected targets without a saved clustering snapshot have their own
      // unclustered bucket alongside the shared known-species bucket.
      await tester.tap(find.textContaining('拒识（未聚类）'));
      for (
        var i = 0;
        i < 60 &&
            find
                .byKey(const ValueKey('dino-rejection-single-line'))
                .evaluate()
                .isEmpty;
        i++
      ) {
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump();
      }
      await _waitForDecodedImage(tester);

      expect(find.text('显示物种'), findsNothing);
      expect(find.byType(DropdownMenu<String>), findsNothing);
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes.map((b) => b.species).toList(), [
        '岩松鼠',
        'Unknown',
      ]);
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump();
    },
  );

  test(
    'dinoValidationLearningFeedbackSummary reads backend message and deduplicates batch reasons',
    () {
      const eligibleMultiBox = DetectionItem(
        filename: 'a.jpg',
        path: 'a.jpg',
        fileType: 'jpg',
        detectionBoxes: [
          DetectionBox(species: '赤狐', bbox: [0.1, 0.1, 0.3, 0.3]),
          DetectionBox(species: '赤狐', bbox: [0.4, 0.4, 0.8, 0.8]),
        ],
        detectionData: {
          'learning_feedback': {
            'status': 'eligible',
            'box_count': 2,
            'manual_count': 2,
            'message': '已按人工单物种标注更新检测框，符合数量条件的裁图将自动学习。',
          },
        },
      );
      const countExceeded = DetectionItem(
        filename: 'b.jpg',
        path: 'b.jpg',
        fileType: 'jpg',
        detectionData: {
          'learning_feedback': {
            'status': 'count_exceeded',
            'box_count': 3,
            'manual_count': 1,
            'message': '已更新框上的物种；检测框数超过人工数量，本图不自动学习，请逐框核对。',
          },
        },
      );

      expect(
        dinoValidationLearningFeedbackSummary([eligibleMultiBox]),
        '已按人工单物种标注更新检测框，符合数量条件的裁图将自动学习。',
      );
      expect(
        dinoValidationLearningFeedbackSummary([
          eligibleMultiBox,
          eligibleMultiBox,
        ]),
        '已按人工单物种标注更新检测框，符合数量条件的裁图将自动学习。',
      );
      expect(
        dinoValidationLearningFeedbackSummary([
          eligibleMultiBox,
          eligibleMultiBox,
          countExceeded,
        ]),
        '2 个文件：已按人工单物种标注更新检测框，符合数量条件的裁图将自动学习。；1 个文件：已更新框上的物种；检测框数超过人工数量，本图不自动学习，请逐框核对。',
      );
      expect(
        dinoValidationLearningFeedbackSummary(const [
          DetectionItem(
            filename: 'c.jpg',
            path: 'c.jpg',
            fileType: 'jpg',
            detectionBoxes: [
              DetectionBox(species: '赤狐', bbox: [0.1, 0.1, 0.3, 0.3]),
              DetectionBox(species: '赤狐', bbox: [0.4, 0.4, 0.8, 0.8]),
            ],
          ),
        ]),
        '',
      );
    },
  );

  test(
    'dinoFeedbackPanelTitle prefers human species for confirmed and image_label boxes',
    () {
      const confirmedBox = DetectionBox(
        species: '赤狐',
        predictedSpecies: '岩松鼠',
        feedbackStatus: 'confirmed',
        bbox: [0.1, 0.1, 0.4, 0.4],
      );
      const imageLabelBox = DetectionBox(
        species: '狍',
        predictedSpecies: '岩松鼠',
        feedbackStatus: 'image_label',
        bbox: [0.2, 0.2, 0.5, 0.5],
      );
      const modelBox = DetectionBox(
        species: 'Unknown',
        predictedSpecies: '岩松鼠',
        bbox: [0.3, 0.3, 0.6, 0.6],
      );

      expect(
        dinoFeedbackPanelTitle(confirmedBox, [confirmedBox]),
        '#1 赤狐 · 检测框校验',
      );
      expect(
        dinoFeedbackPanelTitle(imageLabelBox, [confirmedBox, imageLabelBox]),
        '#2 狍 · 检测框校验',
      );
      expect(dinoFeedbackPanelTitle(modelBox, [modelBox]), '#1 岩松鼠 · 检测框校验');
    },
  );
}

Future<void> _waitForDecodedImage(WidgetTester tester) async {
  for (var attempt = 0; attempt < 100; attempt++) {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump();
    if (tester
        .widgetList<RawImage>(find.byType(RawImage))
        .any((widget) => widget.image != null))
      return;
  }
  fail('The selected photo did not finish decoding');
}
