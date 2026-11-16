import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  late Directory tempDir;
  late NeriApiClient apiClient;
  late List<DetectionItem> items;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('neri_group_expansion_');
    final pngBytes = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
    );
    final paths = <String>[];
    for (var index = 0; index < 4; index++) {
      final file = File('${tempDir.path}/ECSP02${77 + index}.JPG');
      await file.writeAsBytes(pngBytes);
      paths.add(file.path);
    }

    apiClient = NeriApiClient(
      httpClient: MockClient(
        (_) async => http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        ),
      ),
    );

    items = <DetectionItem>[
      for (var index = 0; index < paths.length; index++)
        DetectionItem(
          filename: 'ECSP02${77 + index}.JPG',
          path: paths[index],
          fileType: 'jpg',
          modifiedAt: '2026-09-11T12:00:0${index}Z',
          width: 10,
          height: 10,
          species: const <String>['豹猫'],
          confidence: 0.95,
          detectionData: const <String, dynamic>{'物种名称': '豹猫'},
        ),
    ];
  });

  tearDown(() async {
    apiClient.close();
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  List<DetectionItem> selectionItems({int groupSize = 4}) {
    return [
      for (var group = 0; group < 4; group++)
        for (var member = 0; member < groupSize; member++)
          DetectionItem(
            filename: 'group-$group-$member.jpg',
            path: '${tempDir.path}/group-$group-$member.jpg',
            fileType: 'jpg',
            dateTaken: '2026-09-${10 + group}T12:00:0${member}Z',
            width: 10,
            height: 10,
            species: [group == 2 ? '野猪' : '豹猫'],
            confidence: 0.95,
            detectionData: {'物种名称': group == 2 ? '野猪' : '豹猫'},
          ),
    ];
  }

  void setSelectionTestSize(WidgetTester tester) {
    tester.view.physicalSize = const Size(1600, 1800);
    tester.view.devicePixelRatio = 1;
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

  Future<void> pumpSelection(WidgetTester tester) async {
    await tester.pump();
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump(const Duration(milliseconds: 300));
  }

  Future<void> selectFixtureSpecies(WidgetTester tester) async {
    await pumpSelection(tester);
    await tester.tap(
      find.descendant(
        of: find.byKey(const ValueKey('validation-species-list')),
        matching: find.text('豹猫'),
      ),
    );
    await pumpSelection(tester);
  }

  Finder fileTiles() => find.byWidgetPredicate(
    (widget) =>
        widget is ListTile &&
        widget.subtitle is Text &&
        ((widget.subtitle! as Text).data?.startsWith('第 ') ?? false),
  );

  Finder groupTile(int ordinal) => find.byWidgetPredicate(
    (widget) =>
        widget is ListTile &&
        widget.subtitle is Text &&
        ((widget.subtitle! as Text).data?.startsWith('第 $ordinal/4 组') ??
            false),
  );

  testWidgets(
    'skipped video inherits its photo group without becoming a detection',
    (tester) async {
      setSelectionTestSize(tester);
      // Grouping uses persisted metadata; native image decoding is outside this test.
      await tester.runAsync(() async {
        for (final item in items) {
          await File(item.path).delete();
        }
      });
      final video = DetectionItem(
        filename: 'skipped.mp4',
        path: '${tempDir.path}/skipped.mp4',
        fileType: 'mp4',
        modifiedAt: '2026-09-11T12:00:01Z',
      );
      await tester.pumpWidget(
        _GroupExpansionHarness(
          apiClient: apiClient,
          initialItems: [items.first, video],
          reorderMetadata: false,
          videoMode: 'skip',
        ),
      );
      await pumpSelection(tester);
      final list = find.byKey(const ValueKey('validation-species-list'));
      expect(
        find.descendant(of: list, matching: find.text('豹猫')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: list, matching: find.text('未检测')),
        findsNothing,
      );
      await tester.tap(find.descendant(of: list, matching: find.text('豹猫')));
      await pumpSelection(tester);
      final expand = find.byTooltip('展开组内文件');
      if (expand.evaluate().isNotEmpty) {
        await tester.tap(expand.first);
        await pumpSelection(tester);
      }
      expect(video.species, isEmpty);
      expect(video.detectionData, isEmpty);
      expect(tester.takeException(), isNull);
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 50)),
      );
      await tester.pump();
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.runAsync(() async {
        imageCache.clear();
        imageCache.clearLiveImages();
        await Future<void>.delayed(const Duration(milliseconds: 100));
      });
      await tester.pump();
    },
  );

  testWidgets('video-only skipped bucket changes when mode changes', (
    tester,
  ) async {
    setSelectionTestSize(tester);
    // Grouping uses persisted metadata; native image decoding is outside this test.
    await tester.runAsync(() async {
      for (final item in items) {
        await File(item.path).delete();
      }
    });
    final video = DetectionItem(
      filename: 'skipped.mp4',
      path: '${tempDir.path}/skipped.mp4',
      fileType: 'mp4',
    );
    final key = GlobalKey<_GroupExpansionHarnessState>();
    Future<void> show(String mode) async {
      await tester.pumpWidget(
        _GroupExpansionHarness(
          key: key,
          apiClient: apiClient,
          initialItems: [items.first, video],
          reorderMetadata: false,
          autoGroup: false,
          videoMode: mode,
        ),
      );
      await pumpSelection(tester);
    }

    await show('skip');
    final list = find.byKey(const ValueKey('validation-species-list'));
    expect(
      find.descendant(of: list, matching: find.text('已跳过视频')),
      findsOneWidget,
    );
    expect(find.descendant(of: list, matching: find.text('未检测')), findsNothing);
    await show('fast');
    expect(
      find.descendant(of: list, matching: find.text('未检测')),
      findsOneWidget,
    );
    expect(
      find.descendant(of: list, matching: find.text('已跳过视频')),
      findsNothing,
    );
    expect(tester.takeException(), isNull);
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 50)),
    );
    await tester.pump();
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.runAsync(() async {
      imageCache.clear();
      imageCache.clearLiveImages();
      await Future<void>.delayed(const Duration(milliseconds: 100));
    });
    await tester.pump();
  });

  testWidgets('skip mode preserves historical video detections and errors', (
    tester,
  ) async {
    setSelectionTestSize(tester);
    // Grouping uses persisted metadata; native image decoding is outside this test.
    await tester.runAsync(() async {
      for (final item in items) {
        await File(item.path).delete();
      }
    });
    final video = DetectionItem(
      filename: 'recognized.mp4',
      path: '${tempDir.path}/recognized.mp4',
      fileType: 'mp4',
      species: const ['野猪'],
      detectionData: const {'物种名称': '野猪'},
    );
    final failed = DetectionItem(
      filename: 'failed.mp4',
      path: '${tempDir.path}/failed.mp4',
      fileType: 'mp4',
      error: 'decode failure',
    );
    await tester.pumpWidget(
      _GroupExpansionHarness(
        apiClient: apiClient,
        initialItems: [items.first, video, failed],
        reorderMetadata: false,
        autoGroup: false,
        videoMode: 'skip',
      ),
    );
    await pumpSelection(tester);
    final list = find.byKey(const ValueKey('validation-species-list'));
    expect(
      find.descendant(of: list, matching: find.text('野猪')),
      findsOneWidget,
    );
    expect(
      find.descendant(of: list, matching: find.text('错误')),
      findsOneWidget,
    );
    expect(
      find.descendant(of: list, matching: find.text('已跳过视频')),
      findsNothing,
    );
    expect(tester.takeException(), isNull);
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 50)),
    );
    await tester.pump();
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.runAsync(() async {
      imageCache.clear();
      imageCache.clearLiveImages();
      await Future<void>.delayed(const Duration(milliseconds: 100));
    });
    await tester.pump();
  });

  testWidgets('group stripes follow visible groups and survive expansion', (
    tester,
  ) async {
    setSelectionTestSize(tester);
    await tester.pumpWidget(
      _GroupExpansionHarness(
        apiClient: apiClient,
        initialItems: selectionItems(),
        reorderMetadata: false,
      ),
    );
    await selectFixtureSpecies(tester);

    final tiles = tester.widgetList<ListTile>(fileTiles()).toList();
    expect(tiles.length, 3);
    expect(tiles[0].tileColor, isNull);
    expect(tiles[1].tileColor, isNotNull);
    expect(tiles[2].tileColor, tiles[0].tileColor);

    await tester.tap(
      find.descendant(of: groupTile(2), matching: find.byType(IconButton)),
    );
    await pumpSelection(tester);
    final expandedTiles = tester.widgetList<ListTile>(groupTile(2)).toList();
    expect(expandedTiles.length, 5);
    expect(expandedTiles.map((tile) => tile.tileColor).toSet(), {
      tiles[1].tileColor,
    });
    expect(tester.widget<ListTile>(groupTile(4)).tileColor, isNull);
    expect(tester.takeException(), isNull);
  });

  for (final autoGroup in [false, true]) {
    for (final modifier in [
      LogicalKeyboardKey.controlLeft,
      LogicalKeyboardKey.shiftLeft,
      LogicalKeyboardKey.metaLeft,
    ]) {
      testWidgets(
        'multi-selection reads held ${modifier.keyLabel} after focus changes ($autoGroup)',
        (tester) async {
          setSelectionTestSize(tester);
          final outsideFocus = FocusNode();
          addTearDown(() async {
            await tester.pumpWidget(const SizedBox.shrink());
            await tester.pump();
            outsideFocus.dispose();
          });
          final harness = GlobalKey<_GroupExpansionHarnessState>();
          final fixtures = selectionItems(groupSize: autoGroup ? 4 : 1);
          await tester.pumpWidget(
            Focus(
              focusNode: outsideFocus,
              child: _GroupExpansionHarness(
                key: harness,
                apiClient: apiClient,
                initialItems: fixtures,
                autoGroup: autoGroup,
                reorderMetadata: false,
              ),
            ),
          );
          await selectFixtureSpecies(tester);
          final tileFinder = autoGroup
              ? groupTile(1)
              : find.widgetWithText(ListTile, 'group-0-0.jpg');
          await tester.tap(tileFinder);
          await pumpSelection(tester);

          // The key is pressed outside the validation page's focus subtree.
          outsideFocus.requestFocus();
          await tester.pump();
          await tester.sendKeyDownEvent(modifier);
          final target = autoGroup
              ? groupTile(4)
              : find.widgetWithText(ListTile, 'group-3-0.jpg');
          await tester.tap(target);
          await pumpSelection(tester);
          await tester.sendKeyUpEvent(modifier);
          final photoTiles = autoGroup
              ? fileTiles()
              : find.byWidgetPredicate(
                  (widget) =>
                      widget is ListTile &&
                      widget.title is Text &&
                      ((widget.title! as Text).data?.startsWith('group-') ??
                          false),
                );
          final isRange = modifier == LogicalKeyboardKey.shiftLeft;
          expect(
            tester
                .widgetList<ListTile>(photoTiles)
                .where((t) => t.selected)
                .length,
            isRange ? 3 : 2,
          );

          await tester.tap(find.text('正确').first);
          await pumpSelection(tester);
          final selectedGroups = isRange ? {0, 1, 3} : {0, 3};
          expect(harness.currentState!._markedPaths, {
            for (final item in fixtures)
              if (selectedGroups.any(
                (group) => item.filename.startsWith('group-$group-'),
              ))
                item.path,
          });
          expect(tester.takeException(), isNull);
        },
      );
    }
  }

  testWidgets('Ctrl click deselects the active group', (tester) async {
    setSelectionTestSize(tester);
    await tester.pumpWidget(
      _GroupExpansionHarness(
        apiClient: apiClient,
        initialItems: selectionItems(),
        reorderMetadata: false,
      ),
    );
    await selectFixtureSpecies(tester);
    await tester.tap(groupTile(1));
    await pumpSelection(tester);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.tap(groupTile(2));
    await pumpSelection(tester);
    expect(tester.widget<ListTile>(groupTile(2)).selected, isTrue);
    await tester.tap(groupTile(2));
    await pumpSelection(tester);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    expect(tester.widget<ListTile>(groupTile(2)).selected, isFalse);
    expect(tester.widget<ListTile>(groupTile(1)).selected, isTrue);
    await tester.tap(groupTile(4));
    await pumpSelection(tester);
    expect(tester.widget<ListTile>(groupTile(1)).selected, isFalse);
    expect(tester.widget<ListTile>(groupTile(4)).selected, isTrue);
    expect(tester.takeException(), isNull);
  });

  testWidgets('expanded group stays open when metadata reorders its members', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      _GroupExpansionHarness(apiClient: apiClient, initialItems: items),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    final expandButton = find.byTooltip('展开分组');
    expect(expandButton, findsOneWidget);

    await tester.tap(expandButton);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    // Expanding loads the representative item's full metadata. The harness
    // changes its timestamp so the same four files are reordered and the
    // automatic grouping cache is rebuilt. That rebuild must not discard the
    // user's explicit expansion state.
    expect(find.byTooltip('折叠分组'), findsOneWidget);
    expect(find.text('ECSP0277.JPG'), findsWidgets);
    expect(find.text('ECSP0278.JPG'), findsWidgets);
    expect(find.text('ECSP0279.JPG'), findsWidgets);
    expect(find.text('ECSP0280.JPG'), findsWidgets);
  });

  for (final autoGroup in [false, true]) {
    for (final rejected in ['拒识/Unknown', 'Unknown']) {
      testWidgets(
        'species buckets follow review priority before and after marking ($autoGroup, $rejected)',
        (tester) async {
          tester.view.physicalSize = const Size(1600, 1600);
          tester.view.devicePixelRatio = 1;
          addTearDown(tester.view.resetPhysicalSize);
          addTearDown(tester.view.resetDevicePixelRatio);
          final png = File(items.first.path).readAsBytesSync();
          final labels = ['空', rejected, '豹猫,$rejected', '斑羚', '野猪', '空'];
          final fixtures = <DetectionItem>[];
          for (var index = 0; index < labels.length; index++) {
            final file = File('${tempDir.path}/priority-$index.jpg');
            file.writeAsBytesSync(png);
            fixtures.add(
              DetectionItem(
                filename: 'priority-$index.jpg',
                path: file.path,
                fileType: 'jpg',
                dateTaken: '2026-09-${10 + index}T12:00:00Z',
                width: 10,
                height: 10,
                species: labels[index].split(','),
                confidence: 0.95,
                detectionData: {'物种名称': labels[index]},
                validated: index >= 4,
              ),
            );
          }
          await tester.pumpWidget(
            _GroupExpansionHarness(
              apiClient: apiClient,
              initialItems: fixtures,
              autoGroup: autoGroup,
              reorderMetadata: false,
            ),
          );
          await tester.pump();
          await tester.pump(const Duration(milliseconds: 100));

          final speciesList = find.byKey(
            const ValueKey('validation-species-list'),
          );
          List<String> titles() => tester
              .widgetList<ListTile>(
                find.descendant(
                  of: speciesList,
                  matching: find.byType(ListTile),
                ),
              )
              .map((tile) => (tile.title! as Text).data!.replaceAll('、', ','))
              .toList();
          expect(titles(), ['斑羚', '豹猫,拒识（未分簇）', '拒识（未分簇）', '空', '空', '野猪']);
          await tester.tap(
            find.descendant(of: speciesList, matching: find.text('斑羚')),
          );
          await tester.pump();
          await tester.tap(find.text('正确').first);
          await tester.pump();
          await tester.pump(const Duration(milliseconds: 150));
          expect(titles(), ['豹猫,拒识（未分簇）', '拒识（未分簇）', '空', '斑羚', '空', '野猪']);
          expect(tester.takeException(), isNull);
          await tester.pump(const Duration(seconds: 5));
        },
      );
    }
  }

  for (final autoGroup in [false, true]) {
    testWidgets('rejected observations use stable cluster buckets ($autoGroup)', (
      tester,
    ) async {
      tester.view.physicalSize = const Size(1600, 2200);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      apiClient.close();
      var requests = 0;
      var learningRequests = 0;
      var undoRequests = 0;
      String? operationId;
      final fixtures = <DetectionItem>[];
      apiClient = NeriApiClient(
        httpClient: MockClient((request) async {
          final body = jsonDecode(request.body) as Map;
          if (request.url.path == '/api/dino/feedback/cluster') {
            learningRequests++;
            operationId = body['feedback_operation_id'] as String;
            final selected = (body['observation_ids'] as List)
                .cast<String>()
                .toSet();
            expect(selected, {'a', 'a2', 'mixed', 'reverse', 'both-a'});
            expect(body['cluster_id'], 'registry:fp:2:0');
            expect(body['species_name'], '岩羊');
            final updated = <Map<String, dynamic>>[];
            for (final item in fixtures) {
              if (!item.detectionBoxes.any(
                (box) => selected.contains(box.observationId),
              )) {
                continue;
              }
              final boxes = [
                for (final box in item.detectionBoxes)
                  {
                    'species': selected.contains(box.observationId)
                        ? '岩羊'
                        : box.species,
                    'bbox': box.bbox,
                    'confidence': selected.contains(box.observationId)
                        ? 0.01
                        : box.confidence,
                    if (selected.contains(box.observationId))
                      'candidates': [
                        {'name': '旧模型结果', 'conf': 0.01},
                      ],
                    'observation_id': box.observationId,
                    if (selected.contains(box.observationId))
                      'feedback_status': 'confirmed',
                  },
              ];
              final names = boxes
                  .map((box) => box['species'] as String)
                  .toSet()
                  .toList();
              updated.add({
                'filename': item.filename,
                'path': item.path,
                'file_type': item.fileType,
                'date_taken': item.dateTaken,
                'species': names,
                'detection_boxes': boxes,
                'detection_data': {'物种名称': names.join(',')},
                'validated': boxes.every(
                  (box) => box['feedback_status'] == 'confirmed',
                ),
              });
            }
            return http.Response(
              jsonEncode({
                'operation_id': operationId,
                'feature_events': 10,
                'learning': {
                  'status': 'confirmed',
                  'active': true,
                  'formal': true,
                  'needs_review': false,
                },
                'items': updated,
              }),
              200,
              headers: {'content-type': 'application/json; charset=utf-8'},
            );
          }
          if (request.url.path == '/api/dino/feedback/revert') {
            undoRequests++;
            expect(body['feedback_operation_id'], operationId);
            return http.Response(
              jsonEncode({
                'operation_id': operationId,
                'affected_species': ['岩羊'],
              }),
              200,
              headers: {'content-type': 'application/json; charset=utf-8'},
            );
          }
          expect(request.url.path, '/api/dino/feedback/rejection-clusters');
          requests++;
          final ids =
              (jsonDecode(request.body) as Map)['observation_ids'] as List;
          expect(
            ids,
            containsAll([
              'a',
              'a2',
              'b',
              'mixed',
              'reverse',
              'both-a',
              'both-b',
              'missing',
            ]),
          );
          return http.Response(
            jsonEncode({
              for (final id in ids.where((id) => id != 'missing'))
                id: {
                  'cluster_id':
                      'registry:fp:${id.toString().startsWith('noise') ? (id == 'noise-other' ? 21 : 20) : (id == 'b' || id == 'both-b' ? 10 : 2)}:0',
                  'candidate_number': id.toString().startsWith('noise')
                      ? (id == 'noise-other' ? 21 : 20)
                      : (id == 'b' || id == 'both-b' ? 10 : 2),
                  'model_fingerprint': 'fp',
                  // A large candidate is still noise when density fails.
                  'event_count': id == 'noise-other'
                      ? 20
                      : (id.toString().startsWith('noise') ? 4 : 10),
                  'min_cluster_events': 5,
                  'is_noise': id.toString().startsWith('noise'),
                  'density_verified': !id.toString().startsWith('noise'),
                },
            }),
            200,
            headers: {'content-type': 'application/json'},
          );
        }),
      );
      final png = File(items.first.path).readAsBytesSync();
      void add(
        String name,
        List<String> labels,
        List<String> observations, {
        bool validated = false,
        String? dateTaken,
      }) {
        final file = File('${tempDir.path}/$name.jpg')..writeAsBytesSync(png);
        fixtures.add(
          DetectionItem(
            filename: '$name.jpg',
            path: file.path,
            fileType: 'jpg',
            width: 10,
            height: 10,
            dateTaken: dateTaken ?? '2026-09-${10 + fixtures.length}T12:00:00Z',
            species: labels,
            confidence: 0.95,
            validated: validated,
            detectionData: {'物种名称': labels.join(',')},
            detectionBoxes: [
              for (final label in labels.where(
                (s) => s != '拒识/Unknown' && s != '空',
              ))
                DetectionBox(
                  species: label,
                  bbox: [0, 0, 10, 10],
                  confidence: 0.95,
                ),
              for (final oid in observations)
                DetectionBox(
                  species: '拒识/Unknown',
                  bbox: [0, 0, 10, 10],
                  observationId: oid,
                ),
            ],
          ),
        );
      }

      add('a', ['拒识/Unknown'], ['a']);
      add('a2', ['拒识/Unknown'], ['a2']);
      add('b', ['拒识/Unknown'], ['b']);
      add('mixed', ['家羊', '拒识/Unknown'], ['mixed']);
      add('reverse', ['拒识/Unknown', '家羊'], ['reverse']);
      add('both', ['拒识/Unknown'], ['both-a', 'both-b']);
      add('missing', ['拒识/Unknown'], ['missing']);
      add('no-id', ['拒识/Unknown'], []);
      add('known', ['斑羚'], []);
      add('manual', ['野猪'], [], validated: true);
      add('empty', ['空'], []);
      for (var index = 0; index < 4; index++) {
        add(
          'noise-$index',
          ['拒识/Unknown'],
          ['noise-$index'],
          dateTaken: '2026-10-01T12:00:0${index}Z',
        );
      }
      add(
        'noise-other',
        ['拒识/Unknown'],
        ['noise-other'],
        dateTaken: '2026-10-02T12:00:00Z',
      );
      final harness = GlobalKey<_GroupExpansionHarnessState>();
      await tester.pumpWidget(
        _GroupExpansionHarness(
          key: harness,
          apiClient: apiClient,
          initialItems: fixtures,
          autoGroup: autoGroup,
          reorderMetadata: false,
          classificationModelPath: 'model.neri.json',
        ),
      );
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      final list = find.byKey(const ValueKey('validation-species-list'));
      for (
        var attempt = 0;
        attempt < 100 && list.evaluate().isEmpty;
        attempt++
      ) {
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump(const Duration(milliseconds: 16));
      }
      List<ListTile> tiles() => tester
          .widgetList<ListTile>(
            find.descendant(of: list, matching: find.byType(ListTile)),
          )
          .toList();
      List<String> titles() =>
          tiles().map((tile) => (tile.title! as Text).data!).toList();
      expect(titles(), [
        '斑羚',
        '家羊、拒识簇 #2',
        '拒识簇 #2',
        '拒识簇 #10',
        '噪声',
        '拒识（未分簇）',
        '空',
        '野猪',
      ]);
      String subtitle(String label) =>
          (tiles().firstWhere((t) => (t.title! as Text).data == label).subtitle!
                  as Text)
              .data!;
      expect(subtitle('家羊、拒识簇 #2'), endsWith('2 个文件'));
      expect(subtitle('拒识簇 #2'), endsWith('3 个文件'));
      expect(subtitle('拒识簇 #10'), endsWith('2 个文件'));
      expect(subtitle('拒识（未分簇）'), endsWith('2 个文件'));
      expect(subtitle('噪声'), autoGroup ? '2 组 / 5 个文件' : '5 个文件');
      expect(requests, 1);
      await tester.tap(find.descendant(of: list, matching: find.text('噪声')));
      await tester.pump(const Duration(milliseconds: 300));
      expect(find.byTooltip('确认簇物种并学习'), findsNothing);
      await tester.tap(
        find.descendant(of: list, matching: find.text('拒识簇 #2')),
      );
      await tester.pump(const Duration(milliseconds: 300));
      await tester.tap(find.byTooltip('确认簇物种并学习'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));
      await tester.enterText(
        find.byWidgetPredicate(
          (widget) =>
              widget is TextField && widget.decoration?.labelText == '人工确认物种',
        ),
        '岩羊',
      );
      await tester.pump();
      await tester.tap(find.text('确认并学习'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      expect(learningRequests, 1);
      expect(titles(), isNot(contains('拒识簇 #2')));
      expect(titles(), contains('岩羊、拒识簇 #10'));
      expect(subtitle('拒识簇 #10'), endsWith('1 个文件'));
      expect(find.textContaining('已更新分类头增量记忆'), findsOneWidget);
      await tester.tap(find.byTooltip('撤回上一步校验'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      expect(undoRequests, 1);
      expect(titles(), contains('拒识簇 #2'));
      expect(subtitle('拒识簇 #2'), endsWith('3 个文件'));
      expect(subtitle('拒识簇 #10'), endsWith('2 个文件'));
      harness.currentState!._replaceSpecies(fixtures[5], '岩羊');
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      expect(subtitle('拒识簇 #2'), endsWith('2 个文件'));
      expect(subtitle('拒识簇 #10'), endsWith('1 个文件'));
      expect(titles().indexOf('岩羊'), greaterThan(titles().indexOf('空')));
      expect(
        requests,
        1,
        reason: 'metadata changes must reuse already loaded cluster links',
      );
      expect(tester.takeException(), isNull);
      await tester.pump(const Duration(seconds: 5));
    });
  }
}

class _GroupExpansionHarness extends StatefulWidget {
  const _GroupExpansionHarness({
    required this.apiClient,
    required this.initialItems,
    this.autoGroup = true,
    this.reorderMetadata = true,
    this.classificationModelPath,
    this.videoMode = 'fast',
    super.key,
  });

  final NeriApiClient apiClient;
  final List<DetectionItem> initialItems;
  final bool autoGroup;
  final bool reorderMetadata;
  final String? classificationModelPath;
  final String videoMode;

  @override
  State<_GroupExpansionHarness> createState() => _GroupExpansionHarnessState();
}

class _GroupExpansionHarnessState extends State<_GroupExpansionHarness> {
  late List<DetectionItem> _items = widget.initialItems;
  final Set<String> _markedPaths = {};
  int _metadataRequests = 0;
  bool _metadataApplied = false;

  void _replaceSpecies(DetectionItem item, String species) {
    final updated = item.mergeValidationUpdate(
      DetectionItem(
        filename: item.filename,
        path: item.path,
        fileType: item.fileType,
        species: [species],
        validated: true,
        detectionData: {'物种名称': species, '最低置信度': '人工校验'},
      ),
    );
    setState(() {
      _items = [
        for (final current in _items)
          current.path == item.path ? updated : current,
      ];
    });
  }

  Future<void> _loadMetadata(DetectionItem item) async {
    if (!widget.reorderMetadata) return;
    _metadataRequests += 1;
    if (_metadataRequests < 2 || _metadataApplied) return;
    _metadataApplied = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() {
        _items = [
          for (final current in _items)
            if (current.path == item.path)
              DetectionItem(
                filename: current.filename,
                path: current.path,
                fileType: current.fileType,
                dateTaken: '2026-09-11T12:00:10Z',
                modifiedAt: current.modifiedAt,
                width: current.width,
                height: current.height,
                sizeBytes: current.sizeBytes,
                species: current.species,
                confidence: current.confidence,
                detectionBoxes: current.detectionBoxes,
                detectionData: current.detectionData,
                error: current.error,
                validated: current.validated,
              )
            else
              current,
        ];
      });
    });
  }

  DetectionItem _confirmItem(DetectionItem item) {
    _markedPaths.add(item.path);
    final updated = item.mergeValidationUpdate(
      DetectionItem(
        filename: item.filename,
        path: item.path,
        fileType: item.fileType,
        species: item.species,
        confidence: item.confidence,
        detectionData: {...item.detectionData, '最低置信度': '人工校验'},
        validated: true,
      ),
    );
    setState(() {
      _items = [
        for (final current in _items)
          if (current.path == updated.path) updated else current,
      ];
    });
    return updated;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: widget.apiClient,
          classificationModelPath: widget.classificationModelPath,
          videoMode: widget.videoMode,
          inputPath: File(widget.initialItems.first.path).parent.path,
          rootInputPath: File(widget.initialItems.first.path).parent.path,
          rootItems: _items,
          exportFormatMode: 'ask',
          items: _items,
          loading: false,
          refreshVersion: 0,
          speciesTypes: const <String, String>{'豹猫': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: widget.autoGroup,
          collapseGroups: true,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: 4,
          autoGroupGapSeconds: 1800,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: const <String>['豹猫'],
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
          favoritePhotoExportMode: favoritePhotoExportAsk,
          emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
          onRefresh: () async {},
          onLoadMetadata: _loadMetadata,
          onOpenExternal: (_) {},
          onMarkItem:
              (
                candidate,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => _confirmItem(candidate),
          onMarkItems:
              (
                candidates,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => candidates.map(_confirmItem).toList(),
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
