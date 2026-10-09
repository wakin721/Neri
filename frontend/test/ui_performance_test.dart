import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/async_refresh_gate.dart';
import 'package:neri_flutter/src/utils/job_result_refresh.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';
import 'package:neri_flutter/src/utils/preview_decode_size.dart';
import 'package:neri_flutter/src/utils/result_json_decoder.dart';
import 'package:neri_flutter/src/utils/video_detection_index.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer_impl.dart';
import 'package:neri_flutter/src/widgets/retained_tab.dart';

void main() {
  test(
    'deleted job details are omitted while other API errors remain visible',
    () async {
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          final id = request.url.pathSegments.last;
          if (id == 'removed')
            return http.Response('{"detail":"Job not found"}', 404);
          if (id == 'broken') return http.Response('{"detail":"failure"}', 500);
          return http.Response(
            '{"id":"kept","state":"completed","results":[]}',
            200,
          );
        }),
      );
      addTearDown(api.close);
      final results = await Future.wait(
        ['removed', 'kept'].map(api.fetchJobIfExists),
      );
      expect(results.first, isNull);
      expect(results.last!.id, 'kept');
      await expectLater(
        api.fetchJobIfExists('broken'),
        throwsA(isA<ApiException>()),
      );
    },
  );

  test(
    'explicit refresh survives an in-flight timeout and ignores its late result',
    () async {
      final gate = AsyncRefreshGate();
      final stalled = Completer<void>();
      var finished = 0;
      final first = gate.run(
        () => stalled.future.timeout(const Duration(milliseconds: 10)),
      );
      final failed = expectLater(first, throwsA(isA<TimeoutException>()));
      final second = gate.run(() async {
        finished++;
      });
      await Future.wait([failed, second]);
      expect(finished, 1);
      stalled.complete();
      await Future<void>.delayed(Duration.zero);
      await gate.run(() async {
        finished++;
      });
      expect(finished, 2);
    },
  );

  test(
    'large result decoding preserves all items, boxes, and job metadata',
    () async {
      final items = [
        for (var i = 0; i < 5000; i++)
          {
            'filename': '$i.jpg',
            'path': 'photos/$i.jpg',
            'file_type': 'jpg',
            'detection_boxes': [
              {
                'species': '豹猫',
                'bbox': [10, 20, 30, 40],
                'confidence': 0.9,
              },
            ],
          },
      ];
      final decoded = await decodeDetectionItems(jsonEncode(items));
      expect(decoded.length, 5000);
      expect(decoded.last.path, 'photos/4999.jpg');
      expect(decoded.last.detectionBoxes.single.bbox, [10, 20, 30, 40]);
      final job = await decodeProcessingJob(
        jsonEncode({
          'id': 'job',
          'state': 'completed',
          'processed': 5000,
          'results': items,
        }),
      );
      expect(job.id, 'job');
      expect(job.processed, 5000);
      expect(job.results.last.detectionBoxes.single.species, '豹猫');
    },
  );
  test(
    'timer ticks coalesce and explicit refresh runs after an in-flight request',
    () async {
      final gate = AsyncRefreshGate();
      final release = Completer<void>();
      var requests = 0;
      var active = 0;
      var peak = 0;
      Future<void> refresh() async {
        requests++;
        peak = math.max(peak, ++active);
        await release.future;
        active--;
      }

      final first = gate.run(refresh);
      final ticks = List.generate(20, (_) => gate.run(refresh, coalesce: true));
      final explicit = gate.run(refresh);
      expect(requests, 1);
      release.complete();
      await Future.wait([first, ...ticks, explicit]);
      expect(requests, 2);
      expect(peak, 1);
    },
  );

  test('refresh gate recovers after a failed request', () async {
    final gate = AsyncRefreshGate();
    await expectLater(
      gate.run(() async => throw StateError('offline')),
      throwsStateError,
    );
    var called = false;
    await gate.run(() async => called = true);
    expect(called, isTrue);
  });

  test(
    'unchanged summaries reuse results and deferred changes remain detectable',
    () {
      ProcessingJob job(
        String version, {
        int processed = 1,
        String state = 'running',
        bool active = false,
      }) => ProcessingJob(
        id: 'job',
        state: state,
        inputDir: 'photos',
        createdAt: 'start',
        updatedAt: version,
        processed: processed,
        active: active,
      );
      final complete = job('v1');
      for (var tick = 0; tick < 100; tick++) {
        expect(jobResultsNeedRefresh(job('v1'), complete), isFalse);
      }
      expect(jobResultsNeedRefresh(job('v2'), complete), isTrue);
      expect(jobResultsNeedRefresh(job('v1', processed: 2), complete), isTrue);
      expect(
        jobResultsNeedRefresh(job('v1', state: 'completed'), complete),
        isTrue,
      );
      expect(jobResultsNeedRefresh(job('v1', active: true), complete), isTrue);
      expect(jobResultsNeedRefresh(job('v1'), null), isTrue);
    },
  );

  test(
    'validation list identity is stable until data, scope, or filter changes',
    () {
      final cache = ValidationItemsCache();
      final folder = Directory.systemTemp.absolute.path;
      DetectionItem item(String name) =>
          DetectionItem(filename: name, path: '$folder/$name', fileType: 'jpg');
      final a = item('a.jpg');
      final b = item('b.jpg');
      final source = [a, b];
      final filter = <String>{};
      final first = cache.itemsFor(source, folder, filter);
      for (var tick = 0; tick < 100; tick++) {
        expect(
          identical(cache.itemsFor(source, folder, filter), first),
          isTrue,
        );
      }
      final changed = cache.itemsFor([a], folder, filter);
      expect(changed, [a]);
      expect(identical(changed, first), isFalse);
      final key = Platform.isWindows
          ? b.path.replaceAll('\\', '/').toLowerCase()
          : b.path;
      expect(cache.itemsFor(source, folder, {key}), [b]);
      expect(cache.itemsFor(source, '$folder/another', filter), isEmpty);
    },
  );

  test(
    'bounded file existence checks preserve order and omit missing files',
    () async {
      final directory = await Directory.systemTemp.createTemp('neri_files_');
      addTearDown(() => directory.delete(recursive: true));
      final items = [
        for (var i = 0; i < 70; i++)
          DetectionItem(
            filename: '$i.jpg',
            path: '${directory.path}/$i.jpg',
            fileType: 'jpg',
          ),
      ];
      for (var i = 0; i < items.length; i += 2) {
        await File(items[i].path).writeAsBytes([]);
      }
      expect(await existingLocalDetectionItems(items), [
        for (var i = 0; i < items.length; i += 2) items[i],
      ]);
    },
  );

  test(
    'decode sizing respects aspect ratio, high DPI, resize buckets, and native size',
    () {
      expect(
        previewDecodeWidth(const Size(4000, 3000), const Size(800, 600), 1),
        832,
      );
      expect(
        previewDecodeWidth(const Size(4000, 3000), const Size(805, 600), 1),
        832,
      );
      expect(
        previewDecodeWidth(const Size(4000, 3000), const Size(800, 600), 2),
        1600,
      );
      expect(
        previewDecodeWidth(const Size(3000, 4000), const Size(800, 600), 1),
        512,
      );
      expect(
        previewDecodeWidth(const Size(100, 75), const Size(800, 600), 2),
        100,
      );
    },
  );

  test(
    '20k-frame lookup returns only nearby entries and keeps original tie order',
    () {
      final boxes = [
        for (var frame = 19999; frame >= 0; frame--)
          DetectionBox(
            species: 'animal',
            bbox: const [0, 0, 1, 1],
            frameIndex: frame,
          ),
      ];
      final index = VideoDetectionIndex(boxes);
      final nearby = index.candidates(
        frames: [10000],
        frameTolerance: 3,
        seconds: 0,
        timeTolerance: 0.25,
      );
      expect(nearby.length, 7);
      expect(nearby.map((entry) => entry.box.frameIndex), [
        10003,
        10002,
        10001,
        10000,
        9999,
        9998,
        9997,
      ]);
      expect(index.maxFrame, 19999);
    },
  );

  test(
    'indexed windows agree with a linear lookup for mixed frames and timestamps',
    () {
      final random = math.Random(7);
      final boxes = [
        for (var i = 0; i < 1000; i++)
          DetectionBox(
            species: 'animal',
            bbox: const [0, 0, 1, 1],
            frameIndex: i.isEven ? random.nextInt(1000) : null,
            timestamp: i % 3 == 0 ? random.nextDouble() * 10 : null,
          ),
      ];
      final index = VideoDetectionIndex(boxes);
      for (var i = 0; i < 40; i++) {
        final frames = [random.nextInt(1000), random.nextInt(1000)];
        final seconds = random.nextDouble() * 10;
        final expected = [
          for (var j = 0; j < boxes.length; j++)
            if (boxes[j].frameIndex != null
                ? frames.any((f) => (boxes[j].frameIndex! - f).abs() <= 6)
                : boxes[j].timestamp != null &&
                      (boxes[j].timestamp! - seconds).abs() <= 0.25)
              j,
        ];
        expect(
          index
              .candidates(
                frames: frames,
                frameTolerance: 6,
                seconds: seconds,
                timeTolerance: 0.25,
              )
              .map((entry) => entry.originalIndex),
          expected,
        );
      }
    },
  );

  test(
    'video selection retains per-track nearest matching and original-order ties',
    () {
      const first = DetectionBox(
        species: 'animal',
        bbox: [0, 0, 1, 1],
        frameIndex: 10,
        trackId: '1',
      );
      const second = DetectionBox(
        species: 'animal',
        bbox: [0, 0, 1, 1],
        frameIndex: 20,
        trackId: '1',
      );
      const distant = DetectionBox(
        species: 'animal',
        bbox: [0, 0, 1, 1],
        frameIndex: 99,
        trackId: '2',
      );
      final boxes = [first, second, distant];
      expect(
        currentVideoDetectionBoxes(
          boxes: boxes,
          index: VideoDetectionIndex(boxes),
          position: const Duration(milliseconds: 1500),
          duration: const Duration(seconds: 10),
          detectionData: const {'total_frames_processed': 10, 'vid_stride': 10},
        ),
        [first],
      );
    },
  );

  testWidgets(
    'hidden tabs build lazily, retain local state, and receive latest data on return',
    (tester) async {
      final builds = [0, 0];
      final counterKey = GlobalKey<_CounterState>();
      Future<void> show(int selected, String version) => tester.pumpWidget(
        MaterialApp(
          home: IndexedStack(
            index: selected,
            children: [
              for (var i = 0; i < 2; i++)
                RetainedTab(
                  active: selected == i,
                  builder: (_) {
                    builds[i]++;
                    return _Counter(
                      key: i == 0 ? counterKey : null,
                      label: '$i:$version',
                    );
                  },
                ),
            ],
          ),
        ),
      );
      await show(0, 'old');
      expect(builds, [1, 0]);
      counterKey.currentState!.increment();
      await tester.pump();
      await show(1, 'old');
      await show(1, 'new');
      expect(builds, [1, 2]);
      expect(counterKey.currentState!.ticksEnabled, isFalse);
      await show(0, 'new');
      expect(builds, [2, 2]);
      expect(find.text('0:new count=1'), findsOneWidget);
      expect(counterKey.currentState!.ticksEnabled, isTrue);
    },
  );

  testWidgets(
    'service tabs preload and update only for relevant hidden changes',
    (tester) async {
      var builds = 0;
      Future<void> show(int revision) => tester.pumpWidget(
        MaterialApp(
          home: RetainedTab(
            active: false,
            preload: true,
            refreshKey: revision,
            builder: (_) {
              builds++;
              return Text('revision=$revision');
            },
          ),
        ),
      );
      await show(0);
      await show(0);
      expect(builds, 1);
      await show(1);
      expect(builds, 2);
      expect(find.text('revision=1'), findsOneWidget);
    },
  );

  testWidgets(
    'scaled preview decodes fewer pixels and hit-tests original pixel coordinates',
    (tester) async {
      final directory = Directory.systemTemp.createTempSync('neri_preview_');
      addTearDown(() => directory.deleteSync(recursive: true));
      final path = '${directory.path}/large.png';
      File(
        path,
      ).writeAsBytesSync(img.encodePng(img.Image(width: 4000, height: 3000)));
      tester.view.physicalSize = const Size(1000, 800);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      const box = DetectionBox(
        species: 'animal',
        bbox: [1000, 750, 3000, 2250],
      );
      DetectionBox? selected;
      await tester.pumpWidget(
        MaterialApp(
          home: Center(
            child: SizedBox(
              width: 800,
              height: 600,
              child: DetectionMediaViewer(
                item: DetectionItem(
                  filename: 'large.png',
                  path: path,
                  fileType: 'png',
                  width: 4000,
                  height: 3000,
                ),
                visibleBoxes: const [box],
                showDetections: true,
                onOpenExternal: () {},
                onDetectionBoxSelected: (value) => selected = value,
              ),
            ),
          ),
        ),
      );
      await _pumpUntilImage(tester);
      final image = tester.widget<Image>(find.byType(Image));
      final provider = image.image as ResizeImage;
      expect(provider.width, 832);
      final loaded = Completer<ImageInfo>();
      final stream = provider.resolve(const ImageConfiguration());
      final listener = ImageStreamListener(
        (info, _) {
          if (!loaded.isCompleted) loaded.complete(info);
        },
        onError: (error, stack) {
          if (!loaded.isCompleted) loaded.completeError(error, stack);
        },
      );
      stream.addListener(listener);
      for (var attempt = 0; attempt < 100 && !loaded.isCompleted; attempt++) {
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump();
      }
      expect(
        loaded.isCompleted,
        isTrue,
        reason: 'Scaled image must finish decoding',
      );
      final info = await loaded.future;
      expect(info.image.width, 832);
      expect(info.image.height, 624);
      stream.removeListener(listener);
      info.dispose();
      await tester.pump();
      final origin = tester.getTopLeft(find.byType(Image));
      await tester.tapAt(origin + const Offset(400, 300));
      expect(selected, box);
      await tester.tapAt(origin + const Offset(80, 60));
      expect(selected, isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );
  testWidgets(
    'EXIF dimensions override raw backend hints and preserve box hit testing',
    (tester) async {
      final directory = Directory.systemTemp.createTempSync(
        'neri_orientation_',
      );
      addTearDown(() => directory.deleteSync(recursive: true));
      final path = '${directory.path}/rotated.jpg';
      File(path).writeAsBytesSync(
        base64Decode(
          '/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAeACgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDiqKKK+aPjwooooAKKKKACiiigAooooAKKKKAP/9k=',
        ),
      );
      tester.view.physicalSize = const Size(800, 800);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      const box = DetectionBox(species: 'animal', bbox: [0.1, 0.1, 0.4, 0.4]);
      DetectionBox? selected;
      await tester.pumpWidget(
        MaterialApp(
          home: Center(
            child: SizedBox(
              width: 300,
              height: 400,
              child: DetectionMediaViewer(
                item: DetectionItem(
                  filename: 'rotated.jpg',
                  path: path,
                  fileType: 'jpg',
                  width: 40,
                  height: 30,
                ),
                visibleBoxes: const [box],
                showDetections: true,
                onOpenExternal: () {},
                onDetectionBoxSelected: (value) => selected = value,
              ),
            ),
          ),
        ),
      );
      await _pumpUntilImage(tester);
      final image = tester.widget<Image>(find.byType(Image));
      expect((image.image as ResizeImage).width, 30);
      final origin = tester.getTopLeft(find.byType(Image));
      await tester.tapAt(origin + const Offset(60, 80));
      expect(selected, same(box));
      await tester.tapAt(origin + const Offset(250, 350));
      expect(selected, isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets('preview reuses filtered boxes until data or filters change', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final item = DetectionItem(
      filename: 'missing.jpg',
      path: '/missing.jpg',
      fileType: 'jpg',
      detectionBoxes: const [
        DetectionBox(species: 'animal', bbox: [0, 0, 1, 1], confidence: 0.8),
      ],
    );
    Future<void> show(
      DetectionItem selected,
      double threshold,
      bool detections,
    ) => tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: PreviewScreen(
            inputPath: '/',
            items: [selected],
            selectedIndex: 0,
            selectedItem: selected,
            speciesTypes: const {},
            useCombinedConfidence: false,
            showDetections: detections,
            onShowDetectionsChanged: (_) {},
            selectedSpeciesFilter: previewAllSpeciesLabel,
            onSpeciesFilterChanged: (_) {},
            confidenceThreshold: threshold,
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
    List<DetectionBox> boxes() => tester
        .widget<DetectionMediaViewer>(find.byType(DetectionMediaViewer))
        .visibleBoxes;
    await show(item, 0.25, true);
    final first = boxes();
    await show(item, 0.25, false);
    expect(boxes(), same(first));
    await show(item, 0.9, true);
    expect(boxes(), isEmpty);
    await show(item, 0.25, true);
    expect(boxes(), hasLength(1));
    final replaced = DetectionItem(
      filename: item.filename,
      path: item.path,
      fileType: item.fileType,
    );
    await show(replaced, 0.25, true);
    expect(boxes(), isEmpty);
    await tester.pumpWidget(const SizedBox.shrink());
  });
}

class _Counter extends StatefulWidget {
  const _Counter({required this.label, super.key});
  final String label;
  @override
  State<_Counter> createState() => _CounterState();
}

class _CounterState extends State<_Counter> {
  int count = 0;
  bool ticksEnabled = true;
  void increment() => setState(() => count++);
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    ticksEnabled = TickerMode.valuesOf(context).enabled;
  }

  @override
  Widget build(BuildContext context) => Text('${widget.label} count=$count');
}

Future<void> _pumpUntilImage(WidgetTester tester) async {
  for (var attempt = 0; attempt < 100; attempt++) {
    if (find.byType(Image).evaluate().isNotEmpty) return;
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump();
  }
  expect(find.byType(Image), findsOneWidget);
}
