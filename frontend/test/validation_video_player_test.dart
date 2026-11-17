import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/validation_video_session.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer_impl.dart'
    as viewer_impl;

import 'support/fake_validation_video_backend.dart';

DetectionItem _videoItem(
  String path, {
  List<DetectionBox> boxes = const <DetectionBox>[],
}) {
  return DetectionItem(
    filename: path.split('/').last,
    path: path,
    fileType: 'mp4',
    detectionBoxes: boxes,
  );
}

Widget _buildViewerHarness({
  required DetectionItem item,
  required FakeValidationVideoBackend Function() backendFactory,
  ValidationVideoLogger? logger,
  bool visible = true,
  bool showDetections = true,
  VoidCallback? onOpenExternal,
}) {
  return MaterialApp(
    home: Scaffold(
      body: TickerMode(
        enabled: visible,
        child: SizedBox(
          width: 800,
          height: 600,
          child: viewer_impl.DetectionMediaViewer(
            item: item,
            visibleBoxes: item.detectionBoxes,
            showDetections: showDetections,
            onOpenExternal: onOpenExternal ?? () {},
            videoBackendFactory: backendFactory,
            videoLogger: logger,
          ),
        ),
      ),
    ),
  );
}

Future<void> _flushMicrotasks(WidgetTester tester, [int turns = 4]) async {
  for (var i = 0; i < turns; i++) {
    await tester.pump();
  }
}

void main() {
  testWidgets(
    'stale completed event from previous media does not cancel autoplay of new media',
    (tester) async {
      Completer<void>? pendingOpen;
      late FakeValidationVideoBackend backend;
      var factoryCalls = 0;
      backend = FakeValidationVideoBackend(
        onOpen: (path, b) async {
          if (path == 'video_b.mp4') {
            pendingOpen = Completer<void>();
            await pendingOpen!.future;
          }
          b.currentMediaId = path;
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () {
            factoryCalls++;
            return backend;
          },
        ),
      );
      await _flushMicrotasks(tester);
      expect(factoryCalls, 1);
      expect(backend.isPlaying, isTrue);

      // Switch to video_b while its open stays pending.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4'),
          backendFactory: () {
            factoryCalls++;
            return backend;
          },
        ),
      );
      await tester.pump();

      // Stale completion from video_a arrives while video_b is opening.
      backend.emitCompleted(true);
      await _flushMicrotasks(tester);

      // Finish opening video_b: it must still auto-play.
      pendingOpen!.complete();
      await _flushMicrotasks(tester);

      expect(factoryCalls, 1);
      expect(backend.isPlaying, isTrue);
      expect(backend.operations, ['open:video_a.mp4', 'play', 'open:video_b.mp4', 'play']);

      // Even if a stale completed arrives with old media identity, ignore it.
      backend.currentMediaId = 'video_a.mp4';
      backend.emitRawCompletedEvent(true);
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'queued old completed event arriving after new open finishes while playing is ignored',
    (tester) async {
      final backend = FakeValidationVideoBackend();

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
        ),
      );
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);

      // Switch to video_b and let open + play finish so currentMediaId is
      // already video_b.mp4 and backend.isPlaying is true.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4'),
          backendFactory: () => backend,
        ),
      );
      await _flushMicrotasks(tester);
      expect(backend.currentMediaId, 'video_b.mp4');
      expect(backend.isPlaying, isTrue);

      // Queued old completed=true arrives after video_b is already playing.
      backend.emitRawCompletedEvent(true);
      await _flushMicrotasks(tester);

      expect(backend.isPlaying, isTrue);
      expect(find.byIcon(Icons.pause_rounded), findsOneWidget);
      expect(backend.operations, [
        'open:video_a.mp4',
        'play',
        'open:video_b.mp4',
        'play',
      ]);

      // Genuine EOF on video_b still transitions to completed and replays.
      backend.emitMetadata(duration: const Duration(seconds: 6));
      backend.emitCompleted(true);
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isFalse);
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);
      expect(backend.operations.sublist(backend.operations.length - 2), [
        'seek:0',
        'play',
      ]);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'path switch reuses state and resets playing, duration, position, and size immediately',
    (tester) async {
      Completer<void>? openBCompleter;
      final backend = FakeValidationVideoBackend(
        onOpen: (path, b) async {
          if (path == 'video_b.mp4') {
            openBCompleter = Completer<void>();
            await openBCompleter!.future;
          }
          b.currentMediaId = path;
        },
      );

      const boxA = DetectionBox(
        species: 'deer',
        bbox: [0.1, 0.1, 0.5, 0.5],
        confidence: 0.9,
      );
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4', boxes: const [boxA]),
          backendFactory: () => backend,
        ),
      );
      await _flushMicrotasks(tester);

      backend.emitMetadata(
        duration: const Duration(seconds: 25),
        position: const Duration(seconds: 11),
        videoSize: const Size(1920, 1080),
      );
      await tester.pump(const Duration(milliseconds: 250));
      expect(find.text('00:11 / 00:25'), findsOneWidget);
      expect(
        find.byType(viewer_impl.DinoVideoDetectionOverlay),
        findsOneWidget,
      );

      // Switch to video_b before its open finishes.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4', boxes: const [boxA]),
          backendFactory: () => backend,
        ),
      );
      await tester.pump();

      // Previous metadata and boxes must be cleared at transition.
      expect(find.byKey(const ValueKey('validation-video-loading')), findsOneWidget);
      expect(find.text('00:00 / 00:00'), findsOneWidget);
      expect(find.byType(viewer_impl.DinoVideoDetectionOverlay), findsNothing);
      expect(find.byIcon(Icons.pause_rounded), findsOneWidget);

      openBCompleter!.complete();
      await _flushMicrotasks(tester);
      backend.emitMetadata(
        duration: const Duration(seconds: 8),
        position: const Duration(seconds: 2),
        videoSize: const Size(1280, 720),
      );
      await tester.pump(const Duration(milliseconds: 250));

      expect(find.byKey(const ValueKey('validation-video-loading')), findsNothing);
      expect(find.text('00:02 / 00:08'), findsOneWidget);
      expect(
        find.byType(viewer_impl.DinoVideoDetectionOverlay),
        findsOneWidget,
      );
      expect(backend.maxConcurrentOperations, 1);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'toggling play/pause during open or pending play updates intent and serializes commands',
    (tester) async {
      Completer<void>? openCompleter;
      Completer<void>? playCompleter;
      final backend = FakeValidationVideoBackend(
        onOpen: (path, b) async {
          if (path == 'video_a.mp4' || path == 'video_b.mp4') {
            openCompleter = Completer<void>();
            await openCompleter!.future;
          }
          b.currentMediaId = path;
        },
        onPlay: (b) async {
          final pendingPlay = playCompleter;
          if (pendingPlay != null && !pendingPlay.isCompleted) {
            await pendingPlay.future;
          }
          b.emitPlaying(true);
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
        ),
      );
      await tester.pump();

      // Tap pause once while open is in flight -> should NOT call play when open finishes.
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      openCompleter!.complete();
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isFalse);
      expect(backend.operations, ['open:video_a.mp4']);

      // Switch to video_b, toggle pause then play while opening -> should autoplay.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4'),
          backendFactory: () => backend,
        ),
      );
      await tester.pump();

      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();
      expect(find.byIcon(Icons.pause_rounded), findsOneWidget);

      playCompleter = Completer<void>();
      openCompleter!.complete();
      await _flushMicrotasks(tester);

      // play() is currently awaiting; tapping pause must serialize pause after play.
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      playCompleter.complete();
      await _flushMicrotasks(tester);

      expect(backend.isPlaying, isFalse);
      expect(backend.maxConcurrentOperations, 1);
      expect(backend.operations, [
        'open:video_a.mp4',
        'open:video_b.mp4',
        'play',
        'pause',
      ]);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'hiding tab while play or open is pending preserves intent and resumes on return',
    (tester) async {
      Completer<void>? openBCompleter;
      Completer<void>? playACompleter = Completer<void>();
      final backend = FakeValidationVideoBackend(
        onOpen: (path, b) async {
          if (path == 'video_b.mp4') {
            openBCompleter = Completer<void>();
            await openBCompleter!.future;
          }
          b.currentMediaId = path;
        },
        onPlay: (b) async {
          final pendingPlay = playACompleter;
          if (pendingPlay != null && !pendingPlay.isCompleted) {
            await pendingPlay.future;
          }
          b.emitPlaying(true);
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
          visible: true,
        ),
      );
      await _flushMicrotasks(tester);

      // Hide tab while play() is still pending (backend.isPlaying is still false!).
      expect(backend.isPlaying, isFalse);
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
          visible: false,
        ),
      );
      await tester.pump();

      playACompleter.complete();
      playACompleter = null;
      await _flushMicrotasks(tester);

      // Hidden tab must pause after pending play completes.
      expect(backend.isPlaying, isFalse);
      expect(backend.operations, ['open:video_a.mp4', 'play', 'pause']);

      // Returning to tab must restore the pending play intent.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
          visible: true,
        ),
      );
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);
      expect(backend.operations, ['open:video_a.mp4', 'play', 'pause', 'play']);

      // Also verify hiding while open is in flight, then becoming visible later.
      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4'),
          backendFactory: () => backend,
          visible: false,
        ),
      );
      await tester.pump();
      openBCompleter!.complete();
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isFalse);

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_b.mp4'),
          backendFactory: () => backend,
          visible: true,
        ),
      );
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);
      expect(backend.maxConcurrentOperations, 1);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'seek and replay after completion are serialized in order',
    (tester) async {
      final seekCompleter = Completer<void>();
      final backend = FakeValidationVideoBackend(
        onSeek: (position, b) async {
          if (position.inSeconds == 5 && !seekCompleter.isCompleted) {
            await seekCompleter.future;
          }
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
        ),
      );
      await _flushMicrotasks(tester);
      backend.emitMetadata(
        duration: const Duration(seconds: 10),
        position: const Duration(seconds: 2),
      );
      await tester.pump();

      // Trigger seek via slider change, then immediately tap pause before seek resolves.
      final slider = tester.widget<Slider>(find.byType(Slider));
      slider.onChangeStart?.call(2000);
      slider.onChanged?.call(5000);
      slider.onChangeEnd?.call(5000);
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();

      seekCompleter.complete();
      await _flushMicrotasks(tester);

      expect(backend.maxConcurrentOperations, 1);
      expect(backend.operations, [
        'open:video_a.mp4',
        'play',
        'seek:5000',
        'pause',
      ]);

      // Resume, then complete video with position slightly behind duration.
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await _flushMicrotasks(tester);
      backend.emitMetadata(position: const Duration(seconds: 9));
      backend.emitCompleted(true);
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isFalse);
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      // Replay must seek to zero before playing.
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isTrue);
      expect(backend.operations.sublist(backend.operations.length - 2), [
        'seek:0',
        'play',
      ]);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'queued completed event arriving while replay seek is pending does not cancel replay',
    (tester) async {
      Completer<void>? replaySeekCompleter;
      final backend = FakeValidationVideoBackend(
        onSeek: (position, b) async {
          if (position == Duration.zero && replaySeekCompleter != null) {
            await replaySeekCompleter.future;
          }
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('video_a.mp4'),
          backendFactory: () => backend,
        ),
      );
      await _flushMicrotasks(tester);
      backend.emitMetadata(
        duration: const Duration(seconds: 6),
        position: const Duration(seconds: 6),
      );
      await tester.pump();

      // Genuine EOF reaches completed=true, isPlaying=false.
      backend.emitCompleted(true);
      await _flushMicrotasks(tester);
      expect(backend.isCompleted, isTrue);
      expect(backend.isPlaying, isFalse);

      // User clicks play to replay; seek(0) is pending while backend still
      // reports isCompleted=true and isPlaying=false.
      replaySeekCompleter = Completer<void>();
      await tester.tap(find.byKey(const ValueKey('validation-video-play-pause')));
      await tester.pump();

      // Another queued completed(true) arrives before seek(0) completes.
      backend.emitRawCompletedEvent(true);
      await _flushMicrotasks(tester);

      replaySeekCompleter.complete();
      await _flushMicrotasks(tester);

      expect(backend.isPlaying, isTrue);
      expect(backend.isCompleted, isFalse);
      expect(backend.operations.sublist(backend.operations.length - 2), [
        'seek:0',
        'play',
      ]);

      // Subsequent genuine EOF after replay still completes normally.
      backend.emitCompleted(true);
      await _flushMicrotasks(tester);
      expect(backend.isPlaying, isFalse);
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'logs open exceptions and native errors with path/requestId and supports same-path retry',
    (tester) async {
      final logs = <String>[];
      var openAttempts = 0;
      late FakeValidationVideoBackend backend;
      backend = FakeValidationVideoBackend(
        onOpen: (path, b) async {
          openAttempts++;
          if (openAttempts == 1) {
            throw StateError('decoder init failed');
          }
          b.currentMediaId = path;
        },
      );

      await tester.pumpWidget(
        _buildViewerHarness(
          item: _videoItem('broken.mp4'),
          backendFactory: () => backend,
          logger: (message, {error, stackTrace}) {
            logs.add('$message | $error');
          },
        ),
      );
      await _flushMicrotasks(tester);

      expect(find.text('媒体加载失败'), findsOneWidget);
      expect(
        find.byKey(const ValueKey('validation-video-retry-button')),
        findsOneWidget,
      );
      expect(
        logs.any(
          (entry) =>
              entry.contains('requestId=1') &&
              entry.contains('broken.mp4') &&
              entry.contains('decoder init failed'),
        ),
        isTrue,
      );

      // Explicit retry on the same path succeeds and starts playback.
      await tester.tap(
        find.byKey(const ValueKey('validation-video-retry-button')),
      );
      await _flushMicrotasks(tester);
      expect(openAttempts, 2);
      expect(find.text('媒体加载失败'), findsNothing);
      expect(backend.isPlaying, isTrue);

      // Native error stream transitions active session to error + logs info.
      backend.emitError('demuxer read error');
      await _flushMicrotasks(tester);
      expect(find.text('媒体加载失败'), findsOneWidget);
      expect(
        logs.any(
          (entry) =>
              entry.contains('requestId=2') &&
              entry.contains('broken.mp4') &&
              entry.contains('demuxer read error'),
        ),
        isTrue,
      );

      // Retry again succeeds.
      await tester.tap(
        find.byKey(const ValueKey('validation-video-retry-button')),
      );
      await _flushMicrotasks(tester);
      expect(openAttempts, 3);
      expect(backend.isPlaying, isTrue);

      // Stale native error with mismatched media identity is ignored.
      backend.currentMediaId = 'stale_other.mp4';
      backend.emitError('stale error');
      await _flushMicrotasks(tester);
      expect(find.text('媒体加载失败'), findsNothing);
      expect(backend.isPlaying, isTrue);

      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  test('controller redacts URI credentials and query secrets from logs', () async {
    final logs = <String>[];
    final backend = FakeValidationVideoBackend(
      onOpen: (path, b) async {
        throw StateError('network error');
      },
    );
    final controller = ValidationVideoSessionController(
      backend: backend,
      logger: (message, {error, stackTrace}) {
        logs.add('$message | $error');
      },
    );
    addTearDown(controller.dispose);

    await controller.open('https://user:secret_pass@example.com/clip.mp4?token=xyz');
    expect(logs, hasLength(1));
    expect(logs.single, isNot(contains('secret_pass')));
    expect(logs.single, isNot(contains('token=xyz')));
    expect(logs.single, contains('https://example.com/clip.mp4'));
    expect(logs.single, contains('requestId=1'));
  });

  test('default logger redacts URI secrets inside raw error strings', () async {
    final printed = <String>[];
    final previousDebugPrint = debugPrint;
    debugPrint = (String? message, {int? wrapWidth}) {
      if (message != null) printed.add(message);
    };
    addTearDown(() => debugPrint = previousDebugPrint);

    final backend = FakeValidationVideoBackend(
      onOpen: (path, b) async {
        throw StateError(
          'failed fetching https://admin:top_secret@example.com/v.mp4?sig=abc',
        );
      },
    );
    final controller = ValidationVideoSessionController(backend: backend);
    addTearDown(controller.dispose);

    await controller.open(
      'https://user:secret_pass@example.com/clip.mp4?token=xyz',
    );

    expect(printed, hasLength(1));
    expect(printed.single, isNot(contains('secret_pass')));
    expect(printed.single, isNot(contains('token=xyz')));
    expect(printed.single, isNot(contains('top_secret')));
    expect(printed.single, isNot(contains('sig=abc')));
    expect(printed.single, contains('https://example.com/clip.mp4'));
    expect(printed.single, contains('https://example.com/v.mp4'));
    expect(printed.single, contains('requestId=1'));
  });
}
