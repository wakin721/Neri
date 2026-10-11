import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/utils/preview_snapshot_cache.dart';

import 'startup_loading_test.dart' as startup;

class _DelayedCache extends PreviewSnapshotCache {
  final pending = Completer<PreviewSnapshot?>();

  @override
  Future<PreviewSnapshot?> read(String scope) => pending.future;

  @override
  Future<void> setMaxBytes(int bytes) async {}

  @override
  Future<void> write(
    String scope,
    List<DetectionItem> items, {
    List<String> directories = const [],
  }) async {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('cache size setting handles saved integers and invalid values', () {
    expect(normalizePreviewCacheLimitMb(null), 128);
    expect(normalizePreviewCacheLimitMb(0), 0);
    expect(normalizePreviewCacheLimitMb(512), 512);
    expect(normalizePreviewCacheLimitMb('37'), 37);
    expect(normalizePreviewCacheLimitMb(256.0), 256);
    for (final invalid in [-1, 16385, 'unknown', 1.5, double.infinity]) {
      expect(normalizePreviewCacheLimitMb(invalid), 128);
    }
  });

  group('persistent snapshots', () {
    late Directory directory;
    late PreviewSnapshotCache cache;
    setUp(() async {
      directory = await Directory.systemTemp.createTemp('neri-preview-cache-');
      cache = PreviewSnapshotCache(directory: directory);
    });
    tearDown(() async {
      await cache.flush();
      await directory.delete(recursive: true);
    });

    test(
      'another instance restores media, detection metadata and directories',
      () async {
        final photo = DetectionItem(
          filename: '照片.jpg',
          path: '/camera/照片.jpg',
          fileType: 'jpg',
          dateTaken: '2026-10-11T12:00:00',
          modifiedAt: '2026-10-11T13:00:00',
          width: 640,
          height: 480,
          sizeBytes: 1234,
          species: ['fox'],
          confidence: 0.8,
          validated: true,
          error: 'example',
          detectionData: {'remark': 'checked'},
          detectionBoxes: [
            DetectionBox(
              species: 'fox',
              bbox: [1, 2, 3, 4],
              confidence: 0.8,
              frameIndex: 1,
              timestamp: 1.5,
              trackId: 'track',
              observationId: 'observation',
              registryId: 42,
              predictedSpecies: 'fox',
              feedbackStatus: 'confirmed',
              candidates: [
                {'species': 'fox', 'score': 0.8},
              ],
            ),
          ],
        );
        await cache.write('/camera', [photo], directories: ['/camera/child']);
        final restored = await PreviewSnapshotCache(
          directory: directory,
        ).read('/camera');
        final item = restored!.items.single;
        expect(restored.directories, ['/camera/child']);
        expect(item.filename, photo.filename);
        expect(item.path, photo.path);
        expect(item.fileType, photo.fileType);
        expect(item.dateTaken, photo.dateTaken);
        expect(item.modifiedAt, photo.modifiedAt);
        expect([item.width, item.height, item.sizeBytes], [640, 480, 1234]);
        expect(item.species, ['fox']);
        expect(item.confidence, 0.8);
        expect(item.validated, isTrue);
        expect(item.error, 'example');
        expect(item.detectionData, {'remark': 'checked'});
        final box = item.detectionBoxes.single;
        expect(box.bbox, [1.0, 2.0, 3.0, 4.0]);
        expect(box.species, 'fox');
        expect(box.confidence, 0.8);
        expect(box.frameIndex, 1);
        expect(box.timestamp, 1.5);
        expect(box.trackId, 'track');
        expect(box.observationId, 'observation');
        expect(box.registryId, 42);
        expect(box.predictedSpecies, 'fox');
        expect(box.feedbackStatus, 'confirmed');
        expect(box.candidates, [
          {'species': 'fox', 'score': 0.8},
        ]);
      },
    );

    test(
      'directory identities stay separate and relative paths resolve consistently',
      () async {
        await cache.write('cache-root/camera', [
          const DetectionItem(
            filename: 'one.jpg',
            path: '/camera/one.jpg',
            fileType: 'jpg',
          ),
        ]);
        await cache.write('cache-root/camera-extra', []);
        expect(
          (await cache.read(
            Directory('cache-root/camera').absolute.path,
          ))!.items.single.filename,
          'one.jpg',
        );
        expect((await cache.read('cache-root/camera-extra'))!.items, isEmpty);
        expect(await cache.read('cache-root/unscanned'), isNull);
      },
    );

    test('queued newer and empty scans replace older snapshots', () async {
      final first = cache.write('/camera', [
        const DetectionItem(
          filename: 'removed.jpg',
          path: '/camera/removed.jpg',
          fileType: 'jpg',
        ),
      ]);
      final second = cache.write('/camera', []);
      await Future.wait([first, second]);
      expect(
        (await PreviewSnapshotCache(
          directory: directory,
        ).read('/camera'))!.items,
        isEmpty,
      );
    });

    for (final invalid in ['malformed', 'version', 'scope', 'item']) {
      test('ignores $invalid cached data', () async {
        await cache.write('/camera', []);
        final file = (await directory.list().toList()).whereType<File>().single;
        final data =
            jsonDecode(await file.readAsString()) as Map<String, dynamic>;
        if (invalid == 'version') data['version'] = 999;
        if (invalid == 'scope') data['scope'] = '/another-camera';
        if (invalid == 'item')
          data['items'] = [
            {'width': 'bad'},
          ];
        await file.writeAsString(
          invalid == 'malformed' ? '{invalid' : jsonEncode(data),
        );
        expect(await cache.read('/camera'), isNull);
      });
    }

    test('keeps recent directories within the entry limit', () async {
      cache = PreviewSnapshotCache(directory: directory, maxEntries: 2);
      for (final scope in ['/first', '/second', '/third']) {
        await cache.write(scope, []);
      }
      expect(await cache.read('/first'), isNull);
      expect(await cache.read('/second'), isNotNull);
      expect(await cache.read('/third'), isNotNull);
    });

    test('oversized snapshots do not evict smaller usable snapshots', () async {
      cache = PreviewSnapshotCache(directory: directory, maxBytes: 1024);
      await cache.write('/small', []);
      await cache.write('/large', [
        DetectionItem(
          filename: 'photo.jpg',
          path: '/large/photo.jpg',
          fileType: 'jpg',
          detectionData: {'remark': List.filled(2048, 'x').join()},
        ),
      ]);
      expect(await cache.read('/large'), isNull);
      expect(await cache.read('/small'), isNotNull);
    });

    test('unwritable cache does not interrupt scanning', () async {
      final blocked = File('${directory.path}/blocked');
      await blocked.writeAsString('not a directory');
      final unavailable = PreviewSnapshotCache(
        directory: Directory(blocked.path),
      );
      await unavailable.write('/camera', []);
      expect(await unavailable.read('/camera'), isNull);
    });

    test(
      'lowering the byte limit immediately keeps only recent fitting scans',
      () async {
        await cache.write('/first', []);
        await cache.write('/second', []);
        final files = (await directory.list().toList())
            .whereType<File>()
            .toList();
        final sizes = await Future.wait(files.map((file) => file.length()));
        final size = sizes.reduce((a, b) => a > b ? a : b);
        await cache.setMaxBytes(size);
        expect(await cache.read('/first'), isNull);
        expect(await cache.read('/second'), isNotNull);
        expect(
          (await directory.list().toList()).whereType<File>(),
          hasLength(1),
        );
      },
    );

    test(
      'disabling caching clears existing and queued writes until enabled again',
      () async {
        await cache.write('/old', []);
        final pending = cache.write('/queued', []);
        final disable = cache.setMaxBytes(0);
        final later = cache.write('/later', []);
        await Future.wait([pending, disable, later]);
        expect(cache.maxBytes, 0);
        expect(await cache.read('/old'), isNull);
        expect(await cache.read('/queued'), isNull);
        expect(await cache.read('/later'), isNull);
        expect(await directory.list().toList(), isEmpty);
        await cache.setMaxBytes(4096);
        await cache.write('/later', []);
        expect(await cache.read('/later'), isNotNull);
      },
    );

    test(
      'new limit applies to subsequent writes and increases allow larger scans',
      () async {
        await cache.setMaxBytes(1024);
        final large = DetectionItem(
          filename: 'photo.jpg',
          path: '/large/photo.jpg',
          fileType: 'jpg',
          detectionData: {'remark': List.filled(2048, 'x').join()},
        );
        await cache.write('/large', [large]);
        expect(await cache.read('/large'), isNull);
        await cache.setMaxBytes(4096);
        await cache.write('/large', [large]);
        expect(
          (await cache.read('/large'))!.items.single.filename,
          'photo.jpg',
        );
      },
    );
  });

  group('cached directory browsing', () {
    setUp(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('window_manager'),
            (call) async => call.method == 'isMaximized' ? false : null,
          );
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('neri/windows_shell'),
            (_) async => null,
          );
    });

    for (final limit in [0, 256]) {
      testWidgets('startup applies the saved $limit MB cache limit', (
        tester,
      ) async {
        final input = Directory('startup-test-input').absolute.path;
        final directory = (await tester.runAsync(
          () => Directory.systemTemp.createTemp('neri-limit-test-'),
        ))!;
        final cache = PreviewSnapshotCache(directory: directory);
        await tester.runAsync(
          () => cache.write('startup-test-input', [
            DetectionItem(
              filename: 'cached.jpg',
              path: '$input/cached.jpg',
              fileType: 'jpg',
            ),
          ]),
        );
        final backend = startup.StartupBackend(previewCache: cache);
        backend.savedSettings[previewCacheLimitSettingKey] = limit;
        try {
          await startup.mount(tester, backend);
          await tester.tap(find.text('预览').first);
          await tester.pump();
          await startup.settleStartup(tester);
          expect(cache.maxBytes, limit * previewCacheMegabyte);
          final screen = tester.widget<PreviewScreen>(
            find.byType(PreviewScreen),
          );
          if (limit == 0) {
            expect(screen.items, isEmpty);
            await tester.runAsync(() => cache.flush());
            expect(
              await tester.runAsync(() => directory.list().toList()),
              isEmpty,
            );
          } else {
            expect(screen.items.single.filename, 'cached.jpg');
          }
          expect(tester.takeException(), isNull);
        } finally {
          await startup.cleanup(tester, backend);
          await tester.runAsync(() => directory.delete(recursive: true));
        }
      });
    }

    testWidgets(
      'cache settings save presets, custom sizes and disabling through the app',
      (tester) async {
        final backend = startup.StartupBackend();
        try {
          await startup.mount(tester, backend);
          await tester.tap(find.text('设置').first);
          await tester.pump();
          await startup.settleStartup(tester);
          await tester.tap(find.text('基础设置').first);
          await tester.pumpAndSettle();
          final selector = find.byKey(
            const ValueKey('preview-cache-limit-selector'),
          );
          await tester.ensureVisible(selector);
          await tester.pumpAndSettle();
          Future<void> choose(String label) async {
            await tester.tap(
              find.descendant(of: selector, matching: find.byType(TextButton)),
            );
            await tester.pumpAndSettle();
            final choice = find.widgetWithText(MenuItemButton, label);
            await tester.ensureVisible(choice);
            await tester.pumpAndSettle();
            await tester.tap(choice);
            await tester.pumpAndSettle();
          }

          Future<void> save() async {
            await tester.pump(const Duration(seconds: 1));
            await startup.settleStartup(tester);
            await tester.runAsync(() => backend.previewCache.flush());
            await tester.pump();
          }

          await choose('512 MB');
          await save();
          expect(backend.savedSettings[previewCacheLimitSettingKey], 512);
          expect(backend.previewCache.maxBytes, 512 * previewCacheMegabyte);
          await choose('自定义…');
          final field = find.byKey(const ValueKey('preview-cache-limit-input'));
          for (final invalid in ['-1', '16385', '1.5', '']) {
            await tester.enterText(field, invalid);
            await tester.pump();
            expect(
              tester
                  .widget<FilledButton>(find.widgetWithText(FilledButton, '应用'))
                  .onPressed,
              isNull,
            );
          }
          await tester.enterText(field, '37');
          await tester.pump();
          await tester.tap(find.widgetWithText(FilledButton, '应用'));
          await tester.pumpAndSettle();
          await save();
          expect(backend.savedSettings[previewCacheLimitSettingKey], 37);
          expect(backend.previewCache.maxBytes, 37 * previewCacheMegabyte);
          await choose('不保留');
          await save();
          expect(backend.savedSettings[previewCacheLimitSettingKey], 0);
          expect(backend.previewCache.maxBytes, 0);
          expect(tester.takeException(), isNull);
        } finally {
          await startup.cleanup(tester, backend);
        }
      },
    );

    for (final mode in ['all', 'child', 'child-only', 'removed-child']) {
      testWidgets(
        '$mode shows persisted cache before scans finish and then replaces stale files',
        (tester) async {
          final input = Directory('startup-test-input').absolute.path;
          final camera = '$input/camera-b';
          final sibling = '$input/camera-a';
          final cacheDirectory = await tester.runAsync(
            () => Directory.systemTemp.createTemp('neri-warm-cache-'),
          );
          final cache = PreviewSnapshotCache(directory: cacheDirectory!);
          final cached = DetectionItem(
            filename: 'removed.jpg',
            path: '$camera/removed.jpg',
            fileType: 'jpg',
          );
          await tester.runAsync(
            () => cache.write(
              mode == 'child-only' ? camera : 'startup-test-input',
              [cached],
              directories: [sibling, camera],
            ),
          );
          // A new cache instance must find results written by an earlier session.
          final restartedCache = PreviewSnapshotCache(
            directory: cacheDirectory,
          );
          final backend = startup.StartupBackend(previewCache: restartedCache)
            ..directories = mode == 'child-only' ? [sibling, camera] : null;
          final childResponse = Completer<http.Response>();
          backend.scopedPreviews[camera] = childResponse;
          http.Response media(String directory, String filename) =>
              http.Response(
                jsonEncode([
                  {
                    'filename': filename,
                    'path': '$directory/$filename',
                    'file_type': 'jpg',
                  },
                ]),
                200,
              );
          try {
            await startup.mount(
              tester,
              backend,
              preferences: {
                'last_input_path': 'startup-test-input',
                'last_browse_directory': jsonEncode({
                  'root': 'startup-test-input',
                  'path': mode == 'all' ? null : camera,
                }),
              },
            );
            await tester.tap(find.text('预览').first);
            await tester.pump();
            await startup.settleStartup(tester);
            var screen = tester.widget<PreviewScreen>(
              find.byType(PreviewScreen),
            );
            expect(
              Directory(screen.inputPath).absolute.path,
              mode == 'all' ? input : camera,
            );
            expect(screen.items.single.filename, 'removed.jpg');
            expect(backend.preview.isCompleted, isFalse);
            expect(screen.loading, isTrue);

            if (mode == 'removed-child') {
              // A fresh index corrects a directory restored from an older cache.
              backend.directories = [sibling];
              screen.onRefresh();
              await tester.pump();
              await startup.settleStartup(tester);
              screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
              expect(Directory(screen.inputPath).absolute.path, input);
              childResponse.complete(
                http.Response('{"detail":"missing directory"}', 400),
              );
              await startup.settleStartup(tester);
              expect(find.textContaining('无法读取输入文件夹预览'), findsNothing);
            } else if (mode != 'all') {
              expect(backend.previewPaths, contains(camera));
              childResponse.complete(media(camera, 'new-child.jpg'));
              await startup.settleStartup(tester);
              screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
              expect(screen.items.single.filename, 'new-child.jpg');
              expect(screen.loading, isFalse);
            }
            backend.preview.complete(
              http.Response(
                jsonEncode([
                  {
                    'filename': 'new-root.jpg',
                    'path': '$sibling/new-root.jpg',
                    'file_type': 'jpg',
                  },
                  if (mode != 'removed-child')
                    {
                      'filename': 'stale-child.jpg',
                      'path': '$camera/stale-child.jpg',
                      'file_type': 'jpg',
                    },
                ]),
                200,
              ),
            );
            await startup.settleStartup(tester);
            screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
            expect(screen.loading, isFalse);
            expect(
              screen.items.map((item) => item.filename),
              mode == 'removed-child'
                  ? ['new-root.jpg']
                  : mode == 'all'
                  ? unorderedEquals(['new-root.jpg', 'stale-child.jpg'])
                  : ['new-child.jpg'],
            );
            final persisted = await tester.runAsync(() async {
              await restartedCache.flush();
              return PreviewSnapshotCache(
                directory: cacheDirectory,
              ).read('startup-test-input');
            });
            expect(
              persisted!.items.map((item) => item.filename),
              unorderedEquals([
                'new-root.jpg',
                if (mode != 'removed-child')
                  mode == 'all' ? 'stale-child.jpg' : 'new-child.jpg',
              ]),
            );
            expect(tester.takeException(), isNull);
          } finally {
            await startup.cleanup(tester, backend);
            await tester.runAsync(() => cacheDirectory.delete(recursive: true));
          }
        },
      );
    }

    testWidgets('a late cache read cannot replace a fresh scan', (
      tester,
    ) async {
      final input = Directory('startup-test-input').absolute.path;
      final cache = _DelayedCache();
      final backend = startup.StartupBackend(previewCache: cache);
      backend.preview.complete(
        http.Response(
          jsonEncode([
            {
              'filename': 'fresh.jpg',
              'path': '$input/fresh.jpg',
              'file_type': 'jpg',
            },
          ]),
          200,
        ),
      );
      try {
        await startup.mount(tester, backend);
        await tester.tap(find.text('预览').first);
        await tester.pump();
        await startup.settleStartup(tester);
        cache.pending.complete(
          PreviewSnapshot([
            DetectionItem(
              filename: 'stale.jpg',
              path: '$input/stale.jpg',
              fileType: 'jpg',
            ),
          ], const []),
        );
        await startup.settleStartup(tester);
        final screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
        expect(screen.items.single.filename, 'fresh.jpg');
        expect(tester.takeException(), isNull);
      } finally {
        if (!cache.pending.isCompleted) cache.pending.complete(null);
        await startup.cleanup(tester, backend);
      }
    });
  });
}
