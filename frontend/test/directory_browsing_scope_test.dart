import 'dart:io';
import 'dart:convert';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/directory_browsing_scope.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';
import 'startup_loading_test.dart' as startup;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final root = Directory('scope-test-root').absolute.path;
  DetectionItem item(String relative) => DetectionItem(
    filename: relative.split('/').last,
    path: '$root/$relative',
    fileType: 'jpg',
  );

  test(
    'only terminal directories are listed while all media remain accessible',
    () {
      final first = item('camera-a/photo.jpg');
      final nested = item('camera-a/nested/photo.jpg');
      final second = item('camera-b/photo.jpg');
      final source = [first, nested, second];
      final scope = DirectoryBrowsingScope()..update(source, root);
      expect(scope.directories.map(scope.label), [
        'camera-a/nested',
        'camera-b',
      ]);
      expect(scope.items, source);
      scope.select(File(nested.path).parent.path);
      expect(scope.items, [nested]);
      expect(
        validationItemsInInputFolder(source, scope.inputPath),
        scope.items,
      );
      scope.select(File(second.path).parent.path);
      expect(scope.items, [second]);
      final cached = scope.items;
      scope.update(source, root);
      expect(identical(cached, scope.items), isTrue);
      scope.select(null);
      expect(scope.inputPath, root);
      expect(scope.items, source);
    },
  );

  test('year and site ancestors are omitted from terminal camera choices', () {
    final source = [
      item('2021/1号样地巴尔峡-寒山样区czc/2/photo.jpg'),
      item('2021/1号样地巴尔峡-寒山样区czc/10/photo.jpg'),
      item('2021/2号样地/2/photo.jpg'),
      item('2022/1号样地/2/photo.jpg'),
    ];
    final scope = DirectoryBrowsingScope()..update(source, root);
    expect(scope.directories.map(scope.label), [
      '2021/1号样地巴尔峡-寒山样区czc/10',
      '2021/1号样地巴尔峡-寒山样区czc/2',
      '2021/2号样地/2',
      '2022/1号样地/2',
    ]);
    scope.select(File(source.first.path).parent.path);
    expect(scope.items, [source.first]);
  });

  test('a newly nested media directory removes its ancestor choice', () {
    final parent = item('camera/photo.jpg');
    final child = item('camera/nested/photo.jpg');
    final scope = DirectoryBrowsingScope()..update([parent], root);
    scope.select(scope.directories.single);
    scope.update([parent, child], root);
    expect(scope.directories.map(scope.label), ['camera/nested']);
    expect(scope.selected, isNull);
    expect(scope.items, [parent, child]);
  });

  test('changed roots and removed directories clear obsolete selections', () {
    final source = [item('camera-a/photo.jpg')];
    final scope = DirectoryBrowsingScope()..update(source, root);
    scope.select(scope.directories.single);
    scope.update([], root);
    expect(scope.selected, isNull);
    scope.update(source, root);
    scope.select(scope.directories.single);
    scope.update(source, '$root-other');
    expect(scope.selected, isNull);
    expect(scope.items, isEmpty);
  });

  test('directory cache survives empty media loads and switching roots', () {
    final photo = item('camera/photo.jpg');
    final directory = File(photo.path).parent.path;
    final scope = DirectoryBrowsingScope()..update([], root);
    scope.cacheDirectories([directory], root);
    scope.select(directory);
    scope.update([], root);
    expect(scope.selected, directory);
    scope.update([photo], root);
    expect(scope.items, [photo]);
    expect(scope.selected, directory);
    scope.update([], '$root-other');
    expect(scope.directories, isEmpty);
    scope.update([], root);
    expect(scope.directories, [directory]);
    scope.select(directory);
    scope.cacheDirectories([], root);
    expect(scope.selected, isNull);
    expect(scope.directories, isEmpty);
  });

  testWidgets('directory choices work before media and during refresh', (
    tester,
  ) async {
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
    final input = Directory('startup-test-input').absolute.path;
    final camera = Directory('$input/camera-b').path;
    final backend = startup.StartupBackend()
      ..directories = [Directory('$input/camera-a').path, camera];
    await startup.mount(tester, backend);
    await tester.tap(find.text('预览').first);
    await tester.pump();
    await startup.settleStartup(tester);
    expect(backend.preview.isCompleted, isFalse);
    expect(
      tester.widget<PreviewScreen>(find.byType(PreviewScreen)).loading,
      isTrue,
    );
    expect(
      tester
          .widget<FilledButton>(
            find.byKey(const ValueKey('directory-scope-selector')),
          )
          .onPressed,
      isNotNull,
    );
    await tester.tap(find.byKey(const ValueKey('directory-scope-selector')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    await tester.pump(const Duration(milliseconds: 350));
    await tester.tap(find.text('camera-b').last);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    expect(
      tester.widget<PreviewScreen>(find.byType(PreviewScreen)).inputPath,
      camera,
    );
    final media = http.Response(
      jsonEncode([
        for (final dir in ['camera-a', 'camera-b'])
          {
            'filename': 'same.jpg',
            'path': '$input/$dir/same.jpg',
            'file_type': 'jpg',
          },
      ]),
      200,
    );
    backend.preview.complete(media);
    await startup.settleStartup(tester);
    var screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
    expect(screen.items.single.path, '$input/camera-b/same.jpg');
    backend.refreshedPreview = Completer<http.Response>();
    screen.onRefresh();
    await tester.pump();
    screen = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
    expect(screen.loading, isTrue);
    expect(screen.items.single.path, '$input/camera-b/same.jpg');
    expect(
      tester
          .widget<FilledButton>(
            find.byKey(const ValueKey('directory-scope-selector')),
          )
          .onPressed,
      isNotNull,
    );
    backend.refreshedPreview!.complete(media);
    await startup.settleStartup(tester);
    await startup.cleanup(tester, backend);
  });

  testWidgets('AppBar directory selector switches duplicate media paths', (
    tester,
  ) async {
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
    final backend = startup.StartupBackend();
    await startup.mount(tester, backend);
    final input = Directory('startup-test-input').absolute.path;
    backend.preview.complete(
      http.Response(
        jsonEncode([
          for (final directory in ['camera-a', 'camera-b'])
            {
              'filename': 'same.jpg',
              'path': '$input/$directory/same.jpg',
              'file_type': 'jpg',
            },
        ]),
        200,
      ),
    );
    await startup.settleStartup(tester);
    await tester.tap(find.text('预览').first);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    expect(
      find.descendant(of: find.byType(AppBar), matching: find.text('全部目录')),
      findsOneWidget,
    );
    final selector = tester.widget<FilledButton>(
      find.byKey(const ValueKey('directory-scope-selector')),
    );
    expect(selector.onPressed, isNotNull);
    await tester.tap(find.byKey(const ValueKey('directory-scope-selector')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    await tester.pump(const Duration(milliseconds: 350));
    await tester.pump();
    await tester.tap(find.text('camera-b').last);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    final preview = tester.widget<PreviewScreen>(find.byType(PreviewScreen));
    expect(preview.items.map((item) => item.path), [
      '$input/camera-b/same.jpg',
    ]);
    expect(
      preview.inputPath.replaceAll('\\', '/'),
      '$input/camera-b'.replaceAll('\\', '/'),
    );
    expect(tester.takeException(), isNull);
    await startup.cleanup(tester, backend);
  });

  testWidgets('interleaved camera timestamps preserve per-camera bursts', (
    tester,
  ) async {
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
    final backend = startup.StartupBackend();
    await startup.mount(tester, backend);
    final input = Directory('startup-test-input').absolute.path;
    backend.preview.complete(
      http.Response(
        jsonEncode([
          for (var second = 0; second < 2; second++)
            for (final directory in ['camera-a', 'camera-b'])
              {
                'filename': '$second.jpg',
                'path': '$input/$directory/$second.jpg',
                'file_type': 'jpg',
                'date_taken': '2026-10-10T12:00:0$second',
              },
        ]),
        200,
      ),
    );
    await startup.settleStartup(tester);
    await tester.tap(find.text('校验').first);
    await tester.pump();
    await startup.settleStartup(tester);
    await tester.tap(find.text('设置').first);
    await tester.pump();
    await startup.settleStartup(tester);
    expect(
      tester
          .widget<SettingsScreen>(find.byType(SettingsScreen))
          .autoGroupInferredBurstSize,
      2,
    );
    await startup.cleanup(tester, backend);
  });
}
