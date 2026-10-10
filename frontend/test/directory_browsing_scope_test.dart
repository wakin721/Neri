import 'dart:io';
import 'dart:convert';

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
    'same filenames stay isolated, nested paths share one browsing scope',
    () {
      final first = item('camera-a/photo.jpg');
      final nested = item('camera-a/nested/photo.jpg');
      final second = item('camera-b/photo.jpg');
      final source = [first, nested, second];
      final scope = DirectoryBrowsingScope()..update(source, root);
      expect(scope.directories.map(scope.label), [
        'camera-a',
        'camera-a/nested',
        'camera-b',
      ]);
      expect(scope.items, source);
      scope.select(File(first.path).parent.path);
      expect(scope.items, [first, nested]);
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
    final selector = tester.widget<PopupMenuButton<String>>(
      find.byType(PopupMenuButton<String>),
    );
    expect(selector.enabled, isTrue);
    await tester.tap(find.byType(PopupMenuButton<String>));
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
