import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/main_window.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/privacy/privacy_status.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/utils/preview_snapshot_cache.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'startup_loading_test.dart' show settleStartup, flushPreviewCache;

http.Response response(Object value) => http.Response(
  jsonEncode(value),
  200,
  headers: {'content-type': 'application/json; charset=utf-8'},
);

Map<String, dynamic> settingsData(bool gpu, {int? batchSize = 16}) => {
  'gpu_available': gpu,
  'selected_model': 'model.pt',
  'available_models': [
    {'path': 'model.pt', 'name': 'Model'},
  ],
  'settings': {
    if (batchSize != null) 'batch_size': batchSize,
    'thread_count': 1,
    'preview_cache_max_mb': 0,
  },
};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'thread control is replaced by automatic scheduling and not saved',
    (tester) async {
      await tester.binding.setSurfaceSize(const Size(1400, 1100));
      addTearDown(() => tester.binding.setSurfaceSize(null));
      final api = NeriApiClient(
        httpClient: MockClient((_) async => response({})),
      );
      final theme = ValueNotifier(const ThemeSettings());
      addTearDown(api.close);
      addTearDown(theme.dispose);
      Map<String, dynamic>? saved;
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SettingsScreen(
              settings: NeriSettings.fromJson(settingsData(true)),
              autoGroupInferredBurstSize: null,
              apiClient: api,
              themeNotifier: theme,
              onUpdateTheme: (_) {},
              closeBehavior: 'ask',
              onCloseBehaviorChanged: (_) {},
              onSaveSettings: (value) async => saved = value,
              onCheckForUpdates:
                  ({required channel, required downloadSource}) async {},
              onShowMessage: (_) {},
            ),
          ),
        ),
      );
      await tester.tap(find.text('检测设置').first);
      await tester.pumpAndSettle();
      expect(find.text('线程数'), findsNothing);
      expect(find.textContaining('处理线程由软件根据硬件和任务自动安排'), findsOneWidget);
      final slider = tester.widget<Slider>(
        find.byWidgetPredicate(
          (widget) => widget is Slider && widget.max == 32,
        ),
      );
      expect(slider.onChanged, isNotNull);
      expect(slider.min, 0);
      expect(slider.value, 16);
      slider.onChanged!(8);
      await tester.pump(const Duration(seconds: 1));
      expect(saved?['batch_size'], 8);
      expect(saved?.containsKey('thread_count'), isFalse);
      slider.onChanged!(0);
      await tester.pump(const Duration(seconds: 1));
      expect(saved?['batch_size'], 0);
      expect(find.text('自动'), findsOneWidget);
      await tester.pumpWidget(const SizedBox());
    },
  );

  for (final gpu in [true, false]) {
    testWidgets(
      'batch selection defaults to automatic on ${gpu ? 'GPU' : 'CPU'}',
      (tester) async {
        await tester.binding.setSurfaceSize(const Size(1400, 1100));
        addTearDown(() => tester.binding.setSurfaceSize(null));
        final api = NeriApiClient(
          httpClient: MockClient((_) async => response({})),
        );
        final theme = ValueNotifier(const ThemeSettings());
        addTearDown(api.close);
        addTearDown(theme.dispose);
        Future<void> showSettings(int? batchSize) async {
          await tester.pumpWidget(
            MaterialApp(
              home: Scaffold(
                body: SettingsScreen(
                  key: ValueKey(batchSize),
                  settings: NeriSettings.fromJson(
                    settingsData(gpu, batchSize: batchSize),
                  ),
                  autoGroupInferredBurstSize: null,
                  apiClient: api,
                  themeNotifier: theme,
                  onUpdateTheme: (_) {},
                  closeBehavior: 'ask',
                  onCloseBehaviorChanged: (_) {},
                  onSaveSettings: (_) async {},
                  onCheckForUpdates:
                      ({required channel, required downloadSource}) async {},
                  onShowMessage: (_) {},
                ),
              ),
            ),
          );
          await tester.tap(find.text('检测设置').first);
          await tester.pumpAndSettle();
          final slider = tester.widget<Slider>(
            find.byWidgetPredicate(
              (widget) => widget is Slider && widget.max == 32,
            ),
          );
          expect(slider.min, 0);
          expect(slider.value, 0);
          expect(slider.label, gpu ? '自动' : '自动（=1）');
          expect(slider.onChanged, gpu ? isNotNull : isNull);
        }

        await showSettings(null);
        // Restore a saved automatic choice; CPU also ignores old manual values.
        await showSettings(gpu ? 0 : 16);
        await tester.pumpWidget(const SizedBox());
      },
    );
  }

  for (final scenario in [
    (gpu: true, batchSize: 16),
    (gpu: true, batchSize: 0),
    (gpu: true, batchSize: null),
    (gpu: false, batchSize: 16),
  ]) {
    final gpu = scenario.gpu;
    testWidgets(
      'single video redetection keeps ${gpu ? 'GPU' : 'CPU'} batching ${scenario.batchSize} and automatic threads',
      (tester) async {
        final root = Directory.systemTemp.createTempSync(
          'neri-video-settings-',
        );
        final cache = PreviewSnapshotCache(
          directory: Directory('${root.path}/cache'),
          maxBytes: 0,
        );
        final theme = ValueNotifier(const ThemeSettings());
        SharedPreferences.setMockInitialValues({'last_input_path': root.path});
        for (final channel in ['window_manager', 'neri/windows_shell']) {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .setMockMethodCallHandler(
                MethodChannel(channel),
                (call) async => call.method == 'isMaximized' ? false : null,
              );
          addTearDown(
            () => TestDefaultBinaryMessengerBinding
                .instance
                .defaultBinaryMessenger
                .setMockMethodCallHandler(MethodChannel(channel), null),
          );
        }
        Map<String, dynamic>? submitted;
        final api = NeriApiClient(
          httpClient: MockClient((request) async {
            switch (request.url.path) {
              case '/api/privacy':
                return response({
                  'agreement_version': privacyAgreementVersion,
                  'agreement_accepted': true,
                  'participation_decided': true,
                });
              case '/api/settings':
                return response(
                  settingsData(gpu, batchSize: scenario.batchSize),
                );
              case '/api/jobs':
                if (request.method == 'POST') {
                  submitted = jsonDecode(request.body) as Map<String, dynamic>;
                  return response({'id': 'video-job', 'state': 'queued'});
                }
                return response([]);
              case '/api/preview':
              case '/api/preview/directories':
              case '/api/models/classes':
                return response([]);
              default:
                return response({});
            }
          }),
        );
        tester.view.physicalSize = const Size(1400, 1100);
        tester.view.devicePixelRatio = 1;
        try {
          await tester.pumpWidget(
            MaterialApp(
              home: MainWindow(
                apiClient: api,
                themeNotifier: theme,
                previewCache: cache,
              ),
            ),
          );
          await settleStartup(tester);
          await tester.tap(find.text('校验').first);
          await settleStartup(tester);
          final screen = tester.widget<SpeciesValidationScreen>(
            find.byType(SpeciesValidationScreen),
          );
          final pending = screen.onRedetectItems([
            DetectionItem(
              filename: 'one.mp4',
              path: '${root.path}/one.mp4',
              fileType: 'mp4',
            ),
          ], confidence: 0.3);
          await settleStartup(tester);
          await pending;
          final options = submitted?['options'] as Map<String, dynamic>?;
          expect(options?['batch_size'], gpu ? scenario.batchSize ?? 0 : 0);
          expect(options?.containsKey('thread_count'), isFalse);
          expect(submitted?['input_paths'], ['${root.path}/one.mp4']);
          expect(tester.takeException(), isNull);
        } finally {
          await tester.pumpWidget(const SizedBox());
          await tester.pump(const Duration(seconds: 1));
          await flushPreviewCache(tester, cache);
          api.close();
          theme.dispose();
          await tester.runAsync(() => root.delete(recursive: true));
          tester.view.resetPhysicalSize();
          tester.view.resetDevicePixelRatio();
        }
      },
    );
  }
}
