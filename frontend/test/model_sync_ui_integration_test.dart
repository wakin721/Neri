import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

const _userDetectionPath =
    r'C:\Neri\res\model\detect\user\bird.pt';
const _cloudDetectionPath =
    r'C:\Neri\res\model\detect\sync\bird.pt';
const _userClassificationPath =
    r'C:\Neri\res\model\cls\user\bird.pt';
const _cloudClassificationPath =
    r'C:\Neri\res\model\cls\sync\bird.pt';

NeriSettings _settings({Map<String, dynamic> saved = const {}}) {
  return NeriSettings(
    appTitle: 'Neri',
    appVersion: 'test',
    supportedImageExtensions: const <String>['.jpg'],
    supportedVideoExtensions: const <String>['.mp4'],
    modelDirectory: 'res/model/detect',
    classificationModelDirectory: 'res/model/cls',
    availableModels: const <ModelInfo>[
      ModelInfo(
        name: 'bird.pt',
        path: _userDetectionPath,
        source: 'user',
        kind: 'detect',
      ),
      ModelInfo(
        name: 'bird.pt',
        path: _cloudDetectionPath,
        source: 'sync',
        kind: 'detect',
      ),
    ],
    availableClassificationModels: const <ModelInfo>[
      ModelInfo(
        name: 'bird.pt',
        path: _userClassificationPath,
        source: 'user',
        kind: 'cls',
      ),
      ModelInfo(
        name: 'bird.pt',
        path: _cloudClassificationPath,
        source: 'sync',
        kind: 'cls',
      ),
    ],
    speciesTypes: const <String, String>{},
    settings: saved,
    gpuAvailable: false,
    missingYoloDependencies: const <String>[],
  );
}

Widget _startScreen({NeriSettings? settings}) {
  final inputController = TextEditingController();
  return MaterialApp(
    home: Scaffold(
      body: StartScreen(
        settings: settings,
        inputController: inputController,
        selectedModelPath: '',
        onModelChanged: (_) {},
        selectedClassificationModelPath: '',
        onClassificationModelChanged: (_) {},
        videoMode: 'fast',
        onVideoModeChanged: (_) {},
        vidStride: 3,
        onVidStrideChanged: (_) {},
        useFp16: false,
        onUseFp16Changed: (_) {},
        confidence: 0.25,
        onConfidenceChanged: (_) {},
        iou: 0.30,
        onIouChanged: (_) {},
        submitting: false,
        onCreateJob: () {},
        onCancelJob: (_) {},
        onResumeJob: (_) {},
        onDeleteJob: (_) {},
        onClearJobs: () {},
        pendingStartJobIds: const <String>{},
        pendingStopJobIds: const <String>{},
        jobs: const [],
      ),
    ),
  );
}

void main() {
  testWidgets('start screen uses lowercase canonical model roots when settings are absent', (
    tester,
  ) async {
    await tester.pumpWidget(_startScreen());
    await tester.pump();

    expect(
      find.text('未在 res/model/detect 中找到 .pt 模型'),
      findsOneWidget,
    );
    expect(
      find.text('未在 res/model/cls 中找到 .pt 模型'),
      findsOneWidget,
    );
  });

  testWidgets('start screen groups duplicate model names by source', (
    tester,
  ) async {
    await tester.pumpWidget(_startScreen(settings: _settings()));
    await tester.pump();

    final dropdowns = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .toList();
    final detection = dropdowns.first;
    final labels = detection.dropdownMenuEntries.map((entry) => entry.label).toList();
    final values = detection.dropdownMenuEntries.map((entry) => entry.value).toList();

    expect(labels, containsAllInOrder(<String>['不使用', '用户模型', 'bird.pt', 'NeriCloud', 'bird.pt']));
    expect(values, containsAll(<String>[_userDetectionPath, _cloudDetectionPath]));
    expect(
      detection.dropdownMenuEntries.singleWhere((entry) => entry.label == '用户模型').enabled,
      isFalse,
    );
    expect(
      detection.dropdownMenuEntries.singleWhere((entry) => entry.label == 'NeriCloud').enabled,
      isFalse,
    );
  });

  testWidgets('settings screen shows source-grouped selectors and sync card', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/model-sync/status') {
          return http.Response(
            '{"state":"completed","run_id":"startup-run"}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(client.close);
    addTearDown(themeNotifier.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: _settings(),
            autoGroupInferredBurstSize: null,
            apiClient: client,
            themeNotifier: themeNotifier,
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
    await tester.pump();
    await tester.pump();

    expect(find.text('模型同步'), findsOneWidget);

    final detectionButton = find.widgetWithText(TextButton, '不使用').first;
    await tester.ensureVisible(detectionButton);
    await tester.tap(detectionButton);
    await tester.pumpAndSettle();

    expect(find.text('用户模型'), findsOneWidget);
    expect(find.text('NeriCloud'), findsOneWidget);
    expect(find.text('bird.pt'), findsNWidgets(2));
  });

  testWidgets('terminal startup sync refreshes the parent catalog once', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var statusReads = 0;
    var saveCalls = 0;
    Map<String, dynamic>? savedPayload;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/model-sync/status') {
          statusReads++;
          final body = statusReads == 1
              ? '{"state":"checking","run_id":"startup-run"}'
              : '{"state":"completed","run_id":"startup-run",'
                    '"cloud_detect_count":2,"cloud_cls_count":2}';
          return http.Response(
            body,
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(client.close);
    addTearDown(themeNotifier.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: _settings(saved: const <String, dynamic>{
              'selected_model': _userDetectionPath,
              'selected_classification_model': '',
            }),
            autoGroupInferredBurstSize: null,
            apiClient: client,
            themeNotifier: themeNotifier,
            onUpdateTheme: (_) {},
            closeBehavior: 'ask',
            onCloseBehaviorChanged: (_) {},
            onSaveSettings: (payload) async {
              saveCalls++;
              savedPayload = Map<String, dynamic>.from(payload);
            },
            onCheckForUpdates:
                ({required channel, required downloadSource}) async {},
            onShowMessage: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
    expect(statusReads, 1);
    expect(saveCalls, 0);

    await tester.pump(const Duration(milliseconds: 550));
    await tester.pump();

    expect(statusReads, greaterThanOrEqualTo(2));
    expect(saveCalls, 1);
    expect(savedPayload?['selected_model'], _userDetectionPath);
  });
}
