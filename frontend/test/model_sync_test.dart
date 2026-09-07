import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/model_sync_controller.dart';
import 'package:neri_flutter/src/models/model_sync_status.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

void main() {
  test('model metadata exposes user and NeriCloud source labels', () {
    const user = ModelInfo(
      name: 'bird.pt',
      path: r'C:\Neri\res\model\detect\user\bird.pt',
      source: 'user',
      kind: 'detect',
    );
    const synced = ModelInfo(
      name: 'bird.pt',
      path: r'C:\Neri\res\model\detect\sync\bird.pt',
      source: 'sync',
      kind: 'detect',
    );

    expect(user.sourceLabel, '用户模型');
    expect(user.displayName, '用户模型 / bird.pt');
    expect(synced.sourceLabel, 'NeriCloud');
    expect(synced.displayName, 'NeriCloud / bird.pt');
    expect(user.path, isNot(synced.path));
  });

  test('settings JSON defaults to lowercase canonical model roots', () {
    final settings = NeriSettings.fromJson(const <String, dynamic>{});
    expect(settings.modelDirectory, 'res/model/detect');
    expect(settings.classificationModelDirectory, 'res/model/cls');
  });

  test('model sync status parses active download progress', () {
    final status = ModelSyncStatus.fromJson(<String, dynamic>{
      'state': 'downloading',
      'run_id': 'abc',
      'current_file': 'detect/bird.pt',
      'total_files': 3,
      'completed_files': 1,
      'received_bytes': 25,
      'total_bytes': 100,
      'last_successful_sync': '2026-09-07T01:00:00+00:00',
      'manifest_id': 'a' * 64,
      'error': null,
      'cloud_detect_count': 4,
      'cloud_cls_count': 2,
    });

    expect(status.isActive, isTrue);
    expect(status.progress, 0.25);
    expect(status.currentFile, 'detect/bird.pt');
    expect(status.cloudDetectCount, 4);
    expect(status.cloudClsCount, 2);
  });

  test('model sync API reads status and starts a manual run', () async {
    final requests = <http.Request>[];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requests.add(request);
        return http.Response(
          '{"state":"checking","run_id":"run-1","total_files":2,'
          '"completed_files":0,"received_bytes":0,'
          '"cloud_detect_count":1,"cloud_cls_count":1}',
          request.method == 'GET' || request.method == 'POST' ? 200 : 405,
          headers: const {'content-type': 'application/json'},
        );
      }),
    );

    final status = await client.fetchModelSyncStatus();
    final started = await client.runModelSync();

    expect(status.state, 'checking');
    expect(started.runId, 'run-1');
    expect(requests.map((request) => request.method), <String>['GET', 'POST']);
    expect(
      requests.map((request) => request.url.path),
      <String>['/api/model-sync/status', '/api/model-sync/run'],
    );
  });

  testWidgets('duplicate model filenames show user and NeriCloud sources', (
    tester,
  ) async {
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>[],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: <ModelInfo>[
                ModelInfo(
                  name: 'bird.pt',
                  path: r'C:\Neri\res\model\detect\user\bird.pt',
                  source: 'user',
                  kind: 'detect',
                ),
                ModelInfo(
                  name: 'bird.pt',
                  path: r'C:\Neri\res\model\detect\sync\bird.pt',
                  source: 'sync',
                  kind: 'detect',
                ),
              ],
              availableClassificationModels: <ModelInfo>[],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
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
            iou: 0.45,
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
      ),
    );

    final modelMenu = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .first;
    final entries = modelMenu.dropdownMenuEntries;
    final labels = entries.map((entry) => entry.label);
    expect(labels, containsAllInOrder(<String>['用户模型', 'bird.pt', 'NeriCloud', 'bird.pt']));
    expect(entries.singleWhere((entry) => entry.label == '用户模型').enabled, isFalse);
    expect(entries.singleWhere((entry) => entry.label == 'NeriCloud').enabled, isFalse);
  });

  testWidgets('sync controller refreshes catalog once when active run finishes', (
    tester,
  ) async {
    var statusReads = 0;
    var catalogRefreshes = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'POST') {
          return http.Response(
            '{"state":"checking","run_id":"run-1"}',
            202,
          );
        }
        statusReads++;
        if (statusReads == 1) {
          return http.Response(
            '{"state":"downloading","run_id":"run-1",'
            '"current_file":"detect/bird.pt","received_bytes":5,'
            '"total_bytes":10}',
            200,
          );
        }
        return http.Response(
          '{"state":"completed","run_id":"run-1",'
          '"received_bytes":10,"total_bytes":10}',
          200,
        );
      }),
    );
    final controller = ModelSyncController(
      client,
      pollInterval: const Duration(milliseconds: 10),
      onCatalogChanged: () async => catalogRefreshes++,
    );
    addTearDown(controller.dispose);

    await controller.runNow();
    await tester.pump(const Duration(milliseconds: 15));
    await tester.pump(const Duration(milliseconds: 15));
    await tester.pump(const Duration(milliseconds: 15));

    expect(controller.status?.state, 'completed');
    expect(catalogRefreshes, 1);
    expect(statusReads, greaterThanOrEqualTo(2));
  });
}
