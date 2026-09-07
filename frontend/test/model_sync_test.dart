import 'dart:async';
import 'dart:convert';

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
    const sync = ModelInfo(
      name: 'bird.pt',
      path: r'C:\Neri\res\model\detect\sync\bird.pt',
      source: 'sync',
      kind: 'detect',
    );

    expect(user.sourceLabel, '用户模型');
    expect(sync.sourceLabel, 'NeriCloud');
    expect(user.displayName, '用户模型 / bird.pt');
    expect(sync.displayName, 'NeriCloud / bird.pt');
  });

  test('settings JSON defaults to lowercase canonical model roots', () {
    final settings = NeriSettings.fromJson(const <String, dynamic>{});
    expect(settings.modelDirectory, 'res/model/detect');
    expect(settings.classificationModelDirectory, 'res/model/cls');
  });

  test('model sync status parses active download progress', () {
    final status = ModelSyncStatus.fromJson(const <String, dynamic>{
      'state': 'downloading',
      'run_id': 'run-1',
      'current_file': 'detect/bird.pt',
      'total_files': 3,
      'completed_files': 1,
      'received_bytes': 25,
      'total_bytes': 100,
      'cloud_detect_count': 4,
      'cloud_cls_count': 2,
    });

    expect(status.isActive, isTrue);
    expect(status.progress, 0.25);
    expect(status.cloudDetectCount, 4);
    expect(status.cloudClsCount, 2);
  });

  test('model sync API reads status and starts a manual run', () async {
    final requests = <String>[];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requests.add('${request.method} ${request.url.path}');
        if (request.method == 'GET') {
          return http.Response(
            jsonEncode(<String, dynamic>{
              'state': 'completed',
              'run_id': 'startup',
              'cloud_detect_count': 2,
              'cloud_cls_count': 1,
            }),
            200,
            headers: const {'content-type': 'application/json'},
          );
        }
        return http.Response(
          jsonEncode(<String, dynamic>{
            'state': 'checking',
            'run_id': 'manual',
          }),
          202,
          headers: const {'content-type': 'application/json'},
        );
      }),
    );
    addTearDown(client.close);

    final status = await client.fetchModelSyncStatus();
    final started = await client.runModelSync();

    expect(status.state, 'completed');
    expect(started.state, 'checking');
    expect(requests, <String>[
      'GET /api/model-sync/status',
      'POST /api/model-sync/run',
    ]);
  });

  testWidgets('duplicate model filenames show user and NeriCloud sources', (
    tester,
  ) async {
    const userPath = r'C:\Neri\res\model\detect\user\bird.pt';
    const syncPath = r'C:\Neri\res\model\detect\sync\bird.pt';
    const settings = NeriSettings(
      appTitle: 'Neri',
      appVersion: 'test',
      supportedImageExtensions: <String>['.jpg'],
      supportedVideoExtensions: <String>['.mp4'],
      modelDirectory: r'C:\Neri\res\model\detect',
      classificationModelDirectory: r'C:\Neri\res\model\cls',
      availableModels: <ModelInfo>[
        ModelInfo(
          name: 'bird.pt',
          path: userPath,
          source: 'user',
          kind: 'detect',
        ),
        ModelInfo(
          name: 'bird.pt',
          path: syncPath,
          source: 'sync',
          kind: 'detect',
        ),
      ],
      availableClassificationModels: <ModelInfo>[],
      speciesTypes: <String, String>{},
      gpuAvailable: false,
      missingYoloDependencies: <String>[],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: settings,
            inputController: TextEditingController(),
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
      ),
    );
    await tester.pump();

    final dropdown = tester.widget<DropdownMenu<String>>(
      find.byType(DropdownMenu<String>).first,
    );
    final labels = dropdown.dropdownMenuEntries.map((entry) => entry.label).toList();
    final values = dropdown.dropdownMenuEntries.map((entry) => entry.value).toList();
    final userHeader = dropdown.dropdownMenuEntries.singleWhere(
      (entry) => entry.label == '用户模型',
    );
    final cloudHeader = dropdown.dropdownMenuEntries.singleWhere(
      (entry) => entry.label == 'NeriCloud',
    );

    expect(labels, containsAllInOrder(<String>[
      '不使用',
      '用户模型',
      'bird.pt',
      'NeriCloud',
      'bird.pt',
    ]));
    expect(values, containsAll(<String>[userPath, syncPath]));
    expect(userHeader.enabled, isFalse);
    expect(cloudHeader.enabled, isFalse);
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
            jsonEncode(<String, dynamic>{
              'state': 'checking',
              'run_id': 'run-2',
            }),
            202,
            headers: const {'content-type': 'application/json'},
          );
        }
        statusReads++;
        final active = statusReads == 1;
        return http.Response(
          jsonEncode(<String, dynamic>{
            'state': active ? 'downloading' : 'completed',
            'run_id': 'run-2',
            'received_bytes': active ? 50 : 100,
            'total_bytes': 100,
          }),
          200,
          headers: const {'content-type': 'application/json'},
        );
      }),
    );
    addTearDown(client.close);

    final controller = ModelSyncController(
      client,
      pollInterval: const Duration(milliseconds: 20),
      onCatalogChanged: () async {
        catalogRefreshes++;
      },
    );
    addTearDown(controller.dispose);

    await controller.runNow();
    await controller.refreshStatus();
    expect(controller.status?.state, 'downloading');

    await tester.pump(const Duration(milliseconds: 30));
    await tester.pump();
    await Future<void>.delayed(const Duration(milliseconds: 1));

    expect(controller.status?.state, 'completed');
    expect(catalogRefreshes, 1);

    await tester.pump(const Duration(milliseconds: 50));
    expect(catalogRefreshes, 1);
  });
}
