import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/dinov3_startup_check.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

Future<void> pumpBrief(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 350));
}

DinoV3RegistryEntry registryEntry({
  int id = 1,
  int candidateNumber = 1,
  String status = 'candidate',
  String displayName = '未知物种 #1',
}) {
  return DinoV3RegistryEntry(
    id: id,
    candidateNumber: candidateNumber,
    status: status,
    displayName: displayName,
    commonName: '',
    scientificName: '',
    eventCount: 5,
    cameraCount: 2,
    prototypeCount: 1,
    clusterPurity: 1,
    embeddingConsistency: 1,
    conditions: const <String, bool>{},
    canRegister: false,
  );
}

void main() {
  test('NeriApp wires the persistent DINOv3 startup checker', () {
    final source = File('lib/main.dart').readAsStringSync();
    expect(source, contains('DinoV3StartupCheck()'));
    expect(source, contains('dinoV3StartupCheck: _dinoV3StartupCheck'));
  });

  test('startup check runs once per backend generation', () async {
    var statusRequests = 0;
    final messages = <String>[];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/environment/dinov3-status') {
          statusRequests += 1;
          return http.Response(
            '{"installed":false,"healthy":false,"message":"not installed"}',
            200,
            headers: const {'content-type': 'application/json'},
          );
        }
        return http.Response('{}', 200);
      }),
    );
    addTearDown(client.close);
    final check = DinoV3StartupCheck();

    await check.run(generation: 1, apiClient: client, onMessage: messages.add);
    await check.run(generation: 1, apiClient: client, onMessage: messages.add);

    expect(statusRequests, 1);
    expect(messages, <String>['DINOv3 尚未安装，可进入设置安装。']);

    await check.run(generation: 2, apiClient: client, onMessage: messages.add);
    expect(statusRequests, 2);
    expect(messages.length, 2);
  });

  test(
    'startup status maps unhealthy and PyTorch states without healthy noise',
    () {
      DinoV3ComponentStatus status(Map<String, dynamic> values) {
        return DinoV3ComponentStatus.fromJson(<String, dynamic>{
          'installed': true,
          'healthy': false,
          'message': '',
          ...values,
        });
      }

      expect(
        dinoV3StartupStatusMessage(
          status(<String, dynamic>{'message': "No module named 'torch'"}),
        ),
        'DINOv3 已安装，但 PyTorch 未安装，当前不可推理',
      );
      expect(
        dinoV3StartupStatusMessage(
          status(<String, dynamic>{
            'message': 'checkpoint fingerprint mismatch',
          }),
        ),
        'DINOv3 安装异常：checkpoint fingerprint mismatch',
      );
      expect(
        dinoV3StartupStatusMessage(
          status(<String, dynamic>{'healthy': true, 'message': 'ok'}),
        ),
        isNull,
      );
    },
  );

  testWidgets('registry dialog shows cropped examples and can delete a species', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var deleted = false;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response(
            '[{"id":9,"event_key":"e1","source_path":"/a.jpg","camera_id":"cam-a","timestamp_missing":false,"sample_count":1,"has_example":true}]',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/example')) {
          return http.Response.bytes(
            base64Decode(
              'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
            ),
            200,
            headers: const {'content-type': 'image/png'},
          );
        }
        if (request.method == 'DELETE' &&
            request.url.path == '/api/dinov3/registry/1') {
          deleted = true;
          return http.Response('{"deleted":true,"registration_id":1}', 200);
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: <DinoV3RegistryEntry>[registryEntry()],
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await pumpBrief(tester);

    expect(find.text('裁切例图'), findsOneWidget);
    expect(
      find.byKey(const ValueKey('dinov3-registry-example-9')),
      findsOneWidget,
    );
    expect(find.text('删除物种'), findsOneWidget);

    await tester.tap(find.text('删除物种'));
    await pumpBrief(tester);
    expect(find.textContaining('删除“未知物种 #1”'), findsOneWidget);
    await tester.tap(find.text('确认删除'));
    await pumpBrief(tester);

    expect(deleted, isTrue);
    expect(find.text('暂无候选或已注册物种'), findsOneWidget);
  });

  testWidgets('checkpoint categories do not expose registry deletion', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final client = NeriApiClient(
      httpClient: MockClient((request) async => http.Response('[]', 200)),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: <DinoV3RegistryEntry>[
              registryEntry(
                status: 'checkpoint',
                displayName: 'checkpoint 基础类别',
              ),
            ],
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await pumpBrief(tester);

    expect(find.text('删除物种'), findsNothing);
  });

  test('all local registry lifecycle states are deletable', () {
    for (final status in const <String>[
      'candidate',
      'provisional',
      'confirmed',
      'mature',
    ]) {
      expect(registryEntry(status: status).canDelete, isTrue, reason: status);
    }
    expect(registryEntry(status: 'checkpoint').canDelete, isFalse);
  });

  test('DINO box feedback title uses visible #1/#2 order and species', () {
    const boxes = <DetectionBox>[
      DetectionBox(
        species: '红腹锦鸡',
        bbox: <double>[0, 0, 10, 10],
        observationId: 'obs-1',
        predictedSpecies: '红腹锦鸡',
      ),
      DetectionBox(
        species: '豹猫',
        bbox: <double>[12, 0, 22, 10],
        observationId: 'obs-2',
        predictedSpecies: '豹猫',
      ),
    ];

    expect(dinoV3FeedbackPanelTitle(boxes[0], boxes), '#1 红腹锦鸡 · 检测框校验');
    expect(dinoV3FeedbackPanelTitle(boxes[1], boxes), '#2 豹猫 · 检测框校验');
  });
}
