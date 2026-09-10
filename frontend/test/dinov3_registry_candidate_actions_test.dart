import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

const candidate = DinoV3RegistryEntry(
  id: 1,
  candidateNumber: 1,
  status: 'candidate',
  displayName: '未知物种 #1',
  commonName: '',
  scientificName: '',
  eventCount: 5,
  cameraCount: 2,
  prototypeCount: 1,
  clusterPurity: 1,
  embeddingConsistency: 1,
  conditions: <String, bool>{},
  canRegister: false,
);

const checkpoint = DinoV3RegistryEntry(
  id: -1,
  candidateNumber: 1,
  status: 'checkpoint',
  displayName: 'A',
  commonName: 'A',
  scientificName: '',
  eventCount: 0,
  cameraCount: 0,
  prototypeCount: 3,
  clusterPurity: 1,
  embeddingConsistency: 1,
  conditions: <String, bool>{},
  canRegister: false,
);

String candidateJson({String commonName = ''}) => jsonEncode(<String, dynamic>{
  'id': 1,
  'candidate_number': 1,
  'status': 'candidate',
  'display_name': commonName.isEmpty ? '未知物种 #1' : commonName,
  'common_name': commonName,
  'scientific_name': '',
  'event_count': 5,
  'camera_count': 2,
  'prototype_count': 1,
  'cluster_purity': 1.0,
  'embedding_consistency': 1.0,
  'conditions': <String, bool>{},
  'can_register': false,
});

Future<void> pumpDialog(WidgetTester tester, NeriApiClient client) async {
  await tester.binding.setSurfaceSize(const Size(1200, 800));
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: DinoV3RegistryDialog(
          apiClient: client,
          modelPath: 'head.neri.json',
          initialEntries: const <DinoV3RegistryEntry>[candidate, checkpoint],
          onContinueValidation: (_) {},
        ),
      ),
    ),
  );
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('saving a checkpoint name offers to merge the candidate', (
    tester,
  ) async {
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var mergeCalled = false;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'PATCH' &&
            request.url.path == '/api/dinov3/registry/1/identity') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          expect(body['common_name'], 'A');
          return http.Response(
            candidateJson(commonName: 'A'),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'POST' &&
            request.url.path == '/api/dinov3/registry/1/merge-checkpoint') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          expect(body['checkpoint_species'], 'A');
          mergeCalled = true;
          return http.Response(
            '{"merged":true,"registration_id":1,"checkpoint_species":"A","event_count":5}',
            200,
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response('[]', 200);
        }
        if (request.method == 'GET' &&
            request.url.path == '/api/dinov3/registry/catalog') {
          return http.Response('[]', 200);
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    await pumpDialog(tester, client);
    await tester.enterText(find.byType(TextField).first, 'A');
    await tester.tap(find.text('保存物种名称'));
    await tester.pumpAndSettle();

    expect(find.text('Checkpoint 已存在“A”'), findsOneWidget);
    expect(find.text('合并到已有物种'), findsOneWidget);

    await tester.tap(find.text('合并到已有物种'));
    await tester.pumpAndSettle();

    expect(mergeCalled, isTrue);
  });

  testWidgets('candidate can be marked empty or false detection beside save', (
    tester,
  ) async {
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var emptyCalled = false;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'POST' &&
            request.url.path == '/api/dinov3/registry/1/empty') {
          emptyCalled = true;
          return http.Response(
            '{"discarded":true,"registration_id":1,"event_count":5}',
            200,
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response('[]', 200);
        }
        if (request.method == 'GET' &&
            request.url.path == '/api/dinov3/registry/catalog') {
          return http.Response('[]', 200);
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    await pumpDialog(tester, client);

    expect(find.text('空/误检'), findsOneWidget);
    await tester.tap(find.text('空/误检'));
    await tester.pumpAndSettle();

    expect(find.text('标记为空/误检？'), findsOneWidget);
    await tester.tap(find.text('确认空/误检'));
    await tester.pumpAndSettle();

    expect(emptyCalled, isTrue);
  });
}
