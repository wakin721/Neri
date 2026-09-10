import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

Future<void> pumpBrief(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 350));
}

void main() {
  test('NeriApp wires a DINOv3 startup status check', () {
    final source = File('lib/main.dart').readAsStringSync();
    expect(source, contains('DinoV3StartupCheck('));
  });

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

    const entry = DinoV3RegistryEntry(
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

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: const <DinoV3RegistryEntry>[entry],
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

  test('DINO box feedback panel includes visible index and species prefix', () {
    final source = File(
      'lib/src/screens/species_validation_screen.dart',
    ).readAsStringSync();
    expect(
      source,
      contains('_buildDinoFeedbackPanel(selectedDinoBox, visibleBoxes)'),
    );
    expect(source, contains("Text('#\$boxNumber \$species · ')"));
  });
}
