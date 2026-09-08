import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

void main() {
  test('ModelInfo parses DINOv3 capability metadata', () {
    final model = ModelInfo.fromJson(const <String, dynamic>{
      'name': 'DINOv3 reviewed',
      'path': 'head.neri.json',
      'kind': 'cls',
      'backend': 'dinov3',
      'architecture': 'dinov3_vitb16',
      'feature_dim': 768,
      'requires_detector': true,
      'supports_video_fast': true,
      'supports_video_all': false,
      'checkpoint_path': 'head.pt',
    });

    expect(model.isDinoV3, isTrue);
    expect(model.featureDim, 768);
    expect(model.requiresDetector, isTrue);
    expect(model.supportsVideoAll, isFalse);
    expect(model.checkpointPath, 'head.pt');
  });

  test('DINOv3 registry models parse conditions and events', () {
    final entry = DinoV3RegistryEntry.fromJson(const <String, dynamic>{
      'id': 17,
      'candidate_number': 17,
      'status': 'candidate',
      'display_name': '未知物种 #17',
      'common_name': '豹猫',
      'scientific_name': 'Prionailurus bengalensis',
      'event_count': 5,
      'camera_count': 3,
      'prototype_count': 1,
      'cluster_purity': 0.95,
      'embedding_consistency': 0.88,
      'conditions': <String, dynamic>{
        'events': true,
        'cameras': true,
        'cluster_purity': true,
        'embedding_consistency': true,
        'identity': true,
      },
      'can_register': true,
    });
    final event = DinoV3RegistryEvent.fromJson(const <String, dynamic>{
      'event_key': 'event-1',
      'source_path': '/survey/camera-a/1.jpg',
      'camera_id': 'camera-a',
      'sample_count': 2,
      'timestamp_missing': false,
    });

    expect(entry.canRegister, isTrue);
    expect(entry.conditions['events'], isTrue);
    expect(event.sourcePath, '/survey/camera-a/1.jpg');
    expect(event.sampleCount, 2);
  });

  testWidgets('settings shows DINOv3 registration state only for DINOv3 model', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dinov3/registry') {
          return http.Response(
            '[{"id":1,"candidate_number":1,"status":"candidate","display_name":"未知物种 #1","common_name":"","scientific_name":"","event_count":4,"camera_count":2,"prototype_count":1,"cluster_purity":1.0,"embedding_consistency":1.0,"conditions":{},"can_register":false}]',
            200,
          );
        }
        if (request.url.path == '/api/models/classes') {
          return http.Response('[]', 200);
        }
        return http.Response('{}', 200);
      }),
    );
    final theme = ValueNotifier(const ThemeSettings());
    addTearDown(client.close);
    addTearDown(theme.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: <ModelInfo>[
                ModelInfo(name: 'det.pt', path: 'det.pt'),
              ],
              availableClassificationModels: <ModelInfo>[
                ModelInfo(
                  name: 'DINOv3',
                  path: 'head.neri.json',
                  kind: 'cls',
                  backend: 'dinov3',
                  requiresDetector: true,
                  supportsVideoAll: false,
                ),
              ],
              selectedModel: 'det.pt',
              selectedClassificationModel: 'head.neri.json',
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{
                'selected_model': 'det.pt',
                'selected_classification_model': 'head.neri.json',
              },
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            autoGroupInferredBurstSize: null,
            apiClient: client,
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
    await tester.pumpAndSettle();

    expect(find.text('物种注册状态'), findsOneWidget);
    expect(find.textContaining('Candidate 1'), findsOneWidget);
  });

  testWidgets('continue validation returns representative event paths', (
    tester,
  ) async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path.endsWith('/events')) {
          return http.Response(
            '[{"event_key":"a","source_path":"/a.jpg","camera_id":"cam-a","timestamp_missing":false,"sample_count":1},{"event_key":"b","source_path":"/b.jpg","camera_id":"cam-b","timestamp_missing":false,"sample_count":1}]',
            200,
          );
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);
    Set<String>? paths;
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
            onContinueValidation: (value) => paths = value,
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.tap(find.text('未知物种 #1').first);
    await tester.pump();
    await tester.tap(find.text('继续验证'));
    await tester.pumpAndSettle();

    expect(paths, <String>{'/a.jpg', '/b.jpg'});
  });
}
