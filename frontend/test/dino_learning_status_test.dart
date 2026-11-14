import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/utils/detection_box_label.dart';
import 'package:neri_flutter/src/widgets/dino_learning_status.dart';

http.Response response(Map<String, dynamic> data, [int status = 200]) =>
    http.Response.bytes(
      utf8.encode(jsonEncode(data)),
      status,
      headers: {'content-type': 'application/json; charset=utf-8'},
    );

void main() {
  test('internal event tracks never cover the species label', () {
    final box = DetectionBox(
      species: '马',
      confidence: 0.91,
      bbox: const [0, 0, 1, 1],
      trackId: 'event:${'a' * 64}:2025-04-23T11:04:48+00:00:track_02',
    );
    expect(detectionBoxLabel(box, 2), '#3 马 0.91');
    expect(
      detectionBoxLabel(
        const DetectionBox(species: '马', trackId: '42', bbox: [0, 0, 1, 1]),
        0,
      ),
      '#42 马 N/A',
    );
  });

  testWidgets('settings show the selected classifier learning status', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1400, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dino/learning/status') {
          expect(
            jsonDecode(request.body)['classification_model_path'],
            'active.neri.json',
          );
          return response({
            'state': 'ready',
            'learned_count': 12,
            'pending_count': 0,
            'class_count': 46,
            'learned_species_counts': {'马': 12},
            'calibration_available': true,
            'rejection': {
              'box_threshold': -0.993,
              'sequence_threshold': -0.332,
            },
          });
        }
        if (request.url.path.endsWith('/classes')) {
          return http.Response('[]', 200);
        }
        return response({});
      }),
    );
    addTearDown(api.close);
    final theme = ValueNotifier(const ThemeSettings());
    addTearDown(theme.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: NeriSettings.fromJson({
              'selected_classification_model': 'active.neri.json',
              'available_classification_models': [
                {
                  'name': 'Neri1.1',
                  'path': 'active.neri.json',
                  'backend': 'dinov3',
                },
              ],
            }),
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
    await tester.pumpAndSettle();
    expect(find.text('当前模型学习状态'), findsOneWidget);
    expect(find.text('学习已完成'), findsOneWidget);
    expect(find.text('DINO 拒识门限（只读）'), findsOneWidget);
    expect(find.text('单框门限 -0.993  ·  序列门限 -0.332'), findsOneWidget);
    expect(find.textContaining('已学习 12 个独立样本'), findsOneWidget);
    expect(find.text('更新识别模型'), findsNothing);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets(
    'automatically refreshes running to completed and stops when inactive',
    (tester) async {
      var requests = 0;
      var state = 'running';
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          requests++;
          return response({
            'state': state,
            'pending_count': state == 'ready' ? 0 : 2,
            'learned_count': state == 'ready' ? 2 : 0,
            'class_count': 45,
            'calibration_available': true,
          });
        }),
      );
      addTearDown(api.close);
      Widget screen(bool active) => MaterialApp(
        home: Scaffold(
          body: DinoLearningStatus(
            apiClient: api,
            modelPath: 'model.neri.json',
            active: active,
          ),
        ),
      );
      await tester.pumpWidget(screen(true));
      await tester.pump();
      expect(find.text('正在学习'), findsOneWidget);
      state = 'ready';
      await tester.pump(const Duration(seconds: 3));
      await tester.pumpAndSettle();
      expect(find.text('学习已完成'), findsOneWidget);
      expect(find.textContaining('已学习 2 个独立样本'), findsOneWidget);
      await tester.pumpWidget(screen(false));
      final before = requests;
      await tester.pump(const Duration(seconds: 6));
      expect(requests, before);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'late response from a previous model cannot overwrite current state',
    (tester) async {
      final old = Completer<http.Response>();
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          if (jsonDecode(request.body)['classification_model_path'] ==
              'old.neri.json') {
            return old.future;
          }
          return response({
            'state': 'ready',
            'learned_count': 9,
            'class_count': 45,
            'pending_count': 0,
          });
        }),
      );
      addTearDown(api.close);
      Widget screen(String model) => MaterialApp(
        home: Scaffold(
          body: DinoLearningStatus(apiClient: api, modelPath: model),
        ),
      );
      await tester.pumpWidget(screen('old.neri.json'));
      await tester.pumpWidget(screen('new.neri.json'));
      await tester.pumpAndSettle();
      old.complete(response({'state': 'failed', 'learned_count': 1}));
      await tester.pumpAndSettle();
      expect(find.textContaining('已学习 9 个独立样本'), findsOneWidget);
      expect(find.text('学习失败'), findsNothing);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );
}
