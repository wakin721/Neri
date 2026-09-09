import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';

void main() {
  const dinoActionKey = Key('dinov3-component-action');

  Future<void> pumpEnvironmentPanel(
    WidgetTester tester, {
    required Map<String, dynamic> dinoStatus,
  }) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/environment/dinov3-status') {
          return http.Response(
            jsonEncode(dinoStatus),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.url.path == '/api/environment/maintenance-status') {
          return http.Response(
            jsonEncode({'state': 'idle', 'message': '', 'progress': 0}),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response('{}', 200);
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);

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
              availableModels: <ModelInfo>[],
              availableClassificationModels: <ModelInfo>[],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            autoGroupInferredBurstSize: null,
            apiClient: apiClient,
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
    await tester.tap(find.text('环境维护'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));
  }

  testWidgets('healthy DINOv3 component shows delete and preserves learned data', (
    tester,
  ) async {
    await pumpEnvironmentPanel(
      tester,
      dinoStatus: {
        'installed': true,
        'healthy': true,
        'architecture': 'DINOv3 ViT-B/16',
        'component_version': 2,
        'source_commit': '6876159a11b4df116f30f667f8c9888617df0751',
        'classifier_filename': 'multi_prototype.pt',
        'classifier_fingerprint': 'a' * 64,
        'classifier_head_type': 'multi_prototype',
        'selection_k': 3,
        'message': 'DINOv3 ViT-B/16 Multi-prototype 已安装。',
      },
    );

    expect(find.text('DINOv3 ViT-B/16'), findsOneWidget);
    expect(find.textContaining('K=3'), findsOneWidget);
    final deleteButton = find.byKey(dinoActionKey);
    expect(deleteButton, findsOneWidget);
    expect(tester.widget(deleteButton), isA<OutlinedButton>());
    expect(
      find.descendant(of: deleteButton, matching: find.text('删除')),
      findsOneWidget,
    );

    await tester.tap(deleteButton);
    await tester.pumpAndSettle();

    expect(find.text('删除 DINOv3'), findsWidgets);
    expect(find.textContaining('共享 PyTorch'), findsOneWidget);
    expect(find.textContaining('本地已学习物种'), findsOneWidget);
  });

  testWidgets('unhealthy DINOv3 component offers install repair', (tester) async {
    await pumpEnvironmentPanel(
      tester,
      dinoStatus: {
        'installed': true,
        'healthy': false,
        'architecture': 'DINOv3 ViT-B/16',
        'component_version': 2,
        'source_commit': '6876159a11b4df116f30f667f8c9888617df0751',
        'classifier_filename': null,
        'classifier_fingerprint': null,
        'classifier_head_type': null,
        'selection_k': null,
        'message': 'DINOv3 安装不完整，需要修复。',
      },
    );

    expect(find.text('DINOv3 ViT-B/16'), findsOneWidget);
    expect(find.textContaining('需要修复'), findsWidgets);
    final repairButton = find.byKey(dinoActionKey);
    expect(repairButton, findsOneWidget);
    expect(tester.widget(repairButton), isA<FilledButton>());
    expect(
      find.descendant(of: repairButton, matching: find.text('安装/修复')),
      findsOneWidget,
    );
  });

  testWidgets('missing DINOv3 component offers install', (tester) async {
    await pumpEnvironmentPanel(
      tester,
      dinoStatus: {
        'installed': false,
        'healthy': false,
        'architecture': 'DINOv3 ViT-B/16',
        'component_version': 2,
        'source_commit': '6876159a11b4df116f30f667f8c9888617df0751',
        'classifier_filename': null,
        'classifier_fingerprint': null,
        'classifier_head_type': null,
        'selection_k': null,
        'message': 'DINOv3 未安装。',
      },
    );

    expect(find.text('DINOv3 ViT-B/16'), findsOneWidget);
    final installButton = find.byKey(dinoActionKey);
    expect(installButton, findsOneWidget);
    expect(tester.widget(installButton), isA<FilledButton>());
    expect(
      find.descendant(of: installButton, matching: find.text('安装')),
      findsOneWidget,
    );
  });
}
