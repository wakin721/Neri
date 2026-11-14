import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/export_result.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';

void main() {
  test(
    'export format setting defaults to CSV for missing and invalid values',
    () {
      expect(normalizeExportFormatMode(null), 'csv');
      expect(normalizeExportFormatMode('unknown'), 'csv');
      expect(normalizeExportFormatMode('ask'), 'ask');
      expect(normalizeExportFormatMode('csv'), 'csv');
      expect(normalizeExportFormatMode('excel'), 'excel');
    },
  );

  test('export result carries every output and supports old responses', () {
    final batch = ValidationExportResult.fromJson({
      'output_path': 'a.csv',
      'output_paths': ['a.csv', 'b.csv'],
      'file_format': 'csv',
      'exported_count': 2,
    });
    expect(batch.outputPaths, ['a.csv', 'b.csv']);
    final legacy = ValidationExportResult.fromJson({
      'output_path': 'a.csv',
      'file_format': 'csv',
      'exported_count': 1,
    });
    expect(legacy.outputPaths, ['a.csv']);
  });

  test('API submits per-directory export scope', () async {
    Map<String, dynamic>? body;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode({
            'output_path': 'a.csv',
            'output_paths': ['a.csv', 'b.csv'],
            'file_format': 'csv',
            'exported_count': 2,
          }),
          200,
        );
      }),
    );
    addTearDown(api.close);
    final result = await api.exportValidationData(
      inputPath: 'root',
      fileFormat: 'csv',
      perDirectory: true,
    );
    expect(body?['per_directory'], true);
    expect(result.outputPaths, ['a.csv', 'b.csv']);
  });

  for (final batch in [false, true]) {
    test('confirmation API preserves filters for batch=$batch', () async {
      Map<String, dynamic>? body;
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          body = jsonDecode(request.body) as Map<String, dynamic>;
          final item = {
            'filename': 'a.jpg',
            'path': 'root/a.jpg',
            'file_type': 'jpg',
          };
          return http.Response(jsonEncode(batch ? [item] : item), 200);
        }),
      );
      addTearDown(api.close);
      if (batch) {
        await api.markValidationItems(
          inputPath: 'root',
          filePaths: ['root/a.jpg'],
          action: 'correct',
          confidenceSettings: {'global': 0.8, 'fox': 0.9},
          minFrameRatio: 0.3,
        );
      } else {
        await api.markValidationItem(
          inputPath: 'root',
          filePath: 'root/a.jpg',
          action: 'correct',
          confidenceSettings: {'global': 0.8, 'fox': 0.9},
          minFrameRatio: 0.3,
        );
      }
      expect(body?['confidence_settings'], {'global': 0.8, 'fox': 0.9});
      expect(body?['min_frame_ratio'], 0.3);
    });
  }

  testWidgets('basic settings saves selected export format', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final api = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
    );
    final theme = ValueNotifier(const ThemeSettings());
    addTearDown(api.close);
    addTearDown(theme.dispose);
    Map<String, dynamic>? saved;
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
            apiClient: api,
            themeNotifier: theme,
            onUpdateTheme: (_) {},
            closeBehavior: 'ask',
            onCloseBehaviorChanged: (_) {},
            onSaveSettings: (settings) async => saved = settings,
            onCheckForUpdates:
                ({required channel, required downloadSource}) async {},
            onShowMessage: (_) {},
          ),
        ),
      ),
    );
    await tester.tap(find.text('基础设置').first);
    await tester.pumpAndSettle();
    expect(find.text('导出格式'), findsOneWidget);
    final formatButton = find.widgetWithText(TextButton, 'CSV').first;
    await tester.ensureVisible(formatButton);
    await tester.pumpAndSettle();
    await tester.tap(formatButton);
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(MenuItemButton, '每次询问'));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(TextButton, '每次询问'), findsWidgets);
    await tester.pump(const Duration(seconds: 1));
    expect(saved?['export_format_mode'], 'ask');

    final columnsButton = find.text('全部列').first;
    await tester.ensureVisible(columnsButton);
    await tester.tap(columnsButton);
    await tester.pumpAndSettle();
    expect(find.text('选择导出表格列'), findsOneWidget);
    await tester.tap(find.widgetWithText(TextButton, '清空'));
    await tester.pumpAndSettle();
    final confirm = find.widgetWithText(FilledButton, '确定');
    expect(tester.widget<FilledButton>(confirm).onPressed, isNull);
    expect(find.text('请至少选择一列后再保存。'), findsOneWidget);
    await tester.tap(find.widgetWithText(CheckboxListTile, '相对路径'));
    await tester.pumpAndSettle();
    expect(tester.widget<FilledButton>(confirm).onPressed, isNotNull);
    await tester.tap(confirm);
    await tester.pumpAndSettle();
    await tester.pump(const Duration(seconds: 1));
    expect(saved?['export_columns'], ['相对路径']);
  });
}
