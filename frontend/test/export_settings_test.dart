import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/export_result.dart';

void main() {
  test('export format setting asks for missing and invalid values', () {
    expect(normalizeExportFormatMode(null), 'ask');
    expect(normalizeExportFormatMode('unknown'), 'ask');
    expect(normalizeExportFormatMode('ask'), 'ask');
    expect(normalizeExportFormatMode('csv'), 'csv');
    expect(normalizeExportFormatMode('excel'), 'excel');
  });

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
}
