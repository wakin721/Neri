import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models/export_result.dart';
import 'models/job.dart';
import 'models/settings.dart';

class NeriApiClient {
  NeriApiClient({
    http.Client? httpClient,
    this.baseUrl = 'http://127.0.0.1:721',
  }) : _httpClient = httpClient ?? http.Client();

  final http.Client _httpClient;
  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<bool> health() async {
    final response = await _httpClient.get(_uri('/api/health'));
    return response.statusCode == 200;
  }

  Future<NeriSettings> fetchSettings() async {
    final response = await _httpClient.get(_uri('/api/settings'));
    _ensureSuccess(response);
    return NeriSettings.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<NeriSettings> saveSettings(Map<String, dynamic> settings) async {
    final response = await _httpClient.put(
      _uri('/api/settings'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({'settings': settings}),
    );
    _ensureSuccess(response);
    return NeriSettings.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<ModelClassInfo>> fetchModelClasses(String modelPath) async {
    final uri = Uri.parse(
      '$baseUrl/api/models/classes',
    ).replace(queryParameters: {'model_path': modelPath});
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(ModelClassInfo.fromJson)
        .toList();
  }

  Future<ProcessingJob> createJob({
    required String inputDir,
    String? outputDir,
    String? modelPath,
    required double confidence,
    required double iou,
    required bool useFp16,
    required bool enableDetection,
    List<String> selectedSpeciesNames = const <String>[],
  }) async {
    final response = await _httpClient.post(
      _uri('/api/jobs'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'input_dir': inputDir,
        'output_dir': (outputDir?.isEmpty ?? true) ? null : outputDir,
        'options': {
          'model_path': (modelPath?.isEmpty ?? true) ? null : modelPath,
          'confidence': confidence,
          'iou': iou,
          'use_fp16': useFp16,
          'enable_detection': enableDetection,
          if (selectedSpeciesNames.isNotEmpty)
            'selected_species_names': selectedSpeciesNames,
        },
      }),
    );
    _ensureSuccess(response);
    return ProcessingJob.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<DetectionItem>> fetchPreviewItems({
    required String inputPath,
    String? outputDir,
  }) async {
    final uri = Uri.parse('$baseUrl/api/preview').replace(
      queryParameters: {
        'input_path': inputPath,
        if (outputDir != null && outputDir.isNotEmpty) 'output_dir': outputDir,
      },
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DetectionItem.fromJson)
        .toList();
  }

  Future<DetectionItem> fetchPreviewItem({
    required String filePath,
    String? inputPath,
    String? outputDir,
  }) async {
    final uri = Uri.parse('$baseUrl/api/preview/item').replace(
      queryParameters: {
        'file_path': filePath,
        if (inputPath != null && inputPath.isNotEmpty) 'input_path': inputPath,
        if (outputDir != null && outputDir.isNotEmpty) 'output_dir': outputDir,
      },
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return DetectionItem.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<DetectionItem> markValidationItem({
    required String inputPath,
    required String filePath,
    required String action,
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final response = await _httpClient.post(
      _uri('/api/validation/mark'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'input_path': inputPath,
        'file_path': filePath,
        'action': action,
        if (speciesName != null) 'species_name': speciesName,
        if (speciesCount != null) 'species_count': speciesCount,
        if (speciesType != null) 'species_type': speciesType,
        if (remark != null) 'remark': remark,
      }),
    );
    _ensureSuccess(response);
    return DetectionItem.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ValidationExportResult> exportValidationData({
    required String inputPath,
    required String fileFormat,
    String? outputPath,
    List<String>? columnsToExport,
    Map<String, double>? confidenceSettings,
    double minFrameRatio = 0,
  }) async {
    final response = await _httpClient.post(
      _uri('/api/validation/export'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'input_path': inputPath,
        'file_format': fileFormat,
        if (outputPath != null && outputPath.isNotEmpty)
          'output_path': outputPath,
        if (columnsToExport != null) 'columns_to_export': columnsToExport,
        'confidence_settings': confidenceSettings ?? {'global': 0.25},
        'min_frame_ratio': minFrameRatio,
      }),
    );
    _ensureSuccess(response);
    return ValidationExportResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<ProcessingJob>> listJobs() async {
    final response = await _httpClient.get(_uri('/api/jobs'));
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(ProcessingJob.fromJson)
        .toList();
  }

  Future<ProcessingJob> fetchJob(String id) async {
    final response = await _httpClient.get(_uri('/api/jobs/$id'));
    _ensureSuccess(response);
    return ProcessingJob.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  void close() => _httpClient.close();

  void _ensureSuccess(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ApiException('HTTP ${response.statusCode}: ${response.body}');
    }
  }
}

class ApiException implements Exception {
  const ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}
