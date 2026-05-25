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

  Future<void> shutdownBackend() async {
    final response = await _httpClient.post(_uri('/api/shutdown'));
    _ensureSuccess(response);
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

  Future<MaintenanceStartResponse> installPytorch(
    String envChoice, {
    String packageSource = 'official',
  }) async {
    final response = await _httpClient.post(
      _uri('/api/environment/install-pytorch'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'env_choice': envChoice,
        'package_source': packageSource,
      }),
    );
    _ensureSuccess(response);
    return MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<MaintenanceStartResponse> installYoloDependencies({
    required String envChoice,
    String packageSource = 'official',
  }) async {
    final response = await _httpClient.post(
      _uri('/api/environment/install-yolo-dependencies'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'env_choice': envChoice,
        'package_source': packageSource,
      }),
    );
    _ensureSuccess(response);
    return MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<MaintenanceStartResponse> reinstallPythonPackage(
    String packageSpec,
    String packageSource,
  ) async {
    final response = await _httpClient.post(
      _uri('/api/environment/reinstall-package'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'package': packageSpec,
        'package_source': packageSource,
      }),
    );
    _ensureSuccess(response);
    return MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<MaintenanceStatus> fetchMaintenanceStatus() async {
    final response = await _httpClient.get(
      _uri('/api/environment/maintenance-status'),
    );
    _ensureSuccess(response);
    return MaintenanceStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<InstalledPackageInfo>> fetchInstalledPackages() async {
    final response = await _httpClient.get(_uri('/api/debug/packages'));
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(InstalledPackageInfo.fromJson)
        .toList();
  }

  Future<RuntimeDiagnostics> fetchRuntimeDiagnostics() async {
    final response = await _httpClient.get(_uri('/api/debug/runtime'));
    _ensureSuccess(response);
    return RuntimeDiagnostics.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<DebugLogInfo>> fetchDebugLogs() async {
    final response = await _httpClient.get(_uri('/api/debug/logs'));
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DebugLogInfo.fromJson)
        .toList();
  }

  Future<DebugLogContent> fetchDebugLogContent(String path) async {
    final uri = Uri.parse(
      '$baseUrl/api/debug/logs/content',
    ).replace(queryParameters: {'path': path});
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return DebugLogContent.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ProcessingJob> createJob({
    String? inputDir,
    List<String> inputPaths = const <String>[],
    String? outputDir,
    String? modelPath,
    String? classificationModelPath,
    required double confidence,
    required double iou,
    required bool useFp16,
    bool useAugment = true,
    bool useAgnosticNms = true,
    int batchSize = 16,
    int vidStride = 1,
    String videoMode = 'all',
    required bool enableDetection,
    List<String> selectedSpeciesNames = const <String>[],
  }) async {
    final cleanInputDir = inputDir?.trim();
    final cleanInputPaths = inputPaths
        .map((path) => path.trim())
        .where((path) => path.isNotEmpty)
        .toList();
    if ((cleanInputDir == null || cleanInputDir.isEmpty) &&
        cleanInputPaths.isEmpty) {
      throw ArgumentError('inputDir 或 inputPaths 至少需要提供一个。');
    }

    final response = await _httpClient.post(
      _uri('/api/jobs'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        if (cleanInputDir != null && cleanInputDir.isNotEmpty)
          'input_dir': cleanInputDir,
        if (cleanInputPaths.isNotEmpty) 'input_paths': cleanInputPaths,
        'output_dir': (outputDir?.isEmpty ?? true) ? null : outputDir,
        'options': {
          'model_path': (modelPath?.isEmpty ?? true) ? null : modelPath,
          'classification_model_path':
              (classificationModelPath?.isEmpty ?? true)
              ? null
              : classificationModelPath,
          'confidence': confidence,
          'iou': iou,
          'use_fp16': useFp16,
          'use_augment': useAugment,
          'use_agnostic_nms': useAgnosticNms,
          'batch_size': batchSize,
          'vid_stride': vidStride,
          'video_mode': videoMode,
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
    bool includeCached = true,
  }) async {
    final uri = Uri.parse('$baseUrl/api/preview').replace(
      queryParameters: {
        'input_path': inputPath,
        'include_cached': includeCached.toString(),
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

  Future<List<DetectionItem>> markValidationItems({
    required String inputPath,
    required List<String> filePaths,
    required String action,
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final response = await _httpClient.post(
      _uri('/api/validation/mark/batch'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'input_path': inputPath,
        'file_paths': filePaths,
        'action': action,
        if (speciesName != null) 'species_name': speciesName,
        if (speciesCount != null) 'species_count': speciesCount,
        if (speciesType != null) 'species_type': speciesType,
        if (remark != null) 'remark': remark,
      }),
    );
    try {
      _ensureSuccess(response);
    } on ApiException catch (error) {
      if (error.statusCode != 404) rethrow;
      final updatedItems = <DetectionItem>[];
      for (final filePath in filePaths) {
        updatedItems.add(
          await markValidationItem(
            inputPath: inputPath,
            filePath: filePath,
            action: action,
            speciesName: speciesName,
            speciesCount: speciesCount,
            speciesType: speciesType,
            remark: remark,
          ),
        );
      }
      return updatedItems;
    }
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DetectionItem.fromJson)
        .toList();
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

  Future<List<ProcessingJob>> listJobs({bool includeResults = true}) async {
    final uri = includeResults
        ? _uri('/api/jobs')
        : _uri(
            '/api/jobs',
          ).replace(queryParameters: const {'include_results': 'false'});
    final response = await _httpClient.get(uri);
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

  Future<ProcessingJob> cancelJob(String id) async {
    final response = await _httpClient.post(_uri('/api/jobs/$id/cancel'));
    _ensureSuccess(response);
    return ProcessingJob.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ProcessingJob> resumeJob(String id) async {
    final response = await _httpClient.post(_uri('/api/jobs/$id/resume'));
    _ensureSuccess(response);
    return ProcessingJob.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> deleteJob(String id) async {
    final response = await _httpClient.delete(_uri('/api/jobs/$id'));
    _ensureSuccess(response);
  }

  Future<void> clearJobs() async {
    final response = await _httpClient.delete(_uri('/api/jobs'));
    _ensureSuccess(response);
  }

  void close() => _httpClient.close();

  void _ensureSuccess(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      String message = 'HTTP ${response.statusCode}: ${response.body}';
      String? code;
      List<String> missingDependencies = const <String>[];
      try {
        final decoded = jsonDecode(response.body);
        if (decoded is Map<String, dynamic>) {
          final detail = decoded['detail'];
          if (detail is Map<String, dynamic>) {
            code = detail['code']?.toString();
            final detailMessage = detail['message']?.toString();
            if (detailMessage != null && detailMessage.isNotEmpty) {
              message = detailMessage;
            }
            final missing = detail['missing'];
            if (missing is List) {
              missingDependencies = missing
                  .map((item) => item.toString())
                  .where((item) => item.isNotEmpty)
                  .toList();
            }
          } else if (detail is String && detail.isNotEmpty) {
            message = detail;
          }
        }
      } catch (_) {}
      throw ApiException(
        message,
        statusCode: response.statusCode,
        code: code,
        missingDependencies: missingDependencies,
      );
    }
  }
}

class ApiException implements Exception {
  const ApiException(
    this.message, {
    this.statusCode,
    this.code,
    this.missingDependencies = const <String>[],
  });

  final String message;
  final int? statusCode;
  final String? code;
  final List<String> missingDependencies;

  bool get isMissingYoloDependencies =>
      statusCode == 409 && code == 'missing_yolo_dependencies';

  @override
  String toString() => message;
}

class MaintenanceStartResponse {
  const MaintenanceStartResponse({
    required this.accepted,
    required this.operation,
    required this.message,
  });

  factory MaintenanceStartResponse.fromJson(Map<String, dynamic> json) {
    return MaintenanceStartResponse(
      accepted: json['accepted'] as bool? ?? false,
      operation: json['operation'] as String? ?? '',
      message: json['message'] as String? ?? '',
    );
  }

  final bool accepted;
  final String operation;
  final String message;
}

class MaintenanceStatus {
  const MaintenanceStatus({
    required this.state,
    required this.message,
    this.operation,
    this.logPath,
    this.error,
  });

  factory MaintenanceStatus.fromJson(Map<String, dynamic> json) {
    return MaintenanceStatus(
      operation: json['operation'] as String?,
      state: json['state'] as String? ?? 'idle',
      message: json['message'] as String? ?? '',
      logPath: json['log_path'] as String?,
      error: json['error'] as String?,
    );
  }

  final String? operation;
  final String state;
  final String message;
  final String? logPath;
  final String? error;
}

class InstalledPackageInfo {
  const InstalledPackageInfo({required this.name, required this.version});

  factory InstalledPackageInfo.fromJson(Map<String, dynamic> json) {
    return InstalledPackageInfo(
      name: json['name'] as String? ?? '',
      version: json['version'] as String? ?? '',
    );
  }

  final String name;
  final String version;
}

class RuntimeDiagnostics {
  const RuntimeDiagnostics({
    required this.pytorchInstalled,
    required this.gpuAvailable,
    this.backendVersion = '',
    this.pytorchVersion,
    this.pytorchCudaVersion,
    this.gpuDevices = const <String>[],
    this.hardwareGpus = const <String>[],
    this.error,
  });

  factory RuntimeDiagnostics.fromJson(Map<String, dynamic> json) {
    return RuntimeDiagnostics(
      backendVersion: json['backend_version'] as String? ?? '',
      pytorchInstalled: json['pytorch_installed'] as bool? ?? false,
      pytorchVersion: json['pytorch_version'] as String?,
      pytorchCudaVersion: json['pytorch_cuda_version'] as String?,
      gpuAvailable: json['gpu_available'] as bool? ?? false,
      gpuDevices: (json['gpu_devices'] as List<dynamic>? ?? const <dynamic>[])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList(),
      hardwareGpus:
          (json['hardware_gpus'] as List<dynamic>? ?? const <dynamic>[])
              .map((item) => item.toString())
              .where((item) => item.isNotEmpty)
              .toList(),
      error: json['error'] as String?,
    );
  }

  final bool pytorchInstalled;
  final String backendVersion;
  final String? pytorchVersion;
  final String? pytorchCudaVersion;
  final bool gpuAvailable;
  final List<String> gpuDevices;
  final List<String> hardwareGpus;
  final String? error;
}

class DebugLogInfo {
  const DebugLogInfo({
    required this.name,
    required this.path,
    required this.sizeBytes,
    this.modifiedAt,
  });

  factory DebugLogInfo.fromJson(Map<String, dynamic> json) {
    return DebugLogInfo(
      name: json['name'] as String? ?? '',
      path: json['path'] as String? ?? '',
      sizeBytes: json['size_bytes'] as int? ?? 0,
      modifiedAt: json['modified_at'] as String?,
    );
  }

  final String name;
  final String path;
  final int sizeBytes;
  final String? modifiedAt;
}

class DebugLogContent extends DebugLogInfo {
  const DebugLogContent({
    required super.name,
    required super.path,
    required super.sizeBytes,
    required this.content,
    super.modifiedAt,
    this.truncated = false,
  });

  factory DebugLogContent.fromJson(Map<String, dynamic> json) {
    return DebugLogContent(
      name: json['name'] as String? ?? '',
      path: json['path'] as String? ?? '',
      sizeBytes: json['size_bytes'] as int? ?? 0,
      modifiedAt: json['modified_at'] as String?,
      content: json['content'] as String? ?? '',
      truncated: json['truncated'] as bool? ?? false,
    );
  }

  final String content;
  final bool truncated;
}
