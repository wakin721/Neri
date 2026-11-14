import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'api_client_core.dart' as core;
import 'models/dino_explanation.dart';
import 'models/dino_feedback.dart';
import 'models/job.dart';
import 'models/model_sync_status.dart';

export 'api_client_core.dart' hide NeriApiClient;
export 'models/dino_explanation.dart';
export 'models/dino_feedback.dart';

class NeriApiClient extends core.NeriApiClient {
  factory NeriApiClient({
    http.Client? httpClient,
    String baseUrl = 'http://127.0.0.1:721',
  }) {
    final client = httpClient ?? http.Client();
    return NeriApiClient._(client, baseUrl);
  }

  NeriApiClient._(this._http, String baseUrl)
    : super(httpClient: _http, baseUrl: baseUrl);

  final http.Client _http;

  Future<Map<String, dynamic>> dinoLearning(
    String action,
    String modelPath, {
    String? calibrationPath,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dino/learning/$action'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': modelPath,
        if (calibrationPath != null) 'calibration_path': calibrationPath,
      }),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      String detail = '请求失败 (${response.statusCode})';
      try {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        detail = data['detail']?.toString() ?? detail;
      } catch (_) {}
      throw http.ClientException(detail);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Uri _dinoUri(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$path').replace(queryParameters: query);

  Future<ModelSyncStatus> fetchModelSyncStatus() async {
    final response = await _http.get(_dinoUri('/api/model-sync/status'));
    _ensureSuccess(response);
    return ModelSyncStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ModelSyncStatus> runModelSync() async {
    final response = await _http.post(_dinoUri('/api/model-sync/run'));
    _ensureSuccess(response);
    return ModelSyncStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<DinoComponentStatus> fetchDinoComponentStatus() async {
    final response = await _http.get(
      _dinoUri('/api/environment/dinov3-status'),
    );
    _ensureSuccess(response);
    return DinoComponentStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<core.MaintenanceStartResponse> installDinoV3({
    required String envChoice,
    String packageSource = 'auto',
  }) async {
    final response = await _http.post(
      _dinoUri('/api/environment/install-dinov3'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'env_choice': envChoice,
        'package_source': packageSource,
      }),
    );
    _ensureSuccess(response);
    return core.MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<core.MaintenanceStartResponse> removeDinoV3() async {
    final response = await _http.post(
      _dinoUri('/api/environment/remove-dinov3'),
    );
    _ensureSuccess(response);
    return core.MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  @override
  Future<DinoBoxFeedbackResult> markDinoBoxFeedback({
    required String inputPath,
    required String filePath,
    required String classificationModelPath,
    required String observationId,
    required String action,
    String? speciesName,
    required String feedbackOperationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dino/feedback/box'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'input_path': inputPath,
        'file_path': filePath,
        'classification_model_path': classificationModelPath,
        'observation_id': observationId,
        'action': action,
        if (speciesName != null && speciesName.trim().isNotEmpty)
          'species_name': speciesName.trim(),
        'feedback_operation_id': feedbackOperationId,
      }),
    );
    _ensureSuccess(response);
    return DinoBoxFeedbackResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  @override
  Future<DinoFeatureExplanation> fetchDinoFeatureExplanation(
    String classificationModelPath,
    String observationId,
  ) async {
    final response = await _http.get(
      _dinoUri(
        '/api/dino/feedback/observations/${Uri.encodeComponent(observationId)}/explain',
        {'classification_model_path': classificationModelPath},
      ),
    );
    _ensureSuccess(response);
    return DinoFeatureExplanation.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  @override
  Future<Uint8List> fetchDinoObservationExample(
    String classificationModelPath,
    String observationId,
  ) async {
    final response = await _http.get(
      _dinoUri(
        '/api/dino/feedback/observations/${Uri.encodeComponent(observationId)}/example',
        {'classification_model_path': classificationModelPath},
      ),
    );
    _ensureSuccess(response);
    return response.bodyBytes;
  }

  @override
  Future<DinoFeedbackRevertResult> revertDinoFeedback({
    required String classificationModelPath,
    required String feedbackOperationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dino/feedback/revert'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'feedback_operation_id': feedbackOperationId,
      }),
    );
    _ensureSuccess(response);
    return DinoFeedbackRevertResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  @override
  Future<DetectionItem> markValidationItem({
    required String inputPath,
    required String filePath,
    required String action,
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
    String? classificationModelPath,
    String? feedbackOperationId,
    Map<String, double>? confidenceSettings,
    double minFrameRatio = 0,
  }) async {
    final updated = await super.markValidationItem(
      inputPath: inputPath,
      filePath: filePath,
      action: action,
      speciesName: speciesName,
      speciesCount: speciesCount,
      speciesType: speciesType,
      remark: remark,
      classificationModelPath: classificationModelPath,
      feedbackOperationId: feedbackOperationId,
      confidenceSettings: confidenceSettings,
      minFrameRatio: minFrameRatio,
    );
    return updated;
  }

  void _ensureSuccess(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) return;
    throw http.ClientException(
      'Neri API 请求失败 (${response.statusCode})',
      response.request?.url,
    );
  }
}

class DinoComponentStatus {
  const DinoComponentStatus({
    required this.installed,
    required this.healthy,
    required this.architecture,
    required this.componentVersion,
    required this.displayName,
    required this.message,
    this.classCount,
    this.featureDim,
    this.manifest,
  });

  factory DinoComponentStatus.fromJson(Map<String, dynamic> json) {
    return DinoComponentStatus(
      installed: json['installed'] as bool? ?? false,
      healthy: json['healthy'] as bool? ?? false,
      architecture: json['architecture'] as String? ?? 'DINOv3 ViT-B/16',
      componentVersion: (json['component_version'] as num?)?.toInt() ?? 1,
      displayName: json['display_name'] as String? ?? 'Neri1.1',
      classCount: (json['class_count'] as num?)?.toInt(),
      featureDim: (json['feature_dim'] as num?)?.toInt(),
      manifest: json['manifest'] as String?,
      message: json['message'] as String? ?? '',
    );
  }

  final bool installed;
  final bool healthy;
  final String architecture;
  final int componentVersion;
  final String displayName;
  final int? classCount;
  final int? featureDim;
  final String? manifest;
  final String message;
}
