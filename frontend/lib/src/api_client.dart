import 'dart:convert';

import 'package:http/http.dart' as http;

import 'api_client_core.dart' as core;
import 'models/model_sync_status.dart';

export 'api_client_core.dart' hide NeriApiClient;

class NeriApiClient extends core.NeriApiClient {
  factory NeriApiClient({
    http.Client? httpClient,
    String baseUrl = 'http://127.0.0.1:721',
  }) {
    final client = httpClient ?? http.Client();
    return NeriApiClient._(client, baseUrl);
  }

  NeriApiClient._(this._modelSyncHttpClient, String baseUrl)
      : super(httpClient: _modelSyncHttpClient, baseUrl: baseUrl);

  final http.Client _modelSyncHttpClient;

  Future<ModelSyncStatus> fetchModelSyncStatus() async {
    final response = await _modelSyncHttpClient.get(
      Uri.parse('$baseUrl/api/model-sync/status'),
    );
    _ensureModelSyncSuccess(response);
    return ModelSyncStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ModelSyncStatus> runModelSync() async {
    final response = await _modelSyncHttpClient.post(
      Uri.parse('$baseUrl/api/model-sync/run'),
    );
    _ensureModelSyncSuccess(response);
    return ModelSyncStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<DinoV3ComponentStatus> fetchDinoV3ComponentStatus() async {
    final response = await _modelSyncHttpClient.get(
      Uri.parse('$baseUrl/api/environment/dinov3-status'),
    );
    _ensureModelSyncSuccess(response);
    return DinoV3ComponentStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<core.MaintenanceStartResponse> installDinoV3({
    required String envChoice,
    String packageSource = 'auto',
  }) async {
    final response = await _modelSyncHttpClient.post(
      Uri.parse('$baseUrl/api/environment/install-dinov3'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'env_choice': envChoice,
        'package_source': packageSource,
      }),
    );
    _ensureModelSyncSuccess(response);
    return core.MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<core.MaintenanceStartResponse> removeDinoV3() async {
    final response = await _modelSyncHttpClient.post(
      Uri.parse('$baseUrl/api/environment/remove-dinov3'),
    );
    _ensureModelSyncSuccess(response);
    return core.MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> mergeDinoV3RegistryCandidateIntoCheckpoint({
    required String classificationModelPath,
    required int registrationId,
    required String checkpointSpecies,
  }) async {
    final response = await _modelSyncHttpClient.post(
      Uri.parse(
        '$baseUrl/api/dinov3/registry/$registrationId/merge-checkpoint',
      ),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'checkpoint_species': checkpointSpecies,
      }),
    );
    _ensureModelSyncSuccess(response);
  }

  Future<void> markDinoV3RegistryCandidateEmpty({
    required String classificationModelPath,
    required int registrationId,
  }) async {
    final response = await _modelSyncHttpClient.post(
      Uri.parse('$baseUrl/api/dinov3/registry/$registrationId/empty'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({'classification_model_path': classificationModelPath}),
    );
    _ensureModelSyncSuccess(response);
  }

  void _ensureModelSyncSuccess(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return;
    }
    throw http.ClientException(
      'Neri API 请求失败 (${response.statusCode})',
      response.request?.url,
    );
  }
}

class DinoV3ComponentStatus {
  const DinoV3ComponentStatus({
    required this.installed,
    required this.healthy,
    required this.architecture,
    required this.componentVersion,
    required this.sourceCommit,
    required this.message,
    this.classifierFilename,
    this.classifierFingerprint,
    this.classifierHeadType,
    this.selectionK,
  });

  factory DinoV3ComponentStatus.fromJson(Map<String, dynamic> json) {
    return DinoV3ComponentStatus(
      installed: json['installed'] as bool? ?? false,
      healthy: json['healthy'] as bool? ?? false,
      architecture: json['architecture'] as String? ?? 'DINOv3 ViT-B/16',
      componentVersion: (json['component_version'] as num?)?.toInt() ?? 1,
      sourceCommit: json['source_commit'] as String? ?? '',
      classifierFilename: json['classifier_filename'] as String?,
      classifierFingerprint: json['classifier_fingerprint'] as String?,
      classifierHeadType: json['classifier_head_type'] as String?,
      selectionK: (json['selection_k'] as num?)?.toInt(),
      message: json['message'] as String? ?? '',
    );
  }

  final bool installed;
  final bool healthy;
  final String architecture;
  final int componentVersion;
  final String sourceCommit;
  final String? classifierFilename;
  final String? classifierFingerprint;
  final String? classifierHeadType;
  final int? selectionK;
  final String message;
}
