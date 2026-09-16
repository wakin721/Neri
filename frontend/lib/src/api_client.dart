import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'api_client_core.dart' as core;
import 'dino_validation_selection.dart';
import 'models/dinov2_explanation.dart';
import 'models/dinov2_feedback.dart';
import 'models/dinov2_registry.dart';
import 'models/job.dart';
import 'models/model_sync_status.dart';

export 'api_client_core.dart' hide NeriApiClient;
export 'models/dinov2_explanation.dart';
export 'models/dinov2_feedback.dart';
export 'models/dinov2_registry.dart';

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

  Future<DinoV2ComponentStatus> fetchDinoV2ComponentStatus() async {
    final response = await _http.get(
      _dinoUri('/api/environment/dinov2-status'),
    );
    _ensureSuccess(response);
    return DinoV2ComponentStatus.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<core.MaintenanceStartResponse> installDinoV2({
    required String envChoice,
    String packageSource = 'auto',
  }) async {
    final response = await _http.post(
      _dinoUri('/api/environment/install-dinov2'),
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

  Future<core.MaintenanceStartResponse> removeDinoV2() async {
    final response = await _http.post(
      _dinoUri('/api/environment/remove-dinov2'),
    );
    _ensureSuccess(response);
    return core.MaintenanceStartResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<DinoV2RegistryEntry>> fetchDinoV2Registry(
    String classificationModelPath, {
    String? status,
  }) async {
    final response = await _http.get(
      _dinoUri('/api/dinov2/registry', {
        'classification_model_path': classificationModelPath,
        if (status != null && status.isNotEmpty) 'status': status,
      }),
    );
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV2RegistryEntry.fromJson)
        .toList(growable: false);
  }

  Future<List<DinoV2RegistryEntry>> fetchDinoV2RegistryCatalog(
    String classificationModelPath,
  ) async {
    final response = await _http.get(
      _dinoUri('/api/dinov2/registry/catalog', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV2RegistryEntry.fromJson)
        .toList(growable: false);
  }

  Future<DinoV2RegistryEntry> fetchDinoV2RegistryEntry(
    String classificationModelPath,
    int registrationId,
  ) async {
    final response = await _http.get(
      _dinoUri('/api/dinov2/registry/$registrationId', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
    return DinoV2RegistryEntry.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<List<DinoV2RegistryEvent>> fetchDinoV2RegistryEvents(
    String classificationModelPath,
    int registrationId,
  ) async {
    final response = await _http.get(
      _dinoUri('/api/dinov2/registry/$registrationId/events', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV2RegistryEvent.fromJson)
        .toList(growable: false);
  }

  Future<List<DinoV2RegistryCluster>> fetchDinoV2RegistryClusters(
    String classificationModelPath,
    int registrationId,
  ) async {
    final response = await _http.get(
      _dinoUri('/api/dinov2/registry/$registrationId/clusters', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV2RegistryCluster.fromJson)
        .toList(growable: false);
  }

  Future<Uint8List> fetchDinoV2RegistryExample(
    String classificationModelPath,
    int registrationId,
    int eventId,
  ) async {
    final response = await _http.get(
      _dinoUri(
        '/api/dinov2/registry/$registrationId/events/$eventId/example',
        {'classification_model_path': classificationModelPath},
      ),
    );
    _ensureSuccess(response);
    return response.bodyBytes;
  }

  Future<void> deleteDinoV2RegistryEntry(
    String classificationModelPath,
    int registrationId,
  ) async {
    final response = await _http.delete(
      _dinoUri('/api/dinov2/registry/$registrationId', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
  }

  Future<int> clearDinoV2Candidates(String classificationModelPath) async {
    final response = await _http.delete(
      _dinoUri('/api/dinov2/registry/candidates', {
        'classification_model_path': classificationModelPath,
      }),
    );
    _ensureSuccess(response);
    final decoded = jsonDecode(response.body);
    return decoded is Map<String, dynamic>
        ? (decoded['deleted'] as num?)?.toInt() ?? 0
        : 0;
  }

  Future<DinoV2RegistryEntry> updateDinoV2RegistryIdentity({
    required String classificationModelPath,
    required int registrationId,
    required String commonName,
    String scientificName = '',
  }) async {
    final response = await _http.patch(
      _dinoUri('/api/dinov2/registry/$registrationId/identity'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'common_name': commonName,
        'scientific_name': scientificName,
      }),
    );
    _ensureSuccess(response);
    return DinoV2RegistryEntry.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<DinoV2RegistryEntry> registerDinoV2Species({
    required String classificationModelPath,
    required int registrationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/registry/$registrationId/register'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({'classification_model_path': classificationModelPath}),
    );
    _ensureSuccess(response);
    return DinoV2RegistryEntry.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> mergeDinoV2RegistryCandidateIntoCheckpoint({
    required String classificationModelPath,
    required int registrationId,
    required String checkpointSpecies,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/registry/$registrationId/merge-checkpoint'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'checkpoint_species': checkpointSpecies,
      }),
    );
    _ensureSuccess(response);
  }

  Future<DinoV2RegistryEntry> mergeDinoV2RegistryCandidate({
    required String classificationModelPath,
    required int registrationId,
    required int targetRegistrationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/registry/$registrationId/merge-candidate'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'target_registration_id': targetRegistrationId,
      }),
    );
    _ensureSuccess(response);
    return DinoV2RegistryEntry.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> markDinoV2RegistryCandidateEmpty({
    required String classificationModelPath,
    required int registrationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/registry/$registrationId/empty'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({'classification_model_path': classificationModelPath}),
    );
    _ensureSuccess(response);
  }

  Future<DinoV2BoxFeedbackResult> markDinoV2BoxFeedback({
    required String inputPath,
    required String filePath,
    required String classificationModelPath,
    required String observationId,
    required String action,
    String? speciesName,
    required String feedbackOperationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/feedback/box'),
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
    return DinoV2BoxFeedbackResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<DinoV2FeatureExplanation> fetchDinoV2FeatureExplanation(
    String classificationModelPath,
    String observationId,
  ) async {
    final response = await _http.get(
      _dinoUri(
        '/api/dinov2/feedback/observations/${Uri.encodeComponent(observationId)}/explain',
        {'classification_model_path': classificationModelPath},
      ),
    );
    _ensureSuccess(response);
    return DinoV2FeatureExplanation.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<Uint8List> fetchDinoV2ObservationExample(
    String classificationModelPath,
    String observationId,
  ) async {
    final response = await _http.get(
      _dinoUri(
        '/api/dinov2/feedback/observations/${Uri.encodeComponent(observationId)}/example',
        {'classification_model_path': classificationModelPath},
      ),
    );
    _ensureSuccess(response);
    return response.bodyBytes;
  }

  Future<DinoV2FeedbackRevertResult> revertDinoV2Feedback({
    required String classificationModelPath,
    required String feedbackOperationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/feedback/revert'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'classification_model_path': classificationModelPath,
        'feedback_operation_id': feedbackOperationId,
      }),
    );
    _ensureSuccess(response);
    return DinoV2FeedbackRevertResult.fromJson(
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
    );
    final selection = dinoValidationBoxSelectionFor(filePath);
    final modelPath = classificationModelPath?.trim() ?? '';
    final operationId = feedbackOperationId?.trim() ?? '';
    final confirmedSpecies = speciesName?.trim() ?? '';
    if (action != 'update' ||
        confirmedSpecies.isEmpty ||
        modelPath.isEmpty ||
        operationId.isEmpty ||
        selection == null ||
        selection.learnableObservationCount <= 1) {
      return updated;
    }
    await _recordDinoV2SelectedObservationFeedback(
      filePath: filePath,
      classificationModelPath: modelPath,
      observationId: selection.observationId,
      speciesName: confirmedSpecies,
      feedbackOperationId: operationId,
    );
    return updated;
  }

  Future<void> _recordDinoV2SelectedObservationFeedback({
    required String filePath,
    required String classificationModelPath,
    required String observationId,
    required String speciesName,
    required String feedbackOperationId,
  }) async {
    final response = await _http.post(
      _dinoUri('/api/dinov2/feedback/selection'),
      headers: const {'content-type': 'application/json'},
      body: jsonEncode({
        'file_path': filePath,
        'classification_model_path': classificationModelPath,
        'observation_id': observationId,
        'species_name': speciesName,
        'feedback_operation_id': feedbackOperationId,
      }),
    );
    _ensureSuccess(response);
  }

  void _ensureSuccess(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) return;
    throw http.ClientException(
      'Neri API 请求失败 (${response.statusCode})',
      response.request?.url,
    );
  }
}

class DinoV2ComponentStatus {
  const DinoV2ComponentStatus({
    required this.installed,
    required this.healthy,
    required this.architecture,
    required this.componentVersion,
    required this.message,
    this.classifierFilename,
    this.classifierFingerprint,
    this.classifierHeadType,
    this.prototypeCount,
  });

  factory DinoV2ComponentStatus.fromJson(Map<String, dynamic> json) {
    return DinoV2ComponentStatus(
      installed: json['installed'] as bool? ?? false,
      healthy: json['healthy'] as bool? ?? false,
      architecture: json['architecture'] as String? ?? 'DINOv2 ViT-B/14',
      componentVersion: (json['component_version'] as num?)?.toInt() ?? 1,
      classifierFilename: json['classifier_filename'] as String?,
      classifierFingerprint: json['classifier_fingerprint'] as String?,
      classifierHeadType: json['classifier_head_type'] as String?,
      prototypeCount: (json['prototype_count'] as num?)?.toInt(),
      message: json['message'] as String? ?? '',
    );
  }

  final bool installed;
  final bool healthy;
  final String architecture;
  final int componentVersion;
  final String? classifierFilename;
  final String? classifierFingerprint;
  final String? classifierHeadType;
  final int? prototypeCount;
  final String message;
}
