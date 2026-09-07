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

  void _ensureModelSyncSuccess(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return;
    }
    throw http.ClientException(
      '模型同步 API 请求失败 (${response.statusCode})',
      response.request?.url,
    );
  }
}
