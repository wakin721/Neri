import 'package:http/http.dart' as http;

import 'api_client_core_impl.dart' as impl;
import 'dino_validation_selection.dart';
import 'models/job.dart';

export 'api_client_core_impl.dart' hide NeriApiClient;

class NeriApiClient extends impl.NeriApiClient {
  NeriApiClient({
    http.Client? httpClient,
    String baseUrl = 'http://127.0.0.1:721',
  }) : super(httpClient: httpClient, baseUrl: baseUrl);

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
    var updated = await super.markValidationItem(
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
    final shouldUseSelectedBox =
        action == 'update' &&
        confirmedSpecies.isNotEmpty &&
        modelPath.isNotEmpty &&
        operationId.isNotEmpty &&
        selection != null &&
        selection.learnableObservationCount > 1;
    if (!shouldUseSelectedBox) return updated;

    final result = await super.markDinoV3BoxFeedback(
      inputPath: inputPath,
      filePath: filePath,
      classificationModelPath: modelPath,
      observationId: selection.observationId,
      action: action,
      speciesName: confirmedSpecies,
      feedbackOperationId: operationId,
    );
    updated = updated.mergeValidationUpdate(result.item);
    return updated;
  }
}
