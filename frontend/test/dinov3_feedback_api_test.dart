import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';

void main() {
  test('DetectionBox decodes DINO feedback metadata', () {
    final box = DetectionBox.fromJson(const <String, dynamic>{
      'species': '盘羊',
      'bbox': <num>[1, 2, 3, 4],
      'observation_id': 'obs-1',
      'registry_id': 7,
      'predicted_species': '家牛',
      'feedback_status': 'corrected',
    });

    expect(box.observationId, 'obs-1');
    expect(box.registryId, 7);
    expect(box.predictedSpecies, '家牛');
    expect(box.feedbackStatus, 'corrected');
  });

  test('box feedback posts exact observation and operation ids', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/dinov3/feedback/box');
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode(<String, dynamic>{
            'item': <String, dynamic>{
              'filename': 'a.jpg',
              'path': 'C:/camera/a.jpg',
              'file_type': 'jpg',
            },
            'operation_id': 'op-1',
            'affected_species': <String>['盘羊'],
          }),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    final result = await client.markDinoV3BoxFeedback(
      inputPath: 'C:/camera',
      filePath: 'C:/camera/a.jpg',
      classificationModelPath: 'model.neri.json',
      observationId: 'obs-1',
      action: 'correct',
      feedbackOperationId: 'op-1',
    );

    expect(body, <String, dynamic>{
      'input_path': 'C:/camera',
      'file_path': 'C:/camera/a.jpg',
      'classification_model_path': 'model.neri.json',
      'observation_id': 'obs-1',
      'action': 'correct',
      'feedback_operation_id': 'op-1',
    });
    expect(result.item.filename, 'a.jpg');
    expect(result.operationId, 'op-1');
    expect(result.affectedSpecies, <String>['盘羊']);
  });

  test('box feedback includes explicit corrected species for update', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode(<String, dynamic>{
            'item': <String, dynamic>{
              'filename': 'a.jpg',
              'path': 'C:/camera/a.jpg',
              'file_type': 'jpg',
            },
            'operation_id': 'op-update',
            'affected_species': <String>['家牛', '盘羊'],
          }),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await client.markDinoV3BoxFeedback(
      inputPath: 'C:/camera',
      filePath: 'C:/camera/a.jpg',
      classificationModelPath: 'model.neri.json',
      observationId: 'obs-1',
      action: 'update',
      speciesName: '家牛',
      feedbackOperationId: 'op-update',
    );

    expect(body['species_name'], '家牛');
    expect(body['action'], 'update');
  });

  test('feedback revert posts exact operation id and decodes result', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/dinov3/feedback/revert');
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode(<String, dynamic>{
            'operation_id': 'op-1',
            'affected_species': <String>['盘羊'],
          }),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    final result = await client.revertDinoV3Feedback(
      classificationModelPath: 'model.neri.json',
      feedbackOperationId: 'op-1',
    );

    expect(body, <String, dynamic>{
      'classification_model_path': 'model.neri.json',
      'feedback_operation_id': 'op-1',
    });
    expect(result.operationId, 'op-1');
    expect(result.affectedSpecies, <String>['盘羊']);
  });

  test('single validation forwards DINO model and operation ids', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/validation/mark');
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode(<String, dynamic>{
            'filename': 'a.jpg',
            'path': 'C:/camera/a.jpg',
            'file_type': 'jpg',
          }),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await client.markValidationItem(
      inputPath: 'C:/camera',
      filePath: 'C:/camera/a.jpg',
      action: 'correct',
      classificationModelPath: 'model.neri.json',
      feedbackOperationId: 'op-file-1',
    );

    expect(body['classification_model_path'], 'model.neri.json');
    expect(body['feedback_operation_id'], 'op-file-1');
  });

  test('batch validation forwards one DINO operation id', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/validation/mark/batch');
        body = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode(<Map<String, dynamic>>[
            <String, dynamic>{
              'filename': 'a.jpg',
              'path': 'C:/camera/a.jpg',
              'file_type': 'jpg',
            },
            <String, dynamic>{
              'filename': 'b.jpg',
              'path': 'C:/camera/b.jpg',
              'file_type': 'jpg',
            },
          ]),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await client.markValidationItems(
      inputPath: 'C:/camera',
      filePaths: const <String>['C:/camera/a.jpg', 'C:/camera/b.jpg'],
      action: 'correct',
      classificationModelPath: 'model.neri.json',
      feedbackOperationId: 'op-batch-1',
    );

    expect(body['classification_model_path'], 'model.neri.json');
    expect(body['feedback_operation_id'], 'op-batch-1');
  });
}
