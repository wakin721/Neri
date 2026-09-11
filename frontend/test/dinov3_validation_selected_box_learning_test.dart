import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/dino_validation_selection.dart';
import 'package:neri_flutter/src/models/job.dart';

void main() {
  DetectionItem itemWithBoxes(List<DetectionBox> boxes) {
    return DetectionItem(
      filename: 'camera.jpg',
      path: 'C:/camera/camera.jpg',
      fileType: 'jpg',
      detectionBoxes: boxes,
    );
  }

  test('selected box is used when file-level DINO learning would be ambiguous', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.4, 0.4],
        observationId: 'obs-1',
      ),
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.5, 0.5, 0.9, 0.9],
        observationId: 'obs-2',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: 'obs-2',
        learningRequested: true,
      ),
      'obs-2',
    );
  });

  test('multi-box file without an explicit selected box remains skipped', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.4, 0.4],
        observationId: 'obs-1',
      ),
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.5, 0.5, 0.9, 0.9],
        observationId: 'obs-2',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: null,
        learningRequested: true,
      ),
      isNull,
    );
  });

  test('single-box files stay on the existing automatic feedback path', () {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.9, 0.9],
        observationId: 'obs-1',
      ),
    ]);

    expect(
      dinoValidationSelectedFeedbackObservationId(
        item,
        selectedObservationId: 'obs-1',
        learningRequested: true,
      ),
      isNull,
    );
  });

  test('file validation uses the selected box as DINO feedback fallback', () async {
    final item = itemWithBoxes(const <DetectionBox>[
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.1, 0.1, 0.4, 0.4],
        observationId: 'obs-1',
      ),
      DetectionBox(
        species: 'Unknown',
        bbox: <double>[0.5, 0.5, 0.9, 0.9],
        observationId: 'obs-2',
      ),
    ]);
    recordDinoValidationBoxSelection(item, item.detectionBoxes[1]);

    final requestedPaths = <String>[];
    Map<String, dynamic>? boxFeedbackBody;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requestedPaths.add(request.url.path);
        if (request.url.path == '/api/validation/mark') {
          return http.Response(
            jsonEncode(<String, dynamic>{
              'filename': 'camera.jpg',
              'path': item.path,
              'file_type': 'jpg',
              'species': <String>['骆驼'],
              'validated': true,
            }),
            200,
            headers: const {'content-type': 'application/json'},
          );
        }
        if (request.url.path == '/api/dinov3/feedback/box') {
          boxFeedbackBody = jsonDecode(request.body) as Map<String, dynamic>;
          return http.Response(
            jsonEncode(<String, dynamic>{
              'item': <String, dynamic>{
                'filename': 'camera.jpg',
                'path': item.path,
                'file_type': 'jpg',
                'species': <String>['骆驼'],
                'validated': true,
              },
              'operation_id': 'op-1',
              'affected_species': <String>['骆驼'],
            }),
            200,
            headers: const {'content-type': 'application/json'},
          );
        }
        return http.Response('{}', 404);
      }),
    );
    addTearDown(client.close);

    await client.markValidationItem(
      inputPath: 'C:/camera',
      filePath: item.path,
      action: 'update',
      speciesName: '骆驼',
      classificationModelPath: 'model.neri.json',
      feedbackOperationId: 'op-1',
    );

    expect(requestedPaths, <String>[
      '/api/validation/mark',
      '/api/dinov3/feedback/box',
    ]);
    expect(boxFeedbackBody?['observation_id'], 'obs-2');
    expect(boxFeedbackBody?['species_name'], '骆驼');
    expect(boxFeedbackBody?['feedback_operation_id'], 'op-1');
  });
}
