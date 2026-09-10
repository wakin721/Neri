import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/dinov3_explanation.dart';
import 'package:neri_flutter/src/widgets/dinov3_feature_scatter.dart';

DinoV3FeatureExplanation explanationFixture() {
  return DinoV3FeatureExplanation.fromJson({
    'species': 'Unknown',
    'accepted': false,
    'best_known_species': '赤麂',
    'known_score': 0.74,
    'threshold': 0.82,
    'nearest_prototype_index': 2,
    'squared_distance': 0.18,
    'nearest_species': [
      {
        'name': '赤麂',
        'nearest_prototype_index': 2,
        'squared_distance': 0.18,
        'cosine_score': 0.74,
        'source': 'checkpoint',
        'registry_id': null,
        'registration_status': null,
      },
      {
        'name': '小麂',
        'nearest_prototype_index': 5,
        'squared_distance': 0.31,
        'cosine_score': 0.66,
        'source': 'registry',
        'registry_id': 7,
        'registration_status': 'confirmed',
      },
    ],
    'projection': {
      'method': 'nearest_two_species_axis',
      'species': ['赤麂', '小麂'],
      'points': [
        {
          'kind': 'prototype',
          'species': '赤麂',
          'source': 'checkpoint',
          'registry_id': null,
          'registration_status': null,
          'prototype_index': 0,
          'x': -0.7,
          'y': 0.1,
        },
        {
          'kind': 'prototype',
          'species': '赤麂',
          'source': 'checkpoint',
          'registry_id': null,
          'registration_status': null,
          'prototype_index': 2,
          'x': -0.5,
          'y': -0.1,
        },
        {
          'kind': 'prototype',
          'species': '小麂',
          'source': 'registry',
          'registry_id': 7,
          'registration_status': 'confirmed',
          'prototype_index': 4,
          'x': 0.45,
          'y': 0.08,
        },
        {
          'kind': 'prototype',
          'species': '小麂',
          'source': 'registry',
          'registry_id': 7,
          'registration_status': 'confirmed',
          'prototype_index': 5,
          'x': 0.62,
          'y': -0.12,
        },
        {
          'kind': 'current',
          'species': 'Unknown',
          'source': 'checkpoint',
          'registry_id': null,
          'registration_status': null,
          'prototype_index': 2,
          'x': -0.08,
          'y': 0.34,
        },
      ],
    },
    'current_example_available': true,
    'nearest_example': {
      'kind': 'registry',
      'species': '赤麂',
      'registration_id': 7,
      'event_id': 12,
    },
  });
}

Future<Offset> _centroidForColor(ui.Image image, Color color) async {
  final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  expect(data, isNotNull);
  final bytes = data!.buffer.asUint8List();
  final argb = color.toARGB32();
  final red = (argb >> 16) & 0xff;
  final green = (argb >> 8) & 0xff;
  final blue = argb & 0xff;
  final alpha = (argb >> 24) & 0xff;
  var sumX = 0.0;
  var sumY = 0.0;
  var count = 0;

  for (var y = 0; y < image.height; y++) {
    for (var x = 0; x < image.width; x++) {
      final offset = (y * image.width + x) * 4;
      if (bytes[offset] == red &&
          bytes[offset + 1] == green &&
          bytes[offset + 2] == blue &&
          bytes[offset + 3] == alpha) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }

  expect(count, greaterThan(0));
  return Offset(sumX / count, sumY / count);
}

void main() {
  test('feature explanation parses nearest species and local projection', () {
    final explanation = explanationFixture();

    expect(explanation.species, 'Unknown');
    expect(explanation.accepted, isFalse);
    expect(explanation.nearestSpecies.map((item) => item.name), ['赤麂', '小麂']);
    expect(explanation.nearestSpecies.first.squaredDistance, 0.18);
    expect(explanation.projection.method, 'nearest_two_species_axis');
    expect(explanation.projection.species, ['赤麂', '小麂']);
    expect(
      explanation.projection.points.where((point) => point.isCurrent),
      hasLength(1),
    );
    expect(explanation.nearestExample?.kind, 'registry');
    expect(explanation.nearestExample?.eventId, 12);
  });

  testWidgets('feature scatter renders nearest classes and current sample', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 560,
            height: 360,
            child: DinoV3FeatureScatter(explanation: explanationFixture()),
          ),
        ),
      ),
    );

    expect(
      find.byKey(const ValueKey('dinov3-feature-scatter')),
      findsOneWidget,
    );
    expect(find.text('赤麂'), findsWidgets);
    expect(find.text('小麂'), findsWidgets);
    expect(find.text('当前检测框'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('feature scatter uses equal pixel scale for X and Y', (
    tester,
  ) async {
    final explanation = DinoV3FeatureExplanation(
      species: 'Unknown',
      accepted: false,
      bestKnownSpecies: 'A',
      knownScore: 0,
      threshold: 0,
      nearestSpecies: const [],
      projection: const DinoV3FeatureProjection(
        method: 'nearest_two_species_axis',
        species: ['A', 'B'],
        points: [
          DinoV3ProjectionPoint(
            kind: 'prototype',
            species: 'A',
            source: 'checkpoint',
            prototypeIndex: 0,
            x: 0,
            y: 0,
          ),
          DinoV3ProjectionPoint(
            kind: 'prototype',
            species: 'B',
            source: 'checkpoint',
            prototypeIndex: 0,
            x: 1,
            y: 0,
          ),
          DinoV3ProjectionPoint(
            kind: 'current',
            species: 'Unknown',
            source: 'checkpoint',
            prototypeIndex: 0,
            x: 0,
            y: 1,
          ),
        ],
      ),
      currentExampleAvailable: false,
    );
    final scheme = ColorScheme.fromSeed(seedColor: Colors.blue);

    await tester.pumpWidget(
      MaterialApp(
        theme: ThemeData(colorScheme: scheme),
        home: SizedBox(
          width: 560,
          height: 360,
          child: DinoV3FeatureScatter(explanation: explanation),
        ),
      ),
    );

    final paintWidget = tester.widget<CustomPaint>(find.byType(CustomPaint));
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    paintWidget.painter!.paint(canvas, const Size(400, 200));
    final image = await recorder.endRecording().toImage(400, 200);
    addTearDown(image.dispose);

    final origin = await _centroidForColor(image, scheme.primary);
    final xUnit = await _centroidForColor(image, scheme.tertiary);
    final yUnit = await _centroidForColor(image, scheme.error);
    final horizontalPixels = (xUnit - origin).distance;
    final verticalPixels = (yUnit - origin).distance;

    expect(horizontalPixels, closeTo(verticalPixels, 2.0));
  });

  test(
    'Registry and validation UI are wired to catalog and explanation APIs',
    () {
      final api = File('lib/src/api_client_core.dart').readAsStringSync();
      final registry = File(
        'lib/src/widgets/dinov3_registry_dialog.dart',
      ).readAsStringSync();
      final validation = File(
        'lib/src/screens/species_validation_screen.dart',
      ).readAsStringSync();

      expect(api, contains('fetchDinoV3RegistryCatalog'));
      expect(api, contains('/api/dinov3/registry/catalog'));
      expect(api, contains('fetchDinoV3FeatureExplanation'));
      expect(api, contains('/api/dinov3/feedback/observations/'));
      expect(registry, contains('fetchDinoV3RegistryCatalog'));
      expect(registry, contains('entry.isCheckpoint'));
      expect(registry, contains('分类头基础物种'));
      expect(validation, contains('DinoV3FeatureExplanationPanel('));
    },
  );
}
