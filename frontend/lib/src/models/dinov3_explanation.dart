class DinoV3NearestSpecies {
  const DinoV3NearestSpecies({
    required this.name,
    required this.nearestPrototypeIndex,
    required this.squaredDistance,
    required this.cosineScore,
    required this.source,
    this.registryId,
    this.registrationStatus,
  });

  factory DinoV3NearestSpecies.fromJson(Map<String, dynamic> json) {
    return DinoV3NearestSpecies(
      name: json['name']?.toString() ?? '',
      nearestPrototypeIndex:
          (json['nearest_prototype_index'] as num?)?.toInt() ?? -1,
      squaredDistance: (json['squared_distance'] as num?)?.toDouble() ?? 0,
      cosineScore: (json['cosine_score'] as num?)?.toDouble() ?? 0,
      source: json['source']?.toString() ?? '',
      registryId: (json['registry_id'] as num?)?.toInt(),
      registrationStatus: json['registration_status']?.toString(),
    );
  }

  final String name;
  final int nearestPrototypeIndex;
  final double squaredDistance;
  final double cosineScore;
  final String source;
  final int? registryId;
  final String? registrationStatus;
}

class DinoV3ProjectionPoint {
  const DinoV3ProjectionPoint({
    required this.kind,
    required this.species,
    required this.source,
    required this.prototypeIndex,
    required this.x,
    required this.y,
    this.registryId,
    this.registrationStatus,
  });

  factory DinoV3ProjectionPoint.fromJson(Map<String, dynamic> json) {
    return DinoV3ProjectionPoint(
      kind: json['kind']?.toString() ?? 'prototype',
      species: json['species']?.toString() ?? '',
      source: json['source']?.toString() ?? '',
      prototypeIndex: (json['prototype_index'] as num?)?.toInt(),
      x: (json['x'] as num?)?.toDouble() ?? 0,
      y: (json['y'] as num?)?.toDouble() ?? 0,
      registryId: (json['registry_id'] as num?)?.toInt(),
      registrationStatus: json['registration_status']?.toString(),
    );
  }

  final String kind;
  final String species;
  final String source;
  final int? prototypeIndex;
  final double x;
  final double y;
  final int? registryId;
  final String? registrationStatus;

  bool get isCurrent => kind == 'current';
}

class DinoV3FeatureProjection {
  const DinoV3FeatureProjection({
    required this.method,
    required this.species,
    required this.points,
  });

  factory DinoV3FeatureProjection.fromJson(Map<String, dynamic> json) {
    return DinoV3FeatureProjection(
      method: json['method']?.toString() ?? '',
      species: (json['species'] as List<dynamic>? ?? const <dynamic>[])
          .map((value) => value.toString())
          .toList(growable: false),
      points: (json['points'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(DinoV3ProjectionPoint.fromJson)
          .toList(growable: false),
    );
  }

  final String method;
  final List<String> species;
  final List<DinoV3ProjectionPoint> points;
}

class DinoV3NearestExample {
  const DinoV3NearestExample({
    required this.kind,
    required this.species,
    this.registrationId,
    this.eventId,
    this.observationId,
  });

  factory DinoV3NearestExample.fromJson(Map<String, dynamic> json) {
    return DinoV3NearestExample(
      kind: json['kind']?.toString() ?? '',
      species: json['species']?.toString() ?? '',
      registrationId: (json['registration_id'] as num?)?.toInt(),
      eventId: (json['event_id'] as num?)?.toInt(),
      observationId: json['observation_id']?.toString(),
    );
  }

  final String kind;
  final String species;
  final int? registrationId;
  final int? eventId;
  final String? observationId;
}

class DinoV3FeatureExplanation {
  const DinoV3FeatureExplanation({
    required this.species,
    required this.accepted,
    required this.bestKnownSpecies,
    required this.knownScore,
    required this.threshold,
    required this.nearestSpecies,
    required this.projection,
    required this.currentExampleAvailable,
    this.nearestPrototypeIndex,
    this.squaredDistance,
    this.nearestExample,
  });

  factory DinoV3FeatureExplanation.fromJson(Map<String, dynamic> json) {
    final projection = json['projection'];
    final nearestExample = json['nearest_example'];
    return DinoV3FeatureExplanation(
      species: json['species']?.toString() ?? 'Unknown',
      accepted: json['accepted'] == true,
      bestKnownSpecies: json['best_known_species']?.toString() ?? '',
      knownScore: (json['known_score'] as num?)?.toDouble() ?? 0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0,
      nearestPrototypeIndex: (json['nearest_prototype_index'] as num?)?.toInt(),
      squaredDistance: (json['squared_distance'] as num?)?.toDouble(),
      nearestSpecies:
          (json['nearest_species'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(DinoV3NearestSpecies.fromJson)
              .toList(growable: false),
      projection: DinoV3FeatureProjection.fromJson(
        projection is Map<String, dynamic>
            ? projection
            : const <String, dynamic>{},
      ),
      currentExampleAvailable: json['current_example_available'] == true,
      nearestExample: nearestExample is Map<String, dynamic>
          ? DinoV3NearestExample.fromJson(nearestExample)
          : null,
    );
  }

  final String species;
  final bool accepted;
  final String bestKnownSpecies;
  final double knownScore;
  final double threshold;
  final int? nearestPrototypeIndex;
  final double? squaredDistance;
  final List<DinoV3NearestSpecies> nearestSpecies;
  final DinoV3FeatureProjection projection;
  final bool currentExampleAvailable;
  final DinoV3NearestExample? nearestExample;
}
