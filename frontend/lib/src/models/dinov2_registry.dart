class DinoV2ClusterExampleRef {
  const DinoV2ClusterExampleRef({
    required this.kind,
    this.registrationId,
    this.eventId,
    this.observationId,
  });

  factory DinoV2ClusterExampleRef.fromJson(Map<String, dynamic> json) {
    return DinoV2ClusterExampleRef(
      kind: json['kind']?.toString() ?? '',
      registrationId: (json['registration_id'] as num?)?.toInt(),
      eventId: (json['event_id'] as num?)?.toInt(),
      observationId: json['observation_id']?.toString(),
    );
  }

  final String kind;
  final int? registrationId;
  final int? eventId;
  final String? observationId;
}

class DinoV2RegistryCluster {
  const DinoV2RegistryCluster({
    required this.id,
    required this.label,
    required this.source,
    required this.prototypeIndex,
    required this.eventCount,
    required this.cameraCount,
    required this.sampleCount,
    required this.active,
    required this.exampleRefs,
    this.meanSquaredDistance,
    this.learningStatus,
  });

  factory DinoV2RegistryCluster.fromJson(Map<String, dynamic> json) {
    final rawRefs = json['example_refs'] as List<dynamic>? ?? const <dynamic>[];
    return DinoV2RegistryCluster(
      id: json['id']?.toString() ?? '',
      label: json['label']?.toString() ?? 'Cluster',
      source: json['source']?.toString() ?? '',
      prototypeIndex: (json['prototype_index'] as num?)?.toInt() ?? 0,
      eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
      cameraCount: (json['camera_count'] as num?)?.toInt() ?? 0,
      sampleCount: (json['sample_count'] as num?)?.toInt() ?? 0,
      meanSquaredDistance: (json['mean_squared_distance'] as num?)?.toDouble(),
      active: json['active'] != false,
      learningStatus: json['learning_status']?.toString(),
      exampleRefs: rawRefs
          .whereType<Map<String, dynamic>>()
          .map(DinoV2ClusterExampleRef.fromJson)
          .toList(growable: false),
    );
  }

  final String id;
  final String label;
  final String source;
  final int prototypeIndex;
  final int eventCount;
  final int cameraCount;
  final int sampleCount;
  final double? meanSquaredDistance;
  final bool active;
  final String? learningStatus;
  final List<DinoV2ClusterExampleRef> exampleRefs;

  bool get isCheckpoint => source == 'checkpoint';
  bool get isFeedback => source == 'feedback' || source == 'feedback_evidence';
  bool get isRegistry => source == 'registry';
}

class DinoV2RegistryEntry {
  const DinoV2RegistryEntry({
    required this.id,
    required this.candidateNumber,
    required this.status,
    this.candidateKind = 'candidate',
    required this.displayName,
    required this.commonName,
    required this.scientificName,
    required this.eventCount,
    required this.cameraCount,
    required this.prototypeCount,
    required this.clusterPurity,
    required this.embeddingConsistency,
    required this.conditions,
    required this.canRegister,
    this.feedbackEventCount = 0,
    this.feedbackPrototypeCount = 0,
    this.learningStatus,
    this.clusters = const <DinoV2RegistryCluster>[],
  });

  factory DinoV2RegistryEntry.fromJson(Map<String, dynamic> json) {
    final rawConditions =
        json['conditions'] as Map<String, dynamic>? ?? const {};
    final rawClusters = json['clusters'] as List<dynamic>? ?? const <dynamic>[];
    return DinoV2RegistryEntry(
      id: (json['id'] as num?)?.toInt() ?? 0,
      candidateNumber: (json['candidate_number'] as num?)?.toInt() ?? 0,
      status: json['status']?.toString() ?? 'candidate',
      candidateKind: json['candidate_kind']?.toString() ?? 'candidate',
      displayName: json['display_name']?.toString() ?? '',
      commonName: json['common_name']?.toString() ?? '',
      scientificName: json['scientific_name']?.toString() ?? '',
      eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
      cameraCount: (json['camera_count'] as num?)?.toInt() ?? 0,
      prototypeCount: (json['prototype_count'] as num?)?.toInt() ?? 0,
      clusterPurity: (json['cluster_purity'] as num?)?.toDouble() ?? 0,
      embeddingConsistency:
          (json['embedding_consistency'] as num?)?.toDouble() ?? 0,
      conditions: rawConditions.map(
        (key, value) => MapEntry(key, value == true),
      ),
      canRegister: json['can_register'] == true,
      feedbackEventCount: (json['feedback_event_count'] as num?)?.toInt() ?? 0,
      feedbackPrototypeCount:
          (json['feedback_prototype_count'] as num?)?.toInt() ?? 0,
      learningStatus: json['learning_status']?.toString(),
      clusters: rawClusters
          .whereType<Map<String, dynamic>>()
          .map(DinoV2RegistryCluster.fromJson)
          .toList(growable: false),
    );
  }

  final int id;
  final int candidateNumber;
  final String status;
  final String candidateKind;
  final String displayName;
  final String commonName;
  final String scientificName;
  final int eventCount;
  final int cameraCount;
  final int prototypeCount;
  final double clusterPurity;
  final double embeddingConsistency;
  final Map<String, bool> conditions;
  final bool canRegister;
  final int feedbackEventCount;
  final int feedbackPrototypeCount;
  final String? learningStatus;
  final List<DinoV2RegistryCluster> clusters;

  DinoV2RegistryEntry withClusters(List<DinoV2RegistryCluster> value) {
    return DinoV2RegistryEntry(
      id: id,
      candidateNumber: candidateNumber,
      status: status,
      candidateKind: candidateKind,
      displayName: displayName,
      commonName: commonName,
      scientificName: scientificName,
      eventCount: eventCount,
      cameraCount: cameraCount,
      prototypeCount: prototypeCount,
      clusterPurity: clusterPurity,
      embeddingConsistency: embeddingConsistency,
      conditions: conditions,
      canRegister: canRegister,
      feedbackEventCount: feedbackEventCount,
      feedbackPrototypeCount: feedbackPrototypeCount,
      learningStatus: learningStatus,
      clusters: List.unmodifiable(value),
    );
  }

  bool get isCandidate => status == 'candidate';
  bool get isCheckpoint => status.toLowerCase() == 'checkpoint';
  bool get hasFeedbackLearning => feedbackEventCount > 0;

  bool get canDelete => const <String>{
    'candidate',
    'provisional',
    'confirmed',
    'mature',
  }.contains(status.toLowerCase());
}

class DinoV2RegistryEvent {
  const DinoV2RegistryEvent({
    required this.eventKey,
    required this.sourcePath,
    required this.cameraId,
    required this.sampleCount,
    required this.timestampMissing,
    this.id = 0,
    this.bbox = const <double>[],
    this.frameIndex,
    this.timestampSeconds,
    this.hasExample = false,
    this.startedAt,
    this.endedAt,
  });

  factory DinoV2RegistryEvent.fromJson(Map<String, dynamic> json) {
    return DinoV2RegistryEvent(
      id: (json['id'] as num?)?.toInt() ?? 0,
      eventKey: json['event_key']?.toString() ?? '',
      sourcePath: json['source_path']?.toString() ?? '',
      cameraId: json['camera_id']?.toString() ?? '',
      sampleCount: (json['sample_count'] as num?)?.toInt() ?? 1,
      timestampMissing: json['timestamp_missing'] == true,
      bbox: (json['bbox'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<num>()
          .map((value) => value.toDouble())
          .toList(),
      frameIndex: (json['frame_index'] as num?)?.toInt(),
      timestampSeconds: (json['timestamp_seconds'] as num?)?.toDouble(),
      hasExample: json['has_example'] == true,
      startedAt: json['started_at']?.toString(),
      endedAt: json['ended_at']?.toString(),
    );
  }

  final int id;
  final String eventKey;
  final String sourcePath;
  final String cameraId;
  final int sampleCount;
  final bool timestampMissing;
  final List<double> bbox;
  final int? frameIndex;
  final double? timestampSeconds;
  final bool hasExample;
  final String? startedAt;
  final String? endedAt;
}
