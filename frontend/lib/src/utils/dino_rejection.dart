import '../models/job.dart';

const _scoreKeys = [
  'box_known_score',
  'sequence_known_score',
  'known_score',
  'within_score',
  'ridge_score',
  'raw_ridge_score',
];

const _gateScoreKeys = ['box_known_score', 'sequence_known_score'];

const _legacyOrRidgeKeys = [
  'known_score',
  'within_score',
  'ridge_score',
  'raw_ridge_score',
];

class DinoBoxRejectionInfo {
  const DinoBoxRejectionInfo({
    required this.finalBox,
    required this.finalLabel,
    required this.finalRejected,
    required this.isHumanConfirmed,
    required this.isImageLabeled,
    required this.isLegacySingleGate,
    required this.hasInconsistentRecord,
    this.isExperimentSingleGate = false,
    this.classificationConfidence,
    this.boxScore,
    this.boxThreshold,
    this.boxPassed,
    this.sequenceScore,
    this.sequenceThreshold,
    this.sequencePassed,
    this.legacyScore,
    this.legacyThreshold,
    this.legacyPassed,
    this.modelAccepted,
    this.withinScore,
    this.ridgeScore,
    this.rawRidgeScore,
    this.rawDetConf,
  });

  final DetectionBox finalBox;
  final String finalLabel;
  final bool finalRejected;
  final bool isHumanConfirmed;
  final bool isImageLabeled;
  final bool isLegacySingleGate;
  final bool isExperimentSingleGate;
  final double? classificationConfidence;
  final bool hasInconsistentRecord;
  final double? boxScore;
  final double? boxThreshold;
  final bool? boxPassed;
  final double? sequenceScore;
  final double? sequenceThreshold;
  final bool? sequencePassed;
  final double? legacyScore;
  final double? legacyThreshold;
  final bool? legacyPassed;
  final bool? modelAccepted;
  final double? withinScore;
  final double? ridgeScore;
  final double? rawRidgeScore;
  final double? rawDetConf;

  bool get isDetectorCategory =>
      primaryDinoCandidate(finalBox)?['source'] == 'detector_category';
}

Map<String, dynamic>? primaryDinoCandidate(DetectionBox box) {
  if (box.candidates.isEmpty) return null;
  for (final candidate in box.candidates) {
    if (candidate['source'] == 'dinov3_ridge_within') {
      return candidate;
    }
  }
  for (final candidate in box.candidates) {
    if (_gateScoreKeys.any((key) => candidate[key] != null)) {
      return candidate;
    }
  }
  for (final candidate in box.candidates) {
    if (_legacyOrRidgeKeys.any((key) => candidate[key] != null)) {
      return candidate;
    }
  }
  return box.candidates.first;
}

String dinoConfidenceLabel(DetectionBox box) {
  final candidate = primaryDinoCandidate(box);
  if (candidate?['source'] == 'detector_category') return '探测证据';
  if (candidate?['confidence_kind'] == 'sigmoid_frame_class_margin') {
    return '分类证据';
  }
  if (candidate?['gate_mode'] == 'sequence' ||
      candidate?['inference_recipe'] == 'experiment_full') {
    return '事件接纳';
  }
  return '置信度';
}

Map<dynamic, dynamic>? findRawDinoBox(
  DetectionBox box,
  int index,
  dynamic rawBoxes,
) {
  if (rawBoxes is! List) return null;
  final obsId = box.observationId?.trim() ?? '';
  if (obsId.isNotEmpty) {
    for (final entry in rawBoxes) {
      if (entry is! Map) continue;
      final rawObsId = entry['observation_id']?.toString().trim() ?? '';
      if (rawObsId == obsId) return entry;
    }
  }

  if (box.bbox.length >= 4) {
    for (final entry in rawBoxes) {
      if (entry is! Map) continue;
      if (box.frameIndex != null) {
        final rawFrame = entry['frame_index'] ?? entry['frameIndex'];
        final parsedFrame = rawFrame is int
            ? rawFrame
            : (rawFrame is num ? rawFrame.round() : int.tryParse('$rawFrame'));
        if (parsedFrame != null && parsedFrame != box.frameIndex) continue;
      }
      final rawCoords =
          entry['边界框'] ?? entry['bbox'] ?? entry['box'] ?? entry['xyxy'];
      if (rawCoords is! List || rawCoords.length < 4) continue;
      var matches = true;
      for (var i = 0; i < 4; i++) {
        final rawVal = rawCoords[i];
        final parsed = rawVal is num
            ? rawVal.toDouble()
            : double.tryParse(rawVal?.toString() ?? '');
        if (parsed == null || (parsed - box.bbox[i]).abs() > 1e-3) {
          matches = false;
          break;
        }
      }
      if (matches) return entry;
    }
    return null;
  }

  if (index >= 0 && index < rawBoxes.length && rawBoxes[index] is Map) {
    return rawBoxes[index] as Map;
  }
  return null;
}

bool isDinoDecision(DetectionBox box) =>
    box.observationId?.startsWith('dinov3:') == true ||
    box.candidates.any(
      (c) =>
          c['source'] == 'dinov3_ridge_within' ||
          _scoreKeys.any((key) => c[key] != null),
    );

bool isDinoItem(DetectionItem item) =>
    item.detectionBoxes.any(isDinoDecision) ||
    item.detectionData['sequence_completed'] == true ||
    item.detectionData['sequence_status'] != null;

double? _finiteMetric(Map<dynamic, dynamic>? source, String key) {
  if (source == null) return null;
  final raw = source[key];
  final value = raw is num
      ? raw.toDouble()
      : double.tryParse(raw?.toString() ?? '');
  return value != null && value.isFinite ? value : null;
}

double? _firstFiniteMetric(
  List<Map<dynamic, dynamic>?> sources,
  List<String> keys,
) {
  for (final key in keys) {
    for (final source in sources) {
      final value = _finiteMetric(source, key);
      if (value != null) return value;
    }
  }
  return null;
}

bool? _boolFlag(List<Map<dynamic, dynamic>?> sources, String key) {
  for (final source in sources) {
    final value = source?[key];
    if (value is bool) return value;
  }
  return null;
}

double? dinoMetric(DetectionBox box, String key) =>
    _finiteMetric(primaryDinoCandidate(box), key);

/// The model has already applied its calibrated gates; display never gates again.
DetectionBox dinoFinalBox(DetectionBox box) {
  if (box.feedbackStatus == 'confirmed' ||
      box.feedbackStatus == 'image_label' ||
      box.candidates.isEmpty) {
    return box;
  }
  final primary = primaryDinoCandidate(box);
  if (primary == null) return box;
  final rejected =
      primary['accepted'] == false ||
      primary['box_accepted'] == false ||
      primary['sequence_accepted'] == false;
  final species = rejected ? 'Unknown' : box.species;
  if (species == box.species) return box;
  return DetectionBox(
    species: species,
    bbox: box.bbox,
    confidence: box.confidence,
    frameIndex: box.frameIndex,
    timestamp: box.timestamp,
    trackId: box.trackId,
    observationId: box.observationId,
    registryId: box.registryId,
    predictedSpecies: box.predictedSpecies,
    feedbackStatus: box.feedbackStatus,
    rejectionCluster: box.rejectionCluster,
    candidates: box.candidates,
  );
}

/// Extracts raw DINO gate metrics without ever substituting sigmoid confidences.
DinoBoxRejectionInfo dinoRejectionInfo(
  DetectionBox box, {
  Map<dynamic, dynamic>? rawBox,
}) {
  final primary = primaryDinoCandidate(box);
  final sources = <Map<dynamic, dynamic>?>[primary, rawBox];
  final isExperimentSingleGate = sources.any(
    (source) =>
        source?['gate_mode'] == 'sequence' ||
        source?['inference_recipe'] == 'experiment_full',
  );

  final boxScore = _firstFiniteMetric(sources, const ['box_known_score']);
  final sequenceScore = _firstFiniteMetric(sources, const [
    'sequence_known_score',
  ]);
  final isLegacySingleGate = boxScore == null && sequenceScore == null;

  final legacyScore = isLegacySingleGate
      ? _firstFiniteMetric(sources, const ['known_score'])
      : null;
  final legacyThreshold = isLegacySingleGate
      ? _firstFiniteMetric(sources, const ['threshold', 'score_threshold'])
      : null;

  final boxThreshold = _firstFiniteMetric(sources, const ['box_threshold']);
  final sequenceThreshold = sequenceScore != null
      ? _firstFiniteMetric(sources, const [
          'sequence_threshold',
          'threshold',
          'score_threshold',
        ])
      : _firstFiniteMetric(sources, const ['sequence_threshold']);

  final explicitAccepted = _boolFlag(sources, 'accepted');
  final explicitBoxAccepted = _boolFlag(sources, 'box_accepted');
  final explicitSeqAccepted = _boolFlag(sources, 'sequence_accepted');

  final boxCompared = (boxScore != null && boxThreshold != null)
      ? boxScore >= boxThreshold
      : null;
  final sequenceCompared = (sequenceScore != null && sequenceThreshold != null)
      ? sequenceScore >= sequenceThreshold
      : null;
  final legacyCompared =
      (isLegacySingleGate && legacyScore != null && legacyThreshold != null)
      ? legacyScore >= legacyThreshold
      : null;

  final boxPassed = explicitBoxAccepted ?? boxCompared;
  final sequencePassed = explicitSeqAccepted ?? sequenceCompared;
  final legacyPassed = isLegacySingleGate
      ? (explicitAccepted ?? legacyCompared)
      : null;

  var hasInconsistentRecord = false;
  if (explicitBoxAccepted != null &&
      boxCompared != null &&
      explicitBoxAccepted != boxCompared) {
    hasInconsistentRecord = true;
  }
  if (explicitSeqAccepted != null &&
      sequenceCompared != null &&
      explicitSeqAccepted != sequenceCompared) {
    hasInconsistentRecord = true;
  }
  if (isLegacySingleGate &&
      explicitAccepted != null &&
      legacyCompared != null &&
      explicitAccepted != legacyCompared) {
    hasInconsistentRecord = true;
  }
  if (!isLegacySingleGate && explicitAccepted != null) {
    final comparedCombined = (boxCompared != null && sequenceCompared != null)
        ? (boxCompared && sequenceCompared)
        : null;
    final flagCombined =
        (explicitBoxAccepted != null && explicitSeqAccepted != null)
        ? (explicitBoxAccepted && explicitSeqAccepted)
        : null;
    if ((comparedCombined != null && explicitAccepted != comparedCombined) ||
        (flagCombined != null && explicitAccepted != flagCombined)) {
      hasInconsistentRecord = true;
    }
  }

  bool? modelAccepted;
  if (explicitAccepted == false ||
      boxPassed == false ||
      sequencePassed == false ||
      legacyPassed == false) {
    modelAccepted = false;
  } else if (explicitAccepted == true ||
      (boxPassed == true && sequencePassed == true) ||
      legacyPassed == true) {
    modelAccepted = true;
  }

  final finalBox = dinoFinalBox(box);
  final displaySpecies = finalBox.species.trim();
  final isHumanConfirmed = box.feedbackStatus == 'confirmed';
  final isImageLabeled = box.feedbackStatus == 'image_label';
  final finalRejected =
      !isHumanConfirmed &&
      !isImageLabeled &&
      (modelAccepted == false ||
          displaySpecies == 'Unknown' ||
          displaySpecies == '拒识/Unknown');

  final String finalLabel;
  if (isHumanConfirmed) {
    finalLabel = displaySpecies == '空'
        ? '人工标为空'
        : '人工已确认 · ${displaySpecies.isEmpty ? 'Unknown' : displaySpecies}';
  } else if (isImageLabeled) {
    finalLabel =
        '整图标注 · ${displaySpecies.isEmpty ? 'Unknown' : displaySpecies}';
  } else if (finalRejected) {
    finalLabel = '最终拒识 · Unknown';
  } else if (primary?['source'] == 'detector_category') {
    finalLabel = '探测保留 · $displaySpecies';
  } else if (modelAccepted == true) {
    finalLabel =
        '最终接纳 · ${displaySpecies.isEmpty ? 'Unknown' : displaySpecies}';
  } else {
    finalLabel = displaySpecies.isEmpty ? '未判定' : displaySpecies;
  }

  return DinoBoxRejectionInfo(
    finalBox: finalBox,
    finalLabel: finalLabel,
    finalRejected: finalRejected,
    isHumanConfirmed: isHumanConfirmed,
    isImageLabeled: isImageLabeled,
    isLegacySingleGate: isLegacySingleGate,
    isExperimentSingleGate: isExperimentSingleGate,
    classificationConfidence: _firstFiniteMetric(sources, const [
      'frame_classification_confidence',
    ]),
    hasInconsistentRecord: hasInconsistentRecord,
    boxScore: boxScore,
    boxThreshold: boxThreshold,
    boxPassed: boxPassed,
    sequenceScore: sequenceScore,
    sequenceThreshold: sequenceThreshold,
    sequencePassed: sequencePassed,
    legacyScore: legacyScore,
    legacyThreshold: legacyThreshold,
    legacyPassed: legacyPassed,
    modelAccepted: modelAccepted,
    withinScore: _firstFiniteMetric(sources, const ['within_score']),
    ridgeScore: _firstFiniteMetric(sources, const ['ridge_score']),
    rawRidgeScore: _firstFiniteMetric(sources, const ['raw_ridge_score']),
    rawDetConf: _firstFiniteMetric(sources, const ['raw_det_conf']),
  );
}
