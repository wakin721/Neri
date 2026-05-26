class DetectionBox {
  const DetectionBox({
    required this.species,
    required this.bbox,
    this.confidence,
    this.frameIndex,
    this.timestamp,
    this.trackId,
    this.candidates = const <Map<String, dynamic>>[],
  });

  factory DetectionBox.fromJson(Map<String, dynamic> json) {
    final rawBbox = json['bbox'] as List<dynamic>? ?? const <dynamic>[];
    return DetectionBox(
      species: json['species'] as String? ?? 'Unknown',
      confidence: (json['confidence'] as num?)?.toDouble(),
      frameIndex: _intFromJson(json['frame_index']),
      timestamp: _doubleFromJson(json['timestamp']),
      trackId: _stringFromJson(json['track_id'] ?? json['trackId']),
      bbox: rawBbox
          .map(
            (item) => item is num
                ? item.toDouble()
                : double.tryParse(item.toString()),
          )
          .whereType<double>()
          .toList(),
      candidates: (json['candidates'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .toList(),
    );
  }

  final String species;
  final double? confidence;
  final int? frameIndex;
  final double? timestamp;
  final String? trackId;
  final List<double> bbox;
  final List<Map<String, dynamic>> candidates;

  static int? _intFromJson(Object? value) {
    if (value is int) return value;
    if (value is num) return value.round();
    return int.tryParse(value?.toString() ?? '');
  }

  static double? _doubleFromJson(Object? value) {
    if (value is num) return value.toDouble();
    return double.tryParse(value?.toString() ?? '');
  }

  static String? _stringFromJson(Object? value) {
    final text = value?.toString().trim();
    return text == null || text.isEmpty ? null : text;
  }
}

class DetectionItem {
  const DetectionItem({
    required this.filename,
    required this.path,
    required this.fileType,
    this.dateTaken,
    this.modifiedAt,
    this.width,
    this.height,
    this.sizeBytes,
    this.species = const <String>[],
    this.confidence,
    this.detectionBoxes = const <DetectionBox>[],
    this.detectionData = const <String, dynamic>{},
    this.error,
    this.validated,
  });

  factory DetectionItem.fromJson(Map<String, dynamic> json) {
    return DetectionItem(
      filename: json['filename'] as String? ?? '',
      path: json['path'] as String? ?? '',
      fileType: json['file_type'] as String? ?? '',
      dateTaken: json['date_taken'] as String?,
      modifiedAt: json['modified_at'] as String?,
      width: json['width'] as int?,
      height: json['height'] as int?,
      sizeBytes: json['size_bytes'] as int?,
      species: (json['species'] as List<dynamic>? ?? const <dynamic>[])
          .map((item) => item.toString())
          .toList(),
      confidence: (json['confidence'] as num?)?.toDouble(),
      detectionBoxes:
          (json['detection_boxes'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(DetectionBox.fromJson)
              .toList(),
      detectionData:
          json['detection_data'] as Map<String, dynamic>? ??
          const <String, dynamic>{},
      error: json['error'] as String?,
      validated: json['validated'] as bool?,
    );
  }

  final String filename;
  final String path;
  final String fileType;
  final String? dateTaken;
  final String? modifiedAt;
  final int? width;
  final int? height;
  final int? sizeBytes;
  final List<String> species;
  final double? confidence;
  final List<DetectionBox> detectionBoxes;
  final Map<String, dynamic> detectionData;
  final String? error;
  final bool? validated;

  DetectionItem mergeValidationUpdate(DetectionItem update) {
    return DetectionItem(
      filename: update.filename.isNotEmpty ? update.filename : filename,
      path: update.path.isNotEmpty ? update.path : path,
      fileType: update.fileType.isNotEmpty ? update.fileType : fileType,
      dateTaken: update.dateTaken ?? dateTaken,
      modifiedAt: update.modifiedAt ?? modifiedAt,
      width: update.width ?? width,
      height: update.height ?? height,
      sizeBytes: update.sizeBytes ?? sizeBytes,
      species: update.species,
      confidence: update.confidence,
      detectionBoxes: update.detectionBoxes,
      detectionData: update.detectionData,
      error: update.error,
      validated: update.validated,
    );
  }
}

class ProcessingJob {
  const ProcessingJob({
    required this.id,
    required this.state,
    required this.inputDir,
    required this.createdAt,
    required this.updatedAt,
    this.outputDir,
    this.total = 0,
    this.processed = 0,
    this.elapsedSeconds = 0,
    this.speed = 0,
    this.remainingSeconds,
    this.message = '',
    this.results = const <DetectionItem>[],
    this.error,
    this.active = false,
  });

  factory ProcessingJob.fromJson(Map<String, dynamic> json) {
    return ProcessingJob(
      id: json['id'] as String? ?? '',
      state: json['state'] as String? ?? 'queued',
      inputDir: json['input_dir'] as String? ?? '',
      outputDir: json['output_dir'] as String?,
      total: json['total'] as int? ?? 0,
      processed: json['processed'] as int? ?? 0,
      elapsedSeconds: _doubleFromJobJson(json['elapsed_seconds']) ?? 0,
      speed: _doubleFromJobJson(json['speed']) ?? 0,
      remainingSeconds: _doubleFromJobJson(json['remaining_seconds']),
      message: json['message'] as String? ?? '',
      results: (json['results'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(DetectionItem.fromJson)
          .toList(),
      error: json['error'] as String?,
      active: json['active'] as bool? ?? false,
      createdAt: json['created_at'] as String? ?? '',
      updatedAt: json['updated_at'] as String? ?? '',
    );
  }

  final String id;
  final String state;
  final String inputDir;
  final String? outputDir;
  final int total;
  final int processed;
  final double elapsedSeconds;
  final double speed;
  final double? remainingSeconds;
  final String message;
  final List<DetectionItem> results;
  final String? error;
  final bool active;
  final String createdAt;
  final String updatedAt;

  double get progress => total == 0 ? 0 : processed / total;
  bool get isActive => state == 'queued' || state == 'running';
  bool get isWorkerActive => isActive || active;
  bool get canResume => !active && (state == 'cancelled' || state == 'failed');
}

double? _doubleFromJobJson(Object? value) {
  if (value is num) return value.toDouble();
  return double.tryParse(value?.toString() ?? '');
}
