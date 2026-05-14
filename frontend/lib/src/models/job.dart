class DetectionBox {
  const DetectionBox({
    required this.species,
    required this.bbox,
    this.confidence,
    this.candidates = const <Map<String, dynamic>>[],
  });

  factory DetectionBox.fromJson(Map<String, dynamic> json) {
    final rawBbox = json['bbox'] as List<dynamic>? ?? const <dynamic>[];
    return DetectionBox(
      species: json['species'] as String? ?? 'Unknown',
      confidence: (json['confidence'] as num?)?.toDouble(),
      bbox: rawBbox
          .map((item) => item is num ? item.toDouble() : double.tryParse(item.toString()))
          .whereType<double>()
          .toList(),
      candidates: (json['candidates'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .toList(),
    );
  }

  final String species;
  final double? confidence;
  final List<double> bbox;
  final List<Map<String, dynamic>> candidates;
}

class DetectionItem {
  const DetectionItem({
    required this.filename,
    required this.path,
    required this.fileType,
    this.dateTaken,
    this.width,
    this.height,
    this.sizeBytes,
    this.species = const <String>[],
    this.confidence,
    this.detectionBoxes = const <DetectionBox>[],
    this.detectionData = const <String, dynamic>{},
    this.error,
  });

  factory DetectionItem.fromJson(Map<String, dynamic> json) {
    return DetectionItem(
      filename: json['filename'] as String? ?? '',
      path: json['path'] as String? ?? '',
      fileType: json['file_type'] as String? ?? '',
      dateTaken: json['date_taken'] as String?,
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
      detectionData: json['detection_data'] as Map<String, dynamic>? ?? const <String, dynamic>{},
      error: json['error'] as String?,
    );
  }

  final String filename;
  final String path;
  final String fileType;
  final String? dateTaken;
  final int? width;
  final int? height;
  final int? sizeBytes;
  final List<String> species;
  final double? confidence;
  final List<DetectionBox> detectionBoxes;
  final Map<String, dynamic> detectionData;
  final String? error;
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
    this.message = '',
    this.results = const <DetectionItem>[],
    this.error,
  });

  factory ProcessingJob.fromJson(Map<String, dynamic> json) {
    return ProcessingJob(
      id: json['id'] as String? ?? '',
      state: json['state'] as String? ?? 'queued',
      inputDir: json['input_dir'] as String? ?? '',
      outputDir: json['output_dir'] as String?,
      total: json['total'] as int? ?? 0,
      processed: json['processed'] as int? ?? 0,
      message: json['message'] as String? ?? '',
      results: (json['results'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(DetectionItem.fromJson)
          .toList(),
      error: json['error'] as String?,
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
  final String message;
  final List<DetectionItem> results;
  final String? error;
  final String createdAt;
  final String updatedAt;

  double get progress => total == 0 ? 0 : processed / total;
  bool get isActive => state == 'queued' || state == 'running';
}
