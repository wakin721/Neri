class DetectionItem {
  const DetectionItem({
    required this.filename,
    required this.path,
    required this.fileType,
    this.dateTaken,
    this.width,
    this.height,
    this.species = const <String>[],
    this.confidence,
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
      species: (json['species'] as List<dynamic>? ?? const <dynamic>[])
          .map((item) => item.toString())
          .toList(),
      confidence: (json['confidence'] as num?)?.toDouble(),
      error: json['error'] as String?,
    );
  }

  final String filename;
  final String path;
  final String fileType;
  final String? dateTaken;
  final int? width;
  final int? height;
  final List<String> species;
  final double? confidence;
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
