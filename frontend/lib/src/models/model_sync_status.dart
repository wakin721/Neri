class ModelSyncStatus {
  const ModelSyncStatus({
    required this.state,
    this.runId,
    this.currentFile,
    this.totalFiles = 0,
    this.completedFiles = 0,
    this.receivedBytes = 0,
    this.totalBytes,
    this.lastSuccessfulSync,
    this.manifestId,
    this.error,
    this.cloudDetectCount,
    this.cloudClsCount,
  });

  factory ModelSyncStatus.fromJson(Map<String, dynamic> json) {
    return ModelSyncStatus(
      state: json['state'] as String? ?? 'idle',
      runId: json['run_id'] as String?,
      currentFile: json['current_file'] as String?,
      totalFiles: (json['total_files'] as num?)?.toInt() ?? 0,
      completedFiles: (json['completed_files'] as num?)?.toInt() ?? 0,
      receivedBytes: (json['received_bytes'] as num?)?.toInt() ?? 0,
      totalBytes: (json['total_bytes'] as num?)?.toInt(),
      lastSuccessfulSync: json['last_successful_sync'] as String?,
      manifestId: json['manifest_id'] as String?,
      error: json['error'] as String?,
      cloudDetectCount: (json['cloud_detect_count'] as num?)?.toInt(),
      cloudClsCount: (json['cloud_cls_count'] as num?)?.toInt(),
    );
  }

  final String state;
  final String? runId;
  final String? currentFile;
  final int totalFiles;
  final int completedFiles;
  final int receivedBytes;
  final int? totalBytes;
  final String? lastSuccessfulSync;
  final String? manifestId;
  final String? error;
  final int? cloudDetectCount;
  final int? cloudClsCount;

  bool get isActive => state == 'checking' || state == 'downloading';

  double? get progress {
    final total = totalBytes;
    if (total == null || total <= 0) {
      return null;
    }
    return (receivedBytes / total).clamp(0.0, 1.0).toDouble();
  }
}
