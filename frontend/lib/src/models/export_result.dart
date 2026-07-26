class ValidationExportResult {
  const ValidationExportResult({
    required this.outputPath,
    required this.fileFormat,
    required this.exportedCount,
    this.favoriteOutputDir,
    this.favoriteExportedCount = 0,
    this.deletedEmptyPhotoCount = 0,
    this.emptyPhotoDeleteFailedCount = 0,
  });

  factory ValidationExportResult.fromJson(Map<String, dynamic> json) {
    return ValidationExportResult(
      outputPath: json['output_path'] as String? ?? '',
      fileFormat: json['file_format'] as String? ?? '',
      exportedCount: json['exported_count'] as int? ?? 0,
      favoriteOutputDir: json['favorite_output_dir'] as String?,
      favoriteExportedCount: json['favorite_exported_count'] as int? ?? 0,
      deletedEmptyPhotoCount: json['deleted_empty_photo_count'] as int? ?? 0,
      emptyPhotoDeleteFailedCount:
          json['empty_photo_delete_failed_count'] as int? ?? 0,
    );
  }

  final String outputPath;
  final String fileFormat;
  final int exportedCount;
  final String? favoriteOutputDir;
  final int favoriteExportedCount;
  final int deletedEmptyPhotoCount;
  final int emptyPhotoDeleteFailedCount;
}
