class ValidationExportResult {
  const ValidationExportResult({
    required this.outputPath,
    required this.fileFormat,
    required this.exportedCount,
  });

  factory ValidationExportResult.fromJson(Map<String, dynamic> json) {
    return ValidationExportResult(
      outputPath: json['output_path'] as String? ?? '',
      fileFormat: json['file_format'] as String? ?? '',
      exportedCount: json['exported_count'] as int? ?? 0,
    );
  }

  final String outputPath;
  final String fileFormat;
  final int exportedCount;
}
