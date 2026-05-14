class NeriSettings {
  const NeriSettings({
    required this.appTitle,
    required this.appVersion,
    required this.supportedImageExtensions,
    required this.supportedVideoExtensions,
    required this.modelDirectory,
    required this.availableModels,
    this.selectedModel,
  });

  factory NeriSettings.fromJson(Map<String, dynamic> json) {
    return NeriSettings(
      appTitle: json['app_title'] as String? ?? 'Neri',
      appVersion: json['app_version'] as String? ?? '',
      supportedImageExtensions:
          (json['supported_image_extensions'] as List<dynamic>? ?? const <dynamic>[])
              .map((item) => item.toString())
              .toList(),
      supportedVideoExtensions:
          (json['supported_video_extensions'] as List<dynamic>? ?? const <dynamic>[])
              .map((item) => item.toString())
              .toList(),
      modelDirectory: json['model_directory'] as String? ?? 'res/model',
      availableModels:
          (json['available_models'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(ModelInfo.fromJson)
              .toList(),
      selectedModel: json['selected_model'] as String?,
    );
  }

  final String appTitle;
  final String appVersion;
  final List<String> supportedImageExtensions;
  final List<String> supportedVideoExtensions;
  final String modelDirectory;
  final List<ModelInfo> availableModels;
  final String? selectedModel;
}

class ModelInfo {
  const ModelInfo({
    required this.name,
    required this.path,
    this.sizeBytes,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      name: json['name'] as String? ?? '',
      path: json['path'] as String? ?? '',
      sizeBytes: json['size_bytes'] as int?,
    );
  }

  final String name;
  final String path;
  final int? sizeBytes;
}
