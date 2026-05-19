class NeriSettings {
  const NeriSettings({
    required this.appTitle,
    required this.appVersion,
    required this.supportedImageExtensions,
    required this.supportedVideoExtensions,
    required this.modelDirectory,
    required this.classificationModelDirectory,
    required this.availableModels,
    required this.availableClassificationModels,
    required this.speciesTypes,
    required this.settings,
    required this.gpuAvailable,
    required this.missingYoloDependencies,
    this.selectedModel,
    this.selectedClassificationModel,
  });

  factory NeriSettings.fromJson(Map<String, dynamic> json) {
    return NeriSettings(
      appTitle: json['app_title'] as String? ?? 'Neri',
      appVersion: json['app_version'] as String? ?? '',
      supportedImageExtensions:
          (json['supported_image_extensions'] as List<dynamic>? ??
                  const <dynamic>[])
              .map((item) => item.toString())
              .toList(),
      supportedVideoExtensions:
          (json['supported_video_extensions'] as List<dynamic>? ??
                  const <dynamic>[])
              .map((item) => item.toString())
              .toList(),
      modelDirectory: json['model_directory'] as String? ?? 'res/model',
      classificationModelDirectory:
          json['classification_model_directory'] as String? ?? 'res/model_cls',
      availableModels:
          (json['available_models'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(ModelInfo.fromJson)
              .toList(),
      availableClassificationModels:
          (json['available_classification_models'] as List<dynamic>? ??
                  const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(ModelInfo.fromJson)
              .toList(),
      selectedModel: json['selected_model'] as String?,
      selectedClassificationModel:
          json['selected_classification_model'] as String?,
      speciesTypes: (json['species_types'] as Map<String, dynamic>? ?? const {})
          .map((key, value) => MapEntry(key, value.toString())),
      settings: Map<String, dynamic>.from(
        json['settings'] as Map<String, dynamic>? ?? const {},
      ),
      gpuAvailable: json['gpu_available'] as bool? ?? false,
      missingYoloDependencies:
          (json['missing_yolo_dependencies'] as List<dynamic>? ??
                  const <dynamic>[])
              .map((item) => item.toString())
              .toList(),
    );
  }

  final String appTitle;
  final String appVersion;
  final List<String> supportedImageExtensions;
  final List<String> supportedVideoExtensions;
  final String modelDirectory;
  final String classificationModelDirectory;
  final List<ModelInfo> availableModels;
  final List<ModelInfo> availableClassificationModels;
  final String? selectedModel;
  final String? selectedClassificationModel;
  final Map<String, String> speciesTypes;
  final Map<String, dynamic> settings;
  final bool gpuAvailable;
  final List<String> missingYoloDependencies;

  NeriSettings copyWith({
    String? appTitle,
    String? appVersion,
    List<String>? supportedImageExtensions,
    List<String>? supportedVideoExtensions,
    String? modelDirectory,
    String? classificationModelDirectory,
    List<ModelInfo>? availableModels,
    List<ModelInfo>? availableClassificationModels,
    String? selectedModel,
    String? selectedClassificationModel,
    Map<String, String>? speciesTypes,
    Map<String, dynamic>? settings,
    bool? gpuAvailable,
    List<String>? missingYoloDependencies,
  }) {
    return NeriSettings(
      appTitle: appTitle ?? this.appTitle,
      appVersion: appVersion ?? this.appVersion,
      supportedImageExtensions:
          supportedImageExtensions ?? this.supportedImageExtensions,
      supportedVideoExtensions:
          supportedVideoExtensions ?? this.supportedVideoExtensions,
      modelDirectory: modelDirectory ?? this.modelDirectory,
      classificationModelDirectory:
          classificationModelDirectory ?? this.classificationModelDirectory,
      availableModels: availableModels ?? this.availableModels,
      availableClassificationModels:
          availableClassificationModels ?? this.availableClassificationModels,
      selectedModel: selectedModel ?? this.selectedModel,
      selectedClassificationModel:
          selectedClassificationModel ?? this.selectedClassificationModel,
      speciesTypes: speciesTypes ?? this.speciesTypes,
      settings: settings ?? this.settings,
      gpuAvailable: gpuAvailable ?? this.gpuAvailable,
      missingYoloDependencies:
          missingYoloDependencies ?? this.missingYoloDependencies,
    );
  }
}

class ModelInfo {
  const ModelInfo({required this.name, required this.path, this.sizeBytes});

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

class ModelClassInfo {
  const ModelClassInfo({
    required this.id,
    required this.name,
    required this.displayName,
  });

  factory ModelClassInfo.fromJson(Map<String, dynamic> json) {
    return ModelClassInfo(
      id: json['id'] as int? ?? 0,
      name: json['name'] as String? ?? '',
      displayName: json['display_name'] as String? ?? '',
    );
  }

  final int id;
  final String name;
  final String displayName;

  String get label => displayName.isEmpty ? name : displayName;
}
