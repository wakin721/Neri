class NeriSettings {
  const NeriSettings({
    required this.appTitle,
    required this.appVersion,
    required this.supportedImageExtensions,
    required this.supportedVideoExtensions,
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
    );
  }

  final String appTitle;
  final String appVersion;
  final List<String> supportedImageExtensions;
  final List<String> supportedVideoExtensions;
}
