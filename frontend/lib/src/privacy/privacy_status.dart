const privacyAgreementVersion = '2026-09-05';
const privacyAgreementAsset = 'assets/legal/user_agreement.md';

class PrivacyStatus {
  const PrivacyStatus({
    required this.agreementVersion,
    required this.agreementAccepted,
    required this.participationDecided,
    required this.trainingEnabled,
    required this.stats,
  });

  factory PrivacyStatus.fromJson(Map<String, dynamic> json) {
    final rawStats = json['stats'];
    return PrivacyStatus(
      agreementVersion: json['agreement_version'] as String? ?? '',
      agreementAccepted: json['agreement_accepted'] as bool? ?? false,
      participationDecided: json['participation_decided'] as bool? ?? false,
      trainingEnabled: json['training_enabled'] as bool? ?? false,
      stats: PrivacyQueueStats.fromJson(
        rawStats is Map<String, dynamic> ? rawStats : const <String, dynamic>{},
      ),
    );
  }

  final String agreementVersion;
  final bool agreementAccepted;
  final bool participationDecided;
  final bool trainingEnabled;
  final PrivacyQueueStats stats;

  bool get hasCurrentAgreement =>
      agreementAccepted && agreementVersion == privacyAgreementVersion;

  bool get isConsentComplete => hasCurrentAgreement && participationDecided;
}

class PrivacyQueueStats {
  const PrivacyQueueStats({
    required this.pending,
    required this.uploading,
    required this.uploaded,
    required this.failed,
    required this.skipped,
  });

  factory PrivacyQueueStats.fromJson(Map<String, dynamic> json) {
    return PrivacyQueueStats(
      pending: _integer(json['pending']),
      uploading: _integer(json['uploading']),
      uploaded: _integer(json['uploaded']),
      failed: _integer(json['failed']),
      skipped: _integer(json['skipped']),
    );
  }

  final int pending;
  final int uploading;
  final int uploaded;
  final int failed;
  final int skipped;

  int get waiting => pending + failed;

  static int _integer(Object? value) {
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '') ?? 0;
  }
}
