import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/privacy/privacy_gate_overlay.dart';
import 'package:neri_flutter/src/privacy/privacy_status.dart';

void main() {
  testWidgets('privacy loading overlay uses generic startup copy', (tester) async {
    const fallbackStatus = PrivacyStatus(
      agreementVersion: 'test',
      agreementAccepted: false,
      participationDecided: false,
      trainingEnabled: false,
      stats: PrivacyQueueStats(
        pending: 0,
        uploading: 0,
        uploaded: 0,
        failed: 0,
        skipped: 0,
      ),
    );

    await tester.pumpWidget(
      MaterialApp(
        home: PrivacyGateOverlay(
          status: null,
          loading: true,
          onRetry: () {},
          onSave: (_) async => fallbackStatus,
          onSaved: (_) {},
          onCloseApp: () {},
        ),
      ),
    );

    expect(find.text('程序启动中'), findsOneWidget);
    expect(find.textContaining('读取隐私设置'), findsNothing);
  });
}
