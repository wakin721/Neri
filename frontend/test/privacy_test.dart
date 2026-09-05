import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/privacy/privacy_consent_dialog.dart';
import 'package:neri_flutter/src/privacy/privacy_gate_overlay.dart';
import 'package:neri_flutter/src/privacy/privacy_settings_card.dart';
import 'package:neri_flutter/src/privacy/privacy_status.dart';

void main() {
  const undecidedStatus = PrivacyStatus(
    agreementVersion: privacyAgreementVersion,
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

  Future<void> pumpConsent(
    WidgetTester tester, {
    required Future<PrivacyStatus> Function(bool enabled) onSave,
    ValueChanged<PrivacyStatus>? onSaved,
  }) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: PrivacyConsentDialog(
            agreementText: '# Neri 用户协议\n\n测试协议正文。',
            onSave: onSave,
            onSaved: onSaved ?? (_) {},
            onCloseApp: () {},
          ),
        ),
      ),
    );
  }

  Future<void> tapVisible(WidgetTester tester, Key key) async {
    final finder = find.byKey(key);
    await tester.ensureVisible(finder);
    await tester.pumpAndSettle();
    await tester.tap(finder);
  }

  testWidgets('participation starts with neither option selected', (
    tester,
  ) async {
    await pumpConsent(tester, onSave: (_) async => undecidedStatus);

    final group = tester.widget<RadioGroup<bool>>(
      find.byKey(privacyParticipationGroupKey),
    );
    final save = tester.widget<FilledButton>(find.byKey(privacySaveButtonKey));
    expect(group.groupValue, isNull);
    expect(save.onPressed, isNull);
  });

  testWidgets('agreement opens separately and returning preserves choices', (
    tester,
  ) async {
    var saves = 0;
    await pumpConsent(
      tester,
      onSave: (_) async {
        saves++;
        return undecidedStatus;
      },
    );

    expect(find.byType(PrivacyAgreementDocument), findsNothing);
    await tapVisible(tester, privacyAgreementCheckboxKey);
    await tapVisible(tester, privacyDeclineOptionKey);
    await tapVisible(tester, const Key('privacy-agreement-menu'));
    await tester.pumpAndSettle();

    expect(find.byType(PrivacyAgreementDocument), findsOneWidget);
    expect(find.text('# Neri 用户协议\n\n测试协议正文。'), findsOneWidget);
    await tester.tap(find.byTooltip('返回'));
    await tester.pumpAndSettle();

    expect(find.byType(PrivacyAgreementDocument), findsNothing);
    expect(find.byKey(privacyConsentDialogKey), findsOneWidget);
    expect(
      tester
          .widget<CheckboxListTile>(find.byKey(privacyAgreementCheckboxKey))
          .value,
      isTrue,
    );
    expect(
      tester
          .widget<RadioGroup<bool>>(find.byKey(privacyParticipationGroupKey))
          .groupValue,
      isFalse,
    );
    expect(saves, 0);
  });

  testWidgets('save requires agreement acceptance and a participation choice', (
    tester,
  ) async {
    await pumpConsent(tester, onSave: (_) async => undecidedStatus);

    await tapVisible(tester, privacyAgreementCheckboxKey);
    await tester.pump();
    expect(
      tester.widget<FilledButton>(find.byKey(privacySaveButtonKey)).onPressed,
      isNull,
    );

    await tapVisible(tester, privacyParticipateOptionKey);
    await tester.pump();
    expect(
      tester.widget<FilledButton>(find.byKey(privacySaveButtonKey)).onPressed,
      isNotNull,
    );
  });

  testWidgets('refusing participation still completes first-use consent', (
    tester,
  ) async {
    bool? submittedChoice;
    PrivacyStatus? completedStatus;
    const refusedStatus = PrivacyStatus(
      agreementVersion: privacyAgreementVersion,
      agreementAccepted: true,
      participationDecided: true,
      trainingEnabled: false,
      stats: PrivacyQueueStats(
        pending: 0,
        uploading: 0,
        uploaded: 0,
        failed: 0,
        skipped: 0,
      ),
    );
    await pumpConsent(
      tester,
      onSave: (enabled) async {
        submittedChoice = enabled;
        return refusedStatus;
      },
      onSaved: (status) => completedStatus = status,
    );

    await tapVisible(tester, privacyAgreementCheckboxKey);
    await tapVisible(tester, privacyDeclineOptionKey);
    await tester.pump();
    await tester.tap(find.byKey(privacySaveButtonKey));
    await tester.pumpAndSettle();

    expect(submittedChoice, isFalse);
    expect(completedStatus, refusedStatus);
  });

  testWidgets('privacy API failure keeps the consent dialog visible', (
    tester,
  ) async {
    var completed = false;
    await pumpConsent(
      tester,
      onSave: (_) async => throw Exception('network unavailable'),
      onSaved: (_) => completed = true,
    );

    await tapVisible(tester, privacyAgreementCheckboxKey);
    await tapVisible(tester, privacyParticipateOptionKey);
    await tester.pump();
    await tester.tap(find.byKey(privacySaveButtonKey));
    await tester.pumpAndSettle();

    expect(completed, isFalse);
    expect(find.byKey(privacyConsentDialogKey), findsOneWidget);
    expect(find.textContaining('network unavailable'), findsOneWidget);
  });

  testWidgets('long agreement remains usable in a small viewport', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(620, 440);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(
        home: PrivacyConsentDialog(
          agreementText: List.filled(80, '较长的用户协议内容。').join('\n'),
          onSave: (_) async => undecidedStatus,
          onSaved: (_) {},
          onCloseApp: () {},
        ),
      ),
    );

    expect(tester.takeException(), isNull);
    expect(find.byKey(privacySaveButtonKey), findsOneWidget);
    await tapVisible(tester, const Key('privacy-agreement-menu'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    final document = find.byType(PrivacyAgreementDocument);
    expect(document, findsOneWidget);
    await tester.drag(document, const Offset(0, -300));
    await tester.pumpAndSettle();
    final scrollable = tester.state<ScrollableState>(
      find.descendant(of: document, matching: find.byType(Scrollable)).first,
    );
    expect(scrollable.position.pixels, greaterThan(0));
    await tester.tap(find.byTooltip('返回'));
    await tester.pumpAndSettle();
    expect(find.byKey(privacyConsentDialogKey), findsOneWidget);
    expect(
      tester
          .widget<CheckboxListTile>(find.byKey(privacyAgreementCheckboxKey))
          .value,
      isFalse,
    );
    expect(
      tester.widget<FilledButton>(find.byKey(privacySaveButtonKey)).onPressed,
      isNull,
    );
  });

  testWidgets('agreement load failure cannot be accepted', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: PrivacyConsentDialog(
          agreementLoader: () async => throw Exception('asset missing'),
          onSave: (_) async => undecidedStatus,
          onSaved: (_) {},
          onCloseApp: () {},
        ),
      ),
    );
    await tester.pumpAndSettle();

    final checkbox = tester.widget<CheckboxListTile>(
      find.byKey(privacyAgreementCheckboxKey),
    );
    final save = tester.widget<FilledButton>(find.byKey(privacySaveButtonKey));
    expect(checkbox.enabled, isFalse);
    expect(save.onPressed, isNull);
    expect(find.textContaining('asset missing'), findsOneWidget);
  });

  testWidgets('privacy load error blocks workspace and offers retry and exit', (
    tester,
  ) async {
    var workspaceOpened = false;
    var retried = false;
    var closed = false;
    await tester.pumpWidget(
      MaterialApp(
        home: Stack(
          children: [
            Center(
              child: TextButton(
                key: const Key('workspace-action'),
                onPressed: () => workspaceOpened = true,
                child: const Text('工作区'),
              ),
            ),
            PrivacyGateOverlay(
              status: null,
              loading: false,
              loadError: 'local API unavailable',
              onRetry: () => retried = true,
              onSave: (_) async => undecidedStatus,
              onSaved: (_) {},
              onCloseApp: () => closed = true,
            ),
          ],
        ),
      ),
    );

    await tester.tap(
      find.byKey(const Key('workspace-action')),
      warnIfMissed: false,
    );
    await tester.tap(find.byKey(privacyRetryButtonKey));
    await tester.tap(find.byKey(privacyLoadErrorCloseButtonKey));

    expect(workspaceOpened, isFalse);
    expect(retried, isTrue);
    expect(closed, isTrue);
    expect(find.textContaining('local API unavailable'), findsOneWidget);
  });

  testWidgets('settings enable action reuses explicit unselected disclosure', (
    tester,
  ) async {
    final client = NeriApiClient(
      httpClient: MockClient((_) async {
        return http.Response(
          jsonEncode({
            'agreement_version': privacyAgreementVersion,
            'agreement_accepted': true,
            'participation_decided': true,
            'training_enabled': false,
            'stats': {
              'pending': 2,
              'uploading': 0,
              'uploaded': 4,
              'failed': 1,
              'skipped': 3,
            },
          }),
          200,
        );
      }),
    );
    addTearDown(client.close);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: PrivacySettingsCard(apiClient: client),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('模型改进计划未启用'), findsOneWidget);
    expect(find.textContaining('待处理 2'), findsOneWidget);
    await tester.tap(find.byKey(privacyEnableButtonKey));
    await tester.pumpAndSettle();

    final group = tester.widget<RadioGroup<bool>>(
      find.byKey(privacyParticipationGroupKey),
    );
    expect(group.groupValue, isNull);
    expect(find.byKey(privacyAgreementCheckboxKey), findsOneWidget);
    await tapVisible(tester, privacyAgreementMenuKey);
    await tester.pumpAndSettle();
    expect(find.byType(PrivacyAgreementDocument), findsOneWidget);
    await tester.tap(find.byTooltip('返回'));
    await tester.pumpAndSettle();
    expect(find.byKey(privacyConsentDialogKey), findsOneWidget);
    await tester.tap(find.byKey(privacyConsentCancelButtonKey));
    await tester.pumpAndSettle();
    expect(find.byKey(privacyConsentDialogKey), findsNothing);
  });

  testWidgets('settings privacy load error recovers automatically', (
    tester,
  ) async {
    var requests = 0;
    final client = NeriApiClient(
      httpClient: MockClient((_) async {
        requests++;
        if (requests == 1) return http.Response('offline', 503);
        return http.Response(
          jsonEncode({
            'agreement_version': privacyAgreementVersion,
            'agreement_accepted': true,
            'participation_decided': true,
            'training_enabled': false,
            'stats': {
              'pending': 0,
              'uploading': 0,
              'uploaded': 0,
              'failed': 0,
              'skipped': 0,
            },
          }),
          200,
        );
      }),
    );
    addTearDown(client.close);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: PrivacySettingsCard(apiClient: client)),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.textContaining('503'), findsOneWidget);
    await tester.pump(const Duration(seconds: 2));
    await tester.pumpAndSettle();
    expect(find.text('模型改进计划未启用'), findsOneWidget);
    expect(requests, 2);
  });

  testWidgets(
    'settings polls without overlap and ignores a stale consent response',
    (tester) async {
      var reads = 0;
      var enabled = true;
      final stale = Completer<http.Response>();
      http.Response response(bool value, int uploaded) => http.Response(
        jsonEncode({
          'agreement_version': privacyAgreementVersion,
          'agreement_accepted': true,
          'participation_decided': true,
          'training_enabled': value,
          'stats': {
            'pending': 0,
            'uploading': 0,
            'uploaded': uploaded,
            'failed': 0,
            'skipped': 0,
          },
        }),
        200,
      );
      final client = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.method == 'PUT') {
            enabled = false;
            return response(false, 8);
          }
          reads++;
          if (reads == 2) return stale.future;
          return response(enabled, reads == 1 ? 4 : 9);
        }),
      );
      addTearDown(client.close);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: PrivacySettingsCard(apiClient: client)),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.byTooltip('刷新'), findsNothing);
      expect(find.text('已上传 4'), findsOneWidget);
      await tester.pump(const Duration(seconds: 2));
      await tester.pump(const Duration(seconds: 4));
      expect(reads, 2);
      expect(find.byType(CircularProgressIndicator), findsNothing);
      await tester.tap(find.byKey(privacyDisableButtonKey));
      await tester.pumpAndSettle();
      stale.complete(response(true, 4));
      await tester.pumpAndSettle();
      expect(find.text('模型改进计划未启用'), findsOneWidget);
      expect(find.text('已上传 8'), findsOneWidget);
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();
      expect(find.text('已上传 9'), findsOneWidget);
      await tester.pumpWidget(const SizedBox());
      final readsAfterDispose = reads;
      await tester.pump(const Duration(seconds: 6));
      expect(reads, readsAfterDispose);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets('settings shows agreement status and disables immediately', (
    tester,
  ) async {
    final methods = <String>[];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        methods.add(request.method);
        final disabling = request.method == 'PUT';
        if (disabling) {
          expect(jsonDecode(request.body), {
            'agreement_version': privacyAgreementVersion,
            'training_enabled': false,
          });
        }
        return http.Response(
          jsonEncode({
            'agreement_version': privacyAgreementVersion,
            'agreement_accepted': true,
            'participation_decided': true,
            'training_enabled': !disabling,
            'stats': {
              'pending': 0,
              'uploading': 0,
              'uploaded': 4,
              'failed': 0,
              'skipped': 0,
            },
          }),
          200,
        );
      }),
    );
    addTearDown(client.close);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: PrivacySettingsCard(apiClient: client)),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('协议版本 2026-09-05 · 已同意'), findsOneWidget);
    await tester.tap(find.byKey(privacyDisableButtonKey));
    await tester.pumpAndSettle();
    expect(methods, ['GET', 'PUT']);
    expect(find.text('模型改进计划未启用'), findsOneWidget);
  });

  testWidgets(
    'hidden settings pause polling and refresh immediately on return',
    (tester) async {
      var reads = 0;
      final client = NeriApiClient(
        httpClient: MockClient((_) async {
          reads++;
          return http.Response(
            jsonEncode({
              'agreement_version': privacyAgreementVersion,
              'agreement_accepted': true,
              'participation_decided': true,
              'training_enabled': false,
              'stats': {
                'pending': 0,
                'uploading': 0,
                'uploaded': reads,
                'failed': 0,
                'skipped': 0,
              },
            }),
            200,
          );
        }),
      );
      addTearDown(client.close);
      Future<void> show(bool active) async {
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: PrivacySettingsCard(apiClient: client, isActive: active),
            ),
          ),
        );
        await tester.pumpAndSettle();
      }

      await show(true);
      expect(find.text('已上传 1'), findsOneWidget);
      await show(false);
      await tester.pump(const Duration(seconds: 6));
      expect(reads, 1);
      await show(true);
      expect(find.text('已上传 2'), findsOneWidget);
      await tester.pumpWidget(const SizedBox());
    },
  );

  test(
    'privacy API uses the dedicated status and mutation endpoints',
    () async {
      final requests = <http.Request>[];
      final client = NeriApiClient(
        baseUrl: 'http://127.0.0.1:721',
        httpClient: MockClient((request) async {
          requests.add(request);
          return http.Response(
            jsonEncode({
              'agreement_version': privacyAgreementVersion,
              'agreement_accepted': true,
              'participation_decided': true,
              'training_enabled': request.method == 'PUT',
              'stats': {
                'pending': 2,
                'uploading': 1,
                'uploaded': 4,
                'failed': 3,
                'skipped': 5,
              },
            }),
            200,
            headers: const {'content-type': 'application/json'},
          );
        }),
      );

      final fetched = await client.fetchPrivacyStatus();
      final enabled = await client.savePrivacyStatus(trainingEnabled: true);
      final cleared = await client.clearPrivacyQueue();

      expect(fetched.stats.pending, 2);
      expect(enabled.trainingEnabled, isTrue);
      expect(cleared.stats.skipped, 5);
      expect(requests.map((request) => request.method), [
        'GET',
        'PUT',
        'DELETE',
      ]);
      expect(requests.map((request) => request.url.path), [
        '/api/privacy',
        '/api/privacy',
        '/api/privacy/queue',
      ]);
      expect(jsonDecode(requests[1].body), {
        'agreement_version': privacyAgreementVersion,
        'training_enabled': true,
      });
      client.close();
    },
  );
}
