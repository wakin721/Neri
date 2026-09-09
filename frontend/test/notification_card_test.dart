import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/widgets/notification_card.dart';

void main() {
  testWidgets('short windows scroll cards without overflow', (tester) async {
    tester.view.physicalSize = const Size(420, 320);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      const MaterialApp(
        home: Stack(
          children: [
            NotificationCard(
              child: SizedBox(key: Key('upper'), width: 400, height: 280),
            ),
            NotificationCard(
              child: SizedBox(key: Key('lower'), width: 400, height: 180),
            ),
          ],
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    final upper = tester.getRect(find.byKey(const Key('upper')));
    final lower = tester.getRect(find.byKey(const Key('lower')));
    expect(upper.overlaps(lower), isFalse);
    await tester.drag(find.byType(SingleChildScrollView), const Offset(0, 300));
    await tester.pumpAndSettle();
    expect(
      tester.getRect(find.byKey(const Key('upper'))).top,
      greaterThan(upper.top),
    );
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pumpAndSettle();
  });
  testWidgets('cards stack without overlap and survive dismissal', (
    tester,
  ) async {
    var visible = true;
    late StateSetter update;
    await tester.pumpWidget(
      MaterialApp(
        home: StatefulBuilder(
          builder: (context, setState) {
            update = setState;
            return Stack(
              children: [
                if (visible)
                  const NotificationCard(
                    key: Key('registration'),
                    child: SizedBox(key: Key('first'), width: 400, height: 280),
                  ),
                const NotificationCard(
                  child: SizedBox(key: Key('second'), width: 400, height: 180),
                ),
              ],
            );
          },
        ),
      ),
    );
    await tester.pumpAndSettle();
    final first = tester.getRect(find.byKey(const Key('first')));
    final second = tester.getRect(find.byKey(const Key('second')));
    expect(first.overlaps(second), isFalse);
    expect(second.top - first.bottom, greaterThanOrEqualTo(12));
    update(() => visible = false);
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('first')), findsNothing);
    expect(
      tester.getRect(find.byKey(const Key('second'))).bottom,
      second.bottom,
    );
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });
}
