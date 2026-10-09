import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/widgets/expressive_action_button.dart';
import 'package:neri_flutter/src/widgets/selectable_list_card.dart';

Widget _actionHost(VoidCallback? onPressed, {bool disableAnimations = false}) {
  return MaterialApp(
    theme: buildNeriTheme(ColorScheme.fromSeed(seedColor: Colors.teal)),
    home: MediaQuery(
      data: MediaQueryData(disableAnimations: disableAnimations),
      child: Scaffold(
        body: Center(
          child: ExpressiveActionButton(
            onPressed: onPressed,
            icon: const Icon(Icons.play_arrow_rounded),
            label: const Text('Run'),
          ),
        ),
      ),
    ),
  );
}

double _actionScale(WidgetTester tester) => tester
    .widget<ScaleTransition>(
      find.descendant(
        of: find.byType(ExpressiveActionButton),
        matching: find.byType(ScaleTransition),
      ),
    )
    .scale
    .value;

void main() {
  testWidgets('custom light and dark schemes survive in the rendered app', (
    tester,
  ) async {
    for (final brightness in Brightness.values) {
      final scheme = ColorScheme.fromSeed(
        seedColor: Colors.deepOrange,
        brightness: brightness,
      ).copyWith(primary: const Color(0xff345678));
      late ThemeData inheritedTheme;
      await tester.pumpWidget(
        MaterialApp(
          theme: buildNeriTheme(scheme),
          home: Builder(
            builder: (context) {
              inheritedTheme = Theme.of(context);
              return const Scaffold(body: Text('Content'));
            },
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(inheritedTheme.colorScheme, scheme);
      expect(inheritedTheme.brightness, brightness);
      expect(inheritedTheme.scaffoldBackgroundColor, scheme.surface);
      expect(tester.takeException(), isNull);
    }
  });

  testWidgets('primary action activates once and disabled action stays inert', (
    tester,
  ) async {
    var activations = 0;
    await tester.pumpWidget(_actionHost(() => activations++));
    await tester.tap(find.text('Run'));
    await tester.pumpAndSettle();
    expect(activations, 1);

    await tester.pumpWidget(_actionHost(null));
    await tester.tap(find.text('Run'));
    await tester.pumpAndSettle();
    expect(activations, 1);
    expect(_actionScale(tester), closeTo(1, 0.001));
  });

  testWidgets('holding an action contracts it and releasing restores it', (
    tester,
  ) async {
    var activations = 0;
    await tester.pumpWidget(_actionHost(() => activations++));
    final gesture = await tester.startGesture(tester.getCenter(find.text('Run')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 250));
    expect(_actionScale(tester), lessThan(0.99));
    expect(activations, 0);
    await gesture.up();
    await tester.pumpAndSettle();
    expect(_actionScale(tester), closeTo(1, 0.001));
    expect(activations, 1);
  });

  testWidgets('reduced motion keeps the action still while preserving taps', (
    tester,
  ) async {
    var activations = 0;
    await tester.pumpWidget(
      _actionHost(() => activations++, disableAnimations: true),
    );
    final gesture = await tester.startGesture(tester.getCenter(find.text('Run')));
    await tester.pump(const Duration(milliseconds: 250));
    expect(_actionScale(tester), 1);
    await gesture.up();
    await tester.pumpAndSettle();
    expect(_actionScale(tester), 1);
    expect(activations, 1);
  });

  testWidgets('list selection and lazy secondary-click menus still work', (
    tester,
  ) async {
    int? selected;
    int? menuItem;
    var menuActivations = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SelectableListCard<int>(
            items: const [0, 1, 2],
            selectedIndex: 0,
            titleBuilder: (item) => 'Item $item',
            onSelected: (index, item) => selected = item,
            onMenuOpening: (index, item) => menuItem = item,
            menuChildrenBuilder: (context, index, item) => [
              MenuItemButton(
                onPressed: () => menuActivations++,
                child: Text('Inspect $item'),
              ),
            ],
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Inspect 1'), findsNothing);
    await tester.tap(find.text('Item 1'));
    expect(selected, 1);
    await tester.tap(
      find.text('Item 1'),
      kind: PointerDeviceKind.mouse,
      buttons: kSecondaryMouseButton,
    );
    await tester.pumpAndSettle();
    expect(menuItem, 1);
    expect(find.text('Inspect 1'), findsOneWidget);
    await tester.tap(find.text('Inspect 1'));
    await tester.pumpAndSettle();
    expect(menuActivations, 1);
    expect(find.text('Inspect 1'), findsNothing);
  });

  testWidgets('jumping selection scrolls a cached offscreen row into view', (
    tester,
  ) async {
    var selectedIndex = 0;
    late StateSetter updateSelection;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Center(
            child: SizedBox(
              width: 320,
              height: 240,
              child: StatefulBuilder(
                builder: (context, setState) {
                  updateSelection = setState;
                  return SelectableListCard<int>(
                    items: List.generate(20, (index) => index),
                    selectedIndex: selectedIndex,
                    titleBuilder: (item) => 'Row $item',
                    onSelected: (index, item) {},
                  );
                },
              ),
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    final list = find.byType(ListView);
    final scrollable = tester.state<ScrollableState>(
      find.descendant(of: list, matching: find.byType(Scrollable)),
    );
    expect(scrollable.position.pixels, 0);
    updateSelection(() => selectedIndex = 6);
    await tester.pumpAndSettle();
    expect(scrollable.position.pixels, greaterThan(0));
    final viewport = tester.getRect(list);
    final selectedRow = tester.getRect(find.text('Row 6'));
    expect(selectedRow.top, greaterThanOrEqualTo(viewport.top));
    expect(selectedRow.bottom, lessThanOrEqualTo(viewport.bottom));
    expect(tester.takeException(), isNull);
  });
}
