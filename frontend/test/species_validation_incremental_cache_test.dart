import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/utils/validation_cache_delta.dart';

void main() {
  test('pending echo checks only pending item paths', () {
    final items = <_Item>[
      for (var index = 0; index < 10000; index++) _Item('path-$index'),
    ];
    final indexes = <String, int>{
      for (var index = 0; index < items.length; index++)
        items[index].path: index,
    };
    var pathReads = 0;

    final accepted = ValidationCacheDelta.canAdoptPendingEcho<_Item>(
      nextItems: items,
      cachedLength: items.length,
      cachedIndexByPath: indexes,
      pendingPaths: const <String>{'path-4321'},
      pathOf: (item) {
        pathReads += 1;
        return item.path;
      },
      groupingSettingsUnchanged: true,
      refreshVersionUnchanged: true,
    );

    expect(accepted, isTrue);
    expect(pathReads, 1);
  });

  test('pending echo falls back when structural preconditions are uncertain', () {
    final items = <_Item>[_Item('a'), _Item('b')];
    final indexes = <String, int>{'a': 0, 'b': 1};

    bool evaluate({
      List<_Item>? nextItems,
      int? cachedLength,
      Map<String, int>? cachedIndexByPath,
      Set<String>? pendingPaths,
      bool groupingSettingsUnchanged = true,
      bool refreshVersionUnchanged = true,
    }) {
      return ValidationCacheDelta.canAdoptPendingEcho<_Item>(
        nextItems: nextItems ?? items,
        cachedLength: cachedLength ?? items.length,
        cachedIndexByPath: cachedIndexByPath ?? indexes,
        pendingPaths: pendingPaths ?? const <String>{'a'},
        pathOf: (item) => item.path,
        groupingSettingsUnchanged: groupingSettingsUnchanged,
        refreshVersionUnchanged: refreshVersionUnchanged,
      );
    }

    expect(evaluate(cachedLength: 3), isFalse);
    expect(
      evaluate(cachedIndexByPath: const <String, int>{'b': 1}),
      isFalse,
    );
    expect(
      evaluate(nextItems: <_Item>[_Item('changed'), _Item('b')]),
      isFalse,
    );
    expect(evaluate(groupingSettingsUnchanged: false), isFalse);
    expect(evaluate(refreshVersionUnchanged: false), isFalse);
    expect(evaluate(pendingPaths: const <String>{}), isFalse);
  });
}

class _Item {
  const _Item(this.path);

  final String path;
}
