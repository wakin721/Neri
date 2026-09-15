class ValidationCacheDelta {
  const ValidationCacheDelta._();

  static bool canAdoptPendingEcho<T>({
    required List<T> nextItems,
    required int cachedLength,
    required Map<String, int> cachedIndexByPath,
    required Set<String> pendingPaths,
    required String Function(T item) pathOf,
    required bool groupingSettingsUnchanged,
    required bool refreshVersionUnchanged,
  }) {
    if (!groupingSettingsUnchanged || !refreshVersionUnchanged) return false;
    if (pendingPaths.isEmpty || nextItems.length != cachedLength) return false;

    for (final path in pendingPaths) {
      final index = cachedIndexByPath[path];
      if (index == null || index < 0 || index >= nextItems.length) return false;
      if (pathOf(nextItems[index]) != path) return false;
    }
    return true;
  }
}
