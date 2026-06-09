List<String> autoSortedQuickMarks({
  required Iterable<String> configuredMarks,
  required Map<String, int> usageCounts,
  required Iterable<String> recentHistory,
  Iterable<String> sessionHistory = const <String>[],
}) {
  final history = <String>[...recentHistory, ...sessionHistory];
  final latestIndex = <String, int>{};
  final recentCounts = <String, int>{};
  for (var index = 0; index < history.length; index++) {
    final species = history[index].trim();
    if (species.isEmpty || species == 'Unknown' || species == '\u7a7a') {
      continue;
    }
    latestIndex[species] = index;
    recentCounts[species] = (recentCounts[species] ?? 0) + 1;
  }

  final originalIndex = <String, int>{};
  final configuredSpecies = <String>[];
  var index = 0;
  for (final value in configuredMarks) {
    final species = value.trim();
    if (species.isEmpty) {
      index++;
      continue;
    }
    originalIndex.putIfAbsent(species, () => index);
    configuredSpecies.add(species);
    index++;
  }

  final sorted = <String>{
    ...configuredSpecies,
    ...usageCounts.keys.map((species) => species.trim()),
    ...recentCounts.keys,
  }.where((species) => species.isNotEmpty).toList();

  sorted.sort((a, b) {
    final recentCompare = (recentCounts[b] ?? 0).compareTo(
      recentCounts[a] ?? 0,
    );
    if (recentCompare != 0) return recentCompare;

    final usageCompare = (usageCounts[b] ?? 0).compareTo(usageCounts[a] ?? 0);
    if (usageCompare != 0) return usageCompare;

    final latestCompare = (latestIndex[b] ?? -1).compareTo(
      latestIndex[a] ?? -1,
    );
    if (latestCompare != 0) return latestCompare;

    final aOriginalIndex = originalIndex[a];
    final bOriginalIndex = originalIndex[b];
    if (aOriginalIndex != null && bOriginalIndex != null) {
      return aOriginalIndex.compareTo(bOriginalIndex);
    }
    if (aOriginalIndex != null) return -1;
    if (bOriginalIndex != null) return 1;
    return a.compareTo(b);
  });

  return sorted;
}
