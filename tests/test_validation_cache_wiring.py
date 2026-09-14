from pathlib import Path


SCREEN = Path("frontend/lib/src/screens/species_validation_screen.dart")


def test_incremental_validation_cache_is_wired_before_global_signatures():
    source = SCREEN.read_text(encoding="utf-8")

    assert "../utils/validation_cache_delta.dart" in source
    assert "_pendingValidationEchoPaths" in source
    assert "_bucketCacheItemIndexByPath" in source
    assert "_tryAdoptValidationEchoWithoutGlobalSignatures" in source

    current_buckets = source.index("List<_SpeciesBucket> _currentBuckets()")
    fast_path = source.index(
        "_tryAdoptValidationEchoWithoutGlobalSignatures", current_buckets
    )
    global_signature = source.index(
        "final pathSignature = _itemsPathSignature(widget.items);", current_buckets
    )
    assert fast_path < global_signature

    mark_selected = source.index("Future<void> _markSelected(")
    mark_batch = source.index("Future<void> _markBatch(")
    undo_marks = source.index("Future<void> _undoRecentMarks(")
    assert source.index("_pendingValidationEchoPaths.add(updated.path)", mark_selected) > mark_selected
    assert source.index("_pendingValidationEchoPaths.addAll(", mark_batch) > mark_batch
    assert source.index("_pendingValidationEchoPaths.addAll(", undo_marks) > undo_marks
