# Task 1 RED checkpoint

Focused local TDD harness before production changes:

```text
3 failed
- missing load_detection_index_for_filenames
- missing load_validation_index_for_filenames
- runtime loader wrapper still delegated to the legacy full loader
```

The failures were expected feature-missing failures, not import/setup failures. This file records the RED checkpoint because the current execution sandbox cannot clone GitHub; the branch remains the isolated workspace and focused modules are exercised in a temporary local harness.
