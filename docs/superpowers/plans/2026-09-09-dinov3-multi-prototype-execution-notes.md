# DINOv3 Multi-prototype Execution Corrections

These corrections apply while executing `2026-09-09-dinov3-multi-prototype-sync.md` and preserve the approved UI semantics.

1. Candidate -> Provisional remains an explicit human registration transition. Accumulating four independent confirmed events makes a Candidate eligible for `register`, but does not automatically change its status. After explicit registration, Provisional -> Confirmed -> Mature may promote automatically when the event/camera thresholds are met.
2. Provisional eligibility is at least 4 independent confirmed events. Confirmed requires at least 10 independent events and at least 2 cameras. Mature requires at least 20 independent events and at least 3 cameras.
3. The Task 2 centered-distance test must use mathematically consistent data. If the raw normalized event feature is `e0` and `feature_center[0] = 0.25`, the centered value is `0.75`; a prototype equal to `0.75 * e0` gives squared distance 0. Do not assert zero distance against an `e0` prototype.
4. `multi_prototype.pt` was re-inspected before execution: size 163625 bytes; SHA-256 `4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43`; 17 classes; `feature_center` shape `(768,)`; prototypes shape `(51, 768)`; `prototype_class_indices` shape `(51,)`; all 17 `prototypes_per_class` values are 3; `head_type=multi_prototype`; `selection_k=3`; threshold `0.31004515290260315`; decision `squared_euclidean_to_nearest_prototype`; rejection score `cosine_similarity_to_winning_prototype`.
