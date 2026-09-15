# DINOv3 Registry Dual-Routing Design

## Goal

Route DINOv3 observations with the calibrated cosine and squared-distance gates, while allowing learned prototypes to change only after explicit human confirmation.

## Decisions

- The winning prototype is always the nearest prototype in the classifier's comparison space. Cosine and squared distance are evaluated against that same prototype.
- `cosine >= threshold` and `distance <= threshold` means `update_prototype`, but inference alone never performs that update. The observation remains in the feedback journal until the user confirms its species.
- Cosine pass plus distance fail, or cosine fail plus distance pass, routes to an ordinary `candidate`.
- Both gates failing routes to a `new_mode_candidate`.
- Legacy models without a calibrated squared-distance threshold retain legacy Candidate routing; they still may not update a registered prototype without human confirmation.
- Base checkpoint files remain immutable. A confirmed update changes only local learned Registry/feedback evidence.
- Candidate quality is calculated from actual event-to-prototype coverage; a default database value must not make the purity gate pass.
- Independent-event grouping is order-independent and transitively merges bridged intervals using the strict `< event_gap_seconds` rule.
- The canonical manual-registration minimum remains four independent events, matching current tests and the current overlay lifecycle. The stale five-event statement is corrected.

## Compatibility

Existing SQLite files are migrated in place with additive columns and indexes. Existing API fields remain valid; new routing metadata is additive. Registry summary endpoints remain backward compatible while expensive cluster details become lazy-loadable.

## Safety invariants

- No inference-only path calls `record_observation` for an existing Registry entry.
- Human feedback may attach a confirmed observation to a Registry entry and trigger prototype regeneration.
- One process serializes writers per fingerprint-scoped Registry path; SQLite also uses WAL and a busy timeout.
- Failed writes roll back the Registry transaction without publishing partially refreshed prototypes.
