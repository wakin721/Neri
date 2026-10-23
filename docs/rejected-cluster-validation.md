# Rejected clusters in species validation

The validation list uses the existing DINOv2 Registry candidate numbers. Runtime
rejections are collected into candidate registrations, which have a single
prototype budget. Their cluster is the same prototype index 0 exposed by the
Registry cluster inspector. The stable identity includes the model fingerprint,
registration ID and prototype index; the visible label is `拒识簇 #N`.

New inference persists an observation-to-cluster link in the fingerprint-scoped
feedback database after the Registry accepts the evidence. These writes are
batched and do not invalidate the Registry matching cache.

Older results did not retain that association. On the first lookup, the backend
matches only the requested rejected observations' saved features to current
candidate prototypes using the existing 0.82 join threshold and caches successful
links. This recovers membership against the current Registry; it cannot recreate
an original assignment that was never saved. It does not read source media,
rerun detection, or create new candidates. Missing features, unmatched observations
and deleted clusters are shown as `拒识（未分簇）`. A request failure is shown separately
as `拒识（簇信息未加载）`.

Each distinct cluster gets its own list entry. A file or camera event containing
multiple clusters appears in each relevant entry. Known-species context is
retained for mixed results, and the known names are canonicalized so different
name orders do not create duplicate entries. Original species names, box labels
and exported results are unchanged. Full labels and stable IDs are available by
hovering over an entry's status icon.

Human review updates rebuild all memberships for multi-cluster files together.
Ordinary single-membership groups retain the existing incremental update path.
