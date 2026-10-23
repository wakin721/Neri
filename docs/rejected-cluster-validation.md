# Rejected clusters in species validation

The validation list uses the existing DINOv2 Registry candidate numbers. Runtime
rejections are collected into candidate registrations, which have a single
prototype budget. Their cluster is the same prototype index 0 exposed by the
Registry cluster inspector. The stable identity includes the model fingerprint,
registration ID and prototype index; the visible label is `拒识簇 #N`.

Only candidates with at least four independent Registry events (the shared
`DEFAULT_MIN_CLUSTER_SIZE`) receive a numbered list entry. Smaller candidates
are grouped under `拒识（噪声/证据不足）`; photos/frames from one trigger do not
multiply the event count. This is a minimum-evidence gate, not another density
clustering pass. Event membership uses the existing Registry time/camera rules;
when timestamps are absent, different source files remain separate events.
Support is counted across the candidate's saved evidence, not just the current
folder. Evidence and stable links are retained; every metadata refresh recounts
events, allowing candidates to acquire a numbered entry as they grow. Matching
still considers small candidates so noise cannot be reassigned to a neighboring
supported group. No source media is rescanned. Noise buckets have no cluster
learning action, and the backend enforces the same minimum for that action;
individual manual box review remains available.

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

## Confirming a cluster and learning its features

Select a numbered cluster and use the school icon, `确认簇物种并学习`. The dialog
states the file-update scope and requires one explicit species name. The backend
validates the model fingerprint and every selected observation's cluster before
making any changes. It updates only that cluster's boxes in the current view;
other clusters in the same file retain their labels. Reviewed boxes remain visible
even when their original model confidence is below the display threshold.

For a checkpoint species, all saved independent-event embeddings from the cluster
are added as human feedback and the affected Memory generations are recomputed.
Event grouping, camera requirements, quality gates and rollback to the previous
generation follow the existing learning policy. The UI distinguishes collecting,
provisional assistance, enabled formal learning and evidence needing review. A
new species is named in the Registry candidate state; the separate registration
flow still controls when it participates in inference. No encoder or checkpoint
weight file is retrained or overwritten.

The operation is journaled and retryable under its original operation ID. Undo
restores the affected box labels and feedback generations. Reconfirming a cluster
as another species also recomputes the previous species, so superseded event
features are removed from its active memory. Video summaries retain per-frame
maximum counts instead of summing repeated detections across frames.
