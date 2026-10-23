# Rejected clusters in species validation

The validation list retains DINOv2 Registry collection identities and candidate
numbers after density verification; these are not the transient integer labels
returned by a particular HDBSCAN run. Runtime rejections are collected into
candidate registrations, which have a single prototype budget. The stable
identity includes the model fingerprint, registration ID and prototype index;
the visible label for verified candidates is `拒识簇 #N`.

Numbered candidates must pass the same density policy as Within-Seq discovery:
raw normalized independent-event means, external `hdbscan`, `min_cluster_size=5`,
`min_samples=3`, Euclidean distance, EOM, and `allow_single_cluster=False`.
The policy is implemented once in `system/dinov2/rejection_clustering.py`.
Having five events alone is insufficient; a homogeneous isolated pool can still
be entirely noise. Density is evaluated across all saved candidate events for
the model, not just the visible folder. A Registry collection candidate receives
a numbered entry only if it contains at least five events and every event is
in the same non-noise HDBSCAN group. A collection mixing supported groups or
containing noise is conservatively kept under `拒识（噪声/证据不足）`, rather than
claiming that the whole collection is one coherent learned species. Distinct
Registry collection identities remain distinct; this check does not rewrite
collection membership, merge candidates, or propagate HDBSCAN labels to training.

Photos/frames from one trigger do not multiply the event count. Event membership
uses the existing Registry time/camera rules; when timestamps are absent,
different source files remain separate events. Evidence and stable links are
retained. A feedback-DB cache stores density results keyed by the shared policy
and a transactionally maintained Registry revision. Event inserts, updates,
deletions and status changes invalidate it, including changes to features with
unchanged counts. Older read-only registries use an exact event-content hash.
Unchanged evidence reuses cached density results across requests and restarts;
concurrent requests share a separate density lock without blocking Registry
writes. Views containing only undersized candidates skip the full-pool fit.
First verification of a large pool can take minutes; the client shows a loading
message and allows that first request up to 15 minutes. Density is never
computed in the inference persistence loop or by opening source
media. Registry matching still includes small candidates. Noise buckets have no
cluster learning action, and the backend enforces both size and density checks;
individual manual box review remains available. The UI requires explicit density
verification, so metadata from an older backend cannot silently enable learning.
Density verification and collection of learning evidence share one WAL read
snapshot; later runtime arrivals are excluded from that human-learning operation.
The existing four-event learning/prototype-summary rules are separate from this
rejection-discovery policy; their arithmetic-mean fallback never verifies a
rejected candidate for the numbered review list.

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
