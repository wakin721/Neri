# Training Upload Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development for scoped UI/broker implementation and independent review. Root integrates the persistent queue and verifies deployment.

**Goal:** Implement the approved automatic photo/annotation contribution workflow.
**Architecture:** Flutter explicit consent, local durable queue/worker, bounded cloud upload broker, existing OpenList OneDrive direct upload.
**Tech Stack:** Flutter, Python/Pillow/sqlite3/urllib, FastAPI, nginx/systemd, OpenList.
**Spec:** ../specs/2026-09-05-training-upload.md

## Global constraints

Use the exact API contracts and policy version in the spec. User has approved the feature, server use, destination and operator contact; proceed without further design/permission loops. Work in the clean dedicated codex/training-upload branch in the current workspace so the user's desktop checkout receives changes. Never copy SSH, OpenList, Microsoft tokens or temporary upload URLs into tracked files. Only root operates the remote server. No subagent may dispatch other subagents. Each implementer owns disjoint files; only one implementation subagent runs at a time alongside root's local work.

## Task 1: Consent UI

Files: frontend/lib/src/privacy/*, frontend/lib/src/main_window.dart, frontend/lib/src/api_client.dart, frontend/lib/src/screens/settings_screen.dart, frontend/pubspec.yaml, frontend/assets/legal/user_agreement.md, frontend/test/privacy_test.dart.

- [x] Write dialog tests: neither upload participation option selected initially; acceptance and a choice required; refusal permits normal use; saving error does not mark accepted.
- [x] Run tests to observe missing behavior.
- [x] Implement typed privacy status/API calls, first-start gate before normal refresh, dedicated settings card with current status/disable/clear queue/view agreement.
- [x] Include agreement asset derived from supplied draft, consistent with the spec and operator contact.
- [x] Run focused tests/analyzer, self-review and report. Do not touch Python/cloud files or .gitignore.

## Task 2: Persistent queue (root)

Files: system/training/{policy,queue,media,transport}.py, system/backend/main.py, system/backend/services.py, tests/test_training_upload.py.

- [x] Add unittest behavioral tests using temporary real SQLite/files and fake transport only at the network boundary; run failing tests.
- [x] Persist versioned consent and revisioned jobs. Serialize empty-slot reservations per physical parent identity; first three eligible confirmed-empty photos only. Defer pixel reads/network to the worker. Reject unlabeled/videos/unverified/person/vehicle.
- [x] Generate reencoded stripped JPEG and allowlisted JSON. Build normalized machine box coordinates against encoded pixel orientation; preserve human species/count separately.
- [x] Add retry/debounce, revision checks, cancellation and removal; keep sample secrets private and retry identities stable.
- [x] Add local privacy API, initialize/stop worker, enqueue after successful annotation persistence, exclude generic settings consent overrides.
- [x] Run queue/media/API tests and existing appropriate Python regressions.

## Task 3: Cloud broker

Files: server/training_broker/{app,store,openlist_client}.py, server/training_broker/requirements.txt, server/training_broker/README.md, tests/test_training_broker.py.

- [x] Use unittest/FastAPI tests to exercise ownership, invalid path-like input, pair completion, superseding revisions, quotas and request limits with a fake OpenList network boundary. Run red before implementation.
- [x] Implement the exact cloud contract; sessions from local OpenList /api/fs/get_direct_upload_info, tool HttpDirect. Use fresh attempt names and encoded File-Path. Refresh the parent through /api/fs/list before /api/fs/get; mkdir/remove via scoped OpenList paths.
- [x] Use fixed root from config, protected token from environment, persistent SQLite, sanitized errors and bounded urllib timeouts. HTTP routes do not accept caller-specified root, filename or upstream URL.
- [x] Provide systemd/nginx deployment instructions scoped to /api/training, default loopback binding and no secret examples.
- [x] Run tests and self-review; root performs deployment.

## Task 4: Integration, review and deployment

- [x] Review worker/UI/broker contracts and all privacy boundary tests, fix gaps.
- [x] Independently review the combined diff, focusing on opt-in, empty limits, revision concurrency and arbitrary uploads/deletion.
- [x] Prepare dedicated service account/config, restrict only /Neri_Data/Training, install broker in /opt/neri-training, nginx route to 127.0.0.1:8001, health check and rollback backup.
- [x] Upload only a synthetic JPEG+JSON pair through public broker and Microsoft session; verify expected directory/pair and cleanup exact test files.
- [x] Run final appropriate tests/analyzer/build. Report actual deployment/build status and privacy limitations, with files for review.

## Final verification note

Implemented/reviewed and deployed. Python suite:38passed; privacy widgets:11passed; live synthetic direct-upload/revision/delete and Training ACL probes passed. Full frontend tests retain one pre-existing PowerShell ZIP fixture failure. Flutter asset bundle built successfully. Windows ARM64 executable linking failed against the existing x64 media_kit MPV/ANGLE libraries; installed Flutter chooses native Dart ABI and exposes no alternate Windows target flag. No new executable was produced. See ../../training-upload-deployment.md and server README for evidence and operational limits.
