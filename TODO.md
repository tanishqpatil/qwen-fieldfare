# Authoritative implementation backlog

## Current milestone — S3-01c two-row expert I/O feasibility

- [x] Verify every actual SLRU decision, victim, final slot and partition state offline.
- [x] Classify safe reads and retirement hazards for adjacent rows.
- [x] Bound critical-path credit, extra staging, and implementation overhead.
- [x] Close the scheduling direction and review the evidence-only checkpoint.

Decision C. Most destinations are address-safe. The ordered model's gross
1K I/O and wake credit is 6.19%, and the more generous combined estimate
provides limited headroom. No runtime work is retained.
Next independent priority is a measurement of default sampled-token
selection cost in sustained decode. It remains unstarted.
See the [two-row evidence](docs/evidence/2026-11-18-s301c/README.md).

## Historical milestone — S3-01b bounded router readback

- [x] Establish layer-major ordering, packet lifetimes and safe slot retirement.
- [x] Bound the available savings with short, 1K and suffix diagnostics.
- [x] Select the feasibility outcome and review the evidence-only checkpoint.

Decision D. The bounded schedule is possible, but the optimistic 1K
estimate of 19.52% excludes remaining host work and misses the 20% gate.
No prototype or larger benchmark was justified.
See the [feasibility evidence](docs/evidence/2026-11-18-s301b/README.md).

## Historical milestone — S3-01a bounded chunked prefill

- [x] Isolate development from the published S3-00d tip and define the execution contract.
- [x] Implement and test one eight-row layer-major projection candidate.
- [x] Qualify state, strict reference, sustained decode and prefix reuse.
- [x] Measure fresh prompts and suffixes, decide promotion and prepare the scoped evidence.

Decision C. Projection-only batching passed correctness and missed the 20%
TTFT target. Tokenwise execution and exact-prefix reuse are restored to the
S3-00d source. The next proposed experiment batches router readback by layer.
See the [design](docs/BOUNDED_PREFILL_CONTRACT.md) and
[qualification evidence](docs/evidence/2026-11-18-s301a/README.md).

## Historical milestone — S3-00d restore exact-prefix state reuse

- [x] Restore the one-runner ledger and opt-in CLI/native server controls over fixed router order.
- [x] Run model-free state-contract and terminal-delivery lifecycle tests.
- [x] Recheck the archived 430-token prompt and 512-output sustained boundary with full-logit and hidden-state samples.
- [x] Complete the short tool-cycle, real mismatch and queue/disconnect/clear/restart/overflow qualification.
- [x] Measure short, 1K, 4K and longer repeated workloads and qualify the real OpenCode read/edit/test cycle.
- [x] Verify native settings persistence and owned server controls on the supported Mac.
- [x] Complete final validation and review the scoped implementation and evidence for feature-branch publication.

Qualification passed on the local Mac at 8K. Opt-in remains required.
See the [qualification evidence](docs/evidence/2026-11-18-s300d/README.md),
[state contract](docs/EXACT_PREFIX_STATE_CONTRACT.md) and
[independent strict numerical gate](docs/NUMERICAL_NON_REGRESSION.md).

## Historical milestone — S3-00c reduction decision

S3-00c selects the original fixed router-order expert reduction as the
production full-replay baseline. The grouped-tail experiment is closed.
The strict MLX 131/132 blocker remains. Stop here. The next authorized
S3-00 iteration should qualify opt-in exact-prefix reuse against this baseline.

- [x] Verify the two combine groupings on actual Metal sentinels.
- [x] Compare saved MLX and Swift controls with the grouped variant's two histories.
- [x] Choose original router order and integrate one bounded scratch buffer.
- [x] Verify production against matched captures and run package/release/strict checks.
- [ ] Future S3-00: restore the off-by-default prefix controls and requalify
  512-output continuation, restart/cancellation and repeated-turn latency.

See [S3-00c decision and handoff](docs/evidence/2026-11-18-s300c/README.md).

## Historical milestone — S3-00b deterministic expert reduction

S3-00b ended with Outcome B (revise). Router order eliminates the measured
residency variance and preserves similar aggregate MLX fidelity, but selects
1031 over 285 at the sustained boundary where both MLX schedules select 285.
The original strict 131/132 issue remains. Production matches c1ff7c9.
S3-00 prefix reuse stays blocked pending a qualified reduction.

- [x] Publish the S3-00a investigation on the feature branch (c85edac).
- [x] Compare both teacher histories with native and tokenwise MLX.
- [x] Check bitwise router-order warm/cold agreement and strict non-regression.
- [x] Record the revision outcome and retain the exact probe and metrics.
- [x] Evaluate only the explicit `residual + (routed + shared)` combine variant in S3-00c.

See [S3-00b report](docs/evidence/2026-11-18-s300b/README.md).

## Historical milestone — S3-00a expert-cache residency

S3-00a characterized residency-dependent MoE accumulation. The experiment is
archived and production uses full replay with the original sum. S3-00 remains
blocked. Preserve the strict MLX fixture and the 131/132 release blocker.
The next gate is review of matched logits and hidden values before promoting
a router-order path. Stop at the S3-00a report.

### S3-00a — characterized, experiment only

- [x] Isolate reduction-order changes with identical per-expert weights and outputs.
- [x] Reproduce output 465 under identical teacher history and inspect logits.
- [x] Confirm exact router-order output across the tested cache histories.
- [x] Preserve original strict MLX fixture and record the prototype's existing 131/132.
- [x] Complete 128/512 performance comparisons and restored-tree validation.
- [ ] Separate review of candidate scores against a matched MLX reference before promotion.

S3-00 remains blocked. See [S3-00a evidence](docs/evidence/2026-11-18-s300a/README.md).

## Historical milestone — S3-00 exact-prefix state reuse

Explicit authorization supersedes S2-03 stop. Measure replay baselines first,
then one bounded in-place exact consumed-token prefix per active runner,default
off,with validated commit/invalidation and shared CLI/UI controls. No state
rewind approximation,kernel/math/precision/expert-policy changes or second owner.
Stop at first validated reuse milestone;review/local commit only,no push/merge.

S3-00 qualification failed on sustained same-Swift raw-token equivalence. The
candidate is archived and production code restored to S2-03. Stop after restored
validation/report; no cache controls ship and no math/kernel fix is authorized.
See docs/evidence/2026-11-18-s300/README.md for the failed gate and bounded diagnostic.

### S3-00 — blocked by sustained equivalence; candidate reverted

- [x] Measure original replay: short,1K,4K,6K and three sequential tool results.
- [x] Inspect/test consumed-token,absolute-position,EOS and pending-token contract.
- [x] Prototype bounded in-place reuse,CLI/UI controls and transactional invalidation;
  preserve complete implementation/tests in the rejected candidate patch.
- [x] Five alternating short tool-cycle pairs agree on raw IDs/finish reasons.
- [x] Stop on first sustained512 divergence: output index465,fullReplay1031/cached285.
- [x] One full-replay-only diagnostic reproduces expert-residency history sensitivity.
- [x] Restore all production Package/Sources/configuration to S2-03; no cache flag,
  UI cache control or numerical kernel change ships. Preserve failed evidence.
- [ ] Complete exact-prefix qualification and ship controls — **blocked**, not done.
- [x] Restored-tree153 passes+3 skips,release41.88s,checker6,archived-log stability,
  docs/diff/source-identity checks. Reviewed investigation checkpoint;no feature promotion.

Short cycles show useful potential but no promotion: raw equivalence failed.
Next safe task is a separately scoped proposal for residency-independent expert
reduction and its numerical/reference gates,or independently reviewed acceptance
policy; do not implement either under this no-math-change mission. Strict MLX
131/132 remains a separate release blocker. See
[S3-00 evidence](docs/evidence/2026-11-18-s300/README.md).

## Historical milestone — S2-03 native server management

Explicit user authorization supersedes S2-02 stop. Checkpoint S2-02, reuse the
existing Core/backend/HTTP implementation in native server controls, preserve
single ownership and client-only tools. Stop and locally commit after S2-03.
No prefix reuse, inference math changes, remote push or merge.


## Historical milestone — S2-02 CLI and real OpenCode qualification

Explicit user authorization supersedes the prior S2-01 stop. Preserve completed
server/tools/readers and known131/132 release blocker. Implement validated CLI
controls and qualify installed OpenCode in a disposable repository. Stop after
S2-02; do not begin S2-03 SwiftUI management in this execution. No push/merge.


## Historical milestone — S2-01 structured tool contract

Explicit user authorization supersedes the prior S2-00 stop. Preserve its server
checkpoint and all reader/numerical evidence. Implement native Qwen tools, JSON
and SSE translation, matched tool results and one real continuation. Never execute
tools in the server. Stop after S2-01; no prefix reuse/readers/Hermes/push/merge.


## Historical priority override — S2-00 (2026-11-17)

Explicit user decision supersedes earlier Stage1-only/order/stop directives.
Build and qualify the minimal local QwenFieldfareServer now. S1-01e and remaining
noncritical reliability work are deferred, not completed or abandoned. Preserve
S1-01a–d and strict131/132 release blocker. Stop after real nonstream/SSE HTTP
generation; tools, prefix reuse and further reader migrations remain deferred.
No push/merge. Current implementation/validation status is in DEVELOPMENT_LOG.


Mission accepted 2026-11-17; base `72fb923`. Stable IDs below use the mission's
stage numbers. Existing AGENT_RUNTIME_PLAN optimization stages are historical.

## Working rules

Read AGENTS.md, this file and docs/AGENT_MEMORY.md at session start. Inspect
related source and baseline before edits. Preserve unrelated work and historical
logs. Update status whenever work changes; statuses are pending, in progress,
blocked, completed or rejected. Every stage must pass applicable acceptance
before the next begins. Follow the current milestone boundary. Stop at a new
applicable hardware, checkpoint, environment or correctness blocker. The known
131/132 baseline permits independent engineering. Verify the milestone before
publishing to its development or feature branch. Main is never merged automatically.

Modify only task-related source/tests/docs. Keep app/bench working, checked
allocations, exact model semantics, one model owner, raw weights external and
the selected memory budget. No generic hosting, server-side tools, unrelated
models, silent history truncation, cache-policy replacement without evidence or
unqualified performance claims. Details: [architecture](docs/ARCHITECTURE.md),
[decisions](docs/DECISIONS.md), [validation](docs/PERFORMANCE_BASELINES.md).

Core paths below omit `Sources/QwenFieldfareCore/`; Mac paths omit
`Sources/QwenFieldfareApp/`. Proposed paths are not existing functionality.

## Stage 0 — Repository audit and engineering foundation

### S0-01 — Audit checkout, ownership and runtime/test paths

- Stage/priority/status: 0 / P0 / **completed**.
- Files: `AGENTS.md; Package.swift; Sources/; Tests/; Scripts/`.
- Dependencies: none.
- Acceptance: Source inventory and gaps documented; preserve clean starting work.
- Required tests: Source review, git status/log.
- Risks/blockers: Unchecked mmap readers and fatal Metal paths found.

### S0-02 — Establish persistent recovery documents and reconcile stale claims

- Stage/priority/status: 0 / P0 / **completed**.
- Files: `TODO.md; docs/ARCHITECTURE.md; docs/AGENT_MEMORY.md; docs/DECISIONS.md; docs/DEVELOPMENT_LOG.md; docs/PERFORMANCE_BASELINES.md; docs/AGENT_INTEGRATION.md; README.md`.
- Dependencies: S0-01.
- Acceptance: All required docs exist; implemented/proposed separated; stage IDs unambiguous.
- Required tests: Markdown links; diff check.
- Risks/blockers: Historical narrative must remain intact.

### S0-03 — Record build/tests, parity inventory and feature branch

- Stage/priority/status: 0 / P0 / **blocked**.
- Files: `Scripts/test.sh; Scripts/test_teacher_reference.py; Scripts/qwen_mlx_greedy_reference.py; Package.resolved`.
- Dependencies: S0-01.
- Acceptance: Record exact checks/missing validation and preserve known failing numerical baseline; independent Stage 1 work is authorized by revised policy. Strict parity remains a release gate.
- Required tests: Full package tests; release build; reference inspection; checkpoint-hashed 132/132 teacher-forced control.
- Risks/blockers: Known 131/132 at token 61 remains unresolved and blocks release/full-parity claims, not independent Stage 1. Original fixture and evidence stay intact; no further broad/layer-3 investigation is scheduled. See ADR-008 and docs/NUMERICAL_NON_REGRESSION.md.


## Stage 1 — Reliability and correctness hardening

Stage 1 status: **deferred**; 131/132 is a retained release blocker, not
an independent-task blocker. Completed S1-03a, S1-01a, S1-01b, S1-01c and S1-01d; next proposed task is S1-01e. Parent tasks
remain open until their full acceptance criteria pass.

### S1-03a — Bound and validate model metadata reads

- Stage/priority/status: 1 / P0 / **completed**.
- Files: `QwenModelDirectory.swift; Tests/QwenFieldfareCore/QwenModelDirectoryTests.swift`.
- Dependencies: S0-02; recorded S0-03 baseline. Does not depend on completing the unrelated runner header refactor.
- Acceptance: Existing compatible metadata accepted; reject escaping/non-regular metadata before index read; capped descriptor-checked index read; malformed input returns typed errors. No inference/sampling/kernel changes.
- Required tests: Model-free directory/FIFO/symlink/oversized/malformed index fixtures, positive internal links, full suite and release; inference non-regression identity check.
- Risks/blockers: Selection validation alone does not harden later runner opens; directory mutation/descriptor-relative containment still need lifecycle/IO work. This task does not complete S1-03 or Stage 1.

### S1-01a — Shared bounded Safetensors header reader

- Stage/priority/status: 1 / P0 / **completed**.
- Files: `Infrastructure/ModelIO/QwenSafetensorsHeader.swift; Tests/QwenFieldfareCore/QwenSafetensorsHeaderTests.swift; Package.swift`.
- Dependencies: Recorded known baseline and S0-02; reuse existing partial reader checks before creating a shared utility.
- Acceptance: Shared bounded header/extent validator and descriptor-owner foundation with model-free fault fixtures; active reader integration is a separate task. Delivered 18 focused and 73 full tests, release build. Existing reader/math paths unchanged.
- Required tests: Truncated/oversized/malformed headers, invalid offset count/order/bounds, overflow; focused/full tests before bounded model-backed integration check. Existing teacher integrity verification must remain enabled for that integration check.
- Risks/blockers: Now used by CPU resident and both Q4 runtime readers; pack-builder consumer remains pending. Caller must enforce root containment and immutable file ownership; stat checks cannot lock out concurrent writes. Canonical packing/duplicate JSON key/payload-content validation is not claimed. Preserve original known mismatch set; source-inventory shortcut is not relaxed for the added file.

### S1-01b — Migrate one existing reader to validated handle ownership

- Stage/priority/status: 1 / P0 / **completed**.
- Files: `QwenResidentCoreLoader.swift; QwenSafetensorsHeader.swift; related IO tests` (first consumer); Q4 runners and pack builder remain separate follow-up integrations.
- Dependencies: S1-01a.
- Acceptance: Completed for QwenResidentWeights CPU utility only: shared parser/owned FD, data-relative offsets, contained internal links, typed errors, invalidate/revalidate on failure. Focused/full/release and actual checkpoint integration pass; one strict replay retains exact known 131/132 failure. Q4 runtime/pack readers unchanged.
- Required tests: Valid BF16/F32 reads, corrupt/mutated/replaced files, checked read/row bounds, descriptor cleanup, focused/full/release. Gate affected inference-reader integrations with appropriate bounded model checks and existing integrity verification.
- Risks/blockers: Files must remain immutable while owned; descriptor identity alone cannot prevent concurrent in-place writes/truncation or mmap SIGBUS. Numerical source inventory baseline cannot be treated as proof after integration.

### S1-01c — Migrate the active-streaming shard reader

- Stage/priority/status: 1 / P0 / **completed**.
- Files: `QwenQ4ActiveExpertStreamingRunner.swift; QwenMappedSafetensorsFile.swift; QwenSafetensorsHeader.swift (comment); QwenMappedSafetensorsTests.swift; Package.swift`.
- Dependencies: S1-01a, S1-01b.
- Acceptance: Delivered shared validated descriptor/header and ARC mmap ownership for active streaming only. Metal no-copy buffers retain owners through GPU use; checked page/range and fixed-consumer extents; safe selected pread errors clear slots after hit work completes. Internal links/containment and bounded streaming retained. No kernel/precision/sampling or second-reader migration. 16 focused pass + 1 opt-in skip; full 99 pass + 2 skip (101 reported); checkpoint integration passes; release/checker/docs/diff checks pass. One strict replay exits 1 with unchanged 131/132, token 61 mismatch. Evidence: `docs/evidence/2026-11-17-s101c/validation.json`.
- Required tests: Model-free malformed/shard/path/mapping failures and ownership tests; focused/full/release; one bounded checkpoint load/parity check with required integrity verification and known mismatch-set comparison.
- Risks/blockers: File immutability is still required; descriptor/stat checks cannot prevent concurrent truncation/SIGBUS. Model metadata should validate before unsafe pointer access; allocator/lifecycle hardening remains separate. Stopped after S1-01c; Stage 1 remains open.

### S1-01d — Migrate fallback Q4ChatRunner shard reader

- Stage/priority/status: 1 / P0 / **completed**.
- Files: `QwenQ4ChatRunner.swift; QwenFallbackReaderTests.swift; Package.swift; shared owner reused without modification`.
- Dependencies: S1-01c.
- Acceptance: Delivered shared validated descriptors/header/mmap/Metal ownership in fallback only. Preserved fast-resident/lazy layer-cache/no-copy strategy and layouts; reused existing fixed-Qwen extent guard, checked context/allocation arithmetic, typed initialization errors. Eight focused pass + one skip; package107 pass +3 skips (110 reported), release/checker/docs/diff gates pass. One bounded actual memory-saver integration passes (context8, cache1, two reset forwards846/846,80 creations/79 evictions). Active teacher path unchanged; retained strict131/132 is not fallback validation. Evidence: `docs/evidence/2026-11-17-s101d/validation.json`.
- Required tests: Model-free IO/ownership faults, focused/full/release, bounded affected-mode model integration with existing integrity and numerical policy.
- Risks/blockers: Immutable checkpoint still required; full pre-allocation model contract, index/allocator/encoder safety and production numerical acceptance remain open. Stopped after S1-01d. Cumulative review/local Git checkpoint authorized; no push/merge. Stage1 and production qualification remain open.

### S1-01e — Migrate pack-builder Safetensors input reader (proposed next)

- Stage/priority/status: 1 / P0 / **deferred**; not started.
- Files: `QwenQ4ExpertStoreBuilder.swift; shared descriptor/header owner; focused pack fixtures`.
- Dependencies: S1-01d.
- Acceptance: Inspect pack builder's input validation/ownership and existing256MiB cap; reuse shared infrastructure with an explicit compatibility decision, not an automatic cap reduction. Preserve packed format/source identity/integrity checks and output semantics. No other storage-format migration bundled.
- Required tests: Model-free malformed inputs, offsets/overflow, failure cleanup and packed output fixtures; full/release and appropriate integrity/integration checks.
- Risks/blockers: Input cap policy and output publication/cleanup need review; full model-contract and allocator/lifecycle work remains separate. Requires a subsequent task; stop now.

### S1-01 — Unify bounded Safetensors parsing and safe shard ownership

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `Infrastructure/ModelIO/; Runtime/Generation/QwenQ4*Runner.swift; Storage/`.
- Dependencies: S0-02 and recorded S0-03 baseline; its unresolved strict-release subgate does not block this independent task.
- Acceptance: All readers reject short/oversized headers, invalid offsets and failed stat/read; close/unmap on all failures; no heap shard copies.
- Required tests: Model-free corrupt fixtures, read failures, checked arithmetic; valid model parity.
- Risks/blockers: All runtime readers are migrated; pack-builder Safetensors reader remains independent.

### S1-02 — Preflight exact Qwen model and tensor contract before allocation

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `Infrastructure/ModelIO/; Runtime/Generation/QwenQ4*Runner.swift`.
- Dependencies: S1-01.
- Acceptance: Required tensor names/shapes/dtypes/byte sizes and affine quantization validated; extras permitted; tokenizer/template metadata checked.
- Required tests: Mutated config/tensors, unsupported formats, valid checkpoint; parity.
- Risks/blockers: Do not silently change checkpoint identity or numerical behavior.

### S1-03 — Enforce model-root and regular-file policy on every entry point

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `QwenModelDirectory.swift; runner loaders; expert stores; Bench/; Pack/`.
- Dependencies: S1-01.
- Acceptance: Canonical containment, safe metadata/shards, regular-file checks; define symlink and concurrent modification policy.
- Required tests: Escaping links/paths, missing files, directory/FIFO fixtures; no hangs.
- Risks/blockers: Selection-time validation alone does not protect later opens.

### S1-04 — Make Metal allocation and command failures recoverable

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `Infrastructure/Metal/MetalContext.swift; Kernels/; Runtime/Generation/; Storage/Q4BoundedExpertCache.swift`.
- Dependencies: S1-01,S1-02.
- Acceptance: Replace production fatal errors/forced buffers; propagate command completion errors; invalidate partial state.
- Required tests: Inject buffer/queue/encoder/command failures; retry; parity.
- Risks/blockers: Every runtime mode must preserve existing behavior and bounds.

### S1-05 — Unify cancellable lifecycle and error recovery

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `Mac/Generation/ChatService.swift; Core runtime/loading support`.
- Dependencies: S1-02,S1-03,S1-04.
- Acceptance: Cancellation during load/prefill/decode releases ownership; idempotent unload; recoverable retry; consistent visible/internal history.
- Required tests: Fault injection; cancellation boundaries; repeated load/generate/cancel/unload soak.
- Risks/blockers: Detached tasks and secondary loading path require ownership review.

### S1-06 — Qualify Stage 1 and stop for review

- Stage/priority/status: 1 / P0 / **deferred**.
- Files: `Scripts/; Tests/; docs/DEVELOPMENT_LOG.md; docs/PERFORMANCE_BASELINES.md`.
- Dependencies: S1-01,S1-02,S1-03,S1-04,S1-05.
- Acceptance: Independent reliability tests/package/release and numerical non-regression pass; known strict-parity failure explicitly retained; no memory regression; measured unload tolerance; stop before Stage 2. Production readiness remains separately blocked.
- Required tests: Full serial suite; release; six-case teacher suite; memory and lifecycle soak.
- Risks/blockers: No stage completion based solely on mocks or compilation.


## S2-00 — First local HTTP server (active priority override)

- Status: **completed**, ahead of deferred Stage1 by explicit user decision.
- Validation:113 package passes +3 skips; release; six checker tests; real
  JSON/SSE/reuse/context/disconnect/shutdown HTTP smoke passes. Initial signal
  isolation crash fixed and failed-run evidence preserved. See S2-00 evidence.
- Scope: SwiftNIO HTTP1, one Core runner, loopback bearer auth, bounded serial queue,
  health/models/chat JSON+SSE, cancellation/shutdown; no tools or prefix reuse.
- Tests: protocol/validation/auth/SSE/queue/cancellation/init/shutdown; real HTTP
  generation gate. Preserve strict131/132 release blocker independently.
- Next: S2-01 structured tools after this milestone; not started here.

## Stage 2 — Qwen chat and structured tools

### S2-01 — Native Qwen structured tool contract

- Stage/priority/status:2/P0/**completed** under the current explicit mission.
- Files: ServerSupport ToolJSON/Tools/ToolPrompt/NativeTools,Protocol/Service/HTTP,
  parser/golden/HTTP tests and real client tool-cycle script.
- Acceptance: typed function schemas/history/ID associations, exact checkpoint
  Jinja prompt, bounded incremental parser and validated JSON/SSE tool deltas;
  genuine JSON and SSE call/result/final-answer cycles both pass. Server executes
  no tools. No-tool behavior/auth/loopback/queue/cancellation preserved.
- Validation:127 package passes +3 skips;14 focused tool tests;release/checkers,
  exact Python Jinja golden comparison; real outputs preserved in S2-01 evidence.
- Limitations:developer/named-required/strict/ref-schema modes unsupported; delayed
  tool deltas; delimiter-bearing tool data rejected; no full OpenCode qualification.
- Stop here; next task is OpenCode provider/configuration and real client cycle.

### S2-02 — CLI usability and actual OpenCode qualification (completed checkpoint)

- Stage/priority/status:2/P0/**completed**; supersedes the historical numbering below.
- Files: ServerConfiguration/CLI/admission/health,Core inference lease,SDK schema
  support,configuration/ownership/wire tests,OpenCode scripts and integration docs.
- Delivered: validated host/port/context/budget/queue/model ID; env or private key
  file; help/version/readiness/shutdown; shared app/Bench/server Q4 ownership.
- Real OpenCode1.19.5: configured/listed/selected local provider;8 Chat Completions
  SSE requests,7 successful read/grep/read/test/edit/edit/retest tools,zero tool
  errors in the successful run,final answer and2 passing tests;1696.73s wall.
  It tested before editing, a documented sequence deviation. Earlier bad absolute
  permission attempt stopped after3 denials; evidence retained;one corrected retry.
- Validation:136 package passes +3 skips;focused configuration/lease and precision
  tests;release/checker/docs/diff gates. Real cancellation drains0.542s;context/
  unsupported output errors explicit400. Competing Bench rejected before allocation.
- Limits: slow full prompt replay (3410–5233 tokens),delayed tool deltas;sampled RSS
  is not a footprint peak;no new numerical replay. Original131/132 blocker retained.
- Evidence: [S2-02 report](docs/evidence/2026-11-17-s202/README.md).
- Historical S2-02 stop reached and checkpointed as f2f5672. S2-03 is delivered
  below. No prefix reuse/Hermes implementation is bundled.

### S2-03 — Native macOS server management (current milestone)

- Stage/priority/status:2/P0/**completed**; stop here and locally checkpoint.
- Delivered: native server section,validated/persistent settings,Keychain/private
  CLI-file authentication,endpoint/safe provider guidance,read-only external CLI
  observation,bounded health counters and last-request timing diagnostics.
- Architecture: same-process CoreBackend+LocalServer;native chat explicitly
  cancels/drains/unloads before server start and is blocked until stop/model
  release. No second runner/child process/HTTP stack/tool executor.
- Real gate: native chat before/after server;UI JSON/SSE/tool result continuation;
  actual OpenCode1.19.5;active-prefill Stop;immediate Restart;owned app Quit/port
  closure;reopen stopped with settings persisted;independent CLI observed and
  still generating after observer app Quit. Conflicting CLI rejected while native
  chat holds ownership. Initial used-port failure preserved/fixed/tested.
- Validation:150 package passes +3 opt-in skips (153 total);release36.68s;checker6,
  historical log stability,docs/diff/source review. No fresh strict parity replay;
  all Core/Bench/pins/reference sources unchanged vs f2f5672. No132/132 claim.
- Limits: shared app/server crash fate,cooperative lease,legacy fatal/load/GPU
  cancellation paths;one timing record,flush≠client receipt,Metal≠RAM/peak;
  current health schema needed for external observation. No native server-chat
  transport/tool display because native chat is disabled in server mode.
- Evidence: [S2-03 report](docs/evidence/2026-11-17-s203/README.md).
- Next: measured repeated-turn latency investigation and exact hybrid-state prefix
  reuse design/gates,only after this boundary. No performance implementation here.

### Historical S2-02 — Parser delivered within S2-01

- Stage/priority/status: 2 / P0 / **completed within S2-01**.
- Files: `Core generation protocol/parser; Tests/`.
- Dependencies: S2-01.
- Acceptance: Complete validated calls only; defined malformed/unsupported handling and finish reasons; no tool execution.
- Required tests: Every split point; Unicode/escapes; nested JSON; multiple calls; duplicate/missing IDs; size limits.
- Risks/blockers: No regex-based structured parsing or partial dispatch.

### Historical S2-03 — Tool conversations delivered within S2-01

- Stage/priority/status: 2 / P0 / **completed within S2-01**.
- Files: `Core protocol; Tests/; docs/AGENT_INTEGRATION.md`.
- Dependencies: S2-02.
- Acceptance: Full serialize/generate/parse/result-history exchange representable and tested.
- Required tests: Golden fixtures; malformed output; streamed and nonstreamed equivalence.
- Risks/blockers: Model-backed behavior remains separate from parser mocks.


## Stage 3 — QwenFieldfareServer

Historical decomposition: S2-00/S2-01/S2-02 now supply the local server, tools
and installed-client workflow. Status notes below distinguish that delivered subset
from broader lifecycle/soak qualification. Current S2-03 UI is delivered above; broad soak/release qualification remains open.

### S3-01 — Select maintained minimal HTTP dependency and define API

- Stage/priority/status: 3 / P0 / **completed within S2-00–02**.
- Files: `Package.swift; Package.resolved; proposed Sources/QwenFieldfareServer/`.
- Dependencies: S2-03.
- Acceptance: Verify necessity/license/maintenance/Swift/macOS; define exact fields/errors/model ID.
- Required tests: Dependency/API review; request decoding tests.
- Risks/blockers: SwiftNIO2.101.3 now explicitly depended on; existing pin retained.

### S3-02 — Implement exclusive owner, worker, bounded queue and shutdown

- Stage/priority/status: 3 / P0 / **in progress; implementation delivered within S2-00–02**.
- Files: `Core lifecycle; app/bench/server ownership support`.
- Dependencies: S3-01.
- Acceptance: One active model owner across processes; networking independent from blocking inference; deterministic queue limits.
- Required tests: Concurrent requests/processes; overload; shutdown; crash/stale ownership recovery.
- Risks/blockers: Cooperating Q4 owner/queue/shutdown implemented and tested;
  exhaustive process-crash/stale-lock qualification remains part of S3-05.

### S3-03 — Implement loopback endpoints and authentication

- Stage/priority/status: 3 / P0 / **in progress; implementation delivered within S2-00–02**.
- Files: `Server health/models/chat handlers; configuration`.
- Dependencies: S3-02.
- Acceptance: Health/models/nonstream chat; local auth; limits; context/usage/finish errors; sensitive payloads never logged.
- Required tests: HTTP auth, malformed/oversized requests, context overflow, queue limits.
- Risks/blockers: No remote exposure default or unrelated endpoints.

### S3-04 — Implement bounded SSE and disconnect cancellation

- Stage/priority/status: 3 / P0 / **in progress; implementation delivered within S2-00–02**.
- Files: `Server streaming transport and inference adapter`.
- Dependencies: S3-03.
- Acceptance: Text/tool streams with backpressure; finite buffers; disconnect releases queued/active work.
- Required tests: Slow/disconnected clients; fragmented SSE; cancellation; tool exchange.
- Risks/blockers: Serialized inference implemented; exhaustive slow-client/socket soak remains part of S3-05.

### S3-05 — Qualify server lifecycle/protocol

- Stage/priority/status: 3 / P0 / **in progress; functional subset complete**.
- Files: `Server tests; docs/AGENT_INTEGRATION.md`.
- Dependencies: S3-04.
- Acceptance: Real HTTP clients generate text and simulated structured exchange; all failure gates pass.
- Required tests: Complete server suite and real-model smoke; memory soak.
- Risks/blockers: Real HTTP/tools/OpenCode functional gates pass; exhaustive memory/socket/lifecycle soak remains open.


## Stage 4 — OpenCode integration

### S4-01 — Inspect and pin OpenCode provider configuration

- Stage/priority/status: 4 / P0 / **completed within current S2-02**.
- Files: `docs/AGENT_INTEGRATION.md; disposable integration fixtures`.
- Dependencies: S3-05.
- Acceptance: Publish configuration only for an installed tested version.
- Required tests: Provider schema and connectivity checks.
- Risks/blockers: Pinned installed1.19.5 v1 config; current website v2 differs.

### S4-02 — Run real coding-agent read/edit/test/retry loop

- Stage/priority/status: 4 / P0 / **completed within current S2-02**.
- Files: `Disposable repository; integration report`.
- Dependencies: S4-01.
- Acceptance: Actual backend completes read/search, edit, test, deliberately failing test repair and final answer.
- Required tests: End-to-end task; record retries/tool validity/context/latency/memory.
- Risks/blockers: Tools execute in OpenCode only. First failed permission attempt and successful run
  sequence deviation retained; this is one bounded task, not broad quality/long-context qualification.


## Stage 5 — Exact conversation-state reuse

### S5-01 — Define exact retained-state and invalidation contract

- Stage/priority/status: 5 / P0 / **implemented in S3-00d for the local server**.
- Files: `Core generation; docs/ARCHITECTURE.md`.
- Dependencies: S4-02.
- Acceptance: One session; exact tokens/positions/KV/Conv/DeltaNet/pending token; no truncation.
- Required tests: State-boundary design review and deterministic fixtures.
- Risks/blockers: Thinking stripping and unconsumed final token affect exact prefix proof.

### S5-02 — Implement opt-in exact-prefix continuation and safe replay

- Stage/priority/status: 5 / P0 / **implemented in S3-00d for the local server**.
- Files: `Active runner; generation session owner`.
- Dependencies: S5-01.
- Acceptance: Reuse only proven continuation; reset on divergence/cancel/error/overflow/tools/template changes.
- Required tests: Multi-turn, identical prefix, tools, changed history/system, new chat, disabled cache, all cancellations/boundaries.
- Risks/blockers: Do not snapshot or multiply memory without accounting.

### S5-03 — Qualify equivalence and measured repeated-turn benefit

- Stage/priority/status: 5 / P0 / **qualified in S3-00d for the tested 8K server workloads**.
- Files: `Bench; Tests/; docs/PERFORMANCE_BASELINES.md`.
- Dependencies: S5-02.
- Acceptance: Cached/full replay token IDs and relevant state agree; report actual reused tokens; predeclared 20% gate.
- Required tests: Deterministic parity and repeated-turn A/B, memory/pressure/swap.
- Risks/blockers: Record unmet target honestly; no manufactured gain.


## Stage 6 — Memory and cache controls

### S6-01 — Expose conversation reuse controls and truthful diagnostics

- Stage/priority/status: 6 / P1 / **pending**.
- Files: `Core session policy; Mac diagnostics; Server configuration`.
- Dependencies: S5-03.
- Acceptance: Disabled/exact prefix, idle expiry, retained-token cap, clear action; report actual occupancy.
- Required tests: Expiry/reset/limits/on-off tests; overflow replay.
- Risks/blockers: KV, conversation state and expert caching are distinct.

### S6-02 — Qualify memory/context settings and unload/reload transitions

- Stage/priority/status: 6 / P0 / **pending**.
- Files: `QwenActiveStreamingMemoryConfiguration.swift; UI/server settings`.
- Dependencies: S6-01.
- Acceptance: Preserve 3/4/5 GiB plans; checked supported context/cache controls; report estimates vs measured use.
- Required tests: All plans/bounds; changed settings; allocation failure; measured footprint.
- Risks/blockers: No live expert-arena resizing or implicit memory increase.

### S6-03 — Investigate optional bounded lazy KV allocation

- Stage/priority/status: 6 / P1 / **pending**.
- Files: `Attention KV ownership; memory planner`.
- Dependencies: S6-02.
- Acceptance: Implement only if worthwhile; budget replacement peaks; exact positions/history; otherwise document rejection.
- Required tests: 4K/8K initial capacity; growth/failure/context limits; replay parity; peak memory.
- Risks/blockers: No sliding-window eviction; full-window memory unchanged.


## Stage 7 — Measured performance research

### S7-01 — Establish representative successful-task throughput baseline

- Stage/priority/status: 7 / P1 / **pending**.
- Files: `Bench/integration workloads; docs/PERFORMANCE_BASELINES.md`.
- Dependencies: S6-02.
- Acceptance: Predeclare tasks and complete timing/cache/memory/correctness metadata.
- Required tests: Reproducible controls; no diagnostic headline runs.
- Risks/blockers: Successful tasks per wall time is primary, not synthetic decode.

### S7-02 — Research bounded batched/chunked prefill

- Stage/priority/status: 7 / P1 / **pending**.
- Files: `Active runner; attention/DeltaNet/MoE kernels`.
- Dependencies: S7-01.
- Acceptance: New hypothesis after rejected head experiment; exact causal/recurrent order; finite scratch and expert pins.
- Required tests: Chunks 16/32/64; cancellation/state parity; >=20% latency promotion.
- Risks/blockers: Architecture-specific; no incompatible upstream attention transplant.

### S7-03 — Measure and optimize production sampling if material

- Stage/priority/status: 7 / P1 / **pending**.
- Files: `QwenGenerationSupport.swift; Bench sampler`.
- Dependencies: S7-01.
- Acceptance: Preserve fixed-seed draws/ties/penalties; retain only measured win.
- Required tests: Fixed-seed equality; paired end-to-end >=5% and positive CI.
- Risks/blockers: Greedy fast-path success does not apply to default sampling.

### S7-04 — Profile GPU/expert IO and 3/4/5 GiB task trade-offs

- Stage/priority/status: 7 / P1 / **pending**.
- Files: `Core kernels/cache; Bench; integration workloads`.
- Dependencies: S7-01.
- Acceptance: Only measured bottlenecks justify changes; preserve SLRU absent new causal evidence.
- Required tests: Controlled A/B; all memory profiles; >=5% decode and positive CI.
- Risks/blockers: Preserve rejected packed/prefetch/cache-policy evidence.

### S7-05 — Evaluate speculative decoding prerequisites only

- Stage/priority/status: 7 / P2 / **pending**.
- Files: `Research decisions; no implementation yet`.
- Dependencies: S7-01.
- Acceptance: Compatible artifact, exact verification/rollback, bounded memory and demonstrated win required before implementation.
- Required tests: Artifact/protocol/state-budget feasibility review.
- Risks/blockers: Speculative implementation unauthorized until these gates and explicit design review.


## Stage 8 — Hermes qualification

### S8-01 — Inspect current Hermes protocol/context and pin version

- Stage/priority/status: 8 / P1 / **pending**.
- Files: `docs/AGENT_INTEGRATION.md; integration fixtures`.
- Dependencies: S4-02,S7-01.
- Acceptance: Document actual installed requirements and auxiliary/session requests.
- Required tests: Provider schema; occupied-context workload design.
- Risks/blockers: Allocated 65,536 context does not prove required quality/performance.

### S8-02 — Qualify actual long-history tools and memory

- Stage/priority/status: 8 / P1 / **pending**.
- Files: `Integration workloads; baseline report`.
- Dependencies: S8-01.
- Acceptance: Real backend behavior, single retained-prefix contention and full memory measured; limitations explicit.
- Required tests: Long-history inference/tool loop; total memory and cancellation.
- Risks/blockers: No compatibility claim without completion evidence.


## Stage 9 — Final reliability and release readiness

### S9-01 — Run complete correctness, lifecycle and memory qualification

- Stage/priority/status: 9 / P0 / **pending**.
- Files: `Tests/; Scripts/; integration workloads`.
- Dependencies: S8-02,S7-02,S7-03,S7-04.
- Acceptance: All applicable tests and supported-context gates pass; unresolved exclusions explicit.
- Required tests: Full tests/release/parity/parser/HTTP/OpenCode; load/unload/cancel/fault/memory/long-context soak.
- Risks/blockers: Rejected optional experiments need documented disposition, not forced implementation.

### S9-02 — Review security, dependencies/licenses and verified usage docs

- Stage/priority/status: 9 / P0 / **pending**.
- Files: `SECURITY.md; THIRD_PARTY_NOTICES.md; README.md; docs/`.
- Dependencies: S9-01.
- Acceptance: Reproducible install/use; supported features only; no secrets/weights; documented limits.
- Required tests: Security and license review; Markdown links; diff check.
- Risks/blockers: Production-ready wording requires actual qualification evidence.
