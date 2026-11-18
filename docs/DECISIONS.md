# Architecture decisions

## ADR-001 — Preserve the Qwen hybrid and bounded streaming (accepted)

2026-11-17; base `72fb923`. Keep Qwen3.6-35B-A3B compatible MLX affine Q4,
30 recurrent and 10 full-attention layers, top-eight routing and the existing
SLRU cache. Generic model abstractions and a second owner are out of scope.
This preserves numerical invariants and makes memory costs reviewable.
Alternative: replace the runtime with a generic serving engine; rejected by
mission scope. Source: [architecture](ARCHITECTURE.md).

## ADR-002 — Qwen-specific local server (accepted scope; not implemented)

2026-11-17. The implementation mission explicitly authorizes a dedicated
QwenFieldfareServer after reliability and tool-contract gates. This supersedes
analysis-only server deferrals, while retaining single ownership, loopback by
default, authentication and client-side tool execution. A generic multi-model
host, shell executor and silent second model allocation remain prohibited.
HTTP dependency selection, auth details and process locking remain design work.
No OpenCode/Hermes compatibility is claimed before real integration tests.

## ADR-003 — One authoritative backlog and sequential mission stages (accepted)

2026-11-17. [TODO](../TODO.md) owns stable IDs and current stage gates. Earlier
numbered optimization stages are historical, not the mission's stage numbers.
Preserve existing measurements and rejection records. Do not start Stage 2
before Stage 1 acceptance and the requested review stop. Repository memory
must recover the work without external graph services or past conversations.

## ADR-004 — Keep historical rejection evidence (accepted)

Existing evidence is retained, not remeasured on 2026-11-17:

- Exact greedy fast path was promoted; sampled generation has no such claim.
- Fixed-table cache bookkeeping had about a 1.2% ideal wall-time ceiling.
- Causal SRRIP/BRRIP and transition prefetch did not justify runtime changes;
  future-aware offline optima are not online results.
- Packed one-read experts lost to raw storage and remain opt-in.
- Unused-head prefill candidate failed its 20% TTFT gate and was reverted;
  commit `72fb923` preserves evidence, not candidate code.

Sources: [optimization protocol](OPTIMIZATION_PLAN.md),
[historical runtime plan](AGENT_RUNTIME_PLAN.md), and
[engineering measurements](QWEN_ACTIVE_STREAMING.md). Retrying requires a new
technical hypothesis, exact correctness and the declared controlled A/B gate.

## ADR-005 — Prefix reuse must include recurrent state (accepted constraint)

2026-11-17; implementation deferred to Stage 5. Compare exact token histories,
retain KV plus Conv/DeltaNet, positions and pending-token boundary. Begin with
one retained session. Divergence, cancellation, errors, changed template/tools,
context overflow or invalid state require reset/full replay. A KV-only cache or
arbitrary history truncation cannot reconstruct the hybrid recurrent state.

## ADR-006 — Keep strict parity after observed margin reversal (accepted)

2026-11-17 investigation: both runtimes have distinct top logits at position 61;
MLX gap +0.125 favors 14988, Swift gap +0.120487213 favors 10000. Two MLX
regenerations reproduce the original fixture. Same-input head checks agree;
upstream hidden vectors differ. Verified BF16/FP32 execution differences are
not proof that all upstream divergence is benign. Classification remains D.

Do not replace expected tokens, special-case position 61, change production
precision speculatively, or introduce a near-tie exemption. Preserve the original
failure and 132/132 gate. Alternative tolerance/state-based policy needs explicit
review and broader calibrated evidence; none is adopted. No concrete defect/fix
has been established, so no production regression test is invented to certify it.
See [investigation](evidence/2026-11-17-investigation/REPORT.md).

## ADR-007 — Separate local numerical equivalence from full-model qualification

2026-11-17. Matched-FP32 layer-0 operation/state tests agree closely. Native
execution differs intentionally in activation/history precision and prompt
batching; MLX recurrence state is FP32 despite BF16 activations. Layer-0 routing
sets match in the selected sample. These facts support local semantic consistency
but do not establish the full cause of the final token reversal or exclude a
defect in untested layers/full attention.

Keep current strict 132/132. Proposed supplemental gates (same-input operations,
state evolution, per-layer/logit errors/margins, longer workloads and pinned
precision/hardware) need human review and independently calibrated tolerances.
No near-tie exemption, new threshold, global precision cast or production fix.
See [contract](NUMERICAL_CONTRACT.md) and
[isolation evidence](evidence/2026-11-17-isolation/REPORT.md).

## ADR-008 — Known numerical baseline no longer blocks independent engineering

Accepted by explicit user decision, 2026-11-17. Preserve current 131/132 failure,
original fixture, evidence and strict tests unchanged. It remains unresolved and
blocks release-quality/full-parity/production-readiness claims. Supersedes prior
instructions to stop all development or investigate layer 3 next.

Proceed with independently verifiable Stage 1 reliability tasks, one small task
per iteration. Maintain a separate, explicitly labelled non-regression check;
a known-issue baseline result is never a strict-parity pass. New/worse numerical
failures block affected changes. Performance/numerical work retains appropriate
reference/behavioral gates. Do not repeat expensive model runs or full checkpoint
hashing unless required by a changed path or an integration/identity/integrity gate.
No required integrity check may be bypassed.

Do not resume broad numerical/layer-3 work without a specific hypothesis, bounded
plan and actionable decision, or a change directly requiring it. Numerical-policy
acceptance still needs independent review and sufficient evidence. Current scope
is Stage 1 only, then report before tool/server implementation. No push/merge.

## ADR-009 — Separate header validation from reader integration

2026-11-17, S1-01a. Build an internal bounded parser/descriptor owner using the
existing tensor-info type, 100,000,000-byte cap and positional-read helper; do
not alter active readers or inference mathematics in this iteration. Reject
invalid size/type/range metadata with typed errors before payload access.

Consumers must retain/use the validated descriptor, not reopen a mutable path.
Size/mtime/ctime checks detect observed mutation but do not lock out writes;
model-root containment and immutable-checkpoint/mmap lifecycle policy remain
caller/integration responsibilities. Current readers do not yet use this path.
The source-inventory non-regression baseline is not weakened to accommodate the
new unused file; scoped review establishes isolation, not numerical equivalence.

## ADR-010 — First consumer migration keeps descriptor ownership explicit

2026-11-17, S1-01b. Migrate QwenResidentWeights CPU BF16/F32 utility first: no
mmap/zero-copy coupling and small fixture coverage; do not conflate it with Q4
Fast Resident mode. Reuse validated descriptors/header metadata for every later
tensor/row read; never reopen by cached path. Serialize ownership, invalidate on
failed operations, and require explicit retry/revalidation. Preserve contained
internal links and original data-relative offsets/bit values.

Retain 100,000,000-byte cap: actual checkpoint headers 88,906/67,237/67,875/50,269
bytes validate; cap is not expanded and narrowing export compatibility is not
bundled into this ownership migration. Cap bounds header bytes, not JSON-object
heap or process peaks. FD identity is not an immutability/content guarantee;
checkpoint writes while owned remain unsupported. Index/root-ancestor and Q4
mmap safety work remain open. One direct reader checkpoint test and one strict
parity replay are separate evidence, since active inference uses another reader.
Strict exit 1/131-of-132 remains visible and unresolved. No policy relaxation.

## ADR-011 — Runtime mappings retain validated file ownership

Implemented through S1-01c/d,2026-11-17. Both Q4 runtime readers now reuse the
shared bounded header/descriptor path and ARC whole-file mapping owner. This
supersedes the historical not-yet-migrated statements at the S1-01a/b decision
points. Each no-copy Metal buffer retains the map owner; default command buffers
retain buffers through use. Never manually unmap a surviving lease. Files must
remain immutable: descriptor identity/stat checks do not prevent writes or SIGBUS.
Preserve existing resident/cache strategies and100,000,000-byte header cap;
measured compatible headers88906/67237/67875/50269 fit, with no universal smaller
cap established. Pack-builder input remains a separate compatibility/ownership
task. Full tensor/config/quantization preflight and production readiness are not
claimed. Active teacher parity evidence is not fallback numerical validation.

## ADR-012 — Reject prefix promotion after sustained replay divergence

2026-11-18,S3-00. A context-bounded,in-place consumed-token candidate with default
off,strict complete-prefix checks,no recurrent rewind and validated flush/epoch
commit achieved natural real tool-result hits. Five short-cycle raw-ID pairs
matched,but first512-token continuation differed from same-Swift full replay at
zero-based output465(1031 versus285). This fails the principal gate despite
measured latency reduction. Stop at first mismatch;do not alter expected outputs,
reference fixtures,kernels,precision or expert-cache policy to rescue the result.

One bounded off/full-replay diagnostic clears expert weights,repeats the fixed
warmup,and replays the same430 prompt IDs. It produces the cached512-ID digest,
not the prior warm-expert replay digest. Existing residency-history sensitivity
is reproduced without conversation reuse;hit/miss subset reduction ordering is
a code-level hypothesis,not proof of the first differing operation. No broad
numerical investigation is authorized by this finding.

Archive the exact candidate and failing tests;restore production Sources/Package/
pins to83d28e2. Keep independent consumed-boundary/template tests and all evidence.
No cache flag/UI control ships;S3-00 is blocked,not completed. A future attempt
requires separately scoped residency-independent reduction/reference validation
or independently reviewed acceptance policy. The original MLX131/132 release
blocker remains unchanged;this diagnostic neither resolves nor explains it.

## ADR-013 — Characterize residency-dependent MoE reduction

S3-00a isolated the different FP32 hit/miss grouping with matched weights,
routing, inputs, and per-expert values. The experimental router-order combine
reproduces hidden values and logits across the tested cache histories. It
also changes logits relative to production. The independent strict MLX test
returns the existing 131/132 result. This qualifies neither different scores
on the reference workload nor conversation-state reuse.

Archive the diagnostic, measurements, and failing strict output. Keep the
production execution path at the validated base. Review the deterministic gate
in [the numerical contract](NUMERICAL_CONTRACT.md) before considering a
production reduction change. Use a matched, checkpoint-identified reference
sample to review score and margin differences beyond token counts. Resume S3-00
only after a candidate passes those gates, then independently compare reuse
against that candidate's full replay.

## ADR-014 — Revise the router-order expert combine

S3-00b compared both Swift histories and the experimental router-order
path against native prefill and sequential MLX on two pinned teacher
histories. Router order is consistent across the requested residency
changes and approximately preserves aggregate MLX fidelity. It retains the
wrong ranking at the sustained divergence compared with both MLX schedules
and cold legacy. The strict 131/132 result is unchanged. The throughput
and 64 KiB trade-offs from S3-00a do not decide this result.

Choose Outcome B. Keep production unchanged and archive the diagnostic patch.
The next implementation step is one combine-only variant that preserves
router-order accumulation and evaluates `residual + (routed + shared)`.
Reuse the current matched comparisons. Attempt S3-00 again only after a
reduction passes that review. See the
[qualification report](evidence/2026-11-18-s300b/README.md).
