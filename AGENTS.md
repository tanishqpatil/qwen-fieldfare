# QwenFieldfare

## Current milestone — S3-01b bounded router readback

Use the existing isolated branch at `335e4d9` and preserve S3-00d
tokenwise source. Inspect dependencies and measure exposed router readback.
Decision D closes the milestone. The 1K timing probe gave insufficient
headroom for router-readback batching with ordered expert service.
Preserve the diagnostic archive and keep S3-00d exact-prefix semantics,
fixed reduction, SLRU and the strict fixture. Further runtime work requires
its own milestone.

## Historical milestone — S3-01a bounded chunked prefill

Work on `feature/bounded-chunked-prefill` from verified S3-00d tip `2e5b993`.
Qualify one bounded opt-in candidate. Preserve the tokenwise default and
exact-prefix protocol, fixed router reduction, checkpoint and strict fixture.
Decision C closes this milestone. Preserve the diagnostic patch and
measurements on this feature branch. Full tokenwise inference and exact-prefix
reuse use the S3-00d source. Further runtime development requires its own
milestone. Main and the server branch remain separate.

## Historical milestone — S3-00d exact-prefix state reuse

Qualify one in-place exact consumed-token prefix per active server runner
against fixed router-order full replay. The default is off. Native chat
and the server retain exclusive model ownership. The strict MLX 131/132
release blocker remains.

After reviewing the diff and validation evidence, commit validated
milestones and push to the development or feature branch. Automatic merges
to main remain prohibited. S3-00d stops at its qualification decision.
The earlier local-only instructions below describe their historical milestones.

## Historical milestone — S3-00c reduction decision

S3-00c selects the original fixed router-order expert reduction as the
production full-replay baseline. The grouped-tail experiment is closed.
The strict MLX 131/132 blocker remains. Stop here. The next S3-00 iteration should qualify
opt-in exact-prefix reuse against this baseline.

## Historical milestone — S3-00b deterministic expert reduction

S3-00b ended with Outcome B (revise). Router order eliminates the measured
residency variance and preserves similar aggregate MLX fidelity, but selects
1031 over 285 at the sustained boundary where both MLX schedules select 285.
The original strict 131/132 issue remains. Production matches c1ff7c9.
S3-00 prefix reuse stays blocked pending a qualified reduction.

## Historical milestone — S3-00a expert-cache residency

S3-00a characterized residency-dependent MoE accumulation. The experiment is
archived and production uses full replay with the original sum. S3-00 remains
blocked. Preserve the strict MLX fixture and the 131/132 release blocker.
The next gate is review of matched logits and hidden values before promoting
a router-order path. Stop at the S3-00a report.

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


Native Swift and Metal inference for the MLX Q4 build of Qwen 3.6 35B-A3B on
Apple Silicon.

## Scope

This repository is Qwen 3.6-only. The 2026-11-17 implementation mission
explicitly authorizes a narrowly scoped local Qwen Chat Completions server,
client-side tool calling and exact hybrid-state reuse in gated later stages.
Do not add unrelated model formats, generic multi-model hosting, server-side
tool execution or a second model-owning process. Preserve bounded expert streaming.

## Session recovery and stage gates

Read this file, `TODO.md` and `docs/AGENT_MEMORY.md` at session start. The TODO
owns stable mission task IDs/status; architecture and decisions distinguish
implemented features from proposals. Read relevant source before editing;
update recovery memory, task status and append-only DEVELOPMENT_LOG as work
progresses. Source/tests/verified evidence outrank narrative memory.

Follow the mission stages sequentially, one small testable task per iteration.
The 2026-11-17 revised policy permits independent Stage 1 reliability work while
131/132 teacher forcing remains an explicitly unresolved baseline. Strict 132/132
is a release-quality blocker, not a global development blocker. Preserve its
fixture/evidence/tests; never suppress failures or claim full parity. New or
worsened mismatches block affected changes. Production-readiness claims require
an independently reviewed numerical policy and sufficient evidence.

Historical Stage 1 and S2-02 boundaries are superseded by the milestone above.
No further broad/layer-3 numerical investigation without a bounded hypothesis
and decision it informs, or a change directly requiring it. Model-free tests
first; model-backed checks only for inference-affecting work/integration gates.
Reuse verified evidence; never skip required integrity checks, but do not hash
all weights repeatedly without an identity/integrity reason. Preserve unrelated
work and historical rejections. Follow the current feature-branch policy above. Project memory
must contain no secrets, private prompts or sensitive runtime payloads.

## Products

```bash
swift build -c release
.build/release/QwenFieldfareMac

swift run -c release QwenFieldfareBench \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --prompt "Explain HashMap in Java with an example." \
  --max-new 128

swift run -c release QwenFieldfarePack \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit
```

`QwenFieldfareCore` owns model validation, Swift/Metal inference, telemetry,
the bounded expert cache, and the optional packed expert store.

## Model safety

The app reads a compatible local MLX Safetensors directory selected in the UI
or through `QWENFIELDFARE_MODEL`. Never stage, duplicate, upload, or commit
model weights. Treat tensor names, shapes, quantization, and chat formatting as
model-specific invariants.

Before a real-model run, require macOS 26+, Swift 6.2+, adequate disk and
normal memory pressure, and no process matching
`QwenFieldfareMac|QwenFieldfareBench|QwenFieldfareServer|mlx_lm|mlx-lm|ollama|LM Studio`.
Run only one model-owning process at a time.

## Validation

Run package tests through `Scripts/test.sh`. Build release once before a model
measurement and follow `docs/COMMUNITY_BENCHMARKS.md` exactly. Profiling flags
are diagnostic and must not be used for headline throughput.

Report the commit, hardware and RAM, macOS, Swift version, exact command,
prompt/generated token counts, complete timing/cache footer, memory, and every
protocol deviation. Measurements are not performance ceilings.
