# QwenFieldfare: controlled memory, faster requests, and local agents

Analysis date: 2026-11-05. Source baseline: `5287dc4`. This document is a
proposal, not an implementation or a new benchmark. No model was loaded for
this review. August measurements below are historical, not current machine
performance. Revalidate client documentation against pinned versions when
implementation begins.

## Goal and constraints

Run the compatible Qwen 3.6 35B-A3B MLX Q4 checkpoint with an explicit memory
budget, improve useful request latency, and expose it to local coding agents.
Keep one model owner, exact top-8 routing, and the existing checkpoint format.
Default to the approximately 5 GiB active-streaming envelope. Larger resident
caches must be an explicit user choice, never an automatic response to slowness.

An inference endpoint can enable an agent client to use tools; it does not
itself need to execute shell commands or own the agent's filesystem permissions.

## Verified current implementation

| Area | Present | Gap relevant to this goal |
| --- | --- | --- |
| Products | Native Mac app, Core library, benchmark, optional pack builder | No HTTP server target or general chat CLI in `Package.swift` |
| Active streaming | Resident dense Q4 weights, 40 layer partitions, segmented LRU, top-8 slot locks | UI capacity is a context-based preset, not a selectable byte budget |
| Expert reads | Parallel independent misses; gate/up/down reads also run in parallel | Nine raw ranges per missed expert; reduced syscall count alone previously lost |
| CPU/GPU overlap | Shared and hit experts run while misses are fetched | Router completion remains a true CPU dependency at each layer |
| State | 30 DeltaNet/Conv recurrent layers and 10 full-attention FP16 KV layers | Every `generateResult` calls `resetState`; no conversation prefix reuse |
| Prefill | Calls `forwardToken` separately for every known prompt token | No batched prompt path; final RMSNorm, LM head and argmax run even for unused intermediate predictions |
| Sampling | Temperature, Top-K/Top-P, presence/repetition penalties, optional fixed seed in Core | General selector scans 248,320 logits with membership checks; app uses an unseeded sampler |
| Chat format | System/user/assistant text, thinking switch | No typed tools, tool results, assistant tool-call history, or incremental tool parser |
| Context | UI offers 4K–64K active streaming | A 64K allocation/load measurement is not validation of 64K tool use or long-context quality |
| Lifecycle | Lazy load, stop, idle unload, telemetry | Mutable runners are not a safe concurrent-request interface; allocation/encoder failures include force unwraps and `fatalError` |
| Model checks | Metadata/index and shard presence, basic shard-name checks | Full tensor/dtype/shape/quantization/offset/symlink validation is incomplete |
| Validation | Model-free CI, benchmark harness, MLX reference script | Historical 132-position parity harness is absent; sampled/long-agent evaluation is absent |

Primary local evidence:

- [Runner](../Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift):
  `generateResult`, `forwardToken`, `initStates`, `fetchExpertWithPread`.
- [Cache](../Sources/QwenFieldfareCore/Storage/Q4BoundedExpertCache.swift):
  `acquireSlots`, `acquireSlotsParallel`, layer partitions and fixed arena.
- [Generation support](../Sources/QwenFieldfareCore/Runtime/Generation/QwenGenerationSupport.swift):
  `QwenChatTemplate`, `QwenLogitSelector`, generation stop/limit handling.
- [App orchestration](../Sources/QwenFieldfareApp/Mac/Generation/ChatService.swift):
  `expertCacheSlots`, memory estimate, `send`, load/unload and sampling defaults.

## What the algorithm implies

MoE reduces work per token, but the selected expert set changes. Loading only
eight experts per layer does not imply the next token uses the same eight.
The cache retains a larger working set to reduce rereads. The existing SLRU
and overlap are useful foundations; there is no evidence for replacing them
with another simple eviction policy.

At 8K, historical allocation was 4.928 GiB Metal, of which 3.597 GiB was the
2,183-slot expert cache. Every slot is 1,769,472 bytes. Longer context spends
more memory on full-attention KV, leaving fewer slots under a fixed budget.
Token generation therefore has two coupled costs: GPU execution and blocking
expert reads. Agent turns add a third: replaying a large prompt.

Metal allocation, process physical footprint, mapped file residency, and
system file cache are different measurements. A Metal budget cannot promise
an exact cap on total system RAM or prevent all OS cache residency. Report
Metal peaks, process footprint, normal memory pressure and swap together.

## Implementation order

Each numbered stage is a separate reviewable change. Performance work uses
the existing [promotion protocol](OPTIMIZATION_PLAN.md#common-promotion-protocol).
No speedup percentage is promised before an A/B measurement.

### 0. Restore the measurement and correctness foundation

**Status: in progress.** The following checklist is the stage-0 work queue;
do not start the next stage until the applicable checks are complete.

- [x] Inspect the current reference, benchmark, runtime and model prerequisites.
- [x] Build the current release and perform an initial real-model greedy
  comparison against the installed MLX reference.
- [x] Define a reproducible checkpoint/reference identity and fixture format.
- [x] Implement and test the release-mode teacher-forced parity harness.
- [ ] Measure 128/512-token decode and fixed-seed app-equivalent sampling.
- [ ] Measure 256/1K/4K-token prefill and repeated-turn controls.
- [ ] Record 8K/32K/64K allocation, physical footprint, pressure and swap.
- [ ] Address and test model-validation, read, Metal and cancellation failures.
- [ ] Complete the package suite and document all results before promotion.

**Parity implementation in progress:** a release benchmark mode now compares
each next-token prediction while feeding the MLX reference token at the next
position, and rejects prompt-token mismatches. On 2026-11-05 it matched 132/132
positions with the 21-token HashMap prompt, 8K context, 2,183 slots, and the
MLX versions recorded below. A full package test run passed. The fixture is
currently temporary and does not yet record a verified checkpoint identity or
all dependency versions. A deliberately wrong target and mismatched prompt
both returned nonzero, and the full suite passed. Fixture provenance, automated
negative tests, and validation after the last token-bound change remain required;
this is not yet a completed parity gate.

**2026-11-05 follow-up:** fixture format 2 hashes every indexed shard and the
required model/tokenizer metadata and records MLX, mlx-lm and Transformers
versions. The release harness verified the hashes and matched 132/132 again.
Full checkpoint hashing warms the OS file cache; parity runs must not be used
as throughput baselines. `Scripts/test_teacher_reference.py` passed six
sequential real-model cases: valid, wrong target, wrong prompt, wrong hash,
invalid token and missing provenance. The fixture records exact package
versions but does not independently pin or verify the MLX package source.

**Validation hardening in progress:** the model-directory validator now rejects
shard symlinks resolving outside the selected directory and non-regular shards.
The Safetensors loader caps the header and checks tensor offsets against file
size. Four focused tests passed, including oversized headers, out-of-bounds
offsets and external symlinks. This is partial model validation: shape, dtype,
quantization, all read failures, Metal command errors, and cancellation still
need separate work before the reliability checklist can be marked complete.
The active runner's recurrent/KV state allocations now throw on allocation
failure rather than force-unwrapping. The full package tests passed after this
change; allocation-failure injection, release build and real-model parity
after this final change have not yet been run. Other forced allocations and
command-encoder fatal errors remain.

**Context allocation checks:** 32K/1,883 slots with a 22-token prompt and 16
generated tokens reported 4.904 GiB Metal and 4.780 GiB final footprint.
64K/1,523 slots with the same workload reported 4.937 GiB Metal and 4.796 GiB
final footprint. These are short runs, not filled-context tests. System memory
free was 88% before and after; system-wide swap after the runs was 2,279.44 MiB.
Swap was not recorded before these runs, so no change or attribution is claimed.

The benchmark now offers a fixed-seed app-equivalent sampler. A preliminary
8K/2,183-slot run using the 21-token HashMap prompt, one warmup and two
128-token measured runs produced 31.97 and 31.95 decode tok/s, with mean CPU
selection 2.933 and 2.964 ms/token, and 4.928 GiB Metal. Both generated the
same token sequence. This is a new baseline, not a before/after optimization;
the real app uses an unseeded sampler. The initial output incorrectly called
the sampled determinism result 'greedy'; its label was subsequently corrected.

**512-token greedy control, 2026-11-05:** same 21-token prompt, 8K context,
2,183 slots, seed 42, one retained-runner warmup and two measured runs. Decode
was 33.58 and 33.36 tok/s; TTFT 0.666 and 0.659 s; decode misses 38,017 and
37,994; logical bytes fetched 62.650 and 62.612 GiB; final Metal allocation
4.928 GiB and physical footprint 4.801 GiB in both runs. Outputs were identical.
The OS file-cache state was uncontrolled and checkpoint hashing had occurred
earlier; these are controls, not a gain over the historical measurements.

**128-token greedy control, 2026-11-05:** same setup and one warmup, followed
by two runs: 34.33 and 34.64 tok/s, TTFT 0.647 and 0.670 s, decode misses
7,595 and 7,577, 12.516 and 12.487 GiB logical decode fetch, 4.928 GiB Metal,
and 4.793 GiB final footprint. Outputs matched. Page-cache state uncontrolled.

**Initial check, 2026-11-05:** on macOS 27.2 with Swift 6.4.2,
MLX 0.32.0, mlx-lm 0.31.3 and Transformers 5.15.0, the release build passed.
The local Q4 model and normal memory pressure were present; no conflicting
model process was found. With `Explain HashMap in Java with an example.`,
both implementations produced the same 65 greedy tokens from a 21-token
prompt. QwenFieldfare used 8K context, 2,183 slots, zero warmups and one run;
it reported 26.19 tok/s over 64 transitions, 1.496 s TTFT, 3,852 decode misses,
6.348 GiB logical decode fetch, 4.928 GiB Metal and 4.790 GiB final footprint.
The MLX script reported 19.629 GB peak and 13.27 tok/s over its entire
generation including prefill; that figure is not comparable to the
QwenFieldfare decode-only rate. This is a single free-running check, not
teacher-forced parity or a performance baseline. Checkpoint content/revision
and historical 132-position fixture have not yet been frozen or reproduced.

Restore a callable release-mode teacher-forced parity harness. Freeze the
checkpoint revision and reference implementation; distinguish next-token parity
from free-running sequence identity. Add representative system/user/tool-turn
fixtures as tool support is introduced. Extend benchmarking to fixed-seed
app-equivalent sampling and multi-turn TTFT, beyond exact greedy short prompts.

Record a new control on the actual Mac: 128/512-token decode, 256/1K/4K-token
prefill, repeated turns, and memory at 8K/32K/64K selections. Start large-prompt
tests incrementally. Do not infer filled-context behavior from a load-only test.
Capture non-overlapping timing; `lmHeadMs = 3.0` and the 70/30 attention/router
split in `forwardToken` are estimates, not separately measured kernel durations.

For a server milestone, make model-contract errors, failed reads, Metal
allocation/command failures and cancellation recoverable. Include checked offset
arithmetic, bounded header sizes, and resolved shard confinement. Preserve
unrelated checkpoint tensors such as unused vision weights.

Deliverable: a reproducible baseline and failures that return to a known state.
This stage enables later claims; it is not itself a throughput optimization.

### 1. Expose explicit memory budgets

Introduce one shared runtime configuration for app, benchmark and future server.
Offer 3/4/5 GiB active-streaming targets where the selected context fits; retain
5 GiB as the current default. Derive slots from fixed/dense/recurrent storage,
context KV, scratch, safety headroom and slot bytes. Use checked integer byte
arithmetic and a minimum of eight locked slots per layer (320 total).

The existing historical 8K estimates are about 969, 1,576 and 2,183 slots for
3/4/5 GiB. Recompute for every context; reject impossible combinations. Do not
reuse the current estimator's positive-only delta above 1,024 slots for smaller
caches. Measure 128/512-token speed and bytes for each promoted profile.

Apply profile changes through a controlled unload/reload initially. The cache
is one fixed Metal buffer: do not claim in-place resize or silently allocate a
second full arena. Later pressure handling can cancel/drain and reload a smaller
profile, with explicit state invalidation and user-visible consequences.

Expected outcome: useful control over the memory/speed tradeoff. Fewer slots
are expected to cause more I/O; actual throughput must be measured. An optional
larger budget may improve hit rate but is outside the default 5 GiB goal.

**Status: complete on 2026-11-05.** A single checked
`QwenActiveStreamingMemoryConfiguration` now owns the model-specific plan for
the Mac app and benchmark. It includes 40 dense arenas, 30 recurrent states,
decoded constants, fixed and context-sized scratch, 10-layer FP16 KV state, a
conservative mapped-model/allocator reserve, and one Q4 expert-cache slot per
remaining 1,769,472 bytes. Arithmetic is checked and every valid profile keeps
at least 320 slots (eight for each of 40 routers). `--cache-slots` remains a
diagnostic override; it cannot be combined with `--memory-gib`.

The app exposes a 3/4/5 GiB picker in the Active Expert Streaming inspector.
Changing it unloads the active runner; no second arena is allocated. The default
remains 5 GiB. Unsupported pairs are removed from the context picker: 3 GiB at
64K would leave only 303 slots, below the required 320.

| 8K profile | Slots | Planned Metal | Observed Metal | 128-token decode mean | 512-token decode mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 GiB | 969 | 2.999 GiB | 2.927 GiB | 31.76 tok/s | 30.28 tok/s |
| 4 GiB | 1,576 | 4.000 GiB | 3.928 GiB | 34.07 tok/s | 32.53 tok/s |
| 5 GiB (default) | 2,183 | 5.000 GiB | 4.928 GiB | 35.83 tok/s | 34.15 tok/s |

These are release-mode greedy controls on the local Apple M5 / 32 GiB machine,
macOS 27.2 and Swift 6.4.2, using the same 21-token HashMap prompt, seed 42,
one retained-runner warmup and two measured runs per cell. Raw Safetensors,
8,192 context and GPU profiling disabled were used. The 512-token decode cache
metrics were 57.5% / 135.9 misses per token / 114.4 GiB fetched at 3 GiB,
68.9% / 99.6 / 83.9 GiB at 4 GiB, and 76.8% / 74.4 / 62.6 GiB at 5 GiB.
Outputs were identical between measured repetitions within each profile.
File-cache state was uncontrolled, so this matrix establishes the supported
trade-off; it is not a before/after speedup claim. Memory pressure was 89% free
before and after the set. Post-run system swap was 2,143.44 MiB; no pre-run
swap baseline was captured, so no swap attribution is made.

### 2. Remove unused output work during prompt prefill

Introduce an explicit state-advance operation that does not compute logits.
Use it for every prompt token except the last; preserve all transformer,
router, expert, Conv, DeltaNet and KV updates. Preserve completion/error ordering
before shared scratch reuse. Do not overload the existing nil configuration:
nil currently means unconfigured argmax, not skip-output.

This removes provably unused final normalization, 248,320-row LM-head projection
and token selection from intermediate prompt positions. It changes neither
decode mathematics nor expert selection. Measure the actual removable GPU
cost first; the hardcoded 3 ms telemetry is not evidence of savings.

Gate: identical final-prompt logits/state within the established reference
contract, identical greedy IDs and fixed-seed output, and at least 20% lower
prefill/TTFT on declared long-prompt workloads. Reject a headline claim if the
measured gain misses that threshold. Decode tok/s should remain unchanged.

### 3. Reuse one verified conversation prefix

Retain exact token prefix, logical positions, Conv/DeltaNet state, full-attention
KV, and the pending last generated token. On each request, compare the newly
formatted/tokenized prompt with the actual retained sequence. Process only an
exact append; otherwise reset and replay. Template cleanup, tokenizer boundary
effects, changed tools/system text and thinking history can break that prefix.

For v1, use one retained session and discard/replay on cancellation, divergence
or error. Avoid snapshots initially: recurrent state alone is roughly 60 MiB,
which matters near the 5 GiB limit. KV truncation cannot roll back DeltaNet.
Clients may send full histories while the server internally verifies reuse.

Gate: cached and full-replay outputs/state agree over multi-turn fixtures,
invalidation is deterministic, and repeated-turn prefill falls at least 20%.
Report reused tokens, newly processed tokens, TTFT and total turn time.

### 4. Implement the Qwen tool contract

Use the exact compatible checkpoint template as the contract, including system
instructions, tool schemas, assistant calls, tool results and reasoning handling.
The upstream Qwen template uses structured `<tool_call>` / `<function=...>` /
`<parameter=...>` output; a generic JSON-regex parser is insufficient.

Build a bounded incremental parser that tolerates arbitrary stream boundaries,
preserves argument strings/Unicode, supports multiple calls and reconstructs
call IDs and tool-result history correctly. Do not dispatch tools from partial
or malformed output. Implement or explicitly reject each requested tool-choice
mode; a prompt instruction alone is not guaranteed schema-constrained decoding.

Gate: template/token fixtures, stream fragmentation tests, argument validation,
tool-name checks, stop-reason mapping, malformed-output handling, and real-model
read-file/search/edit/test loops in a disposable test project. Track task success,
valid-call rate and retries separately from speed. The client executes tools.

### 5. Add a single-owner local inference server; validate OpenCode first

Proposed product: `QwenFieldfareServer` (not currently available). Scope it to
this Qwen checkpoint and a documented API subset. Use one serialized runtime
owner, one retained prefix and a bounded queue. Stream SSE with backpressure;
disconnect cancels the active request at supported boundaries. Slow clients
must not accumulate unbounded token/event buffers. Reject unsupported fields
and context overflow explicitly, reporting actual token counts.

Initial proposed endpoints: readiness/health, `GET /v1/models`, and
`POST /v1/chat/completions`, including non-streaming and streaming tool calls,
usage and finish reasons. Add a documented local token-count endpoint if needed;
it is not a universal OpenAI standard. No embeddings, audio, vision, batching,
Responses API or arbitrary model selection in this milestone.

The server owns the model; the app and benchmark must not independently load
another copy at the same time. Initially document mutually exclusive launch
modes and enforce a local ownership lock. Making the app a server client can be
a later change. HTTP event handling must stay responsive during Metal and I/O.
Use loopback binding and a local credential; remote exposure needs a separate
authentication/TLS design. Prompt/tool contents stay out of default logs.

OpenCode is first because it documents a custom OpenAI-compatible provider.
Pin its version: the current classic and v2 configuration schemas differ.
Verify real request payloads and publish only tested configuration examples.
Acceptance requires successful read/search/edit/test/retry tool loops,
cancellation, queue rejection and context overflow, not just a chat response.

### 6. Qualify Hermes, then evaluate Claude Code separately

Hermes documents custom OpenAI-compatible endpoints and currently requires at
least 64,000 context tokens for tool-enabled agents. Our 65,536 selection fits
that advertised minimum, but actual long-history execution is unqualified.
Measure prefill, sustained attention cost, task quality and total memory at
real occupied lengths. Advertise the server's configured effective context,
not the checkpoint's larger theoretical maximum. Ensure auxiliary summarization
or title requests are queued; configure them to the intended local provider
explicitly if fully local operation is required. Interleaving sessions can
invalidate the single retained prefix and must remain correct via replay.

Claude Code uses Anthropic Messages, not merely Chat Completions. Its current
official documentation explicitly does not support routing to non-Claude
models. Treat Qwen support as experimental and version-specific. It would need
a separately tested `/v1/messages` adapter, Anthropic SSE/content blocks,
`tool_use`/`tool_result`, usage/stop mappings and relevant token-count behavior.
Do not advertise official compatibility or silently pretend to serve Claude.
Only undertake this after OpenCode/Hermes pass useful agent tasks.

### 7. Investigate actual sampled-decode cost

The promoted S1 speedup applies to exact greedy. The UI default (temperature 1,
Top-K 20, presence penalty 1.5) still uses the general selector. Measure it
before rewriting it. If material, test indexed seen-token tracking, sparse
penalty application and reusable Top-K/weight scratch. Preserve tie rules,
floating-point behavior and fixed-seed draw order.

Gate: at least 5% paired end-to-end sampled-decode gain and positive confidence
interval, fixed-seed output equality and unchanged memory profile. Do not extend
the greedy claim to sampling or infer this cost from August greedy profiles.

### 8. Build bounded batched prefill for genuinely long prompts

After eliminating unused heads and enabling prefix reuse, optimize remaining
first-request/suffix prefill. Process bounded chunks layer-by-layer so dense
matrix operations and repeated expert selections can reuse weights across rows.
Start with an exact sequential recurrent update inside the chunk, batched
projections, and correctly masked full attention; optimize DeltaNet scans only
after its state contract is verified. Tile routed expert work under the cache
capacity rather than pinning the entire chunk's expert union.

Chunk scratch, cache locks, recurrent ordering, long-dispatch responsiveness
and cancellation require a Qwen-specific design. The upstream Gemma prefill
implementation is a reference, not a drop-in kernel. Test chunk sizes such as
16/32/64 before selecting one. Gate: at least 20% TTFT reduction, reference
quality/state checks and measured peak memory within the selected budget.

## Later research, gated by measurements

- **Lazy FP16 KV:** useful for short conversations with 32K/64K selected.
  Full-window memory is unchanged. Replacing buffers can temporarily double
  storage; prove transition peaks or use chunks/headroom. Reset/attention bounds
  must track physical capacity independently of logical context.
- **Faster Q4 GPU path:** one representative high-cost projection or routed
  operation; investigate a materially different tiled/data-access algorithm.
  Include repack/staging cost and scratch. Require >=5% full-request decode gain
  after a compelling kernel measurement, not another launch-size sweep.
- **Long-context attention:** profile real occupied 8K/32K/64K lengths before
  pursuing split attention, online softmax or attention scratch reduction.
  INT8 KV changes numerics; use a long-context quality suite first.
- **Draft/MTP speculation:** research only. Requires a compatible artifact,
  acceptance evaluation, exact target sampling, recurrent rollback/verification,
  and all draft/snapshot memory inside the budget. Faster resident speculation
  elsewhere does not establish an active-streaming win here.
- **User features:** saved/exportable chats, accurate remaining-token display,
  context-aware truncation/compaction policy, resumable setup and sanitized
  diagnostics. Never discard system instructions or unmatched tool-result pairs.

## Closed directions and evidence discipline

August S3 replay found 43%–46% total-miss reduction with perfect future
knowledge, but tested causal replacements saved at most 1.63% on one prompt
and regressed another. Transition prefetch had only 7.56%–10.01% precision at
the smallest budget. Keep SLRU; revisit only with new causal evidence.
Fixed-table cache bookkeeping had about a 1.2% ideal wall-time ceiling. Packed
one-read storage lost to raw storage. Preserve these rejection records.

For every experiment publish control/candidate IDs, workload, actual prefix
reuse, sampling/seed, all paired runs, prompt/output lengths, TTFT, decode,
p50/p95 latency, expert bytes/misses, peak Metal/process memory, pressure/swap,
and correctness results. Changes that do not pass their declared gate are
removed; the evidence remains in docs. Protocol support has functional gates;
it does not need to pretend to improve decode tok/s.

## External sources checked on 2026-11-05

- [Qwen checkpoint tool template](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/chat_template.jinja).
- [OpenCode providers](https://opencode.ai/docs/providers) and
  [v2 providers](https://opencode.ai/v2/docs/providers): custom compatible endpoints;
  configuration schemas differ by generation.
- [Hermes providers](https://hermes-agent.nousresearch.com/docs/integrations/providers):
  custom endpoints, structured tool calls, and the current 64,000-token minimum.
- [Claude Code gateways](https://code.claude.com/docs/en/llm-gateway) and
  [protocol](https://code.claude.com/docs/en/llm-gateway-protocol): endpoint/streaming
  contract and explicit non-Claude support limitation.
- [TurboFieldfare system design](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md)
  and [server guide](https://github.com/drumih/turbo-fieldfare/blob/main/docs/OPENAI_SERVER.md):
  bounded prefill, one warm server owner, and verified conversational prefix.

## Scope decision

This proposal extends the prior server deferral only to a model-specific local
server and client-side tool execution. It proposes no generic model platform
or second model owner. Before server implementation, update the explicit scope
in `AGENTS.md` and the product roadmap with this agreed design. No core or
workflow change is authorized by this document alone.
