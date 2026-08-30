# Optimization analysis and priority order

Status: roadmap reset on 2026-08-28 after reviewing the current source,
corrected benchmarks, memory ownership, and rejected experiments. S1a/S1b was
promoted and pushed as `4758b8b` on 2026-08-30. S2's measurement gate completed
the same day and closed the fixed-table bookkeeping direction without a
runtime change. S3's four-prompt offline trace gate also completed on
2026-08-30; it found a large non-causal miss ceiling but rejected the tested
causal replacement and prefetch directions.

## Decision

The exact-greedy portion of **S1: token-selection hot paths** passed its
promotion gate. S2 then measured cache planning at only 0.364 ms/token, too
small for a fixed-table rewrite to reach the 5% promotion threshold. S3 found
that perfect future knowledge could remove 43.42%–46.43% of total misses, but
the tested causal policies could not approach that bound: SRRIP changed misses
by -1.63% to +0.22%, and the smallest online transition-prefetch budget wasted
18,393–18,765 reads per trace. No S3 runtime candidate or throughput A/B was
justified.

Before S1b, the benchmark's non-nil `.greedy(...)` configuration routed the
248,320-logit result through the general `QwenLogitSelector` in both runners.
The general selector scanned the complete vocabulary, performed a hash-set
lookup for every logit, and maintained temporary candidate storage; sampled
generation also created weight storage. For the benchmark's exact greedy
configuration, the penalty checks and candidate allocation were unnecessary.

This was a stronger candidate than the previous kernel work because the
semantic fast-path condition is provable. The retained implementation is
deliberately narrower: temperature zero, Top-K one, presence penalty zero, and
repetition penalty one. Top-P and seed cannot change that selection. The app's
default temperature-based sampler does not enter this path and has no new
performance claim.

## Reality check

| Evidence | Verified result | Roadmap consequence |
| --- | ---: | --- |
| Corrected retained-runner 128-token baseline | 20.62 tok/s mean in the original corrected run | Keep as the comparison record for commit `2f49ba2` |
| Later same-session short control | 31.00 tok/s mean | Host/GPU/page-cache state can move short-run results substantially |
| Promoted S1 exact-greedy A/B | 29.02 to 31.97 tok/s; +10.16% paired mean, 95% CI +8.05% to +12.26% | Material short-run gain with identical token IDs and cache demand |
| Promoted S1 512-token confirmation | Three paired gains: +6.22%, +8.85%, +5.01% | Gain remains positive under sustained expert demand |
| S2 cache-planning split | 0.364 ms in a 29.84 ms token step | Complete removal could save only about 1.2%; do not implement the fixed-table rewrite |
| S3 lock-aware offline optimum | 43.42%–46.43% fewer total misses; 46.20%–50.98% fewer decode misses | Future-aware replacement has a large theoretical ceiling, but it is not implementable online |
| S3 causal replacement | SRRIP changed misses by -1.63% to +0.22%; BRRIP was worse on all four traces | No replacement-policy runtime edit or tok/s A/B |
| S3 causal prefetch | Best one-read/layer precision 7.56%–10.01%, with 18,393–18,765 wasted reads | Reject the tested transition predictor; it increases expert traffic even before cache pollution |
| Corrected 512-token baseline | 19.68 tok/s; 50.80 ms/token | Sustained decode and late-run drift are the primary speed metric |
| Recent exact GPU work | 20.25–21.02 ms/token | About 47–49 tok/s is the GPU-only ceiling before CPU work and expert I/O |
| Full-resident MLX reference | 49.86 tok/s; 19.631 GB peak | A reference, not an accepted 5 GiB target |
| Active 8K allocation | 4.928 GiB | The current speed-oriented profile has little headroom under 5 GiB |
| Expert cache share at 8K | 3.597 GiB, about 73% of Metal allocation | Large steady memory reductions require fewer cache slots and more I/O |
| Late-run expert demand | 53.4 to 101.8 misses/token | Miss demand and bytes rise materially and contribute to variance; their share of aggregate slowdown is not isolated |
| Projection Candidates A–F | No repeatable end-to-end gain | Stop projection geometry and instruction-form search |
| MoE subset row packing | 1.13% slower | Idle SIMD groups were not the useful-work bottleneck |
| Packed one-read experts | 24.38 vs 25.04 tok/s raw | Fewer reads without fewer bytes is not a speed strategy |

These are workload-specific observations, not hardware ceilings. The evidence
does **not** support promising 50 tok/s while retaining the approximately 5 GiB
active-streaming envelope. The near-term goal is a repeatable improvement over
the corrected control, with separate reporting for greedy, app-default sampled,
short, and 512-token sustained workloads.

## How priorities are scored

Every workstream is ranked on four independent axes:

- **evidence**: a current measured cost or a provable unnecessary operation;
- **impact ceiling**: enough affected wall time to clear run variance;
- **feasibility**: a bounded reversible change with known correctness checks;
- **product relevance**: whether it improves greedy diagnostics, default chat,
  sustained decode, memory, or time to first token.

A speed candidate is a promotion candidate only if the common alternating A/B
protocol below shows at least a 5% end-to-end point improvement and a positive
95% confidence interval for the paired change. A memory milestone must save at
least 0.25 GiB at the claimed operating point; smaller exact savings are
cleanup, not headline optimization. A conversation-latency feature must reduce
the measured repeated-turn prefill by at least 20%. These are predeclared
decision thresholds, not predicted results.

## Common promotion protocol

Every runtime experiment uses this gate unless a stricter one is documented:

1. freeze control and candidate release executables and record their commit and
   executable SHA-256;
2. use the same model, prompt, tokens, sampler, context, cache capacity,
   retained-runner warmup, initial cache state, power mode, and process
   exclusions;
3. run at least five alternating A/B pairs, reversing initial order across
   pairs, and report every run;
4. use paired percent change as the primary statistic; require a point estimate
   of at least 5% and a positive 95% confidence interval;
5. keep GPU/subcomponent profiling out of headline runs and capture matched
   diagnostic runs separately;
6. confirm the result with at least two alternating 512-token A/B pairs,
   including per-bucket latency, misses, bytes, Metal allocation, physical
   footprint, memory pressure, and before/after system-wide swap;
7. run the candidate-specific correctness tests, cancellation/unload checks,
   release build, complete package tests, and `git diff --check`.

## Speed and latency roadmap

### S1 — token-selection hot paths

**Status: S1a greedy measurement and S1b exact-greedy fast path promoted
locally on 2026-08-30. S1c production-sampler work is not started.**

Stage S1a added measurement-only reporting for the existing `cpuArgmaxMs`.
Exact greedy was measured over 128- and 512-token runs. App-default sampling
was not modified or benchmarked, so S1c remains a separate future experiment.
Any S1c work must use an explicit fixed validation seed and report that the
real UI default remains unseeded. Greedy and sampled rates must never be
combined in one headline.

Stage S1b restored one shared exact greedy fast path for both runners when:

```text
temperature == 0
topK == 1
presencePenalty == 0
repetitionPenalty == 1
```

It uses Accelerate's maximum-index operation and preserves the same token IDs.
The current paired measurement reduced mean token selection from 3.171 ms to
0.028 ms; older scalar measurements are not used as the promoted result.

Stage S1c targets the production sampler only if S1a establishes a material
cost. Candidate design should remove 248,320 hash-set lookups, reuse fixed
Top-K/weight scratch, and apply penalties through an indexed seen-token
representation while preserving exact seeded output. Start with a CPU design;
do not mix it with a GPU Top-K rewrite.

Promotion gates:

1. require exact greedy IDs to match the frozen control; separately restore the
   teacher-forced harness and require all 132 recorded positions to match MLX
   before release qualification;
2. test equal maxima, all-negative logits, infinities, and defined NaN behavior;
   verify that the Accelerate path preserves the current lowest-token-ID tie
   rule;
3. a fixed-seed configuration using app-default sampling parameters produces
   identical output before and after sampler changes;
4. selector microbenchmarks and alternating full-generation A/B both improve;
5. the greedy result is labelled as greedy-only unless the sampled path also
   passes its independent gate;
6. memory, cancellation, unload, and all package tests remain unchanged.

#### S1a/S1b result — promoted locally

The measurement-only harness now reports CPU token-selection latency without
enabling GPU profiling. The frozen instrumented control executable SHA-256 was
`5b34d0bb1b773dce55a2a690b98cde2e6633de518c6ac2af59c0e3cf9318f432`.
The candidate executable SHA-256 was
`28cf14350f0300c5eaf08a63754d3118eb4d21795e7d0167e77c6c905adda257`.

The runtime change is one shared exact-greedy selector path. It uses
Accelerate's maximum-index operation for the guarded configuration and retains
the general selector for penalties, Top-K greater than one, and sampling.

Five alternating 128-token pairs used the same 22-token prompt, 8K context,
2,183 slots, raw Safetensors, one retained-runner warmup, and one measured run:

| Pair | Order | Control | Candidate | Paired change |
| ---: | --- | ---: | ---: | ---: |
| 1 | control, candidate | 29.01 tok/s | 31.85 tok/s | +9.79% |
| 2 | candidate, control | 29.00 tok/s | 31.94 tok/s | +10.14% |
| 3 | control, candidate | 29.09 tok/s | 31.95 tok/s | +9.83% |
| 4 | candidate, control | 29.01 tok/s | 32.74 tok/s | +12.86% |
| 5 | control, candidate | 29.00 tok/s | 31.37 tok/s | +8.17% |

Control mean was 29.02 tok/s and candidate mean was 31.97 tok/s. The paired
mean improvement was 10.16%, with a 95% confidence interval of +8.05% to
+12.26%. Mean CPU token selection fell from 3.171 ms to 0.028 ms. All five
pairs had identical cache hits, misses, fetched bytes, and 4.928 GiB Metal
allocation.

Three alternating 512-token pairs confirmed sustained gains:

| Pair | Order | Control | Candidate | Paired change |
| ---: | --- | ---: | ---: | ---: |
| 1 | control, candidate | 24.75 tok/s | 26.29 tok/s | +6.22% |
| 2 | candidate, control | 24.19 tok/s | 26.33 tok/s | +8.85% |
| 3 | control, candidate | 24.97 tok/s | 26.22 tok/s | +5.01% |

Every 512-token run recorded 40,551 decode misses, 66.826 GiB fetched, and
4.928 GiB Metal allocation. A final candidate memory run held system swap at
0.25 MiB before and after, ended at 4.801 GiB physical footprint, and kept
system memory pressure normal. A separate 128-token output comparison matched
all 128 token IDs and the finish reason against the frozen control. Focused
tests cover equal maxima, all-negative logits, infinities, NaNs, penalties,
and seeded sampling. The complete package suite passed 43/43 tests in 10
suites, and the final release build passed.

The automated 132-position MLX teacher-forced harness is absent from the
current checkout, so this result is not presented as a new MLX parity run.
Restoring that release gate remains P7 in the product roadmap. The retained
performance claim is exact-greedy only; app-default sampled generation is
unchanged.

### S2 — fixed-table expert-cache bookkeeping (closed by measurement)

**Status: measurement gate completed on 2026-08-30. No runtime candidate was
implemented or retained.**

The cache hot path uses dictionaries, multiple `Set` values, temporary arrays,
`map`/`compactMap`, and repeated scans of each layer's slot range. Historical
telemetry attributed about 2.68 ms/token to “cache lookup,” but that metric
combined serial placement planning, hit-command encoding, and physical miss
reads. It overstated the portion a fixed-table rewrite could affect.

A measurement-only build split those phases at their real boundaries. The
frozen control executable SHA-256 was
`26a3969e2c4e7ce83cf2ff583664fd5ce1281b68b1e0df2a71a2932f6c51227e`.
It used the promoted S1 runtime, the standard 22-token prompt, 8K context,
2,183 slots, raw Safetensors, one warmup, and three retained-runner 128-token
measurements:

| Run | Decode | Mean token step | Cache planning | Hit encoding | Miss fetch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.46 tok/s | 29.89 ms | 0.361 ms | 0.203 ms | 5.127 ms |
| 2 | 33.44 tok/s | 29.90 ms | 0.368 ms | 0.207 ms | 5.094 ms |
| 3 | 33.65 tok/s | 29.72 ms | 0.364 ms | 0.202 ms | 4.956 ms |

The means were 33.52 tok/s, 29.84 ms/token, 0.364 ms planning,
0.204 ms hit-command encoding, and 5.059 ms physical miss fetch. Removing every
planning instruction would reduce the token step by only about 1.2%. Even the
impossible upper bound of removing both planning and all hit encoding is below
2%. A fixed expert-to-slot table cannot reach the predeclared 5% end-to-end
gate on this workload.

The measurement instrumentation was removed after the gate. No cache mapping,
replacement, hit/miss behavior, runtime API, test, or production benchmark
code changed. No candidate A/B or 512-token promotion run was justified.
Reopen fixed-table bookkeeping only if a future non-overlapping profile shows
at least 1.5 ms/token of planning cost under the then-current runtime.

### S3 — expert-miss ceiling offline (closed by measurement)

**Status: four-prompt trace gate completed on 2026-08-30. No production cache
policy, prefetch caller, or runtime benchmark candidate was retained.**

The diagnostic used the promoted S1 executable path, exact greedy sampling,
8K context, 2,183 cache slots, raw Safetensors, no warmup, one 512-token run per
prompt, and trace recording enabled. Each trace included prefill and decode.
The SLRU replay reproduced the measured total miss count exactly before any
comparison was accepted. The lock-aware Belady simulator preserved each
layer's real capacity and prevented eviction of an expert already acquired by
the current top-8 group.

| Prompt class | Prompt tokens | SLRU total | Offline total | Total reduction | Decode misses | Offline decode | Decode reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Explanation | 22 | 43,267 | 24,482 | 43.42% | 40,551 | 21,818 | 46.20% |
| Coding | 29 | 39,170 | 21,902 | 44.08% | 36,101 | 18,958 | 47.49% |
| Summarization | 44 | 33,816 | 18,846 | 44.27% | 29,582 | 15,024 | 49.21% |
| Creative | 33 | 36,309 | 19,451 | 46.43% | 32,470 | 15,917 | 50.98% |

This is a real theoretical gap, not an expected speedup. Belady uses the full
future trace and therefore cannot be used by the running model. The measured
diagnostic decode rates were 27.90, 29.01, 30.80, and 30.32 tok/s respectively;
they are controls only, because no causal candidate passed the trace gate.

Two standard causal 2-bit re-reference policies were replayed next with the
same active-group locks:

| Prompt class | SLRU | SRRIP | SRRIP change | BRRIP | BRRIP change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation | 43,267 | 43,361 | 0.22% worse | 49,220 | 13.76% worse |
| Coding | 39,170 | 39,076 | 0.24% better | 43,327 | 10.61% worse |
| Summarization | 33,816 | 33,676 | 0.41% better | 37,488 | 10.86% worse |
| Creative | 36,309 | 35,717 | 1.63% better | 39,406 | 8.53% worse |

Even a prompt-specific dynamic-program allocation of the same 2,183 SLRU
slots reduced misses by only 1.30%, 1.42%, 1.57%, and 2.47%. SRRIP and layer
reallocation therefore remain below the predeclared 5% end-to-end promotion
threshold before accounting for policy overhead or the fact that a percent of
misses is not a percent of wall time.

The previous-token same-layer route overlap was 37.26%–40.89%, but every such
prediction referred to experts already resident in that layer partition and
would initiate zero useful reads. A causal online transition table was then
trained only on earlier groups in the same trace. The optimistic analysis
excluded cache pollution and counted a prediction only when its expert was
absent from the current SLRU state:

| Prompt class | Useful reads | Wasted reads | Precision | Decode misses covered |
| --- | ---: | ---: | ---: | ---: |
| Explanation | 1,535 | 18,765 | 7.56% | 3.79% |
| Coding | 1,634 | 18,747 | 8.02% | 4.53% |
| Summarization | 1,743 | 18,696 | 8.53% | 5.89% |
| Creative | 2,045 | 18,393 | 10.01% | 6.30% |

Those are the smallest one-read-per-layer budgets. Increasing the budget to
eight raised nominal decode-miss coverage to 21.25%–31.33% while issuing
152,391–154,287 wasted reads. Because each speculative miss reads the same
expert payload as a demanded miss, every tested budget increases total expert
traffic before counting eviction pollution. There is no credible wall-time
translation or runtime A/B to perform.

Conclusion: retain production SLRU. The measurement-only Belady, RRIP, and
transition-prediction code was removed after recording these results. Reopen
cache replacement or prefetch only with a causal routing signal that achieves
high precision on held-out prompts and lowers total expert bytes, not merely a
new policy name or an oracle-only miss reduction.

### S4 — incremental multi-turn prefill

**Confidence: medium for user-visible latency; this does not improve
decode tok/s.**

Every send currently calls `resetState()` and forwards the complete formatted
conversation again. A historical debug-harness second turn replayed 94 prompt
tokens and took 6.86 seconds before decode; capture a current release baseline
before applying the 20% promotion gate. Design an append-only state contract
that retains the 30 DeltaNet recurrent states, 10 full-attention KV prefixes,
exact token prefix, and any pending final token, then processes only the new
template/user suffix.

Before implementation, specify invalidation and rollback for New Chat,
thinking/template changes, mode/context/model changes, cancellation, output
limit, context full, and errors.
[MLX-LM's prompt-cache implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
is a useful reference, but QwenFieldfare must validate its hybrid recurrent/KV
state independently. KV rollback can change a logical length, but DeltaNet and
Conv state mutate destructively; a full recurrent snapshot is roughly 60 MiB
before other state and may consume the remaining 64K headroom. Choose and
budget snapshot, replay, or another rollback method before any implementation.

### S5 — one true Q4 weight-access prototype

**Confidence: low-to-medium; potential impact is higher than the rejected
micro-tweaks, but engineering risk is high.**

Do not revisit row count, launch size, byte-versus-native loads, or the existing
contiguous arena experiment. Any new GPU candidate must materially change data
reuse or weight access, such as a block-interleaved Q4 mapping or an MLX-class
tiled QMV design, while retaining the same Q4/BF16 payload.

Prototype one representative high-cost matrix first. Require a profiler-backed
kernel win large enough to predict at least 5% end-to-end improvement before
changing the model layout or all projections. Reject the research branch if
one representative kernel cannot beat the frozen control. Before prototyping,
specify load-time versus durable repack, routed-miss repack cost, raw-checkpoint
fallback, temporary and steady Metal allocation, and attribution if external
kernel ideas are adapted. A second resident LM-head representation, for
example, does not fit the current 5 GiB profile without reclaimed memory.

## Memory roadmap

There is no unexplained steady Metal allocation in the measured 8K load: the
source model predicts 4.927 GiB at 8K/2,183 slots, matching the measured
4.928 GiB. This does not rule out lifecycle, cancellation, repeated-growth, or
mode-switch leaks. Meaningful steady 8K memory reduction still requires a
deliberate cache/speed tradeoff.

### M1 — explicit Active Streaming memory profiles

**Confidence: high; exact parity and context length are preserved. Throughput
is expected to decrease with fewer slots, but must be measured.**

At 8K, non-cache Metal allocation is approximately 1.3295 GiB. The existing
runner already accepts a cache capacity, so expose honest speed/memory presets
instead of presenting one fixed budget as universally optimal. Preserve the
current approximately 0.07 GiB measurement margin rather than filling every
theoretical byte:

| 8K target | Approximate slots | Saving versus current | Required claim |
| ---: | ---: | ---: | --- |
| 5.0 GiB | 2,183 | control | Speed-oriented default |
| 4.0 GiB | about 1,576 | about 1.0 GiB | Lower memory, measured rate reported |
| 3.0 GiB | about 969 | about 2.0 GiB | Minimum-memory active profile, measured tradeoff reported |

These counts are 8K examples, not constants for other contexts. Derive each
context/profile capacity as:

```text
floor((target bytes - non-cache bytes(context) - safety margin) / 1,769,472)
```

Require at least 320 slots because 40 layers each need eight simultaneously
locked top-8 slots. Validate every preset/context at 128 and 512 tokens and
publish cache misses, fetched bytes, tok/s, and memory. Fewer slots cannot
improve the cache-miss ceiling, but overlap and host variance mean measured
throughput is not guaranteed to decrease monotonically.

### M2 — lazy exact FP16 KV growth

**Confidence: medium until a no-double-residency design is proven; full-context
peak is unchanged.**

The runner allocates all selected-context KV storage at load. Begin with a
small power-of-two FP16 capacity and grow while preserving the exact K/V prefix.
Starting at 1K would initially avoid approximately 140 MiB for an 8K selection,
620 MiB for 32K, and 1.23 GiB for 64K. The saving disappears when the
conversation fills the context. The 8K saving is cleanup under the 0.25 GiB
memory milestone; the feature clears that gate from 16K, where the initial
saving is approximately 300 MiB.

This work is blocked until a no-double-residency design is demonstrated. One
candidate is to grow between token steps and replace one K or V buffer at a
time after all commands using the old buffer complete, proving immediate
release before the next allocation. If that cannot stay below the target, use
paged/chunked KV, shrink the expert cache before growth, or reserve explicit
headroom. The logical context limit remains `maxSeqLen`, while reset, physical
strides, bounds, and `qwen_update_kv_cache_f16` must use the current allocated
KV capacity. Measure peak Metal allocation during every growth transition,
plus latency, cancellation, and allocation-failure behavior.

### M3 — compact per-layer-kind dense arenas

**Confidence: high; impact is modest.**

Separate linear-attention and full-attention layouts can reclaim exactly
79.453 MiB of mutually exclusive unused ranges. This is only 1.6% of current
Metal allocation, so classify it as cleanup rather than a priority speed or
memory milestone. Retaining the saving is preferable to claiming that about 47
additional expert slots will materially change throughput.

### M4 — experimental INT8 KV at long context only

**Confidence: low-to-medium; exact parity is not preserved.**

At 64K, the INT8 payload has a 640 MiB gross saving versus FP16; estimated scale
metadata reduces that to roughly 620–635 MiB before additional scratch. At 8K,
the gross saving is 80 MiB and the estimated post-metadata saving is about
77.5–79.4 MiB. This requires per-token/head quantization, attention
dequantization, long-context quality evaluation, and short-context performance
checks. Keep FP16 as the default and do not begin this experiment before M1/M2
and a quality suite.

## Independent lane order

Decode speed and memory are different product outcomes and are not combined
into a misleading single score.

- **Decode speed:** S1 token selection, S2 cache bookkeeping, S3 offline miss
  ceiling, then S5 one Q4 prototype.
- **Conversation latency:** S4 incremental prefill/state reuse.
- **Memory:** M1 explicit budgets, M2 lazy FP16 KV if its blocker is resolved,
  M3 arena cleanup, then experimental M4 INT8 KV.

The next one-at-a-time decode task is S3's offline miss-ceiling analysis.
Product planning may intentionally prioritize M1 or S4 over high-risk S5, but
that is a product-value decision rather than a decode-speed ranking. S2 does
not authorize a cache-policy or prefetch edit.

## Directions closed by evidence

Do not repeat:

- projection Candidates A–F, input threadgroup copies, SIMD16, or launch-size
  search;
- MoE inactive-SIMD row packing;
- packed one-read/three-read experts as a speed strategy;
- frequency-only eviction or prompt-specific layer repartitioning;
- `F_NOCACHE` or a two-worker miss cap;
- command-buffer-count reduction without critical-path evidence;
- removal of the per-layer router wait, which is a real data dependency;
- fixed-table cache bookkeeping under the current profile, whose complete
  planning-time ceiling is only about 1.2%;
- small scratch/collection cleanup as a headline optimization.

Defer speculative decoding, lower-precision routed experts, expert-count
reduction, and generic model support because they change model behavior or
materially expand the architecture.

## Product work

Performance is only one production requirement. The separately ranked
[product roadmap](PRODUCT_ROADMAP.md) covers strict model validation,
recoverable lifecycle, context budgeting, release qualification, signed app
distribution, diagnostics, and later convenience features.

## Comparative evidence

The current upstream
[TurboFieldfare optimization record](https://github.com/drumih/turbo-fieldfare/blob/main/docs/OPTIMIZATION_JOURNEY.md)
supports the decision method used here: microbenchmark wins are not promoted
without full-runtime results, and its Gemma-specific prediction, I/O, fusion,
and packed-KV failures are warnings rather than evidence that Qwen must behave
the same way. MLX-LM currently implements prompt-cache reuse and quantized KV
cache types; those implementations establish feasibility patterns, not a
QwenFieldfare performance or quality result.

## Previous roadmap (superseded)

The sections below preserve the plan that led to the completed/rejected
experiments. They are historical evidence, not the current execution order.

### Previous recommendation

Optimize the resident GPU projection path first, beginning with the Q4 linear-
attention input projection. Do not begin with another packed-expert format,
larger cache, or replacement eviction policy.

The reason is mathematical, not speculative: the validated 512-token active-
streaming run produced 23.28 tok/s, or approximately 42.96 ms/token. Matching
the 49.86 tok/s MLX reference requires approximately 20.06 ms/token. The best
exact GPU profile already measured 20.96–21.30 ms/token before cache-miss I/O,
CPU scheduling, and sampling. Eliminating expert I/O alone therefore cannot
reach the target.

### Previous current evidence

| Signal | Measurement | Consequence |
| --- | ---: | --- |
| Historical active streaming | 23.28 tok/s | Previous timing protocol; retained as historical evidence |
| Current 128-token baseline | 20.62 tok/s mean | Corrected inter-token timing; 2.99% spread across three runs |
| Current 512-token baseline | 19.68 tok/s | 50.80 ms mean inter-token latency |
| Full-resident MLX reference | 49.86 tok/s | Approximately 20.06 ms/token target |
| Best measured aggregate GPU time | 20.96–21.30 ms/token | GPU work already exceeds the complete target budget |
| Current diagnostic GPU time | 29.36–30.73 ms/token | GPU work is the largest short-run floor in the current host state |
| Current linear input projection | 4.683 ms/token | First narrow fixed-geometry target |
| 8K cache | 2,183 slots; 4.928 GiB Metal | Little safe memory headroom remains under 5 GiB |
| 512-token expert demand | 43,267 misses; 71.302 GiB fetched | Approximately 84.5 misses and 142.6 MiB/token |
| Within-run drift | 53.4 to 101.8 misses/token | Rising miss demand explains much of the late-run slowdown |
| Cache promotion from 2,100 to 2,183 slots | 3.31% fewer misses | Helpful, but no deterministic throughput gain was established |
| Raw versus packed one-read storage | 25.04 versus 24.38 tok/s | Fewer reads did not reduce payload bytes or improve decode |

These results are workload-specific measurements, not hardware ceilings.

### Previous measurement gate

Before changing a kernel, capture a reproducible release baseline with the same
model, prompt tokens, generated-token count, context, cache slots, sampler,
power state, and process exclusions. Record at least three warm runs plus one
512-token run. Keep `--gpu-profile` and `--subcomponent-profile` out of headline
throughput results; use them only in paired diagnostic runs.

Every candidate must report:

- complete token IDs and finish reason;
- prefill, TTFT, mean/p50/p95 decode latency, and tok/s;
- GPU component time, cache misses/token, fetched bytes/token, and hit rate;
- Metal allocation, physical footprint, memory pressure, and swap delta;
- all protocol deviations and every run, not only the fastest result.

This gate is necessary because observed GPU time varied from about 21 ms to
30 ms/token and file-page-cache state also changed results.

### Previous Priority 1: resident Q4 projection kernels

Start with the fused linear-attention input projection, then apply successful
work to the generic Q4 out-projection/full-attention/LM-head path.

Measured projection costs have included:

- 4.683 ms/token for the linear input projection in the current subcomponent
  profile, versus 3.658 ms/token in an earlier corrected GPU split;
- 4.93 ms across the 30 linear QKV/Z/B/A input projections in the earlier
  subcomponent profile;
- 2.64 ms for the 30 linear out projections;
- 1.76 ms for the 10 full-attention Q/K/V projections;
- 0.97 ms for the 10 full-attention out projections.

Candidate experiments, in order:

1. Specialize kernels for the fixed Qwen matrix geometries instead of routing
   every projection through one generic row kernel.
2. Evaluate multi-row output tiling per SIMD group so packed Q4 loads, affine
   metadata addressing, and input-vector reads are amortized across more work.
3. Vectorize packed weight/scale/bias access and remove repeated address
   arithmetic while retaining the exact MLX affine calculation order required
   by parity.
4. Re-profile the already-fused QKV/Z/B/A input path for occupancy, register
   pressure, and memory transactions before adding another fusion.
5. Promote a kernel only after an end-to-end release A/B; an isolated
   microbenchmark improvement is insufficient.

Do not repeat the removed threadgroup copy of the 2,048-element input vector or
the SIMD16 variant. The former improved its isolated projection by only
1.57–1.88% with no full-run improvement; the latter was 2.10% slower.

### Previous Priority 2: MoE GPU compute and per-layer synchronization

After the dense projection work, target the work paid on every token regardless
of cache misses:

- shared-expert plus cache-hit GPU work measured 5.22–5.66 ms/token;
- miss/tail/LM-head GPU work measured 3.29–4.44 ms/token;
- the runtime performs a router completion wait at every layer because the CPU
  must know the eight selected experts before it can resolve cache slots.

The first experiments should measure whether the cached gate/up activation and
down projection can use a geometry-specific fused or more tightly tiled path,
and whether shared/routed reductions can avoid intermediate global-buffer
traffic. Any synchronization redesign must preserve the dependency between the
current layer's routed output and the next layer; simply removing a wait is not
correct.

The objective is not a lower command-buffer count by itself. The objective is a
lower end-to-end GPU critical path with identical token output.

### Previous Priority 3: establish the achievable expert-miss floor

Only after the GPU path improves should expert I/O become the main optimization
track. Start offline using recorded traces rather than changing production
cache policy immediately.

1. Compute a per-layer Belady/offline-optimal replacement bound for several
   real prompts at the current layer capacities. The existing dynamic-program
   result tested layer partitioning, not the complete optimal replacement
   ceiling.
2. Compare SLRU against the bound. If the possible reduction is small, stop
   cache-policy work and focus on overlap or representation.
3. Evaluate temporal route prediction offline across prompts and token ranges.
   Record precision, useful-prefetch rate, wasted reads, and added bytes.
4. Implement prefetch only if the offline trace predicts lower wall time without
   materially increasing total expert traffic or exceeding the memory budget.
5. Measure asynchronous/double-buffered staging only against the existing
   shared-expert/cache-hit overlap; do not count already-overlapped time twice.

Avoid these previously rejected directions:

- frequency-only eviction, which reduced hit rate to 68.58% and increased
  fetched payload to 88.317 GiB;
- prompt-specific layer repartitioning, whose trace-specific gain was only
  1.16% at 2,100 slots;
- another layout whose only advantage is fewer `pread` calls, because packed
  one-read storage was slower than raw safetensors;
- `F_NOCACHE`, which reduced decode to 18.39 tok/s;
- limiting miss concurrency to two workers, which reduced a matching run from
  25.72 to 23.08 tok/s.

### Previous Priority 4: incremental conversation prefill

This does not improve decode tok/s, but it is the largest user-visible latency
target after decode. The app currently rebuilds and prefills the complete
retained conversation. A measured second turn with a 94-token full prefill took
6.86 seconds in an earlier workload.

Analyze an append-only prompt-state cache that retains the 30 DeltaNet recurrent
states and the 10 full-attention KV prefixes, then prefills only the newly
appended user/template suffix. The design must define invalidation for New Chat,
model/mode/context changes, edited history, cancellation, and template changes.
No implementation should begin until exact token-prefix identity and state
rollback behavior are specified.

### Previous Priority 5: CPU cache bookkeeping and encoding

Older non-overlapping measurements attributed about 2.68 ms/token to cache
index lookup and 0.67 ms/token to command encoding. These are secondary until
the larger GPU and miss-fetch costs fall. Later optimization can replace
allocation-heavy per-layer arrays/sets and repeated temporary construction with
fixed-size storage, but only after a current release profile confirms the cost.

### Previous memory constraint

The optimization budget remains a maximum of approximately 5 GiB Metal
allocation in Active Expert Streaming mode, including 64K context support. The
8K/2,183-slot run measured 4.928 GiB and the 64K/1,523-slot load measured
4.937 GiB. A candidate requiring another large resident buffer is not viable
unless it first reclaims equivalent memory elsewhere.

Do not trade the bounded-memory design for MLX-like full residency. Fast
Resident Q4 already exists as the high-memory diagnostic control.

### Previous promotion gates

A change is eligible for the production path only when all of these hold:

1. Exact teacher-forced MLX parity remains 132/132 and the fixed greedy token
   sequence is unchanged.
2. Three repeated release runs show a consistent end-to-end improvement; the
   result is not based only on an isolated kernel benchmark.
3. Long-generation output remains coherent and reaches the same finish reason.
4. Metal allocation stays at or below the established approximately 5 GiB
   envelope for every supported context tier.
5. Memory pressure remains normal, swap growth is zero, cancellation/unload
   still release allocations, and all unit tests pass.
6. The experiment is documented whether retained or rejected, preventing the
   same failed direction from being repeated.

### Previous planned order

1. Capture a stable baseline and corrected GPU split.
2. Optimize the fixed-geometry linear input projection.
3. Generalize any proven projection improvement to out/full/LM-head GEMVs.
   Skipped: execution step 2 promoted no projection change.
4. Optimize shared and routed MoE GPU kernels. Candidate A was rejected and
   removed; no production runtime change is active.
5. Measure the optimal cache-replacement bound and route-prediction ceiling.
6. Implement only the I/O strategy justified by those offline bounds.
7. Design incremental conversation prefill.
8. Revisit CPU bookkeeping after the larger costs have fallen.

The first implementation task should therefore be a narrowly scoped,
reversible Metal experiment on the fixed-geometry linear input projection with
paired microbenchmark and full-generation A/B measurements.

## Execution log

### Priority step 1 — reproducible baseline (completed)

Objective: establish the measurement contract before changing a production
kernel. No throughput improvement is expected from this step.

#### Pre-edit finding

The initial `QwenFieldfareBench` process loaded one runner and executed exactly
one generation. Running the executable repeatedly could warm the macOS file
page cache, but each new process discarded the routed-expert cache, so it could
not produce three retained-runner warm measurements.

The original rate calculation also subtracted prefill time and divided the
remaining wall time by every generated token. The first generated token is
computed by the final prompt forward step before `onPrefillComplete` fires, so
that token has no decode interval in the remaining wall time. The correct
steady decode denominator is `generatedTokens - 1`, measured from emission of
the first generated token to emission of the last. The old formula had little
effect on a 512-token result but biased short runs upward and contaminated the
first latency sample with the final prefill step.

#### Measurement-only harness change

- Add explicit `--warmup-runs` and `--runs` controls while retaining one loaded
  runner and its expert cache.
- Reset run counters, recurrent state, and KV state for each run while retaining
  cache contents intentionally.
- Measure decode using inter-token wall-clock intervals, excluding the first
  token already produced during prefill.
- Report per-run mean/p50/p95 decode latency, decode-only cache deltas, sampled
  memory, and complete bucket drift.
- Print every warmup and measured result and verify that repeated greedy token
  sequences and finish reasons are identical.
- Keep GPU profiling disabled in headline runs; capture a separate diagnostic
  run with exact GPU timestamps and the subcomponent profiler.

Headline protocol:

```bash
.build/release/QwenFieldfareBench \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --prompt "tell me what hasmap is in java with example" \
  --context 8192 --cache-slots 2183 --max-new 128 \
  --warmup-runs 1 --runs 3 --metrics-only
```

Long-run drift protocol:

```bash
.build/release/QwenFieldfareBench \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --prompt "tell me what hasmap is in java with example" \
  --context 8192 --cache-slots 2183 --max-new 512 \
  --runs 1 --metrics-only
```

Expected outcome: a stable baseline and variance envelope for evaluating
Priority 2. Expected performance outcome: no material speedup, because this
step changes measurement code only.

#### Preflight record

- Commit before local measurement changes: `f6f556ac90a8f67ae32713f16978666f72ac6452`.
- Hardware: MacBook Pro `Mac17,2`, Apple M5, 10 CPU cores, 32 GiB unified memory.
- Software: macOS 26.6.2 (`25G83`), Apple Swift 6.3.3.
- Model: `/Users/tanishq/models/Qwen3.6-35B-A3B-4bit`; all four indexed
  safetensors shards plus config/tokenizer files were present. The path also
  contains the optional packed store, but baseline commands use raw
  safetensors.
- Capacity: 241 GiB disk available and 82% system-wide memory free.
- Process exclusion: no QwenFieldfare, MLX, Ollama, or LM Studio model process
  was running.
- Power: `pmset` reported AC power; battery state was 82% and reported
  discharging.
- Swap baseline: 1,607.56 MiB used. The pass criterion is zero material growth,
  not zero absolute swap, because the host already contained encrypted swap.
- Protocol limitation: `pmset -g therm` did not expose thermal or CPU power
  warning status on this system. No thermal-state value is claimed.
- Measurement-only validation: release build passed; CLI help and invalid-run
  validation passed; 42 tests in 10 suites passed.

#### Headline result: one warmup plus three retained-runner measurements

The release headline command completed successfully with GPU profiling
disabled. All runs generated 128 tokens and reached `Output limit`; repeated
greedy token sequences and finish reasons were identical.

| Run | TTFT | Decode rate | Mean / p50 / p95 | Decode misses | Hit rate | Fetched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmup | 2.014 s | 18.82 tok/s | 53.14 / 51.94 / 65.01 ms | 7,199 | 82.29% | 11.864 GiB |
| Measured 1 | 1.401 s | 20.69 tok/s | 48.33 / 47.43 / 59.03 ms | 7,185 | 82.32% | 11.841 GiB |
| Measured 2 | 1.052 s | 20.89 tok/s | 47.87 / 47.27 / 58.01 ms | 7,157 | 82.39% | 11.794 GiB |
| Measured 3 | 1.338 s | 20.27 tok/s | 49.33 / 48.71 / 59.37 ms | 7,170 | 82.36% | 11.816 GiB |

The measured decode-rate envelope is 20.27–20.89 tok/s, with a 20.62 tok/s
mean and 2.99% full spread relative to the mean. This is the baseline for
future end-to-end A/B comparisons. Loaded Metal allocation remained 4.928 GiB
and the final physical footprint remained 4.803 GiB in every measured run.

Each rate uses 127 inter-token transitions for 128 generated tokens. It is not
directly comparable to older reports that divided post-prefill time by all 128
tokens. The historical 23.28 tok/s result remains useful as a record of its
original protocol, but it is not the current promotion baseline.

#### Long-run drift result

The 512-token release run completed successfully with GPU profiling disabled:

- TTFT: 1.851 seconds.
- Decode: 511 inter-token transitions in 25.961 seconds, or 19.68 tok/s.
- Inter-token latency: 50.80 ms mean, 50.07 ms p50, and 65.78 ms p95.
- Decode-only cache: 40,551 misses, 75.20% hit rate, and 66.826 GiB fetched.
- Cache including prefill: 43,267 misses and 71.302 GiB fetched, matching the
  established demand totals.
- Final physical footprint: 4.811 GiB; Metal allocation: 4.928 GiB.

The 64-token buckets show increasing expert demand but not a monotonic token
rate decline:

| Ending token | Rate | Misses/token | Fetched/token | Miss-fetch time |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 20.16 tok/s | 53.4 | 90.2 MiB | 14.41 ms |
| 128 | 19.84 tok/s | 59.9 | 101.1 MiB | 11.71 ms |
| 192 | 19.90 tok/s | 68.5 | 115.6 MiB | 6.55 ms |
| 256 | 19.17 tok/s | 69.0 | 116.4 MiB | 9.40 ms |
| 320 | 17.94 tok/s | 85.2 | 143.8 MiB | 17.19 ms |
| 384 | 18.76 tok/s | 100.4 | 169.5 MiB | 13.91 ms |
| 448 | 21.29 tok/s | 96.2 | 162.3 MiB | 28.94 ms |
| 512 | 20.85 tok/s | 101.8 | 171.8 MiB | 26.78 ms |

The reliable conclusion is that routed-expert misses and bytes per token rise
substantially with generation length. Bucket throughput also depends on GPU
state, page-cache state, and overlap, so the data does not support claiming a
simple monotonic speed decline from I/O alone.

#### Diagnostic GPU split

A separate one-warmup/one-measured 128-token run used `--gpu-profile` and
`--subcomponent-profile`. It is diagnostic only and is excluded from the
headline average because profiling changes timing. The measured run reached
23.43 tok/s after file-page-cache warming and reported:

- exact GPU: 29.36–30.73 ms/token at the sampled boundaries;
- attention/router GPU: 12.25–15.07 ms/token;
- shared/cache-hit GPU: 9.18–11.17 ms/token;
- miss/tail/head GPU: 5.94–6.48 ms/token;
- observed expert miss fetch: 5.21–6.73 ms/token;
- attention subcomponents: 12.353 ms total, including 8.940 ms linear and
  3.412 ms full attention; the fused linear input projection was 4.683 ms.

This establishes two distinct bottlenecks. In a short page-cache-warm run, GPU
execution is the largest latency floor. Over a long generation, increasing
expert misses and bytes introduce substantial length-dependent variability.
Both must eventually improve, but the ranked next implementation remains the
fixed-geometry resident Q4 projection experiment because it is paid on every
token and does not consume more expert-cache memory.

#### Resource and protocol outcome

- Swap changed from 1,607.56 MiB before testing to 1,599.56 MiB afterward: no
  benchmark-induced swap growth.
- Final system-wide memory free was 81%.
- No model or benchmark process remained after the runs.
- Peak reported Metal allocation stayed at 4.928 GiB, inside the approximately
  5 GiB Active Expert Streaming budget.
- The only protocol limitation was unavailable thermal-warning telemetry.
- Headline runs used `--metrics-only`, so the log did not print the complete
  token-ID array. The harness compared the arrays directly and reported them
  identical, but future promoted-candidate A/B records must also retain the
  complete IDs as required by the promotion gate.

#### Priority step 1 conclusion

Expected versus actual: no runtime speedup was expected and none is claimed.
The result is a corrected, repeatable baseline with a 2.99% short-run spread,
long-run drift evidence, deterministic-output checking, and a current GPU/I/O
split. Priority step 1 is complete. Do not begin the fixed-geometry projection
edit until it is explicitly selected as the next one-at-a-time task.

### Priority step 2 — fixed-geometry linear input projection (completed; no promotion)

Objective: reduce the fused Q4 linear-attention input-projection GPU critical
path while preserving exact generated tokens and the approximately 5 GiB Metal
allocation envelope.

#### Pre-edit analysis

The production Active Expert Streaming path already uses
`qwen_fused_linear_attn_in_proj_q4_simd_f32`: one 32-lane SIMD group computes
one output row, and eight SIMD groups produce eight rows per threadgroup. Each
lane reloads the same activation elements independently for every output row.
The earlier rejected SIMD16 experiment is therefore not a useful next test.

The current profile attributes 4.683 ms/token to all 30 linear-layer fused
input projections. At the 20.62 tok/s baseline, even a 25% isolated reduction
would save only about 1.17 ms/token, corresponding to an estimated 2–3%
end-to-end ceiling before interactions. A retained candidate must exceed normal
run variance in repeated full generation; an isolated profiler win alone is
not sufficient.

#### Candidate A: two rows per SIMD group

- Compute two adjacent output rows in each SIMD group.
- Load each eight-element activation slice once per lane and apply it to two
  independent packed-Q4 rows.
- Preserve the current per-row affine accumulation order and separate SIMD
  reductions so numerical output remains unchanged.
- Produce 16 rows per 256-thread threadgroup instead of eight, halving the
  threadgroup count from 1,544 to 772 for the fixed 12,352-row projection.
- Add no persistent buffers and do not change expert-cache capacity, model
  format, runtime defaults, or any non-linear-attention projection.

Promotion requires release build/tests, identical greedy token IDs, a paired
subcomponent improvement, three retained-runner headline measurements, no
Metal-allocation increase, and a coherent long run if the headline gain clears
the variance envelope. If it fails those gates, Candidate A will be removed and
documented as rejected before another kernel design is attempted.

#### Candidate A result: rejected

The paired same-session subcomponent measurement regressed from 5.238 to 5.391
ms/token for the fused input projection, a 2.9% slowdown. Release compilation
and all 42 tests passed, but the candidate failed the isolated performance gate
and was not eligible for full-generation testing. The probable cause is added
register/weight/metadata pressure; the activation vector was already served
effectively enough by GPU caches that explicit two-row reuse did not compensate.
The two-row kernel and dispatch are removed before Candidate B.

#### Candidate B: native aligned packed and metadata loads

The fixed 2,048-column geometry gives every packed-weight row a 1,024-byte
stride and every scale/bias row a 64-byte stride. All base offsets and accesses
are therefore aligned for native 32-bit packed-weight loads and 16-bit BF16-bit
pattern loads. The current helper instead assembles a packed `uint` from four
`uchar` loads and a BF16 value from two `uchar` loads.

Candidate B will retain one row per SIMD group and the current accumulation
order, but reinterpret the fixed-geometry row pointers as `uint` and `ushort`.
It removes byte assembly and repeated shifts without changing persistent memory
or the quantization calculation. The same promotion gates apply.

#### Candidate B result: rejected

The controlled three-run comparison averaged 5.280 ms/token for the original
helper (5.721, 4.805, and 5.314 ms) and 5.423 ms/token for aligned loads (5.406,
5.337, and 5.526 ms). Candidate B was 2.7% slower and is removed. Native loads
did not improve generated GPU work; compiler load coalescing or scheduling cost
likely eliminated the intended instruction advantage.

#### Candidate C: 128-thread projection threadgroups

The one-row SIMD arithmetic remains unchanged. Candidate C reduces each
threadgroup from eight SIMD groups/256 threads to four SIMD groups/128 threads,
with four output rows per threadgroup. The kernel uses no threadgroup memory,
so this tests whether smaller scheduling units improve occupancy without adding
memory, changing arithmetic, or revisiting the rejected SIMD16 lane width.

#### Candidate C result: rejected

Three 128-thread runs averaged 5.322 ms/token (5.408, 5.505, and 5.054 ms),
versus the 5.280 ms original mean. The 0.8% regression fails the isolated gate;
Candidate C is removed.

#### Candidate D: 512-thread projection threadgroups

Candidate D tests the opposite scheduling direction once: 16 SIMD groups and
16 rows per 512-thread threadgroup. It changes no arithmetic or memory and
halves threadgroup count relative to production. If this fails, threadgroup-size
search stops and the next candidate must change useful work rather than launch
geometry alone.

#### Candidate D result: rejected

Three 512-thread runs averaged 5.171 ms/token (5.548, 4.812, and 5.152 ms), a
2.1% projection-only improvement over the 5.280 ms original mean. This saves an
estimated 0.11 ms/token end to end, which is far below the measured 2.99%
headline variance. It is not impactful enough to promote; the 256-thread
production geometry is restored and launch-size search stops.

#### Candidate E: contiguous fixed projection arena and output

The first 14,229,504 bytes of every dense arena already equal the exact combined
size of QKV/Z/B/A weights, scales, and biases. Linear layers currently stage
those bytes matrix-by-matrix (`W/S/B`, then the next matrix), forcing the fused
kernel to bind twelve logical inputs and branch every output row to select a
matrix, offsets, and one of four output buffers.

Candidate E changes only the linear-layer interpretation of the same arena
region to `all weights / all scales / all biases`. The four matrices become one
fixed 12,352-row Q4 projection. The kernel binds three arena regions, one input,
and one contiguous output; it removes row-classification branches and 24 offset
constants. QKV, Z, B, and A consumers bind offsets within that output. Full
attention layers retain their existing arena layout. Persistent weight bytes
are unchanged and the four scratch outputs are replaced by one equal-size
buffer, so Metal allocation should remain unchanged.

#### Candidate E result: not promoted alone

Candidate E's three projection profiles averaged 4.979 ms/token (5.068, 5.157,
and 4.712 ms), 5.7% below the 5.280 ms original mean. It preserved 132/132 MLX
greedy token IDs and 4.928 GiB Metal allocation. However, the fair current-state
headline averaged 31.52 tok/s versus 31.47 tok/s for a temporary build of pushed
baseline commit `2f49ba2`: only +0.16%. Candidate E is not impactful alone.

#### Candidate F: precomputed affine input-group sums

Every Q4 output row currently recomputes eight-element partial sums for the
same normalized input, then applies the per-group affine bias. Across 12,352
rows and 30 linear layers this repeats hundreds of millions of additions per
token. A linear-specific RMSNorm kernel can compute the 32 sums of each
64-element quantization group once. The projection then retains all weight-dot
arithmetic but applies one bias FMA per group via the precomputed sums instead
of recomputing partial sums in every row.

Candidate F is tested together with Candidate E's contiguous layout because it
provides the branch-free fixed projection needed by the specialized helper.
The combined change is promoted only if it retains exact token IDs and produces
a repeatable end-to-end gain over commit `2f49ba2`; Candidate E's isolated result
is not counted as sufficient evidence.

#### Candidate F result: rejected

Candidate F initially appeared to reduce the projection profile substantially:
3.678, 3.692, and 3.619 ms/token, or 3.663 ms/token mean. An immediate profile
of the exact pushed baseline measured 3.695, 3.688, and 3.597 ms/token, or 3.660
ms/token mean. The paired means are effectively identical; the apparent gain
relative to older profiles was GPU-state drift, not Candidate F.

A 65-token Candidate F greedy sequence matched the recorded MLX token IDs.
Candidate F changes floating-point association, however, and its run recorded
four different expert-cache misses. Because it failed the performance gate, the
full 132-token promotion parity test was intentionally not used to justify it.

The first ordered headline pair appeared positive:

| Build | Measured decode rates | Mean | Spread |
| --- | --- | ---: | ---: |
| Candidate E+F | 29.88, 30.35, 30.24 tok/s | 30.16 tok/s | 1.58% |
| Exact pushed baseline | 29.44, 29.49, 29.30 tok/s | 29.41 tok/s | 0.64% |

This +2.55% ordered result conflicted with the identical isolated GPU time, so
it was not accepted. Reversing the comparison produced the opposite result:

| Build | Measured decode rates | Mean | Spread |
| --- | --- | ---: | ---: |
| Candidate E+F | 29.77, 31.17, 31.52 tok/s | 30.82 tok/s | 5.67% |
| Exact pushed baseline | 31.38, 31.33, 31.02 tok/s | 31.25 tok/s | 1.14% |

Candidate E+F was 1.38% slower in the closing pair. Across all six measured
runs per build, Candidate E+F averaged 30.49 tok/s and baseline averaged 30.33
tok/s: a 0.53% difference, below the established variance and contradicted by
the paired projection profiles. Metal allocation was 4.928 GiB for both.

Candidate F and Candidate E are therefore removed together. They add a second
dense-arena layout, another RMSNorm kernel, altered floating-point association,
and more runtime bindings without a reproducible end-to-end benefit.

#### Priority step 2 conclusion

No fixed-geometry input-projection candidate met the promotion gate:

- two-row activation reuse regressed the projection by 2.9%;
- aligned packed/metadata loads regressed it by 2.7%;
- 128-thread scheduling regressed it by 0.8%;
- 512-thread scheduling improved only the isolated projection by 2.1%, an
  estimated 0.11 ms/token end-to-end;
- contiguous layout measured only +0.16% end to end;
- precomputed affine sums had identical paired GPU time and no repeatable
  headline gain.

The evidence indicates that the fixed input projection is bandwidth-bound and
already receives coalesced row-major reads; removing address arithmetic or
scalar additions does not materially shorten its critical path. A larger gain
would require a different same-size weight representation and tiled algorithm,
not another launch-size or instruction-level variation.

Production runtime source remains exactly as pushed in commit `2f49ba2`; later
commit `5cbc801` changes CI only. Execution step 3 is skipped below.

### Execution step 3 — projection generalization (skipped)

This step required a projection improvement promoted by execution step 2.
None of Candidates A–F met the end-to-end promotion gate. Applying their launch
or arithmetic changes to linear out, full-attention, or LM-head projections
would therefore generalize an unproven or rejected result. The measured costs
also limit the likely value: the best launch-only candidate saved an estimated
0.11 ms/token on the original input-projection target, below normal headline
variance. No runtime edit is justified, and execution step 3 is skipped.

### Execution step 4 — shared and routed MoE GPU kernels (Candidate A rejected)

Objective: reduce GPU work paid on every token while preserving the current
CPU/GPU overlap for expert-cache misses, exact token output, and the bounded
approximately 5 GiB Metal allocation.

#### Current control flow

For every layer, the production path:

1. encodes attention, post-attention RMSNorm, router logits/top-8, and the
   shared-expert coefficient;
2. waits because the CPU needs the selected expert IDs to resolve cache slots;
3. starts shared-expert and cache-hit GPU work in one command buffer while the
   CPU fetches cache misses;
4. runs miss-only routed work, combines hit/miss/shared outputs with the
   residual, and continues on the same Metal queue.

The router wait is a real data dependency. Removing it, moving shared work in
front of it, or merging hit and miss work after all reads would either be
incorrect or destroy existing I/O overlap.

#### Current diagnostic measurement

Commit `5cbc801` has no runtime difference from control commit `2f49ba2`. A
diagnostic-only release run used the standard 22-token prompt, 8K context,
2,183 cache slots, 128 generated tokens, one warmup, one measured run, raw
safetensors, and GPU profiling. It is excluded from headline throughput.

- measured decode: 30.78 tok/s, 32.49 ms mean inter-token latency;
- exact GPU at token 128: 21.02 ms/token;
- attention/router: 11.26 ms/token;
- shared/cache-hit MoE: 5.55 ms/token;
- miss/tail/LM-head: 4.21 ms/token;
- decode cache: 7,185 misses, 82.32% hit rate, 11.841 GiB fetched;
- Metal allocation: 4.928 GiB;
- repeated greedy output: identical.

The two MoE buckets total 9.76 ms/token, but the miss/tail bucket also includes
the LM head and the two buckets overlap CPU expert reads differently. Their sum
is an optimization opportunity, not an amount that can simply be subtracted
from wall time.

#### First technically credible candidate

The hit-only and miss-only routed-down kernels each dispatch eight SIMD groups
per output row. Each kernel uses only `expert_count` groups and makes the rest
write zero. When both subsets exist, the two kernels schedule sixteen SIMD
groups per row although exactly eight groups perform expert dot products across
the two subsets. The measured 56.6 misses/token equals approximately 1.42 miss
experts per layer, leaving about 6.58 hit experts per layer on this prompt.

A geometry-specific subset-down kernel could pack multiple output rows into a
threadgroup according to `expert_count`, so only useful expert/row SIMD groups
are scheduled. It would retain the hit-before-I/O-completion and miss-after-read
split, the exact router-order reduction, the same Q4 weight bytes, and the same
persistent buffers.

This is not yet approved for implementation. It is certain to remove idle SIMD
groups, but it is not certain to improve wall time because useful weight traffic
is unchanged and the kernel may already be bandwidth-bound. Claiming a speedup
before an A/B would be false.

#### Required stages before promotion

1. Freeze the exact pushed control build and record two alternating diagnostic
   control runs to establish same-session GPU variance.
2. Implement only the reversible subset-down row-packing kernel and dispatch;
   do not alter cache policy, expert format, overlap, or persistent memory.
3. Reject immediately unless paired GPU measurements reduce both hit-subset
   and miss-subset critical paths without increasing attention/router time.
4. Verify the complete 132-token MLX parity sequence and the fixed greedy
   sequence before headline testing.
5. Run alternating control/candidate headline pairs, each with one warmup and
   three measured 128-token generations. Require a repeatable gain exceeding
   the established variance rather than selecting the fastest run.
6. Only after that gate, run the 512-token drift test, 64K load test, memory and
   swap checks, cancellation/unload checks, release build, and all package
   tests.
7. Remove and document the candidate if any gate fails. Do not proceed to a
   second MoE kernel idea on the same evidence.

#### Candidate A result: rejected

An exact control benchmark was frozen in a separate SwiftPM scratch build before
the edit. Its one-warmup/two-measured diagnostic run averaged 31.00 tok/s
(31.04 and 30.96 tok/s) with 0.27% spread. At token 128 its measured GPU splits
were:

| Control run | Exact GPU | Attention/router | Shared/hits | Miss/tail/head |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.25 ms | 10.70 ms | 5.68 ms | 3.88 ms |
| 2 | 20.48 ms | 11.22 ms | 5.26 ms | 4.00 ms |

Candidate A implemented only the planned subset-down row packing. Expert counts
of one, two, or four packed eight, four, or two output rows per 256-thread
threadgroup; other counts used the largest whole-row packing possible. Active
expert dot products, subset split, router-order reduction, cache behavior,
persistent buffers, and I/O overlap were unchanged.

The candidate's matching one-warmup/two-measured diagnostic run averaged 30.65
tok/s (30.67 and 30.64 tok/s) with 0.10% spread, a 1.13% regression from the
control. At token 128 its GPU splits were:

| Candidate run | Exact GPU | Attention/router | Shared/hits | Miss/tail/head |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.83 ms | 10.87 ms | 5.84 ms | 4.12 ms |
| 2 | 20.75 ms | 11.24 ms | 5.49 ms | 4.02 ms |

Generated length, finish reason, cache demand, fetched bytes, repeated greedy
output, and 4.928 GiB Metal allocation matched the control. The candidate
failed the first performance gate, so MLX parity, headline, long-run, and test
promotion stages were correctly not used to rescue it. Runtime source is
restored to the pushed control.

Conclusion: scheduling inactive SIMD groups was visible structurally but was
not the MoE critical-path bottleneck. The useful Q4 dot-product weight traffic
dominates this kernel. Do not repeat row packing or dynamic subset threadgroup
geometry without evidence of a changed weight-access design.
