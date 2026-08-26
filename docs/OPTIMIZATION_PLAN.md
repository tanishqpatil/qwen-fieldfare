# Optimization analysis and priority order

Status: Priority step 1 completed. This document records analysis and benchmark
experiments; no production runtime path or default has been changed.

## Recommendation

Optimize the resident GPU projection path first, beginning with the Q4 linear-
attention input projection. Do not begin with another packed-expert format,
larger cache, or replacement eviction policy.

The reason is mathematical, not speculative: the validated 512-token active-
streaming run produced 23.28 tok/s, or approximately 42.96 ms/token. Matching
the 49.86 tok/s MLX reference requires approximately 20.06 ms/token. The best
exact GPU profile already measured 20.96–21.30 ms/token before cache-miss I/O,
CPU scheduling, and sampling. Eliminating expert I/O alone therefore cannot
reach the target.

## Current evidence

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

## Required measurement gate

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

## Priority 1: resident Q4 projection kernels

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

## Priority 2: MoE GPU compute and per-layer synchronization

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

## Priority 3: establish the achievable expert-miss floor

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

## Priority 4: incremental conversation prefill

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

## Priority 5: CPU cache bookkeeping and encoding

Older non-overlapping measurements attributed about 2.68 ms/token to cache
index lookup and 0.67 ms/token to command encoding. These are secondary until
the larger GPU and miss-fetch costs fall. Later optimization can replace
allocation-heavy per-layer arrays/sets and repeated temporary construction with
fixed-size storage, but only after a current release profile confirms the cost.

## Memory constraint

The optimization budget remains a maximum of approximately 5 GiB Metal
allocation in Active Expert Streaming mode, including 64K context support. The
8K/2,183-slot run measured 4.928 GiB and the 64K/1,523-slot load measured
4.937 GiB. A candidate requiring another large resident buffer is not viable
unless it first reclaims equivalent memory elsewhere.

Do not trade the bounded-memory design for MLX-like full residency. Fast
Resident Q4 already exists as the high-memory diagnostic control.

## Promotion gates

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

## Planned order

1. Capture a stable baseline and corrected GPU split.
2. Optimize the fixed-geometry linear input projection.
3. Generalize any proven projection improvement to out/full/LM-head GEMVs.
4. Optimize shared and routed MoE GPU kernels.
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
