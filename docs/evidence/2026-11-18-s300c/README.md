# S3-00c final combine experiment

**Selected path: original fixed router order.** Both deterministic paths
eliminate measured dependence on cache residency and preserve the existing
strict 131/132 result. Changing the final parentheses gives no useful margin
improvement on the relevant hard cases. Production uses the original router
order with its existing left-associated final adds. This closes the reduction
choice for S3-00. The known MLX discrepancy and the independent prefix
qualification still apply.

## Scope and arithmetic proof

At upstream TurboFieldfare revision `de9d1b4`, the
[decode scheduler](https://github.com/drumih/turbo-fieldfare/blob/de9d1b4d195a17c76a952d32c54c0b91d2605697/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift#L1870-L1904)
combines the cached and fetched paths with a single
[top-eight reduction](https://github.com/drumih/turbo-fieldfare/blob/de9d1b4d195a17c76a952d32c54c0b91d2605697/Sources/TurboFieldfare/Metal/MoE/moe.metal#L446-L483).
Its [fused layer tail](https://github.com/drumih/turbo-fieldfare/blob/de9d1b4d195a17c76a952d32c54c0b91d2605697/Sources/TurboFieldfare/Kernels/Fusions/FusedLayerTail.swift)
supplies scheduling context. Qwen retains its own projections, weighting and
expert math. The tested alternative changes only the final two additions,
from `(residual + shared) + routed` to `residual + (routed + shared)`.
The top-eight loop stays the same. A scoped `clang fp reassociate(off)
contract(off)` applies to the alternative's final two additions. Runtime
Metal dispatch through the app's actual compiler distinguished the original
`[0, 0, 0, 1]` from the candidate `[1, -1, 0, 1]` on cancellation
sentinels. The final two rows verify identical router-order accumulation
where a different fold would change the answer. The offline Metal toolchain
was unavailable. These tests observe real GPU arithmetic.

For real model samples, the grouped GPU output agreed bitwise with an
independently evaluated FP32 router sum and proposed tail equation at all
57,344 checked layer coordinates across both histories and both cache states.
Router IDs and weights, selected expert record digests, expert input,
activations, and weighted contributions match the saved original at all
14 sampled positions in both cache histories. The grouped path matches
bitwise across warm/cold at all 152 strict and 941 sustained full logit
positions, and at all sampled final hidden states. Changing only the tail
is measurable at output 465: grouped versus original raw-logit relative L2
is 1.212%, with 0.955% final hidden relative L2.

## Numerical trade-off

The exact 21+132 strict and 430+512 sustained teacher histories from S3-00b
were reused separately. The [comparison](comparison.json) includes
full-vocabulary errors, logit distribution, top five and margins at all
14 positions against both native and sequential MLX. No MLX, legacy Swift
or router-order capture was repeated. All 168 archived raw control arrays
were verified against their S3-00b digests.

| Native MLX comparison | Legacy cold | Legacy warm | Original router order | Grouped tail |
| --- | ---: | ---: | ---: | ---: |
| Strict mean logit relative L2 | 2.299% | 2.298% | 2.300% | 2.296% |
| Sustained mean logit relative L2 | 7.571% | 7.780% | 7.613% | 7.607% |
| Sustained mean logit RMSE | 0.18104 | 0.18588 | 0.18197 | 0.18161 |
| Sustained mean hidden relative L2 | 8.846% | 9.024% | 8.897% | 8.868% |
| Sustained top-1 matches across 7 samples | 6 | 5 | 5 | 5 |

| Position and signed pair | Native MLX | Original router order | Grouped tail |
| --- | ---: | ---: | ---: |
| Strict token 61, 14988−10000 | +0.12500 | −0.10266 | −0.11650 |
| Strict 82, native MLX top pair | 0 | +0.07501 | +0.06689 |
| Strict 123, native MLX top pair | +0.12500 | +0.16011 | +0.16089 |
| Sustained 645, native MLX top pair | 0 | +0.05772 | +0.05564 |
| Sustained 850, native MLX top pair | +0.12500 | −0.18171 | −0.19407 |
| Sustained output 465, 1031−285 | −0.12500 | +0.01194 | +0.01535 |

Both MLX schedules favor 285 at output 465. Legacy cold chooses 285 and
legacy warm chooses 1031. Both deterministic candidates consistently choose
1031. The grouped change slightly lowers aggregate error while widening the
disputed signed margin and moving the strict mismatch and position 850
farther from native MLX. It changes none of the 512 forced predictions from
the original router-order path. The original is the smaller decision:
it resolves the demonstrated residency inconsistency with aggregate
fidelity within the observed legacy cold/warm spread, preserves the strict
mismatch set, and stays closer to MLX on the three disputed margins above.
It trades cold legacy's ranking at output 465 for stable output. This is
a documented local fidelity limit. No further combine-order search is planned.

## Resource and timing trade-off

The combine uses one 65,536-byte allocation. Production replaces two
8,192-byte subset sums, for **48 KiB net**. The existing memory plan charges
those 48 KiB to its safety reserve and preserves all planned cache sizes.
Both experimental groupings had exactly the same scratch allocation. The
existing A/B against legacy measured −2.76% mean decode change in five
128-output pairs (95% interval −7.58% to +2.07%) and a −0.48% mean across two
512-output pairs. [Legacy timing anchor](../2026-11-18-s300a/README.md)

This run adds only a targeted grouping comparison, using the same opt-in
release executable, 52 prompt tokens, greedy seed 42, context 8192, 5 GiB
and 2,183 slots. Each measured request follows a same-length warmup in its
own process. The 128-output order reverses and the 512 pair is a sustained
spot check. See the [exact commands and measured counters](benchmark/runs.json).

| Output tokens | Pair | Original tok/s | Grouped tok/s | Grouped change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 34.05 | 34.91 | +2.53% |
| 128 | 2 | 34.77 | 34.46 | −0.89% |
| 512 | 1 | 29.37 | 28.84 | −1.80% |

There is no reproducible speed reason to choose the alternative. These are
bounded timing observations without a fresh legacy rerun. System-wide pressure
remained at level 1 during sampled pre/postflights. Benchmark traces preserve
cache hit/miss, logical I/O, system device traffic, CPU time and process
footprint. System device traffic is not attributable to just this process.

## Resume S3-00 from the accepted reduction

1. Freeze this implementation as the independent full-replay control. Keep
   the exact consumed-token boundary and one-owner lifetime contract.
2. Reapply only the bounded prefix ledger and transactional cache controls
   against this control. Start with the mode off. Verify complete consumed-ID
   prefix, configuration identity, normal EOS and successful terminal flush.
   Reset and replay on mismatch, cancellation, invalid output or clear.
3. Requalify the archived tool sequence and sustained 512-token workload
   against same-executable full replay under controlled cold/warm residency.
   Compare raw tokens, finish reasons and the known near ties before measuring
   repeated-turn latency, memory, or considering default promotion.

## Reproduction

Apply [the temporary experiment](combine-probe.patch.gz) to `d821f21` in a
disposable checkout. Use S3-00b raw captures or regenerate them with its
documented reference script if unavailable. On a normal-pressure machine with
one model owner, build the release test, verify and link the existing controls
into an empty local capture directory, then run the opt-in test and comparator.

```bash
gzip -dc docs/evidence/2026-11-18-s300c/combine-probe.patch.gz | git apply
Scripts/test.sh -c release --filter ExpertResidencyProbeTests
python3 docs/evidence/2026-11-18-s300c/prepare_controls.py \
  /absolute/s3-00b-captures docs/evidence/2026-11-18-s300b /absolute/empty-capture-directory
QWENFIELDFARE_PREFIX_CHECKPOINT=/absolute/model-directory \
QWENFIELDFARE_EXPERT_RESIDENCY_PROBE=1 \
QWENFIELDFARE_RESIDENCY_OUT=/absolute/empty-capture-directory \
  Scripts/test.sh -c release --skip-build --filter ExpertResidencyProbeTests
/absolute/numpy-venv/bin/python docs/evidence/2026-11-18-s300c/compare.py \
  /absolute/empty-capture-directory /absolute/comparison.json
python3 docs/evidence/2026-11-18-s300c/benchmark.py \
  /absolute/model-directory .build/release/QwenFieldfareBench /absolute/benchmark-directory
```

The main report records derived metrics and array hashes. No raw captures,
binaries, or checkpoint files are part of the evidence commit.

## Integrated production validation

The focused production tests exercise weighted Q4 down projection and
identical combine results using all hits, all misses, and two mixed masks.
The 8K profile and 3/4/5 GiB plan tests pass. An initial budget test caught
a one-slot reduction if the extra scratch were charged outside the reserve.
The final plan preserves the prior per-profile slot counts.

The bounded [production replay](production-replay.txt) matched seven of
seven reference-probe hidden and raw-logit byte pairs and all 512 sustained
teacher predictions. Target 5 GiB, actual measured Metal delta 5,291,294,720
bytes (4.928 GiB), 2,183 slots. The temporary snapshot harness is archived as
[production replay patch](production-replay.patch.gz) and removed from the
final source. The first harness compilation failed on optional digest
formatting and was corrected before replay.

The clean [package suite](package-tests.txt) passed 155 tests with three
opt-in skips, and the [release build](release-build.txt) passed in 43.42 s.
The final executable's [strict run](production-strict.txt) verified all
original checkpoint hashes and returned the original 131/132 result and
exit 1. The fixture SHA-256 is unchanged. Six checker tests and the separate
[known-mismatch check](checker-production.txt) pass. Pressure was level 1
and swap was 2,309.06 MiB postflight versus 2,395.44 MiB preflight.
See [validation summary](validation.json).
