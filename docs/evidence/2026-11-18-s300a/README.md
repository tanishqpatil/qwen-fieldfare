# S3-00a expert-cache residency evidence

The active streaming runner's legacy MoE path sums weighted expert contributions
separately in the hit and miss subsets. The subsets may split at different
router positions on otherwise identical inputs. The observed layer-output
differences follow that grouping. The router-order diagnostic uses the same
weight reader and projection arithmetic, writes each weighted result to its
router position, and adds the eight positions only after both subsets finish.

## Matched layer 0 at position 0

The probe reset all model state and seeded real layer-0 records using the current
cache and checkpoint readers. It used the same first token of the archived
430-token prompt in every case. Cold starts with no expert records. Warm starts
with all eight. The mixed cases preseed alternating router positions. The even
mask is independently rebuilt and replayed.

| Legacy input case | Miss positions | Layer elements with different bits vs cold | Max abs | RMS | Relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | 0–7 | 0 | 0 | 0 | 0 |
| Warm | None | 0 | 0 | 0 | 0 |
| Even hits | 1, 3, 5, 7 | 195 | 7.451e−9 | 6.367e−10 | 1.265e−8 |
| Odd hits | 0, 2, 4, 6 | 195 | 7.451e−9 | 6.367e−10 | 1.265e−8 |
| Even hits repeated | 1, 3, 5, 7 | 195 | 7.451e−9 | 6.367e−10 | 1.265e−8 |

The repeated even mask matches the first even mask bit for bit. Router IDs are
129, 38, 132, 21, 253, 213, 244, and 193. All eight route weights, per-record
weight hashes, normalized MoE inputs, 4,096 activation values, 16,384 weighted
per-expert values, shared expert outputs, and attention residuals match across
masks. The individual legacy subset sums match FP32 sums reconstructed from the
independently captured contributions. The host's left-to-right reconstruction
of the *final* combine differs in bits even in the cold control, so the source
expression does not establish a literal compiler instruction order. The actual
Metal router-order diagnostic makes the layer output invariant over all masks.
The probe waited after the captured layer and used real GPU work on both
branches. This addresses wrong bytes, activation math, and stale outputs for
the tested cases. It cannot rule out every untested timing or I/O fault.

## Sustained identical-input replay

The [forced input](teacher-input.json) comes from the archived raw warm replay
and matches the original output SHA-256. Both histories start with empty model
state and the same 17-token warmup. The warm history also retains expert weights
from the 355-token setup and its 44 consumed output tokens. Both reset model
state before the continuation. Both consume 430 prompt IDs and the first 511 of
the same 512 teacher IDs. Thus the two different predictions at index 465 see
the same prior tokens. The comparison uses SHA-256 of raw FP32 logit bytes at
each of the 941 input positions and direct element metrics at the boundary. Hidden refers to the last layer's
output before final RMS normalization.

| Mode and prehistory | Winner at output 465 | Logit 1031 | Logit 285 | Top-two margin |
| --- | ---: | ---: | ---: | ---: |
| Legacy, tool retained | 1031 | 25.773659 | 25.758919 | 0.014740 |
| Legacy, warmup only | 285 | 25.693356 | 25.705835 | 0.012480 |
| Router order, tool retained | 1031 | 25.760391 | 25.748455 | 0.011936 |
| Router order, warmup only | 1031 | 25.760391 | 25.748455 | 0.011936 |

| Boundary comparison | Array | Max abs | RMS | Relative L2 |
| --- | --- | ---: | ---: | ---: |
| Legacy history vs legacy history | Hidden | 0.043121 | 0.007820 | 1.452% |
| Legacy history vs legacy history | Logits | 0.172230 | 0.033368 | 1.950% |
| Router order vs router order | Hidden and logits | 0 | 0 | 0 |
| Router order vs legacy warm | Hidden | 0.039019 | 0.005189 | 0.963% |
| Router order vs legacy warm | Logits | 0.111104 | 0.023236 | 1.358% |

Legacy histories differ in logit bits at all 941 positions, starting at
position 0. The first prediction change is the original output 465 (input
position 894, both zero-based). The canonical histories produce equal logit
digests at all 941 positions, equal hidden bits at position 894, and the same
512 predictions. Holding everything except the expert sum path fixed removes
the effect for this fixture. That supports H1 as the cause of the demonstrated
sensitivity and gives no evidence for H2 or an independent H3 cause in this
replay. It leaves downstream amplification mechanisms and other workloads
outside the experiment. The canonical path has different logits from legacy
warm from the first position even though its 512 predicted IDs match.

Full [comparison JSON](comparison.json) includes all masks, hashes, weights,
route scores, top-five logits, and memory and cache telemetry.

## Resource and timing observations

One 8 × 2,048 FP32 buffer requires 65,536 bytes across **all** layers. Measured
Metal allocation rose by exactly 65,536 bytes, from 5,291,294,720 to 5,291,360,256,
under the 5 GiB plan with 2,183 expert slots and context 8,192. Maximum sampled
process footprint was 5,261,353,960 bytes. The single Metal queue orders both
subsets before the combine and the next layer's reuse of the buffer. Hit
projection still overlaps refills into disjoint miss slots. There is no
per-layer buffer collection or activation cache.

| Forced replay | Prefill s | First 128 decode steps s | All 511 decode steps s | Expert hits | Expert misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy warm | 18.678 | 4.451 | 17.878 | 223,876 | 77,244 |
| Router order warm | 18.497 | 4.433 | 17.908 | 223,935 | 77,185 |
| Legacy cold | 18.428 | 4.392 | 17.808 | 224,172 | 76,948 |
| Router order cold | 18.477 | 4.450 | 17.886 | 224,192 | 76,928 |

One instrumented comparison per history, using per-position logit readback and
hashing, gives these timings. They do not estimate uninstrumented throughput.
The records distinguish logical expert bytes from physical SSD traffic.

## Alternating release performance

The benchmark uses one executable with an environment opt-in for router order,
a 52-token public coding prompt, greedy seed 42, one same-length retained-runner
warmup, one measured generation, context 8,192, 5 GiB and 2,183 slots. Each run
loads one model. Pair order reverses. The 14 measurements below exclude GPU
profiling. Prefill and decode are measured separately from startup. Both paths
produce the requested output counts. Within each process, warmup and measured
outputs match. This benchmark does not check token IDs across the two modes.
For that check use the matched teacher history above.

| Output count | Pair | Legacy tok/s | Router order tok/s | Legacy prefill s | Router order prefill s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 29.20 | 29.50 | 1.968 | 1.963 |
| 128 | 2 | 31.84 | 29.90 | 1.785 | 1.937 |
| 128 | 3 | 34.27 | 33.59 | 1.746 | 1.723 |
| 128 | 4 | 29.85 | 30.07 | 1.962 | 1.962 |
| 128 | 5 | 32.14 | 29.74 | 1.760 | 1.940 |
| 512 | 1 | 28.07 | 27.62 | 2.060 | 2.057 |
| 512 | 2 | 28.32 | 28.50 | 2.028 | 2.042 |

The 128-output mean paired decode change is −2.76% (five-pair 95% interval
−7.58% to +2.07%). Prefill averages +3.43% in paired elapsed-time change. The
512-output decode changes are −1.60% and +0.64%, with mean −0.48%. Prefill
averages +0.27%. There is no measured speed gain. The 128 interval also leaves
room for a slowdown. Two long pairs are insufficient to settle a small cost.

The 128-output legacy/router-order decode records show 7,889/7,896 misses,
80.59%/80.57% hit rate, and 13.001/13.012 logical GiB. At 512 outputs they show
38,336/38,276 misses, 76.56%/76.59%, and 63.176/63.077 logical GiB. Bench rounds
Metal to 4.928 GiB on both. Observed final footprints span 4.793–4.818 GiB.
System pressure stayed at level 1. Whole child-process user+system CPU time
spans 9.96–10.34 s at 128 and 32.67–33.37 s at 512, including load and warmup.
System-wide disk0 transfers were 4,967–9,711 MB at 128 and 23,857–26,534 MB at
512. These counters include OS caching and unrelated device activity. There
is no per-process physical SSD attribution. See [all timed runs](benchmark/runs.json)
and [paired summary](benchmark/paired-summary.json), including device snapshots,
exact commands, and CPU measurements.

With separate GPU profiling, a 128-output measured run sums to 3,652.134 ms
of model GPU command duration for legacy and 3,614.822 ms for router order.
This measures command durations across prefill and decode. It is not device
utilization or wall time. The totals come from the existing per-forward GPU
timestamp buckets accumulated for each run in a second temporary build.
See [legacy](gpu-totals-0.txt), [router order](gpu-totals-1.txt), and
[profiling patch](gpu-totals.patch.gz). This one profiling pair supports no
speed claim and is excluded from the headline pairs.

## Independent reference check

The prototype ran the unmodified strict MLX fixture with all required weight and
metadata hashes. It returned **131/132** with the original mismatch at token 61
(expected 14988, got 10000). Bench emitted its strict error. The separate
non-regression checker accepted only that unchanged mismatch record. The
strict command's zsh wrapper tried to assign to the reserved `status` variable
after Bench completed, so it did not record a separate exit-status file.
The strict log preserves the error and the Bench catch exits 1. No repeat hash
or replay was needed. The reference stores expected token IDs rather than a
full matched logit trace. The unchanged result cannot qualify the prototype's
changed logits. See [strict output](candidate-strict-mlx.txt) and
[known-issue check](candidate-known-issue-check.txt).

## Reproduce the bounded experiment

Base `c1ff7c907309b81e67f9b03925b127bdf48f323b`. Host Mac17,2, M5 with
10 GPU cores and 32 GiB unified RAM, macOS 27.2.1 (26B131), Swift 6.4.2.
AC power, raw Safetensors, local compatible Qwen3.6-35B-A3B-4bit checkpoint,
normal pressure, one model owner. Existing MLX fixture and S3-00 artifacts
were read-only. The first diagnostic attempt had a missing zero offset
argument in the cloned Metal projection call. Runtime shader compile rejected
it before any forwards. See [failed compile](initial-metal-compile-failure.txt).
The corrected diagnostic took 190.121 seconds. See [test log](probe.txt).

Apply the [archived diagnostic patch](residency-probe.patch.gz) to an isolated
checkout at the base and provide this evidence directory there. Use a normal
pressure preflight and one model owner. The archived version contains one extra
opt-in for the strict benchmark and timings. With that variable unset it runs
the same forced diagnostic. See [source hashes](experiment-identity.json).

```bash
gzip -dc docs/evidence/2026-11-18-s300a/residency-probe.patch.gz | git apply
Scripts/test.sh -c release --filter ExpertResidencyProbeTests
QWENFIELDFARE_PREFIX_CHECKPOINT=/absolute/model-directory \
QWENFIELDFARE_EXPERT_RESIDENCY_PROBE=1 \
QWENFIELDFARE_RESIDENCY_OUT=/absolute/output/comparison.json \
  Scripts/test.sh -c release --skip-build --filter ExpertResidencyProbeTests

swift build -c release --product QwenFieldfareBench
QWENFIELDFARE_PROBE_CANONICAL_REDUCTION=1 .build/release/QwenFieldfareBench \
  --model /absolute/model-directory --context 8192 --memory-gib 5 \
  --warmup-runs 0 --runs 1 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
python3 docs/evidence/2026-11-18-s300a/benchmark.py \
  /absolute/model-directory .build/release/QwenFieldfareBench /absolute/output/benchmark
```

For the summed GPU measurement, apply the profiling patch after the main
patch, rebuild the Bench product, and run the same 128-output command from
the performance records with the extra `--gpu-profile` flag in each mode.

The experiment ended with **Outcome B, characterized**. The deterministic
policy is proposed in the [numerical contract](../../NUMERICAL_CONTRACT.md).
The experimental reduction is archived, and production uses the existing
replay path. Before considering a production correction, compare a small
checkpoint-verified MLX logit/hidden-state sample at specified teacher
positions, including near ties and output 465, with the precision and
phase conventions pinned. Review deviations with both the legacy and
canonical results. After a canonical runtime is accepted, requalify
conversation reuse independently against canonical full replay.


## Restored tree validation

Production Sources, Tests, manifest and pins match `c1ff7c9` byte for byte.
The focused model probe passed in the temporary build.
On the restored source, `Scripts/test.sh` passed 153 tests with three
checkpoint-only skips. Release build passed in 40.00 seconds. Numerical
checker tests passed 6/6 and the historical mismatch record and original
reference SHA-256 were verified. Markdown and Git diff checks passed.
Pressure level was 1 at postflight, with 81% free memory, 271.69 MiB swap
matching preflight, and no remaining model owner.

[Package suite](restored-package-tests.txt),
[release build](restored-release-build.txt), [checker tests](checker-tests.txt),
[archived mismatch check](archived-known-issue-check.txt),
[documentation check](docs-check.txt), [postflight](postflight.txt).
The working changes are documentation and evidence. The production correction
was not retained, so the investigation remains uncommitted.
