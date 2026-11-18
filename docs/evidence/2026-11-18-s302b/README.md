# S3-02b exact sampled-selection bitmap

Control `f340b5718dd23687ca08044f839e23a111dd3ddf` has the same inference
sources as the verified stable tip `2e5b993`. The comparison changes only
active-streaming generation membership. The shared selector retains its Set
entry point, exact-greedy vDSP path, score traversal, signed penalties,
minimum-candidate replacement, comparison, sorting, weights, Top-P and draw.

## Decision A — promote on the feature branch

The five primary pairs averaged **11.82% less decode latency (95% paired
CI 6.60–17.05%)**. Both 512-output confirmations cleared 5%. Candidate
commit `5d687b1` contains the two runtime edits and the selector tests.
It is eligible for separate integration review. The strict MLX gate stays
at 131/132.

| Generated outputs | Pair order | Set mean ms/token | Bitmap mean ms/token | Reduction |
| --- | --- | ---: | ---: | ---: |
| 128 | Set → bitmap | 36.090 | 29.108 | 19.35% |
| 128 | Bitmap → Set | 32.140 | 28.917 | 10.03% |
| 128 | Set → bitmap | 32.117 | 29.000 | 9.71% |
| 128 | Bitmap → Set | 32.128 | 28.875 | 10.13% |
| 128 | Set → bitmap | 32.022 | 28.851 | 9.90% |
| 512 | Set → bitmap | 37.141 | 33.909 | 8.70% |
| 512 | Bitmap → Set | 37.620 | 34.237 | 8.99% |

The first 128 pair showed the largest change and remains in the estimate.
OS page-cache state was uncontrolled. The two sustained confirmations
have the same output history as their respective Set controls. Their mean
paired reduction is 8.85%; each saw 9.53% or 9.88% higher throughput.

| Workload and variant | Mean / p50 / p95 transition ms | Decode tok/s | Selector CPU-wall ms/token | Selector share |
| --- | ---: | ---: | ---: | ---: |
| 128 Set | 32.899 / 31.896 / 38.663 | 30.40 | 3.107 | 9.44% |
| 128 bitmap | 28.950 / 28.435 / 33.467 | 34.54 | 0.210 | 0.73% |
| 512 Set | 37.380 / 36.115 / 48.094 | 26.75 | 3.267 | 8.74% |
| 512 bitmap | 34.073 / 32.860 / 44.071 | 29.35 | 0.211 | 0.62% |

| Workload and variant | Prefill / TTFT / request (s) | Request expert hits / misses | Logical bytes fetched |
| --- | ---: | ---: | ---: |
| 128 Set | 1.927 / 1.930 / 6.108 | 48,917 / 12,523 | 22,159,097,856 |
| 128 bitmap | 1.891 / 1.893 / 5.569 | 48,917 / 12,523 | 22,159,097,856 |
| 512 Set | 1.951 / 1.953 / 21.055 | 141,178 / 43,142 | 76,338,561,024 |
| 512 bitmap | 2.028 / 2.030 / 19.442 | 141,178 / 43,142 | 76,338,561,024 |

Every headline run allocated 5,291,360,256 Metal bytes (4.928 GiB) and
remained at normal pressure. The maximum observed headline footprint was
4.821 GiB for each variant at the reported precision. Sampled swap across
headline runs ranged from 4,143.81 to 4,159.81 MiB. The CPU bitmap adds
31,040 logical CPU bytes per sampled request. Logical fetched bytes
describe cache demand. Physical SSD traffic was unmeasured.

[Individual timings and resource records](timing.json),
[paired effect and confidence interval](effect.json), and
[frozen release executable hashes](timing-binaries.json).


## Bitset ownership

One sequence-local array holds 3,880 UInt64 words for the existing 248,320
logits, a **31,040-byte logical payload** plus the fixed object, array header
and allocator rounding. An insertion validates the ID before indexing the
array. The scan borrows its storage once. The selector's loop range comes
from the same vocabulary count as the bitmap, so the word index is bounded
by construction. Every prompt ID initializes membership once, including IDs
covered by exact-prefix reuse. A non-EOS generated ID is marked at its
existing point after the callback. Stop and context/output limits retain
their observed consumed-versus-pending boundaries. Only the request owns the
bitmap. Reset, cancellation and model replacement leave no membership for a
subsequent request. Exact greedy skips the bitmap. Explicit Set-based direct
forwards and fallback generation keep their existing APIs.

## Exact equivalence

The original Set selector passed its pinned cases before the implementation
was edited. The focused model-free matrix covers empty, repeated, and
near/full checkpoint-vocabulary membership, first and last indices,
out-of-range input, signed and zero penalties, signed logits and zeroes,
ties, NaNs and infinities, temperature zero and nonzero, Top-K 1/20/256,
Top-P boundaries, multiple seed and position choices, and repeated requests.
The shared loop comparison observes callback order, positions, EOS,
output/context limits, and cancellation.

Both release harnesses compared raw IDs before discarding them, and compared
finish reasons, exact expert-access digests by layer, logical read demand,
and whole hidden/logit digests at six sampled output positions on the
512-output case. The default sampled tests use temperature 1, Top-K 20,
Top-P 0.95, presence 1.5, repetition 1 and seed 42. The tool fixture uses
temperature 1 and Top-K 1 with zero presence penalty to hold its known
canonical tool result while exercising the bitmap path. It produces
`audit_chunk(index: 1)` and reuses 399 IDs of the 430-ID follow-up, with
31 fresh IDs. A separately published request followed by different content
replays all 17 prompt IDs. Cancellation inside the first delivery invalidates
the ledger and a subsequent reset request matches replay. Native tool parsing
runs in the harness, with the public tool response supplied by the fixture.

[Exact matching records](correctness.json) contain the public benchmark's
65 prompt IDs, hashes for outputs and tool prompts, and resource readings.
The test compares raw outputs locally and strips them before publication.
It records no private prompts or logit arrays.

## Performance protocol

The MacBook Pro Mac17,2 has an M5 with 10 GPU cores, 32 GiB RAM, macOS
27.2.1 (26B131), and Swift 6.4.2. Both builds use the same raw four-shard
checkpoint, 8,192 context, the 5 GiB planner with 2,183 slots, AC power and
low-power mode off. Every run starts in a fresh process, with an empty
expert cache and the same fixed default-sampled warmup. The OS file cache
remains uncontrolled. The driver checks checkpoint file metadata, exclusive
model ownership, normal pressure and swap before and after every process.

Five 128-output pairs and two 512-output confirmation pairs reverse
Set/bitmap order across pairs. Headline samples record monotonic delivery
intervals, existing selector CPU-wall telemetry, and periodic footprint.
They omit expert routing traces, GPU profiling and hidden/logit hashing;
selector CPU-wall timing covers a synchronous CPU-only section without GPU
wait. Each run's first token belongs to prefill. Decode uses 127 or 511
transitions. Report all paired latency reductions as
`100 * (Set elapsed - bitmap elapsed) / Set elapsed`. The primary 95%
interval is the two-sided paired Student t interval over the five relative
effects. The promotion threshold is 5% end-to-end decode improvement and a
positive 95% paired interval. A separate diagnostic uses actual thread CPU
clocks on matched synthetic logits, and the standard Bench GPU diagnostic
collects bucket timings and resources outside headline measurement.

## Strict reference and separate diagnostics

The candidate's checkpoint-verified strict replay returned exit 1,
131/132, with only token 61 expected 14988 and actual 10000. The numerical
checker accepted that exact known-only record. The original fixture is
byte-identical to the control. [Strict replay](strict.txt)

The matched-logit microbenchmark used 300 seen IDs, 248,320 scores and
five rounds of 512 positions. It matched the selected IDs at every
position and measured **3.337 ms Set versus 0.191 ms bitmap** of thread
CPU per selection. [Thread CPU diagnostic](thread-cpu.json)

The paired 512-output GPU profile runs use the same released paths with
profiling enabled, outside the promotion statistic. They record 64-output
bucket speed, expert-miss/logical-byte averages, host timings, exact GPU
timings, footprint, Metal and before/after pressure and swap.
[GPU profiling and system samples](diagnostics.json)

## Reproduction

Prepare two clean checkouts. Use `f340b57` for the control and production commit `5d687b1` for the candidate. Set the variables to absolute paths.

```bash
evidence=/absolute/path/to/docs/evidence/2026-11-18-s302b
baseline=/absolute/path/to/set-checkout
candidate=/absolute/path/to/bitmap-checkout
model=/absolute/path/to/Qwen3.6-35B-A3B-4bit
out=/absolute/path/to/temporary-results

for checkout in "$baseline" "$candidate"; do
  cp "$evidence/BitmapModelQualification.swift" "$checkout/Tests/QwenFieldfareServerTests/"
  (cd "$checkout" && swift build -c release &&
    env -u QWEN_BITMAP_CHECKPOINT swift test -c release --no-parallel --filter BitmapModelQualification)
done
for phase in correctness timing; do
  python3 -B "$evidence/qualify.py" --set "$baseline" --bitmap "$candidate" \
    --model "$model" --out "$out" --phase "$phase"
done
python3 -B "$evidence/diagnostics.py" --set "$baseline" --bitmap "$candidate" \
  --model "$model" --out "$out"
python3 -B "$evidence/measure_selector.py" --set "$baseline" --bitmap "$candidate"
```

The harness is archived here and was copied into each release test target
temporarily. It is separate from the production selector and build.
