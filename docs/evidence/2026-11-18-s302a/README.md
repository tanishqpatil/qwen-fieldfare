# S3-02a sampled selector measurement

The source base is `2e5b9934a5a96b18d352f58e0fa4201319fb12e2`, the verified
stable server tip when this work began. The chunked-prefill research closed
on its separate branch at `e6ac9c3`. This study uses the established
tokenwise runtime, fixed router-order reduction, and default-off prefix reuse.

## Decision A — qualify one exact membership replacement

The default sampled selector accounts for **8.50% of sustained inter-token
latency** over 512 outputs. Replacing nonempty Set hashing with one dense,
sequence-local token bitmap is the smallest justified next experiment.
The existing score traversal, penalties, Top-K minimum replacement and
ties, NaN handling, weights, Top-P and draw stay in their original order.
No candidate was implemented.

The bitmap needs 31,040 logical bytes for 248,320 IDs, plus allocation
rounding. Set the same prompt/generated IDs at the same lifecycle points
and preserve existing cancellation, reset, context and prefix behavior.
Keep exact greedy on the existing vDSP path. Run exact matched-logit tests
including ties, NaNs, infinities, penalty combinations and sampling
positions. Require identical seeded model outputs, finish reasons and cache
demand before the sustained paired A/B. The existing promotion gate needs
at least 5% end-to-end improvement and a positive paired 95% confidence interval.

## Real-model results

Three opt-in runs per 128-output row and four per 512-output row. Selector
share is total selector elapsed time divided by total first-to-last
inter-token time. Percentiles use pooled
transitions, with median p50 and nearest-rank p95. Times are milliseconds.

| Configuration | Inter-token mean / p50 / p95 | Selector thread CPU mean | Selector elapsed mean / p50 / p95 / max | Elapsed share |
| --- | ---: | ---: | ---: | ---: |
| App defaults, 128 outputs | 32.499 / 31.919 / 36.632 | 3.199 | 3.202 / 3.143 / 3.912 / 5.411 | 9.85% |
| App defaults, 512 outputs | 36.767 / 35.636 / 46.440 | 3.124 | 3.127 / 3.118 / 3.931 / 6.568 | 8.50% |
| Exact greedy, 128 outputs | 29.702 / 29.026 / 34.722 | 0.028 | 0.029 / 0.032 / 0.035 / 0.045 | 0.098% |
| Exact greedy, 512 outputs | 33.617 / 32.490 / 43.042 | 0.030 | 0.031 / 0.032 / 0.034 / 0.048 | 0.091% |

The per-run sampled 512 selector share ranged from 8.27% to 8.71%. The
worst sampled 512 elapsed selector step was 6.568 ms. The highest
thread CPU step was 5.553 ms. Sampled throughput and greedy throughput use
different output histories. At 512 the sampled history had 38,817 decode
misses while greedy had 39,876, so their wall-time difference cannot be
assigned to selection.

| Configuration | Prefill / TTFT (s) | Decode / request (s) | Decode hits / misses | Logical decode bytes |
| --- | ---: | ---: | ---: | ---: |
| App defaults, 128 | 2.163 / 2.165 | 4.127 / 6.292 | 32,463 / 8,177 | 14,468,972,544 |
| App defaults, 512 | 2.308 / 2.310 | 18.788 / 21.098 | 124,703 / 38,817 | 68,685,594,624 |
| Exact greedy, 128 | 2.120 / 2.122 | 3.772 / 5.894 | 32,875 / 7,765 | 13,739,950,080 |
| Exact greedy, 512 | 2.284 / 2.286 | 17.178 / 19.464 | 123,644 / 39,876 | 70,559,465,472 |

All 28 processes emitted the requested output length and ended at the
output limit. On/off matched every within-sampler output SHA-256 and
cache count. For sampled 512, profiler-on decode time differed from
the paired control by −0.88%, +0.70%, −1.17%, and −2.30% in the reversed pair. Short pairs differed by
as much as 11.38%, so the sustained comparison is the primary measurement.
The direct clock attribution avoids confusing run-wide host variance with
the selector. The paired controls do not establish a precise overhead
percentage or satisfy candidate promotion requirements.

Metal allocation was 5,291,360,256 bytes (4.928 GiB) throughout. The
maximum sampled process footprint ranged from 4.793 to 4.811 GiB, with
periodic rather than continuous sampling. Pressure was normal at every
pre/post sample. System free-memory percentages were 85–87% before and
81–84% after. Existing swap fell from 4,443.69 to 4,263.81 MiB across the
series. No individual run showed an increase at its sampled boundaries.

See all 28 anonymized [timing records](runs.jsonl.gz) and the
[deterministically derived statistics](summary.json).

## Cost attribution and possible return

At 512 sampled outputs, the vocabulary scan consumed **3.124 ms** per
transition. Sorting, weights, Top-P, and draw together averaged about
**1.0 µs**. The scan tests membership and the retain threshold
248,320 times. It performed a mean of 87.1 minimum candidate rescans
(p50 84, p95 141, max 163), each checking the remaining 19 candidates.
Set construction averaged 18.3 µs at the final prompt forward and
1.1 µs at decode initialization. A per-step seen insertion averaged
0.04 µs. Shared buffer pointer access averaged 0.19 µs and the preceding
benchmark callback 0.75 µs. Mean expert lookup was 0.093 ms, expert miss
service 10.050 ms, and all host GPU commit/wait 22.944 ms. The final
command ran for 2.183 ms on the GPU within an existing 2.378 ms host
wait. These values are diagnostic, with overlap described below.

The matched synthetic fixture varied only the seen Set, or Top-K with
an empty Set, while keeping the vocabulary, logits, seeded draw and
selected IDs identical. The 300 seen IDs have deliberately low logits,
isolating routine membership/penalty work while retaining the same
candidate work. Across six rotated rounds of 128 selections, mean
thread CPU time was 3.442 ms with the populated Set, 0.184 ms with an
empty Set at the same 182 minimum rescans, and 0.173 ms with an empty
Set and Top-K 1. This localizes the large cost to populated Set
membership and its dependent penalty branch. The empty Set is a
diagnostic estimate of the work that remains without populated hashing.
Correct sampled generation still needs membership checks. The synthetic
CPU difference cannot be subtracted directly from the real run.
[Raw synthetic timing](synthetic.json)

Eliminating the entire real selector would reduce sustained inter-token
time by at most 8.50% at this operating point, or increase decode
throughput by 9.29%. A 5% latency reduction requires removing 1.838 ms per
token, or 58.8% of current selector elapsed time. Retaining the scan and
sampling work, a bitmap that recovers 60–75% of real selector time would
save 1.88–2.35 ms per token, a conditional **5.10–6.38%** decode reduction.
That allows roughly 0.60–1.07 ms of membership work above the synthetic
no-populated-Set floor. This is a budget for a candidate, with no bitmap
timings yet. The same budget equates to about 4.5–5.7% for this complete
512-output request including prefill. Longer prompts or faster storage
can change the observed shares.


## Execution and attribution

The final layer encodes FP32 RMSNorm and the Q4 language-model head in its
existing tail command. The head writes 248,320 Float32 scores directly into
a Metal shared buffer. The host already waits for that command. The selector
uses the buffer pointer with no staging, materialization, or copy. CPU
first-touch costs remain in the scan. The probe adds no GPU synchronization.

Exact greedy guards the first-logit NaN case and uses vDSP's first maximum,
bypassing the general candidate and penalty loop. App-default sampling uses
temperature 1.0, Top-K 20, Top-P 0.95, presence penalty 1.5 and repetition
penalty 1.0. The real UI uses an unspecified seed. The benchmark supplies
42 and retains the existing seed/position calculation. It scans all scores,
checks Set membership once per ID, skips repetition arithmetic at a
penalty of one, and subtracts the presence penalty for seen IDs. It keeps
20 candidates, scans those candidates for a new minimum when necessary,
sorts by score and token ID, builds Double exponential weights, applies
Top-P, then uses a seeded draw. No sampling or Metal operations were changed.

The probe brackets the whole vocabulary scan (including candidate reservation,
membership, penalty, Top-K comparison and any retained-minimum scans).
A single counter counts retained-minimum rescans. No clock runs inside the
vocabulary loop. Separate brackets measure candidate sorting, exponential
weights plus their total, Top-P filtering, and the retained total/random draw.
A monotonic wall clock measures selector elapsed time and actual callback
intervals. A per-thread CPU clock brackets the selector on the same
synchronous thread. The clock calls and counters are included in the observed
probe overhead, and the whole-selector boundaries include small timing costs.
No malloc interposition or exact allocator attribution was used. The nominal
Top-20 candidate and weight data occupy 160 + 160 bytes per call, subject
to actual allocator rounding and sort internals. Seen-set rehashes and
initial construction are outside selector time.

The first generated output comes from the last prefill forward. Decode
statistics use the 127 or 511 first-to-last output transitions and the
selector associated with each later delivery. They exclude the first
output's selector and the last output callback. Runtime callback time in
each transition belongs to the preceding output, including benchmark metrics
and occasional footprint snapshots. The decode-initial seen-set build is
reported separately because it happens before the first delivery. These are
native-sampler semantics exercised in the CLI. There was no AppKit rendering,
chat publication or SSE/network delivery in the test. Total request time
covers `generateResult` including control flow and terminal handling but
excludes model load, CLI formatting/tokenization, and post-run output hashing.
A separate process wall clock includes setup and teardown. The existing
prefill clock and monotonic start-to-first-callback time are both retained.

Final-tail GPU execution is read from the completed Metal command's timestamps
and includes the tail, RMSNorm and LM head. Whole-forward host GPU commit/wait
and host expert lookup/miss-read times use the existing runner counters.
These are overlapping attribution signals. Summing them would double count
overlap. Total pure GPU execution and isolated LM-head GPU time were not
collected. Logical miss bytes do not measure SSD traffic.

## Reproduction

Apple MacBook Pro Mac17,2 with M5 (10 GPU cores), 32 GiB unified memory, macOS
27.2.1 (26B131) and Swift 6.4.2. The selected 8,192-context 5 GiB plan uses
2,183 expert slots, raw Safetensors, one model owner, AC power and
low-power mode off. Pressure and swap were checked before and after every
process.
The probe ran on the same local compatible checkpoint used for the accepted
baseline. The four shard sizes, stable release and probe binary SHA-256
and source/patch identity are in [provenance.json](provenance.json). The
driver verified stable device/inode/size/mtime metadata throughout the
profile. It did not rehash the weight payload or rerun the strict fixture.

Use a disposable checkout and keep build outputs outside the published evidence. Replace the variables with absolute paths.

```bash
evidence=/absolute/path/to/docs/evidence/2026-11-18-s302a
checkout=/absolute/path/to/disposable-source
model=/absolute/path/to/Qwen3.6-35B-A3B-4bit

mkdir -p "$checkout"
git archive 2e5b9934a5a96b18d352f58e0fa4201319fb12e2 | tar -x -C "$checkout"
gzip -dc "$evidence/selector-probe.patch.gz" | (cd "$checkout" && git apply --check -)
gzip -dc "$evidence/selector-probe.patch.gz" | (cd "$checkout" && git apply -)
(cd "$checkout" &&
  Scripts/test.sh --filter 'QwenGenerationSupportTests|QwenSelectorProbeTests' &&
  swift build -c release)
python3 "$evidence/run_profile.py" --executable "$checkout/.build/release/QwenFieldfareBench" \
  --model "$model" --output /tmp/s302a-runs.jsonl
python3 "$evidence/run_synthetic.py" "$checkout"
python3 "$evidence/run_profile.py" --executable "$checkout/.build/release/QwenFieldfareBench" \
  --model "$model" --output /tmp/s302a-reversed.jsonl --reverse-sustained-only
cat /tmp/s302a-runs.jsonl /tmp/s302a-reversed.jsonl > /tmp/s302a-all.jsonl
python3 "$evidence/summarize.py" /tmp/s302a-all.jsonl
python3 -B "$evidence/test_summarize.py"
```

The driver embeds the public synthetic Java benchmark input, derived from
the community HashMap prompt with a longer-answer instruction to sustain
512 outputs. The records include its length, without storing prompts,
generated text, raw token histories or logits. One fresh process per run
starts with an empty expert cache. The OS file cache was neither reset nor
controlled. Off controls use the same probe executable with the detailed
observer disabled. The controls keep existing benchmark timings and the
common monotonic delivery clock/output digest. The first three pairs kept
the same activation order within each workload despite rotating workload
order. One additional 512-output pair per sampler reversed that order to
check for directional bias. The 128-output pairs are secondary. These are
diagnostic observations. No background compile or alternate model owner
ran alongside the capture.

## Preserved correctness gates

The matched model-free selector probes preserve selected IDs on the same
synthetic logits, seen set and seed, and the loop probe checks unchanged
seen-token histories, sample positions, callbacks and terminal status.
The real controls compare deterministic output fingerprints, finish labels,
cache demand and Metal allocation within each sampler and output length.
Model output was never replaced or constrained. Production source, tests,
package files, dependency pins and the strict fixture are byte-identical to
the stable tip. The existing checkpoint-verified strict MLX result is
**131/132**, reference token 61 expected 14988, actual 10000. The archived
log checker reproduced that mismatch set with no new mismatch records.
It did not execute inference or certify full numerical equivalence.
