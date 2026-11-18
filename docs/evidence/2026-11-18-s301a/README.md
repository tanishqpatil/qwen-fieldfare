# S3-01a bounded chunked prefill qualification

Base `2e5b9934a5a96b18d352f58e0fa4201319fb12e2` on
`feature/bounded-chunked-prefill`. **Decision C — reject projection-only batching for promotion.** The
candidate passed numerical and lifecycle qualification. It missed the
20% fresh-prefill target and the tested suffix took longer. The branch
restores the published S3-00d runtime and retains the candidate as an
exact source patch. See the [execution contract](../../BOUNDED_PREFILL_CONTRACT.md).

The eight-row candidate batches input RMSNorm and Q4 projections
at all 40 layers. In a full chunk it replaces 800 such dispatches
with 100. There are two projections-stage dispatches per linear
layer and four per full-attention layer, including normalization.
It preserves one router readback and one ordered Conv/DeltaNet or
causal-attention step per token and layer. Router-selected expert
acquisition and the fixed top-eight reduction run in original
per-layer order. No temporary expert union is needed. Every prompt
token still runs the LM head, and decode uses the tokenwise schedule.

The plan charges 655,360 bytes for page-rounded scratch, with
624,640 bytes of requested buffers. It retains 2,183 expert slots
at 8K/5 GiB. The 16/32-row logical figures are accounting comparisons
only. The checkpoint and strict reference fixture are unmodified.

## Reproduction

From this evidence branch, apply the diagnostic patch in a disposable
checkout before building. Its gzip SHA-256 is
`16b23d43c2e40e5815f21be225d90628f808afeb0c06278eb9c45e7e7a21134d`.
The measured release executable had SHA-256
`2dec5c1b684f0442c24cbd57edeb542ae9cfc35c50aa011c703681e972ed819d`.
Use the same public fixtures and Mac configuration. The
[preflight record](preflight.txt) contains the hardware and toolchain.
Close competing processes, check normal memory pressure and use
one model owner. Use the compatible checkpoint in place.

```sh
gzip -dc docs/evidence/2026-11-18-s301a/candidate.patch.gz | git apply
Scripts/test.sh
swift build -c release
export QWENFIELDFARE_CHUNK_CHECKPOINT=/absolute/model
Scripts/test.sh -c release --filter QwenChunkedPrefillTests
.build/release/QwenFieldfareBench --model "$QWENFIELDFARE_CHUNK_CHECKPOINT" \
  --context 8192 --memory-gib 5 --prefill chunked8 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
QWENFIELDFARE_CHUNK_SUFFIX_BENCH=1 \
  Scripts/test.sh -c release --skip-build --filter measureFreshExactPrefixSuffix
python3 Scripts/benchmark_chunked_prefill.py --out /private/tmp/qwen-fresh-pairs.json
python3 Scripts/benchmark_chunked_prefill.py --profile --workloads 256 \
  --out /private/tmp/qwen-fresh-gpu-profile.json
python3 Scripts/test_numerical_non_regression.py
ruby Scripts/check_markdown_links.rb
git diff --check
```

The paired process runs start with empty expert caches. Initial order
alternates across sizes. OS page cache is uncontrolled. Prompt count,
first-token time, decode rate, expert counters and logical fetch bytes,
Metal allocation, process footprint and before/after system swap are
kept in the output. GPU profiling is a separate short diagnostic since
command timestamp collection changes the measurement.


## Numerical and lifecycle result

On the real checkpoint, prompts of 1, 2, 8, 9, 15, 16 and 17 tokens
matched first predictions, all sampled router selections and weights,
final hidden vectors and logits, and complete KV, Conv and DeltaNet state
digests. The 430-token sustained prompt matched all 512 generated IDs
and five sampled hidden/logit boundaries. The output ID digest and the
hidden/logit digests at output 465 also matched archived S3-00d.
The 32-position test exercised a 31-position EOS boundary, actual context
exhaustion, overflow, a genuine exact-prefix hit, cancellation after a
partial chunk, and reset/replay. The remainder of the server, client
protocols and default controls were outside the experiment.

The original strict teacher fixture was run with required checkpoint
verification in candidate mode. It still exits 1 with the only mismatch
at token 61, expected 14988 and got 10000. The separate mismatch-set checker
passes. The complete candidate package suite recorded 173 passes and
seven opt-in skips. The release build completed in 44.86 s. See the
[checkpoint tests](candidate-checkpoint.txt), [strict log](candidate-strict.txt)
and [validation](candidate-validation.txt).

## Exact-prefix suffix pair

Both runs retained 18 IDs and processed the same 89 fresh IDs, followed by
16 matching generated IDs. The exact-prefix continuation took 2.674 s
through fresh prefill with tokenwise scheduling and 3.160 s with eight rows.
Total generation wall time was 3.144 s and 3.691 s, respectively. Both runs
had 25,245 hits, 8,035 misses and 14,217,707,520 logical fetched bytes.
The profiling-free analysis runner reserved the candidate scratch for
both schedules and observed 5,291,982,848 bytes of Metal allocation.
See the [suffix record](candidate-suffix.txt).

## Profiled 214-token diagnostic

| Schedule | GPU command time | CPU encoding | CPU command waits | Prefill |
| --- | ---: | ---: | ---: | ---: |
| Tokenwise | 4,710.201 ms | 122.763 ms | 5,020.546 ms | 8.740 s |
| Eight-row | 4,474.762 ms | 127.060 ms | 4,767.700 ms | 8.817 s |

The instrumented pair has equal expert demand. GPU time and CPU waits
fell about 5%, but encoding and total profiled prefill did not improve.
The wait totals overlap GPU execution. This result points to the remaining
per-token router and expert handoff rather than a reduction in logical
weight traffic. See the [separate profile](gpu-profile.json).


## Fresh-prompt performance

| Prompt IDs | Tokenwise prefill | Eight rows | Prefill reduction | Decode tok/s, tokenwise / eight rows |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 9.301 s | 8.322 s | +10.53% | 21.80 / 21.77 |
| 1,010 | 38.578 s | 36.971 s | +4.17% | 19.15 / 19.06 |
| 4,022 | 166.414 s | 153.519 s | +7.75% | 17.76 / 18.36 |
| 6,128 | 228.058 s | 242.185 s | -6.19% | 24.80 / 19.24 |

Each workload uses one same-binary, cold expert-cache pair with 16
generated tokens. Nominal size labels are approximate. These are pilot
pairs, below the repetition count required for promotion. Expert hit and
miss counts, eviction counts and logical fetched bytes agreed in all four
pairs. Measured final Metal allocation was 4.928 GiB tokenwise and
4.929 GiB in the candidate. The largest observed final physical footprints
were 4.917 and 4.892 GiB respectively. Sampled pressure stayed at level 1.

Host swap rose from 937.12 to 7,007.94 MiB during the long eight-row
run. The following tokenwise control started with 7,007.94 MiB used.
This imbalance confounds attribution of the long-prompt and decode
differences. It does not explain away the shorter pairs missing 20%
or the separate suffix regression. See the [paired records](fresh-pairs.json)
and [postflight record](postflight.txt).

## What to try next

The next bounded experiment is one attention/router readback per layer and
chunk. Preserve per-row causal order, process expert requests in the same
per-layer order, and guard slot overwrite until prior GPU readers retire.
Measure synchronization and 1K TTFT before extending the experiment to long
prompts. The observed GPU profile and identical expert traffic motivate
this experiment. It remains outside S3-01a.

## Restored runtime validation

After reversing the exact patch, Core and other product source, tests,
scripts, package manifest and dependency pins matched `2e5b993`. The
restored package reports 171 passes and four opt-in skips, and release
build completed in 43.58 s. Existing S3-00d HTTP, SSE, OpenCode and native
qualification applies to the restored source.
See the [restored checks](restored-validation.txt).
