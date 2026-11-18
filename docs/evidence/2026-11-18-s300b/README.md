# S3-00b deterministic expert reduction qualification

Production baseline `c1ff7c9`, with S3-00a published as `c85edac`. The comparison
uses native local MLX and one release Swift probe containing the archived
router-order implementation. The two models run sequentially, with one owner
at a time. The probe is an evidence artifact and does not change the production
reduction or conversation-state path.

## Decision

**Outcome B — Revise.** Router-order accumulation solves the residency problem
on both tested executions. Average numerical fidelity is similar to legacy.
The candidate still locks in a wrong signed margin at the sustained divergence
where cold legacy agrees with both MLX schedules. That ranking trade-off is
reproducible and relevant to the handoff, despite the small change in average
error. Keep the existing production reduction while testing one targeted
combine variant. This decision does not require a speed improvement.

## Matched results against native MLX

Each mean is across the seven selected positions for that history. Max absolute
error is the maximum across all vocabulary entries and selected positions.
The samples deliberately include hard positions and are not a random estimate
of fidelity. Router cold and warm logits and hidden captures match exactly.

| History | Swift mode | Mean logit RMSE | Max abs | Mean logit relative L2 | Mean hidden relative L2 | Top 1 / 7 | Top-5 members / 5 | Ordered top 5 / 7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | legacy-cold | 0.06462 | 0.77081 | 2.299% | 3.126% | 6 | 5.00 | 4 |
| strict | legacy-warm | 0.06459 | 0.77339 | 2.298% | 3.127% | 6 | 5.00 | 4 |
| strict | router order (both) | 0.06467 | 0.77886 | 2.300% | 3.133% | 6 | 5.00 | 4 |
| sustained | legacy-cold | 0.18104 | 2.43519 | 7.571% | 8.846% | 6 | 4.86 | 4 |
| sustained | legacy-warm | 0.18588 | 2.58557 | 7.780% | 9.024% | 5 | 4.71 | 1 |
| sustained | router order (both) | 0.18197 | 2.62317 | 7.613% | 8.897% | 5 | 4.86 | 3 |

The [full metrics](comparison.json) include per-sample FP64 error metrics,
logit distribution quantiles, top-five scores and exact tied margins, plus
aligned layer-0 expert route scores and overlap for every comparison. Against
tokenwise MLX, mean strict logit RMSE is 0.06987 cold/warm versus 0.07038
router order. Sustained is 0.16623 cold, 0.16383 warm and 0.16755 router
order. This is a modest loss relative to the alternate schedule, alongside
similar mean error against native MLX. BF16 logits quantize some margins to
zero or multiples of 0.125. Raw rank and signed margins therefore give context
that full-vocabulary RMSE by itself cannot supply.

| Sample | Scored pair | MLX native | MLX tokenwise | Legacy cold | Legacy warm | Router order |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Strict 80 | 14988 minus 10000 | +0.12500 | −0.12500 | −0.12049 | −0.11462 | −0.10266 |
| Strict 82 | MLX native top pair | 0 | −0.12500 | +0.07444 | +0.06749 | +0.07501 |
| Strict 123 | MLX native top pair | +0.12500 | +0.62500 | +0.16291 | +0.16102 | +0.16011 |
| Sustained 645 | MLX native top pair | 0 | 0 | +0.05796 | +0.04947 | +0.05772 |
| Sustained 850 | MLX native top pair | +0.12500 | 0 | −0.19220 | −0.20279 | −0.18171 |
| Sustained 894 | 1031 minus 285 | −0.12500 | −0.12500 | −0.01248 | +0.01474 | +0.01194 |

Native MLX reproduced 132/132 argmax IDs on the original fixture. Changing
only its prefill schedule to sequential produced 129/132, with mismatches
at 33, 61 and 63 (one-based). All four Swift runs reproduced the unchanged
131/132 result, only token 61 expected 14988 and actual 10000. At strict 61,
the deterministic sum moves the signed 14988–10000 gap slightly toward the
native reference without changing the ranking. The tokenwise MLX diagnostic
explains sensitivity to scheduling and has no authority to replace that
reference. Sustained 850 also changes ranking under MLX schedule or tie
handling. Sustained 894 is a stronger discriminator: both MLX schedules
select 285, cold legacy selects 285, and warm legacy and router order select
1031. Native relative L2 there is 6.083% cold, 6.517% warm and 6.180% router
order. Router order improves the magnitude error relative to warm while
preserving its ranking error.

## Residency and resource findings

Router order is bitwise equal at all 152 strict and 941 sustained logit
positions across the two tested Swift histories. The 14 sampled points
also match the pre-RMSNorm hidden bytes across router-order histories.
The focused test also requires identical layer-0 IDs, weights, input,
activations, per-slot checkpoint record hashes, weighted contribution hashes,
and layer output hashes at those 14 points. The observed layer-0 masks differ
at four of seven strict samples and two of seven sustained samples. Legacy
full-logit hashes differ at every position across the warm and cold controls.
The canonical execution uses the existing single queue and slot ownership.
It writes the eight weighted results to disjoint router positions and runs one
fixed-order combine after both producers. It makes grouping immaterial at
matched inputs with this executable and device. The evidence covers these
histories and sampled layer values. It does not establish across-device bitwise
determinism or prove the absence of every synchronization fault.

The M5/32 GiB run uses Swift 6.4.2 and macOS 27.2.1. MLX 0.32.4 and mlx-lm
0.32.2 were verified against the fixture. MLX peak allocation was 20,272,756,008
bytes and Swift Metal rose from 5,291,294,720 to 5,291,360,256 bytes with the
existing 64 KiB scratch. Pressure checks stayed at level 1, but system-wide
swap rose from 271.69 MiB before the run to 3,067.69 MiB after MLX, with
2,899.62 MiB after Swift. These observations are system-wide, not a per-process
peak or a claim of swap-free inference. The main performance assessment uses
the already-published alternating measurements: −2.76% mean short decode rate
(95% interval −7.58% to +2.07%) and −0.48% at 512 across two pairs. The
instrumented qualification is not a new throughput benchmark.
See [S3-00a performance](../2026-11-18-s300a/README.md).

## Exact next step

Test one diagnostic router-order combine that preserves the eight FP32 weighted
contributions and fixed routed sum, then adds the shared expert before applying
the attention residual, using the explicit grouping `residual + (routed + shared)`.
This tests the grouping visible in the native MLX model. Keep reassociation from
erasing the proposed distinction and prove the isolated arithmetic with a
matched contribution fixture. This is a targeted hypothesis, including the
possibility that the changed grouping has no meaningful effect on final logits.
Reuse the saved MLX captures and the existing strict fixture. Require exact
warm/cold output on both Swift histories, the current strict mismatch ceiling,
and review the two historical margins, near ties, and aggregate errors against
both current Swift controls. If this variant fails to improve the trade-off,
stop changing the combine and review whether accepting the original deterministic
path is warranted. Only after a reduction is accepted should S3-00 compare
prefix reuse independently with its canonical full-replay control.

## Histories and sampling

| History | Prompt | Teacher outputs | Input positions | Historical boundary |
| --- | ---: | ---: | ---: | --- |
| Strict MLX reference | 21 | 132 | 152 | Input 80 predicts reference token 61 (one-based) |
| Sustained S3-00 continuation | 430 | 512 | 941 | Input 894 predicts output 465 (zero-based) |

Each run consumes all prompt tokens and the teacher outputs except the last.
No prediction from the runtime is fed back. The strict fixture comes from a
HashMap explanation. The sustained input comes from the separate staged tool
and LRU-cache workload. Its 512 teacher IDs are the archived warm-legacy Swift
output. Agreement with those IDs measures trajectory agreement, not MLX accuracy.
The exact reconstructed inputs are stored in [histories](histories.json).
The original strict fixture remains the independent benchmark reference.

MLX verifies all ten original checkpoint file hashes and the fixture version
pins before running. The primary MLX execution batches the first 20 or 429
prompt tokens, respectively, then uses single-token forwards, following native
`generate_step` schedule. The separate sequential control uses the same native
modules but processes all tokens individually. Swift always processes one token
per forward. MLX uses native BF16 hidden states and logits. The raw captures
convert those values to FP32 for storage and to FP64 for metrics, without
adding precision. Swift captures its original FP32 values. Hidden states are
the final layer's residual stream immediately before final RMS normalization.
Taps pass the actual normalization and router values through unchanged.
Raw logits are captured before softmax, penalties or sampling.

The capture takes positions 0 and 1, the first output's prefill boundary, the
historical divergence, and the final position. It also takes the earliest
minimum native MLX top-two margin in each decode half, excluding the fixed
samples. Strict picks positions 82 and 123. Sustained picks 645 and 850.
Ranking sorts descending by score with ascending token ID for exact ties.
Signed margins score the same two reference candidates in each mode. At the
two historical boundaries, the original competing IDs are separately reported
even if they are not that mode's highest scores. Layer-0 routing compares
expert IDs as a set and aligns scores by expert ID.

For strict, Swift cold clears the expert cache and warm replays that same
fixed trajectory once before resetting model state. For sustained, both start
with the same 17-token warmup. Warm also retains the original 355+44 tool
history's expert weights. Both reset all model state before the continuation.
One 8K-context/5 GiB/2,183-slot Swift runner uses both paths. No cache policy
change or conversation reuse is involved.

## Reproduction

Read-only reference input and archived prototype are prerequisites. Apply the
[Swift probe patch](reference-probe.patch.gz) in a disposable checkout at
`c85edac`. Build the release test and use a writable temporary capture directory.
Run in order, with normal memory pressure and a single model owner. Raw capture
files stay local. Published JSON has derived metrics, small metadata, and array
hashes.

```bash
gzip -dc docs/evidence/2026-11-18-s300b/reference-probe.patch.gz | git apply
Scripts/test.sh -c release --filter ExpertResidencyProbeTests
export QWENFIELDFARE_PREFIX_CHECKPOINT=/absolute/model-directory
export QWENFIELDFARE_RESIDENCY_OUT=/absolute/temporary-capture-directory
export QWENFIELDFARE_EXPERT_RESIDENCY_PROBE=1
mkdir -p "$QWENFIELDFARE_RESIDENCY_OUT"
QWENFIELDFARE_RESIDENCY_INPUTS_ONLY=1 Scripts/test.sh -c release --skip-build --filter ExpertResidencyProbeTests
/absolute/mlx-venv/bin/python docs/evidence/2026-11-18-s300b/capture_mlx.py \
  "$QWENFIELDFARE_PREFIX_CHECKPOINT" docs/evidence/2026-11-17-parity/reference.json \
  "$QWENFIELDFARE_RESIDENCY_OUT"
Scripts/test.sh -c release --skip-build --filter ExpertResidencyProbeTests
/absolute/mlx-venv/bin/python docs/evidence/2026-11-18-s300b/compare.py \
  "$QWENFIELDFARE_RESIDENCY_OUT" docs/evidence/2026-11-18-s300b/comparison.json
```

## Validation

The [focused Swift test](swift-run.txt) passed in 180.388 seconds.
The restored tree passed 153 package tests with three opt-in skips and built
in release mode in 42.54 seconds. All six numerical checker tests passed.
The archived strict mismatch check confirms the same original fixture and
failure, independently of the fresh Swift capture. Sources, tests, manifest,
dependency pins and the strict reference files match `c1ff7c9`.
Markdown links and whitespace checks passed. [Validation summary](validation.json)
