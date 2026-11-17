# Stage 0 parity investigation — 2026-11-17

## Outcome and classification

**D — unresolved upstream activation divergence.** A small-margin ranking
reversal is reproduced, but its full mechanism is not established. Evidence is
consistent with mixed BF16/FP32 execution differences; that is a hypothesis,
not proof that the discrepancy is harmless. No stale/invalid reference or
concrete Swift defect was demonstrated. No production fix or acceptance-policy
change is justified. Strict 132/132 remains required and **unsatisfied**.
Stage 0 task S0-03 stays blocked; Stage 1 has not started and is prohibited by
the current investigation request.

## Identity and equivalent inputs

Base commit `72fb923a3e7a5d43e5baeb43d35bdcf1f96e4966`, branch
`feature/local-coding-agent-server`, existing uncommitted foundation preserved.
MacBook Pro Mac17,2, Apple M5/32 GB; macOS 27.2.1 (26B131); Apple Swift 6.4.2
(swiftlang-6.4.2.1.9, clang-2100.4.1.1), arm64. Disk 134 GiB free initially.
No OS/toolchain/package/model changes during investigation. GPU core count not
recorded. Installed and saved reference versions agree: MLX 0.32.4, mlx-lm
0.32.2, Transformers 5.22.2. Historical versions differ, but no causal conclusion
about version drift is established. Relevant installed reference source hashes
are in [provenance](provenance.json); they identify local code, not a historical
source comparison or independently reproducible environment lock.

The [original format-2 fixture](../2026-11-17-parity/reference.json) is unchanged,
SHA-256 `8d552a906d417ceb492ddd045a50e67b0e9630988d9665e8c6fe5e480cb30747`.
It contains all ten checkpoint/metadata SHA-256 identities; every Swift run
verified them, and two newly generated fixtures reproduce those identities and
the complete 132-token sequence exactly. New artifacts never overwrite it.

Public prompt: `Explain HashMap in Java with an example.` Thinking disabled,
checkpoint chat template, greedy Top-K=1, temperature=0, no penalties.
Exact prompt IDs:

```text
248045 846 198 814 20139 10215 303 7693 440 449 3010 13 248046 198
248045 74455 198 248068 271 248069 271
```

Swift re-encodes its own template and rejects a prompt-token mismatch before
teacher forcing. MLX uses the checkpoint template; both regenerated fixtures
match these IDs. The Swift teacher loop uses greedy argmax directly; the
printed sampler seed is immaterial and no sampled penalty is applied.
Reference index **60** (zero-based), token **61** (one-based), consumes input
token ID **332** at absolute position **80** (zero-based) and predicts the token
for position **81**. This is after all 21 prompt and 60 preceding reference
IDs have been consumed. Both sides use that identical history. MLX uses normal
batched prompt prefill; Swift uses sequential prefill, a relevant arithmetic
execution difference that remains unisolated.

## Reproduction matrix

All checks use 8,192 selected context, raw Safetensors, no profiling, and the
same full teacher-forced 132-token input sequence.

| Condition | Full replays | Result |
| --- | ---: | --- |
| Unmodified release, 2,183 slots, separate processes | 2 | Each 131/132; only position 61, actual 10000 |
| Read-only diagnostics, 5 GiB / 2,183 slots, retained runner reset/replay | 2 | Each 131/132; only position 61, actual 10000 |
| Read-only diagnostics, 3 GiB / 969 slots, retained runner reset/replay | 2 | Each 131/132; only position 61, actual 10000 |
| Four selected-layer captures, 5 GiB, retained runner reset/replay | 2 | Each 131/132; only position 61, actual 10000 |
| Restored production source and executable | 1 | 131/132; only position 61, actual 10000 |
| Separately generated MLX references | 2 | Both identical to original fixture; same Top-5 and final normalized hidden vector |

The second retained-runner pass preserves the expert cache but resets sequence
state; it is a warm-cache replay, not prefix reuse. The existing CLI warmup/run
flags do not repeat teacher mode, so temporary diagnostics explicitly ran two
passes. Cache state changes exact scores, **not the winning token here**.

## Raw logits at reference token 61

Scores below are pre-softmax LM-head outputs. MLX outputs BF16 logits converted
to Float32 for inspection; Swift outputs FP32. These are comparable quantities
with different precision, not log probabilities or mixed score definitions.

| Rank | MLX token | MLX raw logit | Swift token | Swift raw logit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14988 | 27.875000000 | 10000 | 27.845386505 |
| 2 | 10000 | 27.750000000 | 14988 | 27.724899292 |
| 3 | 20480 | 23.750000000 | 20480 | 23.785346985 |
| 4 | 2070 | 19.250000000 | 2070 | 19.365528107 |
| 5 | 3436 | 18.625000000 | 3436 | 18.732065201 |

MLX margin `logit(14988) - logit(10000)` = **+0.125**. Swift initial 5-GiB
margin = **-0.120487213** (top-two gap +0.120487213). Signed margin changes by
0.245487213 across these outputs. Neither runtime has an exact top-two tie;
tie-breaking cannot explain this prediction. At logits around 27, one BF16
ULP is 0.125, so the MLX separation is one representable BF16 step. This
establishes a narrow decision margin, not an exemption from the parity gate.

| Swift condition | logit(14988) | logit(10000) | Winner gap |
| --- | ---: | ---: | ---: |
| 5 GiB, pass 1 | 27.724899292 | 27.845386505 | 0.120487213 |
| 5 GiB, pass 2 | 27.730762482 | 27.845378876 | 0.114616394 |
| 3 GiB, pass 1 | 27.728061676 | 27.830921173 | 0.102859497 |
| 3 GiB, pass 2 | 27.729824066 | 27.845741272 | 0.115917206 |

Cache-dependent hit/miss grouping changes floating-point accumulation order in
`qwen_combine_cached_moe_subsets_f32` (`attn_res + shared + hit + miss`). This is
a source-supported possible contributor to score variation, not a proven complete
explanation. No concurrency defect or cache corruption has been demonstrated.

## Localization and controlled head cross-check

Apply the **same loaded MLX quantized LM head** to captured hidden vectors:

| Hidden input / head output precision | logit(14988) | logit(10000) | Winner |
| --- | ---: | ---: | --- |
| MLX hidden / BF16 | 27.875000000 | 27.750000000 | 14988 |
| MLX hidden cast FP32 / FP32 | 27.847143173 | 27.773597717 | 14988 |
| Swift hidden / FP32 | 27.724899292 | 27.845386505 | 10000 |
| Swift hidden rounded BF16 / BF16 | 27.750000000 | 27.875000000 | 10000 |

On **the identical Swift hidden input**, MLX reproduces Swift's five token IDs
in the same order; maximum raw-score discrepancy is
**1.90734863281e-06**. Token 14988 and 10000 scores agree
exactly at FP32 in this comparison. The reversal survives a common FP32 head:
MLX hidden favors 14988 by 0.073545456; Swift hidden favors 10000 by 0.120487213.
Thus the observed reversal resides upstream of this head, rather than argmax or
only the final BF16 logit cast. This check does not certify every head output
or every input.

Selected output vectors at absolute input position 80 (layer indices zero-based):

| Boundary | Max absolute error | RMS error | Relative L2 error |
| --- | ---: | ---: | ---: |
| Layer 0 | 0.002012372 | 0.000079316 | 0.438729% |
| Layer 3 | 0.003219599 | 0.000811885 | 2.476826% |
| Layer 19 | 0.012592316 | 0.003478287 | 4.878141% |
| Layer 39, before final norm | 0.037962437 | 0.005995894 | 2.280052% |
| After final norm | 0.266634464 | 0.059479371 | 2.272128% |

A separate row-bounded embedding probe for input token 332 reproduces the Swift
FP32 affine dequantization formula and MLX's actual `mx.dequantize` path. Of
2,048 values, 262 differ, max absolute 0.00006103515625, RMS 0.000007593922,
relative L2 0.103372%. **Rounding the FP32 formula result to BF16 exactly equals
all MLX embedding outputs.** This is the earliest demonstrated precision
boundary. Layer 0 already differs by 0.438729% relative L2; downstream errors
are not monotonic. The probe is not an ablation proving that embedding rounding
alone causes the token reversal.

Known precision differences include FP32 Swift embedding/activations and
recurrent buffers versus BF16 MLX embedding/activations; Swift full-attention KV
is FP16. Q4/Q8 projections, RMSNorm, Conv/DeltaNet updates, RoPE/full attention,
MoE routing and residual reductions can accumulate or amplify differences.
No individual operation beyond the embedding boundary has yet been isolated as
wrong. Replaying layer 0 from the same prior Conv/DeltaNet state and measuring
component outputs/routed IDs is the next bounded localization step. Pinning an
oracle precision/execution contract is necessary before declaring harmlessness.

## Diagnostic scope and reversibility

Temporary Swift changes expose completed buffers only and record four selected
layer vectors at **one** token position. No shader/math/sampling changes; no
weights copied or modified; no all-40-layer dump. Selected-layer capture adds
four command-buffer waits; no-layer diagnostics only read completed final
buffers. In the observed first pass, traced and untraced Top-5 scores are
identical. Timing from these probes is diagnostic, not performance evidence.

All runtime/benchmark edits were removed afterward. Restored release SHA-256
`f55a81ce3041544c5cd811a1c9acbc04abcb302f8b40a29fd67d2d0af2854c32` equals the
pre-investigation executable byte-for-byte. The [temporary patch](temporary-diagnostics.patch)
and diagnostic scripts are retained solely as reproducible investigation evidence.
No production fix, new runtime API or inference regression test was retained.
A concrete defect must be identified before adding a targeted fix/test.

The first selected-layer MLX wrapper failed before generation because it did
not expose the original script's `input_layernorm` probe attribute. That failed
attempt exited 1, wrote no reference fixture, and was preserved as `mlx-2.log`.
The corrected wrapper forwards the attributes; subsequent fixture and captured
scores agree exactly with the prior successful diagnostic run.

## Reproduction commands

From the repository root, MODEL_DIR is the existing compatible checkpoint and
MLX_PYTHON is the installed environment's Python (`$HOME/venvs/mlx-dspark/bin/python`
in this run). Check macOS/toolchain/disk/normal memory pressure and absence of
other model owners **before each** model process. Never run these concurrently.

```bash
swift build -c release
.build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --cache-slots 2183 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
```

The temporary patch was applied only for diagnostic runs:

```bash
QWEN_PARITY_DIAGNOSTIC=/tmp/qwen-investigation-swift-5 \
  .build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --memory-gib 5 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
QWEN_PARITY_DIAGNOSTIC=/tmp/qwen-investigation-swift-3 \
  .build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --memory-gib 3 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
QWEN_PARITY_LAYERS=1 QWEN_PARITY_DIAGNOSTIC=/tmp/qwen-investigation-layers \
  .build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --memory-gib 5 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
```

MLX wrapper commands (defaults use the same `$HOME/models` checkpoint; scripts
retain the actual temporary paths used, so choose fresh output names):

```bash
"$MLX_PYTHON" docs/evidence/2026-11-17-investigation/capture_mlx.py \
  /tmp/qwen-investigation-mlx-2b
"$MLX_PYTHON" docs/evidence/2026-11-17-investigation/embedding_probe.py
```

`capture_mlx.py` invokes the original reference generator without changing its
sampler. Its cross-head control reads `/tmp/qwen-investigation-swift-5-pass-1.json`;
copy the archived snapshot there if reconstructing the experiment. The initial
MLX capture preceded addition of the selected-layer/head-crosscheck wrapper;
both resulting references match. No supported production diagnostic flags are
introduced by the archived patch.

## Validation and memory

Final results are listed in DEVELOPMENT_LOG and `validation.json`. After removal
of diagnostics, release build and all 49 package tests pass. Full parity remains
131/132, exit 1. Five existing malformed-reference cases are checked separately;
the stock six-case script cannot pass because its valid case remains failing.
No assertions/expected IDs changed. No claim of restored 132/132.

Swift diagnostic snapshots at position 80: 5 GiB plan 4.927902 GiB Metal,
4.791436–4.794808 GiB process footprint; 3 GiB plan 2.927292 GiB Metal,
2.789924–2.793311 GiB footprint. Buffers remain inside selected targets. These
are observed snapshots, **not sampled request peaks or an unload soak**. MLX
reference peak is 19.615 decimal GB, a full-resident reference outside the Swift
Metal streaming budget. No headline throughput/performance claim; complete
ordinary timing/cache footer is unavailable in teacher mode. Hashing warms OS
page cache; no system-swap delta series or physical SSD-traffic measurement.
Production binary is identical, so this task retains no allocation-path change.

## Next safe task

Continue S0-03 only: establish a pinned reference precision/execution contract,
then compare layer-0 components with the exact teacher-forced prefix and prior
Conv/DeltaNet state, beginning with embedding/RMSNorm/projections and router IDs.
Keep the failing fixture. A near-tie policy requires review and broader calibrated
logit/state evidence; it cannot silently convert this failure into a pass.
Do not begin Stage 1 or change production precision to force token 61 to match.
