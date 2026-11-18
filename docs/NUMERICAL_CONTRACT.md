# Numerical execution contract — Stage 0 investigation

Observed 2026-11-17 at base `72fb923`, Apple M5/32 GB, macOS 27.2.1, Swift 6.4.2.
Reference: installed MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.22.2. The compatible
checkpoint is identified by the unchanged [format-2 fixture](evidence/2026-11-17-parity/reference.json).
Installed source hashes and measured dtypes are preserved in
[isolation evidence](evidence/2026-11-17-isolation/provenance.json) and
[MLX summary](evidence/2026-11-17-isolation/mlx-summary.json).
This is a descriptive contract for observed paths, **not a new acceptance policy**.

## Evidence levels

- **Measured:** actual runtime arrays, identical-input comparisons, native-path checks.
- **Source:** the owning Swift/Metal or installed Python implementation was inspected.
- **Unverified:** backend arithmetic or paths not isolated here; do not infer precision
  from tensor storage alone. This document does not certify all layers or contexts.

## Operation contract

| Operation | Swift active streaming | Installed MLX path | Evidence / limits |
| --- | --- | --- | --- |
| Embedding | Q4 integer nibble, BF16 affine scale/bias decoded to FP32; FP32 affine result, no BF16 store | QuantizedEmbedding calls mx.dequantize; BF16 result | Measured. Rounding Swift affine output to BF16 matched the prior input-332 probe exactly. Token-0 input already differs. |
| Input/post/final RMSNorm | FP32 FMA square accumulation, SIMD/two-group reduction, rsqrt(mean + 1e-6), FP32 products/output | mx.fast.rms_norm, BF16 output for native BF16 input; FP32 output for supplied FP32 input | Same FP32 embedding/weights/epsilon gives exact layer-0 output at all seven positions. MLX GPU internal reduction instructions not established by its installed docstring. |
| Q4 projections | Affine group size 64; packed integer weights, BF16 scales/biases decoded FP32; factored FP32 dot/sum, FMA, SIMD reduction; FP32 result | mx.quantized_matmul; native BF16 input/output, FP32 output for diagnostic FP32 input | Native dtypes measured. Same-input QKV errors ~1–2e-7 relative L2. Exact MLX backend instruction/reduction contract unverified, not assumed BF16 accumulation. |
| Q8 router/shared gate | Group-64 affine Q8, FP32 accumulation/output | QuantizedLinear with Q8 weights; BF16 native router output | Same-FP32-input router relative L2 ~6e-8. Full Q8 backend accumulation not inferred from output dtype. |
| Conv history | FP32 [channel,4]; slot 0 is redundant for next step; read slots 1/2/3, append current QKV, shift | BF16 [batch,3,channel]; concatenate history/current, retain last three | Measured values after explicit mapping. Swift prior history is slots 1:4 transposed; same mapping for next-step history after update. Do not compare all four slots to three native slots. |
| Conv + SiLU | FP32 four-term convolution and exp-based SiLU; FP32 output | nn.Conv1d then SiLU, BF16 native output/history | Matched FP32 inputs/history/coefficients through mx.conv1d+SiLU agree ~5–7e-8 relative L2. Backend internal BF16-path MAC details unverified. |
| Q/K preparation | FP32 q / sqrt(sum(q²)+1e-6) / sqrt(128); k / sqrt(sum(k²)+1e-6) | normalize_qk folds identical scaling through RMSNorm; native BF16 q/k output | Formula equivalence from source; epsilon adjusted by 1/128 in MLX RMSNorm. Same-input fused recurrence tested, not every individual normalization intermediate. |
| Beta/decay | FP32 sigmoid(b), exp(-exp(A_log)*softplus(a+dt_bias)) | Native beta BF16; decay FP32 (A_log explicitly cast FP32) | Measured: not all MLX recurrence quantities are BF16. Exact compiled intermediate rounding within compute_g remains unisolated. |
| DeltaNet state | FP32 [value_head,value_row,key_col] = [32,128,128] | FP32 [batch,value_head,value_row,key_col] = [1,32,128,128] | Actual dtype/layout verified. Batch dimension can be omitted only for B=1. Both start at zero. |
| DeltaNet update | Decay S; dot(decayed S,k); delta=(v-dot)*beta; S+=delta*k; read S*q, FP32 FMA/dot | Same update order; FP32 local/state accumulation; native core output cast to BF16 | Installed GPU kernel source + measured state. Matched FP32 inputs/prior state agree 3e-8–1.2e-7 relative L2; native output BF16 does not imply BF16 state. |
| Gated RMSNorm | Fused FP32 core normalization, learned weight and SiLU(z), FP32 result | RMSNorm output follows BF16 core; precise_swiglu casts gate/normalized value to FP32 for SiLU/product, then casts result to core dtype | Measured gated output BF16; source proves an internal FP32 path. Matched-input FP32 gated results agree ~3e-8–1.9e-7 relative L2. |
| Attention projections | Q4 FP32 projection; FP32 head normalization | Native quantized projection/head normalization output BF16 | Source; full-attention component equality not isolated in this round. |
| RoPE | FP32 CPU frequencies/cos/sin and FP32 partial rotation of first 64/256 head dimensions, non-interleaved halves | Installed nontraditional partial RoPE; BF16 output measured on the actual checkpoint module | Offset semantic source inspected. Matched-Q/K RoPE comparison still pending. |
| Attention KV | Token-major [token,2,256], FP16 K/V storage | [batch,2,token,256], BF16 K/V; actual full-model two-token run verified cache dtype | Different dtype and layout; conversion must explicitly transpose and apply declared rounding. No raw memcpy comparison is valid. FP16 vs BF16 can affect later attention; not isolated here. |
| Full-attention execution | FP32 RoPE Q cast to FP16; split attention uses FP16 Q/K/V, FP32 reductions and FP16 attention output before FP32 gate | Native attention path receives BF16 Q/K/V | Swift source verified; matched-input cross-runtime attention comparison remains pending. |
| Routing | FP32 router logits and normalized top-eight weights; selected IDs descending score | precise softmax on router logits; selected IDs via argpartition, weights native BF16 | Seven layer-0 sets match, after aligning by ID rather than array order. No claim about every other layer/token. |
| Expert outputs/accumulation | FP32 shared expert and routed contributions; cache hits/misses summed separately; final attn_res + shared + hit + miss | Native BF16 expert/contribution output; routed weighted reduction, then shared addition, then residual | Same-input FP32 routed/shared controls agree closely. Reverse-order FP32 sum has ~7e-8 relative L2 effect here. Native BF16 arithmetic and residual ordering differ. |
| Final norm/LM head/argmax | FP32 output norm, Q4 FP32 head, largest-logit selection | Native final hidden/logits BF16; greedy Top-K=1 | Earlier common-head test agrees to 1.91e-6 on identical Swift hidden. Neither candidate is exactly tied at failed position; tie-breaking is not that reversal. |

Swift source owners: `Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift`
and `Metal/LinearAttention/deltanet.metal` under QwenFieldfareCore. Installed MLX
owners: qwen3_5.py, qwen3_next.py, gated_delta.py, activations.py, cache.py and
nn/layers/quantized.py. These local source hashes identify the actual reference;
a package version string alone is not a complete source provenance guarantee.

The installed DeltaNet dispatch enables its packed kernel by default when no
mask, scalar gates, Dk=128, Dv divisible by eight and FP32 gate/state conditions
hold. MLX_GDN_PACKED was unset, use_kernel=True, GPU available; captured shapes
meet that predicate. Its different reduction tree must not be assumed bitwise
identical to Swift. Full official wheel/source equivalence was not audited.

## History and execution differences

Swift prefill consumes each prompt token separately. The reference generator
prefills the first 20 prompt tokens together, then the last prompt token and
teacher history individually. Layer 0 is independent of later-layer outputs,
so it can be replayed alone under identical token IDs without changing its inputs.
The diagnostic transcription reproduces native layer output and updated recurrent
state **exactly** at inspected boundaries with identical supplied prior state.

A sequential-only MLX replay is a separate diagnostic, not a replacement reference.
Even MLX's batched versus sequential prompt paths leave different Conv/recurrent
history: at input 80 the state relative L2 difference is 0.091425% and layer-0
output difference 0.399003%. QKV at the current token is identical in those two
MLX paths, locating that difference in history rather than current projection.
The first 20-token batch uses actual same layer modules; early prompt components
are compared to batch rows, with per-step states taken from the separately
labelled sequential replay.

## Correctness boundary

Intentional precision/execution differences are demonstrated locally; no examined
layer-0 operation has shown a semantic defect under matched FP32 inputs/state.
This is not proof of the full model or an attribution of every final-logit error.
Strict 132/132 remains in force; current outcome is 131/132. See the
[isolation report](evidence/2026-11-17-isolation/REPORT.md) for results, limitations
and the proposed additional validation dimensions. No policy is changed here.

## S3-00a: cache-residency consistency

With fixed layer-0 inputs and real checkpoint weights, all examined router
weights, expert records, gate/up activations, and weighted expert contributions
match bit for bit across cold, warm, and mixed masks. Legacy sums change when
the top-eight contributions are regrouped into hits and misses. The same mixed
mask reproduces exactly. The final CPU left-to-right reconstruction is not
bitwise identical to the compiled Metal combine. Source grouping and measured
outputs establish the residency effect without requiring an assumed compiler
instruction order.

A diagnostic path stores the eight FP32 weighted contributions by router
position and reduces in that order. The two matched-input expert histories
produce equal projected logit bytes at all 941 inspected input positions and
equal final hidden bytes at the historical divergence. At that position the
legacy histories differ by 1.950% relative L2 in logits. Router-order logits
differ by 1.358% relative L2 from the legacy retained-tool control, though the
two predict the same IDs on the warm 512-step fixture. This improvement in
residency consistency says nothing by itself about reference accuracy.

Proposed deterministic gate for review: with the same executable, device,
context, checkpoint, teacher tokens, and precision, require exact equality
for matched-input router results and per-expert contributions and for canonical
MoE layer output across masks. Repeat one mask to check same-mask nondeterminism.
Require equal logits across controlled histories at every forced position and
inspect hidden values and top-two scores at any first differing prediction.
Apply an independent, checkpoint-identified MLX reference gate with the
original strict test. Other devices or deliberately different arithmetic
would need separate calibration on matched inputs before adopting tolerances.
The observed differences are descriptive and are not an allowable error budget.
Future conversation reuse must independently compare with canonical full replay.
See [S3-00a evidence](evidence/2026-11-18-s300a/README.md).

## S3-00b: matched reference fidelity

The original 21+132 teacher fixture and 430+512 sustained continuation have
different inputs and independent acceptance roles. Native MLX batch prefill
reproduces 132/132 on the original fixture. A sequential-only MLX diagnostic
with identical teacher tokens returns 129/132. All four Swift controls return
the same known 131/132. Do not use the sequential experiment to redefine
the reference or declare the strict failure resolved.

Router order gives bitwise-equal logits across cold/warm residency for both
teacher histories, 152 and 941 input positions, with equal sampled final
hidden states, layer-0 router weights, per-expert contributions and layer
outputs. Fidelity against native MLX is similar in aggregate. At sustained
output 465, MLX native and sequential both prefer token 285 with raw
1031–285 margin −0.125. Cold legacy also prefers 285 (−0.01248). Router order
prefers 1031 (+0.01194) under both cache histories, as warm legacy does.
This is separate from the strict fixture's token-61 mismatch. Reference
and residency gates test different properties.

Outcome B: defer promotion and test only a fixed-order combine with the
shared expert added to routed contributions before the attention residual.
Require exact warm/cold consistency again and inspect the saved matched
reference scores before deciding whether to keep it. See
[S3-00b metrics and decision](evidence/2026-11-18-s300b/README.md).

## S3-00c: selected active-streaming reduction

The final combine experiment used matched prior captures and a compiler
sentinel to verify `residual + (routed + shared)`. Both deterministic
implementations agree within each cold/warm replay. The grouped form improves
mean sustained logit relative L2 only from 7.613% to 7.607% against native
MLX. Its disputed 1031−285 margin moves from +0.01194 to +0.01535, away from
MLX's −0.125. The strict token-61 rank and 131/132 result remain the same.
The original router-order path is selected as the active-streaming baseline.
Its measured stability is scoped to the tested same-runtime histories.
This does not change the strict reference gate or qualify cross-request state
reuse. [S3-00c decision](evidence/2026-11-18-s300c/README.md)


The S3-01b timing probe preserved the existing execution schedule. The
layer-major design would retain ascending per-layer Conv, DeltaNet and KV
positions and per-row fixed-order expert reduction. No batch implementation
or change to this contract was adopted.
See the [feasibility report](evidence/2026-11-18-s301b/README.md).
