# Stage 0 numerical root-cause isolation — 2026-11-17

## Supported conclusion

**Overall classification 4: the full-model token-61 reversal remains unresolved.**
Within the inspected layer-0 chain, intentional numerical differences are
confirmed and no implementation defect was demonstrated: supplying identical
FP32 inputs/prior state makes independently executed RMSNorm, QKV, Conv,
DeltaNet, router and expert operations agree closely. Native precision and
history differ. This does not certify the remaining layers, full attention or
all possible inputs. No production fix is proposed and no gate is weakened.

The [numerical contract](../../NUMERICAL_CONTRACT.md) distinguishes measured
array dtypes, inspected source behavior and backend details not verified here.
MLX recurrent state is **FP32**, not BF16. MLX Conv history/activation outputs
are BF16; decay is FP32 and beta BF16. Swift uses FP32 there. Attention KV is
FP16 in Swift versus BF16 in the installed MLX path. Treating all MLX arithmetic
as BF16 would be incorrect.

Original fixture, checkpoint and library identities are unchanged. Base commit
72fb923a3e7a5d43e5baeb43d35bdcf1f96e4966 on feature/local-coding-agent-server;
uncommitted foundation and earlier evidence preserved. M5/32 GB, macOS
27.2.1/26B131, Swift 6.4.2; MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.22.2.
Source hashes and unchanged fixture/executable hashes are in provenance.json.
No model file, package or dependency was altered. Git remote was inspected:
origin points to tanishqpatil/qwen-fieldfare; no remote action performed.

## Scope and valid comparison contract

Positions inspected: zero-based absolute input positions 0,1,2,20,40,60,80,
using the exact existing prompt plus first 60 teacher reference tokens. Position
80 consumes token 332 and predicts reference token 61. The full teacher harness
still checks all 132 predictions, not only captured boundaries.

Only layer 0 was traced in detail; layer 3 selected IDs were recorded for cache
independence, not compared against every MLX layer. Raw state/activation buffers
were kept in bounded temporary directories; the repository preserves comparison
metrics, source patch and scripts rather than large tensor payloads or weights.

Layer 0 depends only on the current token embedding and its own recurrent/Conv
history, so direct replay of this layer uses valid model inputs. The MLX
transcription uses the installed modules/functions, not a newly implemented
recurrence; native layer output **and updated state match exactly** at every
inspected boundary when supplied the same prior state. This tests the diagnostic
transcription; it is not independent validation of MLX itself.

Two MLX histories are explicitly separate: (a) every token sequentially, for
per-step prior/post state inspection, (b) the reference generator's actual first
20 prompt tokens batched, followed by tokenwise steps. Early batched components
come from the corresponding batch row. Do not mistake sequential-only results
for the original reference trajectory.

State alignment:

- Swift Conv [8192,4] slots 1:4 transpose to MLX [1,3,8192]. The fourth stored
  Swift slot does not mean a different convolution window: next update reads
  its three newest slots and appends current projection.
- Swift DeltaNet [32,value_row=128,key_col=128] maps directly to MLX
  [1,32,128,128], B=1. Both actual states are FP32. Matched-state tests preserve
  exact captured FP32 values; no BF16 state is substituted.
- BF16 Conv/activation values are promoted exactly to FP32 for comparison;
  converting Swift FP32 to BF16 is explicitly a precision ablation, not an
  assertion that the native states are mathematically identical.

## Earliest divergence and scale-aware component results

The earliest observed difference is **the first input token, position 0**, before
any prior history exists. Both recurrent states start at zero. Errors below
compare Swift with the actual batched-prefix reference (not sequential-only).
All metrics use FP64 host comparison of captured values; relative L2 denominator
is the reference norm. Reference RMS exposes scale; no small absolute error is
automatically declared negligible.

| Position | Component | Max abs | Error RMS | Relative L2 | Reference RMS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | embedding | 0.000122070312 | 1.82386317e-05 | 0.150982% | 0.0120799721 |
| 0 | norm | 0.0204737186 | 0.00284216549 | 0.277599% | 1.02383953 |
| 0 | qkv | 0.0284137726 | 0.00283127624 | 0.228730% | 1.2378261 |
| 0 | conv | 0.0157463551 | 0.000405608841 | 0.297640% | 0.136275024 |
| 0 | gated | 0.0272693634 | 0.000478633729 | 0.993859% | 0.0481591362 |
| 0 | output | 0.00267148018 | 0.00050556564 | 1.012012% | 0.049956469 |
| 80 | embedding | 6.10351562e-05 | 7.59392181e-06 | 0.103372% | 0.00734617908 |
| 80 | norm | 0.0510349274 | 0.00285644488 | 0.270071% | 1.05766607 |
| 80 | qkv | 0.0655517578 | 0.00418377114 | 0.191926% | 2.17988336 |
| 80 | conv | 0.0567626953 | 0.00151601717 | 0.209951% | 0.722082947 |
| 80 | gated | 0.00576519966 | 0.00010522752 | 0.324143% | 0.0324632753 |
| 80 | output | 0.00201237202 | 7.93162352e-05 | 0.438729% | 0.0180786244 |

Complete selected-position metrics (including z/a/b, residual, post-normalization,
router, shared/routed outputs and prior/post recurrent state) are in
comparison.json. Those per-step state trajectories use the separately labelled
sequential MLX replay; actual batched-reference metrics are in
actual-reference-comparison.json. Neither is a tolerance or passing criterion.

## Controlled identical-input operation tests

Feed captured Swift inputs into the installed MLX operations with the same FP32
activation and state values. Quantized coefficients remain the same checkpoint;
Conv coefficients are losslessly promoted from BF16 to FP32 to match the Swift
kernel. Readback changes scheduling but not production math. Range maxima below
cover all seven selected positions; they do not certify all model shapes.

| Operation | Worst max abs | Worst RMS | Worst relative L2 |
| --- | ---: | ---: | ---: |
| Input RMSNorm | 0 | 0 | 0 |
| QKV projection | 3.81469727e-06 | 3.58959777e-07 | 1.92387299e-07 |
| DeltaNet state update | 1.90734863e-06 | 7.02565002e-09 | 1.12181624e-07 |
| DeltaNet gated output | 4.76837158e-07 | 9.13512173e-09 | 1.88512107e-07 |
| Q8 router | 1.90734863e-06 | 3.40533104e-07 | 6.27054267e-08 |
| Routed expert output | 2.98023224e-08 | 3.81900384e-09 | 3.19928173e-07 |
| Shared expert output | 1.49011612e-07 | 5.98679811e-09 | 2.26282033e-06 |
| Conv + SiLU | 9.53674316e-07 | 3.88603341e-08 | 6.65639632e-08 |

This localizes most native-path error to differing numerical inputs and histories,
rather than a mismatch in the examined matched-FP32 operation semantics. It is
not a proof that every backend instruction, full-attention path or deeper-layer
routing decision is correct.

## Current step versus prior-state history

Native sequential post-state errors versus Swift:

| Position | Prior-state relative L2 | Post-state relative L2 |
| ---: | ---: | ---: |
| 0 | 0.000000% | 0.757879% |
| 1 | 0.757879% | 0.650914% |
| 2 | 0.650914% | 0.534176% |
| 20 | 0.446281% | 0.452022% |
| 40 | 0.410813% | 0.404782% |
| 60 | 0.393221% | 0.400720% |
| 80 | 0.397606% | 0.395182% |

At position 80, the matched-input/prior-state DeltaNet control has only
3.13309e-8 relative L2 error versus Swift updated state. Holding Swift current
inputs fixed but substituting the captured MLX sequential FP32 **prior** state
changes updated state by **0.382589%** relative L2. Holding Swift prior state fixed
but substituting native MLX current normalized Q/K/V and a/b changes it by
**0.100589%**. Those are separate counterfactual changes against the same control,
not additive percentages or a complete causal decomposition.

The MLX reference's own batched versus sequential prompt execution changes
position-80 prior/post state by 0.091963%/0.091425% and layer output by 0.399003%,
while its current QKV projection is identical there. Thus accumulated history
is a demonstrated contributor; current projections alone cannot explain the
entire later difference. State errors do not simply grow monotonically.

## Independent precision ablation

Only Swift embedding affine results were rounded to BF16 (round-to-nearest,
ties-to-even), at every token. All subsequent production operations, recurrence
and cache policy remained unchanged. This is a diagnostic hypothesis test,
not a proposal to alter the numerical contract.

| Position | Baseline layer-0 output error | Embedding-rounded output error |
| ---: | ---: | ---: |
| 0 | 1.012012% | 1.046945% |
| 1 | 0.747559% | 0.768869% |
| 2 | 0.663824% | 0.652271% |
| 20 | 0.501082% | 0.498979% |
| 40 | 0.444319% | 0.449751% |
| 60 | 0.461990% | 0.454544% |
| 80 | 0.438729% | 0.434745% |

Rounding alone is not a general repair: it improves some positions and worsens
others. Full parity remains **131/132**, same token-61 reversal. No expected ID,
sampling algorithm or production precision was changed to force a match.

## Routing and expert accumulation

All seven inspected layer-0 selected expert **sets** match across runtimes.
Array order differs (Swift descending-score selection versus MLX argpartition),
so scores are aligned by expert ID. At position 80 Swift IDs are
[179,145,130,91,197,10,79,88]; MLX returns the same set in reverse array order.
The largest per-ID routing-weight absolute difference over inspected positions
is 0.003897175; native weights are BF16 in MLX versus FP32 in Swift. Matched-FP32
router/expert tests above reduce discrepancies to numerical reduction scale.
No earliest routing-set disagreement was found **within this limited sample**.

Changing Swift expert-cache capacity from 2,183 to 969 does not change selected
IDs at layers 0 or 3 at any inspected position. Layer-0 output differences are
at most 7.45058e-9 absolute here. Using identical FP32 expert contributions and
changing only forward/reverse sum order gives a small rounding difference;
full metrics are under expert_reduction_order in mlx-summary.json. This does not
prove cache-independent bitwise output or certify deeper-layer routing. Both
cache profiles still fail full parity at token 61.

## Fix and correctness-policy assessment

No concrete defect is demonstrated in the examined operations. No production
fix, global BF16 conversion or checkpoint/reference replacement is proposed.
The final full-model reversal still requires isolation beyond layer 0.

**Proposal for review only:** supplement the existing strict gate with a
versioned numerical validation matrix: same-input/same-precision operation and
state-transition tests; declared layout/dtype contracts; per-layer/logit-error
and margin distributions; top-token agreement; sampled/longer-history behavior;
and pinned reference/hardware provenance. Calibrate numerical bounds on an
independent multi-prompt suite before proposing a cross-precision tolerance.
Retain 132/132 as the current gate. Do not add a one-ULP/near-tie exemption or
claim all 2.27% hidden-vector error benign from this single case. No acceptance
policy or test threshold is changed, and no approval is inferred.

## Reproduction and instrumentation

Restore the archived patch only in an isolated diagnostic checkout or after
saving the exact pre-edit source. It is **not applied now**. From repo root,
with MODEL_DIR set to the existing checkpoint:

```bash
swift build -c release
QWEN_ISOLATION=/tmp/qwen-isolation-swift-base QWEN_EMBEDDING_BF16=0 \
 .build/release/QwenFieldfareBench --model "$MODEL_DIR" --context 8192 \
 --memory-gib 5 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
QWEN_ISOLATION=/tmp/qwen-isolation-swift-rounded QWEN_EMBEDDING_BF16=1 \
 .build/release/QwenFieldfareBench --model "$MODEL_DIR" --context 8192 \
 --memory-gib 5 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
QWEN_ISOLATION=/tmp/qwen-isolation-swift-cache3 QWEN_EMBEDDING_BF16=0 \
 .build/release/QwenFieldfareBench --model "$MODEL_DIR" --context 8192 \
 --memory-gib 3 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
```

Each command ran separately, after the previous owner exited, with normal
kern.memorystatus_vm_pressure_level=1 and process exclusions. The attempted early
MLX start was blocked by the exclusion while Swift was still active; MLX did
not load. It was started only after Swift completion. Available memory 85%–88%
reported across checks. The checkpoint, OS/toolchain and ample disk were checked
against the continuing same-session baseline; no inference owners overlapped.

MLX scripts use the existing $HOME/models checkpoint and the same installed
$HOME/venvs/mlx-dspark Python. Exact invocations were:

```bash
"$MLX_PYTHON" /tmp/qwen-isolation-mlx.py
"$MLX_PYTHON" /tmp/qwen-isolation-conv.py
"$MLX_PYTHON" /tmp/qwen-isolation-compare.py
```

Archived counterparts are capture_mlx.py, matched_conv.py and compare.py.
The final capture script writes /tmp/qwen-isolation-mlx2; the comparator uses
the first /tmp/qwen-isolation-mlx trajectory. Both successful runs have identical
native state/output checks, and the second adds MoE/KV controls. Temporary
raw snapshots are needed to rerun comparisons; the scripts recreate them.
No script is a new production API. Supplied state files contain activations,
not weights. Archive contains metrics/scripts only.

Layer-0 snapshots add a command-buffer wait and bounded file readbacks at seven
positions. No instrumented timing is a production benchmark. FP32 state snapshots
are about 2 MiB each; total captures are bounded (about 31 MiB per Swift variant)
and stay temporary. No new Metal arena was allocated. First model-free attempt
to import MLX for docstrings failed under sandbox Metal access; read-only retry
with expanded access succeeded. No model or dependency was changed to repair it.

## Final validation and remaining boundary

All temporary production edits were removed. Restored benchmark SHA-256 is
f55a81ce3041544c5cd811a1c9acbc04abcb302f8b40a29fd67d2d0af2854c32, identical to
the original control. Sources/Scripts/Tests/Package files have no diff. Release
build passes (18.11 s), full serial suite passes 49/49 (17 Presentation, 32 Core).
Restored full teacher forcing exits 1, **131/132**, only token 61 expected 14988,
actual 10000. The unchanged malformed-reference suite was not rerun: no fix or
validation-code change was retained; five negatives passed in the prior phase.
No claim that the six-case suite passes. No new defect regression test is
invented because no concrete defect was fixed.

Loaded Metal snapshots remain 4.928 GiB at 5 GiB and 2.927 GiB at 3 GiB; these
are not request peaks. MLX peak in the second diagnostic is 19,631,363,782 bytes,
a full-resident reference outside the Swift streaming allocation. Temporary CPU
readbacks have explicit bounds. No complete cache/timing footer, swap-delta or
unload-soak series; no performance or memory-regression promotion claim.

**Next experiment:** compare layer-3 full attention under identical projected
Q/K/V and prior history, independently changing BF16 versus FP16 storage with
explicit layout/position conversion. Record small selected-ID summaries through
all layers at positions 20 and 80 to find any deeper routing-set disagreement.
This is the next bounded step; do not start Stage 1 or silently replace strict
parity with a tolerance. Original fixture remains untouched. All changes remain
uncommitted; no push or merge.
