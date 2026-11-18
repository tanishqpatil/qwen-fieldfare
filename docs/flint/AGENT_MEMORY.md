# FlintFieldfare Agent Recovery Memory

This document is the authoritative recovery guide for agents operating on FlintFieldfare. Read this file at session start to resume work seamlessly.

## Mission Context
- **Target Model**: Recursive Flint 40B-A2.5B (MoE: 49 layers, 2 dense MLPs + 47 MoE layers, 256 routed experts/layer, top-6 routing, 1 shared expert).
- **Target Hardware**: Apple MacBook Pro M5, 32 GB unified memory, macOS 27.2.1, Apple Swift 6.4.2.
- **Local Checkpoint**: `/Users/tanishq/models/flint-mixed-3-4bit` (17.84 GB on disk, 3-bit experts, INT4 projections, INT8 embeddings).
- **Git Branch**: `feature/recursive-flint-40b-a2p5b`
- **Base SHA**: `7370a1d` (from `origin/feature/local-coding-agent-server`).
- **Independent Clone Path**: `/Users/tanishq/dev/product/flint-fieldfare-isolated`

## Current Milestone Status
- **F0 (Audit, Architecture & Baseline)**: **COMPLETED & REVISED (Review #001 & #002)**.
  - Empirical all-resident MLX baseline measured on M5 (4.39s load, 17.68 GB peak, +657.6 MB swap event, ~70 tok/s decode, ~1,317 tok/s 8K prefill).
  - Reproducible evidence preserved under version control at `docs/flint/evidence/2026-11-18-baseline/` (`benchmark_flint_mlx.py`, `mlx_flint_benchmark_results.json`, `README.md`).
  - Derived SSD streaming bandwidth formula vs hit rate (370.125 MiB/tok cold).
  - Clearly separated empirical measurements from theoretical streaming design targets (~3.2 GiB).
  - Architecture decisions ADR-F001 through ADR-F007 recorded in `docs/flint/DECISIONS.md`.
- **F1 (Metadata & Weight Loading Infrastructure)**: **F1-01, F1-02, F1-03, F1-04 COMPLETED**.
  - `FlintModelDirectory` (F1-01): Strict Case A (INT4) & Case B (141 overrides) validation, safe pread contiguous buffers.
  - `FlintInferenceLease` (F1-02): Dedicated lock at `/tmp/flintfieldfare-inference-\(uid).lock`.
  - `FlintSafetensorsFile` & Cross-Validation (F1-03):
    - POSIX descriptor ownership (`O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`, `close(descriptor)` on deinit).
    - Bounded header parsing (100 MB max bound, 8-byte LE header length decoding, UTF-8 non-NUL verification).
    - Offset & range validation (negative, out-of-bounds, inverted, arithmetic overflow, and zero overlapping allocations across tensors).
    - Safe random-access `preadExact` using contiguous memory buffers.
    - Identity verification via `stat`/`fstat` and mutation detection via `verifyUnchanged()`.
    - Cross-validation against config metadata (`mlx_export` vs `quantization` vs Safetensors shapes/dtypes).
    - Opening safeguards resolved: bounded chunk allocations (16 MiB max) in `readTensorChunks`, safe throwing `FlintTensorInfo` construction.
    - Bitwise SHA-256 match against independent Python/MLX ground truth across 8 empirical tensors.
    - Physical tensor inventory report: `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` ($1,376,256\text{ B} = 1.3125\text{ MiB}$ per 3-bit expert, closed-form offset formula).
  - `FlintResidentLoader` & `FlintResidentWeights` (F1-04):
    - Single authoritative copy per tensor via Metal shared buffers (`MTLBuffer` with `.storageModeShared`).
    - Programmatic `FlintResidentManifest`: All 1,328 permanently resident tensors across all 49 layers. Total resident payload: 1,278,418,496 B (1.1906 GiB / ~1.28 GB).
    - Direct zero-copy I/O: 0 bytes peak temporary staging allocation.
    - Strictly 0 bytes of routed expert weights (`switch_mlp`) loaded into resident memory.
    - Deterministic failure injection seam and verified partial-load cleanup (`weak` pointer deallocation).
    - Real teardown verification in explicit scopes (`weakWeights == nil`, `weakSampleBuffer == nil`).
    - Explicit Metal allocation measurements (`device.currentAllocatedSize` before, during, after).
    - Hardened storage modes: `.gpuRequired` (explicit failure, zero silent CPU fallback) and `.cpuOnly` (aligned heap).
    - 15 independent reference SHA-256 digests verified across early, middle, and final layers.
    - Packed U32 weights verified intact without float conversion.
    - Measured on Apple M5 32 GB: 0.218s – 0.351s load duration.
  - Test Suite Breakdown: **59 tests passing across 4 suites in 0.651 seconds**:
    - `FlintResidentLoaderTests`: 9 tests (live M5 load, repeated cycles, teardown, failure injection, CPU-only mode, bounds).
    - `FlintSafetensorsFileTests`: 28 tests (header validation, O_NOFOLLOW, 64 MB chunk limit, concurrent reads, live inventory).
    - `FlintModelDirectoryTests`: 20 tests (Case A/B bifurcation, strict metadata, 141 MoE projection checks).
    - `FlintInferenceLeaseTests`: 2 tests (mutual exclusion on `/tmp/flintfieldfare-inference-(uid).lock`, path validation).
  - Concurrency & Thread-Safety Qualification:
    - `Sendable` declarations are enforced by the Swift 6 compiler (`-strict-concurrency=complete`).
    - Basic concurrent read safety is validated via `testConcurrentReadsDoNotRace()` and `testSendableAndThreadSafetyVerification()`.
    - Full end-to-end multi-threaded pipeline qualification (e.g. concurrent token prefill/decode) is scheduled for F3 runtime integration.
  - Qwen Test Suite Lock Contention Resolution:
    - Diagnosed that parallel test target execution (`swift test`) causes race conditions on `/tmp/qwenfieldfare_inference.lock` between `QwenFieldfareServerTests` and `QwenFieldfareCoreTests`. Sequential execution (`--no-parallel`) passes 100% (99/99 tests in 0.129s).
    - FlintFieldfare uses an isolated dedicated lease lock `/tmp/flintfieldfare-inference-(uid).lock` preventing any cross-process or cross-target contention.
- **F2 (Mathematical Primitives & Metal Kernels)**: **F2-01, F2-02, & F2-03 COMPLETED**.
  - `FlintRMSNormKernel` & Metal Shading Language (`flint_rmsnorm.metal`) (F2-01):
    - Implemented native Metal compute kernels for width 2048 (hidden state, 256 threads, 8-warp SIMD reduction) and width 128 (head dim for per-head `q_norm` and `k_norm`, 128 threads, 4-warp SIMD reduction).
    - General fallback kernel repaired and qualified: synchronized threadgroup shared sums (`shared_sums[8] = 0.0f`), verified across non-specialized dimensions (16, 64, 129, 192, 256, 512, 1024) in FP32 and BF16 against CPU reference.
    - Swift dispatch bounds hardened with overflow-checked integer arithmetic (`multipliedReportingOverflow`, `addingReportingOverflow`), non-finite epsilon rejection (NaN, ±inf), and offset alignment checks.
    - Canonical Metal shader source equivalence test: verified zero drift between on-disk `flint_rmsnorm.metal` and `embeddedShaderSource`.
    - Direct zero-copy consumption of resident weights. Zero weight mutation (SHA-256 verified invariant before and after execution).
    - Independent Reference Qualification (vs MLX 0.32.4): 20/20 cases passed (FP32 exact match $0.000000$, BF16 max ULP $\le 1$, max relative diff $\le 0.781\%$).
    - Apple M5 Benchmark: Hidden Norm Decode 14.00 µs, Hidden Norm Prefill 25.25 µs (41.86 GB/s), Q Norm 6.42 µs, K Norm 6.37 µs.
  - `FlintAttentionKernel` & `FlintRoPEKernel` (F2-02):
    - GQA architecture: 32 query heads, 4 KV heads (8:1 ratio), head dimension strictly 128, scaling factor $1.0 / \sqrt{128} \approx 0.0883883476$.
    - Split-half RoPE (`traditional=False`, base 10000.0) implemented in `flint_rope.metal` and `FlintRoPEKernel.swift` with 64 threads/threadgroup. Bypassed on 14 full-attention layers (`[0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]`) and applied to 35 sliding-attention layers.
    - Online streaming FlashAttention formulation: threadgroup-based parallel online softmax accumulation in FP32 with zero global memory matrix allocations.
    - Learned Attention Sinks: per-head scalar `softmax_offset[32]` initialized as starting max $m = s_h$, running sum $l = \exp(s_h - m) = 1.0$, and running accumulator $acc_d = 0.0$, strictly contributing to the softmax denominator without adding a spurious value vector.
    - Dual masking modes: full causal attention (all cached keys attended) and sliding window causal attention (window 513 tokens, attended keys $\max(0, p - 512) \le j \le p$).
    - Bounded-KV Cache Eviction Indexing Closure: Formalized contract separating `queryAbsPos`, `retainedKVStart`, `T_kv`, and sliding window bounds. Verified physical indexing $[0, T_{kv} - 1]$ when earlier tokens have been evicted. Verified scenario: absolute position 1024, retained interval 512..1024 (physical indices 0..512), attending all 513 retained keys.
    - Ring KV cache layout explicitly rejected (`FlintKVStorageLayout.ring` throws `invalidParameter`).
    - Combined RoPE + Attention qualification: Split-half RoPE applied at sequence position 600 for queries and position 88 for retained keys, followed by grouped-query SDPA with learned sinks beyond 513 tokens.
    - Direct consumption of resident BF16 attention sinks with FP32 activations (`flint_attention_decode_f32_bf16sinks` and `flint_attention_prefill_f32_bf16sinks`).
    - Pinned MLX 0.32.4 Numerical Qualification: All 38 golden test cases passed.
      - 24 RoPE cases: FP32 exact match (diff $\le 3.5 \times 10^{-5}$, max rel $\le 0.0035\%$), BF16 max rel $\le 0.769\%$.
      - 14 Attention cases: FP32 max abs diff $= 0.000000$ (rel L2 $\le 10^{-6}$), BF16 relative L2 error strictly $\le 5 \times 10^{-3}$ (aggregate $L_2 \le 0.5\%$, measured $0.0029 - 0.0047$, with individual element ULPs reaching hundreds near zero due to reduction tree differences).
    - Canonical Metal shader source equivalence test: verified zero drift between on-disk `flint_attention.metal` / `flint_rope.metal` and their embedded Swift strings.
    - Apple M5 Hardware Benchmark:
      - Decode Attention (1 token, 513 KV cache): P50 = 183.37 µs (Min: 181.63 µs, P95: 807.42 µs, Max: 810.25 µs, Nominal: 5.82 GB/s).
      - RoPE Decode (32 heads, 1 token): P50 = 1.42 µs (Min: 1.37 µs, P95: 1.46 µs, Max: 1.50 µs).
  - `FlintRouterKernel` & Metal Shading Language (`flint_router.metal`) (F2-03):
    - Top-6 Sigmoid Router for 47 MoE layers (layers 2 through 48; layers 0 and 1 are dense SwiGLU MLPs and have NO router gate).
    - Resident BF16 router gate weight tensor `[256, 2048]` (524,288 elements = 1,048,576 bytes per layer, 47 MiB total).
    - Vectorized 4-wide dot product loop in FP32 with 256 threads/threadgroup (1 thread/expert).
    - Sigmoid scoring function: $\text{score}_e = 1 / (1 + \exp(-x \cdot W_e^T))$.
    - Deterministic top-6 selection with documented tie-breaking policy: lower expert ID is strictly chosen first on ties.
    - Normalized weights: $w_k = s_k / (\sum_{m=0}^5 s_m + 10^{-20})$ such that $\sum w_k = 1.0$.
    - Zero GPU heap allocations, zero CPU roundtrips.
    - Direct zero-copy consumption of resident weights. Zero weight mutation (SHA-256 verified invariant before and after execution on layers 2, 24, 48).
    - Pinned MLX 0.32.4 Numerical Qualification: All 10 golden test cases passed (FP32 and BF16 exact matches across layers 2, 24, 48, all-zeros tie-breaking, saturated positive, saturated negative, and multi-token sequence).
    - Actual MLX router oracle (`mlx_vlm.models.flint.language.SparseMoeBlock`) qualification: verified set equality and per-expert weights across FP32, BF16, near-ties, saturation, zeros, sequence (T=8), and batch > 1 (batch=2, T=4).
    - Apple M5 Hardware Benchmark:
      - Decode Router (1 token, BF16): P50 = 31 µs (Throughput: ~32,000 tok/s).
      - Prefill Router (8 tokens, BF16): P50 = 29 µs (Throughput: ~270,000 tok/s).
      - Prefill Router (64 tokens, BF16): P50 = 118 µs (Throughput: ~540,000 tok/s).
      - Decode Router FP32 (1 token): P50 = 39 µs (Throughput: ~25,000 tok/s).
  - `FlintDequantizeKernel` & Metal Shading Language (`flint_dequantize.metal`) (F2-04):
    - 3-bit and 4-bit affine dequantization for MLX packed UInt32 weights with group size 64 and resident BF16 scales & biases.
    - Mathematical formula: $\text{weight} = \text{scale} \times \text{int\_val} + \text{bias}$, computed in FP32 then cast to BF16 or FP32.
    - 4-bit packing: 8 values packed per 32-bit UInt32 word ($64 \times 4 / 32 = 8$ words per group).
    - 3-bit continuous bit-stream: continuous bit packing crossing 32-bit word boundaries ($64 \times 3 / 32 = 6$ words per group).
      - Formula: `combined = (ulong(w_hi) << 32) | ulong(w_lo)`, `int_val = (combined >> (bit_idx % 32)) & 0x7`.
    - Supports routed expert projections: `gate_proj` & `up_proj` `[256, 512, 192]` UInt32 ($2048 \times \text{packed}$), `down_proj` `[256, 2048, 48]` UInt32.
    - Supports resident shared expert projections: 4-bit INT4 group size 64 (`[1024, 256]` and `[2048, 128]` UInt32).
    - Bounded execution memory: dequantizes on-demand into temporary preallocated scratch buffer (only 6 MiB per expert, 36 MiB for all Top-6 experts), zero permanent memory expansion.
    - Direct zero-copy consumption of resident weights. Zero weight mutation (SHA-256 verified invariant before and after execution).
    - Review #009 Cleanup: Added raw-UInt16 bit-exact equality assertions for all 13 BF16 cases (100% bit-exact 0 mismatches), unrounded FP32 oracle with $0.000000$ error across all 5 FP32 cases.
    - Pinned MLX 0.32.4 Numerical Qualification: All 18 golden test cases passed (13 BF16 cases 100% bit-exact, 5 FP32 cases exact match).
    - Apple M5 Hardware Benchmark:
      - Shared Expert Gate Proj (4-bit, [1024, 2048]): P50 = 32.25 µs (166.63 GB/s).
      - Routed Expert Gate Proj (3-bit, [512, 2048]): P50 = 17.71 µs (144.33 GB/s).
      - Routed Expert Down Proj (3-bit, [2048, 512]): P50 = 17.67 µs (144.67 GB/s).
      - Routed Expert Gate Proj FP32 (3-bit, [512, 2048]): P50 = 22.25 µs (209.13 GB/s).
  - `FlintReductionKernel` & Metal Shading Language (`flint_reduction.metal`) (F2-05):
    - Fixed Router-Order Expert Reduction for 47 MoE layers (layers 2 through 48).
    - Combines contributions of Flint's 6 router-selected experts strictly in fixed router order $k = 0, 1, 2, 3, 4, 5$.
    - Input layout: `contributions[batch, tokens, 6, 2048]`, `weights[batch, tokens, 6]`, optional `sharedExpert[batch, tokens, 2048]`.
    - Output layout: `output[batch, tokens, 2048]`.
    - Mathematical formula: $\text{output}_{b,t,d} = \sum_{k=0}^5 w_{b,t,k} c_{b,t,k,d} + \text{shared\_expert}_{b,t,d}$.
    - Explicit FP32 accumulation semantics with BF16 or FP32 output cast boundary.
    - Unweighted shared expert addition performed strictly after router-selected expert accumulation.
    - Absolute Residency Invariance: Output is strictly addressed by router slot $k \in [0..5]$, never by thread completion order, arrival order, or cache residency.
    - Residency Invariance Contract Verified: Tested Natural, Reverse, Shuffled, and Simulated Cold/Warm arrival orders across multiple randomized trials; all produce 100% bitwise-identical output (exact SHA-256 match, 0 byte differences across 4096 bytes).
    - 50-Run Bitwise Determinism Verified: 50 consecutive dispatches produce identical SHA-256 digests.
    - Pinned MLX 0.32.4 Numerical Qualification: All 8 golden test cases passed (BF16 cases 100% bit-exact across thousands of dimensions, FP32 max abs diff $0.000000$, relative L2 $\le 3.6 \times 10^{-8}$).
    - Apple M5 Hardware Benchmark:
      - Single-token reduction (BF16, with shared expert): P50 = 1.75 µs (Min: 1.67 µs, P95: 1.83 µs, Max: 8.17 µs, Mean: 1.84 µs ± 0.64 µs, 18.74 GB/s effective).
      - Steady-state heap allocations: strictly 0 per dispatch.
  - Real Metal vs MLX Router Diagnostic on 100 Deterministic Tokens (Review #010 Task 2.A):
    - Replaced NumPy script with real Metal GPU dispatch (`FlintRouterKernel`) vs actual pinned MLX `SparseMoeBlock.route` with identical input and gate weights (`router_100_tokens_mlx.json`).
    - 96 / 100 tokens (96.0%) match 100% identically in expert set (6/6).
    - 90 / 100 tokens (90.0%) match 100% in exact rank ordering.
    - Max weight difference on matched experts: $0.000178$ ($< 2 \times 10^{-4}$).
    - Boundary Mismatches at Rank 5: exactly 4 tokens (Tokens 11, 25, 27, 65).
    - For ALL 4 tokens, MLX 6th vs 7th score margin was identically $0.00000000$ (genuinely tied down to float32 precision). Metal resolved ties deterministically by lower expert ID!
  - MoE Reduction Multi-Batch Difference Characterization (Review #010 Task 2.B):
    - Identified exact differing element in `multi_batch_multi_token_bf16`: Index 7161, Metal=`0xb696`, MLX=`0xb695`, difference = 1 ULP ($2.98 \times 10^{-8}$ absolute diff, $2.95 \times 10^{-10}$ relative L2).
    - Root cause: MLX `mx.sum` uses a pairwise tree reduction schedule in batched tensors that can differ by 1 ULP at rounding boundaries from sequential sum.
    - Tightened test assertions to evidence-based thresholds: max abs diff $\le 10^{-4}$, relative L2 $\le 10^{-5}$, bit mismatches $\le 1 / 8192$.
  - Residual Documentation Correction (Review #010 Task 2.C):
    - Corrected stale text in `NUMERICAL_CONTRACT.md` and `DEVELOPMENT_LOG.md` to 13 BF16 cases and 5 FP32 cases.
  - F2-06: Native Metal Dense and Shared Expert SwiGLU MLP Kernel & Numerical Qualification:
    - Dense MLP (Layers 0, 1): `hidden_dim = 2048`, `intermediate_dim = 6144`.
    - Shared Expert MLP (Layers 2..48): `hidden_dim = 2048`, `intermediate_dim = 1024`.
    - Integrated with 4-bit affine dequantized weights (group size 64).
    - FP32 SIMD-coalesced GEMM reduction with 100% coalesced memory loads.
    - Pointwise and fused SwiGLU: $\text{SiLU}(\text{gate}) \odot \text{up}$ evaluated in FP32.
    - Zero heap allocations during steady-state forward passes.
    - Preallocated bounded scratch buffers: 48.0 MiB weight scratch, 4.5 MiB activation scratch.
    - Numerical Qualification vs MLX 0.32.4: 11 test cases across all 5 intermediate pipeline stages (`gate_proj`, `up_proj`, `silu_gate`, `gated_product`, `down_proj`) passed. Max abs diff $\le 10^{-5}$ ($0.000000$ to $0.000001$), relative L2 $\le 2.91 \times 10^{-4}$ in BF16, $\le 4.0 \times 10^{-7}$ in FP32.
    - Apple M5 Hardware Benchmark:
      - Dense L0 Decode ($T=1, I=6144$): P50 = 1259.67 µs (Min: 1253.37 µs, P95: 1579.38 µs)
      - Dense L0 Prefill ($T=8, I=6144$): P50 = 2379.63 µs (Min: 2355.79 µs, P95: 2424.21 µs)
      - Shared L2 Decode ($T=1, I=1024$): P50 = 163.37 µs (Min: 163.08 µs, P95: 163.63 µs)
      - Shared L2 Prefill ($T=8, I=1024$): P50 = 373.46 µs (Min: 371.83 µs, P95: 374.25 µs)
  - Test Suite Breakdown: **116 tests passing across 11 suites in 19.43s (clean ./Scripts/test.sh)**:
    - `FlintDecoderLayerTests`: 9 tests (Layer 0 qualification vs MLX, Layer 2 MoE qualification vs MLX, 50-run determinism, canonical shader equivalence, bounds validation, Apple M5 benchmarks).
    - `FlintMLPKernelTests`: 9 tests (MLX qualification, CPU analytical contract, resident weights integration, 50-run determinism, zero allocation verification, parameter/buffer bounds, M5 benchmark, shader equivalence, concurrency exclusivity).
    - `FlintReductionKernelTests`: 8 tests (MLX qualification, CPU analytical contract, residency invariance, 50-run determinism, magnitude/cancellation, buffer bounds, M5 benchmark, shader equivalence).
    - `FlintDequantizeKernelTests`: 7 tests.
    - `FlintRouterKernelTests`: 8 tests (including real Metal vs MLX 100-token diagnostic).
    - `FlintAttentionKernelTests`: 10 tests.
    - `FlintRMSNormKernelTests`: 7 tests.
    - `FlintResidentLoaderTests`: 9 tests.
    - `FlintSafetensorsFileTests`: 28 tests.
    - `FlintModelDirectoryTests`: 20 tests.
    - `FlintInferenceLeaseTests`: 2 tests.
- **F3 (First Real Flint Token Forward)**: **F3-01A (Layer 0 Dense Decoder) & F3-01B (Layer 2 MoE Decoder) COMPLETED**.
  - `FlintDecoderLayerRunner` & Metal Shading Language (`flint_layer.metal`):
    - Implemented unified forward runners for Layer 0 (dense) and Layer 2 (MoE).
    - Layer 0 validates all 10 captured intermediate stages against pinned MLX 0.32.4 (`norm1`, `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `attn_out`, `res1`, `norm2`, `mlp_out`, `final_out`):
      - Uses full causal attention with learned sinks and **NO RoPE** (`0 in no_rope_layer_indices`).
      - Dense SwiGLU MLP ($2048 \to 6144 \to 2048$).
      - Decodes at $T=1$ and prefills at $T=4, T=8$.
      - 50/50 repeat run bitwise determinism confirmed (SHA-256: `ab289efcae1913d144ce991e3bcbd8b57ba70c929694debf273c6a30423c5e2e`).
      - Apple M5 Benchmark: Decode P50 = 9.55 ms (104.7 tok/s), Prefill $T=4$ P50 = 20.32 ms, 0 steady-state heap allocations.
    - Layer 2 validates all 14 captured intermediate stages against pinned MLX 0.32.4 (`norm1`, `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `attn_out`, `res1`, `norm2`, `router_scores`, `routed_contributions`, `shared_out`, `moe_out`, `final_out`):
      - Uses sliding-window attention (window 513) with learned sinks and **split-half RoPE** (`traditional=False`, base 10000.0).
      - Top-6 Sigmoid Router into 6 routed expert slots.
      - Dynamic synchronous streaming of 3-bit weights from `model-00001-of-00004.safetensors` via `FlintSafetensorsFile` into preallocated temporary buffers (`expertBuffers[0..5]`, $6 \times 1.3125\text{ MiB} = 7.875\text{ MiB}$ total).
      - Strictly 0 bytes of routed expert weights retained in resident storage.
      - Shared Expert 4-bit SwiGLU forward ($2048 \to 1024 \to 2048$) evaluated concurrently.
      - Fixed router-order FP32 reduction across slots $0..5$ plus shared expert.
      - Multi-token prefill safety: token multi-buffering with per-token command buffer completion synchronization, eliminating CPU/GPU write races.
      - 50/50 repeat run bitwise determinism confirmed (SHA-256: `ab289efcae1913d144ce991e3bcbd8b57ba70c929694debf273c6a30423c5e2e`).
      - Apple M5 Benchmark: Decode P50 = 9.04 ms (110.6 tok/s), Prefill $T=4$ P50 = 23.65 ms, 0 steady-state heap allocations.
- **F4 (Bounded Expert Streaming & Full Model Pipeline)**: Pending.

## Strict Workspace Isolation Rules
1. **Flint Operations**: Execute all edits, builds, tests, and git operations EXCLUSIVELY in `/Users/tanishq/dev/product/flint-fieldfare-isolated`.
2. **Qwen Preservation**: NEVER write, pull, restore, checkout, or modify `/Users/tanishq/dev/product/qwen-fieldfare`.
3. **Symlink Invariant**: Leave the existing `/Users/tanishq/dev/product/flint-fieldfare` symlink untouched until explicit user/reviewer authorization is granted.

## Key Architectural Invariants
1. **Never use Qwen Recurrent/DeltaNet code for Flint**: Flint is a pure Transformer MoE (14 full-attention, 35 sliding-attention with window 513). All 49 layers feature self-attention.
2. **Fixed Router-Order Reduction**: Always accumulate into `contributions[position * 2048 + row]` where `position \in [0, 5]` and sum in order $0 \to 5$. Never accumulate by cache residency.
3. **Learned Attention Sinks**: `softmax_offset[32]` is an extra logit in softmax denominator without a value vector.
4. **Dedicated Lease Lock**: `/tmp/flintfieldfare-inference-\(getuid()).lock`.
5. **Never Commit Weights**: Checkpoint files remain strictly external.

## Immediate Next Steps
1. Submit Review #011 Completion Report (F2-06 Closure, F3-01A Layer 0 Dense Decoder, F3-01B Layer 2 MoE Decoder).
2. Request authorization for Milestone F3-02 (Full 49-Layer Model Pipeline Integration & End-to-End Generation).
