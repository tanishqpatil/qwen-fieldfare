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
- **F2 (Mathematical Primitives & Metal Kernels)**: **F2-01 & F2-02 COMPLETED**.
  - `FlintRMSNormKernel` & Metal Shading Language (`flint_rmsnorm.metal`):
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
    - Sequence boundary verification: verified across boundary lengths 1, 2, 512, 513, 514, and 1024 tokens.
    - Resident Weight Zero-Copy & Immutability: direct execution using resident buffers for Layer 0 (`model.layers.0.self_attn.softmax_offset`) and Layer 2 (`model.layers.2.self_attn.softmax_offset`). SHA-256 before execution == SHA-256 after execution.
    - Pinned MLX 0.32.4 Numerical Qualification: All 32 golden test cases passed.
      - 24 RoPE cases: FP32 exact match (diff $\le 3.5 \times 10^{-5}$, max rel $\le 0.0035\%$), BF16 max rel $\le 0.769\%$.
      - 8 Attention cases: FP32 max abs diff $= 0.000000$ (max rel diff $\le 9.65 \times 10^{-4}$), BF16 max abs diff $\le 0.015625$ (1-2 ULPs, relative L2 error $\le 10^{-3}$).
    - Canonical Metal shader source equivalence test: verified zero drift between on-disk `flint_attention.metal` / `flint_rope.metal` and their embedded Swift strings.
    - Apple M5 Hardware Benchmark:
      - Decode Attention (1 token, 513 KV cache): P50 = 183.37 µs (Min: 181.63 µs, P95: 807.42 µs, Max: 810.25 µs, Nominal: 5.82 GB/s).
      - RoPE Decode (32 heads, 1 token): P50 = 1.42 µs (Min: 1.37 µs, P95: 1.46 µs, Max: 1.50 µs).
  - Test Suite Breakdown: **74 tests passing across 6 suites in 1.673s (clean ./Scripts/test.sh)**:
    - `FlintAttentionKernelTests`: 8 tests (analytical synthetic sink, RoPE invariants, sliding boundaries 1..514, 32 MLX goldens, resident sinks integration & immutability, buffer bounds, M5 hardware benchmark, shader equivalence).
    - `FlintRMSNormKernelTests`: 7 tests (mathematical invariants, bounds/Int.max validation, general fallback dims 16..1024, 20 MLX goldens, resident weight immutability, M5 benchmark, shader equivalence).
    - `FlintResidentLoaderTests`: 9 tests.
    - `FlintSafetensorsFileTests`: 28 tests.
    - `FlintModelDirectoryTests`: 20 tests.
    - `FlintInferenceLeaseTests`: 2 tests.
- **F3 (First Real Flint Token Forward)**: Pending.
- **F4 (Bounded Expert Streaming)**: Pending.

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
1. Submit Review #006 Completion Report (Prerequisite RMSNorm Fixes + F2-02 Attention Sinks, RoPE, and SDPA Qualification).
2. Request authorization for Milestone F2-03 (Sigmoid Top-6 Router Kernel).
