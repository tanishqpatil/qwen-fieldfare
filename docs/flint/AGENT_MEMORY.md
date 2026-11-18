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
- **F2 (Mathematical Primitives & Metal Kernels)**: **F2-01 COMPLETED**.
  - `FlintRMSNormKernel` & Metal Shading Language (`flint_rmsnorm.metal`):
    - Implemented native Metal compute kernels for width 2048 (hidden state, 256 threads, 8-warp SIMD reduction) and width 128 (head dim for per-head `q_norm` and `k_norm`, 128 threads, 4-warp SIMD reduction).
    - General fallback kernel for arbitrary dimensions.
    - FP32 internal accumulation for mathematical stability matching reference standard.
    - Supports both BF16 and FP32 activation dtypes; BF16 checkpoint weights.
    - Direct zero-copy consumption of resident weights (`readOnlyPointer` / `metalBuffer`). Zero weight mutation (SHA-256 verified invariant before and after execution).
    - Pre-compiled pipeline state cache and zero per-invocation heap allocations in command encoding.
    - Full parameter and buffer bounds validation prior to dispatch.
    - Allocation safety: `FlintResidentBuffer` validates `buf.length >= byteCount`, requires `.storageModeShared`, and enforces non-null pointers (`invalidMetalBuffer`).
  - Independent Reference Qualification (vs MLX 0.32.4):
    - Evaluated across 20 deterministic golden test cases spanning widths 2048 and 128, BF16 and FP32, batch sizes 1 to 32 rows, epsilon values $10^{-6}$ and $10^{-5}$, and boundary inputs (zeros, ones, small/large magnitudes, mixed signs).
    - FP32 exact match: Max absolute difference = $0.000000$ (0.0 ULP, RMS error $0.00\text{e}+00$).
    - BF16 match: Max ULP difference $\le 1$ ($0.0625$ at $|y| \approx 8.0$, $0.015625$ at $|y| \approx 1.0$), max relative difference $\le 0.781\%$ ($1/128$, the mathematical limit of 7-bit BF16 mantissa).
    - 20/20 cases passed.
  - Apple M5 Hardware Performance Benchmark:
    - Hidden Norm Decode (1 token, dim 2048): P50 = 14.00 µs (Min: 13.87 µs, P95: 19.17 µs).
    - Hidden Norm Prefill (128 tokens, dim 2048): P50 = 25.25 µs (Min: 23.75 µs, P95: 26.00 µs, Throughput: 41.86 GB/s).
    - Q Norm (32 heads, dim 128): P50 = 6.42 µs (Min: 6.29 µs, P95: 6.50 µs).
    - K Norm (4 heads, dim 128): P50 = 6.37 µs (Min: 6.25 µs, P95: 6.42 µs).
  - Test Suite: **64 tests passing across 5 suites in 0.542 seconds**.
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
1. Submit Review #005 Completion Report (F2-01 Native Metal RMSNorm Kernel and Numerical Qualification).
2. Request authorization for Milestone F2-02 (Attention Sinks & SDPA Kernel).
