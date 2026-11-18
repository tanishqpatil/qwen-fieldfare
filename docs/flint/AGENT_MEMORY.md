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
- **F1 (Metadata & Weight Loading Infrastructure)**: **F1-01, F1-02, F1-03 COMPLETED**.
  - `FlintModelDirectory` (F1-01): Strict Case A (INT4) & Case B (141 overrides) validation, safe pread contiguous buffers.
  - `FlintInferenceLease` (F1-02): Dedicated lock at `/tmp/flintfieldfare-inference-\(uid).lock`.
  - `FlintSafetensorsFile` & Cross-Validation (F1-03):
    - POSIX descriptor ownership (`O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`, `close(descriptor)` on deinit).
    - Bounded header parsing (100 MB max bound, 8-byte LE header length decoding, UTF-8 non-NUL verification).
    - Offset & range validation (negative, out-of-bounds, inverted, arithmetic overflow, and zero overlapping allocations across tensors).
    - Safe random-access `preadExact` using contiguous memory buffers.
    - Identity verification via `stat`/`fstat` and mutation detection via `verifyUnchanged()`.
    - Cross-validation against config metadata (`mlx_export` vs `quantization` vs Safetensors shapes/dtypes).
    - Empirical validation of live checkpoint `/Users/tanishq/models/flint-mixed-3-4bit` (4 shards, 1,751 tensors, 17.84 GB).
    - Bitwise SHA-256 match against independent Python/MLX ground truth across 8 empirical tensors.
    - Physical tensor inventory report: `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` ($1,376,256\text{ B} = 1.3125\text{ MiB}$ per 3-bit expert, closed-form offset formula).
  - Test Suite: **50 tests passing across 3 suites in 0.019 seconds**.
- **F1-04 (Resident Weight Extraction & Storage)**: Pending authorization.
- **F2 (Mathematical Primitives & Metal Kernels)**: Pending.
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
1. Submit Review #004 Safety Closure completion report.
2. Await authorization for F1-04 (`FlintResidentLoader.swift`).
