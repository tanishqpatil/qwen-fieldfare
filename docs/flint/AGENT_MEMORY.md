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
- **F0 (Audit, Architecture & Baseline)**: **COMPLETED & REVISED (Review #001)**.
  - Empirical all-resident MLX baseline measured on M5 (4.39s load, 17.68 GB peak, +657.6 MB swap, ~70 tok/s decode, ~1,317 tok/s 8K prefill).
  - Derived SSD streaming bandwidth formula vs hit rate (370.1 MiB/tok cold).
  - Clearly separated empirical measurements from theoretical streaming design targets (~3.2 GiB).
  - Architecture decisions ADR-F001 through ADR-F006 recorded in `docs/flint/DECISIONS.md`.
- **F1 (Metadata & Weight Loading Infrastructure)**: **IN PROGRESS (Hardening complete, readiness gate)**.
  - `FlintModelDirectory` hardened: rejects missing mandatory fields, verifies exact 49-layer arrangement (14 full, 35 sliding), and validates all 141 routed-expert projections for consistent bit depth.
  - `FlintInferenceLease` verified: dedicated lock at `/tmp/flintfieldfare-inference-\(uid).lock`.
  - Focused test suite: 12 tests passing across 2 suites.
- **F2 (Mathematical Primitives & Metal Kernels)**: Pending.
- **F3 (First Real Flint Token Forward)**: Pending.
- **F4 (Bounded Expert Streaming)**: Pending.

## Key Architectural Invariants
1. **Never use Qwen Recurrent/DeltaNet code for Flint**: Flint is a pure Transformer MoE (14 full-attention, 35 sliding-attention with window 513).
2. **Fixed Router-Order Reduction**: Always accumulate into `contributions[position * 2048 + row]` where `position \in [0, 5]` and sum in order $0 \to 5$. Never accumulate by cache residency.
3. **Learned Attention Sinks**: `softmax_offset[32]` is an extra logit in softmax denominator without a value vector.
4. **Dedicated Lease Lock**: `/tmp/flintfieldfare-inference-\(getuid()).lock`.
5. **Never Commit Weights**: Checkpoint files remain strictly external.

## Immediate Next Steps
1. Review #001 gate review with user.
2. Receive authorization on workspace transition / symlink swap to make `/Users/tanishq/dev/product/flint-fieldfare` the official clone.
3. Proceed to F1-03 & F1-04: Implement `FlintResidentLoader.swift` and `FlintSafetensorsFile`.
