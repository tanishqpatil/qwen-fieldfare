# FlintFieldfare Agent Recovery Memory

This document is the authoritative recovery guide for agents operating on FlintFieldfare. Read this file at session start to resume work seamlessly.

## Mission Context
- **Target Model**: Recursive Flint 40B-A2.5B (MoE: 49 layers, 2 dense MLPs + 47 MoE layers, 256 routed experts/layer, top-6 routing, 1 shared expert).
- **Target Hardware**: Apple MacBook Pro M5, 32 GB unified memory, macOS 27.2.1, Apple Swift 6.4.2.
- **Local Checkpoint**: `/Users/tanishq/models/flint-mixed-3-4bit` (17.84 GB, 3-bit experts, INT4 projections, INT8 embeddings).
- **Git Branch**: `feature/recursive-flint-40b-a2p5b`
- **Base SHA**: `7370a1d` (from `origin/feature/local-coding-agent-server`).

## Current Milestone Status
- **F0 (Audit, Architecture & Environment)**: **COMPLETED**.
  - All source references verified (Transformers `modeling_flint.py`, `mlx_vlm/models/flint/`).
  - Safetensors header audit verified (1,751 tensors, U32/BF16 distribution, group 64 affine).
  - Feasibility verified: ~1.28 GB resident weights, ~1.31 GB expert cache, total ~3.2 GB working set.
  - Architecture decisions ADR-F001 through ADR-F005 recorded in `docs/flint/DECISIONS.md`.
- **F1 (Metadata & Weight Loading Infrastructure)**: **IN PROGRESS**.
  - Target: `FlintModelDirectory`, `FlintInferenceLease`, `FlintSafetensorsHeader`, and unit tests.
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
1. Implement `FlintModelDirectory.swift` to parse and validate `config.json`, verify 49 layers, layer types, `no_rope_layer_indices`, vocab 150144, top-6 routing, and quantization metadata.
2. Implement `FlintInferenceLease.swift` with dedicated lock path.
3. Add unit tests in `Tests/FlintFieldfareCoreTests/FlintModelDirectoryTests.swift`.
4. Register `FlintFieldfareCore` in `Package.swift`.
