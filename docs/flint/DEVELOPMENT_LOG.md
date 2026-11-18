# FlintFieldfare Append-Only Development Log

## 2026-11-18 — F0: Initialization, Architecture Audit & Environment Verification

- **Author**: Inference Systems Architect
- **Branch**: `feature/recursive-flint-40b-a2p5b` (branched from `origin/feature/local-coding-agent-server` at `7370a1d`).
- **Target Machine**: Apple MacBook Pro M5, 32 GB unified memory, macOS 27.2.1, Apple Swift 6.4.2.
- **Actions Completed**:
  1. Inspected hardware: Confirmed 34,359,738,368 bytes (32 GiB) unified memory and 114 GiB available disk.
  2. Inspected processes: Detected active background Qwen prefix qualification run (PID 17705) in separate directory `/Users/tanishq/dev/github/turbo-fieldfare`. Confirmed isolation from our checkout.
  3. Located Flint model checkpoint: Found `/Users/tanishq/models/flint-mixed-3-4bit` (17.84 GB on disk, 4 shards).
  4. Header & tensor audit: Wrote `audit_checkpoint.py` to parse Safetensors headers without loading tensor payloads. Discovered exact quantization:
     - `embed_tokens`: U32 shape `[150144, 512]` -> INT8 affine, group size 64.
     - Attention projections & dense/shared MLPs: INT4 affine, group size 64.
     - Routed experts: U32 shape `[256, 512, 192]` -> 3-bit affine (6 uint32 words per 64 weights), group size 64.
     - Router gate: BF16 shape `[256, 2048]`.
     - Norms & attention sinks: BF16.
  5. Inspected authoritative reference code:
     - Hugging Face `transformers/models/flint/modeling_flint.py`: Verified learned attention sinks logic (concatenated extra logit in softmax denominator without value vector), sigmoid router with normalized scores, ungated shared expert.
     - `mlx_vlm/models/flint/`: Verified 14 full-attention layers without RoPE, 35 sliding-attention layers (window 513) with RoPE (`traditional=False`), dense MLPs on layers 0 and 1, sparse MoE on layers 2..48.
  6. Tested bit packing in MLX (Python): Proved exact bitstream layout for 3-bit quantization (32 elements per 3 uint32 words, 64 elements per 6 uint32 words).
  7. Formulated Memory Plan:
     - Resident weights: 1.28 GB.
     - Expert cache (1024 slots): 1.31 GB.
     - KV cache (8192 tokens): 0.26 GB.
     - Total working set: ~3.2 GB. Proven highly viable on 32 GB M5.
  8. Created all foundational Flint documentation under `docs/flint/`:
     - `ARCHITECTURE.md`, `TODO.md`, `AGENT_MEMORY.md`, `DECISIONS.md`, `MODEL_SPEC.md`, `MODEL_PROVENANCE.md`, `NUMERICAL_CONTRACT.md`, `MEMORY_PLAN.md`, `PERFORMANCE_BASELINES.md`, `AGENT_INTEGRATION.md`, `RESEARCH_SOURCES.md`.
- **Verdict on F0**: Audit and architectural design complete. All gates verified. Ready to proceed to F1 implementation.

## 2026-11-18 — F1: Metadata & Validation Infrastructure Implementation

- **Implemented Components**:
  1. `FlintInferenceLease.swift`: Dedicated POSIX non-blocking advisory file lock at `/tmp/flintfieldfare-inference-\(uid).lock`. Tested mutual exclusion and release. Completely decoupled from QwenFieldfare lock.
  2. `FlintModelDirectory.swift`: Comprehensive validator for Flint checkpoint directories:
     - Enforces presence and non-escaping symlinks for `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.
     - Bounded file reading (max 16 MB metadata bound).
     - Validates `FlintForCausalLM`, 49 layers, 2048 hidden size, 6144 intermediate size, 513 sliding window, 256 experts, top-6 routing.
     - Validates exact layer composition: 14 full-attention layers without RoPE, 35 sliding-attention layers with RoPE.
     - Detects routed expert quantization (3-bit mixed vs 4-bit reference) and 8-bit embedding.
     - Validates shard presence, safety, and critical tensor mappings in `weight_map`.
  3. Integrated `FlintFieldfareCore` and `FlintFieldfareCoreTests` into `Package.swift`.
  4. Executed `swift test --filter FlintFieldfareCoreTests`: 7 tests passed (0.005s), including real-checkpoint validation of `/Users/tanishq/models/flint-mixed-3-4bit`.
- **Verdict on F1 Start**: F1-01, F1-02, and F1-05 are successfully implemented and verified. Ready for F1-03 (Safetensors reader) and F1-04 (Resident weights).
