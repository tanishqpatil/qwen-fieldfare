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
  7. Formulated Initial Memory Budget & Targets:
     - Resident weights: 1.28 GB (estimated).
     - Expert cache (1024 slots): 1.31 GB (budgeted).
     - KV cache (8192 tokens): 0.26 GB (budgeted).
     - Target working set: ~3.2 GB.
  8. Created all foundational Flint documentation under `docs/flint/`.
- **Verdict on F0**: Initial audit and architectural design completed.

---

## 2026-11-18 — Review #001 Response & F0 Revision / F1 Hardening

- **Author**: Inference Systems Architect
- **Actions Completed**:
  1. **Repository Isolation Audit & Clone Setup**:
     - Verified target paths: `qwen-fieldfare` (`/Users/tanishq/dev/product/qwen-fieldfare`) and symlink `flint-fieldfare` (`/Users/tanishq/dev/product/flint-fieldfare`).
     - Established physically independent Git clone at `/Users/tanishq/dev/product/flint-fieldfare-isolated` checked out on `feature/recursive-flint-40b-a2p5b`.
     - Left the existing symlink and `qwen-fieldfare` directory undisturbed pending user authorization.
  2. **Empirical All-Resident M5 Benchmark (Task 2)**:
     - Executed full MLX benchmark (`benchmark_flint_mlx.py`) using `mlx` 0.32.4 and `mlx_vlm` 0.7.12 on `/Users/tanishq/models/flint-mixed-3-4bit`.
     - Measured load time: 4.39 s.
     - Measured active memory: 16.61 GB. Peak memory: 16.68 GB (short prompt), 17.68 GB (8K context).
     - Measured swap impact: Swap grew by +657.63 MB (from 767 MB to 1,425 MB), proving memory pressure on 32 GB RAM.
     - Short prompt throughput: ~161.7 tok/s prefill, ~70.0 tok/s decode.
     - 8K prompt throughput: ~1,316.9 tok/s prefill, ~60.2 tok/s decode.
     - Derived SSD streaming bandwidth formula vs hit rate: 370.1 MiB/token at cold start down to 18.5 MiB/token at 95% hit rate.
  3. **Hardened FlintModelDirectory (Task 3)**:
     - Removed all silent fallback defaults for mandatory fields (`hidden_size`, `vocab_size`, `num_hidden_layers`, etc.).
     - Strictly enforced `model_type == "flint"` and `layer_types` arrangement (14 full-attention, 35 sliding-attention).
     - Implemented comprehensive quantization validation across all 141 routed-expert projections (47 layers $\times$ 3 projections), rejecting partial or inconsistent bit-depths.
  4. **Improved Test Suite (Task 4)**:
     - Added 5 new focused negative tests covering missing mandatory fields, invalid layer types, mismatched attention arrangements, incorrect expert counts, inconsistent quantization overrides, and unsafe shard path traversal.
     - Updated live checkpoint test to use `FLINT_TEST_MODEL_DIR` with documented fallback `/Users/tanishq/models/flint-mixed-3-4bit`, explicitly recording an issue if absent.
     - Executed `swift test`: **12 tests passed across 2 suites in 0.006s**.
  5. **Updated Documentation (Task 5)**:
     - Updated `MEMORY_PLAN.md`, `PERFORMANCE_BASELINES.md`, `DECISIONS.md` (ADR-F006), `MODEL_SPEC.md`, `ARCHITECTURE.md`.
     - Clearly distinguished empirical measurements from theoretical streaming design estimates.

---

## 2026-11-18 — Review #002 Response: Strict Quantization Contract, Buffer Safety & Baseline Evidence

- **Author**: Inference Systems Architect
- **Actions Completed**:
  1. **Strict Repository Isolation Enforced**:
     - Inspected `qwen-fieldfare` (`pwd -P`: `/Users/tanishq/dev/product/qwen-fieldfare`), confirming clean status on `feature/recursive-flint-40b-a2p5b` (intended Qwen branch: `origin/feature/local-coding-agent-server`).
     - Maintained zero write operations in `qwen-fieldfare`.
     - Preserved symlink `/Users/tanishq/dev/product/flint-fieldfare -> /Users/tanishq/dev/product/qwen-fieldfare` without modification.
     - Confirmed all implementation and tests execute strictly in `/Users/tanishq/dev/product/flint-fieldfare-isolated`.
  2. **Quantization Validator Hardened with Strict Case A / Case B Bifurcation**:
     - Hardened `FlintModelDirectory.swift` to enforce strict bifurcation:
       - **Case A (Official INT4 Checkpoint)**: Base 4-bit, group size 64, exactly 0 `switch_mlp` overrides. Inherits uniform 4-bit.
       - **Case B (Mixed 3/4-Bit Checkpoint)**: Validates that all 141 required `switch_mlp` projection overrides (47 MoE layers $\times$ 3 projections) exist, agree on group size (64), and agree on bit depth (3 or 4).
       - Enforced rejection of partial overrides (1..140), extraneous projections, unsupported bit depths, or unsupported group sizes.
     - Enforced mandatory explicit metadata:
       - `model_type` must be `"flint"` (missing or invalid fails).
       - `norm_topk_prob` must be `true` (missing or false fails).
       - `no_rope_layer_indices` must contain exactly 14 unique layer indices (duplicates fail) matching `expectedFullAttentionIndices`.
       - Embedding quantization is explicitly validated (bits 8 or 4).
  3. **Safe File Buffer Reading (`pread`)**:
     - Replaced ephemeral array buffer references in `readFileBounded` with safe contiguous buffer access using `Data.withUnsafeMutableBytes` and `baseAddress.advanced(by:)`.
     - Added explicit `errno` checking on negative `pread` return values.
  4. **Reproducible Baseline Evidence Preserved**:
     - Created `docs/flint/evidence/2026-11-18-baseline/`.
     - Preserved `benchmark_flint_mlx.py` and `mlx_flint_benchmark_results.json`.
     - Documented Python environment, MLX versions, CLI invocation, synchronization methodology (`mx.eval`), timing breakdown, and swap metrics in `docs/flint/evidence/2026-11-18-baseline/README.md`.
  5. **Documentation Rigor & Claims Clarification**:
     - Updated `MEMORY_PLAN.md`, `PERFORMANCE_BASELINES.md`, `DECISIONS.md` (ADR-F007), and `ARCHITECTURE.md`.
     - Explicitly distinguished system memory paging correlation from thrashing causality.
     - Labeled the **~3.2 GiB** figure strictly as an **architectural design target budget**.
     - Labeled **70–85% cache hit rates** explicitly as working hypotheses.
     - Framed all-resident vs streaming as an explicit engineering trade-off.
     - Verified and corrected SSD bandwidth unit conversions.
  6. **Comprehensive Test Suite Expansion**:
     - Added 10 new unit tests covering Case A official INT4, Case A with `mlx_export` embedding bits, Case B partial overrides (140 projections), single override, extraneous projections on dense layers, unsupported group sizes, missing/invalid `model_type`, missing/false `norm_topk_prob`, duplicate `no_rope_layer_indices`, and malformed embedding quantization.
     - Executed `swift test`: **22 tests passed across 2 suites in 0.008s**.


---

## 2026-11-18 — F1-03: Safe, Bounded Safetensors Access & Physical Tensor Validation

- **Author**: Inference Systems Architect
- **Branch**: `feature/recursive-flint-40b-a2p5b`
- **Authorized Directory**: `/Users/tanishq/dev/product/flint-fieldfare-isolated`
- **Target Machine**: Apple MacBook Pro M5, 32 GB unified memory, macOS 27.2.1, Apple Swift 6.4.2.
- **Actions Completed**:
  1. **Safetensors Reader Implementation (`FlintSafetensorsFile.swift`)**:
     - Created `FlintSafetensorsFile.swift` owning POSIX file descriptors (`open` with `O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`, `close(descriptor)` in `deinit`).
     - Implemented bounded header parsing with 100 MB upper bound, 8-byte little-endian header length decoding, UTF-8 non-NUL verification, and arithmetic overflow detection.
     - Implemented range validation: checked for negative offsets, inverted ranges, payload-overflowing offsets, shape-to-dtype byte count mismatches, and zero overlapping allocations across tensors.
     - Implemented safe, thread-safe random-access reads via `preadExact` using contiguous memory buffers.
     - Implemented file identity tracking via `stat`/`fstat` (device, inode, size, mtime, ctime) and `verifyUnchanged()` mutation detection.
  2. **Metadata Cross-Validation (`FlintModelDirectory.crossValidate`)**:
     - Added `openShards(canonicalPath:shardNames:)` returning descriptor-owning files for all referenced shards.
     - Implemented `crossValidate(config:configData:shardFiles:weightMap:)`:
       - Cross-checks `mlx_export.embedding_bits` against `quantization.model.embed_tokens.bits` and `config.embeddingBits`.
       - Cross-checks `mlx_export.routed_expert_bits` against detected `config.routedExpertBits`.
       - Verifies that all weight map entries physically exist in their assigned shards without omissions.
       - Validates representative physical tensor shapes and dtypes against architectural layout.
  3. **Physical Checkpoint Validation & Independent Byte Verification**:
     - Inspected and validated live checkpoint `/Users/tanishq/models/flint-mixed-3-4bit` across all 4 shards (1,751 tensors, 17.84 GB).
     - Confirmed SHA-256 reference digests matching Python / `mlx.core.load` ground truth for 8 empirical tensors:
       - `model.layers.2.self_attn.softmax_offset`: `e55d79b0bd25bc2d585bc4bf6706ae3d6d62e2561c9b0ec8f9918041f6404404`
       - `model.layers.0.input_layernorm.weight`: `f33f5b184b24b4e62c37c757ee2d90396173e9ef2ef28af130b266ca021cc4a5`
       - `model.embed_tokens.weight` (first 256 bytes): `c5b8b890109befaf2efd23d462ece554855d2c73d1c14301a8c62afff75bed71`
       - Layer 2, expert 0 `gate_proj.weight`: `a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820`
       - Layer 20, expert 128 `gate_proj.weight`: `a3fcc07e846e6c78e5a57c7e89870e97bbb56afe2a0ddb7979043c127acbd678`
       - Layer 48, expert 255 `down_proj.weight`: `2e823253c72b37404f447e5656a08995746460f39b417474eeed19078d9fd15b`
       - Layer 2, expert 0 `gate_proj.scales`: `7cc18d29413197ee959987c119bbb6e0ed834af8af8107cb3e9e937d9ee267d7`
       - Layer 2, expert 0 `gate_proj.biases`: `4629cee3a9d13041dafcf227f7ac56e5c2388a2619ed197595e176a1fd1bab2d`
  4. **Physical Tensor Inventory Artifact**:
     - Authored `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` containing shard distribution, dtype counts (1,265 BF16, 486 U32), per-expert record sizes ($1,376,256\text{ B} = 1.3125\text{ MiB}$ for 3-bit vs $1,769,472\text{ B} = 1.6875\text{ MiB}$ for 4-bit), and the exact closed-form byte-offset formula for random-access expert streaming.
  5. **Benchmark & Evidence Cleanup**:
     - Updated `benchmark_flint_mlx.py` with configurable output path, actual `generation_tokens`, and `finish_reason`.
     - Updated `docs/flint/evidence/2026-11-18-baseline/README.md` with verified Darwin kernel 27.2.0 (macOS 27.2.1).
  6. **Unit Test Suite Execution**:
     - Added 23 unit tests in `FlintSafetensorsFileTests.swift`.
     - Executed full test suite: **45 tests passed across 3 suites in 0.017 seconds**.
  7. **Strict Repository Isolation Maintained**:
     - Verified `/Users/tanishq/dev/product/qwen-fieldfare` has 0 modifications and remained completely untouched.
     - Symlink `/Users/tanishq/dev/product/flint-fieldfare` left untouched.
- **Verdict on F1-03**: Safe, bounded Safetensors access and physical tensor validation fully implemented, tested, and documented. Ready for F1-04 authorization.
