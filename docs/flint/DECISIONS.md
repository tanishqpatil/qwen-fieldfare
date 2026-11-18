# FlintFieldfare Architecture Decisions (ADR)

## ADR-F001: Isolated Flint Implementation Line (Option A)

- **Date**: 2026-11-18
- **Status**: Accepted
- **Context**: QwenFieldfare contains a functioning Swift/Metal inference runtime, NIO server, and macOS UI for Qwen 3.6 35B-A3B. Flint 40B-A2.5B has fundamentally different architecture (no Gated DeltaNet recurrent state, 49 layers with 14 full/35 sliding attention layers, learned attention sinks, 256 routed experts with top-6 sigmoid routing, 3-bit/4-bit quantization, distinct tokenizer and Jinja tool template).
- **Decision**: Implement Flint as isolated targets (`FlintFieldfareCore`, `FlintFieldfareBench`, `FlintFieldfareServer`) within the repository. Reuse proven low-level infrastructure (`MetalContext`, bounded `Safetensors` header reading, NIO HTTP mechanics, Jinja chat templates) while completely isolating model-specific kernels, runners, and loaders.
- **Consequences**:
  - Qwen build targets and existing regression tests remain 100% operational and undisturbed.
  - No risk of breaking Qwen or leaking Qwen recurrent state assumptions into Flint.
  - Clear, focused development path with independent verification gates.

## ADR-F002: Fixed Router-Order Expert Reduction from Inception

- **Date**: 2026-11-18
- **Status**: Accepted
- **Context**: In QwenFieldfare, earlier iterations partitioned expert accumulation by cache residency (hits vs misses), resulting in non-associative floating-point discrepancy and cache-history sensitivity. S3-00c established that storing intermediate outputs by router position and reducing in fixed router order eliminated this variance.
- **Decision**: Design FlintFieldfare's MoE reduction from day one with fixed router-order accumulation. Each of the top-6 experts writes its weighted down-projection into `contributions[position * 2048 + row]` where `position \in [0, 5]`. A dedicated combine kernel sums router positions in fixed order $0 \dots 5$.
- **Consequences**:
  - Identical outputs whether an expert is served from cache hit or miss.
  - Zero residency-dependent drift.

## ADR-F003: Dedicated Flint Inference Lease Lock

- **Date**: 2026-11-18
- **Status**: Accepted
- **Context**: QwenFieldfare uses an exclusive POSIX file lock at `/tmp/qwenfieldfare-inference-\(uid).lock` to ensure only one model process runs at a time. If Flint shared this lock path, running a Qwen qualification job would block Flint tests, or Flint would block Qwen.
- **Decision**: Use a dedicated lock path for Flint: `/tmp/flintfieldfare-inference-\(getuid()).lock`.
- **Consequences**:
  - Eliminates false test failures when a Qwen process is active.
  - Prevents multiple concurrent Flint instances from exhausting unified memory.

## ADR-F004: Dual 3-Bit and 4-Bit Routed Expert Quantization Support

- **Date**: 2026-11-18
- **Status**: Accepted
- **Context**: The official `RecursiveSI/flint-int4` uses 4-bit routed experts, while the popular local community build `mlx-community/flint-mixed-3-4bit` uses 3-bit routed experts (192 uint32s per 2048 weights) to reduce peak memory from 22.6 GB to 17.8 GB.
- **Decision**: Detect routed expert bit-depth from `config.json` (`quantization["model.layers.<idx>.mlp.switch_mlp.gate_proj"]["bits"] == 3` vs `4`) and support both 3-bit contiguous packing and 4-bit nibble packing in the Metal dequantization kernels.
- **Consequences**:
  - Works out-of-the-box with both official INT4 and the 3-bit mixed checkpoint.

## ADR-F005: Learned Attention Sink Softmax Normalization

- **Date**: 2026-11-18
- **Status**: Accepted
- **Context**: Flint incorporates learned per-head scalars (`softmax_offset` of shape `[32]`) into attention. Rather than attending to a dummy token value vector, this scalar acts as an extra logit in the softmax denominator, absorbing excess attention weight.
- **Decision**: Integrate the sink term directly into the Metal attention kernel's online softmax accumulator.
- **Consequences**:
  - Exact mathematical equivalence to Transformers and MLX without allocating dummy KV memory.
