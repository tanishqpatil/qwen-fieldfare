# Flint Research Sources & Verification Matrix

This document tracks all authoritative sources, specifications, and codebases consulted for the FlintFieldfare implementation.

## Authoritative Source Matrix

| Source | Role / URL | Revision / Inspected Location | Verification Status | Key Findings |
|---|---|---|---|---|
| **Local Checkpoint** | Local MLX Safetensors directory | `/Users/tanishq/models/flint-mixed-3-4bit` | **VERIFIED** | 17.84 GB, 4 shards. INT8 embeddings (group 64), INT4 attention/dense/shared/head (group 64), 3-bit routed experts (group 64). BF16 norms & router gates. |
| **Official Model Card** | Recursive Flint | `https://huggingface.co/RecursiveSI/flint` | **VERIFIED** | 39.8B total params, 2.5B active params. 49 layers, 2048 hidden, 256 routed experts (top-6), 1 shared expert. |
| **Official INT4 Checkpoint** | Recursive Flint INT4 | `https://huggingface.co/RecursiveSI/flint-int4` | **VERIFIED** | 22.6 GB on disk, 4-bit trained weights with uint8 storage viewable as uint32. |
| **Hugging Face Transformers** | `modeling_flint.py`, `configuration_flint.py` | Local venv: `/Users/tanishq/venvs/mlx-dspark/.../transformers/models/flint/` | **VERIFIED** | Verified explicit attention sink logit concatenation (`sinks` parameter), FP32 sigmoid top-k router with normalization, ungated shared expert, SwiGLU activation. |
| **MLX Implementation** | `mlx_vlm/models/flint/` (`language.py`, `config.py`) | Local venv: `/Users/tanishq/venvs/mlx-dspark/.../mlx_vlm/models/flint/` | **VERIFIED** | Verified layer layout, 14 full-attention layers with no RoPE, 35 sliding-attention layers (window 513) with RoPE (split-half `traditional=False`). |
| **3-Bit Mixed Converter** | `experts3_from_int4.py`, `eval_kl.py` | `/Users/tanishq/models/flint-mixed-3-4bit/` | **VERIFIED** | MSE per-group clip search (clip ratios 0.70..1.00) from Recursive INT4. 3-bit contiguous packing (6 uint32 words per 64 weights). |
| **TurboFieldfare Architecture** | Original bounded streaming design | `https://github.com/drumih/turbo-fieldfare` | **VERIFIED** | Bounded expert cache, Metal kernel dispatch, SSD streaming principles. |
| **QwenFieldfare Base** | Swift/Metal infrastructure | Git commit `0775cda` / `7370a1d` | **VERIFIED** | Fixed router-order reduction eliminating cache-residency floating-point variance, SwiftNIO server, Metal pipeline management. |

## Source Discrepancies and Clarifications

1. **Quantization Format (INT4 vs 3-bit)**:
   - Official Recursive INT4 uses 4-bit routed experts.
   - The primary local checkpoint `/Users/tanishq/models/flint-mixed-3-4bit` uses 3-bit routed experts (192 uint32s per 2048 weights) and INT4 for attention/shared/dense/lm_head.
   - **Resolution**: The FlintFieldfare engine will support both 3-bit and 4-bit routed expert dequantization dynamically based on the tensor metadata (`bits: 3` vs `bits: 4`).

2. **Attention Sinks**:
   - The checkpoint includes `model.layers.<idx>.self_attn.softmax_offset` of shape `[32]` (one BF16 scalar per query head).
   - In Transformers and MLX, this is an additional logit appended to the attention logits prior to softmax. It has no corresponding value vector.
   - **Resolution**: In Metal attention, the running softmax max and normalizer incorporate $\exp(\text{sink} - m)$, absorbing probability mass when no context token is strongly relevant.

3. **RoPE Layer Assignment**:
   - Config specifies `no_rope_layer_indices: [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]`.
   - These 14 layers correspond exactly to the `full_attention` layers.
   - The remaining 35 layers are `sliding_attention` with window size 513 and standard LLaMA-style RoPE (`traditional=False`, $\theta = 10000.0$).
   - **Resolution**: Strict adherence to the `no_rope_layer_indices` contract in `FlintModelDirectory` and `FlintAttentionKernel`.
