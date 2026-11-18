# Physical Tensor Inventory — Recursive Flint 40B-A2.5B

## 1. Executive Summary & Physical Baseline

This document provides the authoritative physical tensor inventory for the **Recursive Flint 40B-A2.5B** checkpoint located at `/Users/tanishq/models/flint-mixed-3-4bit`, validated under milestone **F1-03**.

Physical validation was performed using `FlintSafetensorsFile` and `FlintModelDirectory.crossValidate`, confirming zero descriptor corruptions, zero duplicate tensor names, zero overlapping tensor payload allocations, and exact 1:1 correspondence between `model.safetensors.index.json` and the physical shard headers.

### High-Level Metrics
- **Total Physical Shards**: 4 Safetensors files
- **Total Registered Tensors**: 1,751 tensors
- **Total Physical Byte Size**: 17,837,740,282 bytes (~17.84 GB / 16.61 GiB)
- **Data Types**: `BF16` (1,265 tensors), `U32` (486 quantized projection tensors)
- **Quantization Scheme**: Base 4-bit (group size 64) with INT8 embeddings (`[150144, 512]` U32) and 3-bit routed MoE projection overrides across all 141 switch_mlp projection groups.

---

## 2. Shard Distribution & Physical Sizes

| Shard Filename | Total File Size (Bytes) | Header Size (Bytes) | Payload Size (Bytes) | Tensor Count |
| :--- | :--- | :--- | :--- | :--- |
| `model-00001-of-00004.safetensors` | 5,297,553,268 | 134,800 | 5,297,418,460 | 542 |
| `model-00002-of-00004.safetensors` | 5,363,650,068 | 126,344 | 5,363,523,716 | 508 |
| `model-00003-of-00004.safetensors` | 5,294,980,456 | 133,264 | 5,294,847,184 | 536 |
| `model-00004-of-00004.safetensors` | 1,881,556,490 | 41,088 | 1,881,515,394 | 165 |
| **Global Total** | **17,837,740,282** | **435,496** | **17,837,304,754** | **1,751** |

*Note: Header byte counts exclude the 8-byte little-endian header length prefix.*

---

## 3. Data Type Distribution

| Data Type | Tensor Count | Percentage | Purpose |
| :--- | :--- | :--- | :--- |
| `BF16` (bfloat16) | 1,265 | 72.24% | Norms, biases, scales, attention projections, router gates, softmax offsets |
| `U32` (uint32) | 486 | 27.76% | Packed quantized projection weights (embeddings, dense MLPs, MoE experts) |
| `F32` / `F64` / `I32` | 0 | 0.00% | None present in physical checkpoint |

Every tensor's byte size matches the exact product:
$$\text{byteCount} = \left(\prod_{d \in \text{shape}} d\right) \times \text{byteWidth}(\text{dtype})$$

---

## 4. Architectural Layer Structure

Recursive Flint 40B-A2.5B contains 49 transformer layers (`num_hidden_layers = 49`, indices 0 to 48). **All 49 layers possess self-attention mechanisms.**

1. **Self-Attention Mechanism (Present on all 49 layers, indices 0..48)**:
   - 32 Query heads, 4 Key/Value heads (Grouped-Query Attention, 8 Q heads per KV head).
   - Head dimension: 128 (Total Query dim: 4,096; Total KV dim: 512).
   - `q_proj`, `k_proj`, `v_proj`, `o_proj` quantized weights (U32 4-bit) with accompanying BF16 scales and biases.
   - Per-head RMSNorm on query and key vectors: `q_norm.weight` and `k_norm.weight` (BF16 `[128]`, 256 bytes per head group).
   - Learned attention sinks: `softmax_offset` (BF16 `[32]`, 64 bytes) present on **all 49 layers**.
   - **Full-Attention Layers (14 layers)**:
     - Indices: `[0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]`
     - Positional Embedding: **NONE** (`no_rope_layer_indices`)
     - Causal mask over full context length.
   - **Sliding-Window Attention Layers (35 layers)**:
     - Indices: All 35 layers other than the 14 full-attention layers.
     - Positional Embedding: RoPE (`rope_theta = 10000.0`, `traditional = false`, split-half rotation).
     - Window size: 513 tokens.

2. **Feed-Forward Networks (Two Variants)**:
   - **Dense MLP Layers (Layers 0 and 1, `mlp_only_layers = [0, 1]`)**:
     - Standard dense feed-forward network (SwiGLU, `intermediate_size = 6144`):
       - `model.layers.<l>.mlp.gate_proj.weight`: U32 `[6144, 256]` (6,291,456 bytes)
       - `model.layers.<l>.mlp.up_proj.weight`: U32 `[6144, 256]` (6,291,456 bytes)
       - `model.layers.<l>.mlp.down_proj.weight`: U32 `[2048, 768]` (6,291,456 bytes)
       - Accompanying BF16 scales and biases (`[6144, 32]` and `[2048, 96]`).
   - **Sparse MoE Layers (Layers 2 through 48, 47 layers)**:
     - Shared Expert (Dense MLP, intermediate size 1,024):
       - `shared_expert.gate_proj.weight`: U32 `[1024, 256]` (1,048,576 bytes)
       - `shared_expert.up_proj.weight`: U32 `[1024, 256]` (1,048,576 bytes)
       - `shared_expert.down_proj.weight`: U32 `[2048, 128]` (1,048,576 bytes)
       - Accompanying BF16 scales and biases (`[1024, 32]` and `[2048, 16]`).
     - Router Gate:
       - `model.layers.<l>.mlp.gate.weight`: BF16 `[256, 2048]` (1,048,576 bytes).
     - Routed MoE Experts:
       - 256 experts per layer, top-6 active routing with sigmoid scoring and normalization.
       - Contiguous 3D tensors per projection across all 256 experts.

---

## 5. Routed MoE Expert Record Sizes & Offsets

Each MoE layer (layers 2..48) stores its 256 routed experts as three packed 3D tensors:
- `gate_proj`
- `up_proj`
- `down_proj`

### 5.1 Physical Shapes and Per-Expert Record Sizes (3-bit MoE)

In `flint-mixed-3-4bit`, all 141 MoE projections (`47 layers * 3 projections`) are quantized at 3-bit with `group_size = 64`.

| Projection | Physical Shape | Dtype | Total Tensor Bytes | Per-Expert Shape | Per-Expert Bytes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gate_proj.weight` | `[256, 512, 192]` | `U32` | 100,663,296 | `[512, 192]` | 393,216 B |
| `gate_proj.scales` | `[256, 512, 32]` | `BF16` | 8,388,608 | `[512, 32]` | 32,768 B |
| `gate_proj.biases` | `[256, 512, 32]` | `BF16` | 8,388,608 | `[512, 32]` | 32,768 B |
| `up_proj.weight` | `[256, 512, 192]` | `U32` | 100,663,296 | `[512, 192]` | 393,216 B |
| `up_proj.scales` | `[256, 512, 32]` | `BF16` | 8,388,608 | `[512, 32]` | 32,768 B |
| `up_proj.biases` | `[256, 512, 32]` | `BF16` | 8,388,608 | `[512, 32]` | 32,768 B |
| `down_proj.weight` | `[256, 2048, 48]` | `U32` | 100,663,296 | `[2048, 48]` | 393,216 B |
| `down_proj.scales` | `[256, 2048, 8]` | `BF16` | 8,388,608 | `[2048, 8]` | 32,768 B |
| `down_proj.biases` | `[256, 2048, 8]` | `BF16` | 8,388,608 | `[2048, 8]` | 32,768 B |

### 5.2 Aggregated Per-Expert Allocation
- **Packed Weights**: $3 \times 393,216 = 1,179,648\text{ B}$ (1.125 MiB)
- **Scales**: $3 \times 32,768 = 98,304\text{ B}$ (0.09375 MiB)
- **Biases**: $3 \times 32,768 = 98,304\text{ B}$ (0.09375 MiB)
- **Total Per-Expert Record Size**:
  $$\mathbf{1,376,256\text{ Bytes}} = \mathbf{1.3125\text{ MiB}}$$

### 5.3 Comparison with Official INT4 Checkpoint
- **Official INT4 Per-Expert Size**:
  - Packed Weights: $3 \times 524,288 = 1,572,864\text{ B}$
  - Scales: $3 \times 32,768 = 98,304\text{ B}$
  - Biases: $3 \times 32,768 = 98,304\text{ B}$
  - Total INT4: $1,769,472\text{ B} = 1.6875\text{ MiB}$
- **Compression Ratio**:
  $$\frac{1.3125\text{ MiB}}{1.6875\text{ MiB}} = 0.7778 \quad (22.2\%\text{ reduction in per-expert weight residency})$$

### 5.4 Closed-Form Offset Formula for Random-Access Expert Reads

To read the weights, scales, or biases for an arbitrary expert $e \in [0, 255]$ in layer $l \in [2, 48]$ without loading any other expert:

$$\begin{aligned}
\text{offset}(\text{gate\_weight}, e) &= \text{base\_offset}(\text{gate\_weight}) + e \times 393,216 \\
\text{offset}(\text{gate\_scales}, e) &= \text{base\_offset}(\text{gate\_scales}) + e \times 32,768 \\
\text{offset}(\text{gate\_biases}, e) &= \text{base\_offset}(\text{gate\_biases}) + e \times 32,768 \\
\text{offset}(\text{up\_weight}, e) &= \text{base\_offset}(\text{up\_weight}) + e \times 393,216 \\
\text{offset}(\text{up\_scales}, e) &= \text{base\_offset}(\text{up\_scales}) + e \times 32,768 \\
\text{offset}(\text{up\_biases}, e) &= \text{base\_offset}(\text{up\_biases}) + e \times 32,768 \\
\text{offset}(\text{down\_weight}, e) &= \text{base\_offset}(\text{down\_weight}) + e \times 393,216 \\
\text{offset}(\text{down\_scales}, e) &= \text{base\_offset}(\text{down\_scales}) + e \times 32,768 \\
\text{offset}(\text{down\_biases}, e) &= \text{base\_offset}(\text{down\_biases}) + e \times 32,768
\end{aligned}$$

Each read is executed via `FlintSafetensorsFile.readPayloadSlice(range:)` using POSIX `pread`, ensuring thread-safe, non-racing, zero-copy bounded reads directly from disk.

---

## 6. Empirical Validation & SHA-256 Checksums

The physical integrity of random-access reads was cross-checked against independent byte references extracted via Python / `mlx.core.load`:

| Tensor Name | Byte Range | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| `model.layers.2.self_attn.softmax_offset` | `0..<64` | `e55d79b0bd25bc2d585bc4bf6706ae3d6d62e2561c9b0ec8f9918041f6404404` | MATCH |
| `model.layers.0.input_layernorm.weight` | `0..<4096` | `f33f5b184b24b4e62c37c757ee2d90396173e9ef2ef28af130b266ca021cc4a5` | MATCH |
| `model.embed_tokens.weight` | `0..<256` | `c5b8b890109befaf2efd23d462ece554855d2c73d1c14301a8c62afff75bed71` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.weight` (Expert 0) | `0..<393216` | `a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820` | MATCH |
| `model.layers.20.mlp.switch_mlp.gate_proj.weight` (Expert 128) | `50331648..<50724864` | `a3fcc07e846e6c78e5a57c7e89870e97bbb56afe2a0ddb7979043c127acbd678` | MATCH |
| `model.layers.48.mlp.switch_mlp.down_proj.weight` (Expert 255) | `100270080..<100663296` | `2e823253c72b37404f447e5656a08995746460f39b417474eeed19078d9fd15b` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.scales` (Expert 0) | `0..<32768` | `7cc18d29413197ee959987c119bbb6e0ed834af8af8107cb3e9e937d9ee267d7` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.biases` (Expert 0) | `0..<32768` | `4629cee3a9d13041dafcf227f7ac56e5c2388a2619ed197595e176a1fd1bab2d` | MATCH |

---

## 7. Integrity Conclusions

1. **Physical Zero-Overlap Invariant**: Every tensor range within all 4 shards is strictly disjoint from all other tensor ranges:
   $$\text{offset}_{i} + \text{byteCount}_{i} \le \text{offset}_{i+1}$$
2. **Deterministic Layout**: Every shard header's metadata and byte bounds completely account for the physical payload without gaps or unmapped trailing space.
3. **No Dynamic Tensor Discovery**: The runtime does not rely on filenames or globbing at inference time; all tensor locations are resolved via `model.safetensors.index.json` validated against `FlintSafetensorsFile` headers.
