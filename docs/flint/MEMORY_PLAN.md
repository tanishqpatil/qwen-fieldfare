# FlintFieldfare Memory Plan — Apple Silicon M5 (32 GB)

## Target Platform Context

- **Device**: Apple MacBook Pro M5
- **Unified Memory**: 32 GiB (34,359,738,368 bytes)
- **Metal Architecture**: Apple Silicon unified memory (shared CPU/GPU address space)
- **Operating System**: macOS 27.2.1

## Weight Residency Analysis

### 1. Resident Weights (Always in Unified Memory)

| Component | Tensor Breakdown | Dimensions / Format | Size (MB) |
|---|---|---|---|
| **Embeddings** | `model.embed_tokens.weight`<br>`model.embed_tokens.scales`<br>`model.embed_tokens.biases` | `[150144, 512]` U32 (INT8)<br>`[150144, 32]` BF16<br>`[150144, 32]` BF16 | 307.49<br>9.61<br>9.61<br>**Subtotal: 326.71** |
| **Output Head** | `lm_head.weight`<br>`lm_head.scales`<br>`lm_head.biases` | `[150144, 256]` U32 (INT4)<br>`[150144, 32]` BF16<br>`[150144, 32]` BF16 | 153.75<br>9.61<br>9.61<br>**Subtotal: 172.97** |
| **Attention (49 layers)** | `q_proj`, `k_proj`, `v_proj`, `o_proj`<br>`q_norm`, `k_norm`, `softmax_offset`<br>`input_layernorm`, `post_attention_layernorm` | INT4 (q: 4.72MB, k: 0.59MB, v: 0.59MB, o: 4.72MB)<br>Norms & Sinks: 12.8 KB/layer<br>$\times$ 49 layers | 49 layers $\times$ 10.63 MB<br>**Subtotal: 520.87** |
| **Dense MLPs (Layers 0 & 1)** | `gate_proj`, `up_proj`, `down_proj` | INT4 group 64, interm 6144<br>$\times$ 2 layers | 2 layers $\times$ 21.23 MB<br>**Subtotal: 42.46** |
| **Routers (Layers 2..48)** | `mlp.gate.weight` | `[256, 2048]` BF16 = 1.05 MB/layer<br>$\times$ 47 layers | 47 layers $\times$ 1.05 MB<br>**Subtotal: 49.35** |
| **Shared Experts (Layers 2..48)** | `shared_expert.gate/up/down_proj` | INT4 group 64, interm 1024<br>$\times$ 47 layers | 47 layers $\times$ 3.54 MB<br>**Subtotal: 166.38** |
| **Final LayerNorm** | `model.norm.weight` | `[2048]` BF16 | **Subtotal: 0.004** |
| **TOTAL RESIDENT WEIGHTS** | | | **1,278.74 MB (~1.25 GiB)** |

### 2. Streamed Weights (Routed Experts)

- **Total MoE Layers**: 47
- **Experts per Layer**: 256
- **Total Expert Instances**: $47 \times 256 = 12,032$
- **Size per Single Expert (3-bit mixed)**:
  - `gate_proj`: 384 KiB weight + 32 KiB scale + 32 KiB bias = 448 KiB
  - `up_proj`: 384 KiB weight + 32 KiB scale + 32 KiB bias = 448 KiB
  - `down_proj`: 384 KiB weight + 32 KiB scale + 32 KiB bias = 448 KiB
  - **Total per Expert**: **1,344 KiB = 1,376,256 Bytes (1.3125 MiB)**
- **Size per Single Expert (4-bit reference)**:
  - $3 \times (512 + 32 + 32)\text{ KiB} = 1,728\text{ KiB} = 1,769,472\text{ Bytes (1.6875 MiB)}$
- **Active Weights per Token**:
  - $47 \text{ layers} \times 6 \text{ experts} \times 1.3125 \text{ MiB} = \mathbf{370.1\text{ MiB}}$

### 3. Bounded Expert Cache Budgets

| Cache Setting | Slot Count | Memory Footprint (3-bit) | Memory Footprint (4-bit) | Slots / Layer (avg) |
|---|---|---|---|---|
| Conservative (Cold/Test) | 384 slots | 504 MiB (0.49 GiB) | 648 MiB (0.63 GiB) | ~8.1 slots/layer |
| Standard (Default) | 1,024 slots | 1,344 MiB (1.31 GiB) | 1,728 MiB (1.69 GiB) | ~21.8 slots/layer |
| Performance | 2,048 slots | 2,688 MiB (2.63 GiB) | 3,456 MiB (3.38 GiB) | ~43.6 slots/layer |
| Generous | 4,096 slots | 5,376 MiB (5.25 GiB) | 6,912 MiB (6.75 GiB) | ~87.1 slots/layer |

### 4. Attention KV Cache Memory

Each token across all 49 layers:
- 4 KV heads $\times 128$ head dim $\times 2$ bytes (FP16) $\times 2$ (K and V) $= 2,048$ bytes / token / layer.
- **Sliding-Window Layers (35 layers)**:
  - Window size = 513 tokens.
  - Max KV memory (bounded!): $35 \times 513 \times 2,048\text{ bytes} = \mathbf{36.77\text{ MiB}}$.
- **Full-Attention Layers (14 layers)**:
  - $14 \times 2,048 = 28,672\text{ bytes / token} \approx 28\text{ KiB / token}$.

| Context Length (Tokens) | Sliding Window KV | Full Attention KV | Total KV Cache |
|---|---|---|---|
| 2,048 | 36.77 MiB | 56.00 MiB | 92.77 MiB |
| 4,096 | 36.77 MiB | 112.00 MiB | 148.77 MiB |
| 8,192 | 36.77 MiB | 224.00 MiB | 260.77 MiB |
| 16,384 | 36.77 MiB | 448.00 MiB | 484.77 MiB |
| 32,768 | 36.77 MiB | 896.00 MiB | 932.77 MiB |

### 5. Overall System Working Set & Safety Budget

For the standard configuration (1,024 cache slots, 8,192 context window):

```
+-------------------------------------------------------+
| Component                               | Memory (GB) |
+-------------------------------------------------------+
| Resident Model Weights                  |    1.28 GB  |
| Expert Cache Buffer (1,024 slots)       |    1.31 GB  |
| Attention KV Cache (8,192 tokens)       |    0.26 GB  |
| Metal Activations & Scratch Allocations |    0.15 GB  |
| Process Runtime, Tokenizer, Swift Heap  |    0.20 GB  |
+-------------------------------------------------------+
| TOTAL PROCESS WORKING MEMORY            |    3.20 GB  |
| Safety Reserve (Free Unified Memory)    |  > 20.00 GB  |
+-------------------------------------------------------+
```

## Feasibility Verdict

**Recursive Flint 40B-A2.5B on Apple Silicon M5 (32 GB) is highly feasible with bounded expert streaming.**
The full checkpoint requires 17.8 GB (or 22.6 GB), which leaves little room for OS, display, and development tools.
Under FlintFieldfare's bounded streaming architecture, the active footprint drops to **~3.2 GiB**, maintaining an overwhelming safety margin of >20 GB on a 32 GB machine and eliminating any risk of memory pressure or swap churn.
