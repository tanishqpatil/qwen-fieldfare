# FlintFieldfare Memory Plan & Streaming Feasibility Analysis

## 1. Physical Target Context

- **Device**: Apple MacBook Pro M5
- **Unified Memory**: 32 GiB (34,359,738,368 bytes)
- **OS**: macOS 27.2.1 (Darwin 27.2.0, Build 26B131)
- **Storage**: Apple NVMe SSD (Internal, ~114 GiB available)
- **Baseline Checkpoint**: `/Users/tanishq/models/flint-mixed-3-4bit` (17.84 GB on disk, 3-bit routed experts, INT4 projections/shared/dense, INT8 embeddings)

---

## 2. Empirical All-Resident Baseline (Measured on M5 32 GB)

Prior to implementing the streaming engine, an empirical all-resident reference run was executed on this exact machine using `mlx` 0.32.4 and `mlx_vlm` 0.7.12:

```
+-------------------------------------------------------------+
| Metric                                        | Measured    |
+-------------------------------------------------------------+
| Model Cold Load Time                          | 4.39 s      |
| MLX Active Weight Memory                      | 16.61 GB    |
| Peak MLX Memory (Short Prompt, 13 tokens)     | 16.68 GB    |
| Peak MLX Memory (Long Prompt, 7,831 tokens)   | 17.68 GB    |
| Process Max RSS (Resident Set Size)           | 6.54 GB     |
| System Swap Usage Before Load                 | 767.56 MB   |
| System Swap Usage During 8K Inference         | 1,425.19 MB |
| Swap Increase Triggered by All-Resident Model | +657.63 MB  |
+-------------------------------------------------------------+
```

### Empirical Inference Throughput (All-Resident MLX on M5)

- **Short Coding Prompt (13 tokens, greedy temp=0.0, 64 gen tokens)**:
  - Run 1 (Cold): TTFT 3.01s (13.76 tok/s), Generation: **70.39 tok/s**
  - Run 2 (Warm): Prefill: **161.91 tok/s**, Generation: **69.91 tok/s**
  - Run 3 (Warm): Prefill: **161.51 tok/s**, Generation: **70.02 tok/s**
- **Long Context (~8K tokens: 7,831 tokens, greedy temp=0.0, 32 gen tokens)**:
  - Run 1: Prefill: **1,228.81 tok/s** (6.91s total), Generation: **60.31 tok/s**, Peak: 17.68 GB
  - Run 2: Prefill: **1,319.89 tok/s** (6.48s total), Generation: **60.63 tok/s**, Peak: 17.68 GB
  - Run 3: Prefill: **1,313.99 tok/s** (6.51s total), Generation: **59.84 tok/s**, Peak: 17.68 GB

**Key Finding**: While all-resident execution achieves ~70 tok/s decode on the M5, holding 17.7 GB resident in unified memory instantly pushed macOS into swapping (+657.6 MB swap increase), reducing headroom for development environments, agent tooling, and multitasking.

---

## 3. Dissected Memory Breakdown (Estimates vs Measurements)

| Component | Status | Breakdown / Formula | Footprint |
|---|---|---|---|
| **1. Explicit Resident Weights** | Theoretical Estimate | Embeddings (326.7 MB) + LM Head (173.0 MB) + 49 Attention Projections & Norms (520.9 MB) + 2 Dense MLPs (42.5 MB) + 47 Routers (49.4 MB) + 47 Shared Experts (166.4 MB) | **1.28 GiB** |
| **2. Expert Cache Allocation** | Configurable Budget | $N_{\text{slots}} \times 1,376,256\text{ bytes}$ (3-bit) or $1,769,472\text{ bytes}$ (4-bit). For 1,024 slots: | **1.31 GiB** (3-bit)<br>**1.69 GiB** (4-bit) |
| **3. Attention KV Cache** | Theoretical Estimate | 35 sliding layers (max $513 \times 2,048\text{ B} = 36.8\text{ MB}$) + 14 full layers ($14 \times 2,048\text{ B/token}$). At 8,192 context: | **0.26 GiB** |
| **4. Metal Activation & Scratch** | Engineering Estimate | Batch 1 decode activations, router scores, and $6 \times 2,048$ intermediate reduction buffers | **~0.15 GiB** |
| **5. Process Runtime & Heap** | Engineering Estimate | Swift runtime, Tokenizer Trie, SwiftNIO event loops | **~0.20 GiB** |
| **6. Total Planned Working Set** | **Target Budget (F3/F4)** | Sum of Components 1 through 5 (with 1,024 cache slots and 8K context) | **~3.20 GiB** |
| **7. Measured All-Resident Peak** | **Empirically Measured** | Real MLX peak allocation on M5 with 8K context | **17.68 GB** |

> **Note on Terminology**: The ~3.2 GiB figure is the **design target budget** for FlintFieldfare's bounded streaming engine. It is NOT yet a measured runtime footprint of FlintFieldfare, as the production streaming runner is under active development in milestones F1–F4.

---

## 4. Theoretical SSD I/O Bandwidth vs Expert-Cache Hit Rate

Flint evaluates $K = 6$ routed experts per MoE layer across 47 MoE layers = **282 expert evaluations per token**.
Each 3-bit expert record is $1,376,256\text{ bytes} \approx 1.3125\text{ MiB}$.
Cold payload per token (0% cache hit):
$$\text{Payload}_{\text{cold}} = 282 \times 1.3125\text{ MiB} = \mathbf{370.1\text{ MiB / token}}$$

The theoretical required SSD read bandwidth as a function of cache hit rate $H \in [0.0, 1.0]$ and decode speed $R$ (tokens/second) is:
$$\text{Bandwidth}(H, R) = R \times 370.1\text{ MiB} \times (1 - H)$$

| Cache Hit Rate ($H$) | Misses / Token | Read Traffic / Token | SSD Bandwidth @ 15 tok/s | SSD Bandwidth @ 25 tok/s | Feasibility on Apple M5 SSD (~3.5–5.0 GB/s read) |
|---|---|---|---|---|---|
| **0.0% (Cold)** | 282 | 370.1 MiB | 5.55 GB/s | 9.25 GB/s | **Exceeds SSD sequential throughput** |
| **30.0%** | 197 | 258.6 MiB | 3.88 GB/s | 6.46 GB/s | Near SSD saturation |
| **50.0%** | 141 | 185.1 MiB | 2.78 GB/s | 4.63 GB/s | Viable at moderate decode speeds |
| **70.0%** | 85 | 111.0 MiB | 1.67 GB/s | 2.78 GB/s | **Comfortably viable** |
| **85.0%** | 42 | 55.5 MiB | 0.83 GB/s | 1.39 GB/s | High efficiency |
| **95.0%** | 14 | 18.5 MiB | 0.28 GB/s | 0.46 GB/s | Minimal I/O overhead |

### I/O Feasibility Conclusion

1. **Cold-start streaming** at 0% cache hit rate cannot match all-resident decode speed because streaming 282 experts per token would require >5 GB/s SSD bandwidth.
2. Bounded expert streaming provides its primary value by **capping unified memory usage at ~3.2 GiB** instead of 18–24 GB, completely preventing system memory thrashing and swap stalls.
3. Interactive streaming speeds (15–30 tok/s) are achievable once the expert cache warms up to $\ge 70\text{--}85\%$ hit rate.
