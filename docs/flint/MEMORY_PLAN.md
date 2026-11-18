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

### System Memory Dynamics: Correlation vs. Causality
During the benchmark run, `vm.swapusage` increased from an initial `767.56 MB` to `1,425.19 MB` (+657.6 MB) before stabilizing between `1,361 MB` and `1,417 MB`.
- **System Memory Event**: On a 32 GB unified memory machine running concurrent development tools (IDE, terminals, Antigravity, browser), allocating ~17.68 GB for MLX elevated system memory pressure, prompting macOS to page out inactive background pages.
- **Rigor Clarification**: This swap increase is an **observed system-level memory response under high unified memory allocation**, not proof of active memory thrashing during decode. Steady decode throughput (~70 tok/s short, ~60 tok/s long context) demonstrates that active model parameters and KV cache remained in physical RAM during generation.

---

## 3. Dissected Memory Breakdown (Estimates vs Measurements)

| Component | Status | Breakdown / Formula | Footprint |
|---|---|---|---|
| **1. Explicit Resident Weights** | Theoretical Estimate | Embeddings (326.7 MB) + LM Head (173.0 MB) + 49 Attention Projections & Norms (520.9 MB) + 2 Dense MLPs (42.5 MB) + 47 Routers (49.4 MB) + 47 Shared Experts (166.4 MB) | **1.28 GiB** |
| **2. Expert Cache Allocation** | Configurable Budget | $N_{\text{slots}} \times 1,376,256\text{ bytes}$ (3-bit) or $1,769,472\text{ bytes}$ (4-bit). For 1,024 slots: | **1.31 GiB** (3-bit)<br>**1.69 GiB** (4-bit) |
| **3. Attention KV Cache** | Theoretical Estimate | 35 sliding layers (max $513 \times 2,048\text{ B} = 36.8\text{ MB}$) + 14 full layers ($14 \times 2,048\text{ B/token}$). At 8,192 context: | **0.26 GiB** |
| **4. Metal Activation & Scratch** | Engineering Estimate | Batch 1 decode activations, router scores, and $6 \times 2,048$ intermediate reduction buffers | **~0.15 GiB** |
| **5. Process Runtime & Heap** | Engineering Estimate | Swift runtime, Tokenizer Trie, SwiftNIO event loops | **~0.20 GiB** |
| **6. Total Planned Working Set** | **Target Design Budget** | Sum of Components 1 through 5 (with 1,024 cache slots and 8K context) | **~3.20 GiB** |
| **7. Measured All-Resident Peak** | **Empirically Measured** | Real MLX peak allocation on M5 with 8K context | **17.68 GB** |

> **Important Terminology**: The **~3.2 GiB** figure is strictly an **architectural design target budget** for FlintFieldfare's bounded streaming engine. It is NOT a measured runtime footprint of FlintFieldfare, as the streaming runner is under active development in milestones F1–F4.

---

## 4. Engineering Trade-off: All-Resident vs. Bounded Streaming

The choice between all-resident execution and bounded expert streaming represents a direct engineering trade-off:

1. **All-Resident Execution (Maximize Inference Throughput)**:
   - When 18+ GB of RAM is unreserved and dedicated to inference, all-resident execution maximizes performance: ~70 tok/s decode and ~1,317 tok/s prefill on M5.
   - Best suited for dedicated inference servers or single-purpose benchmark tasks.

2. **Bounded Expert Streaming (Coexistence in Multitasking Workstations)**:
   - When a developer operates on a 32 GB (or 24 GB) machine running Xcode, language servers, browsers, Docker, or multiple agent conversations, allocating 18 GB to model weights leaves insufficient physical RAM for other tasks, inducing background paging.
   - Bounded streaming aims to constrain the inference working set to the **~3.2 GiB design budget**, leaving >20 GB of unified memory available for the host OS and developer workflows, trading peak decode speed for seamless local coexistence.

---

## 5. Theoretical SSD I/O Bandwidth vs Expert-Cache Hit Rate

Flint evaluates $K = 6$ routed experts per MoE layer across 47 MoE layers = **282 expert evaluations per token**.  
Each 3-bit expert record is $1,376,256\text{ bytes} \approx 1.3125\text{ MiB}$ (1.376 MB decimal) — verified physically in `docs/flint/PHYSICAL_TENSOR_INVENTORY.md`.  
Cold payload per token (0% cache hit):
$$\text{Payload}_{\text{cold}} = 282 \times 1.3125\text{ MiB} = \mathbf{370.125\text{ MiB / token}}\quad (388.11\text{ MB / token})$$

The theoretical required SSD read bandwidth as a function of cache hit rate $H \in [0.0, 1.0]$ and decode speed $R$ (tokens/second) is:
$$\text{Bandwidth}(H, R) = R \times 370.125\text{ MiB} \times (1 - H)$$

| Cache Hit Rate ($H$) | Misses / Token | Read Traffic / Token (MiB) | Read Traffic / Token (MB) | SSD Bandwidth @ 15 tok/s (GB/s) | SSD Bandwidth @ 20 tok/s (GB/s) | Feasibility on Apple M5 SSD (~3.5–5.0 GB/s read) |
|---|---|---|---|---|---|---|
| **0.0% (Cold Start)** | 282 | 370.125 MiB | 388.11 MB | 5.82 GB/s | 7.76 GB/s | **Exceeds SSD sequential throughput** |
| **50.0%** | 141 | 185.06 MiB | 194.05 MB | 2.91 GB/s | 3.88 GB/s | Approaching SSD saturation |
| **70.0% (Hypothesis)** | 84.6 | 111.04 MiB | 116.43 MB | 1.75 GB/s | 2.33 GB/s | **Comfortably viable** |
| **80.0% (Hypothesis)** | 56.4 | 74.025 MiB | 77.62 MB | 1.16 GB/s | 1.55 GB/s | **Highly efficient** |
| **90.0% (Hypothesis)** | 28.2 | 37.01 MiB | 38.81 MB | 0.58 GB/s | 0.78 GB/s | Minimal I/O overhead |
| **95.0% (Hypothesis)** | 14.1 | 18.51 MiB | 19.41 MB | 0.29 GB/s | 0.39 GB/s | Near memory-resident performance |

### I/O Feasibility Conclusion

1. **Cold-start streaming** at 0% cache hit rate cannot match all-resident decode speed because streaming 282 experts per token would require >5.8 GB/s SSD bandwidth.
2. Bounded expert streaming provides its primary value by **capping unified memory usage at the ~3.2 GiB design budget**, preventing memory pressure and preserving system responsiveness.
3. Interactive streaming speeds (15–20 tok/s) are theoretically feasible under the **working hypothesis** that expert cache hit rates reach $\ge 70\text{--}85\%$ on repeated coding prompts. **This hit-rate assumption remains an explicit hypothesis** to be empirically tested and measured in milestone F4.
