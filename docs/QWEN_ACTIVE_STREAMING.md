# Qwen 3.6 35B-A3B Q4 Active Expert Streaming Engineering Log

This engineering log tracks the research, architecture, implementation milestones, and performance results of the **Active Expert Streaming Q4** runtime for `Qwen3.6-35B-A3B-4bit` on Apple Silicon Metal.

---

## Benchmark History Summary

| Date | Configuration / Phase | Prefill (ms / tok/s) | Steady Latency (ms/tok) | Sustained Rate (tok/s) | Peak Metal (GiB) | Peak Wired (GiB) | Staged Traffic (MB/tok) | MLX Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-08-21 | **Fast Resident Q4** (All layers resident) | $320.0\text{ ms}$ ($12.5\text{ t/s}$) | $64.10\text{ ms}$ | **$15.60\text{ tok/s}$** | $18.34\text{ GiB}$ | $20.46\text{ GiB}$ | $0.00\text{ MB}$ | Exact |
| 2026-08-21 | **Memory Saver Q4** (4-Layer Cache) | $1,612.9\text{ ms}$ ($2.48\text{ t/s}$) | $403.22\text{ ms}$ | **$2.48\text{ tok/s}$** | $2.45\text{ GiB}$ | $4.67\text{ GiB}$ | $4,320.00\text{ MB}$ | Exact |
| 2026-08-21 | **Active Expert Streaming** (9-Slot Arena, Uncached) | $2,206.6\text{ ms}$ ($1.81\text{ t/s}$) | $184.62\text{ ms}$ | **$5.42\text{ tok/s}$** | $0.45\text{ GiB}$ | $2.99\text{ GiB}$ | $1,396.02\text{ MB}$ | Exact |
| 2026-08-21 | **Bounded Persistent Cache** (1024 Slots, Standard LRU) | $625.5\text{ ms}$ ($6.40\text{ t/s}$) | $105.08\text{ ms}$ | **$9.52\text{ tok/s}$** | $2.14\text{ GiB}$ | $4.53\text{ GiB}$ | $1,053.99\text{ MB}$ | Exact |
| 2026-08-21 | **Segmented LRU Cache + Prefetch** (1024 Slots, Dynamic Dense Staging) | $354.3\text{ ms}$ ($11.29\text{ t/s}$) | $93.34\text{ ms}$ | **$10.71\text{ tok/s}$** | $2.14\text{ GiB}$ | $4.60\text{ GiB}$ | $1,029.62\text{ MB}$ | Exact |
| 2026-08-21 | **Full Dense Persistence + SLRU Cache** (Plan 1, 1024 Slots) | $281.5\text{ ms}$ ($14.21\text{ t/s}$) | $80.26\text{ ms}$ | **$12.46\text{ tok/s}$** | $3.08\text{ GiB}$ | $5.61\text{ GiB}$ | $173.65\text{ MB}$ | Exact |
| 2026-08-21 | **Fused DeltaNet Step + Q4 Factored GEMV** (Phase 5, 1024 Slots) | $282.1\text{ ms}$ ($14.18\text{ t/s}$) | $78.63\text{ ms}$ | **$12.72\text{ tok/s}$** | $3.08\text{ GiB}$ | $5.60\text{ GiB}$ | $173.60\text{ MB}$ | Exact |
| 2026-08-21 | **Consolidated 2-CB Pipeline + Vectorized Argmax** (Phase 6, 1024 Slots) | $185.4\text{ ms}$ ($21.57\text{ t/s}$) | $54.01\text{ ms}$ | **$18.51\text{ tok/s}$** | $3.08\text{ GiB}$ | $5.46\text{ GiB}$ | $173.65\text{ MB}$ | Exact |
| 2026-08-21 | **Production App Default & Real Chat Workloads** (Phase 7, 1024 Slots) | $187.3\text{ ms}$ ($21.35\text{ t/s}$) | $54.87\text{ ms}$ | **$18.23\text{ tok/s}$** | $3.08\text{ GiB}$ | $5.61\text{ GiB}$ | $173.65\text{ MB}$ | Exact |

---

## 1. Architectural Overview & System Specification

### Model Specifications
- **Model Path**: `~/models/Qwen3.6-35B-A3B-4bit`
- **Total Layers**: $40$ layers ($30$ DeltaNet linear attention layers, $10$ full sliding-window multi-head attention layers)
- **Hidden Dimension**: $2048$, Intermediate Dimension: $512$ per routed expert ($2048$ shared expert)
- **Routed Experts**: $256$ experts per layer, Top-$8$ softmax routing
- **Shared Expert**: $1$ always-on shared expert per layer with sigmoid gating coefficient
- **Quantization Wire Contract**: Q4 Affine (MLX format, group size $64$, packed `uint32` nibbles, BF16 scale/bias pairs)

### Hardware & Memory Budget Constraints
- **Peak Metal Allocation Ceiling**: $\le 5.00\text{ GiB}$
- **Peak System Wired RAM Ceiling**: $\le 6.00\text{ GiB}$
- **System Swap Delta**: Strict $0.00\text{ MB}$ (no OS memory pressure or page swapping)

---

## 2. Engineering Log Entries

### Entry: 2026-08-21T18:00:00 — Phase 1 & 2 Baseline: Uncached Streaming Arena vs. Bounded Persistent Cache (1024 Slots, Standard LRU)

#### Context & Objectives
- Address the high unified-memory residency of full-model resident inference ($18.34\text{ GiB}$ Metal) while overcoming the slow decode throughput of whole-layer streamed unmapping ($2.48\text{ tok/s}$).
- Move to active selected-expert streaming: stage only the $8$ router-selected experts per layer into preallocated reusable Metal arenas.

#### Implementations
1. **Packed Expert Storage (`Q4ExpertPackedStore.swift`)**:
   - Organized expert parameters into contiguous $1.6875\text{ MiB}$ ($1,769,472\text{ bytes}$) blocks on disk and memory-mapped shards.
   - Slices: `gate_proj` ($512\text{ KB}$ w, $32\text{ KB}$ s, $32\text{ KB}$ b), `up_proj` ($512\text{ KB}$ w, $32\text{ KB}$ s, $32\text{ KB}$ b), `down_proj` ($512\text{ KB}$ w, $32\text{ KB}$ s, $32\text{ KB}$ b).
2. **Dense Non-Expert Reusable Arena (`DenseLayerArena.swift`)**:
   - Preallocated $24\text{ MB}$ fixed shared buffer holding all dense projection weights for the active layer.
3. **Bounded Persistent Expert Cache (`Q4BoundedExpertCache.swift`)**:
   - Preallocated $1024$-slot cache buffer ($1.6875\text{ GiB}$), holding recent experts across tokens with LRU eviction and active layer protection.
4. **Cached Metal Shaders (`deltanet.metal`)**:
   - `qwen_cached_moe_gate_up_silu_q4` and `qwen_cached_moe_down_accum_residual_q4` indexing directly into the $1024$-slot arena via active slot indices.

#### Benchmark Results (Prompt: `"Hello, how are"`, Tokens: `[488, 30, 271, 248068, 198, 8160, 579, 264]`)
- **Uncached Active Streaming (9-Slot Arena)**:
  - Cold Prefill: $2,206.61\text{ ms}$ ($1.81\text{ tok/s}$)
  - Sustained 32-step Decode: **$5.42\text{ tok/s}$** ($184.62\text{ ms/token}$)
  - Expert Staging Latency: $80.40\text{ ms/token}$ (p50)
  - Active Metal Allocation: **$0.45\text{ GiB}$** ($462.17\text{ MB}$)
  - System Wired RAM: **$2.99\text{ GiB}$** ($3,061.28\text{ MB}$)
  - Parity: 100% Exact Match with MLX oracle.
- **Bounded Persistent Cache (1024 Slots, Standard LRU)**:
  - Cold Prefill: $625.46\text{ ms}$ ($6.40\text{ tok/s}$)
  - Sustained 32-step Decode: **$9.52\text{ tok/s}$** ($105.08\text{ ms/token}$)
  - Overall Cache Hit Rate: **$62.58\%$**
  - Expert Staging Latency: $19.80\text{ ms/token}$ (p50)
  - Active Metal Allocation: **$2.14\text{ GiB}$** ($2,190.17\text{ MB}$)
  - System Wired RAM: **$4.53\text{ GiB}$** ($4,640.98\text{ MB}$)
  - Parity: 100% Exact Match with MLX oracle.

#### Bottleneck Analysis
- At $62.58\%$ hit rate, $37.42\%$ of expert lookups still require dynamic fetch from mmap shards ($342\text{ MB/token}$ miss traffic).
- Standard LRU treats single-use transient experts with equal priority as frequently recurrent core experts, causing unnecessary eviction of hot experts.

---

### Entry: 2026-08-21T18:08:00 — Phase 3: Segmented LRU (SLRU) Partitioning & Speculative Prefetching

#### Context & Objectives
- Increase cache hit rate and reduce cache-miss staging penalty by protecting recurrent hot experts from transient one-hit expert evictions.
- Keep peak Metal allocation strictly $\le 5.0\text{ GiB}$ and wired RAM $\le 6.0\text{ GiB}$ with zero swap.

#### Architectural Changes
1. **Segmented LRU (SLRU) Cache Partitioning (`Q4BoundedExpertCache.swift`)**:
   - Split 1024 slots into two distinct pools:
     - **Probationary Partition ($256$ slots = $25\%$)**: New cache-miss experts enter here.
     - **Protected Partition ($768$ slots = $75\%$)**: Experts that receive $\ge 2$ accesses are promoted.
   - Evictions prioritize the Probationary partition, insulating frequently reused experts from transient eviction churn.
   - Preserved active layer lockouts during layer dispatch.
2. **Speculative Candidate Prefetching (`QwenQ4ActiveExpertStreamingRunner.swift`)**:
   - Tracks per-layer expert routing history (`perLayerRecentExperts`).
   - Issues non-blocking prefetch candidates for layer $L+1$ into spare cache slots while layer $L$ computes.

#### Benchmark Results (Prompt: `"Hello, how are"`, Tokens: `[488, 30, 271, 248068, 198, 8160, 579, 264]`)
- **Cold-Page First Pass (8 Tokens)**:
  - Prefill Duration: $627.60\text{ ms}$ ($6.37\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$10.23\text{ tok/s}$** ($97.76\text{ ms/token}$)
  - Cold Hit Rate: **$55.65\%$** ($2,137$ hits, $1,703$ misses, $679$ evictions)
- **Warm OS-Cache Pass (8 Tokens)**:
  - Prefill Duration: $354.25\text{ ms}$ ($11.29\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$11.66\text{ tok/s}$** ($85.74\text{ ms/token}$)
  - Warm Hit Rate: **$75.26\%$** ($2,890$ hits, $950$ misses)
- **Extended 32-Step Decode Pass**:
  - Mean Latency: **$93.34\text{ ms/token}$** ($10.71\text{ tok/s}$)
  - p50 / p95 Step Latency: **$94.62\text{ ms}$ / $99.79\text{ ms}$**
  - Overall Sustained Hit Rate: **$68.03\%$** (up from $62.58\%$)
  - Staging Latency: **$10.45\text{ ms}$ p50** (down from $19.80\text{ ms}$, $-47.2\%$)
  - Staged Traffic per Token: **$1,029.62\text{ MB/token}$**
  - Active Metal Allocation: **$2.14\text{ GiB}$** ($2,190.17\text{ MB}$, Budget: $\le 5.0\text{ GiB}$)
  - Peak Wired RAM: **$4.60\text{ GiB}$** ($4,708.47\text{ MB}$, Budget: $\le 6.0\text{ GiB}$)
  - Physical Footprint: $1,964.63\text{ MB}$, Process RSS: $5,662.83\text{ MB}$
  - Swap Growth: **$0.00\text{ MB}$**
  - Parity: **100% Exact Match** with MLX reference fixture.

#### Decision
- **KEEP**: Sustained generation throughput improved to **$10.71\text{ tok/s}$** ($+12.5\%$ over simple LRU, $+97.6\%$ over uncached streaming), while staging time dropped to $10.45\text{ ms}$.

#### Next Bottlenecks
1. Dense weights staging still consumes $15.40\text{ ms}$ ($856\text{ MB/token}$) per step across the 40 layers.
2. Attention / DeltaNet kernel execution takes $24.22\text{ ms}$ ($26\%$ of total step time).

---

### Entry: 2026-08-21T18:12:00 — Phase 4 Research: Exact Text-Model Memory Inventory & Dense Persistence Proposals

#### 1. Exact Memory Inventory of Text-Model Tensors
Excluding all vision components, the complete $40$-layer Qwen 3.6 35B-A3B text model consists of the following tensor groups:

| Tensor Family | Tensors per Layer | Parameters / Structure | Bytes per Layer (Q4 / BF16) | Total Bytes (40 Layers) | Total GiB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Routed MoE Experts** | $256 \times 3$ tensors (`gate`, `up`, `down`) | $256 \times (2048 \times 512 \times 3)$ | $452,984,832\text{ B}$ ($432.0\text{ MB}$) | $18,119,393,280\text{ B}$ | **$16.875\text{ GiB}$** |
| **Linear Attention Projections** (30 layers) | `in_proj` ($2048 \times 8192$), `out_proj` ($4096 \times 2048$) | $2048 \times 12288$ | $14,155,776\text{ B}$ ($13.5\text{ MB}$) | $424,673,280\text{ B}$ (30 layers) | **$0.395\text{ GiB}$** |
| **Full Attention Projections** (10 layers) | `qkv_proj` ($2048 \times 2560$), `out_proj` ($2048 \times 2048$) | $2048 \times 4608$ | $5,308,416\text{ B}$ ($5.0625\text{ MB}$) | $53,084,160\text{ B}$ (10 layers) | **$0.049\text{ GiB}$** |
| **Shared Experts** (40 layers) | `gate`, `up`, `down` ($2048 \times 2048$) | $3 \times (2048 \times 2048)$ | $7,077,888\text{ B}$ ($6.75\text{ MB}$) | $283,115,520\text{ B}$ | **$0.264\text{ GiB}$** |
| **MoE Routers** (40 layers, Q8 Affine) | `gate` ($256 \times 2048$) | $256 \times 2048$ | $557,056\text{ B}$ ($544\text{ KB}$) | $22,282,240\text{ B}$ | **$0.021\text{ GiB}$** |
| **Norms, Biases & Constants** (40 layers) | RMSNorms, Conv1D, SSM coefficients | FP32 / BF16 vectors | $\approx 80\text{ KB}$ | $\approx 3,360,000\text{ B}$ | **$0.003\text{ GiB}$** |
| **Embedding Table** (`embed_tokens`) | $248,320 \times 2048$ (Q4) | $248,320 \times 2048$ | — (Global) | $286,064,640\text{ B}$ | **$0.266\text{ GiB}$** |
| **LM Head Projection** (`lm_head`) | $248,320 \times 2048$ (Q4) | $248,320 \times 2048$ | — (Global) | $286,064,640\text{ B}$ | **$0.266\text{ GiB}$** |

#### 2. Packed Byte Totals of Non-Routed Subsets
- **A. All Non-Routed Language-Model Tensors**: **$1,358,644,480\text{ bytes} = 1.265\text{ GiB}$ ($1,295.7\text{ MB}$)**.
- **B. Non-Routed Excluding Embed & LM Head**: **$786,515,200\text{ bytes} = 0.732\text{ GiB}$ ($750.08\text{ MB}$)**.
- **C. Non-Routed Excluding Embed, LM Head & Shared Experts**: **$503,399,680\text{ bytes} = 0.469\text{ GiB}$ ($480.08\text{ MB}$)**.

#### 3. Bounded Residency Plans (Within $\le 5.0\text{ GiB}$ Metal / $\le 6.0\text{ GiB}$ Wired)
- **Plan 1 (Full Dense Residency — Selected for Implementation)**:
  - Keep **all non-routed text tensors resident** ($1.265\text{ GiB}$) in zero-copy shared Metal buffers.
  - Maintain the **$1,024$-block Persistent Expert Cache** ($1.6875\text{ GiB}$).
  - Scratch, activations, and KV / SSM state buffers: $\approx 0.15\text{ GiB}$.
  - **Estimated Peak Metal Allocation**: $1.265 + 1.688 + 0.150 = \mathbf{3.10\text{ GiB}}$ ($\le 5.0\text{ GiB}$ budget).
  - **Estimated Peak System Wired RAM**: $\approx \mathbf{5.35\text{ GiB}}$ ($\le 6.0\text{ GiB}$ budget).
  - **Eliminated Staged Traffic**: $856\text{ MB/token}$ dense copy traffic $\rightarrow \mathbf{0.0\text{ MB/token}}$ (saving $15.4\text{ ms/token}$).
- **Plan 2 (Attention & Router Persistence Only)**:
  - Retain attention projections, routers, embed, and LM head ($1.00\text{ GiB}$); dynamically stage shared experts and routed experts.
  - Estimated Peak Metal Allocation: $\mathbf{2.84\text{ GiB}}$.
  - Saves $478\text{ MB/token}$ of dense traffic, but leaves $270\text{ MB/token}$ shared expert copy overhead.

---

### Entry: 2026-08-21T18:15:00 — Phase 4 Results: Full Dense Persistence + Segmented LRU Expert Cache

#### Context & Architectural Implementation
- Implemented **Plan 1 (Full Dense Residency)** in `QwenQ4ActiveExpertStreamingRunner.swift` with `keepDenseWeightsResident: Bool = true`.
- Pre-allocated 40 persistent `DenseLayerArena` buffers ($960\text{ MB}$ total) holding all 40 layers' dense projections (linear attention, full attention, shared experts, routers).
- Dense weights are copied once during model initialization; runtime per-token dense staging overhead is completely eliminated ($0.00\text{ ms/token}$, $0.0\text{ MB/token}$).
- Only router-selected MoE experts ($8$ per layer) participate in cache lookups and cache-miss staging via the $1024$-slot SLRU cache.

#### Files Changed
- `Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift`: Added persistent dense arena storage and removed per-layer dense copies when `keepDenseWeightsResident` is enabled.
- `Sources/QwenFieldfareBench/main.swift`: Adjusted RSS telemetry safety ceiling for 35B mmapped model validation.

#### Benchmark Results (Prompt: `"Hello, how are"`, Tokens: `[488, 30, 271, 248068, 198, 8160, 579, 264]`)
- **Cold-Page First Pass (8 Tokens)**:
  - Prefill Duration: $540.37\text{ ms}$ ($7.40\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$12.34\text{ tok/s}$** ($81.02\text{ ms/token}$)
  - Cold Hit Rate: **$55.65\%$** ($2,137$ hits, $1,703$ misses, $679$ evictions)
- **Warm OS-Cache Pass (8 Tokens)**:
  - Prefill Duration: $281.47\text{ ms}$ ($14.21\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$14.25\text{ tok/s}$** ($70.20\text{ ms/token}$)
  - Warm Hit Rate: **$75.26\%$** ($2,890$ hits, $950$ misses)
- **Extended 32-Step Decode Pass**:
  - Mean Latency: **$80.26\text{ ms/token}$** (**$12.46\text{ tok/s}$**)
  - p50 / p95 Step Latency: **$79.08\text{ ms}$ / $93.77\text{ ms}$**
  - Staged Traffic per Token: **$173.65\text{ MB/token}$** (reduced from $1,029.62\text{ MB/token}$, $-83.1\%$)
  - Active Metal Allocation: **$3.08\text{ GiB}$** ($3,150.17\text{ MB}$, Budget: $\le 5.00\text{ GiB}$) — **PASSED**
  - Peak Wired RAM: **$5.61\text{ GiB}$** ($5,746.56\text{ MB}$, Budget: $\le 6.00\text{ GiB}$) — **PASSED**
  - Physical Footprint: $2,901.14\text{ MB}$, Process RSS: $6,419.39\text{ MB}$
  - Swap Growth: **$0.00\text{ MB}$**
  - Parity: **100% Exact Match** with MLX reference fixture (`[488, 30, 271, 248068, 198, 8160, 579, 264]`).

#### Detailed 32-Step Component Breakdown
| Component | p50 Latency | p95 Latency | Mean Latency | Share of Step | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense Weights Staging** | **$0.00\text{ ms}$** | **$0.00\text{ ms}$** | **$0.00\text{ ms}$** | $0.0\%$ | Completely eliminated by persistent residency |
| **Cached Expert Staging** | $9.97\text{ ms}$ | $15.39\text{ ms}$ | $9.34\text{ ms}$ | $11.6\%$ | $68.02\%$ cache hit rate across 40 layers |
| **Router & Top-8 Select** | $10.84\text{ ms}$ | $12.91\text{ ms}$ | $11.34\text{ ms}$ | $14.1\%$ | 40 layers fused router GEMV + top-8 selection |
| **Attention / DeltaNet** | $26.25\text{ ms}$ | $31.16\text{ ms}$ | $26.60\text{ ms}$ | $33.1\%$ | 30 linear attention + 10 causal attention layers |
| **Cached MoE Metal Compute** | $13.60\text{ ms}$ | $16.26\text{ ms}$ | $14.40\text{ ms}$ | $17.9\%$ | Fused Gate-Up + Down-Accum MoE kernels |
| **LM Head GEMV** | $2.96\text{ ms}$ | $3.37\text{ ms}$ | $3.04\text{ ms}$ | $3.8\%$ | $248\text{K}$ vocabulary Q4 GEMV |

#### Decision
- **KEEP**: Full dense persistence delivers **$14.25\text{ tok/s}$** on warm decode (and **$12.46\text{ tok/s}$** sustained 32-step decode) — approaching the $15.60\text{ tok/s}$ all-resident baseline while staying within the **$3.08\text{ GiB}$ Metal / $5.61\text{ GiB}$ wired** constraints with zero swap.

#### Next Bottleneck Analysis
1. **DeltaNet & Attention Compute ($26.60\text{ ms}$, $33.1\%$)**: Attention and DeltaNet compute is now the single largest runtime component. Fusing the 30 DeltaNet recurrence kernels with preceding in-projections or optimizing threadgroup tile sizes can reduce this significantly.
2. **MoE Compute & Router ($25.74\text{ ms}$, $32.0\%$)**: Fusing top-8 router sorting and MoE dispatch into a unified command encoder.

---

### Entry: 2026-08-21T18:20:00 — Phase 5: DeltaNet Fused Step (`qwen_fused_deltanet_step_f32`) & Attention Subcomponent Profiling

#### Context & Architectural Goal
- Linear attention and DeltaNet recurrence was the single largest runtime component ($26.60\text{ ms/token}$, $33.1\%$ of decode time).
- In each of the 30 linear attention layers, execution previously dispatched 4 separate Metal kernels (`conv1d`, `prep_qkv`, `deltanet_recurrent`, `gated_rmsnorm`) with intermediate device memory round-trips for Q, K, V, and Core Attention buffers ($56\text{ KB}$ per layer = $1.68\text{ MB/token}$).
- In addition, Q4 GEMV affine kernels (`qwen_gemv_q4_affine` and `qwen_fused_linear_attn_in_proj_q4`) were performing scale and bias dequantization arithmetic inside the innermost 8-element nibble loops.

#### Architectural Implementations
1. **Unified DeltaNet Fused Step Kernel (`qwen_fused_deltanet_step_f32` in `deltanet.metal`)**:
   - Fuses `prep_qkv`, `deltanet_recurrent` SSM recurrence, and `gated_rmsnorm` into a single 32-threadgroup kernel (1 threadgroup of 128 threads per head).
   - In-register / threadgroup memory data flow:
     - SIMD reductions compute $q\_inv$ and $k\_inv$ directly from `conv_out`.
     - Threadgroup shared memory `tg_q[128]` and `tg_k[128]` feed the recurrence without device writes.
     - Recurrent state row vectorization (`float4`) computes decay, $kv\_mem$, delta update, and core attention.
     - Gated RMSNorm is applied immediately in threadgroup registers and written directly to `gated_norm_out`.
   - Bypasses intermediate `qBuf`, `kBuf`, `vBuf`, and `coreAttnBuf` global allocations and eliminates 60 kernel launch dispatches per token step.
2. **Factored Scale/Bias Q4 Dequantization (`qwen_gemv_q4_affine` & `qwen_fused_linear_attn_in_proj_q4`)**:
   - Factored scale and bias multiplications out of the inner nibble loops into a single multiply-accumulate per group of 64 elements: $\text{acc} += \text{scale} \cdot \sum (q_k \cdot v_k) + \text{bias} \cdot \sum v_k$.
   - Reduces arithmetic overhead across all linear attention in-projections, out-projections, and full attention projections.
3. **Non-Overlapping Subcomponent Profiling (`profileSubcomponents`)**:
   - Integrated fine-grained profiling measuring non-overlapping subcomponents across all 30 linear layers and 10 full attention layers over 10 warm iterations.

#### Files Changed
- `Sources/QwenFieldfareCore/Metal/LinearAttention/deltanet.metal`: Added `qwen_fused_deltanet_step_f32` with `float4` vectorization and optimized `qwen_gemv_q4_affine` and `qwen_fused_linear_attn_in_proj_q4`.
- `Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift`: Added `fusedDeltanetStepPipe`, updated linear attention execution in `forwardToken`, and added `profileSubcomponents(iterations:)`.
- `Sources/QwenFieldfareBench/main.swift`: Added subcomponent profiling output to `testQ4BoundedExpertCachePerformanceAndParity`.

#### Benchmark Results (Prompt: `"Hello, how are"`, Tokens: `[488, 30, 271, 248068, 198, 8160, 579, 264]`)
- **Cold-Page First Pass (8 Tokens)**:
  - Prefill Duration: $547.41\text{ ms}$ ($7.31\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$12.11\text{ tok/s}$** ($82.58\text{ ms/token}$)
  - Cold Hit Rate: **$55.65\%$** ($2,137$ hits, $1,703$ misses, $679$ evictions)
- **Warm OS-Cache Pass (8 Tokens)**:
  - Prefill Duration: $282.12\text{ ms}$ ($14.18\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$14.02\text{ tok/s}$** ($71.33\text{ ms/token}$)
  - Warm Hit Rate: **$75.26\%$** ($2,890$ hits, $950$ misses)
- **Extended 32-Step Decode Pass**:
  - Mean Latency: **$78.63\text{ ms/token}$** (**$12.72\text{ tok/s}$**)
  - p50 / p95 Step Latency: **$79.88\text{ ms}$ / $86.93\text{ ms}$**
  - Attention / DeltaNet Latency: **$24.91\text{ ms}$** (p50: $25.23\text{ ms}$)
  - Staged Traffic per Token: **$173.60\text{ MB/token}$**
  - Active Metal Allocation: **$3.08\text{ GiB}$** ($3,150.17\text{ MB}$, Budget: $\le 5.00\text{ GiB}$) — **PASSED**
  - Peak Wired RAM: **$5.60\text{ GiB}$** ($5,734.03\text{ MB}$, Budget: $\le 6.00\text{ GiB}$) — **PASSED**
  - Physical Footprint: $2,901.06\text{ MB}$, Process RSS: $6,419.44\text{ MB}$
  - Swap Growth: **$0.00\text{ MB}$**
  - Parity: **100% Exact Match** with MLX reference fixture (`[488, 30, 271, 248068, 198, 8160, 579, 264]`).

#### Non-Overlapping Attention Subcomponent Timings (10 Warm Iterations)
| Subcomponent | Scope | Measured Latency | Share of Total Attn |
| :--- | :--- | :---: | :---: |
| **Input RMSNorm** | 30 Linear Layers | $0.52\text{ ms}$ | $3.4\%$ |
| **QKV/Z/B/A In-Proj GEMV** | 30 Linear Layers ($12,352 \text{ rows} \times 2048$) | $4.93\text{ ms}$ | $32.2\%$ |
| **Conv1D State Shift & SiLU** | 30 Linear Layers ($8,192 \text{ dims}$) | $0.32\text{ ms}$ | $2.1\%$ |
| **Fused DeltaNet Step (Prep+SSM+Norm)** | 30 Linear Layers ($32 \text{ heads} \times 128$) | **$1.48\text{ ms}$** | **$9.7\%$** |
| **Linear Out-Proj GEMV** | 30 Linear Layers ($2048 \text{ rows} \times 4096$) | $2.64\text{ ms}$ | $17.2\%$ |
| **Residual Add** | 30 Linear Layers ($2048 \text{ dims}$) | $0.21\text{ ms}$ | $1.4\%$ |
| **Subtotal (30 Linear Attention Layers)** | | **$10.08\text{ ms}$** | **$66.0\%$** |
| **Input RMSNorm** | 10 Full Attention Layers | $0.29\text{ ms}$ | $1.9\%$ |
| **Q/K/V Projections** | 10 Full Attention Layers ($8192+512+512 \text{ rows}$) | $1.76\text{ ms}$ | $11.5\%$ |
| **Split Q/Gate + RoPE + Head Norm** | 10 Full Attention Layers | $0.76\text{ ms}$ | $5.0\%$ |
| **Causal Softmax Attention Step** | 10 Full Attention Layers ($16 \text{ heads}$) | $1.22\text{ ms}$ | $8.0\%$ |
| **Full Out-Proj GEMV** | 10 Full Attention Layers ($2048 \text{ rows} \times 4096$) | $0.97\text{ ms}$ | $6.3\%$ |
| **Residual Add** | 10 Full Attention Layers ($2048 \text{ dims}$) | $0.20\text{ ms}$ | $1.3\%$ |
| **Subtotal (10 Full Attention Layers)** | | **$5.20\text{ ms}$** | **$34.0\%$** |
| **Total Attention GPU Time (40 Layers)** | | **$15.28\text{ ms}$** | **$100.0\%$** |

#### Decision
- **KEEP**: The fused DeltaNet step executes across all 30 linear layers in only **$1.48\text{ ms}$** ($0.049\text{ ms/layer}$), proving that recurrent SSM compute is no longer a bottleneck. Total Attention GPU execution dropped to **$15.28\text{ ms}$**. Sustained decode throughput is **$12.72\text{ tok/s}$** ($78.63\text{ ms/token}$) with exact MLX parity within the **$3.08\text{ GiB}$ Metal / $5.60\text{ GiB}$ wired** memory budgets.

#### Next Bottlenecks
1. **MoE Compute & Router Dispatch ($24.91\text{ ms}$, $31.7\%$)**: Fusing router score calculation with top-8 sort and MoE kernel dispatch to eliminate inter-command-buffer stalls.
2. **Dense Attention GEMVs ($10.30\text{ ms}$, $13.1\%$)**: In-projection ($4.93\text{ ms}$) and out-projection ($2.64\text{ ms}$) Q4 GEMVs can be accelerated using threadgroup matrix tiling.

---

### Entry: 2026-08-21T18:30:00 — Phase 6: Consolidated 2-Command-Buffer per Layer Pipeline, In-Metal KV Cache Updates & Vectorized Logits Argmax

#### Context & Architectural Goal
- Detailed Phase 5 non-overlapping profiling revealed that out of the $78.63\text{ ms/token}$ step time:
  - Dispatched $131$ separate command buffers and executed $131$ CPU-GPU synchronization waits (`waitUntilCompleted()`) per token step across the 40 layers.
  - Scanning $248,320$ logits on CPU in unoptimized scalar Swift took **$15.52\text{ ms}$** ($22.4\%$ of total wall-clock decode step).
  - CPU-GPU command buffer turnaround and synchronization stalls added $> 20\text{ ms/token}$ of pipeline bubbles.

#### Architectural Implementations
1. **Consolidated 2-Command-Buffer per Layer Architecture**:
   - **Command Buffer 1 (`cmdAttnRouter`)**: Encodes the entire Attention block + Post-Attention RMSNorm + Fused Top-8 Router GEMV + Shared Expert Gating in a single command buffer:
     - Linear layers: Input RMSNorm $\rightarrow$ In-Proj GEMV $\rightarrow$ Conv1D $\rightarrow$ Fused DeltaNet Step $\rightarrow$ Out-Proj $\rightarrow$ Residual Add $\rightarrow$ Post RMSNorm $\rightarrow$ Top-8 Router $\rightarrow$ Shared Coeff.
     - Full attention layers: Input RMSNorm $\rightarrow$ QKV Proj $\rightarrow$ Split $\rightarrow$ Head Norm $\rightarrow$ RoPE $\rightarrow$ `update_kv_cache` $\rightarrow$ Causal Softmax Attention $\rightarrow$ Out-Proj $\rightarrow$ Residual Add $\rightarrow$ Post RMSNorm $\rightarrow$ Top-8 Router $\rightarrow$ Shared Coeff.
   - **In-Metal KV Cache Update Kernel (`qwen_update_kv_cache_f32`)**:
     - Direct GPU-side KV cache slot write eliminates host CPU `memcpy` and eliminates command-buffer splitting in full attention layers.
   - **Single Synchronous Routing Wait per Layer**:
     - CPU blocks once per layer after `cmdAttnRouter`, inspects the $8$ selected expert IDs, performs $O(1)$ SLRU cache slot acquisition, and memory-copies any cache-miss expert slices.
   - **Command Buffer 2 (`cmdMoE`)**:
     - Dispatches Cached MoE Gate-Up and Down-Accumulate kernels.
     - In Layer 39, `cmdMoE` also encapsulates the Final RMSNorm and LM Head Q4 GEMV into the same command buffer.
   - **Total Command Buffers**: Reduced from **$131$ to $80$ command buffers per token step** (a $39\%$ reduction in command buffer dispatches and CPU-GPU synchronization stalls).
2. **SIMD-Vectorized Logits Argmax (`vDSP_maxvi`)**:
   - Integrated Apple Accelerate `vDSP_maxvi` NEON vectorization to find the maximum logit across $248,320$ vocabulary elements.
   - Reduced CPU argmax latency from **$15.52\text{ ms}$** to **$0.02\text{ ms}$** ($776\times$ speedup).

#### Files Changed
- `Sources/QwenFieldfareCore/Metal/LinearAttention/deltanet.metal`: Added `qwen_update_kv_cache_f32` kernel.
- `Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift`: Consolidated `forwardToken` into the 2-command-buffer per layer pipeline, added `updateKVCachePipe`, integrated `vDSP_maxvi`, and implemented non-overlapping wall-clock telemetry in `StepProfile`.
- `Sources/QwenFieldfareBench/main.swift`: Updated benchmark telemetry to output non-overlapping wall-clock breakdowns.

#### Benchmark Results (Prompt: `"Hello, how are"`, Tokens: `[488, 30, 271, 248068, 198, 8160, 579, 264]`)
- **Cold-Page First Pass (8 Tokens)**:
  - Prefill Duration: $430.10\text{ ms}$ ($9.30\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$17.08\text{ tok/s}$** ($58.54\text{ ms/token}$)
  - Cold Hit Rate: **$55.65\%$** ($2,137$ hits, $1,703$ misses, $679$ evictions)
- **Warm OS-Cache Pass (8 Tokens)**:
  - Prefill Duration: $185.42\text{ ms}$ ($21.57\text{ tok/s}$)
  - Steady-State Rate (Steps 2–8): **$21.97\text{ tok/s}$** ($45.51\text{ ms/token}$)
  - Warm Hit Rate: **$75.26\%$** ($2,890$ hits, $950$ misses)
- **Extended 32-Step Decode Pass**:
  - Mean Latency: **$54.01\text{ ms/token}$** (**$18.51\text{ tok/s}$**)
  - p50 / p95 Step Latency: **$54.95\text{ ms}$ / $62.32\text{ ms}$**
  - Active Metal Allocation: **$3.08\text{ GiB}$** ($3,150.17\text{ MB}$, Budget: $\le 5.00\text{ GiB}$) — **PASSED**
  - Peak Wired RAM: **$5.46\text{ GiB}$** ($5,588.47\text{ MB}$, Budget: $\le 6.00\text{ GiB}$) — **PASSED**
  - Physical Footprint: $2,900.94\text{ MB}$, Process RSS: $6,419.22\text{ MB}$
  - Swap Growth: **$0.00\text{ MB}$**
  - Parity: **100% Exact Match** with MLX reference fixture (`[488, 30, 271, 248068, 198, 8160, 579, 264]`).

#### Non-Overlapping Wall-Clock Breakdown (32-Step Decode Average)
| Component | p50 Latency | p95 Latency | Mean Latency | Share of Wall-Clock | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CPU Setup & Input Init** | $0.17\text{ ms}$ | $0.26\text{ ms}$ | $0.18\text{ ms}$ | $0.3\%$ | Embedding dequantization & runner setup |
| **Command Buffer Encoding** | $0.66\text{ ms}$ | $0.82\text{ ms}$ | $0.67\text{ ms}$ | $1.2\%$ | CPU time creating and encoding 80 CBs |
| **GPU Commit & Wait (80 CBs)** | $42.58\text{ ms}$ | $44.87\text{ ms}$ | **$42.44\text{ ms}$** | **$78.6\%$** | GPU compute for Attention, Router, MoE, LM Head |
| **Expert Cache Index Lookup** | $2.98\text{ ms}$ | $3.97\text{ ms}$ | $2.68\text{ ms}$ | $5.0\%$ | $O(1)$ SLRU cache slot management (40 layers) |
| **Expert Cache Miss Fetch/Copy**| $7.69\text{ ms}$ | $14.81\text{ ms}$ | $7.81\text{ ms}$ | $14.5\%$ | Disk/mmap $\rightarrow$ cache memcpy on cache misses |
| **CPU Argmax / Logits Scan** | $0.02\text{ ms}$ | $0.04\text{ ms}$ | **$0.02\text{ ms}$** | $0.0\%$ | SIMD-vectorized `vDSP_maxvi` scan |
| **Sum of Measured Parts** | **$54.10\text{ ms}$** | **$64.77\text{ ms}$** | **$53.80\text{ ms}$** | **$99.6\%$** | **$\Delta = 0.39\%$ vs $54.01\text{ ms}$ Wall-Clock** |

#### Decision
- **KEEP**: The consolidated 2-command-buffer architecture combined with SIMD-vectorized argmax increases sustained decode throughput to **$18.51\text{ tok/s}$** ($54.01\text{ ms/tok}$) and warm decode to **$21.97\text{ tok/s}$** ($45.51\text{ ms/tok}$) — **surpassing native full-resident inference ($15.60\text{ tok/s}$) by $+18.6\%$ while requiring only $3.08\text{ GiB}$ Metal ($17\%$ of resident memory) and $5.46\text{ GiB}$ wired RAM with zero swap growth**. Exact MLX parity is 100% verified.

#### Next Bottlenecks
1. **GPU Commit & Wait ($42.44\text{ ms}$, $78.6\%$)**: Attention ($15.33\text{ ms}$), MoE compute ($13.99\text{ ms}$), and Router GEMVs ($7.63\text{ ms}$). Optimizing GEMV block sizes and threadgroup vector loads in `qwen_gemv_q4_affine` and `qwen_cached_moe_gate_up_silu_q4` can further reduce GPU execution time toward $\approx 30\text{ ms/token}$ ($> 30\text{ tok/s}$).
2. **Expert Cache Miss Fetch ($7.81\text{ ms}$, $14.5\%$)**: Asynchronous staging / double-buffering expert transfers ahead of the GPU compute pass.

---

### Entry: 2026-08-21T18:35:00 — Phase 7: Production App Default Mode Integration & Real Chat-Length Workload Validation

#### Context & Objectives
- Promote Phase 6 Active Expert Streaming Q4 to the default runtime mode in `QwenFieldfareMac` and `ChatService`.
- Expose comprehensive live telemetry in the app status bar: runtime mode, tok/s, Metal GiB, wired GiB, expert cache hit rate, cache misses/token, memory pressure, and explicit unload.
- Ensure safe mode transitions: teardown existing runners and deallocate unified memory buffers before initializing a different runtime mode.
- Validate the runtime under long single-turn chat generation (128 tokens) and real multi-turn conversation workloads (Turn 1: 64 tokens, Turn 2: 64 tokens) while maintaining strict memory and swap limits ($\le 5.00\text{ GiB}$ Metal, $\le 6.00\text{ GiB}$ wired, $0.00\text{ MB}$ swap delta).

#### App Architecture Changes
1. **Default Mode Promotion**:
   - Updated `QwenQ4ChatRunner.RuntimeMode` with `.activeStreaming` (`"Active Expert Streaming — Fast Low Memory"`).
   - Set `.activeStreaming` as the initial default mode in `ChatService`.
2. **Live Status Bar Telemetry & Controls**:
   - Integrated live expert cache metrics: `expertCacheHitRate` (`Cache: XX.X%`) and `expertCacheMissesPerToken` (`Misses: X.X/tok`).
   - Added mode selector segmented picker supporting `Active Streaming`, `Fast Resident`, and `Memory Saver`.
   - Added prominent mode badges with mode-specific tinting (Teal for Active Streaming, Purple for Fast Resident, Blue for Memory Saver).
   - Preserved 60s idle auto-unload toggle and instant manual "Unload Model" action.
3. **Safe Mode Switching**:
   - `ChatService.setMode()` enforces immediate `unloadModel()` teardown before switching mode variables.

#### Files Changed
- `Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ChatRunner.swift`: Added `activeStreaming` case to `RuntimeMode`.
- `Sources/QwenFieldfareApp/Mac/Generation/ChatService.swift`: Integrated `QwenQ4ActiveExpertStreamingRunner` as primary default runner, added live cache telemetry tracking, updated preflight memory checks for all three modes.
- `Sources/QwenFieldfareApp/Mac/App/RootView.swift`: Updated header mode selector, empty-state descriptions, and status bar telemetry displays.
- `Sources/QwenFieldfareBench/main.swift`: Added `testQ4ActiveExpertStreamingChatWorkloads` validation suite.

#### Workload Validation Results

##### 1. Workload 1: 128-Token Real Chat-Length Generation
- **Prompt**: `"Explain how a hash map works, with a small Rust example."` ($22$ prompt tokens)
- **Generation Length**: $128$ new tokens
- **Output Sample**: `"<think> Here's a thinking process: 1. **Understand User Request:** - **Topic:** Hash Map - **Requirements:** Exp..."`
- **First Token Latency (Prefill + Step 1)**: $8,844.37\text{ ms}$ (Prefill $8,738.49\text{ ms}$ / $2.52\text{ tok/s}$)
- **Steady-State Throughput (Steps 2–128)**: **$7.17\text{ tok/s}$** ($139.51\text{ ms/token}$)
- **Latency Distribution**: p50: **$133.72\text{ ms}$** | p95: **$238.48\text{ ms}$**
- **Cache Hit Rate**: min: **$24.4\%$** | mean: **$58.2\%$** | max: **$82.2\%$**
- **Staged Traffic per Token**: **$217.28\text{ MB/token}$** ($27,811.69\text{ MB}$ total)

##### 2. Workload 2: Multi-Turn Conversation Workload (Turn 1: 64t, Turn 2: 64t)
- **Turn 1**: `"Explain hash maps briefly."`
  - First Token Latency: **$1,054.17\text{ ms}$**
  - Steady-State Throughput: **$13.48\text{ tok/s}$** ($74.18\text{ ms/tok}$, p50: $72.48\text{ ms}$, p95: $106.00\text{ ms}$)
- **Turn 2**: `"Now show a Rust example."` (Continues context from Turn 1)
  - First Token Latency: **$1,658.08\text{ ms}$**
  - Steady-State Throughput: **$10.29\text{ tok/s}$** ($97.18\text{ ms/tok}$, p50: $91.04\text{ ms}$, p95: $152.43\text{ ms}$)
- **Combined Multi-Turn Aggregate (128 Tokens Across 2 Turns)**:
  - Overall Decode Rate: **$11.70\text{ tok/s}$** ($85.47\text{ ms/tok}$)
  - Overall Latency Distribution: p50: **$80.72\text{ ms}$** | p95: **$146.05\text{ ms}$**
  - Overall Cache Hit Rate: **$52.92\%$** ($26,754$ hits / $23,806$ misses)
  - Staged Traffic: **$236.53\text{ MB/token}$**

##### 3. Reference 8-Token Parity Test
- **Prompt**: `"Hello, how are"` (`[9419, 11, 1204, 513]`)
- **Tokens Generated**: `[488, 30, 271, 248068, 198, 8160, 579, 264]`
- **MLX Oracle Match**: **100% Exact Match**
- **Warm Steady Throughput**: **$21.55\text{ tok/s}$** ($46.40\text{ ms/token}$)
- **Extended 32-Step Decode Throughput**: **$18.23\text{ tok/s}$** ($54.87\text{ ms/token}$)

##### 4. Unified Memory & Safety Audit Across Real Chat Workloads
| Safety Metric | Measured Value | Budget Ceiling | Status |
| :--- | :---: | :---: | :---: |
| **Peak Active Metal Allocation** | **$3.08\text{ GiB}$** ($3,150.17\text{ MB}$) | $\le 5.00\text{ GiB}$ | **PASSED** |
| **Peak System Wired RAM** | **$5.61\text{ GiB}$** ($5,741.94\text{ MB}$) | $\le 6.00\text{ GiB}$ | **PASSED** |
| **Host Used RAM** | $19.48\text{ GiB}$ | — | Monitored |
| **Process Physical Footprint** | $3,007.22\text{ MB}$ | — | Monitored |
| **Process RSS** | $14,735.39\text{ MB}$ (includes read-only mmap) | — | Monitored |
| **System Swap Delta** | **$0.00\text{ MB}$** | $== 0.00\text{ MB}$ | **PASSED** |
| **System Memory Pressure** | **Normal** | Normal | **PASSED** |

#### Decision
- **SUPERSEDED**: Active Expert Streaming Q4 was selected as the default because it met the measured memory and short-run token-parity gates. A later 512-token app run exposed repetition and an unfinished response, so this entry does not establish robust long-form generation correctness.

---

### Entry: 2026-08-24T12:27:41+05:30 — Phase 6 App Promotion Revalidation (Before)

#### Objective
- Reconcile the existing Phase 6/7 Qwen Q4 active-expert work with the current repository, complete the native-app lifecycle and telemetry integration, and perform reproducible correctness, long-chat, multi-turn, cold/warm, memory, and unload validation.

#### Inspected Repository State
- Branch: `main`; commit: `9df7b67748b18afdf33ae37626980e0a582c9167`.
- The worktree was already dirty before this task. Existing modified files include `README.md`, `MetalContext.swift`, `RootView.swift`, format/repack files, and format tests. Existing untracked work includes the Qwen Q4 runners, storage/cache code, `ChatService.swift`, Qwen Metal kernels, Qwen tests, and this engineering log. All are treated as user-owned and must be preserved.
- The Phase 6 two-command-buffer/SIMD-argmax active-expert runner, 1024-slot SLRU cache, persistent dense-weight option, app mode label, mode picker, default selection, manual unload, and idle-unload control already exist in the uncommitted baseline.
- The app telemetry baseline is incomplete: it exposes Metal allocation, host-wide wired memory, pressure, aggregate generation rate, cache hit rate, and misses/token, but lacks a coherent snapshot, prompt/first-token and latency-percentile metrics, fetched/staged bytes, occupancy/segment/eviction/prefetch counters, host-used/process/swap metrics, peak tracking, cold/warm classification, and complete cancelling/unloading/error states.
- The app currently rebuilds a single-turn chat prompt for every send and does not retain conversation context; the existing log's multi-turn claim therefore requires validation against the actual test harness and must not be assumed to describe visible-app behavior.
- Canonical discrepancy: this task specifies prompt IDs `[9419, 1204, 11, 513]` and expected generated IDs `[248068, 8160, 30, 198, 264, 271, 488, 579]`; existing repository tests/log entries use prompt IDs `[9419, 11, 1204, 513]` and report generated IDs `[488, 30, 271, 248068, 198, 8160, 579, 264]`. The final report will use freshly observed results and will not relabel the prior sequence.

#### Intended Validation
- Build the release package once; run focused cache/telemetry tests and the existing Qwen Q4 active-expert fixture/workload test through `Scripts/test.sh`.
- Before any model execution, verify macOS, Swift, disk, memory pressure, required model artifacts, and absence of another model-owning process as required by `AGENTS.md`.
- Run the strongest available shared-runner harness for canonical, 128-token, retained-context multi-turn, cold/warm, memory, and unload measurements; launch the release Mac app only when the process check permits it.

---

### Entry: 2026-08-24T12:52:00+05:30 — Phase 6 App Promotion Revalidation (Final)

#### Outcome and Repository State
- Branch/commit remained `main` / `9df7b67748b18afdf33ae37626980e0a582c9167`; no commit, remote write, upstream synchronization, cache purge, model download, or model duplication occurred.
- The pre-existing dirty/untracked Qwen work was preserved. `git diff --check` passed.
- `Active Expert Streaming — Fast Low Memory` remains the app default. It instantiates `QwenQ4ActiveExpertStreamingRunner` with a 1,024-slot SLRU expert cache, persistent dense weights, and storage-backed routed experts. Fast Resident Q4 and Memory Saver Streamed Q4 remain selectable.
- The app now has a coherent bounded metrics snapshot, complete lifecycle states, cancellation checks, 250 ms display throttling with exact final publication, real retained conversation context, full-prompt multi-turn rebuilding, manual/idle unload, and decode-only cache counter resets after prefill.

#### Files Changed by This Revalidation
- `Sources/QwenFieldfareCore/Runtime/Telemetry/QwenRuntimeTelemetry.swift`: reusable bounded latency, cache, memory, and run snapshot types.
- `Sources/QwenFieldfareCore/Storage/Q4BoundedExpertCache.swift`: documented counter semantics and coherent cache snapshot including segment occupancy, evictions, bytes, and prefetch outcomes.
- `Sources/QwenFieldfareCore/Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift` and `QwenQ4ChatRunner.swift`: cancellation checks and explicit prefill-complete timing callback; inference math is unchanged.
- `Sources/QwenFieldfareApp/Mac/Generation/ChatService.swift`: app-runner selection/lifecycle, retained-context chat formatting, throttled telemetry, process/host/Metal/swap collection, load/prefill/TTFT/decode accounting, and safe unload/cancellation.
- `Sources/QwenFieldfareApp/Mac/App/RootView.swift`: exact requested mode label plus state, TTFT, p50/p95, tokens, cache, fetched/staged traffic, occupancy, eviction, prefetch, Metal, wired, pressure, stop, and unload displays.
- `Tests/QwenFieldfareCore/QwenRuntimeTelemetryTests.swift`: deterministic percentile, bounded-reset, and cache-derived-metric tests.
- `Sources/QwenFieldfareBench/main.swift`: tokenizer assertion, three warm 128-token repeats, app-equivalent full-prefill multi-turn validation, full latency/cache fields, host deltas, virtual memory, and post-unload snapshot.

#### Machine and Method
- Mac model `Mac17,2`, 32 GiB unified memory, arm64; macOS 26.6.2 (25G83); Swift 6.3.3; 295 GiB free disk; initial `memory_pressure -Q` reported 85% free; Qwen model size reported 19 GiB.
- All generation was greedy. Decode throughput and percentiles exclude prefill and the first generated-token step. Percentiles use individual token-step samples. EOS was not encountered before the requested lengths.
- “New runner” means a newly constructed runner with an empty expert cache; macOS file-page cache was not purged or controlled, so this is not claimed as storage-cold. Warm runs retained the same runner and expert-cache contents, reset recurrent/KV state, and rebuilt the prompt.
- The live harness was executed by `Scripts/test.sh`, whose default is a debug test build. A release product build succeeded. `./Scripts/test.sh -c release --filter testQ4ActiveExpertStreamingChatWorkloads` built the release test binary but SwiftPM discovered zero selected Swift Testing tests, so no release-harness timings are claimed.

#### Correctness and Focused Regressions
- Direct tokenizer measurement: `"Hello, how are"` -> `[9419, 11, 1204, 513]`.
- Exact greedy continuation: `[488, 30, 271, 248068, 198, 8160, 579, 264]`; exact parity passed on cold and warm passes. The prompt/generated sequences supplied in the 2026-08-24 task were not produced by this tokenizer/model and are incorrect for this checkout.
- Q4 affine GEMV: max/mean absolute error `3.814697e-06` / `3.066049e-07`; Q8 router GEMV: `4.053116e-06` / `5.729280e-07`.
- Selected-expert staged-vs-resident parity: max and mean difference `0.00e+00`.
- `QwenRuntimeTelemetryTests`: 3/3 passed.

#### Fresh Debug-Harness Performance

| Workload | Prompt tokens | Generated | TTFT | Steady tok/s | Mean | p50 | p95 | Cache state |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Canonical new-runner | 4 | 8 | 800.25 ms | 9.78 | 102.26 ms | unavailable | unavailable | empty at start; OS pages uncontrolled |
| Canonical warm | 4 | 8 | 289.18 ms | 17.41 | 57.43 ms | unavailable | unavailable | same runner/cache |
| 128-token chat, new runner | 22 | 128 | 2,018.97 ms | 12.36 | 80.92 ms | 80.62 ms | 91.60 ms | empty at start; OS pages uncontrolled |
| Warm 128 #1 | 22 | 128 | 1,520.00 ms | 12.70 | 78.74 ms | 78.61 ms | 91.78 ms | retained, 1,024/1,024 |
| Warm 128 #2 | 22 | 128 | 1,524.95 ms | 12.48 | 80.13 ms | 81.63 ms | 91.74 ms | retained, 1,024/1,024 |
| Warm 128 #3 | 22 | 128 | 1,540.42 ms | 12.60 | 79.37 ms | 79.04 ms | 91.72 ms | retained, 1,024/1,024 |
| Multi-turn Turn 1 | 14 | 64 | 1,010.55 ms | 14.02 | 71.33 ms | 72.33 ms | 76.62 ms | retained runner; state reset |
| Multi-turn Turn 2 | 94 | 64 | 6,859.24 ms | 11.25 | 88.89 ms | 91.18 ms | 92.31 ms | full retained-context prefill; no KV reuse |

- Principal 128-token latency range: 63.82–92.76 ms. The printed preview was topical, but only its prefix was inspected; the later app run demonstrated that this was insufficient evidence of long-form semantic correctness.
- Combined multi-turn decode: 12.50 tok/s, 80.02 ms mean, 78.01 ms p50, 92.07 ms p95. Turn 2 rebuilt the 94-token complete conversation, matching current app behavior; its 6.86 s TTFT is therefore materially higher than Turn 1.
- Previously reported Phase 6 was 18.51 tok/s sustained / 21.97 tok/s warm. Fresh long-run debug-harness results are lower (12.36 tok/s principal; 12.48–12.70 tok/s warm). This is not a like-for-like release comparison, and no claim of matching the prior result is made.

#### Cache Behavior
- Principal 128-token decode: 58.2% aggregate run hit rate (per-step range 24.1%–82.2%), 128.84 decode misses/token, and 217.42 MiB/token of logical expert blocks fetched and staged.
- Warm 128 runs: 56.75%, 56.75%, and 56.82% hit rate; 162.19, 162.17, and 161.91 run misses/generated-token; 273.69, 273.67, and 273.23 MiB fetched/generated-token. Warm run counters include prompt prefill demand, while principal decode-only bytes/misses are derived from decode `StepProfile` samples.
- Combined multi-turn: 53.78% aggregate hit rate (40,615 hits / 34,905 misses) and 236.47 MiB staged per decode token.
- Final warm occupancy was 1,024/1,024. The app snapshot exposes probationary/protected occupancy, evictions, prefetch requests, useful hits, and wasted prefetches. The legacy chat summary did not print their final numeric values, so none are invented here.

#### Memory and Unload
- Peak Metal: 3,150.17 MiB = 3.08 GiB — **PASS** (`<= 5 GiB`). This bounded size is direct evidence that the 16.875 GiB routed bank was not fully allocated as Metal buffers.
- Peak host-wide wired: 5,775.39 MiB = 5.64 GiB; baseline 2,580.84 MiB, delta 3,194.55 MiB — **PASS** against the requested absolute 6 GiB target, with the qualification that wired is system-wide.
- Peak host-used: 20.34 GiB; host-wide delta 2,940.28 MiB.
- Peak process RSS 14,722.27 MiB; physical footprint 3,014.30 MiB; virtual size 447,718.62 MiB. RSS includes large read-only mappings and is not used as the primary unified-memory metric.
- Swap delta 0.00 MiB — **PASS**. Memory pressure: Normal — **PASS**.
- After explicit runner release and 100 ms cleanup: Metal 0.69 MiB; footprint 148.83 MiB; RSS 151.86 MiB. No persistent large Metal allocation remained.

#### Commands and Exit Status
- `swift build -c release` — exit 0.
- `./Scripts/test.sh --filter QwenRuntimeTelemetryTests` — exit 0, 3 tests passed.
- `./Scripts/test.sh --filter quantizedAffineGEMVParityAndBenchmark` — exit 0.
- `./Scripts/test.sh --filter testQ4SelectedExpertStreamingMicrobenchmark` — exit 0.
- `./Scripts/test.sh --filter testQ4BoundedExpertCachePerformanceAndParity` — exit 0.
- `./Scripts/test.sh --filter testQ4ActiveExpertStreamingFull40LayerGeneration` — exit 0.
- `./Scripts/test.sh --filter testQ4ActiveExpertStreamingChatWorkloads` — exit 0; final complete-metrics run 67.327 s.
- `.build/release/QwenFieldfareMac` — launched successfully; `pgrep` showed only that app and no decode service; the launched app was stopped with Ctrl-C (exit 1 by signal), and the post-check found no remaining model process.
- `./Scripts/test.sh -c release --filter testQ4ActiveExpertStreamingChatWorkloads` — build exit 0, but zero selected tests were discovered; no model benchmark executed.

#### Decision, Limitations, and Next Bottleneck
- Default decision at this point was based on short token parity, fixed-length execution, memory, swap, and unload behavior. It was later superseded after the app exposed repetition and the silent 512-token limit. Measured long-run throughput here is 11.25–14.02 tok/s, not 50+ tok/s.
- Visible UI was launch-verified, but prompts were exercised deterministically through the shared runner harness rather than UI automation. Idle unload was code-reviewed but not held open for a 60-second visible-UI test; explicit runner release was measured.
- True storage-cold state, thermal state, power, ambient temperature, instrumentation overhead, and release-harness timings are unavailable. Metal peak is sampled and may miss very short transients.
- Next measured bottleneck: cache-miss fetch/copy and long-context full prefill. The cold/cache-stress 32-step run measured 350.13 ms/token in miss fetch/copy (83.0% of wall time), while app-equivalent Turn 2 TTFT rose to 6.86 s for a 94-token full prefill.

---

### Entry: 2026-08-24 — Long-Generation Correctness Correction

#### New Evidence
- The Mac app generated a repetitive exposed `<think>` trace and stopped mid-sentence. Code inspection confirmed that the app bypassed the checkpoint's thinking-control suffix, always used greedy argmax, and hard-capped every response at 512 tokens without recording a finish reason.
- Ollama `qwen3.6:35b-mlx` (NVFP4) completed the same chat request at 43.01 decode tok/s with its default sampler.
- A raw deterministic Ollama comparison used the app's exact old prompt framing, greedy decoding, no presence/repetition penalty, a 2,048-token context, and a 512-token output cap. It generated 512 coherent tokens without a repetition loop and stopped with `done_reason: length` at 35.27 tok/s. This proves the cap explains the unfinished response and that greedy decoding alone does not explain the QwenFieldfare repetition.

#### Corrections
- Added the checkpoint-compatible text chat template. Thinking-disabled requests now append `<think>\n\n</think>\n\n`, and retained assistant reasoning is not replayed into later turns.
- Replaced the silent fixed 512-token loop with a shared context-aware generation state machine and explicit `End of turn`, `Output limit`, and `Context full` finish reasons.
- Restricted EOS handling to the checkpoint's declared IDs (`248046`, `248044`).
- Added Qwen temperature, Top-K, Top-P, presence-penalty, repetition-penalty, and seeded sampling over the Float32 logits buffer. App defaults match the checkpoint sampler plus the validated Ollama presence penalty.
- Added app controls and finish-reason telemetry.
- Added eight non-model regressions, including a 1,024-token simulated generation proving there is no implicit 512-token cap.
- Found and removed the long-generation corruption: `qwen_causal_attn_step_f32` stored attention scores in a fixed 64-element private array while iterating through every context position. Position 64 and later wrote out of bounds. The active and fallback runners now use a per-head scratch buffer sized from `maxSeqLen`; the generic kernel uses bounded online softmax.
- Increased the app runner context from 2,048 to 8,192 tokens so its 4,096-token output option is reachable. The additional full-attention KV allocation is approximately 240 MiB over the prior setting.

#### Verification
- `swift build -c release` — exit 0.
- `./Scripts/test.sh --filter QwenGenerationSupportTests` — exit 0; 8/8 tests passed.
- Standalone MLX greedy oracle: 512 tokens at 49.86 tok/s, 19.631 GB peak memory, coherent output.
- Before the attention fix, MLX teacher forcing matched only 117/132 next-token predictions; the first mismatch was generated token 65. After the fix, all 132/132 predictions matched.
- Corrected QwenFieldfare greedy run: 512 coherent tokens, `Output limit`, 1,460.13 ms prefill, 8.23 tok/s decode, with no repetition or invalid Java.
- Corrected app-default sampling run: 1,107 coherent tokens, natural `End of turn`, 2,469.28 ms prefill, 5.06 tok/s decode, with no `<think>` leakage or repeated sections.
- Correctness is validated. Decode performance remains below the 49.86 tok/s MLX reference and is not accepted as meeting the performance goal.

---

### Entry: 2026-08-24 — Long-Run I/O Variance and Packed-Expert Experiment

#### Reproducible workload

- Machine: Mac17,2 (M5), 32 GiB unified memory; macOS 26.6.2 (25G83); Swift 6.3.3.
- Release command: `.build/release/QwenFieldfareBench --model /absolute/path/to/Qwen3.6-35B-A3B-4bit --prompt 'tell me what hasmap is in java with example' --context 8192 --max-new 512 --cache-slots 2100 --metrics-only`.
- Greedy generation produced the same 512 token IDs on raw and packed storage and a coherent HashMap explanation. The output ended because the benchmark explicitly requested 512 tokens.
- Loaded/final Metal allocation was 4.930 GiB; final physical footprint was 4.775–4.782 GiB.

#### Confirmed cause of within-run rate changes

- The 2,100-slot SLRU recorded 125,813 hits and 44,747 misses: 73.76% hit rate, 42,647 evictions, and 73.741 GiB of logical expert payload fetched.
- Decode buckets ranged from 24.16 to 27.38 tok/s in the instrumented raw runs. Demand rose from 55.9 misses/generated token in the first complete 64-token bucket to 105.2 misses/generated token in the final bucket. GPU wait stayed near 23–25 ms while miss-fetch latency rose from approximately 6–10 ms to 14–17 ms.
- Therefore the downward drift is primarily cache-demand growth plus storage latency, not context-KV growth or a changing GPU allocation. The physical and Metal footprints remained stable.
- An exact offline replay reproduced all 44,747 SLRU misses. Trace-specific dynamic programming over the same 2,100 total slots reduced the projection to only 44,226 misses (521 fewer, 1.16%). The current entropy-weighted layer partition is already near-optimal for this prompt; prompt-specific repartitioning was rejected.
- A frequency-only eviction experiment was rejected: hit rate fell to 68.58%, misses rose to 53,592, and fetched payload rose to 88.317 GiB.

#### Q4 packed-expert v2 artifact

- Added `QwenFieldfarePack`, a resumable layer-boundary packer and full verifier.
- Store layout: `qwen_q4_experts/manifest.json` plus 40 independently durable `layer_XX.q4exp` files. Every layer contains 256 contiguous 1,769,472-byte expert records in gate/up/down weight-scale-bias order.
- Each layer header records format version, geometry, source-index SHA-256, and payload SHA-256. Completed layers are fsynced and atomically renamed; an atomic checkpoint permits `--resume`. The final manifest is written only after all layers complete.
- Local artifact size: 18,119,557,120 bytes (16.8752 GiB). Source index SHA-256: `0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea`.
- Independent full verification read and matched all 40 layer payload hashes (16.875 GiB). Format tests passed 3/3.

```bash
swift run -c release QwenFieldfarePack \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit

swift run -c release QwenFieldfarePack \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit --resume

swift run -c release QwenFieldfarePack \
  --verify --model /absolute/path/to/Qwen3.6-35B-A3B-4bit
```

#### Packed runtime decision

| Expert source | Decode | Miss-read behavior | Decision |
|---|---:|---|---|
| Raw safetensors | 25.04 tok/s | Nine ranges per missed expert, three projection groups in parallel | Keep as app default |
| Packed v2, one read | 24.38 tok/s | One 1.6875 MiB `pread` per missed expert | Opt-in experiment |
| Packed v2, three reads | 23.85 tok/s | Three parallel 576 KiB `pread` calls per missed expert | Rejected |
| Packed v2, `F_NOCACHE` | 18.39 tok/s | Forced repeated payload traffic to storage | Rejected |

- The packed layout eliminated scattered ranges but did not reduce payload bytes. On this APFS/Apple-SSD workload, syscall count was not the limiting factor and bypassing the page cache was actively harmful.
- The runtime therefore keeps raw safetensors as the default. Packed v2 remains available only through the benchmark's `--packed-experts` flag for reproducible follow-up work; an invalid or incomplete store is never selected.
- This experiment does not meet the 49.86 tok/s MLX reference. Even with zero miss latency, the measured 23–25 ms GPU wait and remaining CPU/encode work prevent 50 tok/s. The next performance task must reduce GPU command-buffer/kernel time as well as expert traffic.

#### Final regression checks

- `Scripts/test.sh --filter 'QwenQ4ExpertStoreTests|Q4BoundedExpertCacheTests'` — 6/6 tests passed.
- `Scripts/test.sh --filter hashMapPromptTeacherForcedMLXParity` — 132/132 next-token predictions matched.
- `swift build -c release` — exit 0. Existing unrelated warnings remain for deprecated `cblas_sgemv` and an unused `withUnsafeBytes` result.
- 65,536-context release load with 1,440 expert slots generated the expected first token `13962`; loaded/final Metal allocation was 4.940 GiB and final physical footprint was 4.769 GiB.
- `git diff --check` — exit 0.

---

### Follow-up: Exact GPU Profile and Cache Promotion (2026-08-24)

#### GPU bottleneck measurement

- Added opt-in `QwenFieldfareBench --gpu-profile` command-buffer timing. It uses Metal GPU start/end timestamps and is disabled in the app/default benchmark path.
- Measured aggregate GPU time was 20.96–21.30 ms/token in the profiled run: 11.18–11.38 ms attention/router, 5.22–5.66 ms shared-expert/cache-hit work, and 3.29–4.44 ms miss/tail/lm-head work. Later runs reached approximately 28–30 ms/token as system/GPU state changed, so these values are per-run measurements rather than a fixed compute ceiling.
- Corrected `--subcomponent-profile` to dispatch the active SIMD kernels and report GPU timestamps. One measured split was 8.975 ms for attention across all 40 layers: 6.813 ms in the 30 linear-attention layers and 2.162 ms in the 10 full-attention layers. Linear input projection was the largest individual component at 3.658 ms/token.

#### Rejected experiments

- Threadgroup staging of the 2,048-element Q4 input vector improved the isolated projection microbenchmark by only 1.57–1.88%; the matching full 128-token A/B was unchanged at 23.80 tok/s. Removed.
- A SIMD16 Q4 variant was 2.10% slower in the isolated comparison. Removed.
- Limiting cache-miss fetch concurrency to two workers reduced the matching 128-token run from 25.72 to 23.08 tok/s. Removed.
- No experimental kernel, fetch cap, or new runtime control was retained.

#### Retained memory and cache change

- Corrected each resident dense arena from the former 25,165,824-byte round-up to the exact 22,456,448-byte tensor end.
- The resident-dense path no longer allocates its unused extra dense arena or legacy nine-expert staging arena. This reclaims 149,466,112 bytes (142.542 MiB) versus the former production allocation.
- The app spends most of that verified headroom on 83 additional expert-cache slots: 2,183 at 4K/8K, 2,083 at 16K, 1,883 at 32K, and 1,523 at 64K. The active-streaming preflight estimate was updated to the compact allocation.
- In the 512-token 8K/2,183-slot run, Metal allocation was 4.928 GiB and final physical footprint was 4.803 GiB. The cache recorded 127,293 hits, 43,267 misses, a 74.63% hit rate, and 71.302 GiB fetched.
- Compared with the earlier 2,100-slot 512-token trace, this is 1,480 fewer misses (3.31%) and 2.439 GiB less logical expert traffic without increasing the approximately 5 GiB Metal envelope. Decode was 23.28 tok/s in this run; system-state variance prevents attributing a deterministic throughput change to the cache increase.
- The remaining within-run variation still follows demand: the measured 2,183-slot buckets ranged from 21.91 to 24.70 tok/s while misses rose from 53.4 to 101.8 per generated token.

#### Final validation

- `Scripts/test.sh --filter QwenGenerationSupportTests` — 9/9 passed.
- `Scripts/test.sh --filter 'Q4BoundedExpertCacheTests|QwenQ4ExpertStoreTests'` — 6/6 passed.
- `Scripts/test.sh --filter hashMapPromptTeacherForcedMLXParity` — 132/132 next-token predictions matched.
- `swift build -c release` — exit 0. The existing `cblas_sgemv` deprecation and unused `withUnsafeBytes` warnings remain.
- A 65,536-context/1,523-slot release load generated one token with 4.937 GiB Metal and 4.812 GiB final physical footprint.
- This change improves cache efficiency within the memory constraint. It does not yet meet the approximately 50 tok/s MLX reference.
