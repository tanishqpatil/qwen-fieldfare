# FlintFieldfare Performance Baselines & Measurement Protocols

## Test Machine Specifications

- **Model**: Apple MacBook Pro M5
- **Unified Memory**: 32 GiB (Bandwidth: Apple M5 family)
- **Internal Storage**: Apple NVMe SSD (~114 GiB free capacity)
- **macOS Version**: macOS 27.2.1 (Darwin 27.2.0, Build 26B131)
- **Swift Version**: Apple Swift 6.4.2 (clang-2100.4.1.1)
- **Metal Version**: Metal 4.0 runtime compiler
- **Python Environment**: `/Users/tanishq/venvs/mlx-dspark/bin/python`
  - `mlx` version: `0.32.4`
  - `mlx_vlm` version: `0.7.12`
  - `transformers` version: `5.22.2`

---

## 1. Verified All-Resident Reference Baseline (Measured on M5)

The following empirical measurements were captured on 2026-11-18 using `mlx_vlm` 0.7.12 with `/Users/tanishq/models/flint-mixed-3-4bit` (all-resident in unified memory, temperature 0.0 deterministic greedy):

### 1.1 Model Loading

- **Cold Load Time**: **4.39 seconds**
- **Active Memory (Weights)**: **16.61 GB**
- **Peak Load Memory**: **16.61 GB**
- **Process Max RSS**: **6.54 GB**
- **System Swap Impact**: Increased from 767.56 MB to 1055.06 MB (+287.5 MB immediately upon load).

### 1.2 Scenario A: Short Coding Prompt (13 tokens, max_tokens 64)

*Prompt*: `"Write a Swift function that implements a thread-safe bounded LRU cache."*

| Metric | Run 1 (Cold) | Run 2 (Warm) | Run 3 (Warm) | Average (Warm) |
|---|---|---|---|---|
| **Prompt Throughput** | 13.76 tok/s | 161.91 tok/s | 161.51 tok/s | **161.71 tok/s** |
| **Decode Throughput** | 70.39 tok/s | 69.91 tok/s | 70.02 tok/s | **69.97 tok/s** |
| **Total Latency** | 3.014 s | 1.001 s | 0.999 s | **1.000 s** |
| **Peak Memory** | 16.68 GB | 16.68 GB | 16.68 GB | **16.68 GB** |
| **Process RSS** | 6.54 GB | 6.54 GB | 6.54 GB | **6.54 GB** |
| **System Swap** | 1,425.19 MB | 1,417.19 MB | 1,417.19 MB | 1,417.19 MB |

### 1.3 Scenario B: Long Context (~8K Prompt: 7,831 tokens, max_tokens 32)

*Prompt*: Realistic Swift code analysis prompt calibrated to 7,831 tokens.

| Metric | Run 1 (Cold KV) | Run 2 (Warm) | Run 3 (Warm) | Average (Warm) |
|---|---|---|---|---|
| **Prompt Throughput** | 1,228.81 tok/s | 1,319.89 tok/s | 1,313.99 tok/s | **1,316.94 tok/s** |
| **Prefill Latency** | ~6.37 s | ~5.93 s | ~5.96 s | **~5.95 s** |
| **Decode Throughput** | 60.31 tok/s | 60.63 tok/s | 59.84 tok/s | **60.24 tok/s** |
| **Total Latency** | 6.914 s | 6.480 s | 6.507 s | **6.494 s** |
| **Peak Memory** | 17.68 GB | 17.68 GB | 17.68 GB | **17.68 GB** |
| **Process RSS** | 6.54 GB | 6.54 GB | 6.54 GB | **6.54 GB** |
| **System Swap** | 1,401.19 MB | 1,401.19 MB | 1,361.19 MB | 1,381.19 MB |

---

## 2. Performance Analysis & Target Comparison

1. **All-Resident Strengths**: Excellent prefill throughput on long context (~1,317 tok/s) due to chunked SDPA, and steady decode throughput (~60–70 tok/s).
2. **All-Resident Cost**: Peak memory reaches **17.68 GB**, triggering **>600 MB** of system swap activity on a 32 GB machine and severely limiting memory headroom for concurrent developer tasks.
3. **FlintFieldfare Streaming Target**:
   - Primary metric: Bounded memory footprint of **~3.2 GiB** (saving >14 GB unified memory).
   - Secondary metric: Interactive decode throughput (>15–30 tok/s) once the expert cache achieves $\ge 70\%$ hit rate.
