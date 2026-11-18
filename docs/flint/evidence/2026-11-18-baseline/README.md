# MLX All-Resident Empirical Baseline: Recursive Flint 40B-A2.5B

**Date**: 2026-11-18  
**Platform**: Apple MacBook Pro M5 (32 GB unified memory)  
**OS**: macOS 27.2.1 (Darwin Kernel Version 27.2.0, Build 26B131, arm64)  
**Python Environment**: `/Users/tanishq/venvs/mlx-dspark/bin/python`  
**MLX Version**: `0.32.4`  
**MLX-VLM Version**: `0.7.12`  
**Checkpoint Path**: `/Users/tanishq/models/flint-mixed-3-4bit` (17.84 GB on disk, 4 Safetensors shards)  
**Invocation**: `python benchmark_flint_mlx.py`  

---

## 1. Objective and Setup

To establish an authoritative, reproducible empirical baseline for Recursive Flint 40B-A2.5B on Apple Silicon M5 (32 GB) prior to developing native Metal streaming kernels. This baseline characterizes cold-load latency, all-resident memory occupancy, short-prompt prefill/decode throughput, and ~8K long-context prefill/decode throughput.

### Scripts and Artifacts Preserved
- Benchmark driver: `benchmark_flint_mlx.py`
- Structured raw metrics: `mlx_flint_benchmark_results.json`

---

## 2. Methodology & Instrumentation

1. **Model Loading & Synchronization**:
   - Model weights and architecture loaded via `mlx_vlm.load(model_path)`.
   - Explicit parameter evaluation via `mx.eval(model.parameters())` ensures cold load latency includes all weight mapping and allocation.
2. **Deterministic Generation**:
   - `temperature = 0.0` (greedy deterministic decoding).
   - Short prompt: 13 prompt tokens, 64 generated tokens (`max_tokens=64`).
   - Long prompt: 7,831 prompt tokens, 32 generated tokens (`max_tokens=32`).
3. **Timing Boundaries**:
   - Python `time.perf_counter()` immediately before and after `generate()`.
   - `prompt_tps` and `generation_tps` extracted from generator telemetry.
4. **Memory Telemetry**:
   - MLX active memory: `mx.get_active_memory() / (1024**3)`
   - MLX peak allocation: `mx.get_peak_memory() / (1024**3)` (reset via `mx.reset_peak_memory()` before each run)
   - Process resident memory: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)`
   - System swap: `sysctl vm.swapusage` sampled before load, after load, and after each benchmark scenario.

---

## 3. Measured Results Summary

### Scenario 1: Model Cold Load
- **Load Latency**: `4.39 s`
- **MLX Active Memory**: `16.61 GB`
- **MLX Peak Memory**: `16.61 GB`
- **Process RSS**: `6.54 GB`
- **Initial Swap**: `used = 767.56M` (total 2048M, free 1280.44M)

### Scenario 2: Short Prompt (13 Tokens Prompt, 64 Tokens Generated)

| Run | Prompt Throughput | Decode Throughput | Total Time | MLX Active Mem | MLX Peak Mem | System Swap |
|:---|:---|:---|:---|:---|:---|:---|
| Run 1 (cold) | 13.76 tok/s | 70.39 tok/s | 3.014 s | 16.61 GB | 16.68 GB | 1425.19M |
| Run 2 (warm) | 161.91 tok/s | 69.91 tok/s | 1.001 s | 16.61 GB | 16.68 GB | 1417.19M |
| Run 3 (warm) | 161.51 tok/s | 70.02 tok/s | 0.999 s | 16.61 GB | 16.68 GB | 1417.19M |

### Scenario 3: Long Context (~8K Prompt Tokens, 32 Tokens Generated)

Prompt token count: `7,831 tokens` (Swift LRU & matrix multiplication source code repeated).

| Run | Prompt Latency | Prompt Throughput | Decode Throughput | Total Time | MLX Active Mem | MLX Peak Mem | System Swap |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Run 1 | 6.37 s | 1,228.81 tok/s | 60.31 tok/s | 6.914 s | 16.61 GB | 17.68 GB | 1401.19M |
| Run 2 | 5.93 s | 1,319.89 tok/s | 60.63 tok/s | 6.480 s | 16.61 GB | 17.68 GB | 1401.19M |
| Run 3 | 5.96 s | 1,313.99 tok/s | 59.84 tok/s | 6.507 s | 16.61 GB | 17.68 GB | 1361.19M |

---

## 4. Engineering Observations & Analysis

### 4.1 System Swap: Correlation vs. Causality
During the benchmark run, `vm.swapusage` increased from an initial `767.56 MB` to `1,425.19 MB` (+657.6 MB) before stabilizing between `1,361 MB` and `1,417 MB`.
- **System Memory Context**: On a 32 GB unified memory machine running macOS with concurrent desktop applications (Xcode, Antigravity, browsers), allocating ~17.68 GB of unified memory for MLX pushes memory pressure into elevated zones, causing macOS kernel `memorystatus` and VM pager to page out inactive background pages.
- **Rigor Clarification**: This swap increase is an **observed system-level memory response to large allocations**, not conclusive proof of active memory thrashing during decode. Decode throughput remained consistent at ~70 tok/s (short) and ~60 tok/s (8k context), indicating that active weights and KV cache remained resident in physical memory during these runs.

### 4.2 Architectural Trade-off: All-Resident vs. Bounded Streaming
The empirical data frames the core design trade-off for FlintFieldfare:
1. **All-Resident Mode (Peak Throughput)**:
   - When 18+ GB of RAM is available and dedicated to inference, all-resident execution achieves ~70 tok/s decode and ~1,317 tok/s prefill on M5.
2. **Bounded Expert Streaming (Multitasking & Small Working Set)**:
   - When the user requires substantial unified memory for IDEs, compilation, graphics, or multiple agents, all-resident 17.7 GB allocation creates severe memory pressure.
   - Bounded streaming aims to constrain the inference Metal footprint to an architectural budget of **~3.2 GiB** (dense weights + KV cache + bounded expert cache), eliminating system memory pressure at the cost of SSD I/O.
   - **Hypothesis**: The viability of streaming depends on expert cache hit rates (hypothesized 70–85% on structured coding workloads) keeping required SSD bandwidth within M5 internal SSD capability (3.0–3.5 GB/s sequential read).
