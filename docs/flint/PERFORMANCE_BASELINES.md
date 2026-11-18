# FlintFieldfare Performance Baselines & Measurement Protocols

## Test Machine Specifications

- **Model**: Apple MacBook Pro M5
- **Unified Memory**: 32 GiB (Bandwidth: Apple M5 family)
- **Internal Storage**: Apple NVMe SSD (114 GiB free capacity)
- **macOS Version**: macOS 27.2.1 (Darwin 27.2.0, Build 26B131)
- **Swift Version**: Apple Swift 6.4.2 (clang-2100.4.1.1)
- **Metal Version**: Metal 4.0 runtime compiler

## Measurement Protocol

Follow the methodology in `docs/COMMUNITY_BENCHMARKS.md`:
1. Build in release mode: `swift build -c release`.
2. Ensure normal memory pressure and no other model-owning processes.
3. Cold run (cache empty): Record time to first token (TTFT), disk read traffic, and initial decode rate.
4. Warm run (repeated prompt or sustained generation): Record warm prefill speed, decode tokens/s, and cache hit percentage.
5. Report memory footprint: Peak resident set size (`vm_stat`, `libproc`), Metal allocated buffer memory.

## Reference Targets (MLX Comparison)

Reference numbers reported by community on M5 Max / M5:
- `mlx-vlm` 0.7.10 decode speed: 76–119 tokens/s (all-resident in unified memory, 17.8 GB).
- FlintFieldfare Target: Maintain interactive streaming decode speed (>15–30 tokens/s) within a strictly bounded **3.2 GiB** memory budget, without requiring 18–24 GB unified memory residency.
