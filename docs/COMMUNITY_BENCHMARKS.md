# Community benchmark guide

Use this procedure for comparable QwenFieldfare measurements. Report results
as measurements from one machine and commit, not as hardware ceilings.

## Before running

1. Close Ollama, LM Studio, MLX, and every other local model process.
2. Confirm normal memory pressure with `memory_pressure -Q`.
3. Record `git rev-parse HEAD`, `system_profiler SPHardwareDataType`,
   `sw_vers`, and `swift --version`.
4. Build once with `swift build -c release`.
5. Use an absolute model path and do not publish personal paths or weights.

## Decode benchmark

```bash
.build/release/QwenFieldfareBench \
  --model "/absolute/path/to/Qwen3.6-35B-A3B-4bit" \
  --prompt "Explain HashMap in Java with a short example." \
  --context 8192 \
  --cache-slots 2183 \
  --max-new 512 \
  --metrics-only
```

Run the same command at least three times. Report every run; do not report only
the fastest one. Keep `--gpu-profile`, `--subcomponent-profile`, and
`--analyze-cache` disabled for headline throughput because instrumentation can
change timing.

## Required report fields

- Commit SHA
- Mac model, Apple chip, GPU-core count, and unified-memory capacity
- macOS and Swift versions
- Exact command and sanitized prompt
- Prompt-token and generated-token counts
- Prefill duration and time to first token
- Decode tokens per second
- Context length and expert-cache slots
- Cache hits, misses, hit rate, and logical bytes fetched
- Metal allocation and memory pressure
- Whether the packed store was enabled
- All deviations from this procedure

Use the benchmark issue template when submitting results.
