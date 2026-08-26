# QwenFieldfare

Native Swift and Metal inference for the MLX Q4 build of Qwen 3.6 35B-A3B on
Apple Silicon.

## Scope

This repository is Qwen 3.6-only. Do not add unrelated model formats, generic
OpenAI-server behavior, or a second model-owning process without an explicit
design change. Preserve the bounded-memory active-expert path.

## Products

```bash
swift build -c release
.build/release/QwenFieldfareMac

swift run -c release QwenFieldfareBench \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --prompt "Explain HashMap in Java with an example." \
  --max-new 128

swift run -c release QwenFieldfarePack \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit
```

`QwenFieldfareCore` owns model validation, Swift/Metal inference, telemetry,
the bounded expert cache, and the optional packed expert store.

## Model safety

The app reads a compatible local MLX Safetensors directory selected in the UI
or through `QWENFIELDFARE_MODEL`. Never stage, duplicate, upload, or commit
model weights. Treat tensor names, shapes, quantization, and chat formatting as
model-specific invariants.

Before a real-model run, require macOS 26+, Swift 6.2+, adequate disk and
normal memory pressure, and no process matching
`QwenFieldfareMac|QwenFieldfareBench|mlx_lm|mlx-lm|ollama|LM Studio`.
Run only one model-owning process at a time.

## Validation

Run package tests through `Scripts/test.sh`. Build release once before a model
measurement and follow `docs/COMMUNITY_BENCHMARKS.md` exactly. Profiling flags
are diagnostic and must not be used for headline throughput.

Report the commit, hardware and RAM, macOS, Swift version, exact command,
prompt/generated token counts, complete timing/cache footer, memory, and every
protocol deviation. Measurements are not performance ceilings.
