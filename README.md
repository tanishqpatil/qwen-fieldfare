<p align="center">
  <img src="Sources/QwenFieldfareApp/Mac/Resources/qwenfieldfare-app-icon.png"
       width="128" alt="QwenFieldfare app icon">
</p>

<h1 align="center">QwenFieldfare</h1>

**Local Qwen 3.6 35B-A3B Q4 mixture-of-experts inference near 5 GiB of Metal
allocation.**

QwenFieldfare is a native Swift + Metal runtime and macOS chat app for Apple
Silicon. It reads MLX Q4 Safetensors and streams only the routed MoE experts
selected for each token from SSD into a bounded cache instead of keeping the
complete checkpoint resident.

[Quick start](#quick-start) · [Benchmarks](#measured-results) ·
[How it works](#how-active-expert-streaming-works) ·
[Community benchmarks](docs/COMMUNITY_BENCHMARKS.md) ·
[Contributing](CONTRIBUTING.md)

[![Model: Qwen 3.6 35B-A3B](https://img.shields.io/badge/Model-Qwen%203.6%2035B--A3B-6f42c1)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white)](https://www.swift.org/)
[![Metal 4](https://img.shields.io/badge/Metal-4-5E5CE6)](https://developer.apple.com/metal/)
[![macOS 26+](https://img.shields.io/badge/macOS-26%2B-000000?logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![CI](https://github.com/tanishqpatil/qwen-fieldfare/actions/workflows/ci.yml/badge.svg)](https://github.com/tanishqpatil/qwen-fieldfare/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2ea44f)](LICENSE)

> [!IMPORTANT]
> This is an experimental, model-specific research runtime. Correctness is
> historically validated against MLX, but the automated parity harness must be
> restored. Measured decode performance does not yet match the full-resident
> MLX reference.

## What this project does

Qwen 3.6 35B-A3B is a mixture-of-experts model with 35 billion total
parameters and approximately 3 billion activated parameters per token. Loading
the complete Q4 checkpoint normally consumes roughly 20 GiB of unified memory.

QwenFieldfare keeps dense Q4 weights and recurrent/KV state resident,
while routed MoE experts are fetched into a bounded Metal-visible cache only
when selected by the router. The default 8K configuration uses 2,183 expert
slots and measured 4.928 GiB of Metal allocation during a 512-token run.

The runtime is written directly in Swift and Metal. It is not a wrapper around
MLX, llama.cpp, Ollama, or another inference engine.

## Status at a glance

| Capability | Current status |
| --- | --- |
| Model | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Architecture | 40 layers; 30 Gated DeltaNet + 10 full-attention layers |
| MoE routing | 256 routed experts; top 8 selected per layer/token + 1 shared expert |
| Quantization | MLX affine Q4, group size 64; Q8 routers/shared gates |
| Chat | Multi-turn, thinking on/off, checkpoint-compatible template |
| Sampling | Temperature, Top-K, Top-P, presence and repetition penalties |
| Context | 4K, 8K, 16K, 32K, or 64K in Active Expert Streaming mode |
| Memory | Approximately 5 GiB Metal allocation in validated active-streaming configurations |
| Correctness | Historical validation matched 132/132 teacher-forced next-token predictions; the automated harness must be restored |
| Validated hardware | Apple M5, 32 GiB unified memory |
| Current limitation | Text-only; the checkpoint's vision encoder is not implemented |

## Requirements

- Apple Silicon Mac
- macOS 26 or newer
- Xcode 26 and Swift 6.2 or newer
- Approximately 20 GiB free storage for the raw Q4 model
- Approximately 37 GiB total model storage if the optional packed-expert
  artifact is also created
- Enough currently available unified memory for the selected runtime mode

The validated machine has 32 GiB unified memory. Lower-memory Macs have not yet
been validated for this Qwen port and should not be advertised as supported
until measured.

## Quick start

### 1. Download the model

This runtime expects the MLX Q4 checkpoint as an ordinary local safetensors
directory. Install the current Hugging Face CLI if necessary:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Download the model to the default location:

```bash
mkdir -p "$HOME/models"

hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir "$HOME/models/Qwen3.6-35B-A3B-4bit"
```

The expected directory is:

```text
~/models/Qwen3.6-35B-A3B-4bit/
├── config.json
├── generation_config.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

Model weights are not stored in this source repository. They remain governed
by the model repository's license and terms.

### 2. Build the app

Clone the repository and build all release products:

```bash
git clone https://github.com/tanishqpatil/qwen-fieldfare.git
cd qwen-fieldfare
swift build -c release
```

### 3. Run the app

```bash
.build/release/QwenFieldfareMac
```

The app loads the model lazily when the first prompt is sent. Open the
diagnostics drawer to inspect context, cache occupancy, memory, prompt prefill,
token latency, and generation rate.

## Choose the model directory

Use **Model → Choose Model Directory…** (`⇧⌘O`) or open the inspector and use
**Model Management → Choose Model Directory…**. QwenFieldfare validates the
model index and every referenced Safetensors shard before saving the path for
future launches.

For scripts and launch agents, set `QWENFIELDFARE_MODEL`:

```bash
QWENFIELDFARE_MODEL="/absolute/path/to/Qwen3.6-35B-A3B-4bit" \
  .build/release/QwenFieldfareMac
```

The model must use the exact compatible Qwen 3.6 35B-A3B MLX Q4 tensor names,
shapes, affine quantization layout, tokenizer, and chat configuration. This is
not a generic Safetensors runtime.

## Mac app guide

### Runtime modes

| Mode | Intended use | Memory behavior |
| --- | --- | --- |
| **Active Expert Streaming** | Recommended chat mode | Dense weights stay resident; routed experts use a bounded cache under the approximately 5 GiB Metal budget |
| **Fast Resident Q4** | High-memory diagnostic mode | Loads the full Q4 model into unified memory |
| **Memory Saver Q4** | Low-memory fallback | Streams layer weights and experts with lower throughput |

Only one model-owning process should run at a time. Do not run the Mac app,
Qwen benchmark, MLX, Ollama, or another local model process concurrently when
collecting memory or performance results.

### Active-streaming cache by context

| Context | Expert-cache slots | Validated budget behavior |
| ---: | ---: | --- |
| 4K / 8K | 2,183 | 8K measured 4.928 GiB Metal |
| 16K | 2,083 | Preflight estimate below 5 GiB |
| 32K | 1,883 | Preflight estimate below 5 GiB |
| 64K | 1,523 | Measured 4.937 GiB Metal |

The response limit cannot exceed the selected context. Prompt tokens and prior
conversation turns consume part of the same context window.

### Default generation settings

| Setting | Default |
| --- | ---: |
| Thinking | Off |
| Context | 8,192 tokens |
| Maximum output | 4,096 tokens |
| Temperature | 1.0 |
| Top-K | 20 |
| Top-P | 0.95 |
| Presence penalty | 1.5 |
| Repetition penalty | 1.0 |
| Idle unload | 60 seconds |

Press **Return** to send, **Shift+Return** for a newline, or use the stop button
to cancel generation. **New Chat** clears retained conversation context without
changing the loaded runtime.

## Benchmark command line

`QwenFieldfareBench` is the supported command-line Qwen entry point. It
uses greedy generation so repeated runs can be compared exactly.

```bash
export QWEN_MODEL="$HOME/models/Qwen3.6-35B-A3B-4bit"

swift run -c release QwenFieldfareBench \
  --model "$QWEN_MODEL" \
  --prompt "Explain HashMap in Java with an example." \
  --context 8192 \
  --cache-slots 2183 \
  --max-new 128
```

Useful benchmark options:

| Option | Purpose |
| --- | --- |
| `--metrics-only` | Suppress generated text and print timing/cache metrics |
| `--load-only` | Validate model loading without generation |
| `--warmup-runs N` | Run `N` unmeasured generations while retaining the loaded expert cache |
| `--runs N` | Run and summarize `N` measured generations in one process |
| `--analyze-cache` | Record and analyze the routed-expert access trace |
| `--gpu-profile` | Add exact Metal command-buffer timing; profiling only |
| `--subcomponent-profile` | Profile production attention subcomponents |
| `--packed-experts` | Use the optional packed routed-expert store |

Profiling changes instrumentation and should not be used for headline
throughput comparisons.

## Model format and packed experts

The production runtime reads the compatible MLX Safetensors checkpoint
directly.

The repository includes an optional Qwen-specific packed-expert experiment.
It creates a checksummed `qwen_q4_experts/` directory beside the raw model:

```bash
export QWEN_MODEL="$HOME/models/Qwen3.6-35B-A3B-4bit"

swift run -c release QwenFieldfarePack \
  --model "$QWEN_MODEL"
```

Resume an interrupted pack:

```bash
swift run -c release QwenFieldfarePack \
  --model "$QWEN_MODEL" \
  --resume
```

Verify all packed layer hashes:

```bash
swift run -c release QwenFieldfarePack \
  --verify \
  --model "$QWEN_MODEL"
```

Use a separate output directory when desired:

```bash
swift run -c release QwenFieldfarePack \
  --model "$QWEN_MODEL" \
  --output "/absolute/path/to/qwen_q4_experts"
```

Custom output directories can be packed and verified, but the benchmark
currently auto-discovers only the default `qwen_q4_experts/` directory beside
the model.

The default packed store is approximately 16.875 GiB and contains 40 durable
`layer_XX.q4exp` files plus `manifest.json`. Packing is resumable at layer
boundaries and verification checks every payload hash.

This format remains experimental. On the validated workload, raw safetensors
measured 25.04 tok/s and packed one-read storage measured 24.38 tok/s. The app
therefore continues to use raw safetensors. Use `--packed-experts` only with
`QwenFieldfareBench` for storage experiments.

## Measured results

Machine: Apple M5 (`Mac17,2`), 32 GiB unified memory, macOS 26.6.2, Swift
6.3.3. Results are workload-specific measurements, not hardware ceilings.

| Measurement | Result |
| --- | ---: |
| Historical MLX teacher-forced parity | 132/132 next-token predictions |
| Historical active streaming, 8K, 2,183 slots, 512 tokens | 23.28 tok/s |
| Corrected inter-token control, same 512-token workload | 19.68 tok/s; 50.80 ms/token |
| Exact-greedy fast path, five paired 128-token runs | 29.02 to 31.97 tok/s; +10.16% paired mean |
| Exact-greedy fast path, three paired 512-token runs | +6.22%, +8.85%, and +5.01% |
| Metal allocation in that 512-token run | 4.928 GiB |
| Cache hit rate | 74.63% |
| Cache misses | 43,267 |
| Logical expert data fetched | 71.302 GiB |
| 64K context load, 1,523 slots | 4.937 GiB Metal |
| Full-resident MLX reference | 49.86 tok/s, 19.631 GB peak memory |

Compared with the previous 2,100-slot trace, the promoted 2,183-slot cache
reduced misses by 1,480 (3.31%) and logical expert traffic by 2.439 GiB. It did
not establish a deterministic throughput increase because file-page-cache,
GPU/DVFS, and system state varied between runs.

See the [active-streaming report](docs/QWEN_ACTIVE_STREAMING.md)
for the measured evidence, rejected experiments, and limitations.
See the [optimization analysis](docs/OPTIMIZATION_PLAN.md) for the ranked next
work, measurement protocol, promotion gates, and experiments not to repeat.
See the [product roadmap](docs/PRODUCT_ROADMAP.md) for reliability,
distribution, and feature priorities.

## How active-expert streaming works

For each transformer layer and generated token:

1. Metal executes normalization and the layer's Gated DeltaNet or full-attention path.
2. The Q8 router selects eight of 256 routed experts.
3. The bounded segmented-LRU cache resolves resident experts and assigns slots for misses.
4. Missing Q4 expert records are read from the model's memory-mapped safetensors shards into Metal-visible shared buffers.
5. Shared-expert and cache-hit GPU work overlaps part of the miss I/O.
6. Metal evaluates routed experts and combines shared, routed, and residual outputs.
7. Final RMSNorm, Q4 language-model head, and sampling produce the next token.

Dense weights, recurrent DeltaNet state, FP16 full-attention KV state, and the
expert cache have bounded ownership. The complete routed-expert bank is never
allocated as Metal buffers in Active Expert Streaming mode.

## Build and validation

Build all release products:

```bash
swift build -c release
```

Run focused non-model support and storage tests:

```bash
Scripts/test.sh --filter QwenGenerationSupportTests
Scripts/test.sh --filter 'Q4BoundedExpertCacheTests|QwenQ4ExpertStoreTests'
```

Generate a standalone MLX greedy reference only after installing the model and
closing every other model process:

```bash
python3 Scripts/qwen_mlx_greedy_reference.py \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --prompt "Explain HashMap in Java with an example." \
  --max-tokens 65
```

The historical engineering run matched 132/132 teacher-forced positions, but
the automated parity test that produced that result is not present in the
current checkout. Restoring a callable real-model parity harness is a release
prerequisite; the reference command above is not a substitute for that gate.

Recent validation:

- Qwen generation support: 10/10 passed
- Expert cache/store: 6/6 passed
- Complete package suite: 43/43 passed
- Historical MLX teacher-forced parity record: 132/132; current automated
  harness not checked in
- Exact-greedy fast path: all 128 generated token IDs matched the frozen
  control; app-default sampling is unchanged
- Complete release build: passed
- `git diff --check`: passed

## Repository map

| Path | Purpose |
| --- | --- |
| `Sources/QwenFieldfareCore/Runtime/Generation/` | Qwen chat, sampling, and active-expert generation |
| `Sources/QwenFieldfareCore/Kernels/LinearAttention/` | Gated DeltaNet and Qwen dense/MoE kernel wrappers |
| `Sources/QwenFieldfareCore/Metal/LinearAttention/` | Qwen Metal kernels |
| `Sources/QwenFieldfareCore/Storage/` | Bounded expert cache and optional packed store |
| `Sources/QwenFieldfareApp/Mac/` | Native SwiftUI chat app and diagnostics |
| `Sources/QwenFieldfareBench/` | Reproducible Qwen benchmark executable |
| `Sources/QwenFieldfarePack/` | Resumable packed-expert builder and verifier |
| `Tests/QwenFieldfareCore/` | Qwen correctness, cache, and storage validation |
| `docs/` | Qwen active-streaming measurements and engineering notes |

## Known limitations

- Decode speed remains below the full-resident MLX reference.
- The runtime is text-only; vision, audio, video, tool execution, and MTP
  speculative decoding are not implemented.
- Qwen is not currently exposed through an OpenAI-compatible server.
- The app does not download the model. Install the model with `hf download`
  before first use.
- True cold-storage state, thermal state, and macOS file-page-cache state can
  materially change performance.
- The optional packed-expert format increases disk use and has not improved
  throughput on the validated machine.
- Generated output can be incorrect. Verify important answers.

## Attribution

This project is based on and derived from
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by
[Andrey Mikhaylov](https://github.com/drumih). The original project introduced
the bounded-memory active-expert streaming design on Apple Silicon.
This derivative ports and extends that work for Qwen 3.6 35B-A3B, including
Qwen-specific Gated DeltaNet/full-attention kernels, model loading, caching,
chat behavior, benchmarks, and UI.

The original TurboFieldfare source and this derivative are distributed under
the [Apache License 2.0](LICENSE). Preserve the license and applicable
attribution notices when redistributing source or binaries. See
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

QwenFieldfare is an independent community research project. It is not
affiliated with, sponsored by, or endorsed by the Qwen team, Alibaba, MLX,
Hugging Face, Apple, or the original TurboFieldfare author.

## Model license and redistribution

Model weights are separate from this source repository:

- Base model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Q4 MLX checkpoint: [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)

Both model repositories identify their artifacts as Apache-2.0. If you publish
a separate copy or modification of the model, include the applicable license,
model-card/source attribution, and a clear description of your changes. The
upstream model repositories and their current files remain authoritative.

## License

Source and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE).
