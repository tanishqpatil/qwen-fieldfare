<p align="center">

  <img src="Sources/QwenFieldfareApp/Mac/Resources/qwenfieldfare-app-icon.png"
       width="128" alt="QwenFieldfare app icon">
</p>

<h1 align="center">QwenFieldfare</h1>

Native Swift + Metal inference runtime with bounded expert streaming for
**Recursive Flint 40B-A2.5B** on Apple Silicon.

> [!IMPORTANT]
> This is an experimental, model-specific research runtime. Active development
> targets milestone F1 (Flint metadata & weight-loading infrastructure). The
> runtime is not yet production-qualified for inference.

**Bounded expert streaming: ~3.2 GiB unified memory for a 39.8B-parameter
mixture-of-experts model.**

QwenFieldfare is a native Swift + Metal runtime and macOS chat app for Apple
Silicon. It reads MLX Safetensors and streams only the routed MoE experts
selected for each token from SSD into a bounded cache instead of keeping the
complete checkpoint resident.

[Quick start](#quick-start) · [Benchmarks](#measured-results) ·
[How it works](#how-active-expert-streaming-works) ·
[Contributing](CONTRIBUTING.md)

[![Model: Recursive Flint 40B-A2.5B](https://img.shields.io/badge/Model-Flint%2040B--A2.5B-6f42c1)](https://huggingface.co/RecursiveSI/flint)
[![Swift 6.2+](https://img.shields.io/badge/Swift-6.2+-F05138?logo=swift&logoColor=white)](https://www.swift.org/)
[![Metal 4](https://img.shields.io/badge/Metal-4-5E5CE6)](https://developer.apple.com/metal/)
[![macOS 26+](https://img.shields.io/badge/macOS-26%2B-000000?logo=apple&logoColor=white)](https://apple.com/macos/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2ea44f)](LICENSE)

## What this project does

Recursive Flint 40B-A2.5B is a mixture-of-experts model with 39.8 billion total
parameters and approximately 2.5 billion activated parameters per token. Loading
the complete quantized checkpoint normally consumes 17–24 GB of unified memory.

QwenFieldfare keeps dense weights, attention projections, norms, routers, and
shared experts resident, while routed MoE experts (256 per layer, top-6
selected) are fetched into a bounded Metal-visible cache only when selected by
the router. The default 1,024-slot configuration uses ~1.31 GiB of Metal
allocation during a decoded token.

The runtime is written directly in Swift and Metal. It is not a wrapper around
MLX, llama.cpp, Ollama, or another inference engine.

## Status at a glance

| Capability | Current status |
| --- | --- |
| Model | `RecursiveSI/flint` / `mlx-community/flint-mixed-3-4bit` |
| Architecture | 49 layers; 2 dense MLP + 47 MoE (256 routed, top-6) + 1 shared |
| Quantization | INT8 embeddings, INT4 projections/shared, 3-bit routed experts |
| Attention | 14 full (no RoPE) + 35 sliding (window 513, RoPE) + learned sinks |
| Context | Configurable (planned) |
| Memory | ~3.2 GiB total working set (resident + bounded cache + KV) |
| Milestone | **F1 — Metadata & Weight-Loading** (IN PROGRESS) |
| Validated hardware | Apple M5, 32 GiB unified memory |
| Current limitation | Research runtime; inference not yet implemented |

## Requirements

- Apple Silicon Mac
- macOS 26 or newer
- Xcode 26 and Swift 6.2 or newer
- Approximately 18 GB free storage for the raw Q4 (or mixed 3/4-bit) model
- 32 GB unified memory (validated; lower-memory Macs not yet assessed)
- The model checkpoint installed locally (not shipped in this repository)

The validated machine has 32 GiB unified memory.

## Quick start

### 1. Download the model

Install the Hugging Face CLI if necessary:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Download the model (example — use whichever checkpoint you have):

```bash
mkdir -p "$HOME/models"

# Official INT4 checkpoint (22.6 GB on disk):
hf download RecursiveSI/flint-int4 \
  --local-dir "$HOME/models/flint-int4"

# Or the community 3-bit mixed checkpoint (17.84 GB):
hf download mlx-community/flint-mixed-3-4bit \
  --local-dir "$HOME/models/flint-mixed-3-4bit"
```

The expected directory contains:

```text
~/models/flint-mixed-3-4bit/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Model weights are not stored in this source repository. They remain governed
by the model repository's license and terms.

### 2. Build the project

Clone the repository and build:

```bash
git clone https://github.com/tanishqpatil/qwen-fieldfare.git
cd qwen-fieldfare
swift build -c release
```

### 3. Run

In development, build and run the FlintFieldfareMac target (when F3 completes):

```bash
swift run -c release FlintFieldfareMac \
  --model "$HOME/models/flint-mixed-3-4bit"
```

## Repository structure (Flint targets)

| Path | Purpose |
| --- | --- |
| `Sources/FlintFieldfareCore/` | Flint model loading, Metal kernels, forward runner, bounded expert cache |
| `Sources/FlintFieldfareBench/` | Standalone benchmark CLI (planned F6) |
| `Sources/FlintFieldfareServer/` | Standalone SwiftNIO HTTP server (planned F6) |
| `Tests/FlintFieldfareCoreTests/` | Flint-specific test suite (12 passing) |

Existing `QwenFieldfareCore`, `QwenFieldfareMac`, `QwenFieldfareBench`,
`QwenFieldfareServer` targets remain untouched.

## Measured results (all-resident MLX baseline)

Machine: Apple M5 (`Mac17,2`), 32 GiB unified memory, macOS 27.2.1, Swift 6.4.2.
Measurements captured on 2026-11-18 using `mlx` 0.32.4 / `mlx_vlm` 0.7.12.

| Measurement | Result |
| --- | ---: |
| Cold load time | 4.39 s |
| Active memory (weights) | 16.61 GB |
| Peak memory (8K context) | 17.68 GB |
| Swap triggered | +657.6 MB (on 32 GB machine) |
| Short prompt decode | ~70 tok/s |
| 8K prompt decode | ~60 tok/s |
| 8K prompt prefill | ~1,317 tok/s |

### Streaming design targets

| Component | Footprint |
| --- | ---: |
| Resident weights | ~1.28 GB |
| Expert cache (1,024 slots, 3-bit) | ~1.31 GB |
| KV cache (8K context) | ~0.26 GB |
| Metal activations + scratch | ~0.15 GB |
| Process runtime + heap | ~0.20 GB |
| **Planned total working set** | **~3.20 GB** |

Bounded streaming caps unified memory at ~3.2 GiB (vs 17.7 GB all-resident),
preventing swap on a 32 GB machine. Interactive speeds of 15–30 tok/s are
expected once the expert cache reaches ≥70% hit rate.

## How active-expert streaming works

For each transformer layer and generated token:

1. Metal executes normalization (RMSNorm) and the layer's attention path
   (full or sliding-window with learned sinks).
2. The FP32 sigmoid router scores all 256 experts and selects the top-6.
3. Normalized scores sum to 1 (in FP32).
4. Each selected expert's SwiGLU MLP is evaluated. Experts already resident
   in cache are served directly; missing experts are streamed from the
   model's memory-mapped safetensors shards into Metal-visible buffers.
5. Each expert writes its weighted down-projection into a scratch buffer at
   position $k \times 2048$ ($k \in [0, 5]$), regardless of cache residency.
6. A dedicated combine kernel sums these contributions in fixed order 0→5.
   This guarantees bit-identical output across all-hit, partial-hit, and
   all-miss cache states.
7. The shared expert (ungated SwiGLU) is evaluated independently and added.
8. Final RMSNorm + LM head (FP32) → logits over 150,144 tokens.

Dense weights, attention projections, norms, routers, and shared experts have
permanent resident ownership. The complete routed-expert bank (256 × 47) is
never allocated as Metal buffers in streaming mode.

## Build and validation

```bash
swift build -c release
```

Run the Flint-specific test suite (F1 milestone — 12 tests passing):

```bash
swift test --filter FlintFieldfareCoreTests
```

Teacher-forced parity tests and production inference qualification are planned
for milestones F3 and beyond. No production-readiness claim is made.

## Key design decisions

1. **Isolated implementation line** — Flint targets live alongside untouched
   Qwen code. No risk of breaking the Qwen build.
2. **Fixed router-order reduction** — Accumulate by position 0→5, never by
   cache residency. Zero residency-dependent floating-point drift.
3. **Dual 3-bit / 4-bit support** — Dynamically detects from `config.json`.
   Works with both the official INT4 and the 3-bit mixed community checkpoint.
4. **Learned attention sinks** — Per-head scalars (`softmax_offset`) integrated
   into softmax denominator without dummy KV memory.
5. **Dedicated inference lease** — Lock at `/tmp/flintfieldfare-inference-{uid}.lock`
   prevents multiple concurrent Flint instances from exhausting memory.

## Known limitations

- Inference runtime (F2–F3) is under active development. No generation yet.
- Text-only; vision, audio, and video are not implemented.
- The local server and chat UI are not yet built for Flint.
- The runtime does not download the model. Install with `hf download` first.
- Cold-start performance will be lower than all-resident MLX until the
  expert cache warms up.

## Attribution

This project derives from [TurboFieldfare](https://github.com/drumih/turbo-fieldfare)
by Andrey Mikhaylov — introducing the bounded expert streaming design on
Apple Silicon. QwenFieldfare ports and extends that work for Qwen 3.6 35B-A3B
and is now being extended for Recursive Flint 40B-A2.5B.

The original TurboFieldfare source and this derivative are distributed under
the [Apache License 2.0](LICENSE).

QwenFieldfare is an independent community research project. It is not
affiliated with, sponsored by, or endorsed by the Qwen team, Recursive SI,
MLX, Hugging Face, Apple, or the original TurboFieldfare author.

## Model license and redistribution

Model weights are separate from this source repository:

- Base model: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint)
- Community 3-bit mixed: [mlx-community/flint-mixed-3-4bit](https://huggingface.co/mlx-community/flint-mixed-3-4bit)

Both model repositories identify their licenses. If you publish a separate
copy or modification of the model, include the applicable license,
model-card/source attribution, and a clear description of your changes.

## License

Source and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE).
