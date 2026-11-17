# QwenFieldfare

## Current milestone — S2-02 CLI and real OpenCode qualification

Explicit user authorization supersedes the prior S2-01 stop. Preserve completed
server/tools/readers and known131/132 release blocker. Implement validated CLI
controls and qualify installed OpenCode in a disposable repository. Stop after
S2-02; do not begin S2-03 SwiftUI management in this execution. No push/merge.


## Historical milestone — S2-01 structured tool contract

Explicit user authorization supersedes the prior S2-00 stop. Preserve its server
checkpoint and all reader/numerical evidence. Implement native Qwen tools, JSON
and SSE translation, matched tool results and one real continuation. Never execute
tools in the server. Stop after S2-01; no prefix reuse/readers/Hermes/push/merge.


## Historical priority override — S2-00 (2026-11-17)

Explicit user decision supersedes earlier Stage1-only/order/stop directives.
Build and qualify the minimal local QwenFieldfareServer now. S1-01e and remaining
noncritical reliability work are deferred, not completed or abandoned. Preserve
S1-01a–d and strict131/132 release blocker. Stop after real nonstream/SSE HTTP
generation; tools, prefix reuse and further reader migrations remain deferred.
No push/merge. Current implementation/validation status is in DEVELOPMENT_LOG.


Native Swift and Metal inference for the MLX Q4 build of Qwen 3.6 35B-A3B on
Apple Silicon.

## Scope

This repository is Qwen 3.6-only. The 2026-11-17 implementation mission
explicitly authorizes a narrowly scoped local Qwen Chat Completions server,
client-side tool calling and exact hybrid-state reuse in gated later stages.
Do not add unrelated model formats, generic multi-model hosting, server-side
tool execution or a second model-owning process. Preserve bounded expert streaming.

## Session recovery and stage gates

Read this file, `TODO.md` and `docs/AGENT_MEMORY.md` at session start. The TODO
owns stable mission task IDs/status; architecture and decisions distinguish
implemented features from proposals. Read relevant source before editing;
update recovery memory, task status and append-only DEVELOPMENT_LOG as work
progresses. Source/tests/verified evidence outrank narrative memory.

Follow the mission stages sequentially, one small testable task per iteration.
The 2026-11-17 revised policy permits independent Stage 1 reliability work while
131/132 teacher forcing remains an explicitly unresolved baseline. Strict 132/132
is a release-quality blocker, not a global development blocker. Preserve its
fixture/evidence/tests; never suppress failures or claim full parity. New or
worsened mismatches block affected changes. Production-readiness claims require
an independently reviewed numerical policy and sufficient evidence.

Historical Stage1-only boundary is superseded by the current S2-02 directive above.
No further broad/layer-3 numerical investigation without a bounded hypothesis
and decision it informs, or a change directly requiring it. Model-free tests
first; model-backed checks only for inference-affecting work/integration gates.
Reuse verified evidence; never skip required integrity checks, but do not hash
all weights repeatedly without an identity/integrity reason. Preserve unrelated
work and historical rejections. No push/merge without approval. Project memory
must contain no secrets, private prompts or sensitive runtime payloads.

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
`QwenFieldfareMac|QwenFieldfareBench|QwenFieldfareServer|mlx_lm|mlx-lm|ollama|LM Studio`.
Run only one model-owning process at a time.

## Validation

Run package tests through `Scripts/test.sh`. Build release once before a model
measurement and follow `docs/COMMUNITY_BENCHMARKS.md` exactly. Profiling flags
are diagnostic and must not be used for headline throughput.

Report the commit, hardware and RAM, macOS, Swift version, exact command,
prompt/generated token counts, complete timing/cache footer, memory, and every
protocol deviation. Measurements are not performance ceilings.
