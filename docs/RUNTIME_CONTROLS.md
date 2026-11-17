# Runtime controls

## Model directory

Choose the compatible MLX Q4 directory through **Model → Choose Model
Directory…** or set `QWENFIELDFARE_MODEL` before launch. The selection is
validated and persisted. The environment variable overrides the saved path.

## Runtime modes

- **Active Expert Streaming**: recommended; resident dense weights plus a
  bounded routed-expert cache.
- **Fast Resident Q4**: highest-memory diagnostic path.
- **Memory Saver Q4**: lower-memory, lower-throughput fallback.

Changing the model, mode, or context unloads the current runner. These controls
are disabled while generation or model loading is active.

## Context and memory

Active streaming exposes 4K, 8K, 16K, 32K, and 64K contexts. Longer contexts
increase FP16 KV memory, so the app reduces expert-cache capacity under the
selected 3, 4 or 5 GiB Metal target (default 5 GiB). The checked planner filters
unsupported budget/context combinations. This is not a process-RAM cap.
Changing the budget unloads the runner. Fast Resident and Memory Saver expose
4K and 8K.

## Sampling

The app exposes maximum output tokens, temperature, Top-K, Top-P, presence
penalty, repetition penalty, and Qwen reasoning mode. Maximum output is capped
by the selected context; prompt and conversation tokens consume the same
window.

## Diagnostics

The inspector reports model state, Metal allocation, memory pressure, KV
estimate, prefill duration, time to first token, rolling and run-average decode
rates, token latency percentiles, expert-cache occupancy, and miss rate.

## Conversation state and future server controls

Every request currently resets and replays KV, Conv and DeltaNet state.
Cross-request prefix reuse, cache idle expiry/retained-token limits/clear action,
lazy KV growth and native UI server settings are not implemented. Expert caching
is separate from sequence state and conversation reuse. Selected context is an
allocation capacity, not measured occupied-history qualification. Current
teacher-forced correctness is blocked; see [baseline](PERFORMANCE_BASELINES.md).

## Standalone local server (S2-02)

Build/run/help/version and authenticated health/model/completion commands are in
[AGENT_INTEGRATION](AGENT_INTEGRATION.md). Controls: required model directory;
model-id; host127.0.0.1 only; port0–65535 (0 ephemeral); positive context checked
against the3/4/5GiB Metal plan; queue-capacity0–8 (default2). One active generation,
reset/replay each request, no prefix reuse. Key comes from QWENFIELDFARE_API_KEY
OR a private owned regular --api-key-file; secrets are not command-line values.

Readiness is authenticated GET /health after successful initialization/bind;
it includes active/queued counts and capacity. SIGINT/SIGTERM stop admission,
cancel/drain current work and release ownership on exit. Both Q4 runner entry
points now hold a cooperative per-user lease: app, Bench and standalone server
cannot silently load competing Qwen instances. Stop/unload the current owner
before switching. Other inference products still require manual exclusion.

JSON/SSE content and validated function calls/results are supported; tools execute
only in clients. OpenCode1.19.5 provider qualification is documented separately.
Native app server controls, Keychain management and tool-call display remain S2-03
proposals and are not implemented here. Existing app chat behavior is retained.
The known131/132 numerical release blocker remains unchanged.
