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

## Conversation state

Native chat resets KV, Conv and DeltaNet for each request. The standalone
and native local server can retain one exact consumed-token prefix per
runner. The default is off. Reuse requires identical represented token
IDs and a fresh suffix. Expert caching remains separate. The context
setting bounds the ledger and model allocations. The strict teacher-forced
release blocker remains in the [numerical policy](NUMERICAL_NON_REGRESSION.md).

## Standalone local server (S2-02)

Build/run/help/version and authenticated health/model/completion commands are in
[AGENT_INTEGRATION](AGENT_INTEGRATION.md). Controls: required model directory;
model-id; host127.0.0.1 only; port0–65535 (0 ephemeral); positive context checked
against the3/4/5GiB Metal plan; queue-capacity0–8 (default2). One active generation,
full replay by default. Key comes from QWENFIELDFARE_API_KEY
OR a private owned regular --api-key-file; secrets are not command-line values.

Readiness is authenticated GET /health after successful initialization/bind;
it includes active/queued counts and capacity. SIGINT/SIGTERM stop admission,
cancel/drain current work and release ownership on exit. Both Q4 runner entry
points now hold a cooperative per-user lease: app, Bench and standalone server
cannot silently load competing Qwen instances. Stop/unload the current owner
before switching. Other inference products still require manual exclusion.

JSON/SSE content and validated function calls/results are supported; tools execute
only in clients. OpenCode1.19.5 provider qualification is documented separately.
Native server controls and Keychain authentication are implemented in S2-03.
Click Local Server for validated persistent settings,start/stop/restart and status.
It shares the existing in-process server/backend, cancels/drains/unloads native
chat before server loading and blocks native sends until stop. Native chat works
again after server release. Observe CLI is authenticated/read-only; Disconnect
and app Quit never stop the independent CLI. App Quit drains its own server.
No auto-start on reopen,no child process,no app tool executor or server-chat tool
frame display. OpenCode alone executes tools. Copy/reveal/export credentials are
explicit warned actions; provider guidance contains an environment placeholder.

The server section shows active/queued/completed/rejected/failed/cancelled counts,
current context/budget,observed Metal allocation and one last finished request
record. Admission/queue,format/tokenize,prefill,first generated token,visible SSE,
validated tool buffering,decode-after-first and overall time are measured where
observable; unavailable values stay unavailable. Metal allocation is not total
RAM or a measured peak. Native inspector timings refer to its last native run.
See the integration guide for metric definitions,Keychain/private-file contract,
external observation version requirements and remaining lifecycle limitations.
The known131/132 numerical release blocker remains unchanged.

## Opt-in exact-prefix controls

- Standalone server: `--prefix-cache exact-prefix`. Use `--prefix-cache off`
  or omit the flag for full replay.
- Native app: open **Local Server**, stop the owned server if needed,
  select **Exact prefix**, then start. The preference persists and the
  runner starts with an empty ledger. Restart constructs a fresh runner.
  The **Clear Prefix Cache** button is available while the owned server is
  ready or generating. Independent CLI observation is read-only.
- HTTP: authenticated `POST /v1/cache/clear` accepts an empty request body
  or `{}`. Clear changes the epoch and any generation already in progress
  may finish without publishing a prefix. Expert weights remain cached.

Authenticated health reports mode, validity, last invalidation,
hit/miss/full-replay totals, reused/fresh tokens, prefill timing, context
capacity and token-array element bytes. Health excludes prompts, token IDs
and tool payloads. See the [state contract](EXACT_PREFIX_STATE_CONTRACT.md)
and preserve the [first S3-00 investigation](evidence/2026-11-18-s300/README.md).
