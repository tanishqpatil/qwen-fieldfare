# Agent integration

Status on 2026-11-17: **no server executable or implemented HTTP API**. Existing
products are Core, Mac, Bench and Pack. OpenCode and Hermes are unqualified.
There is no working provider configuration, tested client version, authentication
scheme or server startup command to publish yet.

## Planned contract (Stages 2–4; not available)

Only `GET /health`, `GET /v1/models` and `POST /v1/chat/completions` are in scope.
Chat Completions will provide JSON and bounded SSE, exact model IDs, usage,
finish reasons, structured tools and actionable errors. Supported fields,
unsupported-field rejection and API error mappings must be specified and tested
before this document becomes a usage guide. Do not guess a model identifier or
claim full OpenAI API coverage.

Bind to `127.0.0.1` by default. Specify and test local authentication before
serving. Request body, prompt/context, output, queue and streaming buffers need
explicit finite limits. Slow clients must exert backpressure or be disconnected;
client disconnect cancels generation. Serialize inference on exactly one owner
and prevent simultaneous app/bench/server model loads. Networking must not run
on the blocking inference executor. Do not log prompts, tool arguments/results,
auth secrets or sensitive payloads. Never execute tools in Core or the server.

Stage 2 must derive formatting from the actual compatible checkpoint template,
including system/developer/user/assistant/tool roles, matching IDs, multiple
calls, thinking controls and tool-choice behavior. A bounded incremental parser
must survive arbitrary fragment boundaries, Unicode, escaped strings, nested
JSON, malformed output and unsupported/duplicate/missing tool metadata. Dispatch
only complete valid calls; client-side tools own filesystem and shell access.

## Qualification gates

- Stage 3: HTTP tests for nonstream/stream/tool exchange, malformed input,
  queue limits, backpressure, disconnect, auth and context overflow.
- Stage 4: pin the installed OpenCode version after inspecting its current
  provider schema; run actual read/edit/test/failing-test-repair in a disposable
  repository through this backend. Record retries, tool validity, actual token
  use, latency and memory. A mocked exchange does not qualify OpenCode.
- Stage 8: inspect installed Hermes protocol and requirements, then test actual
  occupied long histories, auxiliary calls and the single-session cache policy.
  A 65,536-token allocation alone does not qualify long-context behavior.
- Claude Code remains separate experimental research.

Integration configuration and tested version records will be added only when
these gates run. See [backlog](../TODO.md) and [architecture](ARCHITECTURE.md).
