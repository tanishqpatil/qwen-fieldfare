# Local agent integration

S2-00 implements one local text model over HTTP/1.1. OpenCode/Hermes end-to-end
qualification and tool calling are **not implemented**. Strict MLX parity remains
131/132; this research backend is not release qualified.

## Build and run

```bash
swift build -c release
export QWENFIELDFARE_API_KEY="$(openssl rand -hex 32)"
.build/release/QwenFieldfareServer \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --model-id qwen-fieldfare --port 8080 --context 8192 --memory-gib 5
```

Keep the environment secret private and configure the same bearer key in clients.
Only127.0.0.1 binding is allowed. Every endpoint requires authorization; browser
Origin requests are rejected and no CORS is enabled. Health becomes available
only after tokenizer/model initialization; startup errors exit nonzero without
printing sensitive details. Default model ID is qwen-fieldfare (single configured
Qwen checkpoint, no model switching). Budget choices3/4/5GiB use Core's existing
planner; these are Metal allocation targets, not process RAM caps.

```bash
curl -H "Authorization: Bearer $QWENFIELDFARE_API_KEY" http://127.0.0.1:8080/health
curl -H "Authorization: Bearer $QWENFIELDFARE_API_KEY" http://127.0.0.1:8080/v1/models
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $QWENFIELDFARE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-fieldfare","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Omit stream or use false for a chat.completion JSON response. SSE sends role,
content deltas, finish chunk, optional usage-only chunk, then data: [DONE].
Stop EOS maps to stop; output/context exhaustion maps to length. Usage counts
Core prompt tokens and emitted completion tokens; excluded EOS is not counted.
Text-only system/user/assistant history is supported, with final user required;
existing Core template disables thinking and cleans assistant history as in app.
Sampling: temperature,top_p,top_k,presence_penalty,repetition_penalty,seed; output
limit max_tokens or max_completion_tokens; n=1 only. Limits/defaults are validated.
Tools/tool_choice,developer/tool roles,content arrays,stop,frequency_penalty,
response_format,logprobs and all other unknown fields return explicit errors.
No tool execution, cross-request prefix cache or generic OpenAI API is present.
Protocol references: [request](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[streaming](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).

## Lifecycle and limits

One retained Core runner/tokenizer, one active generation, two queued requests;
full queue returns429, shutdown admission returns503. Networking uses NIO event
loops; serialized detached work resets/replays Core state for every request.
Disconnect cancels active/queued work; cancellation waits for the current Core
forward/Metal work to finish. SIGINT/SIGTERM stop admission/listening, cancel and
drain work before event-loop shutdown/process exit. File flock excludes another
server for the same OS user; app/Bench/MLX do not honor that lock, so stop them
manually before startup. There is no implemented cross-application owner lock.

1MiB body,128 messages,4096 output tokens (default256), prompt+requested output
must fit selected context.32 accepted connections,15s header/body deadline,one
request per connection,no keep-alive/pipelining.16KiB outbound watermark,one
flushed write at a time,10s write timeout; non-writable clients cancel instead of
accumulating streaming data. No heartbeat or resumable SSE. Errors are structured
JSON before SSE starts; after start an error data event closes the stream (no
false successful DONE). No request/content/secret/tool/weight logging; only safe
startup/load/shutdown aggregates. Existing fatal Metal encoder paths and immutable
checkpoint/ancestor-race limitations remain; this is not crash-proof production
serving. Loading itself is not immediately cancellable.

## Reproduce validation

```bash
Scripts/test.sh
python3 Scripts/test_numerical_non_regression.py
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/path/to/Qwen3.6-35B-A3B-4bit python3 Scripts/test_server_http.py
```

Run model integration only after AGENTS preflight and release build; it starts
one server on18089 and saves aggregate timings/counts, never content/keys. See
[S2-00 evidence](evidence/2026-11-17-s200/README.md) for actual results/limitations.
Next task is S2-01 tool-contract work, followed by actual OpenCode qualification;
clients, not this server, execute tools. No tool compatibility is advertised now.
