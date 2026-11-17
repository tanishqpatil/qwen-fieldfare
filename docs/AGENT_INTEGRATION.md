# Local agent integration

S2-01 adds validated native Qwen function tools to the S2-00 local HTTP server.
JSON and SSE call/result continuations are verified; OpenCode/Hermes end-to-end
client qualification is **not implemented**. Strict MLX parity remains
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
Legacy text-only system/user/assistant history retains final-user validation;
structured tool history additionally supports matching tool results.
existing Core template disables thinking and cleans assistant history as in app.
Sampling: temperature,top_p,top_k,presence_penalty,repetition_penalty,seed; output
limit max_tokens or max_completion_tokens; n=1 only. Limits/defaults are validated.
Developer roles,content arrays,stop,frequency_penalty,
response_format,logprobs and all other unknown fields return explicit errors.
No server-side tool execution, cross-request prefix cache or generic OpenAI API is present.
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
Next task is actual OpenCode provider/tool-cycle qualification. Clients, not this
server, execute tools. The bounded function-tool subset below is implemented;
compatibility with every OpenCode schema/setting is not yet claimed.

## Function-tool contract (S2-01)

Accept standard `tools:[{type:"function",function:{name,description,parameters}}]`,
`tool_choice:"auto"` (default) or `"none"`. `strict:false` is accepted; strict:true,
required/named choice,parallel_tool_calls,legacy functions/function_call and unknown fields reject400.
1–32 uniquely named tools, names/parameter names/IDs1–64 ASCII letters/digits/_/-.
Assistant `tool_calls` use unique IDs,type:function,function:{name,arguments};
arguments is a JSON-object string. Assistant content may be string,null or absent
when carrying calls. Tool results use role:tool,tool_call_id and string content.
Each call needs exactly one result, in call order, before another conversational
turn. Parallel results arriving out of order must be ordered by the client; the
native template carries no IDs, so ambiguous ordering is rejected. Definitions
must accompany replayed calls; unknown functions/duplicate IDs/results reject.

Schema subset: explicit scalar types string/number/integer/boolean/null, objects,
arrays with items, nullable type arrays,enum,description,title,properties,required,
boolean additionalProperties. Max64 properties per object, schema nesting12 and
whole-JSON nesting16 (the tighter whole-request bound also applies). Root function
arguments must be object; native parameter names must be declared. Unknown schema
keywords including $ref/$defs,combinators,pattern/format/bounds reject explicitly;
this is not a general JSON Schema implementation. JSON duplicate keys, bare integer literals beyond Int64 and non-finite numbers
reject. Fraction/exponent numbers use Double; other numbers use existing
Int64/Double template semantics, not arbitrary-precision math. Numeric literals
are capped at128 bytes and precision-losing renderings reject instead of silently
rounding arguments. Argument/object
member order is retained; numbers,booleans,objects,arrays,null preserve JSON types.
Ambiguous string/other-type native values are rejected rather than guessed.

The checkpoint's actual chat_template.jinja is rendered by the existing Jinja
engine for tool conversations. Its exact tool preamble, assistant empty-thinking
prefix, grouped user/tool_response wrappers and im_end turn semantics are used.
Developer messages are rejected because the template does not support them.
Ordinary no-tool requests retain the existing Core formatter. `none` with tools
hides definitions and forbids generated native calls while allowing history replay.
No special tokens or inference/sampling/recurrence behavior are changed.

Parser state holds partial delimiters and never leaks incomplete native markup
as prose. It streams safe text before calls, buffers structured tail up to256KiB,
validates <=8 calls against names/required parameters/schema, and commits only at
actual normal EOS. A token limit or cancellation cannot authorize a tool call.
Complete validated calls get stable call IDs; SSE emits one indexed ID/name delta
then Unicode-safe argument fragments, finish_reason:tool_calls,usage (if asked)
and DONE. Tool deltas are deliberately delayed until whole-turn validation/EOS;
clients must wait for complete arguments/finish and independently authorize tools.
Malformed/truncated/unknown/oversized tool output returns502 (or an error SSE event
if prose already started), no executable calls and no successful DONE. Native
string parameters use the template's raw text value convention, not quoted JSON.
There is no XML entity decoding or XML parser.

Tool text is untrusted. Reserved native tool/role/reasoning delimiters in tool
results,structured assistant content or argument values are explicitly rejected;
no automatic escaping or different conversation protocol is invented. This blocks
structural boundary injection for the supported subset, not semantic prompt
injection. Ordinary malicious instructions can still influence the model; clients
must sandbox/authorize tools. The server never executes a function or shell command.

## Reproduce a genuine tool cycle

```bash
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/path/to/Qwen3.6-35B-A3B-4bit python3 Scripts/test_server_tools.py
```

Run after release/preflight with no other model owner. The client requests
get_project_status(project:string), validates returned tool_calls, supplies the
fixed public mock result status=building-local-agent, and asks the same loaded
model to continue. It checks JSON and SSE, plus no-tool/none/context/cancellation
regressions. Only the client supplies results; no actual project/filesystem tool
is executed. The server logs no content or arguments. The test client saves only
its public synthetic exchanges as explicit reproducibility evidence.
[S2-01 evidence](evidence/2026-11-17-s201/README.md) includes results and limitations.
