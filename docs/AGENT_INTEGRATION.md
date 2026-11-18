# Local agent integration

S2-01 adds validated native Qwen function tools to the S2-00 local HTTP server.
JSON and SSE call/result continuations are verified. S2-02 qualifies installed
OpenCode1.19.5 separately below; Hermes remains unqualified. Strict MLX parity remains
131/132; this research backend is not release qualified.

## Opt-in conversation prefix cache

Start the CLI with `--prefix-cache exact-prefix` or choose **Exact prefix**
in the native **Local Server** settings before starting. The default
`off` replays every prompt. Canonical tool continuations can reuse the
last completed request's consumed state when their formatted token IDs
match its entire represented sequence. A history, template or sampler
mismatch triggers full replay. Preserve validated assistant calls and
matching tool results as received. An equal prompt with no fresh suffix
and a length-limited completion are ineligible.

Send authenticated `POST /v1/cache/clear` with `{}` or an empty body,
or use the native **Clear Prefix Cache** button. The current generation
may continue after clear, with publication disabled for its epoch. The
next request starts from reset state. Authenticated health has
`diagnostics.prefixCache` for current state and `requests.lastRequest`
for timing and the most recent operation's reuse counts. Those records
contain no payload or raw IDs. Tool execution remains with the OpenCode
or other client. See [runtime controls](RUNTIME_CONTROLS.md).

## Build and run

```bash
swift build -c release
umask 077
mkdir -p "$HOME/.config/qwenfieldfare"
openssl rand -hex 32 > "$HOME/.config/qwenfieldfare/api-key"
# Choose key file OR environment key; do not configure both.
unset QWENFIELDFARE_API_KEY
.build/release/QwenFieldfareServer \
  --model /absolute/path/to/Qwen3.6-35B-A3B-4bit \
  --host 127.0.0.1 --port 8080 --model-id qwen-fieldfare \
  --context 8192 --memory-gib 5 --queue-capacity 2 \
  --api-key-file "$HOME/.config/qwenfieldfare/api-key"
```

Keep the key private and configure the same bearer key in clients. Environment
authentication remains supported; a secret command-line option is not supported.
Key files must be regular, owned by the current UID, without group/other access
(0600 or0400), at most512 printable ASCII key bytes plus an optional newline;
leaf symlinks/FIFOs reject through the opened descriptor. Environment keys use
the same16–512 byte printable, whitespace-free policy. Help/version require no
model or secret. Repeated/unknown/missing options and impossible context/memory
combinations fail before model loading. Port0 selects an ephemeral port, reported
at readiness; normal fixed ports are1–65535. Queue capacity0–8, default2.

```bash
.build/release/QwenFieldfareServer --help
.build/release/QwenFieldfareServer --version
# In another terminal, for the authenticated checks/client below:
IFS= read -r QWENFIELDFARE_API_KEY < "$HOME/.config/qwenfieldfare/api-key"
export QWENFIELDFARE_API_KEY
```
Only127.0.0.1 binding is allowed. Every endpoint requires authorization; browser
Origin requests are rejected and no CORS is enabled. Health becomes available
only after tokenizer/model initialization; startup errors exit nonzero without
printing sensitive details. Default model ID is qwen-fieldfare (single configured
Qwen checkpoint, no model switching). Budget choices3/4/5GiB use Core's existing
planner; these are Metal allocation targets, not process RAM caps.

```bash
printf 'header = "Authorization: Bearer %s"\n' "$QWENFIELDFARE_API_KEY" | curl --config - http://127.0.0.1:8080/health
printf 'header = "Authorization: Bearer %s"\n' "$QWENFIELDFARE_API_KEY" | curl --config - http://127.0.0.1:8080/v1/models
printf 'header = "Authorization: Bearer %s"\n' "$QWENFIELDFARE_API_KEY" | \
  curl --config - -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
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

One retained Core runner/tokenizer, one active generation, configurable0–8 queued
requests (default2);
full queue returns429, shutdown admission returns503. Networking uses NIO event
loops. Serialized detached work owns all Core state. The default replays every prompt.
Disconnect cancels active/queued work; cancellation waits for the current Core
forward/Metal work to finish. SIGINT/SIGTERM stop admission/listening, cancel and
drain work before event-loop shutdown/process exit. A Core RAII flock acquired by both
Q4 runner constructors before Metal allocation excludes another cooperating
QwenFieldfare app/Bench/server for the same UID. Runner lifetime owns the lease;
failed initialization and unload release it. The persistent0600 lock file is
never unlinked; deleting it while owned can defeat exclusivity. It is a
cooperative resource guard, not protection against hostile same-UID processes.
MLX/Ollama/LM Studio do not honor it and still require manual preflight.
The server retains its earlier server-instance lock as well.

Use Ctrl-C for foreground shutdown or SIGTERM to the exact server PID, then wait
for exit before starting another owner. Authenticated health includes status,
model,tools,active_generations,queued_requests,queue_capacity. It reports ready
while work is active; counts are observational, not admission reservations.
No loading endpoint exists before the listener binds. S2-03 now offers native server controls in the app, using the same CoreBackend
and LocalServer in-process with exclusive native-chat/server modes (details below).

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
Clients, not this server, execute tools. The bounded function-tool subset below is implemented;
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
boolean additionalProperties. S2-02 additionally accepts the exact2020-12
$schema URI and numeric minimum/maximum/exclusiveMinimum/exclusiveMaximum on
numeric types, enforces those bounds, and preserves Int64 comparison precision.
Other dialects or unsupported constraints reject rather than being discarded. Max64 properties per object, schema nesting12 and
whole-JSON nesting16 (the tighter whole-request bound also applies). Root function
arguments must be object; native parameter names must be declared. Unknown schema
keywords including $ref/$defs,combinators,pattern/format reject explicitly;
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

## Installed OpenCode configuration (S2-02)

Tested client: Homebrew OpenCode1.19.5, Bun1.4.4. Its installed v1 configuration
uses provider/npm/options/agent, while the current website documents a newer v2
contract. The [tested provider example](evidence/2026-11-17-s202/provider-example.json)
is pinned to this client; do not assume it applies to every future version.
Official references: [providers](https://opencode.ai/docs/providers/),
[configuration](https://opencode.ai/docs/config/), [agents](https://opencode.ai/docs/agents/).

Copy the example to opencode.json in an explicitly selected disposable repository.
It points directly to127.0.0.1:8080/v1 and reads the bearer key from the environment,
sets model context8192/output1024 and SDK timeout600000ms. Keep client context/output
within server limits. Its read/grep permissions allow workspace inspection;
edit/bash require client approval and external_directory denies outside access.
The real qualification uses a stricter fixture-only edit/exact-test-command allowlist.
OpenCode permission patterns for read/edit are workspace-relative. Absolute patterns
can silently deny valid workspace files. These permissions are client policy, not
an OS sandbox; use a disposable repository and trust/review the client installation.

```bash
opencode --version
# Canonical working directory and matching PWD prevent inherited session paths.
cd /absolute/path/to/disposable-repository
export PWD="$(pwd -P)"
export OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_AUTOUPDATE=1
export OPENCODE_DISABLE_CLAUDE_CODE=1 OPENCODE_DISABLE_EXTERNAL_SKILLS=1
export OPENCODE_DISABLE_AUTOCOMPACT=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1
opencode debug config --pure
opencode models fieldfare --pure
opencode run --pure --format json --dir "$PWD" \
  --model fieldfare/qwen-fieldfare --agent fieldfare --title 'Local coding task' \
  'Read the source and explain the failing test before proposing an edit.'
```

Model listing above uses OpenCode's configured provider catalog, not HTTP discovery.
The authenticated /v1/models endpoint is checked separately. --pure disables plugins;
the opt-in qualification script isolates all XDG cache/data/state/config directories
so it neither edits nor loads normal user configuration. For comparable isolation,
set those directories to private scratch paths before running commands yourself.
No auto-compaction, auxiliary remote model or delegated agent is used in qualification.

```bash
# After release build, normal-pressure/no-other-model preflight:
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/path/to/Qwen3.6-35B-A3B-4bit \
  python3 Scripts/test_opencode.py
```

The script starts one server at18092 and a byte-preserving timing relay at18093,
creates only a disposable Git workspace, and has OpenCode execute read/grep/edit
and the exact unittest command. The relay records timing/usage aggregates and never
executes tools or modifies payloads. Only public synthetic test conversations are
saved by this explicit test client; production server logs remain content-free.
See [S2-02 evidence](evidence/2026-11-17-s202/README.md) for qualification status,
latency, retries, memory, first failed permission attempt and known limitations.
Tool SSE is still delayed until complete native-turn validation/EOS; safe prose
streams earlier. A600s client timeout accommodates observed prefill/tool latency;
this is functional qualification, not a throughput claim or long-context benchmark.

## Native server management (S2-03)

Open the macOS application and choose **Local Server** in the header. It reuses
CoreBackend/LocalServer in-process; the standalone executable remains independent.
Choose a compatible directory,loopback port (0 ephemeral),context,3/4/5GiB target,
and queue0–8. Invalid port/context/budget combinations disable Start. Missing or
incompatible models return recoverable load errors. Stop before editing/reloading;
settings persist in UserDefaults,not credentials. Reopen never auto-starts.

Start first blocks new native sends, cancels/drains existing native work and
unloads it; only then constructs the server. Native chat is intentionally disabled
while owned or observed server mode is active (option B). Stop cancels/drains HTTP
work, releases the server/runner/lease, then re-enables chat. Restart uses the same
path; SO_REUSEADDR permits the used port to rebind without SO_REUSEPORT or shared
listeners. App Quit waits for drain. There is no supervised child to orphan, and
an app crash shares the server's fate. A blocked GPU/legacy fatal failure is not
made recoverable by this controller; immediate load cancellation is not promised.

The app's bearer key lives in the local,non-synchronizing Keychain item
QwenFieldfare.local-server/default (WhenUnlockedThisDeviceOnly). Each startup uses
a private0700 temporary directory/0600 key file and the exact CLI configuration
parser; the temporary file is removed before serving. No bearer token appears in
argv,defaults,diagnostics or safe provider guidance. Explicit Copy/Reveal/Export
asks for confirmation; exported files are0600,new-file-only,no-follow,never
overwritten. Use a new writable private path if export fails. Keychain access can
require an OS prompt; unlock/retry on failure. Import CLI Key requires its same
owned/private/regular bounded file contract. There is no plaintext-defaults fallback.

**Copy API URL** copies the bound endpoint; port0 is resolved after start.
**Copy OpenCode Guidance** includes a key environment placeholder,observed model/context
and a conservative output limit,using tested1.19.5 v1 provider fields and the
bounded fieldfare agent. Use the supplied --agent fieldfare command; unrestricted
default client agents can send unsupported schemas or require broader permissions. Set the
matching key privately in the client environment; keep client tool permissions
restricted. The GUI-managed instance passed real JSON/SSE,a native function/result
continuation and installed OpenCode; the prior full coding-workflow evidence is
retained and not replaced with a new performance claim. Native chat does not
consume tool-call frames in this exclusive mode; OpenCode executes tools externally.

For an already running CLI, import its matching private key file,select its port
and click **Observe CLI**. It authenticates GET /health only, rejects redirects,
uses no proxy/cookies/cache,and bounds response bytes to64KiB. Current S2-03 health
metadata is required; older processes without the service/diagnostic fields cannot
be observed until restarted with the current executable. This is read-only:
Disconnect or Quit never signals/stops/reconfigures an independent CLI.
Disconnect resumes native input, but the independent CLI must be stopped by its
owner before native model loading can succeed; the Core lease enforces that. A port
conflict offers observation/change-port guidance; a different-port model owner
returns the shared inference-ownership error. Never delete ownership lock files
while a process holds them. Other inference products still require manual exclusion.

Health now adds service,diagnostics and requests to the existing fields. Counters:
completed=successfully flushed chat responses;rejected=HTTP4xx/5xx validation/auth/
route replies before admission;failed=admitted generation/parser errors;
cancelled=active or queued cancellation. Health/models reads do not count as
completed chats. Counters saturate and reset per server lifetime. The UI polls
owned state directly once/sec; external state uses bounded authenticated GET.
Counts are snapshots and can miss brief active phases; they are not reservations.

One last finished request records server-header receipt through admission,queue
wait,prompt-format/tokenize time and token count,Core prefill callback,first
emitted non-EOS token,first successfully flushed visible SSE content,recognized
complete tool-opening delimiter through validated commit,decode-after-first-token
and overall duration/output count where observed. Generated/visible/overall times
share a monotonic request origin; prefill uses the existing Core timer. Decode
includes callbacks/output work and finishing EOS; it is not pure GPU kernel time.
SSE flush is not client receipt. JSON/empty/cancelled/no-tool paths leave relevant
values unavailable. No prompts,arguments,source text or token arrays are stored.
This instrumentation does not optimize or attribute the older long-workflow
latency. S3-00d separately qualifies exact-prefix state reuse.

See [S2-03 evidence](evidence/2026-11-17-s203/README.md) for real lifecycle tests,
initial restart failure/fix,model-free gates,measurement limits and unchanged
numerical release blocker. The private explicit test client can be reproduced
against a running managed instance with `python3 Scripts/test_managed_server.py
--key-file /private/path/key --port 8080`; it never loads another model and saves
only public synthetic fixture exchanges/aggregate measurements.

## Historical S3-00 qualification outcome

The first candidate was reverted after sustained raw-token divergence.
The archived evidence records that failure and the full-replay expert-history
sensitivity. S3-00d uses the fixed router-order control under the opt-in
settings described above. See the
[historical evidence](evidence/2026-11-18-s300/README.md).
