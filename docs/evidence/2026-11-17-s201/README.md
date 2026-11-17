# S2-01 structured tool-contract evidence

S2-00 checkpoint c001a78; S1 reader safety and original strict-parity evidence
preserved. No weights copied/changed, no Core numerical/cache/state changes.
Swift Jinja2.3.6 was already resolved and is now a direct support dependency;
no dependency pins upgraded. Exact template snapshots are text metadata, not weights.

## Template authority and protocol tests

checkpoint-template.jinja is the local model's exact file. official-template.jinja
was retrieved from the official Qwen/Qwen3.6-35B-A3B main template URL; hashes,
URL,EOS/token IDs and tokenizer facts are in template-provenance.json. template.diff
shows training-generation annotations/equivalent prefix relocation only.
Python Jinja3.1.6 renders both identically for three text tool fixtures; Swift
rendering matches those goldens exactly. No developer role support, no vision.
The tokenizer_config has no embedded template; the actual file is the authority.

```bash
# Existing Python environment with Jinja3.1.6; no MLX/model import/run:
python Scripts/generate_tool_goldens.py
Scripts/test.sh --filter 'ToolContractTests|ToolHTTPTests'
Scripts/test.sh
swift build -c release
python3 Scripts/test_numerical_non_regression.py
python3 Scripts/check_numerical_non_regression.py --log docs/evidence/2026-11-17-s101c/strict-parity.log
ruby Scripts/check_markdown_links.rb
git diff --check
```

Final focused14 pass. Package127 passed +3 opt-in skips (130 reported:20 Server,
17 Presentation,93 Core including skips). Release34.36s, checker unit6 pass,
Markdown/diff checks pass. See raw logs. Tests cover fragmentation, multiple calls,
all JSON value types,Unicode/multiline/quotes,malformed/truncated/limited output,
unknown/missing/duplicate/ambiguous input,buffer/depth limits,matching result IDs,
injection-like tool content,JSON and indexed SSE reassembly,context,none and
cancellation while a partial native delimiter is pending. Original text/auth/
loopback/queue/init/disconnect/shutdown tests remain unchanged and pass.

The final enum-number comparison and precision-preservation guards were tightened after real integration,
then focused/full/release gates rerun. The real test schema has no enum, so the
exercised model/parser path is unchanged; no costly replay repeated for that fix.

## Genuine model-generated tool cycles

```bash
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/path/to/Qwen3.6-35B-A3B-4bit python3 Scripts/test_server_tools.py
```

Actual checkpoint `/Users/tanishq/models/Qwen3.6-35B-A3B-4bit`,one server on18090,
context8192,budget5GiB,temperature0,max_tokens512. All tool calls/continuations
are genuine model output. The CLIENT supplies one fixed mock result; the server
executes no function, shell or filesystem tool. Public synthetic request/response
and SSE records are in public-tool-exchanges.json; aggregate metrics in
http-tool-metrics.json. No bearer secret or private conversation is persisted;
server.log contains only safe startup/load/drain aggregates.

| Mode | Call prompt/output | Call latency | Continuation prompt/output | Continuation latency |
| --- | --- | --- | --- | --- |
| JSON |344/27 tokens|16.196s|404/13 tokens|16.985s|
| SSE |344/27 tokens|15.313s|404/13 tokens|17.073s|

Both calls: get_project_status,arguments={"project":"fieldfare"},finish tool_calls.
Both client-supplied results: {"project":"fieldfare","status":"building-local-agent"}.
Both final model answers: `The current status of project fieldfare is "building-local-agent".`
Final finish stop; call IDs uniquely bind results and streaming indices/argument
reassembly validate. No-tool JSON/SSE,tool_choice:none,schema context exhaustion,
client disconnect recovery and clean SIGTERM shutdown also pass. Process exit0.
No fake forced-tool mock is used in these real cycles; deterministic fixtures
are separate model-free tests. Usage counts native model-output tokens, including
markup, not just the translated JSON characters.

Host/preflight: MacBook Pro Mac17,2,M5/32GB/10GPU;macOS27.2.1(26B131),Swift6.4.2;
132GiB disk,normal pressure1/free85%,no competing model owner. Postpressure1/free79%.
Load0.775s,health-ready1.310s,shutdown0.088s. Metal5,291,294,720 bytes (4.928GiB);
loaded footprint1,167,267,816,drained5,166,064,424 bytes. Runner remains retained
through drain until exit. Snapshots not peak/RAM cap; no optimization claim,
full-context/long-agent soak or cross-platform qualification. No full checkpoint
rehash or new strict parity replay was needed. Original131/132/token61 exit1 is
still an unresolved release blocker; historical-log checker proves no new numerical
accuracy and does not execute a model.

## Explicit limitations and trust boundary

Only auto/none and non-strict function tools; supported schema subset/IDs/order
and native-delimiter restrictions are documented in docs/AGENT_INTEGRATION.md.
Native text parameters are raw strings; JSON types follow declared schemas;
ambiguous values reject rather than guessed. Unknown native parameters reject.
Calls commit only after whole-turn EOS validation, then SSE argument fragments
are emitted. Tool events are intentionally delayed. Incomplete/malformed/limited
turns error with no executable tool call or successful DONE. <=256KiB parser,
<=8 calls,JSON depth16,tools<=32,properties<=64. Structural delimiter injection
in tool data is rejected; ordinary tool text remains semantically untrusted.
Client must authorize/sandbox execution. No server tool execution,developer/
vision/strict/reference schemas,required/named choice,prefix reuse,Hermes or
filesystem-capable agent. Actual OpenCode provider/client qualification is next,
not claimed complete. Existing Core fatal paths and immutable-checkpoint races
remain. Stopped after S2-01; local commit only,no push/merge.
