# S2-03 native server management validation

Completed native server controls over the existing in-process CoreBackend and
LocalServer. Standalone CLI preserved. Native chat is cancelled/drained/unloaded
before server start and blocked until server release. External CLI observation
has no stop authority. No child process,second runner,duplicated HTTP/tool stack,
server tool execution,prefix reuse or numerical kernel/state changes were added.

S2-02 was source-identity checked and re-tested136 passes+3skips,then locally
committed as f2f5672. S2-03 is the subsequent scoped checkpoint. No remote push/merge.
See [usage](../../AGENT_INTEGRATION.md) and [architecture](../../ARCHITECTURE.md).

## Validation

Final complete package150 pass+3 opt-in skips (153 total:30 Server,10 Management,
17 Presentation,96 Core including skips). Final release36.68s. Six numerical
checker tests and archived-log stability pass;this does not constitute a fresh
numerical replay or establish equivalence. Original reference fixture SHA
8d552a906d417ceb492ddd045a50e67b0e9630988d9665e8c6fe5e480cb30747 preserved.
All Core/Bench source,pins,teacher harness/reference/baseline are byte-identical
to f2f5672. Only server-side timing callbacks observe the existing runner. Strict
131/132/token61 expected14988 actual10000 remains an unresolved release blocker.

Model-free tests cover configuration/overflow/context bounds,private-file CLI
contract/export/import/mode rejection,start/stop/restart,persisted settings with
no credential defaults,invalid auth/missing model/occupied port,loading cancellation,
read-only external observation/auth/64KiB response bound,runtime health failure/
cleanup/reload,request counters/privacy,queued cancellation without inference,
immediate used-port rebinding and validated tool buffering timing. Final GUI
provider JSON includes the bounded fieldfare agent (workspace read/grep,edit/bash
ask,external-directory/delegation denied),observed model ID/context and no secret.
JSON escaping and exact installed OpenCode1.19.5 debug-config validation pass.

## Real native UI protocol and lifecycle gate

Host Mac17,2/M5/32GB,macOS27.2.1(26B131),Swift6.4.2.134GiB disk,normal pressure1;
initial free87%,later CLI preflight84%,postflight69%/pressure1. No competing model.
The app deliberately remained running **unloaded** for independent CLI observation,
a necessary exception to the blanket process-name preflight;the actual Q4 lease
and visible unload/conflict checks confirmed only one model owner at all times.
No model weights copied,staged,hashed again or changed. Checkpoint:
/Users/tanishq/models/Qwen3.6-35B-A3B-4bit . Existing required loader/extent checks
stay enabled;prior integrity evidence is reused,not bypassed.

CUA could not address the bundleless SwiftPM process. A disposable .app in
/private/tmp wrapped the exact release binary and symlinked its existing resource
bundles (binary identity in tested-binary.json). UI actions used that app;no
packaging/signing/distribution claim follows. Wrapper and temporary client key
were removed after testing. Keychain credential remains local as product state.

Observed in actual native UI:

- Native chat replies hello,1 output token,TTFT2189.9ms initially. Server start
  unloads it and disables native sends;Keychain authentication succeeds.
- UI-owned server ready,context8192,target5GiB,queue2. Authenticated real JSON,
  SSE,function call with client-supplied public result and final continuation pass.
  Actual OpenCode1.19.5 selects this endpoint and receives hello.
- Public2000-word alpha prompt admitted;Stop Server cancels mid-prefill,client
  gets RemoteDisconnected after5.698s total request age,listener closes and UI is
  Stopped/chat released. This is not5.698s cancellation-drain latency;it includes
  time waiting for the UI action. No false completed response was substituted.
- First immediate restart failed on the recently used address. Failure preserved.
  Added SO_REUSEADDR to early probe/actual listener,no SO_REUSEPORT;unit immediate
  rebind and actual UI Restart on18094 then pass. Counters reset,Metal remains4.928GiB.
- Quit while app owns server:process exits,listener closes,no child survives.
  Reopen is unloaded/stopped,settings port18094 persist,no auto-start.
- Native chat reloads and replies hello,1 token,TTFT838ms. Competing CLI while
  native owns model exits1 with ownership error before model allocation.
- Explicit native Unload allows independent CLI to start. UI Start sees occupied
  port and offers observation;Observe CLI is Ready/Independent CLI/Disconnect.
  Quit observer app leaves CLI serving a real hello response. Only our own test
  CLI then receives graceful SIGTERM and exits0;no independent process is killed.

Initial UI export at default save target returned recoverable create/no-overwrite
error;no successful exported file there is claimed. Model-free private export/
import and permissions pass. Actual Keychain credential was retrieved privately
for authenticated test clients and the same private key-file contract started
the standalone CLI. No token in argv,defaults,evidence,diagnostics or clipboard.

Real request client command (does not load a model):

```bash
python3 Scripts/test_managed_server.py --key-file /private/path/key --port 8080
```

UI test ports8080 initially,18094 after persisted setting/restart. Standalone test:

```bash
.build/release/QwenFieldfareServer --model /Users/tanishq/models/Qwen3.6-35B-A3B-4bit \
  --port 18094 --context 8192 --memory-gib 5 --queue-capacity 2 \
  --api-key-file /private/path/key
```

## Measured diagnostics and limits

| Real UI-owned request | Prompt/output tokens | Wall seconds |
| --- | ---: | ---: |
| JSON hello |17/1|0.925447|
| SSE hello |17/1|0.596157|
| get_project_status call |290/27|16.123924|
| SSE client-result continuation |347/16|17.324563|
| OpenCode single no-tool hello |210/1|11.062162 client wall|

The OpenCode server record:admission0.280ms,queue0.040ms,format0.050ms,tokenize4.377ms,
prefill9479.468ms,first generated9490.340ms,visible SSE flush9490.514ms,
decode-after-first44.260ms,overall9534.786ms. Tool buffering is implemented/tested
but unavailable in this retained last record (no tool in that request). It is
measured from recognized complete opening delimiter through validated commit;
only one bounded last-request record is retained. Values not observed remain
unavailable. Phase measurement definitions are in the integration guide.

These small requests are **not** the earlier full OpenCode workflow. Preserve its
28m17s/first content167.9s/first tool173.4s baseline;no speedup or broad bottleneck
claim. Core/prompt reset/replay behavior remains unchanged. No profiling flags.
HTTP provides no Bench cache/timing footer;only usage/wire and phase diagnostics
were captured,so no decode/cache-throughput benchmark is claimed.

Metal allocation5291294720 bytes (4.928GiB). UI combined RSS snapshot5052688KiB,
not physical footprint or exact peak. Independent CLI load footprint1156667344,
drained5141734112 bytes;runner retained until exit. Targets are Metal planner
budgets,not RAM caps. Additional controller state is bounded settings,one timing
record/counters,one recorder per admitted request and64KiB observer responses.
No extra model/expert cache is retained in parallel.

Initial integration release37.90s;corrected restart release36.97s. Final review
changed only safe provider-guidance generation/JSON escaping and tests;exact
OpenCode debug config/full suite/final36.68s release rerun. No expensive model
replay for that non-inference change. Tested/final binary identities are recorded.

Remaining risks:shared app/server crash fate,legacy fatal allocation/Metal paths,
non-immediate load/current-forward cancellation,cooperative same-UID ownership,
immutable-checkpoint/ancestor races,long-history/slow-client/soak coverage and
numerical acceptance. Current health schema needed for external observation;
older running CLI builds require restart with current executable. Native chat
has no server tool-frame transport/display because it is disabled in this mode;
OpenCode executes tools externally. No production-readiness claim. Stop S2-03;
next is a bounded measured repeated-turn investigation and exact hybrid prefix
state design,not an implementation started here.
