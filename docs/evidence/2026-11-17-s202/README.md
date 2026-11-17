# S2-02 — CLI and actual OpenCode qualification

Completed the bounded standalone CLI/OpenCode milestone. S2-03 native UI is not
started. Base c2b16632baefb3aba64136a5b8e03e426baf8c5a plus the uncommitted S2-02
working tree; no push/merge. This is one disposable coding workflow, not broad
agent quality, production readiness, long-context or performance qualification.

## Changes and model-free gates

Validated configuration/help/version; explicit loopback host,port0–65535,
model/context/budget/queue0–8; env OR private owned regular key file with bounded
no-follow descriptor reads and redacted errors. Both Q4 runners now hold a
cooperative RAII per-user owner lease before allocation; app/Bench/server cannot
silently compete. Persistent lock file must never be unlinked while owned;
uncooperative/same-UID actors and other inference products are outside that guard.
No UI controls or second model service were introduced.

Installed OpenCode wire fixture adds exact2020-12 schema metadata/numeric bounds;
unsupported constraints still reject. Adjacent Int64 bound tests prevent Double
rounding from weakening validation. No inference mathematics/template/cache policy
changed. Existing S2-00/01 protocol/parser/transport tests remain intact.

Focused6 configuration tests +3 lease tests pass (initial filter also includes4
existing memory-configuration tests). Final full package136 pass +3 opt-in skips
(139 total:26 Server,17 Presentation,96 Core including skips). Release before
model35.31s and final35.35s pass. Six numerical-checker tests pass; archived log
known-mismatch check passes separately. Markdown/diff checks are in their logs.

## Exact real-client setup

Command, after release and AGENTS preflight:

```bash
QWENFIELDFARE_SERVER_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit \
  python3 Scripts/test_opencode.py
```

Mac17,2/M5/32GB/10GPU,macOS27.2.1(26B131),Swift6.4.2. Initial preflight133GiB disk,
normal pressure1/free84%; retry preflight free82%,normal1,no competing model. End
free80%,pressure1. Checkpoint stays external; no weights copied/staged or full
checkpoint hashing repeated. Earlier verified checkpoint identity is reused;
required normal loader/index/extent validation remains enabled.

One server127.0.0.1:18092,context8192,Metal target5GiB,queue2,private key file; no
secret argv. Load3.806333s. Byte-preserving client-side relay18093 records aggregate
SSE timings/usage only; it neither edits payloads nor executes tools. Direct provider
example uses8080/v1. OpenCode1.19.5/Bun1.4.4,installed v1 provider/npm/options/agent
contract; current website v2 differs. Isolated XDG paths,canonical PWD/--dir,
--pure,disabled external skills/Claude instructions/models fetch/autoupdate/LSP
fetch/autocompaction;no delegation/remote model. Provider listing is configured
catalog listing, not a claim of server discovery. Actual requests select the model.

Client allows workspace read/grep,edits only calculator.py/test_calculator.py,
and exact shell command python3 -m unittest -v;external_directory denied. Patterns
are workspace-relative. Client permissions are not an OS sandbox. All execution
is in OpenCode, never the inference server. The saved provider example asks for
edit/bash approval instead of automatically allowing this fixture command.

## Actual result, latency and retries

8 real Chat Completions SSE requests,7 completed tools,0 tool errors in the
successful run;first7 finish tool_calls,last stop. Actual order:

1. Read calculator.py (subtraction bug).
2. Grep related Python add calls.
3. Read test_calculator.py (deliberate add(2,2)==5 expectation).
4. Run unittest:2 failures.
5. Edit implementation to a+b.
6. Repair incorrect expectation to4.
7. Run unittest:2 passes.
8. Final assistant report,client exit0;independent final tests also pass.

Sequence deviation: the requested prompt asked for implementation edit before the
first test. The model instead inspected both files and ran the failing tests first,
then repaired both. This still exercises real failure diagnosis/repair, but the
strict requested ordering was not followed. No fabricated tool result or model
response was substituted. Public synthetic events/final files are retained.

Workflow wall1696.730207s (28m16.73s),excluding startup/error/cancellation probes.
First request3410 prompt/118 native output tokens;final5233/155. Across replayed
requests35055 prompt/1079 native output tokens. Per-request first content latency
167.929–244.235s;initial first content167.929377s,first tool173.367516s. Delayed
complete tool calls arrive173.368–241.219s after their respective request starts.
The initial extra5.438s between first prose and tool commit is visible; calls are
still held through normal EOS/complete native validation. OpenCode accepts this
behavior with600s SDK timeout. Full prompt replay dominates this workload; no
prefix reuse,heartbeat or optimization claim. Details in requests.json.

One earlier real attempt stopped after631.682666s with0 successful tools and3
client permission errors:directory read,forbidden ls,source-file read. Bad absolute
allow patterns caused valid reads to deny. Its complete artifacts remain under
attempt-1-denied. A scripted/model-free rehearsal proved corrected permissions
(8 allowed tools including read of config;3 explicit outside-read/config-edit/
unauthorized-shell denials) before one corrected real retry. This mock rehearsal
is never counted as model qualification. No further retry was needed.

## Lifecycle/error/memory results

Actual OpenCode output limit5000 gives non-retryable HTTP400 "Output limit must
be1–4096";oversized prompt returns HTTP400 "Prompt plus output reservation exceeds
context". SIGINT after admitted client request exits-2;server active/queued counts
drain to0 in0.542347s and the subsequent workflow succeeds. This cancellation is
an early admitted prefill probe, not exhaustive load/decode/queued fault coverage.
Competing Bench process exits1 with owner error before loading another runner.
Final SIGTERM server drains/exits0.

Metal allocation5291294720 bytes (4.928GiB). Loaded physical footprint1157421032
bytes;shutdown drained footprint5230895984 bytes (runner still retained). Every2s
sampled peak RSS5087821824 bytes;RSS is not physical footprint or an exact peak,
and excludes total client/system costs.5GiB is a Metal planner target,not a RAM cap.
Normal system pressure retained. Process exit releases model/lease. HTTP has no
Bench timing/cache footer;only usage/timing/process snapshots were collected,so
no prefill/decode/cache-throughput benchmark is claimed. Profiling was disabled.

## Numerical and remaining limits

Original fixture SHA8d552a906d417ceb492ddd045a50e67b0e9630988d9665e8c6fe5e480cb30747
verified (small fixture only);teacher harness/expected tokens/dependency pins are
unchanged. Tracked Core diff is two lease lines per Q4 runner plus new resource-only
lease source;kernel/state/sampling/precision code unchanged. The final bound
precision correction affects values outside the actual safe-integer OpenCode
operands exercised here;prompt formatting/schema text are unchanged. Focused/full/
release gates rerun;costly real workflow was not repeated for this isolated fix.

No new strict parity replay was performed. Historical131/132,token61 expected14988
actual10000 remains an unresolved release blocker. Archived checker/log stability
is not new numerical equivalence evidence;the existing unchanged-source shortcut
is not asserted for added Core ownership/manifest files. Original Stage0/readers
and reference artifacts are untouched. Independent numerical acceptance review,
legacy fatal allocator/Metal paths,immutable-checkpoint/ancestor races,long-context/
long-agent/soak,Hermes and native server UI remain open. Next safe task:S2-03 UI
ownership/controller design. Stop here;no prefix reuse or additional readers.

Aggregate/test .log files are retained locally but matched by repository *.log
ignore rules. A later scoped checkpoint must explicitly include the reviewed
evidence logs; no Git ignore policy was broadened and no commit was made here.
