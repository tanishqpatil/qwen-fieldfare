# S2-00 local server evidence

Base cf0d144. Existing reader safety, model weights, reference fixture and
numerical sources preserved. New NIOHTTP1/Posix executable plus small protocol/
admission/backend support target. Existing resolved SwiftNIO2.101.3 is declared
as a direct exact dependency; no dependency versions upgraded.

Model-free suite: six server tests (request validation, HTTP/JSON/SSE, auth,
loopback rejection, saturation/shutdown, disconnect cancellation and failed init),
plus existing package tests.113 passed,three model opt-ins skipped,116 reported.
See package-tests.log; release-build.log records final build; checker-tests.log
records six numerical-checker tests. No tests or parity expectations weakened.

Reproduction: follow preflight in AGENTS, then release build and
`QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/model python3 Scripts/test_server_http.py`.
Server on127.0.0.1:18089,context8192,memory5GiB,temporary randomly generated bearer
key kept only in environment/memory. No prompts/responses/secrets saved. The
script saves aggregate timings/counts and safe server load/drain memory snapshots.
All model requests reuse one Core runner and reset/replay state.

Initial real HTTP run passed JSON/SSE, repeat requests, context overflow and
client-disconnect recovery; shutdown trapped because a Dispatch signal callback
inherited MainActor isolation. Initial JSON/log evidence retained. The callback
is now installed by a nonisolated function; final result is recorded separately
in http-integration.json and server.log. Raw macOS crash report was not copied
because it contains device identifiers. See shutdown-diagnosis.txt.

Metrics are smoke-test observations, not performance promotion: first/repeated
request times include existing OS-cache/Metal warmup effects. Loaded/drained
snapshots are not peak memory; drained runner is retained until process exit.
No full-context, long-agent, default sampled quality, tools, OpenCode or Hermes
qualification. SIGTERM drains one synchronous in-flight forward before cancellation.
Loading is not immediately cancellable. Output writes have10s timeout. No cross-
application owner lock; the server's flock excludes another server only.

Retained strict active teacher evidence remains131/132,exit1,token61 expected14988
actual10000. Existing checker uses the S1-01c historical log, not a fresh parity
run. HTTP generation does not establish numerical equivalence. Kernels, samplers,
precision, cache policy, checkpoint and original fixture are unchanged.

New functional API and limits: docs/AGENT_INTEGRATION.md. Next S2-01 tool contract;
no further reader migration, tools, prefix reuse or push/merge in this milestone.

Final full HTTP run passed (exit0): model load0.897s,health-ready1.367s; first/repeat
JSON1.053/0.680s,23 prompt +2 completion tokens each,stop. SSE0.675s,first content
0.615s,5 events +DONE. Context rejection/disconnect recovery pass; shutdown during
request0.153s. Loaded/drained footprint1,157,519,336/5,143,274,256 bytes; Metal
5,291,294,720 bytes (4.928GiB) under5GiB target. Normal postpressure1/free80%.
Initial failed-run metrics remain alongside final results; no optimization claim.
