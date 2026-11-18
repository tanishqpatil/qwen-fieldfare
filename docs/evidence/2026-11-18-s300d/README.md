# S3-00d exact-prefix qualification over fixed router order

The candidate reuses one context-bounded consumed-token prefix in the
existing server runner. The default remains off. See the
[state contract](../../EXACT_PREFIX_STATE_CONTRACT.md) and
[historical rejection](../2026-11-18-s300/README.md).

## Reproduction

Host Mac17,2, Apple M5 (10 GPU cores), 32 GB unified memory, macOS 27.2.1
(26B131), Swift 6.4.2. Base `7370a1d43d5c50d7fb4356f7651f50a4feeae47c`.
Fixed router order and strict reference fixture are unchanged. All runs
use one model owner, the same compatible MLX Q4 checkpoint, raw Safetensors,
8,192 context, 5 GiB plan and 2,183 expert slots. Close other model processes
and verify normal pressure before running. Default loader checks remain enabled.
The checkpoint identity came from the recent verified strict run.

```sh
Scripts/test.sh
swift build -c release
QWENFIELDFARE_PREFIX_CHECKPOINT=/absolute/model \
QWENFIELDFARE_PREFIX_ROUNDS=1 \
QWENFIELDFARE_PREFIX_OUTPUT=/private/tmp/real-equivalence.json \
  Scripts/test.sh -c release --filter RealPrefixEquivalenceTests
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/model \
  python3 Scripts/qualify_exact_prefix.py --out /private/tmp/s300d --phase all
QWENFIELDFARE_SERVER_CHECKPOINT=/absolute/model \
  python3 Scripts/qualify_exact_prefix.py --out /private/tmp/s300d-off --phase opencode-replay
python3 Scripts/test_numerical_non_regression.py
python3 Scripts/check_numerical_non_regression.py \
  --log docs/evidence/2026-11-18-s300c/production-strict.txt
ruby Scripts/check_markdown_links.rb
git diff --check
```

## Same-runtime raw-token gate

[Direct runs](real-equivalence.json) use one runner per invocation. Each mode
starts with an empty expert cache and the same fixed warmup. The sustained
gate runs first and reconstructs the original 355/44 tool setup and
430/512 continuation from the fixed public fixture. The raw predicted IDs,
finish reasons and actual continuation prompt IDs match. Reuse processes
31 fresh prompt IDs after the 399 consumed-prefix IDs. At output positions
0, 216, 421, 464, 465, 466 and 511, SHA-256 matches for every full 248,320-logit
and 2,048-element pre-norm hidden array. Index 465 maps to hidden/logit state
after position 894. The full output digest also matches the earlier archived warm full replay. The 512-output result ended at the limit and was
ineligible to publish; its final returned token was pending.

The four archived short turns produce three sequential `audit_chunk`
calls and final `DONE`, with matching raw output IDs and finish reasons.
Actual prompts contain 1,141 / 1,255 / 1,369 / 1,482 tokens. Three continuation
hits reuse 1,166 / 1,280 / 1,394 tokens and prefill 89 / 89 / 88 fresh IDs.
Total prefill is 214.40 s (5,247 IDs) for replay and 61.66 s (1,407 IDs) for
reuse. Total measured generation plus output validation is 217.78 s and
65.33 s, a 70.0% reduction for this four-request pair. The sampled sustained
gate instrumentation adds hash work and is excluded from headline timing.
The test invocation took 377.356 s including loads and warmups.
[Gate log](real-equivalence.txt)

Metal allocation was 5,291,360,256 bytes for both modes. Peak sampled process
footprints in the short pair were 5,337,523,296 replay and 5,370,766,456 with
reuse. Token element storage measured up to 10,208 bytes on the short
fixture. Process footprint includes runtime and OS effects. Timing pairs
measure this host and fixture with uncontrolled OS page-cache history.
Throughput in the per-request records includes application overhead and
has no GPU profiling enabled. Logical expert reads are not SSD bytes.

The raw-token gate ran before the final error-handling and diagnostics
edits. Those edits preserve successful forward arithmetic and are covered
by the rebuilt CLI lifecycle run and the complete package suite.

The full package run recorded 171 passes and four opt-in skips. Final release
build took 42.66 s. All six numerical checker unit tests passed. The checker
accepted the existing checkpoint-verified [strict log](../2026-11-18-s300c/production-strict.txt).
Fixture identity and the sole known mismatch at reference token 61,
expected 14988 and got 10000, remain unchanged at 131/132. This check of
archived evidence did not run strict inference again.


## Real CLI lifecycle

On the rebuilt executable, JSON and SSE completed real tool continuations
with terminal delivery. The SSE continuation reused 1,166 tokens. Changing
an earlier tool result forced a 1,375-token full replay with the same
validated call and usage as the cleared control. A 400 context rejection
invalidated the previous ledger. The real server processed an active
generation and queued successor serially, cleared the active epoch,
cancelled a disconnected queued client, cancelled an active disconnect,
and drained on SIGTERM during generation. A restart with the flag omitted
came up empty in mode off with a new runner identity. The detailed
[loopback records](http-qualification.json) include only aggregate metrics.


## Repeated continuation latency

Each pair compares the same final HTTP request and validated tool call.
The cache-hit request follows its live setup. The full-replay request runs
on the same retained runner after explicit metadata clear and processes
every prompt token. This order avoids rebuilding each large setup twice.
The default-off comparison also ran independently in the raw-token gate.
One pair per size limits redundant long-prefill work. These HTTP observations
describe synthetic tool continuations on this host.

| Continuation | Prompt / output | Reused / fresh | Prefill, replay / reuse | First generated token, replay / reuse | HTTP wall, replay / reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approximately 1K | 1,255 / 25 | 1,166 / 89 | 49.04 / 3.64 s | 49.04 / 3.65 s | 49.96 / 4.65 s |
| Approximately 4K | 4,116 / 25 | 4,027 / 89 | 159.40 / 3.91 s | 159.42 / 3.93 s | 160.48 / 5.09 s |
| Longer 8K configuration | 6,114 / 25 | 6,025 / 89 | 242.71 / 4.13 s | 242.73 / 4.16 s | 243.88 / 5.31 s |

| Continuation | Decode tokens/s, replay / reuse | Expert hits, replay / reuse | Expert misses, replay / reuse | Sampled process footprint, replay / reuse |
| --- | ---: | ---: | ---: | ---: |
| Approximately 1K | 26.22 / 23.95 | 270,711 / 24,912 | 138,889 / 11,568 | 5.185 / 5.182 GB |
| Approximately 4K | 22.58 / 20.64 | 835,102 / 25,039 | 490,018 / 11,441 | 5.164 / 5.161 GB |
| Longer 8K configuration | 20.87 / 20.91 | 1,222,260 / 25,165 | 742,220 / 11,315 | 5.222 / 5.219 GB |

The [loopback records](http-qualification.json) contain exact byte counts
and the full cache, timing and swap measurements. Metal stayed 5,291,360,256
bytes. The largest reported token-array element capacity was 49,120 bytes
on the long pair, including Swift Array growth capacity. Host memory pressure at each sample was 1. Used swap
decreased from 1,998.50 MiB at performance-phase readiness to 1,140.06 MiB
after shutdown. Host swap includes other applications.


## Native macOS server

A disposable, ad-hoc signed app wrapper used the same SwiftPM release
build and a separate preferences domain. The initial wrapper launch
failed because SwiftPM expected its resource bundles at the wrapper
root. Adding those resource links fixed the harness without a source
change. The wrapper tests UI behavior on this Mac.

The settings sheet showed off on first launch. Selecting Exact prefix
persisted after Quit and reopen, which remained Stopped. Start brought
up the real authenticated in-process server. The 430-token continuation
reused 399 IDs and processed 31 fresh IDs. A subsequent EOS request
published a valid ledger. Clear Prefix Cache invalidated it and released
its token storage without restarting. Restart came up empty with a new
session. Stop closed the listener and restored the native chat UI.
The [native records](native-qualification.json) contain aggregate
health evidence and both build and app wrapper hashes.


## Real OpenCode workflow

OpenCode 1.19.5 read the disposable repository's calculator module,
edited its implementation and ran the allowed test command. Both modes
completed all three tools and a final response, and the independent
unittest passed. Each run used the same release server binary, task,
provider settings and tool permissions. The two temporary repositories
have different absolute paths and the actual token counts differ. Treat
this as one paired behavioral and wall-time measurement. The separate
raw-token gate supplies the fixed-input equivalence proof.

| Mode | Workflow wall | Prompt tokens by turn | Reused tokens by turn | Fresh tokens by turn |
| --- | ---: | --- | --- | --- |
| Full replay | 567.13 s | 3,006 / 3,192 / 3,319 / 3,400 | 0 / 0 / 0 / 0 | 3,006 / 3,192 / 3,319 / 3,400 |
| Exact prefix | 154.40 s | 3,008 / 3,196 / 3,324 / 3,405 | 0 / 3,080 / 3,302 / 3,355 | 3,008 / 116 / 22 / 50 |

Wall time fell 72.8% on this workflow. First-request runtime was
135.12 s on replay and 136.99 s with reuse. The benefit accumulated on
later requests. Both trials sampled normal pressure and 961.12 MiB of
used swap before requests and after shutdown. The qualification records
contain aggregate timings, reuse counters and tool names. The first trial was interrupted
after starting the server and produced no workflow result. The resumed
paired runs use [cache on](opencode-exact.json) and [cache off](opencode-replay.json).
