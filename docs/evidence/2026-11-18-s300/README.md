# S3-00: blocked exact-prefix candidate, safe replay restored

**S3-00 is not complete.** The candidate failed the principal same-Swift
raw-token equivalence gate and was reverted from production. No cache mode,
cache-clear endpoint or UI cache controls ship. S2-03 behavior remains intact.
The rejected implementation and failing tests are preserved as an exact patch.
No push or merge occurred. This is an investigation checkpoint, not a feature
promotion or production-readiness claim.

## Result and bounded attribution

Five alternating short tool-cycle pairs matched every raw generated ID and
finish reason: three sequential actual `audit_chunk` calls, client-supplied
public results, and a final DONE response. Each candidate cycle genuinely reused
1166/1280/1394 tokens and processed89/89/88 fresh continuation tokens. Across the
four requests, fresh prompt work was1407 versus replay5247 tokens.

The first sustained continuation then failed. Both modes received the same430
prompt IDs and produced512 greedy IDs with maxNewTokens finish. Candidate reused
399 consumed IDs. At zero-based output index465, full replay returned1031 and
candidate285. First tool setup355/44 agreed. The test failed after1332.333s and
stopped before the second sustained pair. No output/expected token was changed,
no failure was skipped, and the short-cycle gain cannot override this failure.
See [divergence](sustained-divergence.json), [raw records](raw-equivalence.json)
and [failed test log](raw-equivalence.log).

One bounded diagnostic tested a specific alternative to a state-boundary bug:
existing expert-residency history sensitivity. With prefix mode off, clear expert
weights, repeat the same fixed warmup, and fully replay the same430-token input.
All430 prompt tokens were processed. The512-token digest matched the prior
cached output and differed from the warm-expert replay control. Diagnostic
56.347s total, continuation34.048454s. See [result](residency-diagnostic.json)
and [log](residency-diagnostic.log). This reproduces full replay's history
sensitivity without conversation reuse; it does not prove the exact first
floating-point operation responsible. Existing hit/miss subset MoE reduction
and combination order is the code-level hypothesis. No kernel, precision,
quantization, sampler or expert-policy change was made to investigate or fix it.

The next safe task is a separately scoped design for residency-independent
expert reduction plus numerical/reference gates, or an independently reviewed
acceptance policy. Neither is authorized as an implicit fix to this mission.
Do not use recurrent rewind, expected-token edits or weaker equality to proceed.

## Paired short-cycle measurements — rejected candidate

Same release test executable, two mode settings, same token arrays, greedy,
8192 context,5GiB plan/2183 expert slots. Each sequence starts from an empty expert
cache plus the same17-token ready warmup. Mode order reverses each pair.
Times below sum generation and native-output validation over four turns; exclude
model load, formatting/tokenization, HTTP and client/tool execution.

| Pair | Order | Replay seconds | Candidate seconds | Prefill reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | off/exact | 194.042673 | 54.514489 | 72.991% |
| 2 | exact/off | 194.298296 | 54.628704 | 72.979% |
| 3 | off/exact | 193.686133 | 53.873989 | 73.260% |
| 4 | exact/off | 194.047324 | 53.947460 | 73.279% |
| 5 | off/exact | 194.356761 | 54.353940 | 73.124% |

Paired mean prefill reduction73.126% (Student-t df4 interval
72.950–73.303%); generation/validation reduction
72.042% (71.857–72.226%). These describe only this short
fixture/host. **No optimization is promoted**: sustained correctness failed.
[Full derived report](paired-report.json) includes each pair's logical fetches,
misses, reused/fresh counts and failure status. The reporter intentionally exits1
for failed/incomplete sustained qualification.

A separate first functional HTTP trial took54.005739s for the tool cycle versus
original193.193920s. This was one uncontrolled cross-run observation, not the
paired statistic above. See [HTTP candidate records](first-exact-tools/requests.json).

## Original replay baseline, before implementation

A frozen diagnostic-only server retained byte-identical Core/Bench/pins relative
to83d28e2. It only added actual-prefill, expert-fetch and footprint telemetry.
Full profile warmup+11 requests,total712.904308s,tool cycle193.193920s,exit0.
Calibrated supplemental profile total619.645961s,exit0. Rough initial labels did
not equal actual1K/4K counts, so the calibrated profile measured1026/4038/6144.
Model initialization/readiness is separate in each summary, not included in
per-request inference time. No GPU/subcomponent profiling flags were used.

| Profile/request | Actual prompt/output | Fresh prefill | Prefill seconds | First generated token seconds | HTTP wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline/single-turn | 30/1 | 30 | 1.229584 | 1.231658 | 1.274277 |
| baseline/two-turn-short | 47/2 | 47 | 1.532566 | 1.534816 | 1.621428 |
| baseline/approx-1k-first | 826/1 | 826 | 31.750659 | 31.756069 | 31.807194 |
| baseline/approx-1k-followup | 849/1 | 849 | 32.660623 | 32.665558 | 32.709374 |
| baseline/approx-4k-first | 3282/1 | 3282 | 129.645858 | 129.656618 | 129.705103 |
| baseline/approx-4k-followup | 3305/1 | 3305 | 127.234476 | 127.251176 | 127.290324 |
| baseline/longer-first | 5010/1 | 5010 | 189.909679 | 189.929013 | 189.980262 |
| baseline/tool-step-1 | 1141/25 | 1141 | 41.383756 | 41.391531 | 42.350120 |
| baseline/tool-step-2 | 1255/25 | 1255 | 46.032065 | 46.038692 | 46.968155 |
| baseline/tool-step-3 | 1369/25 | 1369 | 49.391328 | 49.398380 | 50.303122 |
| baseline/tool-final | 1482/1 | 1482 | 53.518983 | 53.526225 | 53.568284 |
| baseline-calibrated/calibrated-1k-first | 1026/1 | 1026 | 35.403092 | 35.408287 | 35.455087 |
| baseline-calibrated/calibrated-1k-followup | 1049/1 | 1049 | 36.584187 | 36.589618 | 36.625156 |
| baseline-calibrated/calibrated-4k-first | 4038/1 | 4038 | 150.121700 | 150.136136 | 150.186134 |
| baseline-calibrated/calibrated-4k-followup | 4061/1 | 4061 | 151.324398 | 151.337724 | 151.378032 |
| baseline-calibrated/longer-first | 6144/1 | 6144 | 241.160190 | 241.182941 | 241.237703 |

Complete per-request timing/cache/Metal/footprint records and decode-after-first
rates are in [original records](baseline/requests.json) and
[calibrated records](baseline-calibrated/requests.json). Decode-after-first
includes finishing/EOS/callback work; one-output-token requests have no meaningful
sustained decode rate. Logical fetched bytes/record reads are not physical SSD
traffic. OS file-cache state was not controlled. Do not infer a hardware ceiling.

**Recorder correction:** original earlier tool snapshots aliased a growing list,
as did the short single-turn snapshot; actual sent requests, usage, timings and
responses are intact. Preserve original corpus and
[workload source](baseline-workload-original.py.txt). Reconstruct tool messages
at2/4/6/8 boundaries from the final public history in
[ordered request-only corpus](reconstructed-tool-requests.json); guards verified
1141/1255/1369/1482 token counts. The live recorder now deep-copies. Do not sort
schema fields through Foundation: ordering affects checkpoint template tokens.
The first invalid-fixture attempt and a strict-JSON rejection of unrelated
telemetry before model loading remain archived; neither is numerical evidence.

## State contract and candidate ownership

The [consumed-state contract](../../EXACT_PREFIX_STATE_CONTRACT.md) records the
candidate's invariant: EOS itself is unconsumed; normal EOS represents every
returned non-EOS token, limits leave a pending final token. Exact reuse needs the
whole represented sequence as a strict prefix of new actual prompt IDs, not an
LCP rewind. Ordinary chat cleans earlier thinking framing and often misses;
canonical same-user tool continuations can match. There is no recurrent rollback.

Candidate used one context-bounded CPU token vector, existing KV/Conv/DeltaNet
buffers in place, no GPU snapshot/model copy/multi-session map. Metadata staging,
normalEOS/decoder/tool validation and complete terminal flush precede publication.
A lock/epoch clear blocks in-flight commits without touching live GPU state;
errors, cancellation, context rejection and shutdown invalidate. Expected close
after complete flush is not an error; flush is not remote receipt. These were
implemented/prototyped, then reverted, not shipped capabilities.

## Host, memory, identity and limitations

MacBook Pro Mac17,2,Apple M5,10 GPU cores,32GiB unified RAM;macOS27.2.1(26B131),
Swift6.4.2,arm64.134GiB disk available. Checkpoint
/Users/tanishq/models/Qwen3.6-35B-A3B-4bit,existing identity/integrity evidence
reused;required loader checks stayed enabled. No weight copied,committed or
rehashed unnecessarily. Base83d28e28745d43000ad7e732e8ba2a069591575e.
[Experiment identity](experiment-identity.json) records source/test executable
and frozen-control SHA256. Candidate's tensor-forward body stayed identical
([proof](forward-body-proof.json)); changing skipped work still required the
real equivalence gate and it failed.

Original baseline free85%,swap279.69MiB; initial numeric pressure unavailable,
not inferred from later readings. Corrected paired preflight free86%,pressure1,
swap279.69MiB,AC attached80%,low-power mode0 observed earlier. Observed paired/diagnostic preflight and mid-run checks showedpressure1 and
279.69MiB swap. Final restored-tree postflight pressure1,free87%,swap271.69MiB
(lower);no swap increase at observed checkpoints. No
competing model owner; each run drains/exits before the next. Metal observed
5291294720bytes(4.928GiB) in both modes; maximum observed harness footprint
5972403752bytes(5.562GiB),not a peak claim. Maximum token element capacity10208
bytes,plus fixed header/lock; no additional Metal snapshot. Logical fetches and
whole-request misses are available,physical SSD effects unavailable. No exact
hidden-state bitwise comparison was made.

Common-protocol limitations: no second sustained pair or per-bucket512 promotion
report after the first correctness failure; no final UI/cache/OpenCode or enabled
calibrated-history gate after rejection. Existing S2-03 UI/client qualification
remains preserved. Further expensive runs cannot turn the failed gate into a pass.

## Reproduction and restored-tree validation

Original baseline command (frozen binary is ignored/local only):

```bash
QWENFIELDFARE_SERVER_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit \
  python3 Scripts/benchmark_prefix_reuse.py --phase baseline \
  --server .build/release/QwenFieldfareServer.s300-control
# Supplemental: --phase baseline-calibrated --profile calibrated
```

The exact rejected [candidate patch](rejected-candidate.patch.gz) targets fresh
83d28e2,not the current tree with retained boundary tests. Its reverse-apply check
passed before restoration; [archive identity](candidate-archive.json) records SHA.
Decompress the archive before `git apply` on that fresh base. Its uncompressed
SHA is recorded separately; no patch line was altered. It contains production prototype and all candidate tests,including the failing
real test and bounded diagnostic. Do not apply it as a qualified feature. With
that candidate on a disposable83d28e2 checkout plus this evidence,build release
harness with env unset,preflight one owner/normal pressure,then run:

```bash
QWENFIELDFARE_PREFIX_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit \
QWENFIELDFARE_PREFIX_OUTPUT=/absolute/checkout/docs/evidence/2026-11-18-s300/raw-equivalence.json \
QWENFIELDFARE_PREFIX_ROUNDS=5 \
  Scripts/test.sh -c release --skip-build --filter RealPrefixEquivalenceTests
# Expected historical result: failure,not a qualified cache.
# Bounded diagnostic instead:
QWENFIELDFARE_PREFIX_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit \
QWENFIELDFARE_PREFIX_RESIDENCY_DIAGNOSTIC=1 \
  Scripts/test.sh -c release --skip-build --filter PrefixResidencyDiagnosticTests
```

Final production Sources,Package.swift and pins are restored byte-for-byte to
83d28e2. Retained new model-free coverage: generation consumed-sequence/absolute
and sampling positions atEOS/output/context boundaries; two template-framing
tests distinguishing ordinary thinking cleanup from possible tool prefixes.
Final restored package/release/checker/docs/diff results are recorded below. Candidate model-free successes are not substituted for failed raw
qualification. The original strict MLX131/132 fixture/mismatch remains intact;
archived-log checker success is not a new numerical replay or equivalence proof.


Final restored gates:

- `Scripts/test.sh`:153 passing,3 opt-in checkpoint skips,156 total;32 Server,
  10 Management,17 Presentation,97 Core including skips;debug build11.41s.
- `swift build -c release`:exit0,41.88s.
- Numerical checker tests6/6;original archived131/132 log stability passes.
  No fresh MLX replay,no numerical equivalence claim.
- Markdown30files pass;`git diff --check` passes;full production Sources,
  Package.swift and pins byte-identical to83d28e2;reference fixture SHA unchanged.
- No model owner remains;postflight pressure1/free87%,swap271.69MiB.

Logs: [package](restored-package-tests.log),[release](restored-release-build.log),
[checker](final-checker-tests.log),[archived log](final-known-log-check.log),
[docs](final-doc-links.log),[source/fixture identity](restored-source-identity.json),
[postflight](restored-postflight.json). This validates the restored safe tree;
it does not convert the rejected candidate's failed test into a pass.
