# Development log

Append-only journal. Corrections go in new entries; do not rewrite old outcomes.
Task status lives in [TODO](../TODO.md), recovery index in [memory](AGENT_MEMORY.md).

## 2026-11-17 — Stage 0 foundation and correctness stop

Branch `feature/local-coding-agent-server`; base commit `72fb923`; clean starting
checkout. No runtime/test/kernel/dependency modifications; no commit or push.
The attached implementation mission authorizes Stages 0–1 and requires stopping
at an unresolved correctness issue. Current work stops in Stage 0, S0-03.

### Audit and decisions

Read AGENTS, README, Package/lock, contribution/security docs, runtime controls,
agent/runtime/product/optimization plans, active-streaming history and benchmark
protocol. Inspected model-directory and resident loaders, runner initialization,
mmap/slice/state/generation paths, generation support, expert caches/stores,
ChatService lifecycle, attention bridge/kernels, CPU diagnostic scaffolding,
tests, CI and reference/parity scripts. This is a source audit, not proof of
complete runtime safety or numerical correctness.

Findings: actual four-shard mmap readers bypass the resident loader's bounded
header checks; model-contract preflight incomplete; forced buffer allocations
and fatal encoder/command creation remain; loading lacks comprehensive cancel
boundaries; visible and committed history can diverge after failure/cancel.
KV/Conv/DeltaNet are reset per generation; no retained conversation state.
3/4/5 GiB controls and teacher-reference CLI already exist. Old claims that the
harness is absent and memory is fixed at 5 GiB are stale. Current full attention
uses the occupied prefix, not sliding-window history eviction. No server or
typed tools exist. No code fixes were attempted before establishing the gate.

Created TODO, ARCHITECTURE, DEVELOPMENT_LOG, AGENT_MEMORY, DECISIONS,
PERFORMANCE_BASELINES and AGENT_INTEGRATION. Updated AGENTS, README and current
status overlays in runtime/product plans and controls; historical records kept.
Saved public-prompt reference and logs under `docs/evidence/2026-11-17-parity/`.
Only hashes/token IDs/public benchmark content are saved; no model weights.

### Commands and exact outcomes

- `git status --short`, `git branch --show-current`, `git log -5 --oneline`:
  clean main at 72fb923. Initial sandbox branch creation failed writing refs;
  expanded-access `git switch -c feature/local-coding-agent-server` succeeded.
- `sw_vers`: macOS 27.2.1, 26B131. `swift --version`: Apple Swift 6.4.2,
  swiftlang-6.4.2.1.9, clang-2100.4.1.1, arm64 macOS 27.2 target.
- `system_profiler SPHardwareDataType`: MacBook Pro Mac17,2, Apple M5, 32 GB,
  10 CPU cores (4 performance/6 efficiency). Device identifiers not persisted.
- `df -h .`: 136 GiB available. `memory_pressure -Q`: 85% system free before
  reference generation. Model directory present; no weights staged/modified.
- `pgrep -ifl 'QwenFieldfareMac|QwenFieldfareBench|mlx_lm|mlx-lm|ollama|LM Studio'`:
  no match, exit 1. MLX and Swift model runs were sequential.
- `Scripts/test.sh`: initial sandbox attempt blocked on Swift cache writes;
  expanded-access rerun exit 0, Core 32 tests/8 suites, Presentation 17 tests/3
  suites; 49 total. Metal bridge/cache tests ran, not skipped. No model required.
- `swift build -c release`: expanded-access exit 0, 41.68 s, all products built.
- System `python3` lacks MLX. Located an existing environment at
  `$HOME/venvs/mlx-dspark/bin/python`; package metadata reports MLX 0.32.4,
  mlx-lm 0.32.2, Transformers 5.22.2. No environment/dependency modified.

Commands below used the existing local checkpoint as MODEL_DIR and that Python
as MLX_PYTHON; generated temporary fixture/logs were later preserved in evidence:

```bash
"$MLX_PYTHON" Scripts/qwen_mlx_greedy_reference.py \
  --model "$MODEL_DIR" --prompt 'Explain HashMap in Java with an example.' \
  --max-tokens 132 --fixture /tmp/qwen-stage0-reference.json
.build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --cache-slots 2183 \
  --teacher-reference /tmp/qwen-stage0-reference.json
```

MLX generator exited 0: 21 prompt/132 output tokens, length stop, greedy,
8,841.21 ms whole generation including prefill (14.93 tok/s), reported peak
19.615 decimal GB. Swift exited **1**: checkpoint SHA-256 verified, **131/132**
teacher-forced predictions matched. One mismatch at generated reference position
61 (one-based): expected 14988, actual 10000. Loaded Metal 4.928 GiB, loaded
footprint 0.996 GiB, 0.124 s load; these are loaded snapshots, not request peaks.
Context 8,192, raw storage, 2,183 slots, profiling disabled. The teacher mode
returns before the ordinary timing/cache footer, so full latency/cache/memory
qualification is unavailable. Token positions consume reference inputs rather
than free-running predictions. Comparison used all 132 expected positions.

Protocol deviations/limits: this is a correctness run, not the community
three-repeat throughput protocol; hashing warms OS file cache, no retained
warmup, raw cache-slot override instead of budget selection, no physical SSD
traffic measurement, no pre/post swap or peak-footprint series, no GPU core count.
No speedup, ceiling, memory-regression or passing numerical baseline is claimed.
The historical oracle had different versions; whether that explains the mismatch
is **unknown**. Historical 132/132 remains historical evidence only.

### Boundary and next action

S0-01 and S0-02 complete; S0-03 blocked on current parity. Stage 0 is not complete.
Stage 1 not started. Do not run later performance/integration stages or treat
mock success as parity. Six-case teacher test not run because valid case already
fails. Next: use saved checkpoint-hashed reference to investigate position 61,
compare top logits and reference numerical settings with a reproducibly pinned
oracle. Preserve failure, do not relax assertions or change tokens to hide it.
The requested stop at unresolved correctness takes precedence over proceeding
with reliability edits. Stage 1 begins only once this baseline boundary clears.

### Foundation verification before handoff

`ruby Scripts/check_markdown_links.rb`: exit 0, all local links/anchors in 18
Markdown files resolve. `git diff --check`: exit 0. Saved fixture JSON checked:
21 prompt IDs, 132 reference IDs, position 61 is 14988. Backlog contains 34
stable tasks across all ten mission stages, each with status, priority, source,
dependencies, acceptance, tests and risks. Inspected tracked diff and new docs;
no source/test/dependency change. Evidence scan found no absolute home paths,
device serial/UUID/UDID, credentials or model weights. Added a current erratum
to the historical streaming log rather than rewriting historical measurements.
All changes remain uncommitted on the feature branch; no remote action taken.


## 2026-11-17 — Follow-up Stage 0 correctness investigation (S0-03)

User scope: investigate/resolve or characterize the current failure; **do not
begin Stage 1**. Branch/base unchanged; all existing foundation edits preserved.
[Full report](evidence/2026-11-17-investigation/REPORT.md) records exact scores,
position arithmetic, identity, commands, selected activations and limitations.

### Reproduction and identity

Read AGENTS, TODO, recovery memory, this log, baseline and historical agent plan;
inspected teacher loop, argmax, embedding, final norm/head, MoE reduction and
installed MLX model/generator/quantized layers. Original fixture SHA-256 remains
8d552a906d417ceb492ddd045a50e67b0e9630988d9665e8c6fe5e480cb30747. Fixture and
installed versions match (MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.22.2).
Checkpoint hashes verified in every Swift replay; new fixtures preserved under
separate names. Public prompt/history/template match exactly: 21 prompt tokens,
132 teacher targets. Failure at reference index 60 consumes ID 332 at zero-based
absolute input position 80 and predicts position 81. Greedy, no penalties.

Hardware/toolchain: same M5/32 GB, macOS 27.2.1/26B131, Swift 6.4.2,
base 72fb923a3e7a5d43e5baeb43d35bdcf1f96e4966. Disk initially 134 GiB free.
Runs were sequential with model process exclusions and memory-pressure query
before each. Reported free memory ranged 74%–87%; final validation driver also
required kern.memorystatus_vm_pressure_level=1 and >10 GiB disk free. No weights,
packages or dependencies changed. Local reference-source hashes saved, not a
claim that the historical oracle environment has been reconstructed.

Two separate uninstrumented controls each exit 1, 131/132 at position 61.
Read-only diagnostic binary: two reset/replay passes in one retained runner for
each 5 GiB and 3 GiB cache plan; every pass exits parity failure at that position.
Two selected-layer passes do likewise. Final restored-source run again fails
131/132. Total nine full investigation replays, same single mismatch/prediction.
Warm expert cache changes exact scores slightly, never the winner in these runs.
Two independently generated MLX fixtures exactly equal the saved original;
MLX Top-5 and final normalized hidden vector agree between those repetitions.

### Narrow instrumentation and findings

Temporary SPI readback of completed raw logits/final normalized hidden plus an
environment-gated teacher diagnostic; second variant captures only layer outputs
0/3/19/39 at position 80. No math/kernel/sampler change. Layer captures add waits;
first-pass Top-5 exactly matches the untraced run. Archived patch, scripts, raw
scores/vectors, command logs, comparison.json and provenance.json preserve evidence.
No runtime instrumentation remains applied.

MLX BF16 raw Top-5: 14988 27.875; 10000 27.75; 20480 23.75; 2070 19.25;
3436 18.625. Swift initial 5-GiB FP32: 10000 27.845386505; 14988 27.724899292;
20480 23.785346985; 2070 19.365528107; 3436 18.732065201. Same score quantity,
different precision. Signed expected-minus-actual margin +0.125 vs -0.120487213;
MLX gap is one BF16 ULP here. No exact tie in either, so tie-breaking is not the
cause. Swift winner gaps across budget/warm passes: 0.120487213, 0.114616394,
0.102859497, 0.115917206. Possible hit/miss reduction-order contribution is
supported by source, not proven as the complete explanation.

Same loaded MLX head on identical Swift hidden reproduces five scores within
1.9073486328125e-6 (the two disputed scores exactly agree). Common FP32 head on
MLX hidden still favors 14988, on Swift hidden favors 10000. Final hidden
relative L2 differs 2.272128%, max absolute 0.266634464; layer-0 output already
0.438729% relative L2. Selected layers 3/19/39 have 2.476826%/4.878141%/2.280052%
relative L2. Earliest demonstrated precision boundary: input-332 embedding,
262/2048 values differ; rounding the Swift FP32 formula to BF16 exactly equals
MLX dequantize. This does **not** prove embedding rounding alone explains the
whole discrepancy or that no Swift defect exists. No large all-layer tensor dump.

One temporary MLX wrapper attempt failed before generation due to missing
input_layernorm attribute forwarding; exited 1, no fixture created. Fixed the
wrapper only, not model code. Successful rerun reproduces the original sequence.
Failure log retained. No attempt or result hidden.

### Changes, validation and interpretation

Removed all temporary Core/Bench edits from exact pre-investigation copies.
No Sources/Scripts/Tests/Package diff remains. Restored release benchmark SHA-256
f55a81ce3041544c5cd811a1c9acbc04abcb302f8b40a29fd67d2d0af2854c32 equals initial
executable exactly. Release build exit 0 (16.54 s); Scripts/test.sh exit 0:
17 Presentation plus 32 Core tests (49 total). These are validations, not speed
measurements. No concrete production fix identified; no targeted defect regression
invented. User's requested fix validation cannot be called satisfied without a fix.

The restored binary's valid fixture still exits 1 (131/132). Five existing
negative fixture cases independently pass: wrong target, wrong prompt, wrong
checkpoint hash, negative token, missing provenance; each command returns 1 and
the intended error. Stock six-case script not claimed passing; valid case fails.
No expected tokens, assertions, tolerances or sampling semantics changed.

At position 80: 5 GiB plan 4.927902 GiB Metal / 4.791436–4.794808 GiB footprint;
3 GiB plan 2.927292 / 2.789924–2.793311 GiB. Snapshots, not sampled request peaks
or an unload soak. MLX full-resident reference peak 19.615 decimal GB. Hash-warmed
OS cache, diagnostic waits, no complete ordinary timing/cache footer, no swap
delta/physical SSD traffic series. No memory/performance promotion claim.

Updated TODO, recovery memory, baselines and ADR-006; preserved original fixture
and all prior documentation. S0-03 remains blocked. **Category D**: a small-margin
ranking reversal and upstream activation differences are characterized, but the
full cause is unresolved. Precision differences are a supported hypothesis,
not a passing numerical-equivalence policy. Strict 132/132 remains unsatisfied.
Next safe task: pin reference precision/execution and compare layer-0 components
with identical prefix and prior Conv/DeltaNet state, then router IDs; retain the
failure. Stop here; no Stage 1, commit, push or merge.

Final handoff checks: Markdown links exit 0 (19 files); git diff --check exit 0;
archived temporary patch passes git apply --check against restored source.
Original fixture and restored executable hashes independently rechecked unchanged;
all six recorded failure outcomes verified (one unresolved valid control and five
intended malformed-reference rejections). Evidence paths sanitized. No production
source, tests, scripts or dependency diff remains; all work is uncommitted.


## 2026-11-17 — Stage 0 numerical root-cause isolation (S0-03)

Scope from latest request: establish numerical contract, earliest token/component
and state/history effects; no Stage 1 or policy changes. Same branch/base and
uncommitted foundation preserved. Inspected git remotes (origin GitHub repository
unchanged), status, AGENTS/TODO/memory/log, architecture, prior report and archived
diagnostics, Swift active runner/Metal kernels, installed MLX layer/recurrence/
quantized/activation/cache code. No remote operations or dependency changes.

Detailed evidence, commands, metric definitions and limitations:
[isolation report](evidence/2026-11-17-isolation/REPORT.md),
[numerical contract](NUMERICAL_CONTRACT.md). Full checkpoint hashes, prompt IDs
and positions still match the original fixture; it was never regenerated or
altered here. Installed source hashes recorded again; packed DeltaNet dispatch
eligibility/default examined rather than assuming the package-version label
fully identifies arithmetic. No official-wheel equivalence claim.

Temporary runner-only diagnostics read layer-0 embedding, norm, projections,
Conv history/output, prior/post FP32 recurrent state, gated/output projections,
residual/postnorm, router IDs/scores and shared/routed results at positions
0,1,2,20,40,60,80. Small layer-3 selected IDs recorded for cache comparisons.
One extra command-buffer wait at selected layer-0 outputs; finite CPU/file
readbacks, no extra Metal arena. Bounded transient captures ~31 MiB per Swift
variant; no large all-layer dumps or weights archived. All readbacks/embedding
ablation code removed after investigation; patch kept as evidence only.

Three sequential teacher runs: baseline diagnostics (5 GiB, 2,183 slots), only
embedding BF16 rounding (5 GiB), smaller expert cache (3 GiB, 969 slots). Each
exit 1, 131/132, only token 61 expected 14988 actual 10000. Each run verified
checkpoint SHA-256, same 8K context, raw storage and no profiling. Commands in
report set QWEN_ISOLATION and QWEN_EMBEDDING_BF16 explicitly. No expected IDs,
sampler or production recurrence changed. Early MLX start was prevented by
process exclusion while the final Swift run was active (exit 90, no load).
Retried after owner completion; no concurrent model processes. Normal pressure
level=1 checked before starts; memory-free reports 85%–88%; continuing same
M5/32 GB, macOS 27.2.1, Swift 6.4.2 environment and ample disk.

MLX direct layer-0 transcription uses native modules and native gated_delta_update.
Its output AND updated state exactly match native layer calls at every sampled
boundary using identical prior state. Two histories: all tokens sequential, and
actual first-20-token batched prompt followed by sequential tokens. Early batch
row components saved separately; per-step prior state belongs to the labelled
sequential path, not falsely presented as baseline batched state. Dtypes measured:
embedding/projections/Conv/core/gated/residual/router/expert outputs BF16;
Conv history BF16; DeltaNet state and decay FP32; beta BF16. Actual two-token
full-model cache verified BF16 K/V; actual RoPE module probe BF16 output.
MLX activation helpers contain FP32 gate/product work before BF16 output casts.
Internal closed-backend MAC/reduction details remain explicitly unverified.

Earliest native-path difference: token position 0 embedding, before nonzero prior
state. Against actual batched reference, layer-0 output error is 1.012012%
relative L2 at 0 and 0.438729% at 80. Full max/RMS/relative/reference-scale metrics
for all boundaries in comparison artifacts; no small absolute error excused.

Matched FP32 tests: RMSNorm exactly matches; QKV relative L2 <=1.924e-7;
Conv+SiLU <=6.657e-8; DeltaNet state <=1.122e-7; gated result <=1.886e-7.
Q8 router/shared/routed expert tests agree closely (full metrics in summary).
State layout [B=1,32,value_row=128,key_col=128] explicitly aligned. Conv Swift
[channel,4] slots 1:4 transpose to MLX [1,3,channel]. No BF16 recurrent state
was copied into FP32 and mislabeled numerically identical.

At 80, matched-input/prior-state update relative L2 error is 3.133e-8. Holding
current inputs fixed and changing only prior state changes next state 0.382589%;
holding prior fixed and substituting native current inputs changes it 0.100589%.
These are separate counterfactuals, not additive causal shares. Even MLX's own
batched vs sequential prompt changes later state: at 80 post-state difference
0.091425%, output difference 0.399003%, current QKV identical. History effects
are demonstrated. Native state discrepancy is not just the current token.

Embedding-only rounding has mixed effects and leaves full parity 131/132.
Seven layer-0 expert sets match across runtimes after aligning IDs rather than
array order; maximum native per-ID weight discrepancy 0.003897175. Cache size
changes selected IDs at neither layer 0 nor 3 in the sampled positions; local
output differences <=7.451e-9 absolute. Separate same-contribution forward vs
reverse FP32 expert summation gives small rounding changes. No claim that every
other layer/token's routing matches, or that cache changes are bitwise inert.

First attempt to import MLX docstrings inside sandbox failed (Metal unavailable);
read-only expanded-access retry succeeded. Docstrings do not specify internal
MAC precision; contract labels that unknown instead of assuming BF16 or FP32.
No backend configuration was altered to force the expected prediction.

Removed all temporary source edits from the saved exact original. Release build
exit 0 (18.11 s), Scripts/test.sh exit 0 (17 Presentation + 32 Core = 49).
Restored full parity exit 1 (131/132, same one mismatch). Benchmark SHA-256 again
f55a81ce3041544c5cd811a1c9acbc04abcb302f8b40a29fd67d2d0af2854c32. No production
source/test/dependency diff; no concrete fix or new defect regression retained.
Five negative-reference cases passed in previous phase; not rerun because no
validation-code/fix changed. No claim that the six-case suite passes.

Loaded Metal snapshots 4.928 GiB and 2.927 GiB stay under selected budgets, not
request-peak measurements. MLX diagnostic peak 19,631,363,782 bytes. Temporary
CPU readbacks bounded; no speed claim or complete timing/cache/swap/unload series.

Added NUMERICAL_CONTRACT, updated TODO/memory/baselines/ADR-007 and architecture
index. Overall classification 4 remains: intentional numerical differences are
established locally and examined operations are consistent under matched FP32
inputs/state, but remaining full-model token-61 cause is unisolated. No policy
change or production fix. Proposed additional operation/state/logit/long-history
validation dimensions are for review, not an adopted tolerance gate.

Next bounded experiment: layer-3 attention on identical Q/K/V and prior history,
changing only FP16 versus BF16 KV storage with explicit layout/position conversion;
small routing-ID summaries through layers at positions 20/80. Stage 1 remains
prohibited. Stop with original fixture and strict 132/132 preserved. No commit,
push or merge.

Isolation handoff checks: all evidence JSON parses; original fixture and restored
executable SHA-256 independently rechecked unchanged; no Sources/Scripts/Tests/
Package diff. git diff --check passes; all links/anchors in 21 Markdown files
resolve. Archived patch applies cleanly with git apply --check. Evidence paths
sanitized. Restored parity process confirmed exit 1; both MLX diagnostic processes
confirmed exit 0. All requested documents updated; work remains uncommitted.


## 2026-11-17 — Revised priorities and first independent Stage 1 iteration

Explicit user policy supersedes the global numerical stop: retain unresolved
131/132 as a release blocker while permitting independent reliability work.
Expected IDs, strict parity tests, fixture/evidence and production-readiness
limitations remain intact. No layer-3 investigation scheduled. Updated AGENTS,
TODO, ADR-008 and recovery policy. Current implementation iteration S1-03a:
metadata regular-file/containment validation and bounded recoverable index reads.
Model-free tests first. Numerical non-regression uses preserved failure evidence
plus unchanged inference artifact/source identity for this selection-only task;
model runs remain required for affected inference paths and integration gates.
No push/merge authorized. Parent Stage 1 tasks remain open.


### S1-03a result — metadata input safety and non-regression setup

Completed one bounded implementation iteration. QwenModelDirectory now rejects
required metadata directories/FIFOs/escaping symlinks before index read, preserves
internal links, opens the resolved index leaf with NONBLOCK/NOFOLLOW/CLOEXEC,
checks the opened descriptor is regular, caps index payload at 16 MiB (including
a 16 MiB+1 read bound to detect concurrent growth), and returns typed errors for
malformed/read/open/size failures. File descriptors close on every handled exit.
Index metadata is never a shard copy. Concurrent directory/ancestor substitution
and later runner opens remain out of scope/open; no claim of complete model IO
safety. No tensor/sampling/kernel/state/allocation algorithm changed.

Added model-free regressions: each required metadata path as a directory, FIFO,
external link; internal index-link success; sparse oversized index and malformed
JSON; preserved previous valid/missing-shard/bounds tests. Local checkpoint
metadata stat-only check: config 23,591 B, index 215,755 B, tokenizer 19,989,343 B,
tokenizer config 1,139 B; all regular, root-contained. The 16 MiB cap applies only
to index reads, not the tokenizer. No weights read, copied or hashed.

Added a separate known-issue checker plus six model-free tests. It rejects new/
changed mismatch triples, incomplete/duplicate/malformed summaries and changed
fixture identity. Optional numerical-source/dependency-inventory guard rejects
changed/added inference code. Only the selection metadata validator is excluded
for this task. Original strict teacher test/fixture/132 targets are unchanged;
this is explicitly not a parity pass or universal numerical-tolerance policy.
Changed arithmetic/state/IO integration still needs bounded model-backed score,
state and behavioral gates; token count alone cannot certify material accuracy.

Commands/results:
- Scripts/test.sh --filter QwenModelDirectoryTests: exit 0, 10 tests; parameterized
  metadata directory/FIFO/link cases cover all four required filenames.
- python3 Scripts/test_numerical_non_regression.py: exit 0, six tests, including
  changed dependency/source inventory and new mismatch rejection.
- Scripts/test.sh: exit 0, 38 Core + 17 Presentation = 55 tests.
- swift build -c release: exit 0, 32.88 s.
- python3 Scripts/check_numerical_non_regression.py --log
  docs/evidence/2026-11-17-isolation/restored-parity.log --unchanged-inference:
  exit 0, archived known failure 131/132, no additional mismatch records, numerical
  source/dependency identity unchanged (named selection validator exception).
- git diff --check and Markdown links pass (22 files at this check).

No current model-backed parity run performed: benchmark inference sources and
pins are unchanged and the changed validator is called by app directory selection,
not benchmark inference. No repeated checkpoint hashing or integrity bypass.
The completed new release benchmark SHA-256 differs from the old control:
49cf77927f0a747ba395a3918fa74796f9b66aed3763399fe3cafd102065ef26. Therefore no
byte-identical executable claim; inference source identity and archived evidence
are the current task's non-regression evidence, not a fresh numerical measurement.
Metadata heap reads are bounded (payload <=16 MiB+1 plus read/allocation overhead);
no new Metal buffer/budget change and no process peak measurement claimed.

Evidence: docs/evidence/2026-11-17-stage1-metadata/; original numerical artifacts
unchanged. Updated policy, TODO, recovery memory, decisions, architecture and
baseline docs. S1-03a completed; parent S1-03 remains in progress; Stage 1 overall
remains **in progress**, not independently qualified. Allocation/command failures,
shared header/tensor validation and lifecycle work remain outstanding.

Next iteration S1-01a: shared bounded header/offset parser with model-free faults,
then separately integrate reader ownership with affected-path model validation.
Strict 132/132 remains release blocker; no more broad layer-3 investigation.
Stop before Stage 2 until Stage 1 independent criteria pass; no push/merge.

Final S1-03a handoff verification: all 22 Markdown files resolve, diff whitespace
check passes, six checker tests pass. Strict teacher script, benchmark generation,
active runner and dependency files remain unchanged. Known-issue source guard
passes against the preserved failing log; no fresh inference result implied.
Original numerical evidence/fixture preserved. One implementation task completed;
Stage 1 remains open with S1-01a next. Work remains uncommitted; no push/merge.


## 2026-11-17 — S1-01a shared bounded header validation (started)

Reloaded AGENTS/TODO/memory/architecture/log and inspected all four independent
Safetensors parser paths and tests. Resident loader caps at 100,000,000 bytes,
checks offset endpoints but skips malformed entries and does not verify byte
extents; Q4 runner mmap readers lack equivalent bounds checks; pack builder has
a different 256 MiB cap and partial numeric checks. Reuse QwenTensorInfo and the
existing EINTR-aware positional-read helper, without changing those active callers.
Plan: one isolated shared validator with typed errors, exact integer/size/range
checks and descriptor ownership. S1-01a stops before reader/mmap integration.
No inference math/path, prior S1-03a changes, numerical fixture or tests altered.
Only local checkpoint headers were inventoried (no payload/hash): sizes 88,906,
67,237,67,875,50,269 bytes; current types BF16/U32. Existing F32 metadata remains
supported. Fixed-width metadata support does not add model/runtime formats.


### S1-01a completed — isolated parser/descriptor-owner foundation

Added Infrastructure/ModelIO/QwenSafetensorsHeader.swift and registered it and
QwenSafetensorsHeaderTests.swift in Package.swift. Reused QwenTensorInfo and the
existing EINTR-aware QwenQ4ExpertStore.preadFully. Existing resident/Q4/pack
reader source was not changed and has no call sites to this foundation yet.
S1-03a metadata implementation/tests and all prior numerical evidence preserved.

Prefix reads are exactly eight bytes, decoded bytewise little-endian. Positive
header length <=100,000,000 decimal bytes and available file size checked before
allocation. Positional helper calls are proven bounded to <=100,000,008 offset+
count; no full-file copy/mapping or payload hashing. UTF-8 object JSON required;
raw NUL/alternate encodings rejected explicitly because JSONSerialization can
auto-detect UTF-16/32. String __metadata__, exact nonnegative integers (no bool/
float/string coercion), offset count/order/bounds, checked dimension/byte products,
exact dtype extents and nonoverlapping nonempty ranges validated. Scalars and zero
shapes supported; zero dimensions avoid spurious overflow. Fixed-width metadata
widths do not add runtime/model support. Unknown encodings return typed errors.

Owner opens caller-resolved URL NONBLOCK/NOFOLLOW/CLOEXEC, checks actual descriptor
regularity/size, compares device/inode/size/mtime/ctime around parse, closes on
failure/deinit, and exposes a borrowed descriptor plus verifyUnchanged. Caller
must resolve legitimate links/enforce root containment before open and retain the
same descriptor for later reads. Descriptor pins namespace replacement, not content
immutability. Metadata checks are not write locks; ancestor races, post-check
writes/truncation and mmap SIGBUS prevention remain integration/lifecycle work.
No general-purpose framework, payload reader or loader rewrite introduced.

Tests cover quantized U32/BF16 and F32/scalar/empty fixtures, supported widths,
truncated prefix/header, malformed/non-object/non-UTF8/UTF16/NUL headers, missing
fields, fractional/bool/negative/oversized integers, unsupported types, shape and
byte-count overflow, offset mismatch/out-of-bounds/overlap, exact cap prefix and
virtual Int.max absolute file boundary, FIFO/directory/link/missing-file errors,
mutation detection, positional-read FD preservation, descriptor deinit and path
replacement. Virtual arithmetic tests allocate no enormous payload. Added the
exact Int.max boundary and alternate-encoding cases after focused review; checks
were rerun for these new tests/changes. No failing assertion was weakened.

Final commands/results:
- Scripts/test.sh --filter QwenSafetensorsHeaderTests: exit 0, **18 tests**.
- Scripts/test.sh: exit 0, **56 Core + 17 Presentation = 73 tests**.
- swift build -c release: exit 0, **32.34 s** (build duration, not inference).
- python3 Scripts/test_numerical_non_regression.py: exit 0, six tests.
- python3 Scripts/check_numerical_non_regression.py --log
  docs/evidence/2026-11-17-isolation/restored-parity.log: exit 0, archived known
  131/132 log only. This does not prove numerical equivalence or constitute a run.
- Scoped byte/diff review: every previously fingerprinted numerical implementation
  and dependency pin unchanged; Package.swift only adds source/test registration;
  no existing runtime/benchmark/pack call sites refer to the new validator.
  Original fixture SHA-256 independently checked unchanged.

The new unused Core source changes inventory/manifest, so --unchanged-inference
fast path cannot certify it. The checker/baseline were deliberately NOT relaxed
or regenerated. User explicitly permits isolated parser-only work without a full
model replay; manual source isolation and scope review are the justification.
No parity/model run, checkpoint payload hashing, numerical/performance promotion
or new memory-peak claim. Known unresolved 131/132 remains a release blocker.
Header cap bounds read bytes, not Foundation object/process allocation overhead;
no Metal allocation/cache budget changed. Gaps/trailing payloads are permitted;
duplicate JSON key and payload-content validation are not claimed. Model-specific
tensor contracts remain separate. These limits are recorded in ARCHITECTURE.

Updated TODO (S1-01a complete, S1-01b pending), recovery memory, architecture,
ADR-009 and non-regression documentation. Logs/scoped validation are preserved
under docs/evidence/2026-11-17-s101a/. Stage 1 overall remains open. Next task:
S1-01b migrate one reader (resident loader first) to descriptor-owned validation,
then separately gate Q4/pack integrations. Stop after S1-01a as requested; no
Stage 2/server, inference math edits, push or merge.

S1-01a handoff checks: git diff --check and local Markdown links pass (22 files).
Existing numerical implementation bytes, original fixture hash, strict checker,
reference script, dependency pins and existing readers remain unchanged; only
manifest registration plus the new unused utility/tests added. Evidence JSON
parses and paths are sanitized. Final release/test process exit 0 confirmed.
No commits/push/merge; stop at S1-01a as requested.


## 2026-11-17 — S1-01b started: resident CPU tensor reader

Reloaded required project documents/shared implementation and compared resident,
active Q4, fallback Q4 and pack-builder readers. Choose QwenResidentWeights:
smallest CPU BF16/F32 reader, no mmap/zero-copy path to disturb. Other readers
remain untouched. Replace its duplicated header parsing and path reopens with
shared descriptor-owned validation; preserve root containment/internal links and
correct data-region-relative offsets. Serialize ownership/reads and invalidate
an affected cached shard on failure, with explicit caller retry/revalidation.
No automatic retry on mutation. Add bounded payload reads to the shared owner,
using existing checked header metadata and pread helper. No kernel/math changes.
Plan focused/model-free/full/release checks first, then one explicit local-checkpoint
reader integration and one strict benchmark parity replay with required hashing
intact. Those exercise different reader paths; no claim otherwise.

### S1-01b result and validation

Migrated QwenResidentWeights CPU BF16/F32 tensor/row reader only. It is distinct
from QwenQ4ChatRunner Fast Resident mode. Removed its independent Safetensors
header parser and metadata-only cache/path reopens; retain shared validated
owners per encountered shard. Existing lock serializes ownership/reads. Resolved
root, safe basename and direct-root containment preserve internal shard links;
unresolved substituted leaf links fail through the shared NOFOLLOW open.

Added QwenSafetensorsFile.readPayload: validate a data-region-relative range,
verify opened identity before/after positional read, compute absolute position
8 + header byte count + relative offset, support empty reads safely. Bounds prove
Int/off_t arithmetic fits; unchanged pread primitive handles partial/EINTR reads.
No full-shard read, mmap, hash or extra Metal allocation introduced. CPU full-tensor
API retains selected raw tensor + decoded Float storage; rows retain row scratch
plus requested output. Overflow guards precede reads/allocations; full-tensor
output allocation stays after successful payload read (review corrected ordering,
then tests/release rerun before any real-model check). Unaligned LE bit loads
preserve BF16/F32 values without inference mathematics changes.

Validate decoder dtype before payload access, embedding dimensionality/hidden
size/token bounds and row/output byte arithmetic before use. Every failure on
an opened shard invalidates/releases its cached owner; explicit subsequent retry
opens/revalidates a fresh file, not stale offsets. Successful reads do not reopen
paths or reparse headers. Reader deinit releases owners. Public metadata tuple
retains header-size semantics; paths are advisory, not protected read leases.

Security limits: descriptor pins the opened inode, not immutable contents or a
multi-shard snapshot. Before/after size/mtime/ctime checks are best effort, not
write locks/hashes. Ancestor/directory substitution and changes after checks
remain possible; require immutable checkpoints while owned. Retargeted links
cannot redirect an existing owner; replacement reads original descriptor or
returns controlled identity error, never silently switches paths within the
operation. QwenResidentCoreLoader index JSON read remains unchanged and outside
this Safetensors migration. Both Q4 mmap readers/pack builder remain unmigrated.

Focused tests cover BF16/F32 bytes at nonzero/unaligned data-region offsets,
last/repeated/out-of-order/empty rows, empty tensors, unsafe/traversing/NUL names,
internal links and retargeting, external links, FIFO/directory/missing files,
malformed/oversized/truncated headers, row overflow and unsupported decoder types,
mutation/cache invalidation/repair retry, path replacement, reader/failed-operation
FD release. Initial test compilation failed on Swift Testing key-path allSatisfy
macro expansion; replaced with equivalent closure (finite-value assertion kept).
Initial failure log preserved; no assertion weakened.

Final model-free commands/results:
- Scripts/test.sh --filter QwenResidentWeightsTests: exit 0; ten model-free
  test functions passed (parameterized unsafe names include six cases); one
  opt-in checkpoint test explicitly skipped.
- Scripts/test.sh: exit 0; 84 tests reported (67 Core, 17 Presentation), of which
  83 passed and the one opt-in checkpoint test was skipped.
- swift build -c release: exit 0, 32.32 s, all existing release products.
- python3 Scripts/test_numerical_non_regression.py: exit 0, six tests.

Hardware/preflight: MacBook Pro Mac17,2, Apple M5, 32 GB, ten GPU cores;
macOS 27.2.1/26B131; Swift 6.4.2 (swiftlang-6.4.2.1.9, clang-2100.4.1.1), arm64.
133 GiB free disk, memory_pressure -Q 87% free and pressure level=1; no other
process matching required exclusions. Rechecked normal pressure/no model owner
before each sequential checkpoint process. Required OS/toolchain/disk/checkpoint
were available. No new MLX reference run or package/model modifications.

With MODEL_DIR the existing local checkpoint:

```bash
QWENFIELDFARE_READER_CHECKPOINT="$MODEL_DIR" \
  Scripts/test.sh --filter localCheckpointHeadersAndSmallTensorIntegration
.build/release/QwenFieldfareBench --model "$MODEL_DIR" --context 8192 \
  --memory-gib 5 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
python3 Scripts/check_numerical_non_regression.py --log /tmp/qwen-s101b-parity.log
```

Explicit checkpoint reader test: exit 0, one test passed. Shared validator read
all four actual headers (88,906 / 67,237 / 67,875 / 50,269 bytes) once per owner,
then read one 2,048-element norm with finite values. No full-model residency or
whole weight file copy. This is the migrated-reader integration, not generated
text parity. Retain 100,000,000-byte cap: supported checkpoint fits comfortably,
existing resident cap not expanded and narrowing compatible export policy is
not bundled into this migration. A smaller universal limit is not justified by
one checkpoint alone. Cap bounds header input bytes, not Foundation heap/process.

Exactly one strict release teacher replay: checkpoint SHA-256 verification intact
and performed once; exit **1**, **131/132**, only reference token 61 (one-based;
index60) expected14988/actual10000. Original prompt 21 tokens, 132 teacher targets,
8,192 context, 5 GiB plan, 2,183 expert slots, raw storage, greedy/no profiling.
Loaded snapshots 4.928 GiB Metal / 0.996 GiB footprint, load 0.526 s. Teacher mode
exits before ordinary timing/cache footer; full footer, request peak/TTFT/decode,
physical SSD traffic, pressure/swap delta series and unload soak unavailable.
Hashing warms OS cache; this correctness run is not headline throughput evidence.
No memory/performance improvement claim.

The separate mismatch-set checker exits 0 for that fresh log, not a strict parity
pass or numerical-equivalence certification. Active benchmark uses another reader;
its result alone does not exercise the migrated CPU utility. Both separate checks
are retained. No extra mismatch/load regression observed; no repeat real-model
or expensive hash iteration needed. Original fixture and expected IDs unchanged.

Updated TODO/memory/architecture, ADR-010, numerical baseline/non-regression docs.
Evidence under docs/evidence/2026-11-17-s101b/ includes raw logs, strict exit record,
preflight and candidate hash. Existing Q4/math/cache/pack readers and dependency
pins verified unchanged. S1-01b complete; Stage 1 aggregate in progress. Next safe
task S1-01c: one active-streaming reader migration preserving zero-copy mapping,
expert reads and cleanup/lifetime ownership, with its own tests/model gate.
Stop here as requested; no second reader, kernel changes, Stage 2/server, push
or merge. All changes remain uncommitted.

S1-01b handoff checks: all local links/anchors resolve (22 Markdown files), diff
whitespace passes, original fixture hash and unchanged Q4/pack/strict-test sources
rechecked. Strict exit1 artifact remains intact. Evidence JSON/logs are sanitized,
including the synthetic NUL-name test display. No code changes followed the final
model checks; no second reader was migrated. Work remains uncommitted.

## 2026-11-17 — S1-01c ownership inspection and bounded plan

Before migration the active reader opens four shards, maps each whole file, parses JSON directly from the mapping, and stores raw fd/pointer structs. Failed parsing before dictionary insertion leaks the mapping/fd. Runner deinit manually unmaps; LM-head bytesNoCopy buffers have no deallocator/owner lease. CPU embedding/dense/expert operations borrow mappings synchronously; expert dispatch joins before returning. Default retained-reference Metal command buffers retain their buffers, and the synchronous forward waits for the final same-queue command. Cancellation is checked between forwards; ChatService's task retains its runner and defers unload until task cleanup.

Plan: one RAII whole-file mapping owner retains QwenSafetensorsFile; validate/map the same descriptor and check metadata around mmap. Each no-copy Metal buffer retains that owner in its deallocator, extending ownership through retained command buffers, even on failed initialization. No forced-close/unmap API. Validate consumed tensor byte extents before fixed-size copies; checked page/range calculations. Keep bounded expert pread and existing immutable-checkpoint requirement. Descriptor identity is not pathname identity or immutability; concurrent truncation can still SIGBUS. Model-free tests must never intentionally touch a truncated mapping. No second reader, kernel, sampling or precision change.

## 2026-11-17 — S1-01c completed: active-streaming mapped-descriptor reader

Changes: added QwenMappedSafetensorsFile ARC owner around QwenSafetensorsFile,
whole-file read-only mmap from its validated descriptor, checked page-rounding,
data-relative tensor ranges, bounded pread and owner-retaining Metal no-copy
buffers. Transactional shard-set opening preserves internal links/direct-root
containment and unwinds partial failure. Replaced only active-streaming header
parsing/manual fd+munmap. Fixed-consumer dtype/byte extents bound all existing
constant/dense/embedding/head/expert accesses; full shapes/config/quantization
remain S1-02. Bit loads tolerate actual unaligned payloads without changing bit
interpretation or arithmetic. Validated token/position/expert ranges prove hot
address arithmetic fits the shared tensor/file bounds. Expert short-read errors
join parallel work, wait for hit commands, clear cache and throw before failed
misses are consumed; no fault-prone mmap fallback. Kernel, sampling, precision,
quantization, memory-arena/cache budgets and other readers remain unchanged.

Validation in order: model-free focused tests, full serial package suite,
release, preflight, exactly one real mapping integration, exactly one required
checkpoint-hashed strict replay, mismatch-set checker. Final focused16 passed +
1 opt-in skip; package99 passed +2 opt-in skipped (101 reported,84 Core including
skips +17 Presentation); release passed33.29s; checker unit6 passed; Markdown and
diff checks passed. Integration validated all four real headers and mapped a
zero-copy LM head; headers88906/67237/67875/50269 bytes, retained100M decimal cap.

Real gate command: `.build/release/QwenFieldfareBench --model /Users/tanishq/models/Qwen3.6-35B-A3B-4bit --context 8192 --memory-gib 5 --teacher-reference docs/evidence/2026-11-17-parity/reference.json`.
Checkpoint integrity verified once by unchanged teacher CLI. Strict exit1 is
preserved:131/132, token61 expected14988/actual10000. Separate checker accepts
exact known mismatch stability only, not equivalence/strict parity. Fixture hash
unchanged8d552a906d417ceb492ddd045a50e67b0e9630988d9665e8c6fe5e480cb30747.
Candidate Bench38480688e1bebd6feda215e383d3af630237c7dc89d8fbab7a348f983f1e6743.
Host Mac17,2 M5/32GB/10GPU; macOS27.2.1(26B131),Swift6.4.2;133GiB free disk,
normal pressure1/87% free. Context8192,5GiB/2183slots,raw,greedy42,no profiling;
load0.125s,footprint0.996GiB,Metal4.928GiB. Teacher mode provides no normal
cache/timing footer after strict failure; no throughput/peak claim. No extra hash
bypass, model replay, checkpoint mutation or broad numerical investigation.

Test development: initial new Metal test incorrectly required the lease to
survive after GPU completion; Metal released it immediately when permitted.
Corrected this new assertion to require retention through pending GPU use, verify
copied bytes, and require final cleanup. Existing regression tests were untouched.
An accidental no-argument invocation of test_teacher_reference.py returned its
argument-usage error before any model execution; it is not counted as a passing
check. Its expensive six-case real harness was not repeated; the one strict
replay retains the original known failure. A sandboxed ps inspection was denied;
preflight process exclusion had already passed under the approved host check.

Evidence: docs/evidence/2026-11-17-s101c (manifest, source hashes, focused/package/
release/integration/strict/exit/checker/preflight logs). Git verifies all Metal,
selection support, cache, packed helpers, fallback and teacher harness unchanged
from base72fb923. Original prior evidence is intact. Documents updated: TODO,
AGENT_MEMORY, ARCHITECTURE, numerical non-regression record.

Ownership findings: no cycle from buffer lease; command buffers retain resources
through use; closing FD alone never unmaps. No explicit invalidation API permits a
live owner to outlive its mapping. Cancellation remains between synchronous
forwards and task ownership defers unload. Mutable checkpoints are still unsafe:
stat does not lock file bytes, truncation can SIGBUS, parent races and multi-shard
snapshots remain unqualified. Mutation fixtures never access a truncated map.
Active index ingestion, pre-allocation full contract, existing allocator/encoder
fatal paths, fallback Q4ChatRunner and pack-builder readers remain open. Release
and production claims still blocked by numerical acceptance evidence/review.

Stopped after S1-01c. Proposed next S1-01d: inspect/migrate fallback Q4ChatRunner
ownership only, with its independent gates. No second-reader implementation,
Stage2/server, commit, push or merge.

## 2026-11-17 — S1-01d inspection and bounded plan

Read project directives, TODO, memory, architecture, development log, shared
header/descriptor/mapping implementation, resident reader and both Q4 loaders.
Fallback owns raw fd/mmap structs, parses unchecked mapped headers and creates
no-copy buffers without owner leases, with manual unmap in runner deinit.
It maps entire shards; memory-saver lazily retains a bounded layer LRU (default4),
fast-resident populates40 layers. Quantized layer buffers reference mappings;
BF16 constants are decoded/copied. All fallback command submissions synchronously
wait before continuing, including before later cache eviction. Cancellation is
between forwards and task retention defers unload. Partial initialization currently
can leak a map before dictionary insertion. No copying of whole weight shards.

Plan: reuse QwenMappedSafetensorsFile and existing fixed-Qwen consumer extent
validator, preserve layer cache strategy and GPU math; no new filesystem/parser
framework. Descriptor-based mapping + buffer-held map owners fix partial-init/
eviction/unload lifetimes. Check context-byte arithmetic before allocations and
use recoverable initialization allocations; retain exact valid context behavior.
Preserve header cap100,000,000 bytes; existing measured requirements88906/67237/
67875/50269 bytes remain compatible. Full shape/quantization preflight is S1-02.

Model-free tests first, then full suite/release/checkers/docs. A single opt-in
actual fallback memory-saver test uses short context and configured one-layer
cache, not 40-layer fast-resident; record memory and pressure. The unchanged
teacher CLI instantiates QwenQ4ActiveExpertStreamingRunner only, never fallback;
therefore reuse its S1-01c known-failure evidence without another expensive hash
or strict replay, and do not claim it validates fallback numerics. Review cumulative
diff/artifacts and make scoped local commits after gates; no push/merge/Stage2.

## 2026-11-17 — S1-01d completed; cumulative checkpoint review

Only fallback QwenQ4ChatRunner migrated: reuse validated descriptor/header/map
owner and existing active fixed-Qwen extent guard. Whole-file/no-copy strategy,
fast-resident40 and lazy configured layer-cache policy retained. Removed direct
mapped-header JSON parsing, unchecked path reopen and manual deinit fd/munmap.
Buffer deallocator leases retain map owners through Metal use/eviction/unload;
partial-init failure unwinds via ARC. Checked context lengths precede allocation,
FP32 KV/reset byte counts unchanged; constant/state allocation failures now throw.
CPU loads allow unaligned bit positions without changing BF16/Q4 math. Invalid
indices and observed file metadata changes fail before forward. Full shape/
quantization preflight, index bounds, encoder/command failures and lifecycle soak
remain separate work. Shared parser/map/active implementations are byte-identical
to S1-01c; no other reader, kernels, sampling or cache policy changed this task.

Validation: focused8 pass +1 skip; full107 pass +3 skips (110 reported;93 Core
including skips +17 Presentation); release33.45s; numerical checker unit6 pass;
Markdown links and diff/whitespace checks pass. One explicit actual fallback test
passed8.110s,exit0: memorySaver,context8,configured cache1,budget5/hard6GiB (these
fields are not process caps). Two independent reset forwards of input248045 each
return846/846;80 creations/misses,79 evictions,0 hits,all40 layers per forward.
No generated sequence, MLX equivalence or full-context/fast-resident claim.

Preflight MacBook Pro Mac17,2,M5,32GB,10GPU; macOS27.2.1(26B131),Swift6.4.2,
133GiB free disk,no competing model process,normal pressure1/free86%. Postflight
pressure1/free85%. Footprint/Metal bytes before9618224/65536; loaded88100152/
645824512; after forwards121687376/1117323264; after unload34751352/720896.
These are snapshots, not peaks or a soak. No throughput/timing-cache footer is
available from this integration test; only duration and cache counts above.

Exact model command: `QWENFIELDFARE_FALLBACK_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit Scripts/test.sh --filter localFallbackCheckpointIntegration`.
Teacher harness instantiates only active-streaming and its execution path is
unchanged. No new strict replay or full checkpoint hash required/performed; no
integrity bypass introduced. Existing S1-01c checkpoint-verified131/132,exit1,
token61 expected14988/actual10000 is retained, not fallback validation. Historical
log checker passes exact known mismatch set only. Original fixture/hash intact.
Header cap100,000,000 unchanged; earlier measured88906/67237/67875/50269 fit.

Cumulative diff review: all pending runtime changes belong to authorized metadata/
Safetensors/ownership work; only Package source/test lists changed, no dependency
or server target. Original diagnostic kernel patches remain textual evidence,
not applied source. Evidence audit: textual UTF8/no NUL,about1.5MB; no model weights,
large payload blobs or build artifacts among candidates. Ignored .DS_Store and
build/Python caches remain excluded and unmodified. Prior logs are ignored by the
repo-wide *.log rule, so explicitly stage reviewed evidence logs rather than lose
their links in the checkpoint. No unrelated user changes reverted or staged.

Scoped local commit plan authorized by current user request:
1. Reader-safety implementation/package/tests, accumulated S1-03a/S1-01a/b/c/d.
2. Foundation/current-state documentation, numerical policy/checker tests and all
   existing textual evidence, preserving strict failure and implemented/proposed
   distinctions. Runtime commit stands alone/builds; documentation follows as a
   separate review unit. No push/merge or Stage2/server. Completion/commit results
   are recorded in the following handoff entry and final response.

Remaining: pack-builder Safetensors reader with its256MiB cap needs its own
compatibility/ownership task (proposed S1-01e, not begun); full contract, index,
allocator/encoder and lifecycle gates remain. Immutable trusted checkpoints still
required; identity/stat checks cannot prevent concurrent mutation/truncation/SIGBUS,
ancestor substitution or multi-shard inconsistency. Strict132/132 and independent
numerical acceptance/validation still block production readiness. Stop after S1-01d.

## 2026-11-17 — Local Git handoff after S1-01d

Created local runtime/package/test checkpoint `8b263e05f9c332ffa56f44be4d13c676042ba3b2`:
`fix: validate Safetensors readers and retain mapped file ownership`.
It contains only the12 reviewed Stage1 source/package/test paths, including all
prior completed reader work. Validation predates the commit with matching source
hashes; no source edits followed tests/release/model integration. The second local
commit contains foundation/current docs, policy/checker tests and textual evidence,
including explicitly added ignored logs. Its identity is available in git log;
this entry is included in that commit to keep the final tree clean. No push or merge.

Final review excludes model artifacts, .DS_Store, build and Python caches; archived
patches remain evidence only. Markdown links/anchors and staged whitespace pass.
The strict known issue remains unresolved131/132, not a clean release gate. Stage1
parent tasks remain open. Stop after S1-01d; proposed next S1-01e has not begun.

Evidence whitespace review before the documentation commit found intentional
context-only blank lines in two archived diagnostic patches and captured
trailing docstring spaces. Preserved all3 raw files byte-for-byte; added exact-
path .gitattributes -whitespace entries for those archives only. Source, tests,
ordinary docs and logs retain normal whitespace checks. This records raw evidence
handling, not suppression of numerical or implementation failures.

## S2-00 started — explicit priority override

Read directives/recovery/architecture/roadmaps/package, Core template/sampling/
active runner, app ChatService and NIO HTTP example. Clean cf0d144 baseline.
Remaining Stage1 deferred by user; no pack migration. Architecture documented
before code. Choose already-resolved NIO2.101.3 HTTP1 with direct dependency, no
extra general framework. Preserve original numerical fixture/release blocker.

## S2-00 functional milestone completed

Implemented QwenFieldfareServer plus ServerSupport Protocol.swift/Service.swift/
HTTP.swift; Package directly names existing exact NIO2.101.3. No pin upgrades.
One configured Core active runner/tokenizer, existing template/planner/config and
reset/replay, no duplicated inference. HTTP1 GET health/models +POST chat JSON/SSE,
usage and finish reasons, explicit unsupported-field validation. One active/two
queued,32 connections,15s body deadline,1MiB/128messages/4096output/context limits,
16KiB outbound watermark,one flush at a time,10s stalled-write deadline. Required
loopback bearer secret; no body/key/content logging. SIGINT/SIGTERM/disconnect
cancel and drain retained work; server flock excludes another server for user,
not app/Bench/MLX. Earlier Stage1-only ordering superseded by explicit attachment.

Final Scripts/test.sh:113 passed +3 opt-in skips (116 reported:93 Core including
skips,17 Presentation,6 server). Server tests cover invalid parameters, actual
HTTP JSON/SSE/auth/loopback, admission saturation/shutdown, disconnect cancellation,
initialization failure. Release passed33.34s; checker unit6 passes. Markdown links,
source/evidence checks and diff checks pass. All tracked Core sources byte-identical
to cf0d144, original fixture SHA unchanged. No kernels/precision/weights modified.

Actual test command: QWENFIELDFARE_SERVER_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit python3 Scripts/test_server_http.py .
One server at a time,127.0.0.1:18089,context8192,budget5GiB. Preflight Mac17,2/M5/
32GB/10GPU,macOS27.2.1/26B131,Swift6.4.2,133GiB disk,normal pressure. Two controlled
HTTP processes total: initial generation/SSE/reuse/context/disconnect succeeded,
shutdown trapped in Dispatch callback actor-isolation assertion. Diagnosis from
triggered stack only (no device-identifying crash report copied). Fixed signal
installation in nonisolated function. Final replay passes entire script,exit0;
no suppression of initial failure and no unrelated numerical investigation.

Final load0.896549s; health-ready1.367499s; first JSON1.052619s,repeat0.680336s,
each23 prompt/2 emitted completion tokens,finish stop. SSE0.674879s,first content
0.615240s,5 JSON events +DONE,then context400/disconnect recovery and SIGTERM during
request pass. Shutdown0.153305s. First-run metrics remain in initial JSON/log files,
not omitted to claim an optimization. OS-cache/warmup uncontrolled; no performance
promotion. No tools or actual OpenCode/Hermes test. Footprint loaded1157519336,
drained5143274256; Metal5291294720 bytes (4.928GiB) remains under selected5GiB
allocation plan. Drained runner is intentionally retained until process exits;
these are snapshots, not peak/RAM cap or unload soak. Postpressure1/free80%.

Strict teacher baseline131/132 exit1/token61 is retained. HTTP smoke proves no
MLX equivalence; no fresh strict replay/hash, checker reuses S1-01c log only.
Known Core fatal paths, checkpoint mutation/ancestor races and full-context/agent
qualification remain open. Remaining Stage1 tasks deferred, not abandoned.

Updated AGENTS/TODO/memory/architecture/roadmaps/integration/runtime controls/README.
Evidence docs/evidence/2026-11-17-s200 retains initial/final logs,aggregate metrics,
pre/postflight and source manifest; no conversations/keys/model artifacts. Work
uncommitted on cf0d144, no push/merge. Stop after S2-00. Next S2-01 validates
structured Qwen tool contracts, then OpenCode; none implemented in this milestone.

## S2-01 started

Read required recovery/plans/non-regression docs, server/backend/tests, Core
template/generation and local exact Jinja/tokenizer metadata. Created scoped
S2-00 checkpoint c001a78 (no push). Saved local/official template and diff. The
official generation annotations change training boundaries only; rendered tool
text and roles match. Earlier commentary suggesting a semantic reasoning-prefix
difference was corrected after byte comparison. Developer unsupported.
Design recorded before editing: exact Jinja tool formatter, bounded ordered JSON
schema/history validation, safe incremental prose + commit-on-EOS tool parser,
OpenAI JSON/SSE translation; retain no-tool Core path and unchanged math.

## S2-01 completed — structured native tools and real continuation

Scoped source changes: ToolJSON ordered/duplicate-key/depth-bounded decoder,
ToolSchema/FunctionTool/FunctionCall/ToolConversation typed contracts, ToolPrompt
Jinja renderer with HF-compatible ordered Unicode JSON filter, NativeToolParser
incremental safe prose/partial-prefix state and buffered structured validation.
Protocol accepts tools/auto/none/assistant calls/matched tool results; Service
renders exact checkpoint template for advanced turns and checks encoded context
reservation. HTTP translates committed native calls into JSON/SSE indexed deltas
with stable IDs, argument fragments, tool_calls finish and model-token usage.
No server-side tool execution. Schemas/modes/native delimiter restrictions are
explicit in integration guide; developer unsupported by template and rejected.
Existing no-tool path and auth/loopback/admission/model ownership preserved.

Local template SHAe84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259;
official1b8361565cc35d47551d342214cff341c0f1a9685ce529c7ce31de0ed3616248.
Saved exact template metadata and diff, not weights. Official only adds training
annotations/equivalent prefix relocation. Python Jinja3.1.6 goldens for prefix,
all-value-type history continuation and tool_choice:none match both templates
and Swift rendering exactly. Regeneration script uses existing local Python
Jinja environment, never imports/loads MLX. Existing Swift Jinja2.3.6 made direct,
no dependency pin upgrade. Tokenizer im_end248046 remains EOS; tool-call tags are
not EOS. Thinking disabled with exact existing assistant prefix.

Final tests:14 focused tool tests pass (every byte delimiter split,Unicode/multiline,
multiple calls,nested JSON,unknown/missing/duplicate inputs,malformed/truncated/
limited output,buffer/depth overflow,IDs/results/injection,JSON/SSE reassembly,
context,disabled tools and cancellation while partial delimiter pending). Existing
six server auth/loopback/admission/disconnect/init/shutdown tests unchanged/pass.
Full suite127 pass +3 skips (130 reported:20 Server,17 Presentation,93 Core incl
skips);release34.36s. Checker unit6 pass and historical mismatch-set check pass;
original strict harness and expected fixture untouched. No new numerical replay
or expensive checkpoint hash. All20 tracked Core files byte-identical to c001a78.
Post-real-test review fixed enum numeric comparison outside Decimal exponent
range and rejects JSON numbers that Jinja would silently round; focused/full/release rerun. Real function schema has no enum, so its
exercised code path is unchanged and no expensive replay was repeated.

Real command: QWENFIELDFARE_SERVER_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit python3 Scripts/test_server_tools.py .
One server127.0.0.1:18090,context8192,budget5GiB,greedy512 output allowance. Genuine
JSON and SSE calls both select get_project_status with {project:fieldfare} and
normal EOS; HTTP finish tool_calls. CLIENT supplies fixed public mock result
{project:fieldfare,status:building-local-agent}; both continuations return:
The current status of project fieldfare is "building-local-agent".
This is real model output, not a mock backend. Parser/transport fixtures are
separate. Each call344 prompt/27 output tokens;continuation404/13,finish stop.
JSON call16.195933s/continuation16.985182s;SSE15.313213/17.073172s. Existing text
JSON/SSE,none,context overflow from schema text,disconnect recovery also pass.
Server loads once0.775344s;ready1.310015s;SIGTERM shutdown0.088268s,exit0. No
actual filesystem/network tool invoked; only client-supplied constant test data.

Preflight Mac17,2 M5/32GB/10GPU,macOS27.2.1(26B131),Swift6.4.2,132GiB disk,
pressure1/free85% and no other model owner. Postpressure1/free79%. Metal
5291294720 bytes (4.928GiB),loaded footprint1167267816,drained5166064424 bytes;
retained runner releases with process exit. Snapshots not peaks or optimization.
No long-agent/full-context or actual OpenCode/Hermes qualification. Tool SSE waits
for whole-turn/EOS validation; declared unknown parameters/native delimiter data
restrictions and unsupported strict/ref/required/developer modes remain explicit.
Result text remains semantically untrusted, clients must authorize/sandbox tools.

Evidence docs/evidence/2026-11-17-s201 includes public synthetic HTTP exchanges
and metrics,source/template identities,logs and pre/postflight. Server emits no
conversation/tool arguments/secret logs; only the explicit public test client
persists its test exchanges. Original131/132 evidence is untouched and remains
a release blocker, not a fresh parity pass. Updated required recovery/docs;
remaining Stage1 deferred. Stop after S2-01; next OpenCode client qualification.
No prefix reuse,filesystem agent,Hermes,new reader migration,remote push or merge.

S2-01 final review: numeric JSON validation now rejects values whose Jinja Double
rendering would change their decimal meaning (rather than silently round), and
caps numeric literal length128 bytes. Added precision/enum tests, reran focused14,
full127+3skip and release34.36s. Real test's string schema and integer settings
are unaffected, so its genuine cycle evidence is reused without expensive replay.
All source changes remain outside Core. Scoped final commit includes only this
contract, tests, documentation and synthetic evidence; no remote push/merge.

## S2-02 started

Read current directives/recovery/architecture/plans/server/tests and installed
CLI. Clean c2b1663 baseline. User explicitly says stop after S2-02; S2-03 UI is
next, not bundled now. OpenCode1.19.5 at Homebrew; initial sandbox cache creation
denied, so isolated XDG cache/data/state/config in /tmp. Website current v2 docs
differ; actual debug config accepts v1 provider/npm/options/agent contract.
Explicit --dir and matching PWD needed to avoid inherited checkout session path.
Local capture-only endpoint showed Chat Completions JSON/SSE,auto/usage and five
allowed tool schemas with $schema2020-12/numeric constraints. This was not a model
run or qualification success. No remote model or real repository tool execution.
Design above precedes edits; keep secrets out of argv/logs and no server tools.

S2-02 iteration: implemented parsed CLI/help/version/private key file, bounded
queue settings and health occupancy; both Core Q4 runners acquire a cooperative
RAII owner lease before allocations. Focused configuration/lease tests passed,
then complete package136 pass +3 skips (139 total), release35.31s. Actual client
wire fixture preserves five tool schemas; explicit2020-12 dialect/numeric bounds
added and enforced rather than removed. Later focused review/tests cover precise
Int64 bounds and extreme exponents; final full/build gates will follow model exit.

First real OpenCode attempt stopped deliberately after631.68s: zero successful
tools, three client permission denials (directory read, forbidden ls, source read).
Absolute permission patterns were incorrect for installed client's relative-path
matching. No production server error was hidden; complete failed attempt retained.
Scripted/model-free permission rehearsal with corrected workspace-relative edit
rules completed read/grep/edit/fail-test/read/repair/pass-test, and separately
denied outside reads, config edits and unauthorized shell. This is permission
contract evidence, not model success. One hypothesis-bounded real retry follows
with unchanged model/protocol and corrected client policy; all paths disposable.
Actual client output5000 and context-overflow errors already return understandable
HTTP400/non-retryable messages, and SIGINT client cancellation drains admission.
No expensive checkpoint hash or numerical replay repeated. Original strict gate
and fixture are intact; lightweight checker tests alone prove no new math result.

## S2-02 completed — actual OpenCode result and boundary

Corrected run genuine model workflow passed:OpenCode1.19.5 selected configured
local model,8 SSE completions,7 successful read/grep/read/bash/edit/edit/bash tools,
zero tool errors,final model response and independent2 passing unittests. Client
ran failing tests before either edit,contrary to prompt's implementation-first
sequence;explicit deviation retained. Source subtraction changed to addition,
wrong fixture expectation5 changed to4. These are disposable test fixtures only;
no repository test expectations or numerical fixture were weakened. First bad
permission attempt preserved;one corrected retry after model-free policy checks.

Successful command QWENFIELDFARE_SERVER_CHECKPOINT=/Users/tanishq/models/Qwen3.6-35B-A3B-4bit python3 Scripts/test_opencode.py .
Same M5/32GB/macOS27.2.1/Swift6.4.2,no competitors,normal pressure,133GiB disk.
Load3.806333s,workflow1696.730207s excluding probes/startup. Initial3410 prompt/
118 native output,final5233/155,aggregate replayed35055/1079. First visible prose
167.929377s,first complete tool173.367516s. OpenCode accepts whole-turn-validated
buffered tool deltas;full prompt replay/first output latency remain substantial.
No throughput ceiling/cache/prefill optimization claim. HTTP lacks Bench footer,
so only wire timings/usage/process metrics are reported;no profiling flags used.

Actual client unsupported output5000 and context overflow are non-retryable400
with actionable errors. Client SIGINT after admission exits-2;active/queued drain
in0.542347s,and workflow thereafter succeeds. Bench load-only ownership probe
exits1 before allocating another model. Final server SIGTERM drain/exit0. Metal
5291294720,loaded physical footprint1157421032,drained5230895984,every2s sampled
peak RSS5087821824 bytes;these are not equivalent metrics or exact memory peaks.
Postpressure1/free80%;no model weights staged/copied/hashed again.

Final complete suite136 pass +3 skips (139 reported:26 Server,17 Presentation,
96 Core incl skips),release35.35s;focused precision/config6 pass plus3lease tests
and existing4 memory-config tests;checker6 pass. Historical log known-mismatch
check passes;original small reference fixture SHA preserved. No fresh strict
parity replay or numerical-equivalence claim. Core tracked diff only2 ownership
lines per Q4 runner,new resource-only lease and manifest registrations;all math/
state/sampling/precision,teacher harness,dependency pins and expected tokens
unchanged. Real run used pre-model release35.31s;post-review bound precision fix
changes comparisons outside actual OpenCode safe-integer operands,not schema text
or inference,so final focused/full/release checks suffice without costly replay.

Updated current mission statuses,architecture,recovery,integration/runtime guide
and evidence,while retaining historical stage/readers/numerical records. Current
S2-02 stop is explicit;S2-03 UI,Keychain/external-owner controller,and prefix/cache
work remain proposed. Task changes remain uncommitted for review;no push/merge.
See evidence/2026-11-17-s202/README.md and validation.json for commands,artifacts,
failed attempt,security/memory/numerical limitations. Final docs/diff checks follow.

Final S2-02 review: all27 Markdown local-link/anchor checks pass;git diff --check
passes. Reviewed task source/tests/docs/public synthetic evidence only;no keys,
weights,model databases or generated binaries. Both runtime key files removed.
Historical broad S3-02–05 qualification remains in progress despite delivered
functional implementation;no exhaustive crash/socket/memory-soak claim. Logs
remain local under existing *.log ignore and require explicit reviewed staging
for a future checkpoint. No local commit,push or merge;S2-02 boundary reached.

## S2-03 started

Reloaded directives/TODO/recovery/architecture/log/integration and app/ChatService/
Core lease/server/tests. S2-02 recorded source hashes match;rerun136 pass +3skip,
prior release/evidence intact,diff clean. Scoped local checkpoint f2f5672 includes
reviewed synthetic evidence/aggregate logs;no remote action. Chose in-process
CoreBackend+LocalServer with exclusive native-chat/server modes for simplest
lifetime/no-orphan behavior;design/tradeoffs recorded before edits. Implement
one small testable iteration at a time,model-free first. Strict131/132 preserved.

S2-03 real UI iteration:disposable .app wrapper required because CUA cannot target
bundleless SwiftPM process. Exact release binary/resource bundles;no source/model
artifact changes. Native chat hello succeeds,then server mode unloads native model
and starts same CoreBackend/LocalServer with Keychain credential. Real JSON/SSE,
function/result/final continuation and actual OpenCode1.19.5 single response pass.
Health reports5 completed/1 rejected,Metal4.928GiB and last request phase timings.
Private export to default save target returned recoverable no-overwrite/create
error;no target overwritten;Keychain-backed loopback authentication verified
privately without token logging. Unit export/import and permissions pass.

Real Stop during admitted2000-word prefill cancels request/RemoteDisconnected and
closes listener;UI stopped/chat released. Immediate Start exposed recently used
port/TCP cleanup rejection. Preserve failure;add SO_REUSEADDR to probe and actual
listener (no SO_REUSEPORT),model-free immediate rebind test and rerun UI lifecycle.
No numerical/math/cache changes. Final parser-failure counters also distinguish
failed native output from transport cancellation;existing protocol errors unchanged.

## S2-03 complete — lifecycle, diagnostics and boundary

Actual final runtime UI tests:fixed SO_REUSEADDR release immediate Restart works
on18094,counters reset,Metal5291294720. App Quit while owning server closes listener/
exits. Reopen unloaded/stopped,port18094 persisted;native hello reload succeeds,
1 token/TTFT838ms. CLI probe while native chat owns exits1 before allocation.
After explicit native Unload,independent CLI starts0.820851s and UI Start correctly
reports occupied port. Observe CLI is authenticated Ready/Independent CLI with
Disconnect only. Observer app Quit leaves CLI generating hello;then only our own
test CLI getsSIGTERM/exits0. No unrelated process was stopped. App intentionally
remained running unloaded for this observation gate;one Q4 owner/normal pressure
verified. Wrapper app was exact release/resources,not a signed distribution.

UI-owned real requests:JSON17/1 0.925447s,SSE17/1 0.596157s,tool290/27 16.123924s,
continuation347/16 17.324563s;OpenCode1.19.5 hello client11.062162s/server210/1.
Retained server phases:admission0.280,queue0.040,format0.050,tokenize4.377,
prefill9479.468,first generated9490.340,first visibleSSE9490.514,
decode-after-first44.260,overall9534.786 milliseconds. Last record is no-tool;
parser buffering validated with focused mock timing test,not invented real value.
Existing28m17s/167.9s/173.4s S2-02 baseline retained,no speedup/optimization claim.

Final guidance review adds the bounded fieldfare agent/permission contract and
JSON-safe observed model ID/context. Exact generated JSON,including quoted model
ID,passes installed OpenCode1.19.5 debug config with isolated XDG/no real key/no
model. This configuration-text-only refinement after model integration gets
full tests/release,not redundant model replay. Native inference/server lifecycle
source is unchanged from corrected real-tested release. All Core/Bench/pins,
teacher harness/reference/baseline byte-identical to f2f5672;small reference SHA
verified. No fresh strict parity:131/132 remains release blocker;historical checker
is not new equivalence evidence. No expected tokens/tests weakened.

Final150 package pass +3 skips (153:30 Server,10 Management,17 Presentation,
96 Core incl skips),release36.68s,checker6. Current docs describe implemented
versus proposed controls,Keychain/private-file auth,read-only CLI operation,
actual telemetry boundaries and limitations. Remaining fatal/load/GPU/immutable
checkpoint/soak/numerical acceptance issues explicit. Final doc/diff/artifact gates
and scoped local checkpoint follow;no push/merge and stop after S2-03.
Temporary private client key and disposable app wrapper removed. App credential
persists only in Keychain as designed;no model/secret artifact in Git evidence.

Final S2-03 review:all28 Markdown local links/anchors andgit diff --check pass;
final source/release hashes match validation inventory. Core/Bench/pins/reference
protected diff empty. Scoped public synthetic text/aggregate evidence only;no
keys,weights,model database or generated binaries. Local checkpoint includes
reviewed ignored .log files explicitly. Stop S2-03;no push/merge/performance work.

S2-03 local checkpoint created with title `feat: manage exclusive local server
mode in native macOS app`;recovery bookkeeping included in the same unpublished
commit. S2-02 remains f2f5672. Working tree clean after checkpoint;no remote
push/merge. S2-03 stop reached;next performance milestone not started.

## S3-00 started

Restored directives/plans/architecture/recovery/integration/benchmark procedure
and source. Clean83d28e2 baseline (S2-02 f2f5672). Both server entry points use
CoreBackend -> QwenQ4ActiveExpertStreamingRunner.generateResult,serialized by
Admission. Native chat remains separate/exclusive. Before reuse implementation,
add diagnostic-only replay counters/memory capture and record repeated-turn
control;Core numerical sources remain byte-identical for that baseline.

State review:forwardToken consumes token atabsolute position;generation emits a
prediction beforefeeding it back. EOS is unconsumed;onEOSall returned non-EOS
tokens were consumed,onoutput/context limit lastreturnedtoken is pending.
KV append positions,DeltaNet andConvall must match consumedIDs. No recurrent
rewind exists. Ordinaryhistorystrips empty thinking preamble fromearlier assistant
turns;tool steps aftersame lastuser canpreserve it,but canonicalcall whitespace
maydiffer. Full exacttokenprefix checkmust refuseany ahead/divergentstate;LCP
alone is insufficient. No state snapshot or altered template is authorized by
assumption; first attempt will keep existing stateinplace and reportactual misses.

S3-00 original replay baseline completes before cache implementation:one retained
server,warmup+11 measured requests,baseline Core/Bench/pins unchanged vs83d28e2.
Short30/47 prompt tokens;initial826/849,3282/3305,5010 histories. Three sequential
actual model audit_chunk calls(1141/25,1255/25,1369/25) and final tool result answer
pass;cycle193.193920s,total712.904308s,server exit0. Initial fixture labels were
rough targets,notactualcounts;calibrated1K/4K/6K supplemental baseline nowruns on
samefrozenoriginalbinary. Fullrequest/publicsyntheticresponse/timing/expertlogical
read/footprint records preserved. No physicalSSDtraffic or decodeceiling claim.
Initialswap279.69MiB unchanged atfirstpostflight;free85%before/80%after. Current
pressure1verified during calibrated run;initial preflight capturedfreepercentage
but didnotcapturethe numericpressurelevel,which remains unavailable for that
initial point. HardwareM5/32GB/macOS27.2.1/Swift6.4.2,134GiBdisk,noothermodelowner.
Two model-freetemplateboundary tests andexplicitgeneration consumed-token/EOS/
limit/context/sampling-position test pass. Currentstatecontract document records
these invariants andstrictno-rewind rules;noactualcachecode yet.

S3-00 calibrated replay baseline completed:1026/1049 and4038/4061 prompt tokens,
6144 longer history; total619.645961s,exit0. Control preserved separately.
Implemented in-place consumed-token ledger and active-runner integration; tensor
forward body is unchanged (proof artifact). Defaultoff,strict full-prefix compare,
no rewind,normalEOS staging then decoder/tool/HTTP terminalflush commit. Epoch
clear blocks in-flight publication. CLI/sharedUI settings/clear/diagnostics added;
legacy persisted settings preserve values and defaultoff. Candidate package160pass
+3skip,release36.67s before latest UI/harness additions. Focused UI/model-free
checks pass; additional lifecycle tests pending final suite.

First real exact-prefix HTTP tool trial:3 sequential natural hits,reused1166/
1280/1394,fresh89/89/88,correct native calls and finalDONE,cycle54.005739s versus
original193.193920s. This is one functional observation,not a promoted estimate.
Token capacity10208bytes;no KV/recurrent/Metal snapshot. Raw-token equivalence
and5 alternating tool-cycle pairs now running in a single runner with identical
prompt arrays,greedy,empty expert cache+same warmup before each mode sequence.
Release harness build43.22s; preflight normalpressure1/free86%,swap279.69MiB,
AC80%,no competing model. Stop qualification on any raw-token disagreement.

Raw-equivalence harness correction before qualification: original Python evidence
recorder retained the same growing messages list for earlier tool rows and the
short two-turn row. Timing,usage,responses and requests actually sent were valid;
serialized earlier request snapshots were mutated later. Preserve originals and
original workload source. Reconstruct tool turns by2/4/6/8 message boundaries from
final public history,keep original ordered schema fields,and assert actual
1141/1255/1369/1482 token counts before model construction. Recorder now deep-copies
requests. First aborted comparison used wrong1483-token prompts (expected-call
assertion failed); preserved aborted artifact. A second attempt rejected unrelated
high-precision telemetry in strict ToolJSON before loading any model. Isolate
ordered request-only corpus rather than weaken the production JSON parser.
Corrected harness now running; initial expected real25-token calls/counts match.
No kernel,fixture expectation or numerical non-regression rule was changed.

S3-00 qualification BLOCKED: all five alternating short tool-cycle pairs agree
on raw IDs/finish reasons (off194.04/194.30/193.69/194.05/194.36s;exact54.51/
54.63/53.87/53.95/54.35s),but first sustained512 continuation diverges at output
index465(zero-based),fullReplay1031 versus cached285. Both512/maxNewTokens;
identical430-token continuation input,399-token actual reused prefix. First tool
355/44 agrees in both modes. Stop at first mismatch;second sustained pair not run.
Test exits1 after1332.333s;do not call this qualified or promote the feature.
Pressure1 and swap279.69MiB unchanged;model process exited. No MLX fixture change.

Bounded diagnostic plan: hypothesis is existing MoE expert-residency-dependent
reduction ordering,not unsupported rewind. Decision: distinguish a consumed-state
boundary bug from full replay itself depending on prior expert-cache history.
One experiment only: regenerate the same public355/44 tool setup,then disable
prefix reuse and fully replay the same430-token continuation after clearing only
expert weights and repeating the fixed warmup. Compare its512-token digest with
both already captured results. No kernel/precision/expert-policy change,no new
broad numerical investigation. Regardless of attribution,the sustained candidate
failure blocks shipping reuse. Preserve the candidate patch/evidence and restore
safe replay if no narrowly scoped fix exists within the no-math-change mission.

Bounded diagnostic completes56.347s,off/full replay430fresh/512output. Cold-expert
replay digest540ba1ff7eee7ce8f5d48c0f878a5b5a49438a360ce9fef15d9ed1d2d3fe25b3
matches prior cached result,not warm-expert replay6ad5b1517561e95688839adc58023094ba07bf52e7c2991187126cad6663dad0.
This confirms existing replay depends on expert-residency history; subset reduction
ordering is a code-level explanation,not first-operation instrumentation proof.
Do not weaken the failed same-Swift gate or infer the original MLX cause from it.

Rejected candidate archived as complete Package/Sources/Tests patch against83d28e2,
SHA in candidate-archive.json;reverse-apply check passes before rollback. All
production Sources/Package/pins restored byte-identically to83d28e2. CLI/UI cache
controls,new telemetry/commit hooks and prototype-only tests are removed from the
active build,not shipped. Failed test and diagnostic remain in reproducible patch
and logs,not suppressed or relabelled as passing. Retain independent consumed-
boundary test and two template-framing tests. No inference/kernel/sampler/tool-
protocol/expert-policy change remains in production. S3-00 BLOCKED/not complete;
no unsafe opt-in is exposed. Final restored suite/release/checker/docs/diff pending.
Next requires separately scoped residency-independent-reduction design and gates
or independent numerical-policy review;no implicit math change under this task.

Final restored-tree gates pass:Scripts/test.sh153pass+3opt-in skips (156total:
32Server/10Management/17Presentation/97Core including skips),debugbuild11.41s;
release41.88s;checker6/6;historical131/132 log stability;Markdown30files anddiff
checks. Sources/Package/pins byte-identical83d28e2;originalfixture SHA unchanged.
No new MLX replay or candidate-equivalence claim. Final no model owner,pressure1,
free87%,swap271.69MiB versus279.69MiB preflight. Complete report/derived paired
statistics/failed log/diagnostic/patch retained. Safe investigation checkpoint
only;S3-00 remains blocked,cache feature not delivered. Final Git review and
local docs/evidence/boundary-test commit follow;no push or merge.

Final cumulative review: only documentation,public benchmark/report scripts,three
independent boundary tests and reviewed evidence remain. No generated model,
weight,symlink or literal high-entropy bearer credential in63 evidence files.
Candidate patch is explicitly rejected/reproduction-only. Scoped local checkpoint
message:docs: record blocked exact-prefix qualification and replay-history sensitivity.
Git log owns its resulting ID;no push/merge. S3-00 stays blocked and work stops.

Staged diff review caught whitespace inherent in unified-patch blank context lines
when archiving the patch as text. Store exact bytes as deterministicgzip instead;
uncompressed SHA remains unchanged,decompression verified,no patch content edit
or diff-check exclusion. Re-run full staged diff/Markdown checks after this
container-only correction. The candidate failure remains visible in text logs.
