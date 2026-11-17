# Agent recovery memory

## Current milestone — S2-01 structured tool contract

Explicit user authorization supersedes the prior S2-00 stop. Preserve its server
checkpoint and all reader/numerical evidence. Implement native Qwen tools, JSON
and SSE translation, matched tool results and one real continuation. Never execute
tools in the server. Stop after S2-01; no prefix reuse/readers/Hermes/push/merge.


## S2-01 completed — recovery handoff

S2-00 was checkpointed locally as c001a78 before tool work. S2-01 now implements
native function-tool schemas/history/ID validation, exact checkpoint Jinja tool
formatting, bounded parser and OpenAI JSON/SSE calls/results. Stop after this
milestone. Next is OpenCode provider/client tool-cycle qualification, not a new
filesystem agent, prefix reuse or reader migration. Remaining Stage1 stays deferred.

New files ServerSupport ToolJSON/Tools/ToolPrompt/NativeTools plus parser/golden/
HTTP tests and test_server_tools.py. Existing resolved Jinja2.3.6 made direct;
no dependency pin upgraded. Core math/precision/state/cache and original fixture
unchanged. Ordinary no-tool requests retain old Core formatter. Jinja local vs
current official templates differ only in training-generation annotations; three
Python3/Jinja3.1.6 goldens prove identical text rendering for tested tool cases.
Local template SHA e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259.
Developer unsupported by actual template; reject. Advanced history must finish
with user or complete matching tool results. Client results must be in call order
because native template has no call IDs; duplicate/unmatched/pending IDs reject.

Safe prose streams incrementally; partial delimiter held. Structured tail <=256KiB,
<=8 calls commits only on real EOS and whole-turn/schema validity. Output limit/
cancellation/malformed blocks never publish calls; errors502 (SSE error then close
if already begun), no successful DONE. Stable IDs, zero-based indices, Unicode-safe
argument fragments after commit. Delayed tool deltas intentional. Server executes
nothing. Only auto/none,strict:false supported. Schema subset and structural-
delimiter-bearing tool text rejection are documented in AGENT_INTEGRATION; results
remain semantically untrusted. No generic schema/vision/developer/required-choice.

Final focused14 pass; full127 pass +3 opt-ins skipped (130 reported:20 Server,
17 Presentation,93 Core including skips),release34.36s,checker6 pass. Real single
server test passed both JSON and SSE tool cycles: project fieldfare,model requests
get_project_status; CLIENT supplies status building-local-agent; model final answer
uses it. Calls344 prompt/27 completion tokens, continuations404/13,stop. JSON times
16.196s call/16.985s continuation;SSE15.313/17.073s. Actual public test exchanges
and aggregate metrics preserved in evidence/2026-11-17-s201; server logs no content.
No-tool JSON/SSE,none,schema context overflow,disconnect recovery,shutdown pass.
Load0.775s,ready1.310s,shutdown0.088s. Metal5291294720 bytes,loaded footprint1167267816,
drained5166064424 (runner retained through drain),normal pressure1/free79% after.
Snapshots not peaks/RAM cap or optimization. Host M5 Mac17,2/32GB/10GPU,macOS27.2.1,
Swift6.4.2. No extra model runs or full weight hashing. Enum-number comparison/precision-preservation guards tightened after the enum-free real test, then focused/full/release revalidated;
real test's exercised branch and Core unchanged, no unnecessary replay.

Original strict131/132/token61 release blocker remains unresolved; log checker
reuses S1-01c evidence only, no fresh parity/equivalence claim. Existing fatal
Metal paths, immutable-checkpoint/race limitations and broad client/long-context
qualification remain. Local final commit authorized; no remote push/merge.
Historical sections below are superseded by this current completed handoff.

## Active priority override — S2-00 (2026-11-17)

Explicit user decision supersedes earlier Stage1-only/order/stop directives.
Build and qualify the minimal local QwenFieldfareServer now. S1-01e and remaining
noncritical reliability work are deferred, not completed or abandoned. Preserve
S1-01a–d and strict131/132 release blocker. Stop after real nonstream/SSE HTTP
generation; tools, prefix reuse and further reader migrations remain deferred.
No push/merge. Current implementation/validation status is in DEVELOPMENT_LOG.


Updated 2026-11-17. Read AGENTS.md, TODO.md and this file at session start.
Git source/tests/evidence outrank narrative memory. No secrets or private runtime
payloads belong here. The saved HashMap prompt is the public benchmark prompt.

## S2-00 completed — current recovery state

User explicitly reordered priorities: minimal server before deferred Stage1.
Do not resume S1-01e now. Server/ServerSupport plus six tests implement NIO HTTP1,
one retained Core active runner, one active +two queued, JSON/SSE and sampling,
loopback bearer auth,32 connections,1MiB/128messages/4096output/context reservation,
15s body deadline/16KiB watermark/10s write deadline. Every request reset/replay;
no tools or prefix reuse. Secret required in QWENFIELDFARE_API_KEY, no body/key
logging. Server flock is server-only; manual exclusion of app/Bench/MLX remains.
See AGENT_INTEGRATION for exact commands/unsupported fields and limits.

Final package113 passed +3 opt-ins skipped; release passed33.34s; checker6 pass.
Real HTTP run uses existing checkpoint,context8192,budget5GiB. JSON/SSE/reuse,
context rejection,disconnect recovery,SIGTERM during request passed;exit0.
Initial HTTP run shutdown trapped on MainActor-inherited Dispatch signal callback;
fixed by nonisolated signal installation, bounded repeat passes. Preserve initial
logs/diagnosis alongside final metrics, no raw device-identifying crash report.
Loaded Metal5291294720 bytes; loaded footprint1157519336,drained5143274256 bytes;
runner remains retained through drain until process exit. Snapshots not peaks;
no optimization or long-context/agent qualification claim. Strict131/132 remains
unresolved; no numerical sources/fixture/math or precision changes. No fresh
strict replay/hash required; checker reuses old log only. Evidence S2-00 directory.
New work is uncommitted on cf0d144; no push/merge. Stop after S2-00. Next S2-01:
validated Qwen tool contract, then real OpenCode integration; not begun here.
Remaining historical policy/status below is superseded by this override.

## Current policy and milestone (supersedes earlier stop instructions)

Explicit revised user decision: **Stage 1 reliability is authorized**. Work one
small testable task per iteration. S1-03a, S1-01a, S1-01b, S1-01c and S1-01d are complete.
Current request stops after S1-01d; next proposed task is S1-01e, pack-builder input reader.
Stop/report at the Stage 1 boundary; do not start tools/server in this milestone.
Scoped local commits are now explicitly authorized after cumulative review and
passing relevant gates. No push/merge. Stage1 remains in progress, not qualified complete.

Known strict parity is still 131/132, expected 14988/actual 10000 at token 61.
It is an unresolved release-quality blocker, **not a global development blocker**.
Preserve fixture/evidence/strict tests; never relabel as full parity. New/worsened
numerical behavior blocks affected changes. Production readiness needs independent
numerical-policy review and sufficient qualification.

No layer-3/broad numerical investigation is scheduled. Resume only for a change
requiring it or a bounded hypothesis with an actionable decision. Model-free tests
first; reuse evidence and unchanged-inference identity for unrelated tasks. Model
checks at inference-affecting changes/integration gates; no skipped mandatory
integrity checks and no repeated full checkpoint hashing without a reason.

Branch feature/local-coding-agent-server, base72fb923. Prior foundation/evidence
are preserved; S1-01d authorizes a reviewed local Git checkpoint (see latest log). Numerical evidence and local operation comparisons
below are historical facts, not instructions to stop independent development.

## Numerical isolation outcome and exact next action

Historical numerical isolation summary (current policy is above).
[Contract](NUMERICAL_CONTRACT.md) and [current isolation report](evidence/2026-11-17-isolation/REPORT.md)
are the recovery entry points. Earlier token-61 evidence remains preserved.

- Four further full parity replays (baseline trace, embedding-only BF16 rounding,
  3-GiB cache, restored production) all fail 131/132, same token 61.
- Earliest difference is input position 0 embedding, before recurrent history.
- MLX Conv history/activations are BF16; recurrent state and decay are **FP32**;
  beta BF16. Swift uses FP32; full-attention KV is Swift FP16 vs MLX BF16.
- With identical FP32 values/state, layer-0 RMSNorm matches exactly; QKV relative
  L2 ~1–2e-7, Conv ~5–7e-8, DeltaNet updated state ~3e-8–1.2e-7. Same-input
  router and expert tests also agree closely. No tested layer-0 defect shown.
- State layout is [B=1,value_head,value_row,key_col]; Swift omits B. Conv maps
  Swift [channel,4] slots 1:4 to MLX [1,3,channel]. Do not compare incompatible
  layouts or claim FP32→BF16 substitution preserves identical numerical values.
- At position 80, replacing only prior FP32 state changes next state 0.382589%
  relative L2; replacing only current native inputs changes it 0.100589%.
  These counterfactual effects are not additive causal fractions.
- MLX batching itself changes state history; sequential-only MLX is a separate
  diagnostic, not the baseline reference. Actual batched-prefix layer-0 output
  error at position 80 remains 0.438729% vs Swift.
- Layer-0 expert sets match MLX at all seven inspected positions. Cache changes
  preserve selected IDs at layers 0/3 in that sample; small reduction errors
  remain. Other layers/tokens are not certified.
- Embedding rounding alone does not restore parity. No global casts/fix adopted.
- Overall classification **4: full-model reversal unresolved**. Intentional
  numerical differences explain the observed local layer-0 trajectory; do not
  infer that the remaining full-model difference is harmless or bug-free.

At the end of the historical isolation phase, temporary production changes
were removed; that prior benchmark hash equals
f55a81ce3041544c5cd811a1c9acbc04abcb302f8b40a29fd67d2d0af2854c32. Full package
49/49 and release pass. Original fixture unchanged; no tests, source, dependencies
or acceptance policy changed. No commit/push. Strict 132/132 remains unsatisfied, retained as release blocker.

Prior result S1-01b: migrated **QwenResidentWeights CPU tensor reader only**,
not Q4 Fast Resident/active streaming or pack builder. Shared parser owns cached
FDs; subsequent selected tensor/row reads use the same descriptor. Root/basename
checks preserve contained internal links; public header size excludes prefix.
Any failed shard operation invalidates its owner; explicit retry reparses, no
automatic mutation retry. BF16/F32 values/offsets, empty reads, rows/overflow and
FD cleanup tested. Original selected CPU allocation behavior retained; no mmap
path was changed. Index JSON reader remains unchanged/unhardened here.

Current validation: 10 focused model-free tests passed (one opt-in skipped);
full suite **83 passed + 1 opt-in skipped** (84 reported); explicit checkpoint
reader integration then passes that opt-in test separately. All four headers
validate (88,906/67,237/67,875/50,269 bytes), plus a small 2,048-element norm read.
100,000,000-byte cap retained for compatibility, not claimed as a heap/RAM cap.
Release passes; six known-issue-checker unit tests pass.

Exactly **one** strict release teacher replay performed with original integrity
verification intact: exit **1**, **131/132**, only reference token 61 expected
14988/actual10000. Known mismatch-set checker passes separately, not numerical
accuracy/full-parity certification. Same original fixture/evidence preserved.
Hardware M5/32 GB/10 GPU cores, macOS27.2.1, Swift6.4.2; normal pressure/process
exclusions before both sequential checks. Logs: evidence/2026-11-17-s101b/.
No new MLX run or repeated full checkpoint hashing.

Security limits: FD identity/stat checks detect observed changes, not immutable
contents. Checkpoints must remain immutable; ancestor-directory races and
multi-shard snapshots are not solved. Path replacement never silently redirects
a cached read: original handle or controlled error. At the S1-01b boundary Q4 mmap readers were unchanged; S1-01c supersedes this for
active streaming only. No kernel/sampling/expert-cache allocation change.

Prior result **S1-01c completed**: active-streaming Q4 reader uses the shared
validated file descriptor/header and new QwenMappedSafetensorsFile. Runner owns
maps; zero-copy LM-head buffers retain maps via deallocator closure; default
command buffers retain buffer leases through GPU use. Last lease ARC unmaps,
then file owner closes FD. Partial init, mmap failures and repeated releases
are tested. Whole-file read-only/page-rounded mapping, data-relative offsets,
internal links/direct-root containment, bounded expert streaming and memory
budgets are preserved. Fixed consumer byte/dtype extents guard pointer/copy
sites; unaligned bit loads support the actual non-aligned headers without math
changes. Token/position/expert indices bounded. Expert read failure no longer
falls back to potentially faulting mmap: join reads, wait for hit work, clear
cache slots, throw typed error; abandon/reset failed request.

Validation: 16 focused pass + 1 opt-in skip; package 99 pass + 2 opt-in skips
(101 reported: 84 Core including skips, 17 Presentation); one real four-shard
mapping/LM-head integration passes; release 33.29s; six checker tests, Markdown
links and diff checks pass. Exactly one mandatory checkpoint-hashed strict replay
returns **exit 1**, 131/132, same token-61 expected14988/actual10000. Separate
known-issue checker passes mismatch-set stability only, not equivalence. Evidence:
`docs/evidence/2026-11-17-s101c/validation.json`, strict log and exit file. Fixture
SHA256 unchanged. Candidate Bench SHA256:
`38480688e1bebd6feda215e383d3af630237c7dc89d8fbab7a348f983f1e6743`.
Host M5 Mac17,2/32GB/10GPU, macOS27.2.1/26B131, Swift6.4.2; 133GiB disk, normal
pressure1/87% free before real checks. Context8192, 5GiB/2183slots, raw, greedy42,
profiling disabled; load0.125s, loaded footprint0.996GiB/Metal4.928GiB. Strict CLI
exits before normal timing/cache footer; no throughput claim or peak measurement.

Limitations: FD pins inode, not immutable contents. Checks around mmap and before
normal forward detect observed size/mtime/ctime changes only. Concurrent writes,
truncation/SIGBUS, ancestor replacement and coherent multi-shard snapshot remain
unsolved. Immutable trusted checkpoint required throughout CPU/GPU leases.
Mutation tests never intentionally dereference truncated maps; this is a safe
observed-error test strategy, not mapping-fault recovery. Active index JSON and
pre-allocation full tensor/quantization contract remain future hardening. Fallback
Q4ChatRunner and pack builder remain unmigrated; existing allocator/encoder fatal
paths are not addressed. No inference kernel/sampler/precision changes.

Latest **S1-01d completed**: fallback QwenQ4ChatRunner uses existing shared
validated descriptor/header/mapping owner and Metal no-copy leases, retaining
whole-file mappings, fast-resident40/lazy layer-cache behavior and dtype layouts.
No shared parser/map owner or active runner implementation changed in this task.
Fixed consumer extents reuse the active runner's existing internal guard, avoiding
a second table. Context-byte arithmetic is checked before allocation; original
FP32 KV sizes retained. Constant/state allocation errors now throw. Unaligned
bit loads preserve data bits. Token/position and observed-mutation checks precede
forward; immutable checkpoint, ancestor-race and concurrent SIGBUS limitations
remain. Index ingestion and full pre-allocation contract still need hardening;
encoder/command fatal paths remain. Pack builder is the remaining independent
Safetensors reader, with a different cap needing explicit compatibility review.

Validation: eight focused pass + one skip; full107 pass +three skips (110 reported,
93 Core including skips +17 Presentation); release33.45s; checker6 pass;
docs/diff checks pass. One actual fallback memory-saver integration passed:
context8, configured layer cache1, two independent reset forwards with input248045,
predictions846/846,80 cache creations/misses,79 evictions,0 hits. This is an IO/
state smoke test, not teacher parity/quality or fast-resident qualification.
Footprint/Metal bytes: before9618224/65536; loaded88100152/645824512;
after forwards121687376/1117323264; after unload34751352/720896. Snapshots, not
peaks or a full memory soak. M5 Mac17,2/32GB/10GPU, macOS27.2.1(26B131), Swift6.4.2;
133GiB disk; normal pressure1, free86% before/85% after. No full model copy/hash,
checkpoint modification, kernel/sampler/quantization/context policy change.

Teacher harness uses only active-streaming; that execution path/shared owners,
selection, shaders and Bench source remain unchanged. No new strict replay was
required or run. Existing S1-01c checkpoint-verified131/132 exit1 remains the
known release blocker. Checker reuses that historical log for exact mismatch-set
stability only; it does not certify fallback or fresh numerical equivalence.
Header cap100,000,000 retained; previously measured88906/67237/67875/50269 fit.
Evidence: `docs/evidence/2026-11-17-s101d/`.

Stop after S1-01d. Next proposed S1-01e is pack-builder input migration only,
not started. Runtime/package/tests now committed locally as8b263e0. The following local docs/
evidence commit completes the checkpoint; see git log. Current request authorizes
reviewed local commits for accumulated
Stage0 docs/evidence and Stage1 reader safety. Keep ignored diagnostic logs in
that checkpoint explicitly; exclude .DS_Store/caches/builds and weights. No
Stage2/server, push or merge. Production/strict132 acceptance remains blocked.

```bash
Scripts/test.sh
swift build -c release
# Existing model directory only; fresh normal-pressure/process/disk checks first.
.build/release/QwenFieldfareBench --model "$MODEL_DIR" \
  --context 8192 --memory-gib 5 \
  --teacher-reference docs/evidence/2026-11-17-parity/reference.json
ruby Scripts/check_markdown_links.rb
git diff --check
```

The archived isolation patch is not applied. Metrics and scripts are durable;
bounded raw state/activation captures remain transient under /tmp. Reconstruct
only in a saved/isolated diagnostic checkout, never overwrite the failing fixture.

## Locations and constraints

- [Architecture](ARCHITECTURE.md): source inventory and implemented/proposed split.
- [TODO](../TODO.md): authoritative stable mission task IDs and acceptance.
- [Development log](DEVELOPMENT_LOG.md): commands, outcomes and exact blockers.
- [Baseline/evidence](PERFORMANCE_BASELINES.md): historical/current distinction.
- [Decisions](DECISIONS.md): accepted scope and rejection evidence.
- `Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift`: production core;
  resets KV, Conv and DeltaNet each call; no prefix reuse or batched prefill.
- `QwenQ4ChatRunner.swift`: fallback modes/tokenizer; shared mapped-descriptor reader.
- `Infrastructure/ModelIO/QwenModelDirectory.swift` and
  `QwenResidentCoreLoader.swift`: CPU and both Q4 readers use shared validation; pack-builder input remains independent.
- `Mac/Generation/ChatService.swift`: UI lifecycle, detached inference, secondary
  loading path, visible/internal conversation mismatch on cancellation/error.
- Keep Qwen-only hybrid, raw weights external, SLRU, one owner, bounded 3/4/5 GiB
  allocation plans. Never truncate recurrent state as KV. Forced allocation and
  command-encoder crashes remain; full model contract/IO ownership unqualified.
- No implemented server/tool protocol/OpenCode/Hermes support. Specific local
  server is authorized for later stages; generic hosting/tools inside server are not.
- Historical head-removal prefill, cache bookkeeping, causal replacement/prefetch
  and packed speed candidates were rejected; see decisions before retrying.
