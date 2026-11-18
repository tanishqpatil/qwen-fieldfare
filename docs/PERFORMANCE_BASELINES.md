# Performance and correctness baselines

## S3-02b matched default-sampled decode

At 512 outputs, the Set and bitmap variants averaged 37.380 and 34.073
ms/token and 26.75 and 29.35 tok/s. Selector timing fell from 3.267 to
0.211 ms/token. Five primary pairs show an 11.82% latency reduction
(95% CI 6.60–17.05%). The sustained pairs give 8.70% and 8.99%.
On the M5/32 GiB system, both release variants use the 8K/5 GiB plan and
the same seeded app-default output history. OS file-cache variance remains.
See the [pairs and confidence interval](evidence/2026-11-18-s302b/README.md).

## S3-02a diagnostic sampled-decode baseline

| Sampler and output target | Inter-token mean | Selector thread CPU | Selector elapsed share |
| --- | ---: | ---: | ---: |
| App defaults, seeded, 128 | 32.499 ms | 3.199 ms | 9.85% |
| App defaults, seeded, 512 | 36.767 ms | 3.124 ms | 8.50% |
| Exact greedy, 128 | 29.702 ms | 0.028 ms | 0.098% |
| Exact greedy, 512 | 33.617 ms | 0.030 ms | 0.091% |

Each sampler used three pairs at 128 outputs and four pairs at 512 outputs
on the M5 with 8,192 context, 5 GiB, and 2,183 expert slots. The exact-greedy
control has a different output and expert history. The measurements
are phase attribution with uncontrolled OS file-cache history. See
[reproduction and per-run data](evidence/2026-11-18-s302a/README.md).

The initial S3-00 candidate was reverted on 2026-11-18. Five short
same-Swift tool-cycle pairs matched and measured73.126% paired mean prefill
reduction,then the first sustained512 continuation diverged at output465.
Full-replay-only cold/warm expert-history diagnostic reproduced the difference.
The complete production candidate was reverted;no cache controls ship. See
[all measurements,failed gates and limitations](evidence/2026-11-18-s300/README.md).
Strict MLX131/132 remains unresolved. Dated prior status below is historical.

S1-01d completed the fallback reader migration on 2026-11-17. Latest gates and separate fallback smoke
integration are in [S1-01d evidence](evidence/2026-11-17-s101d/README.md). Strict
active-streaming131/132 remains unresolved. Dated sections below preserve prior
results/status; they are not instructions to restart numerical investigation.

## Evidence policy

Record commit/build identity; chip/GPU/RAM; OS/Swift; checkpoint hashes and
reference versions; exact command; sampler/seed; prompt/output lengths; selected
and occupied context; prefill/TTFT/decode/total and repeated-turn latency; cache
hits/misses and logical fetch bytes; peak Metal allocation/physical footprint;
pressure and before/after swap; correctness; protocol deviations. Mark missing
fields unavailable. Logical fetch bytes are not measured physical SSD traffic.

Follow [community procedure](COMMUNITY_BENCHMARKS.md) and
[controlled promotion gates](OPTIMIZATION_PLAN.md#common-promotion-protocol).
Diagnostic, hashing/parity and page-cache-uncontrolled runs are not headline
performance comparisons. Never infer a ceiling or gain from mismatched runs.

## Historical records, not current measurements

| Evidence | Recorded outcome | Authority and limitations |
| --- | --- | --- |
| August exact-greedy A/B | 29.02 → 31.97 decode tok/s; paired +10.16%, CI +8.05% to +12.26% | OPTIMIZATION_PLAN; not sampled generation or current host performance |
| November 5 teacher forcing | 132/132 predictions | AGENT_RUNTIME_PLAN; format-2 checkpoint hashes; then MLX 0.32.0, mlx-lm 0.31.3, Transformers 5.15.0 |
| November 5 512-token greedy | 33.58 and 33.36 tok/s; 4.928 GiB Metal | AGENT_RUNTIME_PLAN; 21-token prompt, 8K, 2,183 slots, uncontrolled OS cache |
| November 5 context allocation | 32K 4.904 GiB; 64K 4.937 GiB Metal | Short 22-prompt/16-output runs, not filled-context tests; pre-run swap missing |
| Removed unused-head prefill candidate | Failed 20% TTFT gate | AGENT_RUNTIME_PLAN; candidate reverted, no retained gain |

Detailed historical commands, footers and missing metadata remain in
[active-streaming log](QWEN_ACTIVE_STREAMING.md) and
[prior plan](AGENT_RUNTIME_PLAN.md). These summaries are not new complete
benchmark reports. Historical full parity fixtures are not committed.

## Current audit baseline — 2026-11-17

Base `72fb923`, branch `feature/local-coding-agent-server`, clean before audit.
MacBook Pro Mac17,2, Apple M5, 32 GB unified memory; macOS 27.2.1 build 26B131;
Apple Swift 6.4.2 (`swiftlang-6.4.2.1.9`, clang-2100.4.1.1), arm64.
Disk reported 136 GiB available; initial `memory_pressure -Q` reported 85% free;
no conflicting model process found. GPU core count not recorded.

`Scripts/test.sh`: exit 0; 32 Core tests and 17 Presentation tests passed (49
total, parameterized cases counted by Swift Testing). No checkpoint required;
attention bridge and cache tests do require working Metal. Build/test summary
is recorded separately from inference performance. Initial sandbox-only attempt
failed on Swift cache writes; rerun with expanded access succeeded.

`swift build -c release`: exit 0, 41.68 seconds. Build duration is not inference
latency. Direct dependency resolves swift-transformers 1.3.3; all other resolved
versions remain pinned in Package.resolved. No dependency changed in this audit.

No current throughput, multi-turn latency, Metal/process peak, SSD traffic or
swap-delta measurement is claimed by this foundation entry. Current reference
validation outcome is recorded in DEVELOPMENT_LOG; a changed MLX environment
must not be treated as the historical oracle without checking parity.

### Current parity failure

Fresh format-2 reference, identical public 21-token prompt, 132 expected outputs;
checkpoint SHA-256 verified. Swift teacher forcing exit 1: **131/132**, position
61 expected 14988, actual 10000. Current MLX 0.32.4, mlx-lm 0.32.2, Transformers
5.22.2. Root cause unknown. Stage 0 is blocked; Stage 1 not started.

[Reference fixture](evidence/2026-11-17-parity/reference.json),
[Swift result](evidence/2026-11-17-parity/swift-parity.txt),
[MLX output](evidence/2026-11-17-parity/mlx-reference.txt),
[package results](evidence/2026-11-17-parity/package-tests.txt),
[release build](evidence/2026-11-17-parity/release-build.txt).

The reference log reports 8,841.21 ms entire MLX generation including prefill,
14.93 tok/s and 19.615 decimal GB peak; this is a correctness run with hashing,
not a comparable decode benchmark. Swift reports only loaded snapshots here:
4.928 GiB Metal, 0.996 GiB footprint and 0.124 s load. Full request/cache footer,
physical SSD traffic, peak process series and pre/post swap are unavailable.
Raw 2,183-slot override, 8K capacity/21 occupied prompt tokens, no profiling,
no warmup, one run; OS cache uncontrolled/hash-warmed. No memory-regression or
speed comparison claimed. Reproduce from commands in DEVELOPMENT_LOG after
fresh exclusive-process/normal-pressure/disk/toolchain checks.

### Follow-up correctness investigation

[Detailed evidence and report](evidence/2026-11-17-investigation/REPORT.md):
nine full Swift replays still fail 131/132 at position 61; two MLX regenerations
reproduce the original unchanged fixture. Category D, upstream activation
cause unresolved. Raw MLX logits 14988=27.875, 10000=27.75; initial Swift FP32
14988=27.724899292, 10000=27.845386505. Same-input MLX head agrees with Swift
Top-5 within 0.00000190735, locating the observed reversal upstream of the head.
Final normalized hidden relative L2 difference 2.272128%; embedding BF16 rounding
and layer-0 divergence demonstrated, but not proven to explain the full failure.

5 GiB/2,183 slots: 4.927902 GiB Metal, 4.791436–4.794808 GiB footprint at sampled
position 80. 3 GiB/969 slots: 2.927292 GiB Metal, 2.789924–2.793311 GiB footprint.
Snapshots only, not peaks. No changed allocation path retained; final executable
hash matches control exactly. No performance comparison or harmlessness claim.
Restored release build and 49 tests pass; five malformed-reference cases correctly
fail; strict 132/132 remains unsatisfied. Stage 1 not started.

### Numerical root-cause isolation (same date, not performance work)

[Contract](NUMERICAL_CONTRACT.md), [report](evidence/2026-11-17-isolation/REPORT.md).
Four further full replays still fail 131/132. Earliest difference is initial
embedding at position 0. Native MLX recurrent state is FP32, while Conv history
and activations are BF16. Identical-input FP32 layer-0 RMSNorm/projection/Conv/
recurrence/router/expert tests agree closely; embedding-only rounding does not
repair the full parity result. Seven selected layer-0 expert sets match.
Overall full-model cause remains unresolved; strict gate unchanged.

Restored executable hash is unchanged, release build passes (18.11 s) and full
suite passes 49/49. Instrumentation used bounded CPU/file snapshots and waits;
no production source or allocation change retained. Loaded Metal remains 4.928
GiB (5-GiB plan) and 2.927 GiB (3-GiB plan), not measured request peaks. MLX
reference peak 19,631,363,782 bytes. No latency/throughput/swap-delta comparison.

### Revised development policy and S1-03a

ADR-008 preserves strict 131/132 as an unresolved release blocker while permitting
independent reliability work. [Known-issue checker and evidence reuse](NUMERICAL_NON_REGRESSION.md)
are separate from strict parity. S1-03a changes only metadata selection validation;
numerical source inventory/bytes and dependency pins match the preserved control.
New benchmark executable SHA-256 is
49cf77927f0a747ba395a3918fa74796f9b66aed3763399fe3cafd102065ef26;
**binary identity is not claimed**. No model was run or weights hashed in this
iteration; no new numerical, speed or memory measurement is claimed.

Full package 55/55, six model-free checker tests, focused metadata suite and release
build pass. [Iteration evidence](evidence/2026-11-17-stage1-metadata/validation.json).
Current known strict failure remains documented, unaltered and unresolved.

### S1-01b — one CPU reader migration

[Validation](evidence/2026-11-17-s101b/validation.json) and raw logs preserve one
checkpoint-backed CPU reader test and one separate release parity replay. Actual
headers 88,906/67,237/67,875/50,269 B fit retained 100,000,000 B cap; a small norm
read passes. No full-shard copies or mmap change introduced. Default full suite:
83 passed, one opt-in test skipped; that test passes in its explicit checkpoint
run. Release passes, as do six checker unit tests.

Strict replay: 21 prompt/132 teacher-target positions, 8,192 context, 5-GiB plan,
2,183 slots, raw storage, no profiling, greedy. Exit 1; known 131/132 only (token
61 expected14988/actual10000). Loaded snapshot: 4.928 GiB Metal, 0.996 GiB
footprint, 0.526 s load. Those are not request peaks/TTFT/throughput. The teacher
mode exits before ordinary timing/cache footer; no full footer, pressure/swap
series, physical SSD traffic or unload-soak result is available. Required hashes
verified once; hashing warms OS cache. No performance/memory-regression claim.
M5/32 GB/10 GPU cores, macOS27.2.1/26B131, Swift6.4.2; 133 GiB disk free and
normal pressure/no other model owner before sequential checks. Runtime Q4 math
and readers remain unchanged; CPU reader behavior has independent test coverage.

## S3-00a expert-cache residency prototype

Five alternating 128-output pairs measured −2.76% mean decode-rate change
(95% interval −7.58% to +2.07%). Two 512-output pairs measured −1.60% and
+0.64%. The prototype adds one 65,536-byte Metal buffer. These measurements
support characterization of a numerical issue. Production remains on the
existing reduction path. [Host, full runs, and limits](evidence/2026-11-18-s300a/README.md).
