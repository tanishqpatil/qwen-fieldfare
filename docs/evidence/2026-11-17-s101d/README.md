# S1-01d validation evidence

Migrated only fallback QwenQ4ChatRunner; reused shared descriptor/header/map
implementation and existing fixed-Qwen byte/dtype guard. No changes to S1-01a/b/c
implementations, shaders, quantization, sampling, context policy or layer cache
policy. Source-audit base72fb923; validated uncommitted tree prior to local commits.

## Commands and results

```bash
Scripts/test.sh --filter QwenFallbackReaderTests
Scripts/test.sh
swift build -c release
python3 Scripts/test_numerical_non_regression.py
QWENFIELDFARE_FALLBACK_CHECKPOINT="$MODEL_DIR" Scripts/test.sh --filter localFallbackCheckpointIntegration
python3 Scripts/check_numerical_non_regression.py --log docs/evidence/2026-11-17-s101c/strict-parity.log
ruby Scripts/check_markdown_links.rb
git diff --check
```

MODEL_DIR was the existing `/Users/tanishq/models/Qwen3.6-35B-A3B-4bit`.
[Preflight](preflight.log) excluded other model processes and checked disk,
hardware/toolchain and normal pressure. Exactly one checkpoint test process.

- [Focused tests](focused-tests.log): eight passed, one opt-in skipped.
- [Full package](package-tests.log):107 passed, three opt-in skipped,110 reported
  across93 Core including skips +17 Presentation; exit0.
- [Release](release-build.log): passed33.45s.
- [Numerical checker tests](checker-tests.log): six passed.
- [Fallback integration](checkpoint-integration.log): one passed8.110s,
  [exit0](integration-exit.txt); no checkpoint mutation/full-file copying.
- [Known-issue checker](known-issue-check.log): historical S1-01c log only,
  not a new model measurement, strict pass or fallback-parity result.
- [Documentation/diff/artifact checks](checks.log): see final review record.

Fallback settings: memorySaver,context8,configured layer cache1,budget5GiB,
hard-limit6GiB. Those budget fields are not implemented process caps. Avoided
fastResident40 and default cache4 to keep this integration bounded. Two separate
reset forwards, one input token248045 each, predictions846/846; no generated
sequence or public-prompt text completion. Cache80 creations/misses,79 evictions,
0 hits. All40 layers are exercised each forward, including lazy reload/eviction.
This is an IO/reset smoke test, not MLX equivalence or broad behavioral validation.

Host: MacBook Pro Mac17,2,M5,32GB,10GPU cores; macOS27.2.1(26B131),Swift6.4.2;
133GiB disk. Normal pressure1 before/after, free86%/85%. Footprint/Metal byte
snapshots: before9,618,224/65,536; loaded88,100,152/645,824,512; after forwards
121,687,376/1,117,323,264; after unload34,751,352/720,896. Snapshots are not peak
GPU/system memory or a full lifecycle soak. No headline timing/cache throughput
footer exists for this test; creation/eviction counts above are its complete
available cache metrics. No throughput or default/fast-resident memory claim.

The strict teacher harness constructs only QwenQ4ActiveExpertStreamingRunner.
Fallback migration does not alter that execution path. No new strict replay or
expensive checkpoint hash was performed or bypassed. Existing S1-01c checkpoint-
verified131/132,exit1,token61 expected14988/actual10000 remains the release blocker;
it does not validate fallback numerics. Original fixture/hash and evidence intact.
Header cap remains100,000,000; previously measured compatible headers are88906/
67237/67875/50269 bytes. No cap reduction is bundled with ownership work.

## Review boundaries and limitations

RAII map/file owners are retained by no-copy Metal buffers and pending GPU
resource references. Failed-init fixture retries leave no checkpoint descriptors;
map/Metal lease tests verify correct byte offsets and cleanup. Context/allocation
arithmetic is checked, constants/state failures recoverable. Broader encoder/
command failure recovery, full pre-allocation model contract, runtime index
bounds and lifecycle soak remain pending. Concurrent file writes/truncation can
still change results or SIGBUS; immutable trusted files are required while any
CPU/GPU lease survives. Descriptor pinning is not content immutability or a root-
ancestor race guarantee. Mutation fixtures never dereference a truncated mapping.
Pack-builder Safetensors input is the remaining unmigrated reader.

Cumulative review found only authorized foundation/docs/numerical evidence and
Stage1 reader-safety changes. Archived diagnostic patches are evidence, not
applied kernels. Text evidence is about1.5MB; .DS_Store, Python/build caches and
weights are excluded. Ignored *.log evidence is explicitly included in the local
checkpoint. Exact-path .gitattributes whitespace exceptions preserve two archived
patches and one raw docstring capture byte-for-byte; ordinary source/docs checks
remain enabled. See [manifest](validation.json) for hashes and exact commit scopes.
Stop after S1-01d; next proposed task S1-01e, not implemented. No push/merge.
