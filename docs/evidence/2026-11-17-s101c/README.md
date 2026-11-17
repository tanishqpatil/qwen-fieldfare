# S1-01c validation evidence

Active-streaming Safetensors migration only. Base commit `72fb923`; candidate is
an uncommitted local tree. No weights copied, staged, mutated or uploaded. No
kernel, sampling, quantization or precision changes. Fallback Q4ChatRunner and
pack-builder readers remain unmigrated.

## Commands and results

```bash
Scripts/test.sh --filter QwenMappedSafetensorsTests
Scripts/test.sh
swift build -c release
python3 Scripts/test_numerical_non_regression.py
QWENFIELDFARE_READER_CHECKPOINT="$MODEL_DIR" Scripts/test.sh --filter localCheckpointMappingIntegration
.build/release/QwenFieldfareBench --model "$MODEL_DIR" --context 8192 --memory-gib 5 --teacher-reference docs/evidence/2026-11-17-parity/reference.json
python3 Scripts/check_numerical_non_regression.py --log docs/evidence/2026-11-17-s101c/strict-parity.log
ruby Scripts/check_markdown_links.rb
git diff --check
```

`MODEL_DIR` was the existing `/Users/tanishq/models/Qwen3.6-35B-A3B-4bit`.
Preflight checked host/toolchain, free disk, normal pressure and no competing
model processes; see [preflight](preflight.log). One model process at a time.

- [Focused](focused-tests.log): 16 passed, one opt-in skipped (17 reported).
- [Package](package-tests.log): 99 passed, two opt-in skipped (101 reported across
  84 Core + 17 Presentation). No package failures.
- [Release](release-build.log): passed, 33.29s.
- [Checker unit tests](checker-tests.log): six passed.
- [Checkpoint mapping integration](checkpoint-integration.log): one passed;
  four validated/mapped files and LM-head bytesNoCopy buffer, no whole-shard copy.
- [Strict parity](strict-parity.log): mandatory checkpoint SHA-256 verification
  performed once, **131/132, exit 1**. Original one-based token61 expected14988,
  actual10000. [Exit status](strict-parity-exit.txt) is preserved, not suppressed.
- [Known-issue checker](known-issue-check.log): same complete mismatch set, no new
  mismatch records. This does not prove numerical equivalence or strict parity.
- Markdown local links/anchors and diff whitespace passed; see [checks](checks.log).

Exact identity, settings, source hashes and limitations: [manifest](validation.json).
Host: MacBook Pro Mac17,2, Apple M5, 10 GPU cores, 32GB; macOS27.2.1/26B131;
Swift6.4.2. Normal pressure1, 87% free, 133GiB disk. Fixture prompt tokens and
132 teacher tokens recorded in manifest. Greedy42, context8192, raw experts,
5GiB/2183 slots, profiling disabled. Load0.125s, loaded footprint0.996GiB,
Metal allocation4.928GiB, planned5.000GiB. These are load snapshots, not peak
memory. Teacher mode exits at parity failure before normal cache/timing footer;
no throughput result or performance promotion is claimed. No repeated checkpoint
hashing or additional model replay was performed. No protocol flags bypassed.

## Ownership and remaining limitations

Runner and Metal buffer leases retain a whole-file mapping owner, which retains
the shared validated descriptor. Metal command references prevent unmapping during
GPU use, including owner release. ARC failure/unload tests release mappings and
FDs. Page-rounded mapped windows keep logical tensor accesses within actual file
bounds. Header sizes88,906/67,237/67,875/50,269 bytes justify compatibility under
the retained100,000,000-byte policy, not an expanded cap.

Mutation tests operate only on tiny fixtures: truncate, then check metadata/pread
errors and release without dereferencing the truncated map. No intentional fault
can kill the package process. This safe strategy does not test/recover SIGBUS.
Concurrent writers/truncation can still invalidate a mapping or its contents;
immutable trusted checkpoints are required until every CPU/GPU lease ends.
Descriptor identity is not pathname identity or immutability. Ancestor races,
coherent multi-shard snapshots, full pre-allocation model validation, other reader
migrations and allocator/encoder fatal paths remain open. Production readiness and
strict132/132 remain blocked. Stop after S1-01c; next proposed task is S1-01d.
