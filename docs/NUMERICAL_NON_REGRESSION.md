# Known numerical issue and non-regression gates

Revised policy accepted explicitly on 2026-11-17 (ADR-008). The original
teacher fixture and strict harness remain unchanged. Current known result is
**131/132**, reference token 61 expected 14988, actual 10000. This is unresolved,
not full parity. Strict 132/132 and independently reviewed numerical acceptance
remain release/production-readiness blockers, not global engineering blockers.

## Separate known-issue stability check

```bash
python3 Scripts/test_numerical_non_regression.py
python3 Scripts/check_numerical_non_regression.py \
  --log docs/evidence/2026-11-17-isolation/restored-parity.log \
  --unchanged-inference
```

The first command is model-free testing of the checker. The second verifies the
unchanged original fixture SHA-256; one complete checkpoint-verified 132-position
log; consistent mismatch count; and no mismatches outside the precise known
position/expected/actual triple. Duplicate, incomplete, malformed or changed
records fail. Elimination of a mismatch is non-regression, not automatic release
qualification. A known-failure result is explicitly printed as such; the original
strict benchmark still exits nonzero, and its six-case test is not weakened.

`--unchanged-inference` also compares the numerical source inventory and bytes,
benchmark source, package definition and dependency pins with the recorded base
commit in `numerical_non_regression_baseline.json`. New numerical files or changed
files fail this fast path. Only the named model-directory selection validator is
excluded for S1-03a; it is not called by benchmark inference. This exception must
not be expanded casually or used if that validator is made part of inference.
Review call sites and scope before using it for another task.

This checker reads existing evidence; it **does not execute a model or rehash
weights**. Its log result must never be presented as a new model measurement.
When a completed release benchmark also has the identical executable SHA-256
under the unchanged toolchain/checkpoint, numerical behavior was not changed by
that binary. For selection-only work, this supplies stronger non-regression
support than repeating expensive deterministic controls. System/environment drift
and nondeterminism are not proven absent by a source fingerprint alone.

## Changes affecting inference

A matching mismatch count alone cannot demonstrate that logits/state/behavior
have not materially worsened. If numerical sources, model IO used by inference,
allocation/state ownership, dependency implementation, compiler or relevant
execution settings change, this source-identity shortcut is insufficient.

Run model-free/fault tests first, then the bounded relevant model-backed check
and Stage integration gate. Use the same fixture/tokens/context/sampler and
compare the entire mismatch set, not just the count. Where a change can affect
arithmetic or state, compare matched inputs, captured scores/margins and state
against preserved controls and relevant behavioral tests. Existing evidence has
no approved universal logit/state tolerance; obtain review of materially changed
behavior instead of treating a token-only pass as permission. Performance-sensitive
changes still require the existing reference/behavioral and promotion gates.

Keep strict failure output alongside known-issue stability reports. Do not edit
expected tokens, suppress failures, skip mandatory checkpoint verification, or
relabel the original benchmark result. Reuse identity/integrity evidence only
where applicable. The current teacher CLI hashes its checkpoint as required;
this policy does not add a hash-bypass flag. Reserve complete model runs/hashing
for changed inference/identity/integrity paths and stage integration gates.

No further broad numerical/layer-3 experiment is scheduled without a direct
implementation need or a bounded hypothesis and an actionable decision.

## Isolated S1-01a parser foundation

The added internal parser has no call sites in existing runtime/benchmark/pack
sources. All previously fingerprinted implementation sources and dependency pins
remain byte-identical; Package.swift only registers the new source and tests.
The old --unchanged-inference inventory fast path cannot pass with a new Core
file/manifest change. Its baseline/checker are deliberately **not** relaxed or
silently regenerated. Manual scoped source/diff review justifies this user's
explicit allowance for an isolated parser-only change without a model replay.
The lightweight archived-log check is evidence reuse, not numerical equivalence.
Once a consumer adopts the validator, changed IO/ownership behavior requires its
appropriate runtime/behavioral and bounded model-backed integration checks.

## S1-01b reader integration gate

QwenResidentWeights now consumes the shared validator; its selected-payload reads
use retained descriptors. One opted-in local checkpoint test validates all four
headers plus a small norm tensor through that migrated reader. Separately, one
release benchmark replay with required checkpoint hashing returned strict exit 1,
131/132 with exactly the known token-61 triple. The log-only checker confirms the
mismatch set, not numerical equivalence. Active Q4 reader/math sources are unchanged
and not exercised through the migrated CPU utility; neither result certifies all
load modes. Original fixture and strict scripts remain unchanged.

## S1-01c active-streaming descriptor/mapping integration

The active reader now uses shared descriptor validation and ARC mapping/Metal
buffer leases. This is an inference-I/O change; `--unchanged-inference` is not an
applicable shortcut and its baseline/exceptions were not changed. Kernels,
sampling and precision are unchanged. One actual four-shard mapping integration
and one mandatory checkpoint-verified teacher replay completed. Strict result
remains131/132, exit1, same token61 expected14988/actual10000. The separate log
checker reports no new mismatches, not numerical equivalence or full parity.
[Evidence](evidence/2026-11-17-s101c/validation.json) includes hashes, exact gate
settings, unchanged source inventory and remaining immutable-file limitations.
No broad numerical investigation or tolerance change was performed. Strict132/132
and independently reviewed acceptance still block release/production claims.

## S1-01d fallback integration scope

QwenQ4ChatRunner now reuses the already-migrated shared mapping/descriptor path.
Active runner, shared parser/map owner, fixed-layout guard, numerical selection,
shaders and Bench remain unchanged in this task. The teacher CLI constructs only
active-streaming; no new strict replay or full checkpoint hashing was required
or performed. Its retained S1-01c131/132,exit1 remains the known release blocker.
The log checker reuses that evidence and does not execute or validate fallback.
A separate bounded memory-saver integration (context8,one-layer cache,two reset
forwards) passed. It is IO/state smoke evidence, not MLX parity or a new numerical
acceptance policy. [Validation](evidence/2026-11-17-s101d/validation.json) records
the exact scope and limitations. No source-inventory baseline or strict test is
weakened to accommodate the migration.

## S3-01a bounded projection batching gate

The eight-row candidate preserved per-layer causal positions and recurrent
ordering, all tested router IDs and weight bits, and full state digests at
uneven and context boundaries. It matched the tokenwise reference on the
430-token, 512-output continuation and five logit/hidden samples. The
independent S3-00d digest at output 465 also matched. The original fixture
and required checkpoint verification both ran on the candidate. Its
strict result remains 131/132 with the sole original mismatch. See the
[qualification record](evidence/2026-11-18-s301a/README.md).


## S3-01b router-readback timing scope

S3-01b used a temporary observer on the tokenwise schedule. Instrumented
and uninstrumented short, 1K and exact-prefix suffix cases had identical
first output, full logits, hidden state and cache demand. The observer
was removed and no batch scheduler was implemented. The stable source
and strict fixture match the baseline. Archived checkpoint-verified
131/132 is still the release blocker at token 61.
See the [timing qualification](evidence/2026-11-18-s301b/README.md).
