# S3-01c two-row expert I/O overlap feasibility

**Decision C — reject a runtime prototype and close this chunked-prefill
scheduling direction.** Most next-row misses have address-safe destinations
under exact SLRU replay. The modeled saving attributable to the two-row
mechanism is too small to establish headroom beyond the 20% fresh-prefill
gate. Production remains tokenwise.

This study starts at experiment commit `b49517b` and uses the stable
`2e5b993` inference source. It builds on the
[S3-01a projection result](../2026-11-18-s301a/README.md) and the
[S3-01b critical-path measurements](../2026-11-18-s301b/README.md).
The raw-weight checkpoints and default exact-prefix and numerical
contracts are unchanged.

## Exact replay and slot lifetimes

The [cache planner](../../../Sources/QwenFieldfareCore/Storage/Q4BoundedExpertCache.swift)
visits the actual router's top eight in order. It first takes the lowest
free slot, then the least-recently used unlocked probationary slot, then
an unlocked protected slot if necessary. A hit refreshes its tick and may
promote it and demote an unlocked protected slot. Ties use the lower slot.
A row locks only the destinations it has already selected. An early miss
within the row can evict an expert whose request appears later in that
row. The replay preserves that order, including exact victims, recency and
demotions.

The application submits the row's shared/hit GPU work before servicing its
misses with parallel raw reads. The hit command reads the hit subset.
The later miss/tail command reads the validated miss subset, then runs
fixed-order combine. The next router wait on the single queue currently
retires those readers before the CPU plans and writes the next row.
A future overlap schedule needs immutable row-0 GPU metadata through
retirement, separate host planning storage for row 1, and a slot write
after its last relevant reader. Two rows may share a slot for the same
expert when both only read it. A free address is useful only when SLRU
actually selected it. Redirecting an evicted expert's replacement to an
alternate address would change the observed decision.

A production tokenwise row cannot start these reads early. Its next
authoritative router outputs require that row's earlier layers. The
offline counterfactual assumes a valid layer-major schedule has already
completed both actual router packets for one layer. It then preserves
the measured expert request order. It uses no prediction or lookahead
past that explicit dependency.

The earlier traces lack per-access slots, victims and read intervals.
A disposable checkout collected only those fields and relevant GPU
and host clocks. The short and 1K workloads start from an empty cache.
The suffix includes its 18 setup rows per layer, so its 89 fresh rows
replay from actual retained state. The collector matched the S3-01b
first-output, hidden and full-logit digests. Replay independently matched
**425,920 decisions**, the actual GPU miss lists and fetched expert IDs,
all hit/miss and eviction counts, and the final slot owners, segments
and timestamps for every case. The three fresh demands also match
S3-01b exactly. See [offline results](offline-analysis.json).

## Adjacent-row access classes

The sliding counts below compare every adjacent pair within each layer.
The first fresh row has no predecessor in its respective workload.

| Fresh tokens | Adjacent layer-pairs | Pairs sharing at least one expert | A, safe next misses | A into previously empty slots | B, blocked until prior tail | Pairs with a blocked write |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 214 | 8,520 | 5,453 | 20,902 | 1,863 | 37 | 28 |
| 1,010 | 40,360 | 21,695 | 105,758 | 1,863 | 458 | 294 |
| 89 | 3,520 | 3,130 | 6,563 | 99 | 1 | 1 |

- **A.** Write the miss directly to the exact SLRU address when the prior
  row does not read that slot. CPU planning is conditional on both routers
  and on the prior row's cache decisions already being applied.
- **B.** Delay the write until the last prior reader retires. The model
  distinguishes hit-command and tail-command readers. Every observed
  conflict was with a prior miss and required the tail boundary.
- **C.** Start a B read before retirement only by loading a separate
  expert block and copying it to the planned destination after retirement.
  This is an alternative to B. The address-safe schedule needs no staging.
- **D.** An early in-place B write, rerouting to a different victim, or
  issuing reads before the real router is available violates this design.
  No disagreements or unresolved access classes remained after exact replay.

The constructive timing calculation uses only **disjoint** pairs from
the first fresh row in each layer. That prevents billing savings twice.
At 1K that is 20,200 pairs with 52,799 direct misses and 209 blocked
misses across 139 pairs. Holding two rows' complete expert unions
was never required.

## Critical-path estimate and memory

Per pair, use the captured GPU durations and individual host miss
completion offsets. In the ordered-service model, row 1 read-ahead
can start once row 0 CPU reads have joined, during the remaining shared/hit
work and tail. It can use only its own planned slots. A reader conflict
delays that individual launch until the relevant GPU retirement.
The next GPU row starts after the previous tail, and its tail needs
its own joined misses. Model row 1's existing read overlap with its
own shared/hit command before taking incremental savings. The dual-worker
sensitivity starts safe reads even before row 0's reads join, with zero
extra contention. Both models credit zero added cache planning, command
encoding, enqueue, host fence, or worker costs. These are scheduling
estimates. They are not proven hardware limits or measured runtime gains.

| Fresh tokens | Extra I/O hidden, ordered | Extra I/O hidden, ideal dual worker | Remaining row-local I/O, ordered | Gross ordered I/O plus every second-row wake |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0.307 s | 1.171 s | 3.087 s | 7.26% |
| 1,010 | 1.175 s | 1.805 s | 3.732 s | 6.19% |
| 89 | 0.093 s | 0.248 s | 0.732 s | 6.66% |

At 1K, safe paired reads have 2.713 s of aggregate remaining row-0 GPU
windows. Joining the next reads and accounting for their own existing
GPU overlap reduces the useful incremental saving to 1.175 s. The gross
wake credit includes 0.951 s measured for every second-row router wake.
Retirement timing for a future scheduler was not measured.

Adding the new modeled saving to S3-01b's disjoint GPU and row-local I/O
floor raises its already optimistic 1K reduction from **19.52% to 22.94%**.
The same-capture ordered estimate is 23.16%. The combined historical
estimate assumes all host work outside GPU and the modeled miss floor
disappears. It leaves 1.010 s above the 20% gate. Charging just the prior
0.764 s of command encoding as a sensitivity leaves 0.246 s for every
other cost. An ideal second worker raises the historical estimate to
24.78% while crediting zero bandwidth contention or added dispatch and
retirement. Independent worker performance and replacement fences are
unverified. The measured mechanism and these margins do not justify
building a scheduler.

The direct design needs no extra expert blocks or additional logical
fetches. A two-row sketch costs about 190,112 logical bytes for the
layer-major projection/hidden and packet arenas, or 262,144 bytes with
the previously described page rounding. Host-only independent planning
and command resources would be additional. With optional staging for
the five blocked blocks at the worst 1K pair, reserve **8,847,360 bytes
(8.44 MiB)**. Staging all 209 blocked misses across disjoint 1K pairs
adds **369,819,648 payload copy bytes** and about twice that in
source-plus-destination memory accesses. It adds no logical checkpoint
reads if each block is read once. The zero-copy-cost dual-worker estimate
improves by only **0.038 s** from adding staging. No arena or staging
was allocated for this offline analysis.

The 1K fresh case fetched 188,512,468,992 logical bytes with 106,536
misses, matching S3-01b. Host read intervals include raw preads,
parallel worker scheduling and any OS page-cache service. No physical
storage throughput was measured. The limited sample, uncontrolled host
file-cache state and zero-overhead sensitivity tables constrain the
result. In the diagnostic capture, Metal stayed at 5,291,360,256 bytes
and 2,183 slots. Sampled pressure was normal and swap decreased 8 MiB.
See [preflight](preflight.txt) and [postflight](postflight.txt).

## Reproduction and validation

The gzip metadata traces ([short](short.json.gz), [1K](1k.json.gz),
[suffix](suffix.json.gz)) contain routed experts, slot decisions, final
state and relative times. They contain no prompts, token IDs, model
paths or weight bytes. See [capture hashes](capture-identity.txt) and
[model-backed test](checkpoint-tests.txt). The suffix prefill and
modeled timing cover 89 fresh tokens. Its initial rows and cumulative
collector counters are retained to reconstruct and verify the seeded
cache.

Run the simulator without a model from the repository root.

```sh
python3 -B -m unittest discover -s docs/evidence/2026-11-18-s301c -p 'test_*.py'
python3 -B docs/evidence/2026-11-18-s301c/replay.py \
  --trace-dir docs/evidence/2026-11-18-s301c \
  --previous docs/evidence/2026-11-18-s301b/timings.json
```

To recollect, apply the [collector patch](collector.patch.gz) only in a
disposable checkout, with one model owner and normal memory pressure.
The compatible checkpoint remains external.

```sh
gzip -dc docs/evidence/2026-11-18-s301c/collector.patch.gz | git apply
mkdir -p /private/tmp/qwen-two-row-output
QWENFIELDFARE_TWO_ROW_MODEL=/absolute/model \
QWENFIELDFARE_TWO_ROW_OUTPUT=/private/tmp/qwen-two-row-output \
  Scripts/test.sh -c release --filter traceTwoRowDecisions
```

Five model-free simulator tests, the focused Swift checks, the checkpoint
capture, and the production package and release build passed. The
[production checks](production-validation.txt) retained the known 131/132
strict result through its archived log. Sources, scripts, package pins
and the strict fixture match `2e5b993`.

**Next independent priority:** measure the CPU cost of default sampled
token selection during 128- and 512-token decode at the existing 8K/5 GiB
profile. Use the exact-greedy fast path as a timed control and report
selector share of inter-token wall time. Keep sampler settings and policy
intact. That measurement is separate work and was not started here.
