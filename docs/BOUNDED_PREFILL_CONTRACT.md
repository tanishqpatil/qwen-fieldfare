# S3-01a bounded layer-major prefill contract

Start from the deterministic router-order and exact-prefix implementation
at `2e5b993`. The first candidate is diagnostic and opt-in. The tokenwise
schedule remains the default and the independent reference.

Process up to eight fresh tokens at a time. At each layer, a two-dimensional
Metal dispatch batches input RMSNorm and Q4 input projections across the
chunk. Linear layers batch the fused QKV/Z/B/A dispatch. Full-attention layers
batch each Q/K/V dispatch. Every row keeps the original quantization,
lane mapping, accumulation and reduction. This initial design reduces
dispatches. It makes no claim of sharing Q4 loads or reducing expert reads.

For a chunk starting at absolute position S, layer L receives the complete
outputs of layer L−1 for positions S through S+count−1, or embeddings
for layer zero. Within each layer, Conv history and DeltaNet advance once
per position in ascending order. Full-attention K/V updates use the same
absolute positions, and causal reads end at the current position. RoPE,
routing, the per-layer expert-cache access sequence, and final reduction
of eight weighted contributions follow the original per-token schedule.
Only one token's experts are acquired at a time. The layer schedule
requires resident dense weights and all 40 cache partitions.

Keep the original per-token LM-head GPU work. The final prompt row produces
the first generation prediction using the complete request's token set,
sample position zero and the original sampler. Decode continues through
the tokenwise path. A one-row tail uses the tokenwise forward. A completed
chunk appends precisely its fresh prompt IDs to the existing ledger.
Cancellation or command error after a layer update aborts the request.
Drain submitted work before resetting buffers on the next full replay.
Prefix publication continues to require successful inference and the
existing terminal request transaction.

## Scratch and acceptance

Two hidden batches and one normalized input batch use 6,144 floats per
row. QKV, Z, B and A use 12,352. Separate full-attention K and V use 1,024.
Total is 78,080 logical bytes per row, 624,640 bytes at eight rows,
1,249,280 at 16 and 2,498,560 at 32. Evaluate only eight rows initially.
Buffers live with one runner and are reused across layers and requests.
Each experimental buffer is page-aligned in the planner charge. Existing
KV, Conv, DeltaNet, router, MoE and decode scratch are reused in place.
Allocate on opt-in and charge scratch to the existing safety reserve,
with measured Metal headroom and unchanged expert slots required.

Gate the candidate on projection tests, numerical agreement with the
tokenwise runner at state and generation boundaries, the checkpoint-verified
strict fixture, exact-prefix/lifecycle behavior and measured cost. The
known strict 131/132 mismatch remains independent. Production promotion
requires at least 20% TTFT improvement on representative fresh prompts.
Preserve a diagnostic checkpoint if the candidate is useful but falls
short. See [numerical policy](NUMERICAL_NON_REGRESSION.md).
