# FlintFieldfare Master Engineering Backlog

## Status Codes
- `PENDING`: Not yet started.
- `IN PROGRESS`: Currently being implemented.
- `BLOCKED`: Blocked by a documented dependency or validation failure.
- `COMPLETED`: Implemented, tested, and verified.
- `REJECTED`: Formally rejected after controlled experiment.
- `DEFERRED`: Postponed to a later stage.

---

## Milestone F0 — Repository and Model Audit

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F0-01** | P0 | `COMPLETED` | Environment & Process Audit | Host | Verify M5 32 GB, macOS, Swift, isolate running processes | Shell commands | `docs/flint/DEVELOPMENT_LOG.md` |
| **F0-02** | P0 | `COMPLETED` | Git Isolation & Branch Setup | Git | Create `feature/recursive-flint-40b-a2p5b` from `7370a1d` | `git status` | `git branch -vv` |
| **F0-03** | P0 | `COMPLETED` | Checkpoint Header & Tensor Audit | Checkpoint | Parse Safetensors headers, shapes, dtypes for 1,751 tensors | `audit_checkpoint.py` | `docs/flint/MODEL_SPEC.md` |
| **F0-04** | P0 | `COMPLETED` | Reference Source Architecture Audit | Research | Verify Transformers & MLX implementations of Flint | Source inspection | `docs/flint/RESEARCH_SOURCES.md` |
| **F0-05** | P0 | `COMPLETED` | Memory & Resource Feasibility | Memory | Calculate resident, streamed, cache, KV budgets for 32 GB M5 | Mathematical plan | `docs/flint/MEMORY_PLAN.md` |
| **F0-06** | P0 | `COMPLETED` | Establish Project Documentation | Docs | Populate all required `docs/flint/` files | File checks | `docs/flint/` |

---

## Milestone F1 — Flint Metadata & Weight-Loading Infrastructure

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F1-01** | P0 | `COMPLETED` | Flint Model Configuration & Directory | `Sources/FlintFieldfareCore/` | Strict Case A (INT4) & Case B (141 Mixed overrides) validation, pread safety | `FlintModelDirectoryTests` | Validated in test suite (22 tests) |
| **F1-02** | P0 | `COMPLETED` | Flint Dedicated Inference Lease | `Sources/FlintFieldfareCore/` | Acquire `/tmp/flintfieldfare-inference-\(uid).lock` without Qwen conflict | `FlintInferenceLeaseTests` | Validated in test suite |
| **F1-03** | P0 | `COMPLETED` | Bounded Safetensors Access & Validation | `Sources/FlintFieldfareCore/` | Map all 1,751 tensors across 4 shards, capacity-checked buffers, 64 MB Data limit, zero-trap shapes, chunked streaming | `FlintSafetensorsFileTests` | Validated in test suite (28 tests), `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` |
| **F1-04** | P0 | `COMPLETED` | Resident Weight Extraction & Storage | `Sources/FlintFieldfareCore/` | Load 1,328 resident tensors (1.28 GB) into Metal shared memory, zero temporary staging, zero routed experts, verified against independent MLX references | `FlintResidentLoaderTests` (9 tests) | Implemented & verified (59 tests passing across 4 suites) |
| **F1-05** | P0 | `COMPLETED` | Model-Free Malformed Checkpoint Tests | `Tests/FlintFieldfareCoreTests/` | Reject missing tensors, corrupted shapes, wrong layer types | Unit tests | Validated in test suite |

---

## Milestone F2 — Flint Mathematical Primitives & Metal Kernels

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F2-01** | P0 | `COMPLETED` | Flint RMSNorm Kernel | `Sources/FlintFieldfareCore/Metal/` | Match FP32 RMSNorm on hidden_size (2048) and head_dim (128), zero resident weight copy/mutation, MLX 0.32.4 qualification | `FlintRMSNormKernelTests` (5 tests) | Implemented & verified (64 tests passing across 5 suites) |
| **F2-02** | P0 | `COMPLETED` | Attention Sink & SDPA Kernel | `Sources/FlintFieldfareCore/Metal/` | Correct softmax logit concatenation, full & sliding attention, split-half RoPE, bounded-KV indexing with eviction, ring layout rejection, combined RoPE+Attention qualification, MLX 0.32.4 qualification | `FlintAttentionKernelTests` (10 tests) | Implemented & verified (76 tests passing across 6 suites) |
| **F2-03** | P0 | `COMPLETED` | Sigmoid Top-6 Router Kernel | `Sources/FlintFieldfareCore/Metal/` | 47 MoE layers (2..48), 256 experts, sigmoid activation, deterministic top-6 tie-breaking, normalized weights (sum=1.0), zero heap allocation/CPU roundtrips, MLX 0.32.4 qualification | `FlintRouterKernelTests` (7 tests) | Implemented & verified (83 tests passing across 7 suites) |
| **F2-04** | P0 | `COMPLETED` | 3-Bit & 4-Bit Metal Dequantization | `Sources/FlintFieldfareCore/Metal/` | Native 3-bit & 4-bit affine dequantization (group size 64) with continuous boundary crossing, BF16/FP32 output, zero-copy immutability, bounded memory, MLX 0.32.4 qualification | `FlintDequantizeKernelTests` (7 tests) | Implemented & verified (90 tests passing across 8 suites) |
| **F2-05** | P0 | `COMPLETED` | Fixed Router-Order Expert Reduction | `Sources/FlintFieldfareCore/Metal/` | Position-indexed accumulation strictly in router order (k=0..5) independent of completion order or cache residency, unweighted shared expert addition, FP32 accumulation with BF16/FP32 output, MLX 0.32.4 qualification | `FlintReductionKernelTests` (8 tests) | Implemented & verified (98 tests passing across 9 suites) |
| **F2-06** | P0 | `COMPLETED` | SwiGLU Dense & Shared Expert MLP | `Sources/FlintFieldfareCore/Metal/` | Dense MLP (L0, L1: 2048->6144) and Shared Expert MLP (L2..48: 2048->1024), 4-bit affine dequant integration, FP32 accumulation, fused SiLU*up SwiGLU, 0 heap allocs per dispatch, bounded 48 MiB scratch, MLX 0.32.4 qualification | `FlintMLPKernelTests` (8 tests) | Implemented & verified (107 tests passing across 10 suites) |

---

## Milestone F3 — First Real Flint Forward Token

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F3-01A** | P0 | `COMPLETED` | Layer 0 Dense End-to-End Forward | Runtime | Decompose and qualify all 10 intermediate stages of Layer 0 (Input RMSNorm, QKV 4-bit dequant+GEMM, Q/K per-head norm 128, causal attention with sinks, O-proj, Res1, Post-attn norm, Dense SwiGLU 6144, Res2) for T=1 decode and T=4, T=8 prefill vs MLX 0.32.4 | `FlintDecoderLayerTests` (5 tests) | Implemented & verified (111 tests passing) |
| **F3-01B** | P0 | `COMPLETED` | Layer 2 MoE End-to-End Forward | Runtime | Decompose and qualify all 14 intermediate stages of Layer 2 (Sliding attention 513 with RoPE, Sigmoid top-6 router, synchronous 3-bit weight reads from `model-00001-of-00004.safetensors`, 6 routed experts into contribution slots, 4-bit shared expert forward, fixed router-order reduction, Res1 & Res2) for T=1 decode and T=4 prefill vs MLX 0.32.4 | `FlintDecoderLayerTests` (8 tests) | Implemented & verified (116 tests passing across 11 suites) |
| **F3-02** | P0 | `PENDING` | Full 49-Layer Forward Runner | Runtime | Teacher-forced forward on prompt token IDs matching logits | Teacher forcing test | Pending |
| **F3-03** | P0 | `PENDING` | Greedy Deterministic Generation | Runtime | Generate first coherent text continuation | First token test | Pending |

---

## Milestone F4 — Bounded Expert Streaming

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F4-01** | P0 | `PENDING` | Bounded SLRU Expert Cache for Flint | Storage | Bounded capacity (e.g. 1024 slots), 47 layers, top-6 routing | Cache unit tests | Pending |
| **F4-02** | P0 | `PENDING` | SSD Bounded Expert Store & Reader | Storage | Read 1.31 MiB / 1.69 MiB expert slices by descriptor | Expert reader tests | Pending |
| **F4-03** | P0 | `PENDING` | Streaming Runner Invariance Qualification | Runtime | Bitwise/ULP-identical output between cold cache and warm cache | Residency qualification | Pending |

---

## Milestones F5 to F10 — High-Level Roadmap

- **F5**: Bounded Chunked Prefill & Attention KV Management
- **F6**: Standalone CLI (`FlintFieldfareBench`) and SwiftNIO Server (`FlintFieldfareServer`)
- **F7**: Native Flint Tool Calling & SSE Streaming Translation
- **F8**: OpenCode Coding Agent Qualification & Native macOS UI Integration
- **F9**: Exact-Prefix KV Reuse & Metal Kernel Fusion
- **F10**: Full Release Qualification & Community Benchmarking
