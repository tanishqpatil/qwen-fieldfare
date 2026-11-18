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
| **F1-03** | P0 | `IN PROGRESS` | Bounded Safetensors Parsing for Flint | `Sources/FlintFieldfareCore/` | Map all 1,751 tensors across 4 shards with descriptor ownership | `FlintSafetensorsTests` | Pending |
| **F1-04** | P0 | `PENDING` | Resident Weight Extraction & Storage | `Sources/FlintFieldfareCore/` | Load embeddings, lm_head, attention projections, dense MLPs, routers, shared experts | `FlintResidentWeightsTests`| Pending |
| **F1-05** | P0 | `COMPLETED` | Model-Free Malformed Checkpoint Tests | `Tests/FlintFieldfareCoreTests/` | Reject missing tensors, corrupted shapes, wrong layer types | Unit tests | Validated in test suite |

---

## Milestone F2 — Flint Mathematical Primitives & Metal Kernels

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F2-01** | P0 | `PENDING` | Flint RMSNorm Kernel | `Sources/FlintFieldfareCore/Metal/` | Match FP32 RMSNorm on hidden_size and head_dim | Numerical unit test | Pending |
| **F2-02** | P0 | `PENDING` | Attention Sink & SDPA Kernel | `Sources/FlintFieldfareCore/Metal/` | Correct softmax logit concatenation, full & sliding attention | Attention numerical tests | Pending |
| **F2-03** | P0 | `PENDING` | Sigmoid Top-6 Router Kernel | `Sources/FlintFieldfareCore/Metal/` | Match FP32 sigmoid and normalized top-6 weights | Router numerical tests | Pending |
| **F2-04** | P0 | `PENDING` | 3-Bit & 4-Bit Metal Dequantization | `Sources/FlintFieldfareCore/Metal/` | Dequantize affine weights matching MLX | Dequant tests | Pending |
| **F2-05** | P0 | `PENDING` | Fixed Router-Order Expert Reduction | `Sources/FlintFieldfareCore/Metal/` | Position-indexed accumulation independent of cache residency | Residency invariance test | Pending |
| **F2-06** | P0 | `PENDING` | SwiGLU Dense & Shared Expert MLP | `Sources/FlintFieldfareCore/Metal/` | Match reference dense and shared expert forward | MLP numerical tests | Pending |

---

## Milestone F3 — First Real Flint Forward Token

| Task ID | Priority | Status | Title | Affects | Acceptance Criteria | Tests | Commit / Evidence |
|---|---|---|---|---|---|---|---|
| **F3-01** | P0 | `PENDING` | Single Layer End-to-End Forward | Runtime | Numerically qualify Layer 0 (dense) and Layer 2 (MoE) vs MLX | Layer numerical tests | Pending |
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
