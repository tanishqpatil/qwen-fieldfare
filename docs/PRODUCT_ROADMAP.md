# Product roadmap

Status: analysis-only roadmap created on 2026-08-28. No item in this document is
implemented or promised until its acceptance criteria pass.

## Product position

QwenFieldfare is currently a strong model-specific research runtime with a
usable native chat interface. It is not yet a production-distributed Mac app.
The next product work should close predictable failure modes before adding
large convenience or platform features.

The roadmap preserves these constraints:

- Qwen 3.6 35B-A3B MLX Q4 only;
- local inference with one model-owning process;
- bounded Active Expert Streaming rather than silent full-model residency;
- no model weights in source control or release artifacts;
- measured, hardware-labelled performance and memory claims;
- no generic server, multi-model abstraction, or cloud dependency without an
  explicit scope decision.

## Current position

| Area | Present capability | Remaining production gap |
| --- | --- | --- |
| Runtime | Three modes; active streaming at 4K–64K; cancellation and unload | Allocation/encoder failures can still terminate instead of returning recoverable errors |
| Model setup | Dynamic directory choice and basic shard/index validation | Architecture, tensor contract, dtype/shape, quantization, and offset bounds are not fully preflighted |
| Chat | Multi-turn, thinking control, sampling, Markdown/code rendering | No visible context budget, deterministic history policy, or persisted conversations |
| Diagnostics | Live timing, cache, memory, and finish-reason telemetry | No sanitized support-bundle export or durable benchmark record |
| Validation | Model-free CI, benchmark harness, historical 132/132 parity record, and an MLX reference generator | The named automated parity test is not present; real-model release qualification must be restored |
| Distribution | A source checkout can build a release-mode SwiftPM executable | No published signed/notarized `.app`, ZIP/DMG, checksums, or clean-Mac install validation |
| Hardware support | Measured on Apple M5 with 32 GiB | Other RAM/chip combinations are unvalidated and should not be advertised as supported |

Effort labels are relative engineering sizes, not delivery promises:

- **S**: bounded change with little architectural work;
- **M**: multiple components and a dedicated validation pass;
- **L**: architectural or release work requiring soak/hardware/external setup.

## Production priority order

### P1 — strict Qwen model-contract preflight

**Impact: critical. Effort: M. Confidence: high.**

Before allocating large Metal buffers, validate:

- compatible architecture constants; exact checkpoint identity only if a
  separate trusted manifest and payload-hash policy is designed;
- tokenizer and chat-template metadata;
- every runtime-consumed tensor name, shape, dtype, quantization group/bits,
  and shard while permitting unrelated extra tensors such as vision weights;
- a capped Safetensors header length, overflow-safe offset arithmetic, and
  data bounds against actual file sizes;
- a defined symlink policy, canonical shard containment inside the selected
  model directory, and regular-file checks;
- unsupported or conflicting required variants with actionable errors.

Acceptance:

1. the supported checkpoint passes using header-bounded reads;
2. mutated fixtures fail before Metal allocation and name the violated contract;
3. oversized headers, arithmetic overflow, escaping links/paths, non-regular
   shards, malformed offsets, missing tensors, and wrong quantization are
   covered by model-free tests;
4. no complete shard or weight tensor is copied into heap memory for validation.

### P2 — recoverable model lifecycle

**Impact: critical. Effort: L. Confidence: medium-high.**

Replace production-path `fatalError`, force-unwrapped Metal allocations, and
non-cancellable load phases with typed errors and explicit ownership stages.
Expose load progress, cancellation checkpoints, idempotent unload, and clear
recovery after a failed allocation or encoder creation. Move lifecycle/state
orchestration into a testable support target and use injectable runtime,
resource, allocation, and encoder factories.

Acceptance:

1. injected allocation, command-buffer, tokenizer, and file-read failures
   surface as recoverable UI errors;
2. cancellation keeps the UI responsive, is observed after the current bounded
   stage, prevents generation from starting, and eventually releases ownership;
3. repeated load/generate/cancel/unload soak tests deallocate the runner and
   return Metal allocation and physical footprint to a defined baseline
   tolerance within a bounded observation window;
4. a failed load can be retried without restarting the app.

### P3 — context and conversation correctness

**Impact: high. Effort: M. Confidence: high.**

After P1 has validated and loaded the selected tokenizer, show exact
prompt/history/output token use before generation. Define one explicit policy
when the selected context cannot hold the request: either block with an
actionable choice or prune complete oldest turns under a documented,
deterministic policy. Never silently truncate a partial turn.

Acceptance:

1. the composer displays remaining context and the effective maximum output;
2. New Chat, cancellation, errors, and context-full exits cannot
   desynchronize visible messages from the internal prompt;
3. boundary tests cover every context tier and multi-turn template overhead;
4. any pruning is visible, deterministic, and operates on complete turns.

### P4 — memory-pressure safety and hardware qualification

**Impact: high. Effort: L plus hardware time. Confidence: medium.**

Turn the current point-in-time preflight into runtime protection. Validate each
advertised mode/context on a small Apple Silicon/RAM matrix, and mark all other
combinations unvalidated instead of guessing support.

Acceptance:

1. warning or critical pressure during load/generation produces a controlled
   warning, cancellation, or unload policy;
2. every supported Active profile remains within its stated Metal budget;
   controlled-protocol runs report before/after system-wide swap and reject a
   profile when repeatable positive deltas occur;
3. the support table records chip, RAM, macOS, context, cache profile, peak
   allocation, and validation date;
4. unsupported combinations are disabled or clearly labelled experimental.

### P5 — first-run readiness flow

**Impact: high. Effort: M. Confidence: high after P1.**

Guide users through OS/chip/RAM/disk checks, the compatible checkpoint
contract, folder selection, and model validation. A copyable `hf download`
command is sufficient for the first version; automatic downloading is not
required.

Acceptance:

1. Send remains disabled until static prerequisites pass, and a lightweight
   dynamic memory/model-availability preflight runs immediately before each
   generation;
2. validation progress and errors are visible and actionable;
3. the flow never implies that source code includes the model;
4. relaunch remembers only a validated model directory.

### P6 — sanitized diagnostic export

**Impact: medium-high. Effort: S–M. Confidence: high.**

Export a support bundle from an allowlist of structured fields containing
embedded build-time version/commit/compiler metadata, hardware, OS, runtime
mode/context/profile, lifecycle state, timing, cache, memory, and categorized
errors. Do not depend on a developer Swift toolchain being installed.

Acceptance:

1. free-form prompts, output, raw errors, credentials, absolute paths,
   usernames, and weights are never admitted to the export schema;
2. allowlist tests prove that only approved structured fields serialize;
3. benchmark and profiled results are clearly distinguished;
4. users can preview the export before saving or attaching it.

### P7 — real-model release qualification

**Impact: critical for releases. Initial effort: L. Confidence: high after the
harness exists.**

Keep public CI model-free. Add a separate local or self-hosted release gate
that records commit/hardware/toolchain and runs the complete private-model
qualification without uploading weights or prompts.

Acceptance:

1. add or restore a release-mode automated teacher-forced parity harness and
   require all 132 recorded next-token positions to match MLX;
2. add a fixed-seed test using the app's temperature, Top-K, Top-P, and penalty
   defaults, plus multi-turn behavior, a 512-token stability run, and a 64K
   allocation load pass;
3. memory, swap, cancellation, unload, and model-contract failures pass;
4. output is sanitized, and enforcement is either a required self-hosted status
   check or a signed qualification artifact consumed by the release workflow.

### P8 — native signed distribution

**Impact: high. Effort: L. Confidence: medium-high after credentials exist.**

Package a versioned arm64 `.app`, sign it with Developer ID, notarize it, and
publish a notarized ZIP or DMG with checksums and release notes. Decide between
an unsandboxed hardened-runtime app and App Sandbox with security-scoped
bookmarks for the external model directory before packaging.

Acceptance:

1. bundle identifier, semantic version, icon, resources, entitlements, NOTICE,
   third-party licenses, hardened-runtime settings, and sandbox policy are
   correct;
2. Gatekeeper accepts the artifact on a clean supported Mac, and packaged
   Metal libraries/resources load successfully from the `.app`;
3. tag, app version, checksum, and release notes agree;
4. the release contains no model weights, local paths, benchmark traces, or
   credentials.

### P9 — accessibility and UI lifecycle qualification

**Impact: medium-high. Effort: M. Confidence: high.**

Add explicit VoiceOver labels/values, full keyboard navigation, predictable
focus after send/error/dialog actions, Reduce Motion behavior, and contrast
review. Move testable app behavior into support components where needed.

Acceptance:

1. model selection, send, stop, New Chat, error recovery, and unload are usable
   without a pointer;
2. dynamic status and performance values expose meaningful accessibility
   values rather than decorative labels;
3. app-level lifecycle tests cover the primary success and failure paths.

## Later convenience features

These are useful after P1–P6 establish a reliable and supportable base.

| Feature | Effort | Dependency | Acceptance boundary |
| --- | ---: | --- | --- |
| Persist generation presets | S | Stable settings schema | Versioned migration and Reset Defaults |
| Retry/regenerate/edit last turn | M | P3 context/state contract | One deterministic conversation; no hidden duplicate turns |
| Local chat history, search, and Markdown export | M–L | Versioned local privacy schema | Opt-in local-only storage, atomic writes, migration, delete-all |
| Interactive Qwen CLI | S–M | Stable generation and validation APIs | Streaming output, cancellation, deterministic exit codes, same memory gates |
| Performance history and comparison export | S–M | P6 diagnostics schema | Commit/workload-labelled data; profiled and headline runs never mixed |
| In-app resumable model download | L | P1/P5, storage and license design | Resume/cancel across restart, bounded scratch, final contract validation |
| Signed update checker | M | P8 signed releases | Verified updates, manual control, never interrupts generation |
| Plain-text file attachment | M–L | P3 context budgeting | Local parsing, explicit token limits, visible truncation policy |

## Explicitly deferred

Do not prioritize these in the current product scope:

- a generic OpenAI-compatible server or a second model-owning process;
- vision, audio, video, tool execution, or generic multi-model support;
- the packed-expert store as the app default;
- cloud accounts, analytics, or prompt telemetry;
- automatic model download before readiness, integrity, cancellation, disk,
  licensing, and recovery behavior are designed;
- speculative decoding without a validated compatible draft/MTP artifact.

## Delivery phases

1. **Reliability foundation:** P1–P4.
2. **Readiness and support:** P5–P6.
3. **Release gate:** P7.
4. **Distribution beta:** P8.
5. **1.0 quality gate:** P9 plus successful hardware-matrix evidence.

Performance work remains governed by
[Optimization analysis and priority order](OPTIMIZATION_PLAN.md). Product and
performance work share the same correctness, memory, and evidence gates.
