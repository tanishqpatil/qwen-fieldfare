# FlintFieldfare System Architecture

## 1. System Vision & Purpose

**FlintFieldfare** is a native Swift and Metal inference runtime designed specifically for **Recursive Flint 40B-A2.5B** on Apple Silicon Macs.

Flint is a 39.8B-parameter decoder-only mixture-of-experts Transformer with 2.5B active parameters per token. Loading the complete quantized model into unified memory requires 18–24 GB, which severely constrains or overwhelms 24 GB and 32 GB Apple Silicon Macs.

FlintFieldfare applies the **bounded expert streaming** architecture:
- Core model weights (embeddings, attention projections, layer norms, routers, dense MLPs, and shared experts) remain permanently resident in Apple Silicon unified memory (~1.28 GB).
- The 12,032 routed experts (256 experts $\times$ 47 MoE layers) reside on high-speed NVMe SSD storage.
- Active experts (6 per layer per token) are streamed into a bounded Metal buffer cache (e.g. 1,024 slots $\approx$ 1.31 GB).
- The complete running working set targets an architectural budget of **~3.2 GiB**, leaving >20 GB of unified memory completely free on a 32 GB machine.

## 2. Implemented vs Proposed Behavior

| Component | Status | Implementation Details |
|---|---|---|
| **Safetensors Inspection** | Implemented | Bounded header parsing without loading full tensor payloads |
| **Flint Checkpoint Audit** | Implemented | Verified 1,751 tensors, U32 packed format, BF16 parameters |
| **Model Specification** | Implemented | 49 layers, 14 full/35 sliding attention, sink offsets, top-6 routing |
| **Memory Plan** | Implemented | Empirical 19.5 GB baseline on 64 GB, ~1.28 GB resident, ~1.31 GB expert cache, ~3.2 GB working set budget |
| **FlintModelDirectory** | Implemented (F1-01) | Swift type validating `config.json`, strict Case A/B quantization, shards |
| **FlintInferenceLease** | Implemented (F1-02) | Single-process POSIX lock at `/tmp/flintfieldfare-inference-\(uid).lock` |
| **FlintSafetensorsFile** | Implemented (F1-03) | Bounded reader, POSIX pread, 8-byte LE validation, zero-trap shape validation, capacity-checked buffers, 64 MB Data cap, chunked streaming |
| **Physical Tensor Inventory** | Implemented (F1-03) | 4 shards, 1,751 tensors, 17.84 GB, 1.3125 MiB per-expert record, closed-form offset formula |
| **FlintResidentLoader** | Proposed (F1-04) | Safe binary loader for permanently resident weights (~1.28 GiB) |
| **Flint Metal Attention** | Proposed (F2) | Q/K RMSNorm, learned sink `softmax_offset`, sliding window 513 |
| **Flint Metal MoE** | Proposed (F2) | FP32 sigmoid top-6 router, fixed router-order position reduction |
| **Flint Forward Runner** | Proposed (F3) | 49-layer forward evaluation and greedy generation |
| **Bounded Expert Cache** | Proposed (F4) | Segmented LRU cache buffer with SSD streaming |
| **Flint HTTP Server** | Proposed (F6) | Loopback SwiftNIO OpenAI-compatible server with Jinja formatting |

## 3. Structural Decomposition

FlintFieldfare adopts **Option A** (Isolated Model Line) to ensure absolute stability of the existing QwenFieldfare codebase:

```
Package.swift
├── Sources/
│   ├── QwenFieldfareCore/              # Untouched Qwen 3.6 reference
│   ├── QwenFieldfareServer/            # Untouched Qwen server
│   │
│   ├── FlintFieldfareCore/             # New Flint-specific engine
│   │   ├── Infrastructure/
│   │   │   ├── Metal/
│   │   │   │   └── FlintMetalContext.swift   # Reusable Metal device/queue/library wrapper
│   │   │   └── ModelIO/
│   │   │       ├── FlintModelDirectory.swift # Config & directory validator
│   │   │       ├── FlintInferenceLease.swift # Isolated lease lock
│   │   │       ├── FlintSafetensorsFile.swift # Safe, bounded Safetensors reader
│   │   │       └── FlintResidentLoader.swift # Resident weight owner (F1-04)
│   │   ├── Kernels/
│   │   │   ├── Attention/
│   │   │   │   └── FlintAttentionKernel.swift
│   │   │   └── MoE/
│   │   │       └── FlintMoEKernel.swift
│   │   ├── Metal/
│   │   │   ├── Attention/
│   │   │   │   └── flint_attention.metal
│   │   │   └── MoE/
│   │   │       └── flint_moe.metal
│   │   ├── Runtime/
│   │   │   ├── Generation/
│   │   │   │   ├── FlintGenerationSupport.swift
│   │   │   │   └── FlintStreamingRunner.swift
│   │   │   └── Inference/
│   │   │       └── FlintForwardRunner.swift
│   │   └── Storage/
│   │       ├── FlintBoundedExpertCache.swift
│   │       └── FlintExpertStore.swift
│   │
│   ├── FlintFieldfareBench/            # Standalone benchmarking CLI
│   └── FlintFieldfareServer/           # Standalone SwiftNIO server
│
└── Tests/
    ├── QwenFieldfareCoreTests/         # Untouched Qwen tests
    └── FlintFieldfareCoreTests/        # New Flint test suite
```

## 4. Tensor Flow & Mathematical Pipeline

For each token position $t$:

```
Token ID
   │
   ▼
[model.embed_tokens] (INT8 dequantized to BF16)
   │
   ▼
Layer 0..48 Loop:
   ├── Residual Save (x_res = x)
   ├── input_layernorm(x) -> RMSNorm(2048)
   ├── Self-Attention:
   │     ├── q = q_norm(q_proj(x)) -> RMSNorm(128) per head
   │     ├── k = k_norm(k_proj(x)) -> RMSNorm(128) per head
   │     ├── v = v_proj(x)
   │     ├── RoPE: If layer in sliding_attention -> apply RoPE; If full_attention -> no RoPE
   │     ├── KV Cache Update (Rotating for sliding_attention, linear for full_attention)
   │     ├── Attention Softmax:
   │     │     logits = (q · k^T) / sqrt(128)
   │     │     denom = sum(exp(logits)) + exp(softmax_offset[head])
   │     │     weights = exp(logits) / denom
   │     │     out = weights · v
   │     └── o_proj(out)
   ├── Residual Add: x = x_res + out
   │
   ├── Residual Save (x_res = x)
   ├── post_attention_layernorm(x) -> RMSNorm(2048)
   ├── MLP Block:
   │     ├── If Layer 0 or 1 (Dense MLP):
   │     │     dense_out = down_proj(silu(gate_proj(x)) * up_proj(x))
   │     │     mlp_out = dense_out
   │     └── If Layer 2..48 (Sparse MoE):
   │           ├── router_scores = sigmoid(gate(x)) (FP32)
   │           ├── top6_indices, top6_scores = topk(router_scores, 6)
   │           ├── normalized_weights = top6_scores / sum(top6_scores)
   │           ├── For each k in 0..5:
   │           │     expert_k = cache.get_or_stream(layer, top6_indices[k])
   │           │     contrib[k] = normalized_weights[k] * expert_k(x)
   │           ├── routed_out = sum_{k=0..5} contrib[k] (fixed router order!)
   │           ├── shared_out = shared_expert(x) (ungated SwiGLU)
   │           └── mlp_out = routed_out + shared_out
   └── Residual Add: x = x_res + mlp_out
   │
   ▼
[model.norm] -> Final RMSNorm(2048)
   │
   ▼
[lm_head] -> FP32 Linear Projection -> Logits [150144]
```

## 5. Active Expert Streaming & Reduction Order

To guarantee mathematical determinism across arbitrary cache hit/miss patterns:
1. The router selects $K=6$ experts in priority order $\pi(0), \ldots, \pi(5)$.
2. For each expert, regardless of whether it is already resident in cache or read asynchronously from SSD:
   It computes its down-projection output into an allocated buffer at row offset $k \times 2048$:
   $$\text{scratch}[k \cdot 2048 + r] = w_k \cdot \text{down\_proj}^{(k)}(r)$$
3. The combine kernel evaluates:
   $$\text{routed\_sum}[r] = \sum_{k=0}^5 \text{scratch}[k \cdot 2048 + r]$$
   strictly in order $k = 0, 1, 2, 3, 4, 5$.
4. The result is mathematically identical across all-hit, partial-hit, and all-miss cache states.

## 6. Single Process Lease & Resource Guardrails

FlintFieldfare enforces single model-process ownership using a non-blocking POSIX advisory lock:
`let fd = open("/tmp/flintfieldfare-inference-\(getuid()).lock", O_CREAT|O_RDWR|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC, 0o600)`
`guard flock(fd, LOCK_EX|LOCK_NB) == 0 else { throw FlintInferenceLeaseError.inUse }`

This guarantees that:
- Two competing Flint instances cannot run simultaneously and exhaust unified memory.
- Running Flint tests does not interfere with or get blocked by independent QwenFieldfare jobs.
