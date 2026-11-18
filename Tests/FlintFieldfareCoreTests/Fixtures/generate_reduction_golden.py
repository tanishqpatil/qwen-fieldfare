#!/usr/bin/env python3
"""
Generate golden fixtures for Flint Fixed Router-Order Expert Reduction (F2-05).
Reference generated using MLX 0.32.4.

Mathematical contract:
  routed_sum = sum_{k=0..5} (weights[b, t, k] * contributions[b, t, k, :])
  if shared_expert is present:
    output = routed_sum + shared_expert[b, t, :]
  else:
    output = routed_sum

Accumulation is strictly FP32 with bfloat16 conversion at the output boundary.
"""

import json
import struct
import numpy as np
import mlx.core as mx

DIM = 2048
EXPERTS = 6

def to_hex(arr_bytes: bytes) -> str:
    return arr_bytes.hex()

def generate_fixtures():
    mx.random.seed(42)
    np.random.seed(42)
    
    cases = []
    
    # 1. Single token, no shared expert, BF16
    B, T = 1, 1
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.bfloat16)
    w = mx.random.uniform(low=0.05, high=0.35, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_bf16 = routed.astype(mx.bfloat16)
    
    cases.append({
        "name": "single_token_no_shared_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": False,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": None,
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 2. Single token, with shared expert, BF16
    B, T = 1, 1
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.bfloat16)
    w = mx.random.uniform(low=0.05, high=0.35, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    s = mx.random.normal(shape=(B, T, DIM), dtype=mx.bfloat16)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_f32 = routed + s.astype(mx.float32)
    out_bf16 = out_f32.astype(mx.bfloat16)
    
    cases.append({
        "name": "single_token_with_shared_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": True,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": bytes(s.view(mx.uint16)).hex(),
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 3. Multi-token (T=4), with shared expert, BF16
    B, T = 1, 4
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.bfloat16)
    w = mx.random.uniform(low=0.01, high=0.5, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    s = mx.random.normal(shape=(B, T, DIM), dtype=mx.bfloat16)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_f32 = routed + s.astype(mx.float32)
    out_bf16 = out_f32.astype(mx.bfloat16)
    
    cases.append({
        "name": "multi_token_with_shared_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": True,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": bytes(s.view(mx.uint16)).hex(),
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 4. Multi-batch (B=2, T=2), with shared expert, BF16
    B, T = 2, 2
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.bfloat16)
    w = mx.random.uniform(low=0.05, high=0.35, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    s = mx.random.normal(shape=(B, T, DIM), dtype=mx.bfloat16)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_f32 = routed + s.astype(mx.float32)
    out_bf16 = out_f32.astype(mx.bfloat16)
    
    cases.append({
        "name": "multi_batch_multi_token_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": True,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": bytes(s.view(mx.uint16)).hex(),
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 5. Zero contributions (only shared expert), BF16
    B, T = 1, 1
    c = mx.zeros(shape=(B, T, EXPERTS, DIM), dtype=mx.bfloat16)
    w = mx.ones(shape=(B, T, EXPERTS), dtype=mx.float32) / EXPERTS
    s = mx.random.normal(shape=(B, T, DIM), dtype=mx.bfloat16)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_f32 = routed + s.astype(mx.float32)
    out_bf16 = out_f32.astype(mx.bfloat16)
    
    cases.append({
        "name": "zero_contributions_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": True,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": bytes(s.view(mx.uint16)).hex(),
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 6. Cancellation test: expert 0 and expert 1 have equal weights and opposite signs
    B, T = 1, 1
    c_np = np.zeros((B, T, EXPERTS, DIM), dtype=np.float32)
    base_vec = np.random.normal(scale=2.0, size=(DIM,)).astype(np.float32)
    c_np[0, 0, 0, :] = base_vec
    c_np[0, 0, 1, :] = -base_vec
    c = mx.array(c_np, dtype=mx.bfloat16)
    w_np = np.zeros((B, T, EXPERTS), dtype=np.float32)
    w_np[0, 0, 0] = 0.5
    w_np[0, 0, 1] = 0.5
    w = mx.array(w_np)
    c_f32 = c.astype(mx.float32)
    routed = mx.sum(c_f32 * w[:, :, :, None], axis=2)
    out_bf16 = routed.astype(mx.bfloat16)
    
    cases.append({
        "name": "cancellation_bf16",
        "batch": B,
        "tokens": T,
        "mode": "bf16",
        "has_shared": False,
        "contributions_hex": bytes(c.view(mx.uint16)).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": None,
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    # 7. Single token, FP32 in, FP32 out
    B, T = 1, 1
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.float32)
    w = mx.random.uniform(low=0.05, high=0.35, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    s = mx.random.normal(shape=(B, T, DIM), dtype=mx.float32)
    routed = mx.sum(c * w[:, :, :, None], axis=2)
    out_f32 = routed + s
    
    cases.append({
        "name": "single_token_with_shared_f32",
        "batch": B,
        "tokens": T,
        "mode": "f32",
        "has_shared": True,
        "contributions_hex": bytes(c).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": bytes(s).hex(),
        "expected_hex": bytes(out_f32).hex(),
    })
    
    # 8. Single token, FP32 in, BF16 out
    B, T = 1, 1
    c = mx.random.normal(shape=(B, T, EXPERTS, DIM), dtype=mx.float32)
    w = mx.random.uniform(low=0.05, high=0.35, shape=(B, T, EXPERTS), dtype=mx.float32)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    routed = mx.sum(c * w[:, :, :, None], axis=2)
    out_bf16 = routed.astype(mx.bfloat16)
    
    cases.append({
        "name": "single_token_f32_in_bf16_out",
        "batch": B,
        "tokens": T,
        "mode": "f32_in_bf16_out",
        "has_shared": False,
        "contributions_hex": bytes(c).hex(),
        "weights_hex": bytes(w).hex(),
        "shared_hex": None,
        "expected_hex": bytes(out_bf16.view(mx.uint16)).hex(),
    })
    
    output = {
        "metadata": {
            "source": "MLX 0.32.4",
            "model": "Recursive Flint 40B-A2.5B",
            "dimension": DIM,
            "experts": EXPERTS,
            "semantics": "sum_{k=0..5} w_k * c_k (+ shared_expert)"
        },
        "cases": cases
    }
    
    out_path = "Tests/FlintFieldfareCoreTests/Fixtures/flint_reduction_golden.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Generated {len(cases)} test cases in {out_path}")

if __name__ == "__main__":
    generate_fixtures()
