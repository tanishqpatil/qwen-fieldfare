#!/usr/bin/env python3
"""
Generates pinned reference golden test vectors for Flint MoE Sigmoid Top-6 Router
using MLX 0.32.4 on Apple Silicon.

Output: flint_router_golden.json
"""

import json
import os
import numpy as np
import mlx.core as mx

def to_hex(arr: mx.array) -> str:
    return bytes(arr.view(mx.uint8)).hex()

def to_hex_np(arr: np.ndarray) -> str:
    return arr.tobytes().hex()

def compute_router(x_np: np.ndarray, w_np: np.ndarray):
    # x_np: [batch, T, 2048] float32
    # w_np: [256, 2048] float32
    # logits = x @ W^T
    logits = np.matmul(x_np, w_np.T)
    # sigmoid = 1 / (1 + exp(-logits))
    scores = 1.0 / (1.0 + np.exp(-logits))
    batch, T, _ = scores.shape
    top_experts = np.zeros((batch, T, 6), dtype=np.uint32)
    top_weights = np.zeros((batch, T, 6), dtype=np.float32)

    for b in range(batch):
        for t in range(T):
            row = scores[b, t].copy()
            for k in range(6):
                best_idx = int(np.argmax(row)) # ties broken by lowest index
                top_experts[b, t, k] = best_idx
                top_weights[b, t, k] = row[best_idx]
                row[best_idx] = -1e30
            # Normalize
            s = np.sum(top_weights[b, t]) + 1e-20
            top_weights[b, t] /= s

    return top_experts, top_weights

def main():
    mx.random.seed(42)
    np.random.seed(42)

    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    shard2 = mx.load(os.path.join(model_dir, "model-00002-of-00004.safetensors"))
    shard4 = mx.load(os.path.join(model_dir, "model-00004-of-00004.safetensors"))

    w_l2_bf16 = shard1["model.layers.2.mlp.gate.weight"]
    w_l24_bf16 = shard2["model.layers.24.mlp.gate.weight"]
    w_l48_bf16 = shard4["model.layers.48.mlp.gate.weight"]

    layers = [
        (2, w_l2_bf16),
        (24, w_l24_bf16),
        (48, w_l48_bf16),
    ]

    cases = []

    for layer_idx, w_bf16 in layers:
        w_f32 = np.array(w_bf16.astype(mx.float32))
        w_bf16_bytes = bytes(w_bf16.view(mx.uint8))

        # 1. Random FP32 (T=1)
        x_f32 = np.random.randn(1, 1, 2048).astype(np.float32)
        exp, wgt = compute_router(x_f32, w_f32)
        cases.append({
            "name": f"layer_{layer_idx}_random_f32",
            "layer_idx": layer_idx,
            "dtype": "float32",
            "batch": 1,
            "num_tokens": 1,
            "input_hex": to_hex_np(x_f32),
            "gate_weight_hex": w_bf16_bytes.hex(),
            "expected_experts_hex": to_hex_np(exp),
            "expected_weights_hex": to_hex_np(wgt)
        })

        # 2. Random BF16 (T=1)
        x_bf16_arr = mx.array(x_f32).astype(mx.bfloat16)
        x_bf16_as_f32 = np.array(x_bf16_arr.astype(mx.float32))
        exp_bf16, wgt_bf16 = compute_router(x_bf16_as_f32, w_f32)
        cases.append({
            "name": f"layer_{layer_idx}_random_bf16",
            "layer_idx": layer_idx,
            "dtype": "bfloat16",
            "batch": 1,
            "num_tokens": 1,
            "input_hex": bytes(x_bf16_arr.view(mx.uint8)).hex(),
            "gate_weight_hex": w_bf16_bytes.hex(),
            "expected_experts_hex": to_hex_np(exp_bf16),
            "expected_weights_hex": to_hex_np(wgt_bf16)
        })

    # Edge Case 1: All Zeros input -> all logits 0, all sigmoid 0.5, test tie-breaking (experts 0..5, weights 1/6)
    w_l2_f32 = np.array(w_l2_bf16.astype(mx.float32))
    w_l2_bf16_bytes = bytes(w_l2_bf16.view(mx.uint8))
    x_zeros = np.zeros((1, 1, 2048), dtype=np.float32)
    exp_zeros, wgt_zeros = compute_router(x_zeros, w_l2_f32)
    cases.append({
        "name": "layer_2_zeros",
        "layer_idx": 2,
        "dtype": "float32",
        "batch": 1,
        "num_tokens": 1,
        "input_hex": to_hex_np(x_zeros),
        "gate_weight_hex": w_l2_bf16_bytes.hex(),
        "expected_experts_hex": to_hex_np(exp_zeros),
        "expected_weights_hex": to_hex_np(wgt_zeros)
    })

    # Edge Case 2: Saturated large positive input (+50.0) -> sigmoid ~ 1.0
    x_pos = np.ones((1, 1, 2048), dtype=np.float32) * 5.0
    exp_pos, wgt_pos = compute_router(x_pos, w_l2_f32)
    cases.append({
        "name": "layer_2_large_positive",
        "layer_idx": 2,
        "dtype": "float32",
        "batch": 1,
        "num_tokens": 1,
        "input_hex": to_hex_np(x_pos),
        "gate_weight_hex": w_l2_bf16_bytes.hex(),
        "expected_experts_hex": to_hex_np(exp_pos),
        "expected_weights_hex": to_hex_np(wgt_pos)
    })

    # Edge Case 3: Saturated large negative input (-50.0) -> sigmoid ~ 0.0
    x_neg = np.ones((1, 1, 2048), dtype=np.float32) * (-5.0)
    exp_neg, wgt_neg = compute_router(x_neg, w_l2_f32)
    cases.append({
        "name": "layer_2_large_negative",
        "layer_idx": 2,
        "dtype": "float32",
        "batch": 1,
        "num_tokens": 1,
        "input_hex": to_hex_np(x_neg),
        "gate_weight_hex": w_l2_bf16_bytes.hex(),
        "expected_experts_hex": to_hex_np(exp_neg),
        "expected_weights_hex": to_hex_np(wgt_neg)
    })

    # Edge Case 4: Multi-token sequence (T = 8)
    x_seq = np.random.randn(1, 8, 2048).astype(np.float32)
    exp_seq, wgt_seq = compute_router(x_seq, w_l2_f32)
    cases.append({
        "name": "layer_2_sequence_t8_f32",
        "layer_idx": 2,
        "dtype": "float32",
        "batch": 1,
        "num_tokens": 8,
        "input_hex": to_hex_np(x_seq),
        "gate_weight_hex": w_l2_bf16_bytes.hex(),
        "expected_experts_hex": to_hex_np(exp_seq),
        "expected_weights_hex": to_hex_np(wgt_seq)
    })

    payload = {
        "metadata": {
            "source": "MLX 0.32.4 / NumPy",
            "model": "Recursive Flint 40B-A2.5B",
            "hidden_dim": 2048,
            "num_experts": 256,
            "top_k": 6,
            "moe_layers": 47,
            "moe_layer_range": "2..48",
            "dense_layers": "0..1"
        },
        "cases": cases
    }

    out_path = os.path.join(os.path.dirname(__file__), "flint_router_golden.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(cases)} router golden cases in {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
