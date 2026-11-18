#!/usr/bin/env python3
"""
Generate 100 deterministic tokens routed through actual pinned Flint MLX SparseMoeBlock
for live Layer 2 gate weights.

Saves:
- Input activations: 100 x 2048 BF16
- Gate weights: 256 x 2048 BF16
- Actual MLX SparseMoeBlock top-6 indices and weights
- Top-7 MLX scores and margins
- Independent FP32 reference indices and weights
"""

import os
import json
import struct
import mlx.core as mx
import numpy as np
from mlx_vlm.models.flint.language import SparseMoeBlock, ModelConfig

def main():
    mx.random.seed(42)
    np.random.seed(42)

    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    args = ModelConfig.from_dict(text_cfg)
    block = SparseMoeBlock(args)

    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    w_l2 = shard1["model.layers.2.mlp.gate.weight"]
    block.gate.weight = w_l2

    num_tokens = 100
    x = mx.random.normal(shape=(num_tokens, 2048)).astype(mx.bfloat16)

    # 1. Actual MLX route output
    mlx_raw_idx, mlx_raw_w = block.route(x)
    # Full sigmoid scores from MLX gate projection
    mlx_scores = mx.sigmoid(block.gate(x).astype(mx.float32))

    # 2. Independent FP32 NumPy reference
    w_f32 = np.array(w_l2.astype(mx.float32))
    x_f32 = np.array(x.astype(mx.float32))

    records = []
    for t in range(num_tokens):
        # MLX raw partition
        t_raw_idx = [int(i) for i in mlx_raw_idx[t]]
        t_raw_w = [float(w) for w in mlx_raw_w[t]]

        # MLX sorted by weight descending
        order = np.argsort(-np.array(t_raw_w))
        t_sorted_idx = [t_raw_idx[i] for i in order]
        t_sorted_w = [t_raw_w[i] for i in order]

        # MLX scores for all 256 experts
        scores_t = np.array(mlx_scores[t])
        scores_sorted_indices = np.argsort(-scores_t)
        top7_scores = [float(scores_t[scores_sorted_indices[i]]) for i in range(7)]
        margin_5_6 = top7_scores[5] - top7_scores[6]

        # Independent FP32 reference (dot product + sigmoid + greedy top-6 with lowest-index tie breaking)
        py_logits = np.dot(x_f32[t], w_f32.T)
        py_scores = 1.0 / (1.0 + np.exp(-py_logits))
        # Greedy top-6
        py_scores_copy = py_scores.copy()
        py_idx = []
        py_top_scores = []
        for _ in range(6):
            best_i = int(np.argmax(py_scores_copy))
            py_idx.append(best_i)
            py_top_scores.append(py_scores[best_i])
            py_scores_copy[best_i] = -1e30
        py_top_scores = np.array(py_top_scores)
        py_weights = [float(w) for w in (py_top_scores / (np.sum(py_top_scores) + 1e-20))]

        records.append({
            "token_idx": t,
            "mlx_top6_indices": t_sorted_idx,
            "mlx_top6_weights": t_sorted_w,
            "mlx_raw_indices": t_raw_idx,
            "mlx_raw_weights": t_raw_w,
            "mlx_top7_scores": top7_scores,
            "mlx_margin_rank5_vs_rank6": float(margin_5_6),
            "py_top6_indices": py_idx,
            "py_top6_weights": py_weights
        })

    out_data = {
        "metadata": {
            "source": "Actual Flint MLX SparseMoeBlock.route (MLX 0.32.4)",
            "layer": 2,
            "tokens": num_tokens,
            "dim": 2048,
            "num_experts": 256,
            "top_k": 6
        },
        "input_hex": bytes(x.view(mx.uint16)).hex(),
        "gate_weight_hex": bytes(w_l2.view(mx.uint16)).hex(),
        "tokens": records
    }

    out_path = "Tests/FlintFieldfareCoreTests/Fixtures/router_100_tokens_mlx.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved {num_tokens} tokens to {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
