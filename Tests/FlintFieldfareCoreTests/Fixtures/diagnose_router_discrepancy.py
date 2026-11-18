#!/usr/bin/env python3
"""
Diagnostic script for characterizing router selection discrepancy between
Metal serial accumulation and MLX GEMM / argpartition on Apple Silicon.
"""

import json
import os
import numpy as np
import mlx.core as mx
from mlx_vlm.models.flint.config import ModelConfig
from mlx_vlm.models.flint.language import SparseMoeBlock

def main():
    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    args = ModelConfig.from_dict(text_cfg)

    block = SparseMoeBlock(args)
    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    block.gate.weight = shard1["model.layers.2.mlp.gate.weight"]

    np.random.seed(42)
    num_samples = 100
    mismatches = 0
    boundary_margins = []
    detailed_cases = []

    w_f32 = np.array(block.gate.weight.astype(mx.float32))

    for s in range(num_samples):
        x_np = np.random.randn(1, 1, 2048).astype(np.float32)
        x_bf16 = mx.array(x_np).astype(mx.bfloat16)

        # 1. MLX reference
        mlx_idx, mlx_w = block.route(x_bf16)
        mlx_set = set(np.array(mlx_idx[0, 0]).tolist())

        # MLX all scores
        scores_mlx = np.array(mx.sigmoid(block.gate(x_bf16).astype(mx.float32))[0, 0])
        mlx_sorted = np.argsort(-scores_mlx)

        # 2. Metal serial FP32 accumulation
        x_bf16_as_f32 = np.array(x_bf16.astype(mx.float32))[0, 0]
        logits_serial = np.zeros(256, dtype=np.float32)
        for e in range(256):
            acc = 0.0
            for i in range(0, 2048, 4):
                acc += float(x_bf16_as_f32[i+0]) * float(w_f32[e, i+0])
                acc += float(x_bf16_as_f32[i+1]) * float(w_f32[e, i+1])
                acc += float(x_bf16_as_f32[i+2]) * float(w_f32[e, i+2])
                acc += float(x_bf16_as_f32[i+3]) * float(w_f32[e, i+3])
            logits_serial[e] = acc
        scores_serial = 1.0 / (1.0 + np.exp(-logits_serial))
        serial_sorted = np.argsort(-scores_serial)
        serial_set = set(serial_sorted[:6].tolist())

        if mlx_set != serial_set:
            mismatches += 1
            e5_mlx = mlx_sorted[5]
            e6_mlx = mlx_sorted[6]
            margin_mlx = abs(scores_mlx[e5_mlx] - scores_mlx[e6_mlx])
            margin_serial = abs(scores_serial[serial_sorted[5]] - scores_serial[serial_sorted[6]])
            boundary_margins.append((margin_mlx, margin_serial))

            diff_expert_mlx = mlx_set - serial_set
            diff_expert_serial = serial_set - mlx_set

            if len(detailed_cases) < 5:
                detailed_cases.append({
                    "sample": s,
                    "diff_mlx": list(diff_expert_mlx),
                    "diff_serial": list(diff_expert_serial),
                    "score_mlx_e5": float(scores_mlx[e5_mlx]),
                    "score_mlx_e6": float(scores_mlx[e6_mlx]),
                    "margin_mlx": float(margin_mlx),
                    "score_serial_e5": float(scores_serial[serial_sorted[5]]),
                    "score_serial_e6": float(scores_serial[serial_sorted[6]]),
                    "margin_serial": float(margin_serial),
                    "score_diff_for_disputed": float(abs(scores_mlx[list(diff_expert_mlx)[0]] - scores_serial[list(diff_expert_mlx)[0]]))
                })

    print("=== ROUTER 100-TOKEN DETERMINISTIC CHARACTERIZATION ===")
    print(f"Total Tokens Evaluated   : {num_samples}")
    print(f"Expert Set Match (6/6)   : {num_samples - mismatches} / {num_samples} ({((num_samples - mismatches)/num_samples)*100:.1f}%)")
    print(f"Expert Set Mismatches    : {mismatches} / {num_samples} ({(mismatches/num_samples)*100:.1f}%)")
    if boundary_margins:
        m_mlx = [m[0] for m in boundary_margins]
        print(f"Boundary Score Margin Min: {min(m_mlx):.8e}")
        print(f"Boundary Score Margin P50: {np.median(m_mlx):.8e}")
        print(f"Boundary Score Margin Max: {max(m_mlx):.8e}")
    print("\nDetailed Mismatch Cases:")
    for d in detailed_cases:
        print(d)

if __name__ == "__main__":
    main()
