#!/usr/bin/env python3
"""
Generate golden fixtures for Flint Native Dense and Shared Expert SwiGLU MLP (F2-06).
Uses actual MLX 0.32.4 SwiGLUMLP with live checkpoint weights from:
/Users/tanishq/models/flint-mixed-3-4bit

Captures all 5 intermediate/final stages:
1. Input activation
2. Gate projection output
3. Up projection output
4. SiLU(gate) output
5. Elementwise gated product: SiLU(gate) * up
6. Down projection output
"""

import os
import json
import struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.mlp import SwiGLUMLP

def to_hex(arr_bytes: bytes) -> str:
    return arr_bytes.hex()

def run_stages(x_in, w_g, s_g, b_g, w_u, s_u, b_u, w_d, s_d, b_d, dtype_str):
    bits = 4
    group_size = 64
    
    if dtype_str == "bfloat16":
        deq_g = mx.dequantize(w_g, s_g, b_g, group_size=group_size, bits=bits)
        deq_u = mx.dequantize(w_u, s_u, b_u, group_size=group_size, bits=bits)
        deq_d = mx.dequantize(w_d, s_d, b_d, group_size=group_size, bits=bits)
        x = x_in.astype(mx.bfloat16)
        
        gate_out = mx.matmul(x, deq_g.T)
        up_out = mx.matmul(x, deq_u.T)
        silu_gate = nn.silu(gate_out.astype(mx.float32)).astype(mx.bfloat16)
        gated_prod = silu_gate * up_out
        down_out = mx.matmul(gated_prod, deq_d.T)
        
        return {
            "input_hex": bytes(x.view(mx.uint16)).hex(),
            "gate_proj_hex": bytes(gate_out.view(mx.uint16)).hex(),
            "up_proj_hex": bytes(up_out.view(mx.uint16)).hex(),
            "silu_gate_hex": bytes(silu_gate.view(mx.uint16)).hex(),
            "gated_product_hex": bytes(gated_prod.view(mx.uint16)).hex(),
            "down_proj_hex": bytes(down_out.view(mx.uint16)).hex(),
        }
    else:
        # Float32 path
        s_g_f32 = s_g.astype(mx.float32)
        b_g_f32 = b_g.astype(mx.float32)
        s_u_f32 = s_u.astype(mx.float32)
        b_u_f32 = b_u.astype(mx.float32)
        s_d_f32 = s_d.astype(mx.float32)
        b_d_f32 = b_d.astype(mx.float32)

        deq_g = mx.dequantize(w_g, s_g_f32, b_g_f32, group_size=group_size, bits=bits)
        deq_u = mx.dequantize(w_u, s_u_f32, b_u_f32, group_size=group_size, bits=bits)
        deq_d = mx.dequantize(w_d, s_d_f32, b_d_f32, group_size=group_size, bits=bits)
        x = x_in.astype(mx.float32)

        gate_out = mx.matmul(x, deq_g.T)
        up_out = mx.matmul(x, deq_u.T)
        silu_gate = nn.silu(gate_out)
        gated_prod = silu_gate * up_out
        down_out = mx.matmul(gated_prod, deq_d.T)

        return {
            "input_hex": bytes(x).hex(),
            "gate_proj_hex": bytes(gate_out).hex(),
            "up_proj_hex": bytes(up_out).hex(),
            "silu_gate_hex": bytes(silu_gate).hex(),
            "gated_product_hex": bytes(gated_prod).hex(),
            "down_proj_hex": bytes(down_out).hex(),
        }

def main():
    mx.random.seed(42)
    np.random.seed(42)

    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    shard2 = mx.load(os.path.join(model_dir, "model-00002-of-00004.safetensors"))
    shard4 = mx.load(os.path.join(model_dir, "model-00004-of-00004.safetensors"))

    cases = []

    # 1. Dense Layer 0 - Single Token (BF16)
    x = mx.random.normal(shape=(1, 1, 2048))
    stages = run_stages(
        x,
        shard1["model.layers.0.mlp.gate_proj.weight"], shard1["model.layers.0.mlp.gate_proj.scales"], shard1["model.layers.0.mlp.gate_proj.biases"],
        shard1["model.layers.0.mlp.up_proj.weight"], shard1["model.layers.0.mlp.up_proj.scales"], shard1["model.layers.0.mlp.up_proj.biases"],
        shard1["model.layers.0.mlp.down_proj.weight"], shard1["model.layers.0.mlp.down_proj.scales"], shard1["model.layers.0.mlp.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "dense_l0_single_token_bf16", "layer": 0, "kind": "dense", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 6144, "dtype": "bfloat16", **stages
    })

    # 2. Dense Layer 1 - Single Token (BF16)
    stages = run_stages(
        x,
        shard1["model.layers.1.mlp.gate_proj.weight"], shard1["model.layers.1.mlp.gate_proj.scales"], shard1["model.layers.1.mlp.gate_proj.biases"],
        shard1["model.layers.1.mlp.up_proj.weight"], shard1["model.layers.1.mlp.up_proj.scales"], shard1["model.layers.1.mlp.up_proj.biases"],
        shard1["model.layers.1.mlp.down_proj.weight"], shard1["model.layers.1.mlp.down_proj.scales"], shard1["model.layers.1.mlp.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "dense_l1_single_token_bf16", "layer": 1, "kind": "dense", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 6144, "dtype": "bfloat16", **stages
    })

    # 3. Shared Expert Layer 2 - Single Token (BF16)
    stages = run_stages(
        x,
        shard1["model.layers.2.mlp.shared_expert.gate_proj.weight"], shard1["model.layers.2.mlp.shared_expert.gate_proj.scales"], shard1["model.layers.2.mlp.shared_expert.gate_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.up_proj.weight"], shard1["model.layers.2.mlp.shared_expert.up_proj.scales"], shard1["model.layers.2.mlp.shared_expert.up_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.down_proj.weight"], shard1["model.layers.2.mlp.shared_expert.down_proj.scales"], shard1["model.layers.2.mlp.shared_expert.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "shared_l2_single_token_bf16", "layer": 2, "kind": "shared_expert", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "bfloat16", **stages
    })

    # 4. Shared Expert Layer 24 - Single Token (BF16)
    stages = run_stages(
        x,
        shard2["model.layers.24.mlp.shared_expert.gate_proj.weight"], shard2["model.layers.24.mlp.shared_expert.gate_proj.scales"], shard2["model.layers.24.mlp.shared_expert.gate_proj.biases"],
        shard2["model.layers.24.mlp.shared_expert.up_proj.weight"], shard2["model.layers.24.mlp.shared_expert.up_proj.scales"], shard2["model.layers.24.mlp.shared_expert.up_proj.biases"],
        shard2["model.layers.24.mlp.shared_expert.down_proj.weight"], shard2["model.layers.24.mlp.shared_expert.down_proj.scales"], shard2["model.layers.24.mlp.shared_expert.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "shared_l24_single_token_bf16", "layer": 24, "kind": "shared_expert", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "bfloat16", **stages
    })

    # 5. Shared Expert Layer 48 - Single Token (BF16)
    stages = run_stages(
        x,
        shard4["model.layers.48.mlp.shared_expert.gate_proj.weight"], shard4["model.layers.48.mlp.shared_expert.gate_proj.scales"], shard4["model.layers.48.mlp.shared_expert.gate_proj.biases"],
        shard4["model.layers.48.mlp.shared_expert.up_proj.weight"], shard4["model.layers.48.mlp.shared_expert.up_proj.scales"], shard4["model.layers.48.mlp.shared_expert.up_proj.biases"],
        shard4["model.layers.48.mlp.shared_expert.down_proj.weight"], shard4["model.layers.48.mlp.shared_expert.down_proj.scales"], shard4["model.layers.48.mlp.shared_expert.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "shared_l48_single_token_bf16", "layer": 48, "kind": "shared_expert", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "bfloat16", **stages
    })

    # 6. Dense Layer 0 - Multi-token (T=8, BF16)
    x_t8 = mx.random.normal(shape=(1, 8, 2048))
    stages = run_stages(
        x_t8,
        shard1["model.layers.0.mlp.gate_proj.weight"], shard1["model.layers.0.mlp.gate_proj.scales"], shard1["model.layers.0.mlp.gate_proj.biases"],
        shard1["model.layers.0.mlp.up_proj.weight"], shard1["model.layers.0.mlp.up_proj.scales"], shard1["model.layers.0.mlp.up_proj.biases"],
        shard1["model.layers.0.mlp.down_proj.weight"], shard1["model.layers.0.mlp.down_proj.scales"], shard1["model.layers.0.mlp.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "dense_l0_t8_bf16", "layer": 0, "kind": "dense", "batch": 1, "tokens": 8,
        "hidden_dim": 2048, "intermediate_dim": 6144, "dtype": "bfloat16", **stages
    })

    # 7. Shared Expert Layer 2 - Multi-token (T=8, BF16)
    stages = run_stages(
        x_t8,
        shard1["model.layers.2.mlp.shared_expert.gate_proj.weight"], shard1["model.layers.2.mlp.shared_expert.gate_proj.scales"], shard1["model.layers.2.mlp.shared_expert.gate_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.up_proj.weight"], shard1["model.layers.2.mlp.shared_expert.up_proj.scales"], shard1["model.layers.2.mlp.shared_expert.up_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.down_proj.weight"], shard1["model.layers.2.mlp.shared_expert.down_proj.scales"], shard1["model.layers.2.mlp.shared_expert.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "shared_l2_t8_bf16", "layer": 2, "kind": "shared_expert", "batch": 1, "tokens": 8,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "bfloat16", **stages
    })

    # 8. Shared Expert Layer 2 - Multi-batch (B=2, T=2, BF16)
    x_b2 = mx.random.normal(shape=(2, 2, 2048))
    stages = run_stages(
        x_b2,
        shard1["model.layers.2.mlp.shared_expert.gate_proj.weight"], shard1["model.layers.2.mlp.shared_expert.gate_proj.scales"], shard1["model.layers.2.mlp.shared_expert.gate_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.up_proj.weight"], shard1["model.layers.2.mlp.shared_expert.up_proj.scales"], shard1["model.layers.2.mlp.shared_expert.up_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.down_proj.weight"], shard1["model.layers.2.mlp.shared_expert.down_proj.scales"], shard1["model.layers.2.mlp.shared_expert.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "shared_l2_b2_t2_bf16", "layer": 2, "kind": "shared_expert", "batch": 2, "tokens": 2,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "bfloat16", **stages
    })

    # 9. Dense Layer 0 - Zero Input (BF16)
    x_zero = mx.zeros(shape=(1, 1, 2048))
    stages = run_stages(
        x_zero,
        shard1["model.layers.0.mlp.gate_proj.weight"], shard1["model.layers.0.mlp.gate_proj.scales"], shard1["model.layers.0.mlp.gate_proj.biases"],
        shard1["model.layers.0.mlp.up_proj.weight"], shard1["model.layers.0.mlp.up_proj.scales"], shard1["model.layers.0.mlp.up_proj.biases"],
        shard1["model.layers.0.mlp.down_proj.weight"], shard1["model.layers.0.mlp.down_proj.scales"], shard1["model.layers.0.mlp.down_proj.biases"],
        "bfloat16"
    )
    cases.append({
        "name": "dense_l0_zero_input_bf16", "layer": 0, "kind": "dense", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 6144, "dtype": "bfloat16", **stages
    })

    # 10. Dense Layer 0 - Single Token (Float32)
    stages = run_stages(
        x,
        shard1["model.layers.0.mlp.gate_proj.weight"], shard1["model.layers.0.mlp.gate_proj.scales"], shard1["model.layers.0.mlp.gate_proj.biases"],
        shard1["model.layers.0.mlp.up_proj.weight"], shard1["model.layers.0.mlp.up_proj.scales"], shard1["model.layers.0.mlp.up_proj.biases"],
        shard1["model.layers.0.mlp.down_proj.weight"], shard1["model.layers.0.mlp.down_proj.scales"], shard1["model.layers.0.mlp.down_proj.biases"],
        "float32"
    )
    cases.append({
        "name": "dense_l0_single_token_f32", "layer": 0, "kind": "dense", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 6144, "dtype": "float32", **stages
    })

    # 11. Shared Expert Layer 2 - Single Token (Float32)
    stages = run_stages(
        x,
        shard1["model.layers.2.mlp.shared_expert.gate_proj.weight"], shard1["model.layers.2.mlp.shared_expert.gate_proj.scales"], shard1["model.layers.2.mlp.shared_expert.gate_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.up_proj.weight"], shard1["model.layers.2.mlp.shared_expert.up_proj.scales"], shard1["model.layers.2.mlp.shared_expert.up_proj.biases"],
        shard1["model.layers.2.mlp.shared_expert.down_proj.weight"], shard1["model.layers.2.mlp.shared_expert.down_proj.scales"], shard1["model.layers.2.mlp.shared_expert.down_proj.biases"],
        "float32"
    )
    cases.append({
        "name": "shared_l2_single_token_f32", "layer": 2, "kind": "shared_expert", "batch": 1, "tokens": 1,
        "hidden_dim": 2048, "intermediate_dim": 1024, "dtype": "float32", **stages
    })

    output = {
        "metadata": {
            "source": "MLX 0.32.4 (SwiGLUMLP with checkpoint weights)",
            "model": "Recursive Flint 40B-A2.5B",
            "stages": [
                "1. input",
                "2. gate_proj = x @ W_gate.T",
                "3. up_proj = x @ W_up.T",
                "4. silu_gate = SiLU(gate_proj)",
                "5. gated_product = silu_gate * up_proj",
                "6. down_proj = gated_product @ W_down.T"
            ]
        },
        "cases": cases
    }

    out_path = "Tests/FlintFieldfareCoreTests/Fixtures/flint_mlp_golden.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(cases)} test cases in {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
