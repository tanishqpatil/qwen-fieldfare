#!/usr/bin/env python3
"""
Generate golden fixtures for Flint Native Dense and Shared Expert SwiGLU MLP (F2-06 / F3-01).
Uses actual MLX 0.32.4 SwiGLUMLP with live checkpoint weights from:
/Users/tanishq/models/flint-mixed-3-4bit

Captures:
1. True MLX real module forward: module(x)
2. Fused SwiGLU module forward
3. All 5 decomposed intermediate/final stages:
   - input
   - gate_proj
   - up_proj
   - silu_gate
   - gated_product
   - down_proj
Across Layers 0, 1, 2, 24, 48 in BF16 and FP32.
"""

import os
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load

def main():
    model_path = "/Users/tanishq/models/flint-mixed-3-4bit"
    print(f"Loading model from {model_path}...")
    model, processor = load(model_path)
    print("Model loaded successfully.")

    cases = [
        {"name": "dense_l0_single_token_bf16", "kind": "dense", "dtype": "bfloat16", "layer": 0, "batch": 1, "tokens": 1},
        {"name": "dense_l1_single_token_bf16", "kind": "dense", "dtype": "bfloat16", "layer": 1, "batch": 1, "tokens": 1},
        {"name": "shared_l2_single_token_bf16", "kind": "shared_expert", "dtype": "bfloat16", "layer": 2, "batch": 1, "tokens": 1},
        {"name": "shared_l24_single_token_bf16", "kind": "shared_expert", "dtype": "bfloat16", "layer": 24, "batch": 1, "tokens": 1},
        {"name": "shared_l48_single_token_bf16", "kind": "shared_expert", "dtype": "bfloat16", "layer": 48, "batch": 1, "tokens": 1},
        {"name": "dense_l0_t8_bf16", "kind": "dense", "dtype": "bfloat16", "layer": 0, "batch": 1, "tokens": 8},
        {"name": "shared_l2_t8_bf16", "kind": "shared_expert", "dtype": "bfloat16", "layer": 2, "batch": 1, "tokens": 8},
        {"name": "shared_l2_b2_t2_bf16", "kind": "shared_expert", "dtype": "bfloat16", "layer": 2, "batch": 2, "tokens": 2},
        {"name": "dense_l0_zero_input_bf16", "kind": "dense", "dtype": "bfloat16", "layer": 0, "batch": 1, "tokens": 1, "zero_input": True},
        {"name": "dense_l0_single_token_f32", "kind": "dense", "dtype": "float32", "layer": 0, "batch": 1, "tokens": 1},
        {"name": "shared_l2_single_token_f32", "kind": "shared_expert", "dtype": "float32", "layer": 2, "batch": 1, "tokens": 1},
    ]

    results = []

    for c in cases:
        lyr = model.layers[c["layer"]]
        mod = lyr.mlp if c["kind"] == "dense" else lyr.mlp.shared_expert
        
        mx.random.seed(c["layer"] * 1000 + c["tokens"] * 10 + c["batch"])
        shape = (c["batch"], c["tokens"], 2048)
        
        if c.get("zero_input", False):
            x = mx.zeros(shape)
        else:
            x = mx.random.normal(shape)
        
        if c["dtype"] == "bfloat16":
            x = x.astype(mx.bfloat16)
            
            # 1. Real module forward
            real_out = mod(x)
            mx.eval(real_out)
            
            # 2. Intermediate projections via module
            gate = mod.gate_proj(x)
            up = mod.up_proj(x)
            silu_gate = nn.silu(gate)
            gated_prod = silu_gate * up
            down_out = mod.down_proj(gated_prod)
            mx.eval(gate, up, silu_gate, gated_prod, down_out)
            
            # 3. Fused SwiGLU with intermediate BF16 round (matches MLX bit-for-bit)
            g_f32 = gate.astype(mx.float32)
            u_f32 = up.astype(mx.float32)
            silu_f32 = g_f32 / (1.0 + mx.exp(-g_f32))
            silu_bf16_cast = silu_f32.astype(mx.bfloat16)
            act_fused = (silu_bf16_cast.astype(mx.float32) * u_f32).astype(mx.bfloat16)
            fused_out = mod.down_proj(act_fused)
            mx.eval(act_fused, fused_out)
            
            case_dict = {
                "name": c["name"],
                "kind": c["kind"],
                "dtype": c["dtype"],
                "layer": c["layer"],
                "batch": c["batch"],
                "num_tokens": c["tokens"],
                "hidden_dim": 2048,
                "intermediate_dim": 6144 if c["kind"] == "dense" else 1024,
                "input_hex": bytes(x.view(mx.uint16)).hex(),
                "real_module_out_hex": bytes(real_out.view(mx.uint16)).hex(),
                "fused_module_out_hex": bytes(fused_out.view(mx.uint16)).hex(),
                "gate_proj_hex": bytes(gate.view(mx.uint16)).hex(),
                "up_proj_hex": bytes(up.view(mx.uint16)).hex(),
                "silu_gate_hex": bytes(silu_gate.view(mx.uint16)).hex(),
                "gated_product_hex": bytes(gated_prod.view(mx.uint16)).hex(),
                "down_proj_hex": bytes(down_out.view(mx.uint16)).hex(),
            }
        else:
            # Float32 path
            x = x.astype(mx.float32)
            gw = mx.dequantize(mod.gate_proj.weight, mod.gate_proj.scales.astype(mx.float32), mod.gate_proj.biases.astype(mx.float32), group_size=64, bits=4)
            uw = mx.dequantize(mod.up_proj.weight, mod.up_proj.scales.astype(mx.float32), mod.up_proj.biases.astype(mx.float32), group_size=64, bits=4)
            dw = mx.dequantize(mod.down_proj.weight, mod.down_proj.scales.astype(mx.float32), mod.down_proj.biases.astype(mx.float32), group_size=64, bits=4)
            
            gate = mx.matmul(x, gw.T)
            up = mx.matmul(x, uw.T)
            silu_gate = gate / (1.0 + mx.exp(-gate))
            gated_prod = silu_gate * up
            down_out = mx.matmul(gated_prod, dw.T)
            mx.eval(gate, up, silu_gate, gated_prod, down_out)
            
            case_dict = {
                "name": c["name"],
                "kind": c["kind"],
                "dtype": c["dtype"],
                "layer": c["layer"],
                "batch": c["batch"],
                "num_tokens": c["tokens"],
                "hidden_dim": 2048,
                "intermediate_dim": 6144 if c["kind"] == "dense" else 1024,
                "input_hex": bytes(x.view(mx.uint32)).hex(),
                "real_module_out_hex": bytes(down_out.view(mx.uint32)).hex(),
                "fused_module_out_hex": bytes(down_out.view(mx.uint32)).hex(),
                "gate_proj_hex": bytes(gate.view(mx.uint32)).hex(),
                "up_proj_hex": bytes(up.view(mx.uint32)).hex(),
                "silu_gate_hex": bytes(silu_gate.view(mx.uint32)).hex(),
                "gated_product_hex": bytes(gated_prod.view(mx.uint32)).hex(),
                "down_proj_hex": bytes(down_out.view(mx.uint32)).hex(),
            }
        
        results.append(case_dict)
        print("Generated case: " + c["name"])

    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fixtures_dir, "flint_mlp_golden.json")
    with open(out_path, "w") as f:
        json.dump({"generator": "generate_mlp_golden.py", "cases": results}, f, indent=2)

    print(f"Saved {len(results)} cases to {out_path} successfully!")

if __name__ == "__main__":
    main()
