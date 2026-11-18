#!/usr/bin/env python3
"""
Generate golden fixtures for Flint Layer 0 Dense Decoder Forward (F3-01A).
Uses actual MLX 0.32.4 Flint model with live checkpoint weights from:
/Users/tanishq/models/flint-mixed-3-4bit

Captures all 10 intermediate states:
1. Layer input hidden state: x
2. Input RMSNorm output: norm1
3. Raw Q/K/V projections: q_raw, k_raw, v_raw
4. Q/K normalized states: q_norm, k_norm
5. Attention output before output projection: attn_out
6. Attention output projection: attn_proj
7. First residual state: res1 = x + attn_proj
8. Post-attention RMSNorm: norm2
9. Dense MLP output: mlp_out
10. Final Layer 0 hidden state: final_out = res1 + mlp_out
"""

import os
import json
import numpy as np
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import scaled_dot_product_attention

def main():
    model_path = "/Users/tanishq/models/flint-mixed-3-4bit"
    print(f"Loading model from {model_path}...")
    model, processor = load(model_path)
    l0 = model.layers[0]
    print("Layer 0 loaded successfully.")

    cases = [
        {"name": "layer0_decode_t1_bf16", "batch": 1, "tokens": 1, "dtype": "bfloat16"},
        {"name": "layer0_prefill_t4_bf16", "batch": 1, "tokens": 4, "dtype": "bfloat16"},
        {"name": "layer0_prefill_t8_bf16", "batch": 1, "tokens": 8, "dtype": "bfloat16"},
        {"name": "layer0_decode_zero_bf16", "batch": 1, "tokens": 1, "dtype": "bfloat16", "zero": True},
    ]

    results = []

    for c in cases:
        mx.random.seed(c["tokens"] * 100 + c["batch"])
        shape = (c["batch"], c["tokens"], 2048)
        if c.get("zero", False):
            x = mx.zeros(shape).astype(mx.bfloat16)
        else:
            x = mx.random.normal(shape).astype(mx.bfloat16)

        # 1. Input hidden state
        s1_input = x

        # 2. Input RMSNorm output
        s2_norm1 = l0.input_layernorm(x)

        # 3. Raw Q/K/V projections
        batch, length, _ = s2_norm1.shape
        q_raw = l0.self_attn.q_proj(s2_norm1).reshape(batch, length, 32, 128)
        k_raw = l0.self_attn.k_proj(s2_norm1).reshape(batch, length, 4, 128)
        v_raw = l0.self_attn.v_proj(s2_norm1).reshape(batch, length, 4, 128)

        # 4. Q/K normalized states
        q_norm_raw = l0.self_attn.q_norm(q_raw)
        k_norm_raw = l0.self_attn.k_norm(k_raw)
        q_norm = q_norm_raw.transpose(0, 2, 1, 3) # [batch, 32, length, 128]
        k_norm = k_norm_raw.transpose(0, 2, 1, 3) # [batch, 4, length, 128]
        v = v_raw.transpose(0, 2, 1, 3) # [batch, 4, length, 128]

        # 5. Attention output before output projection
        mask = "causal" if length > 1 else None
        attn_out = scaled_dot_product_attention(
            q_norm, k_norm, v,
            cache=None,
            scale=l0.self_attn.scale,
            mask=mask,
            sinks=l0.self_attn.softmax_offset.astype(q_norm.dtype)
        )

        # 6. Attention output projection
        attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        attn_proj = l0.self_attn.o_proj(attn_flat)

        # 7. First residual
        s7_res1 = x + attn_proj

        # 8. Post-attention RMSNorm
        s8_norm2 = l0.post_attention_layernorm(s7_res1)

        # 9. Dense MLP output
        s9_mlp_out = l0.mlp(s8_norm2)

        # 10. Final Layer 0 hidden state
        s10_final = s7_res1 + s9_mlp_out

        # Full layer forward directly via module
        full_layer_out = l0(x, mask=mask)

        mx.eval(s1_input, s2_norm1, q_raw, k_raw, v_raw, q_norm, k_norm, v,
                attn_out, attn_flat, attn_proj, s7_res1, s8_norm2, s9_mlp_out, s10_final, full_layer_out)

        case_dict = {
            "name": c["name"],
            "batch": c["batch"],
            "num_tokens": c["tokens"],
            "dtype": c["dtype"],
            "input_hex": bytes(s1_input.view(mx.uint16)).hex(),
            "norm1_hex": bytes(s2_norm1.view(mx.uint16)).hex(),
            "q_raw_hex": bytes(q_raw.view(mx.uint16)).hex(),
            "k_raw_hex": bytes(k_raw.view(mx.uint16)).hex(),
            "v_raw_hex": bytes(v_raw.view(mx.uint16)).hex(),
            "q_norm_hex": bytes(q_norm_raw.view(mx.uint16)).hex(),
            "k_norm_hex": bytes(k_norm_raw.view(mx.uint16)).hex(),
            "attn_out_hex": bytes(attn_out.view(mx.uint16)).hex(),
            "attn_flat_hex": bytes(attn_flat.view(mx.uint16)).hex(),
            "attn_proj_hex": bytes(attn_proj.view(mx.uint16)).hex(),
            "res1_hex": bytes(s7_res1.view(mx.uint16)).hex(),
            "norm2_hex": bytes(s8_norm2.view(mx.uint16)).hex(),
            "mlp_out_hex": bytes(s9_mlp_out.view(mx.uint16)).hex(),
            "final_out_hex": bytes(s10_final.view(mx.uint16)).hex(),
            "full_layer_out_hex": bytes(full_layer_out.view(mx.uint16)).hex(),
        }

        diff = mx.max(mx.abs(s10_final - full_layer_out)).item()
        print(f"Generated {c['name']} (deconstruction vs full layer max diff: {diff:.8f})")
        results.append(case_dict)

    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fixtures_dir, "flint_layer0_golden.json")
    with open(out_path, "w") as f:
        json.dump({"generator": "generate_layer0_golden.py", "cases": results}, f, indent=2)

    print(f"Saved {len(results)} cases to {out_path} successfully!")

if __name__ == "__main__":
    main()
