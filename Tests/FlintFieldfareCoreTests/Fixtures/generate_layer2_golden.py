#!/usr/bin/env python3
"""
Generate golden fixtures for Flint Layer 2 MoE Decoder Forward (F3-01B).
Uses actual MLX 0.32.4 Flint model with live checkpoint weights from:
/Users/tanishq/models/flint-mixed-3-4bit

Captures all 14 intermediate states:
1. Layer input hidden state: x
2. Input RMSNorm output: norm1
3. Raw Q/K/V projections: q_raw, k_raw, v_raw
4. Q/K normalized states: q_norm, k_norm
5. Q/K RoPE rotated states: q_rope, k_rope
6. Attention output before output projection: attn_out
7. Attention output projection: attn_proj
8. First residual state: res1 = x + attn_proj
9. Post-attention RMSNorm: norm2
10. Router selected expert IDs and normalized weights: indices, weights
11. Routed expert contributions: routed [batch, tokens, 6, 2048]
12. Shared expert output: shared_out [batch, tokens, 2048]
13. MoE block output: moe_out [batch, tokens, 2048]
14. Final Layer 2 hidden state: final_out = res1 + moe_out
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
    l2 = model.layers[2]
    print("Layer 2 loaded successfully.")

    cases = [
        {"name": "layer2_decode_t1_bf16", "batch": 1, "tokens": 1, "dtype": "bfloat16"},
        {"name": "layer2_prefill_t4_bf16", "batch": 1, "tokens": 4, "dtype": "bfloat16"},
        {"name": "layer2_decode_zero_bf16", "batch": 1, "tokens": 1, "dtype": "bfloat16", "zero": True},
    ]

    results = []

    for c in cases:
        mx.random.seed(c["tokens"] * 200 + c["batch"])
        shape = (c["batch"], c["tokens"], 2048)
        if c.get("zero", False):
            x = mx.zeros(shape).astype(mx.bfloat16)
        else:
            x = mx.random.normal(shape).astype(mx.bfloat16)

        # 1. Input hidden state
        s1_input = x

        # 2. Input RMSNorm output
        s2_norm1 = l2.input_layernorm(x)

        # 3. Raw Q/K/V projections
        batch, length, _ = s2_norm1.shape
        q_raw = l2.self_attn.q_proj(s2_norm1).reshape(batch, length, 32, 128)
        k_raw = l2.self_attn.k_proj(s2_norm1).reshape(batch, length, 4, 128)
        v_raw = l2.self_attn.v_proj(s2_norm1).reshape(batch, length, 4, 128)

        # 4. Q/K normalized states
        q_norm_raw = l2.self_attn.q_norm(q_raw)
        k_norm_raw = l2.self_attn.k_norm(k_raw)
        q_norm = q_norm_raw.transpose(0, 2, 1, 3) # [batch, 32, length, 128]
        k_norm = k_norm_raw.transpose(0, 2, 1, 3) # [batch, 4, length, 128]
        v = v_raw.transpose(0, 2, 1, 3) # [batch, 4, length, 128]

        # 5. RoPE rotated states
        q_rope = l2.self_attn.rope(q_norm, offset=0)
        k_rope = l2.self_attn.rope(k_norm, offset=0)

        # 6. Attention output before output projection
        mask = "causal" if length > 1 else None
        attn_out = scaled_dot_product_attention(
            q_rope, k_rope, v,
            cache=None,
            scale=l2.self_attn.scale,
            mask=mask,
            sinks=l2.self_attn.softmax_offset.astype(q_rope.dtype)
        )

        # 7. Attention output projection
        attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        attn_proj = l2.self_attn.o_proj(attn_flat)

        # 8. First residual
        s8_res1 = x + attn_proj

        # 9. Post-attention RMSNorm
        s9_norm2 = l2.post_attention_layernorm(s8_res1)

        # 10. Router selection
        indices, router_weights = l2.mlp.route(s9_norm2)
        if c.get("zero", False):
            # For zero input, all router scores are identical; Metal selects the canonical first 6 experts [0..5]
            indices = mx.broadcast_to(mx.arange(6, dtype=indices.dtype), indices.shape)
            scores = mx.sigmoid(l2.mlp.gate(s9_norm2).astype(mx.float32))
            router_weights = mx.take_along_axis(scores, indices, axis=-1)
            router_weights = router_weights / (router_weights.sum(axis=-1, keepdims=True) + 1e-20)
        else:
            # Sort descending by weight to match Metal's canonical descending-score router output:
            sort_order = mx.argsort(-router_weights, axis=-1)
            indices = mx.take_along_axis(indices, sort_order, axis=-1)
            router_weights = mx.take_along_axis(router_weights, sort_order, axis=-1)

        # 11. Routed expert contributions
        routed = l2.mlp.switch_mlp(s9_norm2, indices)

        # 12. Shared expert output
        shared_out = l2.mlp.shared_expert(s9_norm2)

        # 13. MoE block reduction
        moe_out = (routed.astype(mx.float32) * router_weights[..., None]).sum(axis=-2).astype(x.dtype) + shared_out

        # 14. Final Layer 2 hidden state
        s14_final = s8_res1 + moe_out

        # Full layer forward directly via module
        full_layer_out = l2(x, mask=mask)

        mx.eval(s1_input, s2_norm1, q_raw, k_raw, v_raw, q_norm_raw, k_norm_raw,
                q_rope, k_rope, attn_out, attn_flat, attn_proj, s8_res1, s9_norm2,
                indices, router_weights, routed, shared_out, moe_out, s14_final, full_layer_out)

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
            "q_rope_hex": bytes(q_rope.view(mx.uint16)).hex(),
            "k_rope_hex": bytes(k_rope.view(mx.uint16)).hex(),
            "attn_out_hex": bytes(attn_out.view(mx.uint16)).hex(),
            "attn_flat_hex": bytes(attn_flat.view(mx.uint16)).hex(),
            "attn_proj_hex": bytes(attn_proj.view(mx.uint16)).hex(),
            "res1_hex": bytes(s8_res1.view(mx.uint16)).hex(),
            "norm2_hex": bytes(s9_norm2.view(mx.uint16)).hex(),
            "router_indices": indices.tolist(),
            "router_weights": router_weights.tolist(),
            "routed_hex": bytes(routed.view(mx.uint16)).hex(),
            "shared_out_hex": bytes(shared_out.view(mx.uint16)).hex(),
            "moe_out_hex": bytes(moe_out.view(mx.uint16)).hex(),
            "final_out_hex": bytes(s14_final.view(mx.uint16)).hex(),
            "full_layer_out_hex": bytes(full_layer_out.view(mx.uint16)).hex(),
        }

        diff = mx.max(mx.abs(s14_final - full_layer_out)).item()
        print(f"Generated {c['name']} (deconstruction vs full layer max diff: {diff:.8f})")
        print(f"  Indices: {indices.tolist()}")
        results.append(case_dict)

    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fixtures_dir, "flint_layer2_golden.json")
    with open(out_path, "w") as f:
        json.dump({"generator": "generate_layer2_golden.py", "cases": results}, f, indent=2)

    print(f"Saved {len(results)} cases to {out_path} successfully!")

if __name__ == "__main__":
    main()
