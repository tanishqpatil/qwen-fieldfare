#!/usr/bin/env python3
"""
Generates pinned reference golden test vectors for Flint Attention (GQA, Sinks, RoPE, SDPA)
using MLX 0.32.4 and mlx_vlm 0.7.12 on Apple Silicon.

Output: flint_attention_golden.json
"""

import json
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import create_causal_mask

def to_hex(arr: mx.array) -> str:
    return bytes(arr.view(mx.uint8)).hex()

def main():
    mx.random.seed(42)
    np.random.seed(42)

    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    shard1 = os.path.join(model_dir, "model-00001-of-00004.safetensors")
    weights = mx.load(shard1)

    l0_sinks_bf16 = weights["model.layers.0.self_attn.softmax_offset"]
    l2_sinks_bf16 = weights["model.layers.2.self_attn.softmax_offset"]

    l0_sinks_f32 = l0_sinks_bf16.astype(mx.float32)
    l2_sinks_f32 = l2_sinks_bf16.astype(mx.float32)

    cases = []
    scale = float(1.0 / (128.0 ** 0.5))
    rope = nn.RoPE(dims=128, traditional=False, base=10000.0)

    # =========================================================================
    # 1. RoPE Isolated Cases (Split-Half, traditional=False)
    # =========================================================================
    for num_heads in [32, 4]:
        for seq_len in [1, 8]:
            for offset in [0, 5, 599]:
                q = mx.random.normal(shape=(1, num_heads, seq_len, 128)).astype(mx.float32)
                q_rot_f32 = rope(q, offset=offset)
                q_bf16 = q.astype(mx.bfloat16)
                q_rot_bf16 = rope(q_bf16, offset=offset)
                mx.eval(q, q_rot_f32, q_bf16, q_rot_bf16)

                case_name_f32 = f"rope_h{num_heads}_t{seq_len}_off{offset}_f32"
                cases.append({
                    "kind": "rope",
                    "name": case_name_f32,
                    "num_heads": num_heads,
                    "seq_len": seq_len,
                    "offset": offset,
                    "dtype": "float32",
                    "input_hex": to_hex(q),
                    "output_hex": to_hex(q_rot_f32)
                })

                case_name_bf16 = f"rope_h{num_heads}_t{seq_len}_off{offset}_bf16"
                cases.append({
                    "kind": "rope",
                    "name": case_name_bf16,
                    "num_heads": num_heads,
                    "seq_len": seq_len,
                    "offset": offset,
                    "dtype": "bfloat16",
                    "input_hex": to_hex(q_bf16),
                    "output_hex": to_hex(q_rot_bf16)
                })

    # =========================================================================
    # 2. Full-Attention (Layer 0, No RoPE, Causal Masking, Real Sinks)
    # =========================================================================
    # Case A: Prefill (T_q=4, T_kv=4)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l0_sinks_f32), ("bfloat16", mx.bfloat16, l0_sinks_bf16)]:
        T_q = 4
        q = mx.random.normal(shape=(1, 32, T_q, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_q, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_q, 128)).astype(dtype_mx)

        # Full causal mask
        mask = create_causal_mask(T_q, offset=0)
        additive_mask = mx.where(mask, 0.0, -1e9).astype(dtype_mx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_prefill",
            "name": f"l0_full_prefill_t4_{dtype_str}",
            "is_sliding": False,
            "layer_idx": 0,
            "T_q": T_q,
            "T_kv": T_q,
            "cache_offset": 0,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # Case B: Decode (T_q=1, T_kv=8)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l0_sinks_f32), ("bfloat16", mx.bfloat16, l0_sinks_bf16)]:
        T_kv = 8
        q = mx.random.normal(shape=(1, 32, 1, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)

        # In full attention decode, all cached keys are attended
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_decode",
            "name": f"l0_full_decode_tkv8_{dtype_str}",
            "is_sliding": False,
            "layer_idx": 0,
            "T_kv": T_kv,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # =========================================================================
    # 3. Sliding-Attention (Layer 2, Window 513, Real Sinks)
    # =========================================================================
    # Case A: Prefill (T_q=8, T_kv=8, sliding window)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l2_sinks_f32), ("bfloat16", mx.bfloat16, l2_sinks_bf16)]:
        T_q = 8
        q = mx.random.normal(shape=(1, 32, T_q, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_q, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_q, 128)).astype(dtype_mx)

        mask = create_causal_mask(T_q, offset=0, window_size=513)
        additive_mask = mx.where(mask, 0.0, -1e9).astype(dtype_mx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_prefill",
            "name": f"l2_sliding_prefill_t8_{dtype_str}",
            "is_sliding": True,
            "layer_idx": 2,
            "T_q": T_q,
            "T_kv": T_q,
            "cache_offset": 0,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # Case B: Single-token decode beyond sliding boundary (T_kv = 520 > 513)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l2_sinks_f32), ("bfloat16", mx.bfloat16, l2_sinks_bf16)]:
        T_kv = 520
        q = mx.random.normal(shape=(1, 32, 1, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)

        # In sliding window of size 513 with total cached 520 keys:
        # Attends to keys from index (520 - 513) = 7 up to 519
        mask = mx.arange(T_kv) >= (T_kv - 513)
        additive_mask = mx.where(mask, 0.0, -1e9)[None, None, None, :].astype(dtype_mx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_decode",
            "name": f"l2_sliding_decode_tkv520_{dtype_str}",
            "is_sliding": True,
            "layer_idx": 2,
            "T_kv": T_kv,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # =========================================================================
    # 4. Bounded-KV Evicted Cache Cases (retained_kv_start > 0)
    # =========================================================================
    # Case A: Reproduce exact Review #007 scenario (pos 1024, retained interval 512..1024, phys 0..512)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l2_sinks_f32), ("bfloat16", mx.bfloat16, l2_sinks_bf16)]:
        T_kv = 513
        query_abs_pos = 1024
        retained_kv_start = 512

        q = mx.random.normal(shape=(1, 32, 1, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)

        # In sliding window of 513 for pos 1024: attended window is [1024 - 512, 1024] = [512, 1024].
        # Physical buffer holds exactly positions [512, 1024], so all 513 physical keys are attended.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_decode_bounded",
            "name": f"l2_sliding_decode_evicted_pos1024_{dtype_str}",
            "is_sliding": True,
            "layer_idx": 2,
            "T_kv": T_kv,
            "query_abs_pos": query_abs_pos,
            "retained_kv_start": retained_kv_start,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # Case B: Partial intersection with retained buffer (retained 450..1049, pos 1024, attended 512..1024 -> phys 62..574)
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l2_sinks_f32), ("bfloat16", mx.bfloat16, l2_sinks_bf16)]:
        T_kv = 600
        query_abs_pos = 1024
        retained_kv_start = 450

        q = mx.random.normal(shape=(1, 32, 1, 128)).astype(dtype_mx)
        k = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)

        # Attended absolute: [512, 1024] -> physical: [512 - 450, 1024 - 450] = [62, 574] (indices 62..574 inclusive)
        indices = mx.arange(T_kv)
        mask = (indices >= 62) & (indices <= 574)
        additive_mask = mx.where(mask, 0.0, -1e9)[None, None, None, :].astype(dtype_mx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q, k, v, out)

        cases.append({
            "kind": "attention_decode_bounded",
            "name": f"l2_sliding_decode_evicted_partial_pos1024_{dtype_str}",
            "is_sliding": True,
            "layer_idx": 2,
            "T_kv": T_kv,
            "query_abs_pos": query_abs_pos,
            "retained_kv_start": retained_kv_start,
            "dtype": dtype_str,
            "q_hex": to_hex(q),
            "k_hex": to_hex(k),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    # =========================================================================
    # 5. Combined RoPE + Attention Qualification (beyond 513 tokens)
    # =========================================================================
    for dtype_str, dtype_mx, sinks_mx in [("float32", mx.float32, l2_sinks_f32), ("bfloat16", mx.bfloat16, l2_sinks_bf16)]:
        pos = 600
        retained_start = 88 # 600 - 512 = 88
        T_kv = 513

        q_raw = mx.random.normal(shape=(1, 32, 1, 128)).astype(dtype_mx)
        k_raw = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)
        v = mx.random.normal(shape=(1, 4, T_kv, 128)).astype(dtype_mx)

        # Apply split-half RoPE to Q at pos 600 and K at positions 88..600
        q_rot = rope(q_raw, offset=pos)
        k_rot = rope(k_raw, offset=retained_start)

        out = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale, mask=None, sinks=sinks_mx, stream=mx.cpu)
        mx.eval(q_raw, k_raw, q_rot, k_rot, v, out)

        cases.append({
            "kind": "combined_rope_attention",
            "name": f"combined_rope_attn_pos600_{dtype_str}",
            "is_sliding": True,
            "layer_idx": 2,
            "T_kv": T_kv,
            "query_abs_pos": pos,
            "retained_kv_start": retained_start,
            "dtype": dtype_str,
            "q_raw_hex": to_hex(q_raw),
            "k_raw_hex": to_hex(k_raw),
            "q_rot_hex": to_hex(q_rot),
            "k_rot_hex": to_hex(k_rot),
            "v_hex": to_hex(v),
            "sinks_hex": to_hex(sinks_mx),
            "out_hex": to_hex(out)
        })

    payload = {
        "metadata": {
            "source": "MLX 0.32.4 / mlx_vlm 0.7.12",
            "model": "Recursive Flint 40B-A2.5B",
            "head_dim": 128,
            "num_q_heads": 32,
            "num_kv_heads": 4,
            "sliding_window": 513,
            "rope_theta": 10000.0
        },
        "cases": cases
    }

    out_path = os.path.join(os.path.dirname(__file__), "flint_attention_golden.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(cases)} golden cases in {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
