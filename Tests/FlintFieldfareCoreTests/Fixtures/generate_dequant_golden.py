#!/usr/bin/env python3
"""
Generates pinned reference golden test vectors for Flint 3-bit and 4-bit affine dequantization
using MLX 0.32.4 (mx.dequantize) on Apple Silicon.

Output: flint_dequant_golden.json
"""

import json
import os
import numpy as np
import mlx.core as mx

def to_hex_bytes(b: bytes) -> str:
    return b.hex()

def to_hex_mx(arr: mx.array) -> str:
    return bytes(arr.view(mx.uint8)).hex()

def to_hex_np(arr: np.ndarray) -> str:
    return arr.tobytes().hex()

def main():
    cases = []

    # -------------------------------------------------------------
    # 1. Synthetic 4-bit cases (rows = 2, cols = 64)
    # -------------------------------------------------------------
    # Group size = 64. 1 group per row = 8 words per row.
    # Row 0: alternating 0x01234567 and 0x89abcdef
    # Row 1: all 0x00000000 and 0xffffffff
    w4_words = np.array([
        [0x76543210, 0xfedcba98, 0x76543210, 0xfedcba98, 0x76543210, 0xfedcba98, 0x76543210, 0xfedcba98],
        [0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff]
    ], dtype=np.uint32)

    # Scales and biases in BF16
    s4_np = np.array([[1.5], [0.5]], dtype=np.float32)
    b4_np = np.array([[-2.0], [1.0]], dtype=np.float32)

    w4_mx = mx.array(w4_words)
    s4_mx = mx.array(s4_np).astype(mx.bfloat16)
    b4_mx = mx.array(b4_np).astype(mx.bfloat16)

    # 4-bit -> BF16 output
    deq4_bf16 = mx.dequantize(w4_mx, s4_mx, b4_mx, group_size=64, bits=4)
    cases.append({
        "name": "synthetic_4bit_bf16",
        "bits": 4,
        "rows": 2,
        "cols": 64,
        "out_dtype": "bfloat16",
        "packed_hex": to_hex_np(w4_words),
        "scales_hex": to_hex_mx(s4_mx),
        "biases_hex": to_hex_mx(b4_mx),
        "expected_hex": to_hex_mx(deq4_bf16)
    })

    # 4-bit -> FP32 output
    deq4_f32 = deq4_bf16.astype(mx.float32)
    cases.append({
        "name": "synthetic_4bit_f32",
        "bits": 4,
        "rows": 2,
        "cols": 64,
        "out_dtype": "float32",
        "packed_hex": to_hex_np(w4_words),
        "scales_hex": to_hex_mx(s4_mx),
        "biases_hex": to_hex_mx(b4_mx),
        "expected_hex": to_hex_np(np.array(deq4_f32))
    })

    # -------------------------------------------------------------
    # 2. Synthetic 3-bit cases (rows = 2, cols = 64)
    # -------------------------------------------------------------
    # Group size = 64. 1 group per row = 6 words per row.
    # Test boundary-crossing values:
    # Word 0: (1 << 30) | (1 << 31) -> element 10 takes 2 bits from word 0
    # Word 1: (1 << 0)               -> element 10 takes 1 bit from word 1
    # Element 10 = (1) | (1<<1) | (1<<2) = 7!
    # Word 1: (1 << 31)              -> element 21 takes 1 bit from word 1
    # Word 2: (3 << 0)               -> element 21 takes 2 bits from word 2
    # Element 21 = (1) | (1<<1) | (1<<2) = 7!
    w3_row0 = np.array([
        (1 << 30) | (1 << 31) | 0x12492492,
        (1 << 0) | (1 << 31) | 0x24924920,
        (3 << 0) | 0x92492490,
        0x49249249,
        0x24924924,
        0x12492492
    ], dtype=np.uint32)

    # Row 1: alternating all zeros and all ones
    w3_row1 = np.array([
        0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff
    ], dtype=np.uint32)

    w3_words = np.vstack([w3_row0, w3_row1])

    s3_np = np.array([[2.0], [0.25]], dtype=np.float32)
    b3_np = np.array([[-1.5], [0.75]], dtype=np.float32)

    w3_mx = mx.array(w3_words)
    s3_mx = mx.array(s3_np).astype(mx.bfloat16)
    b3_mx = mx.array(b3_np).astype(mx.bfloat16)

    # 3-bit -> BF16 output
    deq3_bf16 = mx.dequantize(w3_mx, s3_mx, b3_mx, group_size=64, bits=3)
    cases.append({
        "name": "synthetic_3bit_bf16",
        "bits": 3,
        "rows": 2,
        "cols": 64,
        "out_dtype": "bfloat16",
        "packed_hex": to_hex_np(w3_words),
        "scales_hex": to_hex_mx(s3_mx),
        "biases_hex": to_hex_mx(b3_mx),
        "expected_hex": to_hex_mx(deq3_bf16)
    })

    # 3-bit -> FP32 output
    deq3_f32 = deq3_bf16.astype(mx.float32)
    cases.append({
        "name": "synthetic_3bit_f32",
        "bits": 3,
        "rows": 2,
        "cols": 64,
        "out_dtype": "float32",
        "packed_hex": to_hex_np(w3_words),
        "scales_hex": to_hex_mx(s3_mx),
        "biases_hex": to_hex_mx(b3_mx),
        "expected_hex": to_hex_np(np.array(deq3_f32))
    })

    # -------------------------------------------------------------
    # 3. Checkpoint Real Tensors: Layer 2 Shared Expert (4-bit INT4)
    # -------------------------------------------------------------
    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    shard2 = mx.load(os.path.join(model_dir, "model-00002-of-00004.safetensors"))
    shard4 = mx.load(os.path.join(model_dir, "model-00004-of-00004.safetensors"))

    # Shared Expert gate_proj slice (rows=4, cols=2048) -> 32 groups
    w_se_gate = shard1["model.layers.2.mlp.shared_expert.gate_proj.weight"][:4]
    s_se_gate = shard1["model.layers.2.mlp.shared_expert.gate_proj.scales"][:4]
    b_se_gate = shard1["model.layers.2.mlp.shared_expert.gate_proj.biases"][:4]
    deq_se_gate = mx.dequantize(w_se_gate, s_se_gate, b_se_gate, group_size=64, bits=4)

    cases.append({
        "name": "layer_2_shared_expert_gate_proj_4bit_bf16",
        "bits": 4,
        "rows": 4,
        "cols": 2048,
        "out_dtype": "bfloat16",
        "packed_hex": to_hex_mx(w_se_gate),
        "scales_hex": to_hex_mx(s_se_gate),
        "biases_hex": to_hex_mx(b_se_gate),
        "expected_hex": to_hex_mx(deq_se_gate)
    })

    # Shared Expert down_proj slice (rows=4, cols=1024) -> 16 groups
    w_se_down = shard1["model.layers.2.mlp.shared_expert.down_proj.weight"][:4]
    s_se_down = shard1["model.layers.2.mlp.shared_expert.down_proj.scales"][:4]
    b_se_down = shard1["model.layers.2.mlp.shared_expert.down_proj.biases"][:4]
    deq_se_down = mx.dequantize(w_se_down, s_se_down, b_se_down, group_size=64, bits=4)

    cases.append({
        "name": "layer_2_shared_expert_down_proj_4bit_bf16",
        "bits": 4,
        "rows": 4,
        "cols": 1024,
        "out_dtype": "bfloat16",
        "packed_hex": to_hex_mx(w_se_down),
        "scales_hex": to_hex_mx(s_se_down),
        "biases_hex": to_hex_mx(b_se_down),
        "expected_hex": to_hex_mx(deq_se_down)
    })

    # -------------------------------------------------------------
    # 4. Checkpoint Real Tensors: Routed Experts (3-bit)
    #    First expert (0), middle expert (128), last expert (255)
    #    Layers 2, 24, 48
    # -------------------------------------------------------------
    expert_configs = [
        (2, shard1, 0, "first"),
        (24, shard2, 128, "middle"),
        (48, shard4, 255, "last")
    ]

    for layer_idx, shard, expert_id, expert_tag in expert_configs:
        # gate_proj (rows=4, cols=2048)
        w_gate = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.weight"][expert_id, :4]
        s_gate = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.scales"][expert_id, :4]
        b_gate = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.biases"][expert_id, :4]
        deq_gate = mx.dequantize(w_gate, s_gate, b_gate, group_size=64, bits=3)

        cases.append({
            "name": f"layer_{layer_idx}_expert_{expert_id}_{expert_tag}_gate_proj_3bit_bf16",
            "bits": 3,
            "rows": 4,
            "cols": 2048,
            "out_dtype": "bfloat16",
            "packed_hex": to_hex_mx(w_gate),
            "scales_hex": to_hex_mx(s_gate),
            "biases_hex": to_hex_mx(b_gate),
            "expected_hex": to_hex_mx(deq_gate)
        })

        # up_proj (rows=4, cols=2048)
        w_up = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.up_proj.weight"][expert_id, :4]
        s_up = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.up_proj.scales"][expert_id, :4]
        b_up = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.up_proj.biases"][expert_id, :4]
        deq_up = mx.dequantize(w_up, s_up, b_up, group_size=64, bits=3)

        cases.append({
            "name": f"layer_{layer_idx}_expert_{expert_id}_{expert_tag}_up_proj_3bit_bf16",
            "bits": 3,
            "rows": 4,
            "cols": 2048,
            "out_dtype": "bfloat16",
            "packed_hex": to_hex_mx(w_up),
            "scales_hex": to_hex_mx(s_up),
            "biases_hex": to_hex_mx(b_up),
            "expected_hex": to_hex_mx(deq_up)
        })

        # down_proj (rows=4, cols=512)
        w_down = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.weight"][expert_id, :4]
        s_down = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.scales"][expert_id, :4]
        b_down = shard[f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.biases"][expert_id, :4]
        deq_down = mx.dequantize(w_down, s_down, b_down, group_size=64, bits=3)

        cases.append({
            "name": f"layer_{layer_idx}_expert_{expert_id}_{expert_tag}_down_proj_3bit_bf16",
            "bits": 3,
            "rows": 4,
            "cols": 512,
            "out_dtype": "bfloat16",
            "packed_hex": to_hex_mx(w_down),
            "scales_hex": to_hex_mx(s_down),
            "biases_hex": to_hex_mx(b_down),
            "expected_hex": to_hex_mx(deq_down)
        })

        # Also add FP32 output variant for down_proj
        deq_down_f32 = deq_down.astype(mx.float32)
        cases.append({
            "name": f"layer_{layer_idx}_expert_{expert_id}_{expert_tag}_down_proj_3bit_f32",
            "bits": 3,
            "rows": 4,
            "cols": 512,
            "out_dtype": "float32",
            "packed_hex": to_hex_mx(w_down),
            "scales_hex": to_hex_mx(s_down),
            "biases_hex": to_hex_mx(b_down),
            "expected_hex": to_hex_np(np.array(deq_down_f32))
        })

    payload = {
        "metadata": {
            "source": "MLX 0.32.4 (mx.dequantize)",
            "model": "Recursive Flint 40B-A2.5B",
            "group_size": 64,
            "bits_supported": [3, 4],
            "dtypes_supported": ["bfloat16", "float32"]
        },
        "cases": cases
    }

    out_path = os.path.join(os.path.dirname(__file__), "flint_dequant_golden.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(cases)} dequantization golden cases in {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
