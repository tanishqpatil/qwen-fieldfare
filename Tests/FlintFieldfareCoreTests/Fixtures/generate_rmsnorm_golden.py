#!/usr/bin/env python3
"""
Generates pinned reference golden data for Flint RMSNorm qualification using MLX 0.32.4.
Extracts real weights from /Users/tanishq/models/flint-mixed-3-4bit and runs mlx.fast.rms_norm.
"""

import json
import os
import mlx.core as mx
import numpy as np

def to_hex(arr):
    if arr.dtype == mx.bfloat16:
        u = arr.view(mx.uint16)
        return np.array(u).tobytes().hex()
    elif arr.dtype == mx.float32:
        return np.array(arr).tobytes().hex()
    elif arr.dtype == mx.float16:
        u = arr.view(mx.uint16)
        return np.array(u).tobytes().hex()
    else:
        return np.array(arr).tobytes().hex()

def main():
    model_dir = "/Users/tanishq/models/flint-mixed-3-4bit"
    output_path = "/Users/tanishq/dev/product/flint-fieldfare-isolated/Tests/FlintFieldfareCoreTests/Fixtures/flint_rmsnorm_golden.json"

    # Pinned versions
    info = {
        "framework": "mlx",
        "mlx_version": "0.32.4",
        "platform": "Apple Silicon M5",
        "description": "Flint RMSNorm golden test vectors for widths 2048 and 128 across BF16 and FP32"
    }

    # Load real checkpoint weights
    shard1 = mx.load(os.path.join(model_dir, "model-00001-of-00004.safetensors"))
    shard4 = mx.load(os.path.join(model_dir, "model-00004-of-00004.safetensors"))

    w_model_norm = shard4["model.norm.weight"]  # (2048,) bf16
    w_l0_in_norm = shard1["model.layers.0.input_layernorm.weight"]  # (2048,) bf16
    w_l0_q_norm = shard1["model.layers.0.self_attn.q_norm.weight"]  # (128,) bf16
    w_l0_k_norm = shard1["model.layers.0.self_attn.k_norm.weight"]  # (128,) bf16

    cases = []

    # -----------------
    # WIDTH 2048 TESTS
    # -----------------
    mx.random.seed(42)

    # 1. Normal Random vector (BF16, real model.norm.weight, eps=1e-6)
    x_rand_2048_bf16 = mx.random.normal((1, 2048)).astype(mx.bfloat16)
    y_rand_2048_bf16 = mx.fast.rms_norm(x_rand_2048_bf16, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_random_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_rand_2048_bf16),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_rand_2048_bf16)
    })

    # 2. Normal Random vector (BF16, real input_layernorm.weight, eps=1e-6)
    y_rand_l0_in_bf16 = mx.fast.rms_norm(x_rand_2048_bf16, w_l0_in_norm, 1e-6)
    cases.append({
        "name": "w2048_random_bf16_real_l0_input_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_rand_2048_bf16),
        "weight_name": "model.layers.0.input_layernorm.weight",
        "weight_hex": to_hex(w_l0_in_norm),
        "output_hex": to_hex(y_rand_l0_in_bf16)
    })

    # 3. All zeros vector (BF16, eps=1e-6)
    x_zero_2048 = mx.zeros((1, 2048), dtype=mx.bfloat16)
    y_zero_2048 = mx.fast.rms_norm(x_zero_2048, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_zeros_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_zero_2048),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_zero_2048)
    })

    # 4. Constant vector 1.0 (BF16, identity weights, eps=1e-6)
    w_ones_2048 = mx.ones((2048,), dtype=mx.bfloat16)
    x_ones_2048 = mx.ones((1, 2048), dtype=mx.bfloat16)
    y_ones_2048 = mx.fast.rms_norm(x_ones_2048, w_ones_2048, 1e-6)
    cases.append({
        "name": "w2048_constant_ones_bf16_identity_weight_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_ones_2048),
        "weight_name": "identity",
        "weight_hex": to_hex(w_ones_2048),
        "output_hex": to_hex(y_ones_2048)
    })

    # 5. Very small magnitude vector (1e-5, BF16, eps=1e-6)
    x_small_2048 = (mx.ones((1, 2048), dtype=mx.bfloat16) * 1e-5).astype(mx.bfloat16)
    y_small_2048 = mx.fast.rms_norm(x_small_2048, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_small_magnitude_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_small_2048),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_small_2048)
    })

    # 6. Very large magnitude vector (1000.0, BF16, eps=1e-6)
    x_large_2048 = (mx.ones((1, 2048), dtype=mx.bfloat16) * 1000.0).astype(mx.bfloat16)
    y_large_2048 = mx.fast.rms_norm(x_large_2048, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_large_magnitude_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_large_2048),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_large_2048)
    })

    # 7. Mixed sign vector (alternating +1.5, -1.5, BF16, eps=1e-6)
    pattern = [1.5 if i % 2 == 0 else -1.5 for i in range(2048)]
    x_mixed_2048 = mx.array([pattern], dtype=mx.bfloat16)
    y_mixed_2048 = mx.fast.rms_norm(x_mixed_2048, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_mixed_signs_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_mixed_2048),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_mixed_2048)
    })

    # 8. Multi-row batch (4 tokens, BF16, eps=1e-6)
    x_batch_2048 = mx.random.normal((4, 2048)).astype(mx.bfloat16)
    y_batch_2048 = mx.fast.rms_norm(x_batch_2048, w_model_norm, 1e-6)
    cases.append({
        "name": "w2048_batch4_random_bf16_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 4,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_batch_2048),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_batch_2048)
    })

    # 9. FP32 input and output (FP32, real model.norm.weight cast to FP32, eps=1e-6)
    x_rand_2048_f32 = mx.random.normal((1, 2048)).astype(mx.float32)
    w_norm_f32 = w_model_norm.astype(mx.float32)
    y_rand_2048_f32 = mx.fast.rms_norm(x_rand_2048_f32, w_norm_f32, 1e-6)
    cases.append({
        "name": "w2048_random_f32_real_model_norm_eps1e6",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "float32",
        "input_hex": to_hex(x_rand_2048_f32),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm), # BF16 weight representation on disk
        "output_hex": to_hex(y_rand_2048_f32)
    })

    # 10. Epsilon variation: eps=1e-5 (BF16)
    y_eps1e5_2048 = mx.fast.rms_norm(x_rand_2048_bf16, w_model_norm, 1e-5)
    cases.append({
        "name": "w2048_random_bf16_real_model_norm_eps1e5",
        "dimension": 2048,
        "rows": 1,
        "epsilon": 1e-5,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_rand_2048_bf16),
        "weight_name": "model.norm.weight",
        "weight_hex": to_hex(w_model_norm),
        "output_hex": to_hex(y_eps1e5_2048)
    })

    # ----------------
    # WIDTH 128 TESTS
    # ----------------
    # 11. Normal Random vector (BF16, real q_norm.weight, eps=1e-6)
    x_rand_128_bf16 = mx.random.normal((1, 128)).astype(mx.bfloat16)
    y_rand_q_bf16 = mx.fast.rms_norm(x_rand_128_bf16, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_random_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_rand_128_bf16),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_rand_q_bf16)
    })

    # 12. Normal Random vector (BF16, real k_norm.weight, eps=1e-6)
    y_rand_k_bf16 = mx.fast.rms_norm(x_rand_128_bf16, w_l0_k_norm, 1e-6)
    cases.append({
        "name": "w128_random_bf16_real_l0_k_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_rand_128_bf16),
        "weight_name": "model.layers.0.self_attn.k_norm.weight",
        "weight_hex": to_hex(w_l0_k_norm),
        "output_hex": to_hex(y_rand_k_bf16)
    })

    # 13. All zeros vector (BF16, eps=1e-6)
    x_zero_128 = mx.zeros((1, 128), dtype=mx.bfloat16)
    y_zero_128 = mx.fast.rms_norm(x_zero_128, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_zeros_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_zero_128),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_zero_128)
    })

    # 14. Constant vector 1.0 (BF16, identity weights, eps=1e-6)
    w_ones_128 = mx.ones((128,), dtype=mx.bfloat16)
    x_ones_128 = mx.ones((1, 128), dtype=mx.bfloat16)
    y_ones_128 = mx.fast.rms_norm(x_ones_128, w_ones_128, 1e-6)
    cases.append({
        "name": "w128_constant_ones_bf16_identity_weight_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_ones_128),
        "weight_name": "identity",
        "weight_hex": to_hex(w_ones_128),
        "output_hex": to_hex(y_ones_128)
    })

    # 15. Very small magnitude vector (1e-5, BF16, eps=1e-6)
    x_small_128 = (mx.ones((1, 128), dtype=mx.bfloat16) * 1e-5).astype(mx.bfloat16)
    y_small_128 = mx.fast.rms_norm(x_small_128, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_small_magnitude_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_small_128),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_small_128)
    })

    # 16. Very large magnitude vector (1000.0, BF16, eps=1e-6)
    x_large_128 = (mx.ones((1, 128), dtype=mx.bfloat16) * 1000.0).astype(mx.bfloat16)
    y_large_128 = mx.fast.rms_norm(x_large_128, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_large_magnitude_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_large_128),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_large_128)
    })

    # 17. Mixed sign vector (alternating +2.0, -2.0, BF16, eps=1e-6)
    pattern128 = [2.0 if i % 2 == 0 else -2.0 for i in range(128)]
    x_mixed_128 = mx.array([pattern128], dtype=mx.bfloat16)
    y_mixed_128 = mx.fast.rms_norm(x_mixed_128, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_mixed_signs_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_mixed_128),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_mixed_128)
    })

    # 18. Multi-head Q batch: 32 heads of 128 (BF16, eps=1e-6)
    x_q_batch_128 = mx.random.normal((32, 128)).astype(mx.bfloat16)
    y_q_batch_128 = mx.fast.rms_norm(x_q_batch_128, w_l0_q_norm, 1e-6)
    cases.append({
        "name": "w128_q_heads32_random_bf16_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 32,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_q_batch_128),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_q_batch_128)
    })

    # 19. Multi-head K batch: 4 heads of 128 (BF16, eps=1e-6)
    x_k_batch_128 = mx.random.normal((4, 128)).astype(mx.bfloat16)
    y_k_batch_128 = mx.fast.rms_norm(x_k_batch_128, w_l0_k_norm, 1e-6)
    cases.append({
        "name": "w128_k_heads4_random_bf16_real_l0_k_norm_eps1e6",
        "dimension": 128,
        "rows": 4,
        "epsilon": 1e-6,
        "dtype": "bfloat16",
        "input_hex": to_hex(x_k_batch_128),
        "weight_name": "model.layers.0.self_attn.k_norm.weight",
        "weight_hex": to_hex(w_l0_k_norm),
        "output_hex": to_hex(y_k_batch_128)
    })

    # 20. FP32 input and output (FP32, real q_norm.weight, eps=1e-6)
    x_rand_128_f32 = mx.random.normal((1, 128)).astype(mx.float32)
    w_q_f32 = w_l0_q_norm.astype(mx.float32)
    y_rand_128_f32 = mx.fast.rms_norm(x_rand_128_f32, w_q_f32, 1e-6)
    cases.append({
        "name": "w128_random_f32_real_l0_q_norm_eps1e6",
        "dimension": 128,
        "rows": 1,
        "epsilon": 1e-6,
        "dtype": "float32",
        "input_hex": to_hex(x_rand_128_f32),
        "weight_name": "model.layers.0.self_attn.q_norm.weight",
        "weight_hex": to_hex(w_l0_q_norm),
        "output_hex": to_hex(y_rand_128_f32)
    })

    payload = {
        "metadata": info,
        "cases": cases
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully generated {len(cases)} golden test cases to {output_path}")

if __name__ == "__main__":
    main()
