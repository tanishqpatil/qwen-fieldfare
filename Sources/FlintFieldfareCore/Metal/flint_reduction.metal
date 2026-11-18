#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint MoE Fixed Router-Order Expert Reduction Kernel
//
// Mathematical Specification:
//   - 47 MoE layers (layers 2 through 48).
//   - Hidden dimension: strictly 2048.
//   - Top-K selected routed experts: strictly 6 (slots k = 0, 1, 2, 3, 4, 5).
//   - Contributions layout: [batch, num_tokens, 6, 2048].
//   - Routing weights layout: [batch, num_tokens, 6].
//   - Optional shared expert layout: [batch, num_tokens, 2048].
//   - Output layout: [batch, num_tokens, 2048].
//
// Accumulation Contract:
//   - All contributions are stored by router slot k in [0..5], NEVER by arrival
//     order, cache residency, or physical expert ID.
//   - Accumulation is performed strictly in order k = 0, 1, 2, 3, 4, 5 in FP32:
//       acc_d = c_{0,d} * w_0 + c_{1,d} * w_1 + c_{2,d} * w_2 +
//               c_{3,d} * w_3 + c_{4,d} * w_4 + c_{5,d} * w_5
//   - If shared expert is present (has_shared_expert != 0):
//       acc_d += shared_expert_d
//   - Output cast to target dtype (BF16 or FP32).
//   - Zero data races, zero atomic operations, zero GPU heap allocations.
// ==============================================================================

/// Expert reduction for bfloat16 contributions, float32 routing weights, and bfloat16 output.
/// Grid dispatch: 3D (2048, num_tokens, batch), threadsPerThreadgroup: (256, 1, 1).
kernel void flint_expert_reduction_bf16(
    device const bfloat* contributions [[buffer(0)]], // [batch, num_tokens, 6, 2048]
    device const float* weights [[buffer(1)]],        // [batch, num_tokens, 6]
    device const bfloat* shared_expert [[buffer(2)]], // [batch, num_tokens, 2048] (optional)
    device bfloat* out [[buffer(3)]],                 // [batch, num_tokens, 2048]
    constant uint& num_tokens [[buffer(4)]],
    constant uint& has_shared_expert [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint t = pos.y;
    uint b = pos.z;
    if (d >= 2048 || t >= num_tokens) {
        return;
    }

    uint token_w_offset = (b * num_tokens + t) * 6;
    float w0 = weights[token_w_offset + 0];
    float w1 = weights[token_w_offset + 1];
    float w2 = weights[token_w_offset + 2];
    float w3 = weights[token_w_offset + 3];
    float w4 = weights[token_w_offset + 4];
    float w5 = weights[token_w_offset + 5];

    uint token_c_offset = (b * num_tokens + t) * (6 * 2048);
    device const bfloat* c0 = contributions + token_c_offset + 0 * 2048;
    device const bfloat* c1 = contributions + token_c_offset + 1 * 2048;
    device const bfloat* c2 = contributions + token_c_offset + 2 * 2048;
    device const bfloat* c3 = contributions + token_c_offset + 3 * 2048;
    device const bfloat* c4 = contributions + token_c_offset + 4 * 2048;
    device const bfloat* c5 = contributions + token_c_offset + 5 * 2048;

    // Strict fixed router-order FP32 accumulation
    float acc = float(c0[d]) * w0;
    acc += float(c1[d]) * w1;
    acc += float(c2[d]) * w2;
    acc += float(c3[d]) * w3;
    acc += float(c4[d]) * w4;
    acc += float(c5[d]) * w5;

    if (has_shared_expert != 0) {
        uint token_s_offset = (b * num_tokens + t) * 2048;
        acc += float(shared_expert[token_s_offset + d]);
    }

    uint token_out_offset = (b * num_tokens + t) * 2048;
    out[token_out_offset + d] = bfloat(acc);
}

/// Expert reduction for float32 contributions, float32 routing weights, and float32 output.
kernel void flint_expert_reduction_f32(
    device const float* contributions [[buffer(0)]], // [batch, num_tokens, 6, 2048]
    device const float* weights [[buffer(1)]],        // [batch, num_tokens, 6]
    device const float* shared_expert [[buffer(2)]], // [batch, num_tokens, 2048] (optional)
    device float* out [[buffer(3)]],                 // [batch, num_tokens, 2048]
    constant uint& num_tokens [[buffer(4)]],
    constant uint& has_shared_expert [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint t = pos.y;
    uint b = pos.z;
    if (d >= 2048 || t >= num_tokens) {
        return;
    }

    uint token_w_offset = (b * num_tokens + t) * 6;
    float w0 = weights[token_w_offset + 0];
    float w1 = weights[token_w_offset + 1];
    float w2 = weights[token_w_offset + 2];
    float w3 = weights[token_w_offset + 3];
    float w4 = weights[token_w_offset + 4];
    float w5 = weights[token_w_offset + 5];

    uint token_c_offset = (b * num_tokens + t) * (6 * 2048);
    device const float* c0 = contributions + token_c_offset + 0 * 2048;
    device const float* c1 = contributions + token_c_offset + 1 * 2048;
    device const float* c2 = contributions + token_c_offset + 2 * 2048;
    device const float* c3 = contributions + token_c_offset + 3 * 2048;
    device const float* c4 = contributions + token_c_offset + 4 * 2048;
    device const float* c5 = contributions + token_c_offset + 5 * 2048;

    float acc = c0[d] * w0;
    acc += c1[d] * w1;
    acc += c2[d] * w2;
    acc += c3[d] * w3;
    acc += c4[d] * w4;
    acc += c5[d] * w5;

    if (has_shared_expert != 0) {
        uint token_s_offset = (b * num_tokens + t) * 2048;
        acc += shared_expert[token_s_offset + d];
    }

    uint token_out_offset = (b * num_tokens + t) * 2048;
    out[token_out_offset + d] = acc;
}

/// Expert reduction for float32 contributions with bfloat16 output cast.
kernel void flint_expert_reduction_f32_in_bf16_out(
    device const float* contributions [[buffer(0)]], // [batch, num_tokens, 6, 2048]
    device const float* weights [[buffer(1)]],        // [batch, num_tokens, 6]
    device const float* shared_expert [[buffer(2)]], // [batch, num_tokens, 2048] (optional)
    device bfloat* out [[buffer(3)]],                 // [batch, num_tokens, 2048]
    constant uint& num_tokens [[buffer(4)]],
    constant uint& has_shared_expert [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint t = pos.y;
    uint b = pos.z;
    if (d >= 2048 || t >= num_tokens) {
        return;
    }

    uint token_w_offset = (b * num_tokens + t) * 6;
    float w0 = weights[token_w_offset + 0];
    float w1 = weights[token_w_offset + 1];
    float w2 = weights[token_w_offset + 2];
    float w3 = weights[token_w_offset + 3];
    float w4 = weights[token_w_offset + 4];
    float w5 = weights[token_w_offset + 5];

    uint token_c_offset = (b * num_tokens + t) * (6 * 2048);
    device const float* c0 = contributions + token_c_offset + 0 * 2048;
    device const float* c1 = contributions + token_c_offset + 1 * 2048;
    device const float* c2 = contributions + token_c_offset + 2 * 2048;
    device const float* c3 = contributions + token_c_offset + 3 * 2048;
    device const float* c4 = contributions + token_c_offset + 4 * 2048;
    device const float* c5 = contributions + token_c_offset + 5 * 2048;

    float acc = c0[d] * w0;
    acc += c1[d] * w1;
    acc += c2[d] * w2;
    acc += c3[d] * w3;
    acc += c4[d] * w4;
    acc += c5[d] * w5;

    if (has_shared_expert != 0) {
        uint token_s_offset = (b * num_tokens + t) * 2048;
        acc += shared_expert[token_s_offset + d];
    }

    uint token_out_offset = (b * num_tokens + t) * 2048;
    out[token_out_offset + d] = bfloat(acc);
}
