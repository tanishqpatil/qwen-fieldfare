#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint RoPE (Rotary Position Embedding) Metal Compute Kernels
//
// Mathematical Specification (Split-Half Permutation, traditional=False):
//   For head dimension D = 128 (64 dimension pairs d in [0, 63]):
//     freq_d = 1.0 / (base ** (2.0 * d / D)) = base ** (-d / 64.0)
//     pos = offset + t
//     theta = pos * freq_d
//     x1 = x[t, h, d]
//     x2 = x[t, h, d + 64]
//     y1 = x1 * cos(theta) - x2 * sin(theta)
//     y2 = x2 * cos(theta) + x1 * sin(theta)
//
// Invariants:
// - Applied ONLY to sliding-attention layers (bypassed on 14 full-attention layers).
// - Base theta is 10000.0f.
// - Dimension D is strictly 128.
// - Supports in-place (y == x) or out-of-place execution.
// - Trigonometric functions evaluate in full FP32 (float) for numerical stability.
// ==============================================================================

/// Split-half RoPE for bfloat16 activations.
/// Grid dispatch: threadsPerThreadgroup = (64, 1, 1), threadgroups = (seq_len, num_heads, batch).
kernel void flint_rope_split_half_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device bfloat* out_y [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant float& base [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]] // (t, h, b)
) {
    uint d = tid.x; // 0 .. 63
    uint t = pos_in_grid.x;
    uint h = pos_in_grid.y;
    uint b = pos_in_grid.z;

    if (t >= seq_len || h >= num_heads) {
        return;
    }

    // Input layout: [batch, num_heads, seq_len, 128]
    // Strides:
    //   batch_stride = num_heads * seq_len * 128
    //   head_stride  = seq_len * 128
    //   time_stride  = 128
    uint row_offset = b * (num_heads * seq_len * 128) + h * (seq_len * 128) + t * 128;

    device const bfloat* x_ptr = in_x + row_offset;
    device bfloat* y_ptr = out_y + row_offset;

    // Angular frequency: freq = base ** (-d / 64.0)
    float exponent = -float(d) / 64.0f;
    float freq = pow(base, exponent);
    float position = float(offset + t);
    float theta = position * freq;

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x1 = float(x_ptr[d]);
    float x2 = float(x_ptr[d + 64]);

    float y1 = x1 * cos_theta - x2 * sin_theta;
    float y2 = x2 * cos_theta + x1 * sin_theta;

    y_ptr[d] = bfloat(y1);
    y_ptr[d + 64] = bfloat(y2);
}

/// Split-half RoPE for float32 activations.
kernel void flint_rope_split_half_f32(
    device const float* in_x [[buffer(0)]],
    device float* out_y [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant float& base [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint d = tid.x;
    uint t = pos_in_grid.x;
    uint h = pos_in_grid.y;
    uint b = pos_in_grid.z;

    if (t >= seq_len || h >= num_heads) {
        return;
    }

    uint row_offset = b * (num_heads * seq_len * 128) + h * (seq_len * 128) + t * 128;

    device const float* x_ptr = in_x + row_offset;
    device float* y_ptr = out_y + row_offset;

    float exponent = -float(d) / 64.0f;
    float freq = pow(base, exponent);
    float position = float(offset + t);
    float theta = position * freq;

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x1 = x_ptr[d];
    float x2 = x_ptr[d + 64];

    float y1 = x1 * cos_theta - x2 * sin_theta;
    float y2 = x2 * cos_theta + x1 * sin_theta;

    y_ptr[d] = y1;
    y_ptr[d + 64] = y2;
}
