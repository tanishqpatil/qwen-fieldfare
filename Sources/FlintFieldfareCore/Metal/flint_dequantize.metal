#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Native 3-bit and 4-bit Metal Dequantization Kernels
//
// Mathematical Specification:
//   - Group size: strictly 64 elements per group across all Flint projections.
//   - Scales & Biases: stored as resident bfloat16 tensors [rows, num_groups].
//   - Quantized affine formula:
//       weight = scale * int_val + bias
//   - Dequantization arithmetic performed in FP32, then cast to target dtype (BF16 / FP32).
//
// 4-bit Packing Contract:
//   - 8 values packed per 32-bit uint32 word (64 elements = 8 words per group).
//   - Unpacked value j in group:
//       word_idx = j / 8
//       bit_offset = (j % 8) * 4
//       int_val = (group_words[word_idx] >> bit_offset) & 0xF
//
// 3-bit Packing Contract:
//   - Continuous bit-stream packed across 32-bit word boundaries (64 elements * 3 bits = 192 bits = 6 words per group).
//   - Unpacked value j in group:
//       bit_idx = j * 3
//       w0 = bit_idx / 32
//       b0 = bit_idx % 32
//       combined = (ulong(w_hi) << 32) | ulong(w_lo)
//       int_val = (combined >> b0) & 0x7
// ==============================================================================

/// 4-bit affine dequantization with output to bfloat16.
/// Grid dispatch: 2D (cols, rows), threadsPerThreadgroup: (min(cols, 256), 1, 1).
kernel void flint_dequantize_4bit_bf16(
    device const uint* packed_weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    if (col >= cols || row >= rows) {
        return;
    }

    uint num_groups = cols / 64;
    uint group_idx = col / 64;
    uint j = col % 64;

    bfloat scale = scales[row * num_groups + group_idx];
    bfloat bias = biases[row * num_groups + group_idx];

    uint word_idx = j / 8;
    uint bit_shift = (j % 8) * 4;

    uint packed_word = packed_weights[row * (num_groups * 8) + group_idx * 8 + word_idx];
    uint int_val = (packed_word >> bit_shift) & 0xFu;

    float val = float(scale) * float(int_val) + float(bias);
    out[row * cols + col] = bfloat(val);
}

/// 4-bit affine dequantization with output to float32.
kernel void flint_dequantize_4bit_f32(
    device const uint* packed_weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    if (col >= cols || row >= rows) {
        return;
    }

    uint num_groups = cols / 64;
    uint group_idx = col / 64;
    uint j = col % 64;

    bfloat scale = scales[row * num_groups + group_idx];
    bfloat bias = biases[row * num_groups + group_idx];

    uint word_idx = j / 8;
    uint bit_shift = (j % 8) * 4;

    uint packed_word = packed_weights[row * (num_groups * 8) + group_idx * 8 + word_idx];
    uint int_val = (packed_word >> bit_shift) & 0xFu;

    float val = float(scale) * float(int_val) + float(bias);
    out[row * cols + col] = val;
}

/// 3-bit affine dequantization with output to bfloat16.
kernel void flint_dequantize_3bit_bf16(
    device const uint* packed_weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    if (col >= cols || row >= rows) {
        return;
    }

    uint num_groups = cols / 64;
    uint group_idx = col / 64;
    uint j = col % 64;

    bfloat scale = scales[row * num_groups + group_idx];
    bfloat bias = biases[row * num_groups + group_idx];

    uint bit_idx = j * 3;
    uint w0 = bit_idx / 32;
    uint b0 = bit_idx % 32;

    device const uint* group_words = packed_weights + row * (num_groups * 6) + group_idx * 6;
    uint w_lo = group_words[w0];
    uint w_hi = (w0 + 1 < 6) ? group_words[w0 + 1] : 0u;
    ulong combined = (ulong(w_hi) << 32) | ulong(w_lo);
    uint int_val = uint((combined >> b0) & 0x7ul);

    float val = float(scale) * float(int_val) + float(bias);
    out[row * cols + col] = bfloat(val);
}

/// 3-bit affine dequantization with output to float32.
kernel void flint_dequantize_3bit_f32(
    device const uint* packed_weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    if (col >= cols || row >= rows) {
        return;
    }

    uint num_groups = cols / 64;
    uint group_idx = col / 64;
    uint j = col % 64;

    bfloat scale = scales[row * num_groups + group_idx];
    bfloat bias = biases[row * num_groups + group_idx];

    uint bit_idx = j * 3;
    uint w0 = bit_idx / 32;
    uint b0 = bit_idx % 32;

    device const uint* group_words = packed_weights + row * (num_groups * 6) + group_idx * 6;
    uint w_lo = group_words[w0];
    uint w_hi = (w0 + 1 < 6) ? group_words[w0 + 1] : 0u;
    ulong combined = (ulong(w_hi) << 32) | ulong(w_lo);
    uint int_val = uint((combined >> b0) & 0x7ul);

    float val = float(scale) * float(int_val) + float(bias);
    out[row * cols + col] = val;
}
