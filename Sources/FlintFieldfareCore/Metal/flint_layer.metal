#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Decoder Layer Metal Compute Kernels (F3-01)
//
// Operations:
//   - Attention head transpositions: [batch, T, H, D] <-> [batch, H, T, D]
//   - Residual additions: y = a + b
// ==============================================================================

/// Transposes [batch, T, H, D] -> [batch, H, T, D] for bfloat16.
/// Grid dispatch: (D, T, batch * H)
kernel void flint_transpose_0213_bf16(
    device const bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint t = pos.y;
    uint bh = pos.z;
    uint b = bh / H;
    uint h = bh % H;

    if (d >= D || t >= T || h >= H || b >= batch) return;

    uint in_idx = (b * T + t) * (H * D) + h * D + d;
    uint out_idx = (b * H + h) * (T * D) + t * D + d;
    out[out_idx] = in[in_idx];
}

/// Transposes [batch, H, T, D] -> [batch, T, H, D] for bfloat16.
/// Grid dispatch: (D, H, batch * T)
kernel void flint_transpose_0213_back_bf16(
    device const bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint h = pos.y;
    uint bt = pos.z;
    uint b = bt / T;
    uint t = bt % T;

    if (d >= D || h >= H || t >= T || b >= batch) return;

    uint in_idx = (b * H + h) * (T * D) + t * D + d;
    uint out_idx = (b * T + t) * (H * D) + h * D + d;
    out[out_idx] = in[in_idx];
}

/// Elementwise residual addition: y = a + b
kernel void flint_residual_add_bf16(
    device const bfloat* in_a [[buffer(0)]],
    device const bfloat* in_b [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    out_y[pos] = bfloat(float(in_a[pos]) + float(in_b[pos]));
}
