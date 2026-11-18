#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Native Dense and Shared Expert SwiGLU MLP Metal Compute Kernels
//
// Mathematical Specification:
//   - Dense MLP (Layers 0, 1): hidden_dim = 2048, intermediate_dim = 6144
//   - Shared Expert MLP (Layers 2..48): hidden_dim = 2048, intermediate_dim = 1024
//   - gate = gate_proj(x)       [batch, num_tokens, intermediate_dim]
//   - up = up_proj(x)           [batch, num_tokens, intermediate_dim]
//   - activated = SiLU(gate) * up
//   - out = down_proj(activated) [batch, num_tokens, hidden_dim]
//
// Accumulation Contract:
//   - Matrix-vector projections evaluate with FP32 SIMD-coalesced reduction.
//   - SiLU evaluates in FP32: silu(g) = g / (1.0f + exp(-g)).
//   - Product silu(g) * u evaluates in FP32 prior to casting.
//   - Zero data races, zero atomic operations, zero GPU heap allocations.
// ==============================================================================

// MARK: - Matrix-Vector Projections (GEMM: Y = X * W^T)

/// GEMM for bfloat16 activations and bfloat16 weights -> bfloat16 output.
/// Dispatch: 3D grid (M * 32, num_tokens, batch), threadsPerThreadgroup: (256, 1, 1).
/// Each SIMD group of 32 threads computes one output scalar Y[b, t, m] with 100% coalesced memory loads.
kernel void flint_gemm_bf16(
    device const bfloat* in_x [[buffer(0)]],         // [batch, num_tokens, K]
    device const bfloat* weight [[buffer(1)]],       // [M, K]
    device bfloat* out_y [[buffer(2)]],              // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const bfloat* x_vec = in_x + (b * num_tokens + t) * K;
    device const bfloat* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += float(x_vec[k]) * float(w_row[k]);
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = bfloat(total);
    }
}

/// GEMM for float32 activations and float32 weights -> float32 output.
kernel void flint_gemm_f32(
    device const float* in_x [[buffer(0)]],          // [batch, num_tokens, K]
    device const float* weight [[buffer(1)]],        // [M, K]
    device float* out_y [[buffer(2)]],               // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const float* x_vec = in_x + (b * num_tokens + t) * K;
    device const float* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += x_vec[k] * w_row[k];
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = total;
    }
}

/// GEMM for float32 activations and float32 weights -> bfloat16 output cast.
kernel void flint_gemm_f32_in_bf16_out(
    device const float* in_x [[buffer(0)]],          // [batch, num_tokens, K]
    device const float* weight [[buffer(1)]],        // [M, K]
    device bfloat* out_y [[buffer(2)]],              // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const float* x_vec = in_x + (b * num_tokens + t) * K;
    device const float* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += x_vec[k] * w_row[k];
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = bfloat(total);
    }
}

// MARK: - Pointwise Activations & Operations

/// SiLU activation: y = x / (1.0 + exp(-x))
kernel void flint_silu_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device bfloat* out_y [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float x = float(in_x[pos]);
    float silu_x = x / (1.0f + exp(-x));
    out_y[pos] = bfloat(silu_x);
}

kernel void flint_silu_f32(
    device const float* in_x [[buffer(0)]],
    device float* out_y [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float x = in_x[pos];
    float silu_x = x / (1.0f + exp(-x));
    out_y[pos] = silu_x;
}

/// Elementwise multiplication: y = a * b
kernel void flint_elementwise_mul_bf16(
    device const bfloat* in_a [[buffer(0)]],
    device const bfloat* in_b [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float a = float(in_a[pos]);
    float b = float(in_b[pos]);
    out_y[pos] = bfloat(a * b);
}

kernel void flint_elementwise_mul_f32(
    device const float* in_a [[buffer(0)]],
    device const float* in_b [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    out_y[pos] = in_a[pos] * in_b[pos];
}

/// Fused SwiGLU activation: y = SiLU(gate) * up
kernel void flint_swiglu_bf16(
    device const bfloat* in_gate [[buffer(0)]],
    device const bfloat* in_up [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float g = float(in_gate[pos]);
    float u = float(in_up[pos]);
    float silu_g = g / (1.0f + exp(-g));
    out_y[pos] = bfloat(silu_g * u);
}

kernel void flint_swiglu_f32(
    device const float* in_gate [[buffer(0)]],
    device const float* in_up [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float g = in_gate[pos];
    float u = in_up[pos];
    float silu_g = g / (1.0f + exp(-g));
    out_y[pos] = silu_g * u;
}
