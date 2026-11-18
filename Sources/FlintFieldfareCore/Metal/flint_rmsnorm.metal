#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint RMSNorm Metal Compute Kernels
//
// Mathematical Specification:
//   y = x * rsqrt(mean(x * x, axis=-1) + epsilon) * weight
//
// Invariants:
// - Weights are stored in bfloat16 representation matching checkpoint storage.
// - All squared accumulations, means, epsilon additions, and scale computations
//   evaluate in full FP32 (single precision) to prevent underflow/overflow.
// - Reductions utilize SIMD shuffle operations (`simd_sum`) and threadgroup memory.
// - Zero memory allocations occur during kernel execution.
// - Weights are strictly read-only (`device const bfloat*`).
// ==============================================================================

// MARK: - Hidden Dimension 2048 (Input, Post-Attention, and Final Norms)

/// RMSNorm for hidden dimension 2048 with bfloat16 input and output.
/// Dispatch configuration: 256 threads per threadgroup, (rows, 1, 1) threadgroups.
kernel void flint_rmsnorm_2048_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[8];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    device const bfloat* x_row = in_x + row * 2048;
    device bfloat* y_row = out_y + row * 2048;

    // Strided accumulation in FP32 across 8 elements per thread (256 * 8 = 2048)
    float local_sq = 0.0f;
    for (uint i = tid; i < 2048; i += 256) {
        float v = float(x_row[i]);
        local_sq += v * v;
    }

    // Warp-level reduction
    float simd_sq = simd_sum(local_sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup-level reduction (8 warps)
    if (simd_id == 0) {
        float s = (simd_lane < 8) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / 2048.0f) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < 2048; i += 256) {
        y_row[i] = bfloat(float(x_row[i]) * scale * float(weight[i]));
    }
}

/// RMSNorm for hidden dimension 2048 with float32 input and output.
kernel void flint_rmsnorm_2048_f32(
    device const float* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[8];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    device const float* x_row = in_x + row * 2048;
    device float* y_row = out_y + row * 2048;

    float local_sq = 0.0f;
    for (uint i = tid; i < 2048; i += 256) {
        float v = x_row[i];
        local_sq += v * v;
    }

    float simd_sq = simd_sum(local_sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < 8) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / 2048.0f) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < 2048; i += 256) {
        y_row[i] = x_row[i] * scale * float(weight[i]);
    }
}

// MARK: - Head Dimension 128 (Attention Q and K Norms)

/// RMSNorm for head dimension 128 with bfloat16 input and output.
/// Dispatch configuration: 128 threads per threadgroup (1 element per thread), (rows, 1, 1) threadgroups.
kernel void flint_rmsnorm_128_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[4];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    device const bfloat* x_row = in_x + row * 128;
    device bfloat* y_row = out_y + row * 128;

    float v = float(x_row[tid]);
    float sq = v * v;

    float simd_sq = simd_sum(sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < 4) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / 128.0f) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    y_row[tid] = bfloat(v * scale * float(weight[tid]));
}

/// RMSNorm for head dimension 128 with float32 input and output.
kernel void flint_rmsnorm_128_f32(
    device const float* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[4];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    device const float* x_row = in_x + row * 128;
    device float* y_row = out_y + row * 128;

    float v = x_row[tid];
    float sq = v * v;

    float simd_sq = simd_sum(sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < 4) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / 128.0f) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    y_row[tid] = v * scale * float(weight[tid]);
}

// MARK: - General Dimension Fallback (Arbitrary D up to 8192)

/// General RMSNorm for arbitrary dimension D with bfloat16 input and output.
kernel void flint_rmsnorm_general_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[8];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    if (tid < 8) {
        shared_sums[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const bfloat* x_row = in_x + row * D;
    device bfloat* y_row = out_y + row * D;

    float local_sq = 0.0f;
    for (uint i = tid; i < D; i += 256) {
        float v = float(x_row[i]);
        local_sq += v * v;
    }

    float simd_sq = simd_sum(local_sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < 8) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / float(D)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < D; i += 256) {
        y_row[i] = bfloat(float(x_row[i]) * scale * float(weight[i]));
    }
}

/// General RMSNorm for arbitrary dimension D with float32 input and output.
kernel void flint_rmsnorm_general_f32(
    device const float* in_x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sums[8];
    threadgroup float shared_scale;

    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;

    if (tid < 8) {
        shared_sums[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* x_row = in_x + row * D;
    device float* y_row = out_y + row * D;

    float local_sq = 0.0f;
    for (uint i = tid; i < D; i += 256) {
        float v = x_row[i];
        local_sq += v * v;
    }

    float simd_sq = simd_sum(local_sq);
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float s = (simd_lane < 8) ? shared_sums[simd_lane] : 0.0f;
        float total_sq = simd_sum(s);
        if (simd_lane == 0) {
            shared_scale = rsqrt((total_sq / float(D)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < D; i += 256) {
        y_row[i] = x_row[i] * scale * float(weight[i]);
    }
}
