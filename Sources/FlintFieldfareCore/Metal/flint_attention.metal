#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Attention (SDPA, Grouped-Query Attention, and Attention Sinks)
//
// Mathematical Specification:
//   - 32 query heads, 4 KV heads (GQA with 8 query heads sharing 1 KV head).
//   - Head dimension D = 128.
//   - Attention scale: 1.0 / sqrt(128.0) ~ 0.0883883476f.
//   - Learned Attention Sinks: Each query head h in [0, 31] has a learned scalar sink s_h
//     stored in `softmax_offset[h]`.
//     The sink participates as an extra logit in the softmax denominator:
//       Denom = sum_{j in attended} exp(S_{i, j}) + exp(s_h)
//       P_{i, j} = exp(S_{i, j}) / Denom
//       O_{i} = sum_{j in attended} P_{i, j} * V_{j}
//     Notice: The sink contributes NO value vector to the output.
//
// Bounded-KV Indexing Contract:
//   - `kv_retained_start`: Absolute sequence position stored at physical KV index 0.
//   - `T_kv`: Number of tokens physically retained in the K/V buffers.
//   - Physical buffers store keys for absolute positions [kv_retained_start, kv_retained_start + T_kv - 1].
//   - For a query at absolute position p:
//       Attended window: [k_start_abs, p] where k_start_abs = is_sliding ? max(0, p - 512) : 0.
//       Intersection with retained buffer: [max(k_start_abs, kv_retained_start), min(p, kv_end_abs)].
//       Physical index range: [phys_start, phys_end).
//       Every accessed physical index is strictly bounded within [0, T_kv - 1].
//
// Architecture:
//   - Streaming online softmax (FlashAttention formulation) in threadgroup memory.
//   - Zero global memory allocations for intermediate score matrices.
//   - Internal softmax statistics (m, l, acc) accumulate in FP32.
// ==============================================================================

// MARK: - Single-Token Decode Attention (T_q = 1)

/// Single-token decode attention for bfloat16 activations with bfloat16 learned sinks.
/// Grid dispatch: threadsPerThreadgroup = (128, 1, 1), threadgroups = (32, 1, batch).
kernel void flint_attention_decode_bf16(
    device const bfloat* in_q [[buffer(0)]],          // [batch, 32, 1, 128]
    device const bfloat* in_k [[buffer(1)]],          // [batch, 4, T_kv, 128]
    device const bfloat* in_v [[buffer(2)]],          // [batch, 4, T_kv, 128]
    device const bfloat* sinks [[buffer(3)]],         // [32] learned scalar sinks (BF16)
    device bfloat* out_o [[buffer(4)]],               // [batch, 32, 1, 128]
    constant uint& T_kv [[buffer(5)]],                // Total physical KV tokens
    constant uint& phys_start_kv [[buffer(6)]],       // First physical key index to attend
    constant uint& phys_end_kv [[buffer(7)]],         // One past the last physical key index
    constant float& scale [[buffer(8)]],              // Attention scaling factor (1 / sqrt(128))
    uint3 tid [[thread_position_in_threadgroup]],     // (0 .. 127, 0, 0)
    uint3 pos_in_grid [[threadgroup_position_in_grid]] // (head_q, 1, batch)
) {
    uint d = tid.x;
    uint h_q = pos_in_grid.x;
    uint b = pos_in_grid.z;

    if (h_q >= 32) return;
    uint h_kv = h_q >> 3; // h_q / 8

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    // Pointers for query head
    uint q_offset = b * (32 * 128) + h_q * 128;
    float q_val = float(in_q[q_offset + d]);

    // Pointers for KV head
    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    // Initialize online softmax with the learned attention sink
    float s_h = float(sinks[h_q]);
    float m = s_h;
    float l = 1.0f; // exp(s_h - m) = exp(0) = 1.0
    float acc_d = 0.0f; // Sink contributes zero value vector

    // Stream through attended physical keys
    for (uint j = phys_start_kv; j < phys_end_kv; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = float(in_k[j_offset + d]);

        // Dot product Q * K
        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        // Online softmax update
        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = float(in_v[j_offset + d]);
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    // Write normalized output vector
    out_o[q_offset + d] = bfloat(acc_d / l);
}

/// Single-token decode attention for float32 activations with float32 sinks.
kernel void flint_attention_decode_f32(
    device const float* in_q [[buffer(0)]],
    device const float* in_k [[buffer(1)]],
    device const float* in_v [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out_o [[buffer(4)]],
    constant uint& T_kv [[buffer(5)]],
    constant uint& phys_start_kv [[buffer(6)]],
    constant uint& phys_end_kv [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint d = tid.x;
    uint h_q = pos_in_grid.x;
    uint b = pos_in_grid.z;

    if (h_q >= 32) return;
    uint h_kv = h_q >> 3;

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    uint q_offset = b * (32 * 128) + h_q * 128;
    float q_val = in_q[q_offset + d];

    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    float s_h = sinks[h_q];
    float m = s_h;
    float l = 1.0f;
    float acc_d = 0.0f;

    for (uint j = phys_start_kv; j < phys_end_kv; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = in_k[j_offset + d];

        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = in_v[j_offset + d];
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    out_o[q_offset + d] = acc_d / l;
}
/// Single-token decode attention for float32 activations with bfloat16 learned sinks (direct checkpoint weights).
kernel void flint_attention_decode_f32_bf16sinks(
    device const float* in_q [[buffer(0)]],
    device const float* in_k [[buffer(1)]],
    device const float* in_v [[buffer(2)]],
    device const bfloat* sinks [[buffer(3)]],
    device float* out_o [[buffer(4)]],
    constant uint& T_kv [[buffer(5)]],
    constant uint& phys_start_kv [[buffer(6)]],
    constant uint& phys_end_kv [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint d = tid.x;
    uint h_q = pos_in_grid.x;
    uint b = pos_in_grid.z;

    if (h_q >= 32) return;
    uint h_kv = h_q >> 3;

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    uint q_offset = b * (32 * 128) + h_q * 128;
    float q_val = in_q[q_offset + d];

    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    float s_h = float(sinks[h_q]);
    float m = s_h;
    float l = 1.0f;
    float acc_d = 0.0f;

    for (uint j = phys_start_kv; j < phys_end_kv; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = in_k[j_offset + d];

        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = in_v[j_offset + d];
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    out_o[q_offset + d] = acc_d / l;
}
// MARK: - Multi-Token Prefill Attention (T_q >= 1)

/// Multi-token prefill attention for bfloat16 activations with bfloat16 sinks.
/// Grid dispatch: threadsPerThreadgroup = (128, 1, 1), threadgroups = (T_q, 32, batch).
kernel void flint_attention_prefill_bf16(
    device const bfloat* in_q [[buffer(0)]],          // [batch, 32, T_q, 128]
    device const bfloat* in_k [[buffer(1)]],          // [batch, 4, T_kv, 128]
    device const bfloat* in_v [[buffer(2)]],          // [batch, 4, T_kv, 128]
    device const bfloat* sinks [[buffer(3)]],         // [32]
    device bfloat* out_o [[buffer(4)]],               // [batch, 32, T_q, 128]
    constant uint& T_q [[buffer(5)]],                 // Number of queries
    constant uint& T_kv [[buffer(6)]],                // Number of physically retained keys/values
    constant uint& cache_offset [[buffer(7)]],        // Absolute starting position of queries
    constant uint& kv_retained_start [[buffer(8)]],   // Absolute starting position of physical KV index 0
    constant uint& is_sliding [[buffer(9)]],          // 1 if sliding window (513), 0 if full causal
    constant float& scale [[buffer(10)]],             // 1 / sqrt(128)
    uint3 tid [[thread_position_in_threadgroup]],     // (0 .. 127, 0, 0)
    uint3 pos_in_grid [[threadgroup_position_in_grid]] // (t, head_q, batch)
) {
    uint d = tid.x;
    uint t = pos_in_grid.x;
    uint h_q = pos_in_grid.y;
    uint b = pos_in_grid.z;

    if (t >= T_q || h_q >= 32) return;
    uint h_kv = h_q >> 3;

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    // Query offset
    uint q_head_stride = T_q * 128;
    uint q_offset = b * (32 * q_head_stride) + h_q * q_head_stride + t * 128;
    float q_val = float(in_q[q_offset + d]);

    // KV offset
    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    // Determine attended key range via bounded-KV indexing contract
    uint p = cache_offset + t;
    uint k_start_abs = (is_sliding != 0 && p >= 512) ? (p - 512) : 0;
    uint k_end_abs = p;
    uint kv_end_abs = kv_retained_start + T_kv - 1;

    uint eff_start = max(k_start_abs, kv_retained_start);
    uint eff_end = min(k_end_abs, kv_end_abs);

    uint phys_start = 0;
    uint phys_end = 0;
    if (eff_start <= eff_end && eff_start >= kv_retained_start) {
        phys_start = eff_start - kv_retained_start;
        phys_end = eff_end - kv_retained_start + 1;
    }

    // Initialize online softmax with sink
    float s_h = float(sinks[h_q]);
    float m = s_h;
    float l = 1.0f;
    float acc_d = 0.0f;

    for (uint j = phys_start; j < phys_end; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = float(in_k[j_offset + d]);

        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = float(in_v[j_offset + d]);
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    out_o[q_offset + d] = bfloat(acc_d / l);
}

/// Multi-token prefill attention for float32 activations with float32 sinks.
kernel void flint_attention_prefill_f32(
    device const float* in_q [[buffer(0)]],
    device const float* in_k [[buffer(1)]],
    device const float* in_v [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out_o [[buffer(4)]],
    constant uint& T_q [[buffer(5)]],
    constant uint& T_kv [[buffer(6)]],
    constant uint& cache_offset [[buffer(7)]],
    constant uint& kv_retained_start [[buffer(8)]],
    constant uint& is_sliding [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint d = tid.x;
    uint t = pos_in_grid.x;
    uint h_q = pos_in_grid.y;
    uint b = pos_in_grid.z;

    if (t >= T_q || h_q >= 32) return;
    uint h_kv = h_q >> 3;

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    uint q_head_stride = T_q * 128;
    uint q_offset = b * (32 * q_head_stride) + h_q * q_head_stride + t * 128;
    float q_val = in_q[q_offset + d];

    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    uint p = cache_offset + t;
    uint k_start_abs = (is_sliding != 0 && p >= 512) ? (p - 512) : 0;
    uint k_end_abs = p;
    uint kv_end_abs = kv_retained_start + T_kv - 1;

    uint eff_start = max(k_start_abs, kv_retained_start);
    uint eff_end = min(k_end_abs, kv_end_abs);

    uint phys_start = 0;
    uint phys_end = 0;
    if (eff_start <= eff_end && eff_start >= kv_retained_start) {
        phys_start = eff_start - kv_retained_start;
        phys_end = eff_end - kv_retained_start + 1;
    }

    float s_h = sinks[h_q];
    float m = s_h;
    float l = 1.0f;
    float acc_d = 0.0f;

    for (uint j = phys_start; j < phys_end; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = in_k[j_offset + d];

        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = in_v[j_offset + d];
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    out_o[q_offset + d] = acc_d / l;
}
/// Multi-token prefill attention for float32 activations with bfloat16 sinks (direct checkpoint weights).
kernel void flint_attention_prefill_f32_bf16sinks(
    device const float* in_q [[buffer(0)]],
    device const float* in_k [[buffer(1)]],
    device const float* in_v [[buffer(2)]],
    device const bfloat* sinks [[buffer(3)]],
    device float* out_o [[buffer(4)]],
    constant uint& T_q [[buffer(5)]],
    constant uint& T_kv [[buffer(6)]],
    constant uint& cache_offset [[buffer(7)]],
    constant uint& kv_retained_start [[buffer(8)]],
    constant uint& is_sliding [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint d = tid.x;
    uint t = pos_in_grid.x;
    uint h_q = pos_in_grid.y;
    uint b = pos_in_grid.z;

    if (t >= T_q || h_q >= 32) return;
    uint h_kv = h_q >> 3;

    threadgroup float shared_warp_sums[4];
    threadgroup float shared_score;

    uint simd_id = d >> 5;
    uint lane = d & 31;

    uint q_head_stride = T_q * 128;
    uint q_offset = b * (32 * q_head_stride) + h_q * q_head_stride + t * 128;
    float q_val = in_q[q_offset + d];

    uint kv_head_stride = T_kv * 128;
    uint kv_base_offset = b * (4 * kv_head_stride) + h_kv * kv_head_stride;

    uint p = cache_offset + t;
    uint k_start_abs = (is_sliding != 0 && p >= 512) ? (p - 512) : 0;
    uint k_end_abs = p;
    uint kv_end_abs = kv_retained_start + T_kv - 1;

    uint eff_start = max(k_start_abs, kv_retained_start);
    uint eff_end = min(k_end_abs, kv_end_abs);

    uint phys_start = 0;
    uint phys_end = 0;
    if (eff_start <= eff_end && eff_start >= kv_retained_start) {
        phys_start = eff_start - kv_retained_start;
        phys_end = eff_end - kv_retained_start + 1;
    }

    float s_h = float(sinks[h_q]);
    float m = s_h;
    float l = 1.0f;
    float acc_d = 0.0f;

    for (uint j = phys_start; j < phys_end; ++j) {
        uint j_offset = kv_base_offset + j * 128;
        float k_val = in_k[j_offset + d];

        float local_prod = q_val * k_val;
        float warp_sum = simd_sum(local_prod);
        if (lane == 0) {
            shared_warp_sums[simd_id] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float total = shared_warp_sums[0] + shared_warp_sums[1] + shared_warp_sums[2] + shared_warp_sums[3];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_j = shared_score;

        float m_new = max(m, S_j);
        float alpha = exp(m - m_new);
        float beta = exp(S_j - m_new);

        l = l * alpha + beta;
        float v_val = in_v[j_offset + d];
        acc_d = acc_d * alpha + beta * v_val;
        m = m_new;
    }

    out_o[q_offset + d] = acc_d / l;
}
