#include <metal_stdlib>
using namespace metal;

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

kernel void qwen_rmsnorm_f32(
    device const float* in       [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant uint& D             [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint lid                     [[thread_position_in_threadgroup]],
    uint lsize                   [[threads_per_threadgroup]],
    uint simd_lane_id            [[thread_index_in_simdgroup]],
    uint simd_group_id           [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = in[i];
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < (lsize / 32)) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        out[i] = in[i] * inv * weight[i];
    }
}

kernel void qwen_fused_residual_rmsnorm_f32(
    device const float* in_vec       [[buffer(0)]], // [D]
    device const float* attn_out     [[buffer(1)]], // [D]
    device const float* weight       [[buffer(2)]], // [D]
    device float* res_out            [[buffer(3)]], // [D]
    device float* norm_out           [[buffer(4)]], // [D]
    constant uint& D                 [[buffer(5)]],
    constant float& eps              [[buffer(6)]],
    uint lid                         [[thread_position_in_threadgroup]],
    uint lsize                       [[threads_per_threadgroup]],
    uint simd_lane_id                [[thread_index_in_simdgroup]],
    uint simd_group_id               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float r = in_vec[i] + attn_out[i];
        res_out[i] = r;
        acc = fma(r, r, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < (lsize / 32)) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        norm_out[i] = res_out[i] * inv * weight[i];
    }
}

kernel void qwen_gemv_f32(
    device const float* weight   [[buffer(0)]], // [rows, cols]
    device const float* vec      [[buffer(1)]], // [cols]
    device float* out            [[buffer(2)]], // [rows]
    constant uint& rows          [[buffer(3)]],
    constant uint& cols          [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint offset = gid * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        acc += weight[offset + c] * vec[c];
    }
    out[gid] = acc;
}

kernel void qwen_conv1d_f32(
    device const float* in_token       [[buffer(0)]],
    device float* conv_state           [[buffer(1)]], // [dim, 4]
    device const float* conv_weight    [[buffer(2)]], // [dim, 4]
    device float* out_token            [[buffer(3)]], // [dim]
    constant uint& dim                 [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    uint state_idx = gid * 4;
    float s0 = conv_state[state_idx + 1];
    float s1 = conv_state[state_idx + 2];
    float s2 = conv_state[state_idx + 3];
    float s3 = in_token[gid];

    conv_state[state_idx + 0] = s0;
    conv_state[state_idx + 1] = s1;
    conv_state[state_idx + 2] = s2;
    conv_state[state_idx + 3] = s3;

    float acc = s0 * conv_weight[state_idx + 0]
              + s1 * conv_weight[state_idx + 1]
              + s2 * conv_weight[state_idx + 2]
              + s3 * conv_weight[state_idx + 3];

    out_token[gid] = silu(acc);
}

kernel void qwen_prep_qkv_f32(
    device const float* conv_out  [[buffer(0)]], // [8192]
    device float* q_out           [[buffer(1)]], // [32 * 128]
    device float* k_out           [[buffer(2)]], // [32 * 128]
    device float* v_out           [[buffer(3)]], // [32 * 128]
    uint gid                      [[thread_position_in_grid]] // head idx 0..<32
) {
    if (gid >= 32) return;
    uint src_head = gid / 2;
    uint q_src_offset = src_head * 128;
    uint k_src_offset = 2048 + src_head * 128;
    uint v_src_offset = 4096 + gid * 128;

    uint dst_offset = gid * 128;

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint i = 0; i < 128; ++i) {
        float qv = conv_out[q_src_offset + i];
        float kv = conv_out[k_src_offset + i];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
        v_out[dst_offset + i] = conv_out[v_src_offset + i];
    }

    float q_inv = rsqrt(q_sum_sq + 1e-6f) * (1.0f / sqrt(128.0f));
    float k_inv = rsqrt(k_sum_sq + 1e-6f);

    for (uint i = 0; i < 128; ++i) {
        q_out[dst_offset + i] = conv_out[q_src_offset + i] * q_inv;
        k_out[dst_offset + i] = conv_out[k_src_offset + i] * k_inv;
    }
}

kernel void qwen_deltanet_recurrent_f32(
    device const float* q         [[buffer(0)]], // [32 * 128]
    device const float* k         [[buffer(1)]], // [32 * 128]
    device const float* v         [[buffer(2)]], // [32 * 128]
    device const float* a         [[buffer(3)]], // [32]
    device const float* b         [[buffer(4)]], // [32]
    device const float* a_log     [[buffer(5)]], // [32]
    device const float* dt_bias   [[buffer(6)]], // [32]
    device float* ssm_state       [[buffer(7)]], // [32, 128, 128]
    device float* out             [[buffer(8)]], // [32 * 128]
    uint2 gid                     [[thread_position_in_grid]] // (row in 0..<128, head in 0..<32)
) {
    uint row = gid.x;
    uint head = gid.y;
    if (head >= 32 || row >= 128) return;

    float beta_h = sigmoid(b[head]);
    float g_h = exp(-exp(a_log[head]) * softplus(a[head] + dt_bias[head]));

    uint head_state_base = head * 128 * 128;
    uint row_state_base = head_state_base + row * 128;
    uint head_k_base = head * 128;
    uint head_q_base = head * 128;

    // 1. Decay state row
    for (uint col = 0; col < 128; ++col) {
        ssm_state[row_state_base + col] *= g_h;
    }

    // 2. kv_mem_r = sum(state[row, :] * k)
    float kv_mem_r = 0.0f;
    for (uint col = 0; col < 128; ++col) {
        kv_mem_r += ssm_state[row_state_base + col] * k[head_k_base + col];
    }

    // 3. delta_r = (v[head, row] - kv_mem_r) * beta_h
    float v_r = v[head * 128 + row];
    float delta_r = (v_r - kv_mem_r) * beta_h;

    // 4. state[row, col] += k[col] * delta_r
    for (uint col = 0; col < 128; ++col) {
        ssm_state[row_state_base + col] += k[head_k_base + col] * delta_r;
    }

    // 5. core_r = sum(state[row, :] * q)
    float core_r = 0.0f;
    for (uint col = 0; col < 128; ++col) {
        core_r += ssm_state[row_state_base + col] * q[head_q_base + col];
    }

    out[head * 128 + row] = core_r;
}

kernel void qwen_gated_rmsnorm_f32(
    device const float* core_attn   [[buffer(0)]], // [32 * 128]
    device const float* norm_weight [[buffer(1)]], // [128]
    device const float* z           [[buffer(2)]], // [32 * 128]
    device float* out               [[buffer(3)]], // [32 * 128]
    uint head                       [[thread_position_in_grid]] // 0..<32
) {
    if (head >= 32) return;
    uint offset = head * 128;

    float sum_sq = 0.0f;
    for (uint i = 0; i < 128; ++i) {
        float val = core_attn[offset + i];
        sum_sq += val * val;
    }
    float inv = rsqrt(sum_sq / 128.0f + 1e-6f);

    for (uint i = 0; i < 128; ++i) {
        uint idx = offset + i;
        out[idx] = core_attn[idx] * inv * norm_weight[i] * silu(z[idx]);
    }
}

kernel void qwen_fused_deltanet_step_f32(
    device const float* conv_out    [[buffer(0)]], // [8192]
    device const float* z           [[buffer(1)]], // [4096]
    device const float* a           [[buffer(2)]], // [32]
    device const float* b           [[buffer(3)]], // [32]
    device const float* a_log       [[buffer(4)]], // [32]
    device const float* dt_bias     [[buffer(5)]], // [32]
    device float* ssm_state         [[buffer(6)]], // [32, 128, 128]
    device const float* norm_weight [[buffer(7)]], // [128]
    device float* gated_norm_out    [[buffer(8)]], // [4096]
    uint r                          [[thread_position_in_threadgroup]], // 0..<128
    uint head                       [[threadgroup_position_in_grid]],    // 0..<32
    uint simd_lane_id               [[thread_index_in_simdgroup]],       // 0..<31
    uint simd_group_id              [[simdgroup_index_in_threadgroup]]   // 0..<3
) {
    if (head >= 32 || r >= 128) return;

    threadgroup float tg_q[128];
    threadgroup float tg_k[128];
    threadgroup float tg_core[128];
    threadgroup float partial_sum[8];

    // 1. Q, K, V Prep
    uint src_head = head / 2;
    uint q_src_offset = src_head * 128 + r;
    uint k_src_offset = 2048 + src_head * 128 + r;
    uint v_src_offset = 4096 + head * 128 + r;

    float qv = conv_out[q_src_offset];
    float kv = conv_out[k_src_offset];
    float vr = conv_out[v_src_offset];

    float sg_q_sq = simd_sum(qv * qv);
    float sg_k_sq = simd_sum(kv * kv);
    if (simd_lane_id == 0) {
        partial_sum[simd_group_id * 2 + 0] = sg_q_sq;
        partial_sum[simd_group_id * 2 + 1] = sg_k_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float q_tot = (simd_lane_id < 4) ? partial_sum[simd_lane_id * 2 + 0] : 0.0f;
        float k_tot = (simd_lane_id < 4) ? partial_sum[simd_lane_id * 2 + 1] : 0.0f;
        q_tot = simd_sum(q_tot);
        k_tot = simd_sum(k_tot);
        if (simd_lane_id == 0) {
            partial_sum[0] = rsqrt(q_tot + 1e-6f) * (1.0f / sqrt(128.0f));
            partial_sum[1] = rsqrt(k_tot + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_inv = partial_sum[0];
    float k_inv = partial_sum[1];
    tg_q[r] = qv * q_inv;
    tg_k[r] = kv * k_inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. DeltaNet Recurrence for Row r
    float beta_h = sigmoid(b[head]);
    float g_h = exp(-exp(a_log[head]) * softplus(a[head] + dt_bias[head]));

    device float* state_row = ssm_state + head * 16384 + r * 128;
    device float4* state_row4 = reinterpret_cast<device float4*>(state_row);
    threadgroup const float4* tg_k4 = reinterpret_cast<threadgroup const float4*>(tg_k);
    threadgroup const float4* tg_q4 = reinterpret_cast<threadgroup const float4*>(tg_q);

    float kv_mem_r = 0.0f;
    for (uint c4 = 0; c4 < 32; ++c4) {
        float4 s4 = state_row4[c4] * g_h;
        state_row4[c4] = s4;
        kv_mem_r += dot(s4, tg_k4[c4]);
    }

    float delta_r = (vr - kv_mem_r) * beta_h;
    float4 delta4 = float4(delta_r);

    float core_r = 0.0f;
    for (uint c4 = 0; c4 < 32; ++c4) {
        float4 s4 = fma(tg_k4[c4], delta4, state_row4[c4]);
        state_row4[c4] = s4;
        core_r += dot(s4, tg_q4[c4]);
    }

    tg_core[r] = core_r;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Gated RMSNorm for Head
    float sg_core_sq = simd_sum(core_r * core_r);
    if (simd_lane_id == 0) {
        partial_sum[simd_group_id] = sg_core_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float c_tot = (simd_lane_id < 4) ? partial_sum[simd_lane_id] : 0.0f;
        c_tot = simd_sum(c_tot);
        if (simd_lane_id == 0) {
            partial_sum[0] = rsqrt(c_tot / 128.0f + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = partial_sum[0];
    uint out_idx = head * 128 + r;
    gated_norm_out[out_idx] = core_r * inv_norm * norm_weight[r] * silu(z[out_idx]);
}

kernel void qwen_fused_gemv_ab_f32(
    device const float* b_weight [[buffer(0)]], // [32, 2048]
    device const float* a_weight [[buffer(1)]], // [32, 2048]
    device const float* vec      [[buffer(2)]], // [2048]
    device float* b_out          [[buffer(3)]], // [32]
    device float* a_out          [[buffer(4)]], // [32]
    constant uint& rows          [[buffer(5)]], // 32
    constant uint& cols          [[buffer(6)]], // 2048
    uint gid                     [[thread_position_in_grid]] // 0..<32
) {
    if (gid >= rows) return;
    uint offset = gid * cols;
    float b_acc = 0.0f;
    float a_acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        float v = vec[c];
        b_acc += b_weight[offset + c] * v;
        a_acc += a_weight[offset + c] * v;
    }
    b_out[gid] = b_acc;
    a_out[gid] = a_acc;
}

kernel void qwen_fused_deltanet_gated_rmsnorm_f32(
    device const float* q           [[buffer(0)]], // [32 * 128]
    device const float* k           [[buffer(1)]], // [32 * 128]
    device const float* v           [[buffer(2)]], // [32 * 128]
    device const float* a           [[buffer(3)]], // [32]
    device const float* b           [[buffer(4)]], // [32]
    device const float* a_log       [[buffer(5)]], // [32]
    device const float* dt_bias     [[buffer(6)]], // [32]
    device float* ssm_state         [[buffer(7)]], // [32, 128, 128]
    device const float* norm_weight [[buffer(8)]], // [128]
    device const float* z           [[buffer(9)]], // [32 * 128]
    device float* out               [[buffer(10)]], // [32 * 128]
    uint tid                        [[thread_position_in_threadgroup]], // 0..<128 (row)
    uint head                       [[threadgroup_position_in_grid]]    // 0..<32 (head)
) {
    if (head >= 32 || tid >= 128) return;

    threadgroup float shared_core[128];
    threadgroup float shared_inv;

    float beta_h = sigmoid(b[head]);
    float g_h = exp(-exp(a_log[head]) * softplus(a[head] + dt_bias[head]));

    uint head_state_base = head * 128 * 128;
    uint row_state_base = head_state_base + tid * 128;
    uint head_k_base = head * 128;
    uint head_q_base = head * 128;

    // 1. Decay state row
    for (uint col = 0; col < 128; ++col) {
        ssm_state[row_state_base + col] *= g_h;
    }

    // 2. kv_mem_r = sum(state[row, :] * k)
    float kv_mem_r = 0.0f;
    for (uint col = 0; col < 128; ++col) {
        kv_mem_r += ssm_state[row_state_base + col] * k[head_k_base + col];
    }

    // 3. delta_r = (v[head, row] - kv_mem_r) * beta_h
    float v_r = v[head * 128 + tid];
    float delta_r = (v_r - kv_mem_r) * beta_h;

    // 4. state[row, col] += k[col] * delta_r
    for (uint col = 0; col < 128; ++col) {
        ssm_state[row_state_base + col] += k[head_k_base + col] * delta_r;
    }

    // 5. core_r = sum(state[row, :] * q)
    float core_r = 0.0f;
    for (uint col = 0; col < 128; ++col) {
        core_r += ssm_state[row_state_base + col] * q[head_q_base + col];
    }

    shared_core[tid] = core_r;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. Compute RMSNorm inverse
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (uint i = 0; i < 128; ++i) {
            float val = shared_core[i];
            sum_sq += val * val;
        }
        shared_inv = rsqrt(sum_sq / 128.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 7. Gated output
    uint idx = head * 128 + tid;
    out[idx] = shared_core[tid] * shared_inv * norm_weight[tid] * silu(z[idx]);
}

kernel void qwen_residual_add_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float* out       [[buffer(2)]],
    constant uint& count    [[buffer(3)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}

kernel void qwen_gemv_bf16_to_f32(
    device const bfloat* weight  [[buffer(0)]], // [rows, cols]
    device const float* vec      [[buffer(1)]], // [cols]
    device float* out            [[buffer(2)]], // [rows]
    constant uint& rows          [[buffer(3)]],
    constant uint& cols          [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint offset = gid * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        acc += float(weight[offset + c]) * vec[c];
    }
    out[gid] = acc;
}

kernel void qwen_gate_up_silu_f32(
    device const float* gate_up [[buffer(0)]], // [1024]
    device float* act           [[buffer(1)]], // [512]
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= 512) return;
    float gate = gate_up[gid];
    float up = gate_up[512 + gid];
    act[gid] = silu(gate) * up;
}

kernel void qwen_silu_mul_f32(
    device const float* gate [[buffer(0)]], // [dim]
    device const float* up   [[buffer(1)]], // [dim]
    device float* out        [[buffer(2)]], // [dim]
    constant uint& dim       [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    out[gid] = silu(gate[gid]) * up[gid];
}

kernel void qwen_expert_accum_f32(
    device const float* expert_out [[buffer(0)]], // [2048]
    device float* moe_accum        [[buffer(1)]], // [2048]
    constant float& weight_score   [[buffer(2)]],
    constant uint& dim             [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    moe_accum[gid] += expert_out[gid] * weight_score;
}

kernel void qwen_shared_expert_gate_f32(
    device const float* shared_down [[buffer(0)]], // [2048]
    device const float* gate_coeff  [[buffer(1)]], // [1]
    device float* out               [[buffer(2)]], // [2048]
    constant uint& dim              [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    float coeff = sigmoid(gate_coeff[0]);
    out[gid] = shared_down[gid] * coeff;
}

kernel void qwen_layer_final_residual_f32(
    device const float* attn_residual [[buffer(0)]], // [2048]
    device const float* moe_out       [[buffer(1)]], // [2048]
    device const float* shared_out    [[buffer(2)]], // [2048]
    device float* final_out           [[buffer(3)]], // [2048]
    constant uint& dim                [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    final_out[gid] = attn_residual[gid] + moe_out[gid] + shared_out[gid];
}

kernel void qwen_fused_expert_down_accum_bf16_to_f32(
    device const bfloat* down_weight [[buffer(0)]], // [rows, cols] = [2048, 512]
    device const float* act          [[buffer(1)]], // [cols] = [512]
    device float* moe_accum          [[buffer(2)]], // [rows] = [2048]
    constant float& weight_score     [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint offset = gid * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        acc += float(down_weight[offset + c]) * act[c];
    }
    moe_accum[gid] += acc * weight_score;
}

kernel void qwen_fused_shared_gate_up_silu_f32(
    device const float* gate_weight  [[buffer(0)]], // [rows, cols] = [512, 2048]
    device const float* up_weight    [[buffer(1)]], // [rows, cols] = [512, 2048]
    device const float* vec          [[buffer(2)]], // [cols] = [2048]
    device float* act_out            [[buffer(3)]], // [rows] = [512]
    constant uint& rows              [[buffer(4)]],
    constant uint& cols              [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint offset = gid * cols;
    float g_acc = 0.0f;
    float u_acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        float v = vec[c];
        g_acc += gate_weight[offset + c] * v;
        u_acc += up_weight[offset + c] * v;
    }
    act_out[gid] = silu(g_acc) * u_acc;
}

kernel void qwen_fused_shared_down_final_residual_f32(
    device const float* shared_down_weight [[buffer(0)]], // [rows, cols] = [2048, 512]
    device const float* sh_act             [[buffer(1)]], // [cols] = [512]
    device const float* gate_coeff_buf     [[buffer(2)]], // [1]
    device const float* attn_residual      [[buffer(3)]], // [2048]
    device const float* moe_accum          [[buffer(4)]], // [2048]
    device float* final_out                [[buffer(5)]], // [2048]
    constant uint& rows                    [[buffer(6)]],
    constant uint& cols                    [[buffer(7)]],
    uint gid                               [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    float coeff = sigmoid(gate_coeff_buf[0]);
    uint offset = gid * cols;
    float down_acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        down_acc += shared_down_weight[offset + c] * sh_act[c];
    }
    final_out[gid] = attn_residual[gid] + moe_accum[gid] + (down_acc * coeff);
}

kernel void qwen_split_q_gate_f32(
    device const float* q_chunk [[buffer(0)]], // [16 * 512 = 8192]
    device float* q_out         [[buffer(1)]], // [16 * 256 = 4096]
    device float* gate_out      [[buffer(2)]], // [16 * 256 = 4096]
    uint head                   [[thread_position_in_grid]] // 0..<16
) {
    if (head >= 16) return;
    uint in_base = head * 512;
    uint out_base = head * 256;
    for (uint i = 0; i < 256; ++i) {
        q_out[out_base + i] = q_chunk[in_base + i];
        gate_out[out_base + i] = q_chunk[in_base + 256 + i];
    }
}

kernel void qwen_head_rmsnorm_f32(
    device float* x              [[buffer(0)]], // [num_heads * head_dim]
    device const float* weight   [[buffer(1)]], // [head_dim]
    constant uint& num_heads     [[buffer(2)]],
    constant uint& head_dim      [[buffer(3)]],
    uint head                    [[thread_position_in_grid]]
) {
    if (head >= num_heads) return;
    uint base = head * head_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
        float v = x[base + i];
        sum_sq += v * v;
    }
    float inv = rsqrt(sum_sq / float(head_dim) + 1e-6f);
    for (uint i = 0; i < head_dim; ++i) {
        x[base + i] = x[base + i] * inv * weight[i];
    }
}

kernel void qwen_rope_partial_f32(
    device float* x              [[buffer(0)]], // [num_heads * 256]
    device const float* cos_arr  [[buffer(1)]], // [64]
    device const float* sin_arr  [[buffer(2)]], // [64]
    constant uint& num_heads     [[buffer(3)]],
    uint head                    [[thread_position_in_grid]]
) {
    if (head >= num_heads) return;
    uint base = head * 256;
    for (uint i = 0; i < 32; ++i) {
        float x0 = x[base + i];
        float x1 = x[base + 32 + i];
        float x_rot0 = -x1;
        float x_rot1 = x0;
        x[base + i]      = x0 * cos_arr[i] + x_rot0 * sin_arr[i];
        x[base + 32 + i] = x1 * cos_arr[32 + i] + x_rot1 * sin_arr[32 + i];
    }
}

kernel void qwen_update_kv_cache_f32(
    device const float* k_in      [[buffer(0)]], // [512]
    device const float* v_in      [[buffer(1)]], // [512]
    device float* k_cache         [[buffer(2)]], // [2, max_seq, 256]
    device float* v_cache         [[buffer(3)]], // [2, max_seq, 256]
    constant uint& curr_pos       [[buffer(4)]],
    constant uint& max_seq        [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]] // 0..<512
) {
    if (gid >= 512) return;
    uint kh = gid / 256;
    uint d = gid % 256;
    uint offset = kh * max_seq * 256 + curr_pos * 256 + d;
    k_cache[offset] = k_in[gid];
    v_cache[offset] = v_in[gid];
}

// Qwen decode production KV path. The cache is token-major to match the
// shared split-K/V attention kernels: [sequence, kv_head, head_dim]. Keeping
// K/V in FP16 halves the long-context footprint and memory traffic while the
// attention accumulators remain FP32.
kernel void qwen_update_kv_cache_f16(
    device const float* k_in      [[buffer(0)]], // [2, 256]
    device const float* v_in      [[buffer(1)]], // [2, 256]
    device half* k_cache          [[buffer(2)]], // [max_seq, 2, 256]
    device half* v_cache          [[buffer(3)]], // [max_seq, 2, 256]
    constant uint& curr_pos       [[buffer(4)]],
    constant uint& max_seq        [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= 512 || curr_pos >= max_seq) return;
    uint kv_head = gid / 256;
    uint d = gid % 256;
    uint offset = (curr_pos * 2 + kv_head) * 256 + d;
    k_cache[offset] = half(k_in[gid]);
    v_cache[offset] = half(v_in[gid]);
}

kernel void qwen_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid < count) output[gid] = half(input[gid]);
}

kernel void qwen_apply_attention_gate_f16_f32(
    device const half* attention [[buffer(0)]],
    device const float* gate     [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& count         [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid < count) output[gid] = float(attention[gid]) * sigmoid(gate[gid]);
}

kernel void qwen_causal_attn_step_f32(
    device const float* q        [[buffer(0)]], // [16 * 256]
    device const float* k_cache  [[buffer(1)]], // [2, max_seq, 256]
    device const float* v_cache  [[buffer(2)]], // [2, max_seq, 256]
    device const float* gate     [[buffer(3)]], // [16 * 256]
    device float* out            [[buffer(4)]], // [16 * 256 = 4096]
    constant uint& curr_pos      [[buffer(5)]], // 0..<max_seq
    constant uint& max_seq       [[buffer(6)]],
    uint head                    [[thread_position_in_grid]] // 0..<16
) {
    if (head >= 16) return;
    uint kv_head = head / 8; // 16 query heads, 2 KV heads (8 query heads per KV head)
    uint q_base = head * 256;
    uint kv_head_stride = max_seq * 256;

    float scale = 1.0f / sqrt(256.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float weighted_value[256];
    for (uint d = 0; d < 256; ++d) {
        weighted_value[d] = 0.0f;
    }

    // Online softmax keeps storage independent of context length. The previous
    // fixed scores[64] array was written out of bounds from position 64 onward.
    for (uint s = 0; s <= curr_pos; ++s) {
        uint k_offset = kv_head * kv_head_stride + s * 256;
        float dot = 0.0f;
        for (uint d = 0; d < 256; ++d) {
            dot += q[q_base + d] * k_cache[k_offset + d];
        }
        float score = dot * scale;
        float next_max = max(running_max, score);
        float previous_scale = running_sum == 0.0f
            ? 0.0f
            : exp(running_max - next_max);
        float current_scale = exp(score - next_max);
        for (uint d = 0; d < 256; ++d) {
            uint v_offset = kv_head * kv_head_stride + s * 256;
            weighted_value[d] = weighted_value[d] * previous_scale
                + v_cache[v_offset + d] * current_scale;
        }
        running_sum = running_sum * previous_scale + current_scale;
        running_max = next_max;
    }

    for (uint d = 0; d < 256; ++d) {
        float g = sigmoid(gate[q_base + d]);
        out[q_base + d] = (weighted_value[d] / running_sum) * g;
    }
}

kernel void qwen_causal_attn_step_scratch_f32(
    device const float* q              [[buffer(0)]],
    device const float* k_cache        [[buffer(1)]],
    device const float* v_cache        [[buffer(2)]],
    device const float* gate           [[buffer(3)]],
    device float* out                  [[buffer(4)]],
    constant uint& curr_pos            [[buffer(5)]],
    constant uint& max_seq             [[buffer(6)]],
    device float* score_scratch        [[buffer(7)]], // [16, max_seq]
    uint head                          [[thread_position_in_grid]]
) {
    if (head >= 16) return;
    uint kv_head = head / 8;
    uint q_base = head * 256;
    uint kv_head_stride = max_seq * 256;
    device float* scores = score_scratch + head * max_seq;
    float max_score = -INFINITY;
    float scale = 1.0f / sqrt(256.0f);

    for (uint s = 0; s <= curr_pos; ++s) {
        uint k_offset = kv_head * kv_head_stride + s * 256;
        float dot = 0.0f;
        for (uint d = 0; d < 256; ++d) {
            dot += q[q_base + d] * k_cache[k_offset + d];
        }
        float score = dot * scale;
        scores[s] = score;
        max_score = max(max_score, score);
    }

    float sum_exp = 0.0f;
    for (uint s = 0; s <= curr_pos; ++s) {
        float value = exp(scores[s] - max_score);
        scores[s] = value;
        sum_exp += value;
    }

    for (uint d = 0; d < 256; ++d) {
        float acc = 0.0f;
        for (uint s = 0; s <= curr_pos; ++s) {
            uint v_offset = kv_head * kv_head_stride + s * 256;
            acc += (scores[s] / sum_exp) * v_cache[v_offset + d];
        }
        out[q_base + d] = acc * sigmoid(gate[q_base + d]);
    }
}

inline float read_bf16_unaligned(device const uchar* ptr, uint index) {
    uint byte_idx = index * 2;
    ushort u = ushort(ptr[byte_idx]) | (ushort(ptr[byte_idx + 1]) << 8);
    uint u32 = uint(u) << 16;
    return as_type<float>(u32);
}

kernel void qwen_gemv_q4_affine(
    device const uchar* raw_weight  [[buffer(0)]],
    device const uchar* raw_scales  [[buffer(1)]],
    device const uchar* raw_biases  [[buffer(2)]],
    device const float* vec         [[buffer(3)]],
    device float* out               [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& w_byte_offset    [[buffer(7)]],
    constant uint& s_byte_offset    [[buffer(8)]],
    constant uint& b_byte_offset    [[buffer(9)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint groups_per_row = cols / 64;
    device const uchar* w_row = raw_weight + w_byte_offset + gid * (cols / 2);
    device const uchar* s_row = raw_scales + s_byte_offset + gid * (groups_per_row * 2);
    device const uchar* b_row = raw_biases + b_byte_offset + gid * (groups_per_row * 2);

    float acc = 0.0f;
    for (uint g = 0; g < groups_per_row; ++g) {
        float scale = read_bf16_unaligned(s_row, g);
        float bias = read_bf16_unaligned(b_row, g);
        device const uchar* g_ptr = w_row + g * 32;
        uint v_base = g * 64;

        float g_w_acc = 0.0f;
        float g_v_acc = 0.0f;

        for (uint u = 0; u < 8; ++u) {
            uint val = uint(g_ptr[u * 4 + 0])
                     | (uint(g_ptr[u * 4 + 1]) << 8)
                     | (uint(g_ptr[u * 4 + 2]) << 16)
                     | (uint(g_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float v = vec[vi + k];
                float q = float((val >> (k * 4)) & 0x0F);
                g_w_acc = fma(q, v, g_w_acc);
                g_v_acc += v;
            }
        }
        acc = fma(g_w_acc, scale, fma(g_v_acc, bias, acc));
    }
    out[gid] = acc;
}

kernel void qwen_gemv_q8_affine(
    device const uchar* raw_weight  [[buffer(0)]],
    device const uchar* raw_scales  [[buffer(1)]],
    device const uchar* raw_biases  [[buffer(2)]],
    device const float* vec         [[buffer(3)]],
    device float* out               [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& w_byte_offset    [[buffer(7)]],
    constant uint& s_byte_offset    [[buffer(8)]],
    constant uint& b_byte_offset    [[buffer(9)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint groups_per_row = cols / 64;
    device const uchar* w_row = raw_weight + w_byte_offset + gid * cols;
    device const uchar* s_row = raw_scales + s_byte_offset + gid * (groups_per_row * 2);
    device const uchar* b_row = raw_biases + b_byte_offset + gid * (groups_per_row * 2);

    float acc = 0.0f;
    for (uint g = 0; g < groups_per_row; ++g) {
        float scale = read_bf16_unaligned(s_row, g);
        float bias = read_bf16_unaligned(b_row, g);
        device const uchar* g_ptr = w_row + g * 64;
        uint v_base = g * 64;

        for (uint u = 0; u < 16; ++u) {
            uint val = uint(g_ptr[u * 4 + 0])
                     | (uint(g_ptr[u * 4 + 1]) << 8)
                     | (uint(g_ptr[u * 4 + 2]) << 16)
                     | (uint(g_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 4;
            for (uint k = 0; k < 4; ++k) {
                float w_deq = float((val >> (k * 8)) & 0xFF) * scale + bias;
                acc += w_deq * vec[vi + k];
            }
        }
    }
    out[gid] = acc;
}

kernel void qwen_batched_expert_gate_up_silu_q4(
    device const uchar* gate_w        [[buffer(0)]],
    device const uchar* gate_s        [[buffer(1)]],
    device const uchar* gate_b        [[buffer(2)]],
    device const uchar* up_w          [[buffer(3)]],
    device const uchar* up_s          [[buffer(4)]],
    device const uchar* up_b          [[buffer(5)]],
    device const float* vec           [[buffer(6)]], // [2048]
    device float* batched_act_out     [[buffer(7)]], // [8 * 512]
    device const uint* expert_indices [[buffer(8)]], // [8]
    constant uint& gate_w_off         [[buffer(9)]],
    constant uint& gate_s_off         [[buffer(10)]],
    constant uint& gate_b_off         [[buffer(11)]],
    constant uint& up_w_off           [[buffer(12)]],
    constant uint& up_s_off           [[buffer(13)]],
    constant uint& up_b_off           [[buffer(14)]],
    uint gid                          [[thread_position_in_grid]] // 0..<4096 (8 * 512)
) {
    if (gid >= 4096) return;
    uint e = gid / 512;
    uint r = gid % 512;
    uint exp_idx = expert_indices[e];
    uint exp_byte_shift = exp_idx * 524288;
    uint exp_s_byte_shift = exp_idx * 32768;

    device const uchar* g_w_row = gate_w + gate_w_off + exp_byte_shift + r * 1024;
    device const uchar* g_s_row = gate_s + gate_s_off + exp_s_byte_shift + r * 64;
    device const uchar* g_b_row = gate_b + gate_b_off + exp_s_byte_shift + r * 64;

    device const uchar* u_w_row = up_w + up_w_off + exp_byte_shift + r * 1024;
    device const uchar* u_s_row = up_s + up_s_off + exp_s_byte_shift + r * 64;
    device const uchar* u_b_row = up_b + up_b_off + exp_s_byte_shift + r * 64;

    float g_acc = 0.0f;
    float u_acc = 0.0f;

    for (uint g = 0; g < 32; ++g) {
        float gs = read_bf16_unaligned(g_s_row, g);
        float gb = read_bf16_unaligned(g_b_row, g);
        float us = read_bf16_unaligned(u_s_row, g);
        float ub = read_bf16_unaligned(u_b_row, g);

        device const uchar* g_ptr = g_w_row + g * 32;
        device const uchar* u_ptr = u_w_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint g_val = uint(g_ptr[u * 4 + 0])
                       | (uint(g_ptr[u * 4 + 1]) << 8)
                       | (uint(g_ptr[u * 4 + 2]) << 16)
                       | (uint(g_ptr[u * 4 + 3]) << 24);
            uint u_val = uint(u_ptr[u * 4 + 0])
                       | (uint(u_ptr[u * 4 + 1]) << 8)
                       | (uint(u_ptr[u * 4 + 2]) << 16)
                       | (uint(u_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float v = vec[vi + k];
                float gw = float((g_val >> (k * 4)) & 0x0F) * gs + gb;
                float uw = float((u_val >> (k * 4)) & 0x0F) * us + ub;
                g_acc += gw * v;
                u_acc += uw * v;
            }
        }
    }
    batched_act_out[gid] = silu(g_acc) * u_acc;
}

kernel void qwen_batched_expert_down_accum_q4(
    device const uchar* down_w        [[buffer(0)]],
    device const uchar* down_s        [[buffer(1)]],
    device const uchar* down_b        [[buffer(2)]],
    device const float* batched_act   [[buffer(3)]], // [8 * 512]
    device const uint* expert_indices [[buffer(4)]], // [8]
    device const float* scores        [[buffer(5)]], // [8]
    device float* moe_accum_out       [[buffer(6)]], // [2048]
    constant uint& down_w_off         [[buffer(7)]],
    constant uint& down_s_off         [[buffer(8)]],
    constant uint& down_b_off         [[buffer(9)]],
    uint gid                          [[thread_position_in_grid]] // 0..<2048
) {
    if (gid >= 2048) return;
    float total_acc = 0.0f;

    for (uint e = 0; e < 8; ++e) {
        float score = scores[e];
        if (score <= 0.0f) continue;
        uint exp_idx = expert_indices[e];
        uint exp_byte_shift = exp_idx * 524288;
        uint exp_s_byte_shift = exp_idx * 32768;

        device const float* act_e = batched_act + e * 512;
        device const uchar* d_w_row = down_w + down_w_off + exp_byte_shift + gid * 256;
        device const uchar* d_s_row = down_s + down_s_off + exp_s_byte_shift + gid * 16;
        device const uchar* d_b_row = down_b + down_b_off + exp_s_byte_shift + gid * 16;

        float exp_acc = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            float ds = read_bf16_unaligned(d_s_row, g);
            float db = read_bf16_unaligned(d_b_row, g);
            device const uchar* d_ptr = d_w_row + g * 32;
            uint v_base = g * 64;

            for (uint u = 0; u < 8; ++u) {
                uint d_val = uint(d_ptr[u * 4 + 0])
                           | (uint(d_ptr[u * 4 + 1]) << 8)
                           | (uint(d_ptr[u * 4 + 2]) << 16)
                           | (uint(d_ptr[u * 4 + 3]) << 24);
                uint vi = v_base + u * 8;
                for (uint k = 0; k < 8; ++k) {
                    float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                    exp_acc += dw * act_e[vi + k];
                }
            }
        }
        total_acc += exp_acc * score;
    }
    moe_accum_out[gid] = total_acc;
}

kernel void qwen_fused_router_top8_q8(
    device const uchar* raw_weight       [[buffer(0)]], // [256, 2048]
    device const uchar* raw_scales       [[buffer(1)]],
    device const uchar* raw_biases       [[buffer(2)]],
    device const float* vec              [[buffer(3)]], // [2048]
    device uint* expert_indices          [[buffer(4)]], // [8]
    device float* routing_scores         [[buffer(5)]], // [8]
    constant uint& w_byte_offset         [[buffer(6)]],
    constant uint& s_byte_offset         [[buffer(7)]],
    constant uint& b_byte_offset         [[buffer(8)]],
    uint tid                             [[thread_position_in_threadgroup]]
) {
    threadgroup float s_logits[256];

    if (tid < 256) {
        device const uchar* w_row = raw_weight + w_byte_offset + tid * 2048;
        device const uchar* s_row = raw_scales + s_byte_offset + tid * 64;
        device const uchar* b_row = raw_biases + b_byte_offset + tid * 64;

        float acc = 0.0f;
        for (uint g = 0; g < 32; ++g) {
            float scale = read_bf16_unaligned(s_row, g);
            float bias = read_bf16_unaligned(b_row, g);
            device const uchar* g_ptr = w_row + g * 64;
            uint v_base = g * 64;

            for (uint u = 0; u < 16; ++u) {
                uint val = uint(g_ptr[u * 4 + 0])
                         | (uint(g_ptr[u * 4 + 1]) << 8)
                         | (uint(g_ptr[u * 4 + 2]) << 16)
                         | (uint(g_ptr[u * 4 + 3]) << 24);
                uint vi = v_base + u * 4;
                for (uint k = 0; k < 4; ++k) {
                    float w_deq = float((val >> (k * 8)) & 0xFF) * scale + bias;
                    acc += w_deq * vec[vi + k];
                }
            }
        }
        s_logits[tid] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint top_idx[8];
        float top_val[8];
        for (int k = 0; k < 8; ++k) {
            top_val[k] = -1e30f;
            top_idx[k] = 0;
        }

        for (uint i = 0; i < 256; ++i) {
            float v = s_logits[i];
            if (v > top_val[7]) {
                top_val[7] = v;
                top_idx[7] = i;
                for (int k = 6; k >= 0; --k) {
                    if (top_val[k + 1] > top_val[k]) {
                        float tv = top_val[k];
                        top_val[k] = top_val[k + 1];
                        top_val[k + 1] = tv;
                        uint ti = top_idx[k];
                        top_idx[k] = top_idx[k + 1];
                        top_idx[k + 1] = ti;
                    } else {
                        break;
                    }
                }
            }
        }

        float max_l = top_val[0];
        float sum_exp = 0.0f;
        for (uint i = 0; i < 256; ++i) {
            sum_exp += exp(s_logits[i] - max_l);
        }

        float top8_sum = 0.0f;
        float top8_prob[8];
        for (int k = 0; k < 8; ++k) {
            float p = exp(top_val[k] - max_l) / sum_exp;
            top8_prob[k] = p;
            top8_sum += p;
        }

        for (int k = 0; k < 8; ++k) {
            expert_indices[k] = top_idx[k];
            routing_scores[k] = top8_prob[k] / top8_sum;
        }
    }
}

kernel void qwen_fused_linear_attn_in_proj_q4(
    device const uchar* qkv_w     [[buffer(0)]],
    device const uchar* qkv_s     [[buffer(1)]],
    device const uchar* qkv_b     [[buffer(2)]],
    device const uchar* z_w       [[buffer(3)]],
    device const uchar* z_s       [[buffer(4)]],
    device const uchar* z_b       [[buffer(5)]],
    device const uchar* b_w       [[buffer(6)]],
    device const uchar* b_s       [[buffer(7)]],
    device const uchar* b_b       [[buffer(8)]],
    device const uchar* a_w       [[buffer(9)]],
    device const uchar* a_s       [[buffer(10)]],
    device const uchar* a_b       [[buffer(11)]],
    device const float* vec       [[buffer(12)]], // [2048]
    device float* qkv_out         [[buffer(13)]], // [8192]
    device float* z_out           [[buffer(14)]], // [4096]
    device float* b_out           [[buffer(15)]], // [32]
    device float* a_out           [[buffer(16)]], // [32]
    constant uint& qkv_w_off      [[buffer(17)]],
    constant uint& qkv_s_off      [[buffer(18)]],
    constant uint& qkv_b_off      [[buffer(19)]],
    constant uint& z_w_off        [[buffer(20)]],
    constant uint& z_s_off        [[buffer(21)]],
    constant uint& z_b_off        [[buffer(22)]],
    constant uint& b_w_off        [[buffer(23)]],
    constant uint& b_s_off        [[buffer(24)]],
    constant uint& b_b_off        [[buffer(25)]],
    constant uint& a_w_off        [[buffer(26)]],
    constant uint& a_s_off        [[buffer(27)]],
    constant uint& a_b_off        [[buffer(28)]],
    uint gid                      [[thread_position_in_grid]] // 0..<12352
) {
    if (gid >= 12352) return;

    device const uchar* raw_w;
    device const uchar* raw_s;
    device const uchar* raw_b;
    uint w_off, s_off, b_off;
    uint row;
    device float* out_target;

    if (gid < 8192) {
        raw_w = qkv_w; raw_s = qkv_s; raw_b = qkv_b;
        w_off = qkv_w_off; s_off = qkv_s_off; b_off = qkv_b_off;
        row = gid;
        out_target = qkv_out;
    } else if (gid < 12288) {
        raw_w = z_w; raw_s = z_s; raw_b = z_b;
        w_off = z_w_off; s_off = z_s_off; b_off = z_b_off;
        row = gid - 8192;
        out_target = z_out;
    } else if (gid < 12320) {
        raw_w = b_w; raw_s = b_s; raw_b = b_b;
        w_off = b_w_off; s_off = b_s_off; b_off = b_b_off;
        row = gid - 12288;
        out_target = b_out;
    } else {
        raw_w = a_w; raw_s = a_s; raw_b = a_b;
        w_off = a_w_off; s_off = a_s_off; b_off = a_b_off;
        row = gid - 12320;
        out_target = a_out;
    }

    // Every matrix has cols = 2048 (32 groups of 64 elements, 1024 bytes per row)
    device const uchar* w_row = raw_w + w_off + row * 1024;
    device const uchar* s_row = raw_s + s_off + row * 64;
    device const uchar* b_row = raw_b + b_off + row * 64;

    float acc = 0.0f;
    for (uint g = 0; g < 32; ++g) {
        float scale = read_bf16_unaligned(s_row, g);
        float bias = read_bf16_unaligned(b_row, g);
        device const uchar* g_ptr = w_row + g * 32;
        uint v_base = g * 64;

        float g_w_acc = 0.0f;
        float g_v_acc = 0.0f;

        for (uint u = 0; u < 8; ++u) {
            uint val = uint(g_ptr[u * 4 + 0])
                     | (uint(g_ptr[u * 4 + 1]) << 8)
                     | (uint(g_ptr[u * 4 + 2]) << 16)
                     | (uint(g_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float v = vec[vi + k];
                float q = float((val >> (k * 4)) & 0x0F);
                g_w_acc = fma(q, v, g_w_acc);
                g_v_acc += v;
            }
        }
        acc = fma(g_w_acc, scale, fma(g_v_acc, bias, acc));
    }
    out_target[row] = acc;
}

kernel void qwen_fused_moe_gate_up_silu_q4(
    device const uchar* switch_gw     [[buffer(0)]],
    device const uchar* switch_gs     [[buffer(1)]],
    device const uchar* switch_gb     [[buffer(2)]],
    device const uchar* switch_uw     [[buffer(3)]],
    device const uchar* switch_us     [[buffer(4)]],
    device const uchar* switch_ub     [[buffer(5)]],
    device const uchar* shared_gw     [[buffer(6)]],
    device const uchar* shared_gs     [[buffer(7)]],
    device const uchar* shared_gb     [[buffer(8)]],
    device const uchar* shared_uw     [[buffer(9)]],
    device const uchar* shared_us     [[buffer(10)]],
    device const uchar* shared_ub     [[buffer(11)]],
    device const float* vec           [[buffer(12)]], // [2048]
    device float* batched_act         [[buffer(13)]], // [8 * 512]
    device float* shared_act          [[buffer(14)]], // [512]
    device const uint* expert_indices [[buffer(15)]], // [8]
    constant uint& switch_gw_off      [[buffer(16)]],
    constant uint& switch_gs_off      [[buffer(17)]],
    constant uint& switch_gb_off      [[buffer(18)]],
    constant uint& switch_uw_off      [[buffer(19)]],
    constant uint& switch_us_off      [[buffer(20)]],
    constant uint& switch_ub_off      [[buffer(21)]],
    constant uint& shared_gw_off      [[buffer(22)]],
    constant uint& shared_gs_off      [[buffer(23)]],
    constant uint& shared_gb_off      [[buffer(24)]],
    constant uint& shared_uw_off      [[buffer(25)]],
    constant uint& shared_us_off      [[buffer(26)]],
    constant uint& shared_ub_off      [[buffer(27)]],
    uint gid                          [[thread_position_in_grid]] // 0..<4608
) {
    if (gid >= 4608) return;

    device const uchar* gw_row;
    device const uchar* gs_row;
    device const uchar* gb_row;
    device const uchar* uw_row;
    device const uchar* us_row;
    device const uchar* ub_row;

    if (gid < 4096) {
        uint e = gid / 512;
        uint r = gid % 512;
        uint exp_idx = expert_indices[e];
        uint exp_byte_shift = exp_idx * 524288;
        uint exp_s_byte_shift = exp_idx * 32768;

        gw_row = switch_gw + switch_gw_off + exp_byte_shift + r * 1024;
        gs_row = switch_gs + switch_gs_off + exp_s_byte_shift + r * 64;
        gb_row = switch_gb + switch_gb_off + exp_s_byte_shift + r * 64;
        uw_row = switch_uw + switch_uw_off + exp_byte_shift + r * 1024;
        us_row = switch_us + switch_us_off + exp_s_byte_shift + r * 64;
        ub_row = switch_ub + switch_ub_off + exp_s_byte_shift + r * 64;
    } else {
        uint r = gid - 4096;
        gw_row = shared_gw + shared_gw_off + r * 1024;
        gs_row = shared_gs + shared_gs_off + r * 64;
        gb_row = shared_gb + shared_gb_off + r * 64;
        uw_row = shared_uw + shared_uw_off + r * 1024;
        us_row = shared_us + shared_us_off + r * 64;
        ub_row = shared_ub + shared_ub_off + r * 64;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint g = 0; g < 32; ++g) {
        float gs = read_bf16_unaligned(gs_row, g);
        float gb = read_bf16_unaligned(gb_row, g);
        float us = read_bf16_unaligned(us_row, g);
        float ub = read_bf16_unaligned(ub_row, g);

        device const uchar* g_ptr = gw_row + g * 32;
        device const uchar* u_ptr = uw_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint g_val = uint(g_ptr[u * 4 + 0])
                       | (uint(g_ptr[u * 4 + 1]) << 8)
                       | (uint(g_ptr[u * 4 + 2]) << 16)
                       | (uint(g_ptr[u * 4 + 3]) << 24);
            uint u_val = uint(u_ptr[u * 4 + 0])
                       | (uint(u_ptr[u * 4 + 1]) << 8)
                       | (uint(u_ptr[u * 4 + 2]) << 16)
                       | (uint(u_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;

            for (uint k = 0; k < 8; ++k) {
                float gw = float((g_val >> (k * 4)) & 0x0F) * gs + gb;
                float uw = float((u_val >> (k * 4)) & 0x0F) * us + ub;
                float v = vec[vi + k];
                gate_acc += gw * v;
                up_acc += uw * v;
            }
        }
    }

    float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    float act_val = silu_gate * up_acc;

    if (gid < 4096) {
        batched_act[gid] = act_val;
    } else {
        shared_act[gid - 4096] = act_val;
    }
}

kernel void qwen_fused_moe_down_accum_residual_q4(
    device const uchar* switch_dw     [[buffer(0)]],
    device const uchar* switch_ds     [[buffer(1)]],
    device const uchar* switch_db     [[buffer(2)]],
    device const uchar* shared_dw     [[buffer(3)]],
    device const uchar* shared_ds     [[buffer(4)]],
    device const uchar* shared_db     [[buffer(5)]],
    device const float* batched_act   [[buffer(6)]], // [8 * 512]
    device const float* shared_act    [[buffer(7)]], // [512]
    device const uint* expert_indices [[buffer(8)]], // [8]
    device const float* scores        [[buffer(9)]], // [8]
    device const float* shared_coeff  [[buffer(10)]], // [1]
    device const float* attn_res      [[buffer(11)]], // [2048]
    device float* cur_out             [[buffer(12)]], // [2048]
    constant uint& switch_dw_off      [[buffer(13)]],
    constant uint& switch_ds_off      [[buffer(14)]],
    constant uint& switch_db_off      [[buffer(15)]],
    constant uint& shared_dw_off      [[buffer(16)]],
    constant uint& shared_ds_off      [[buffer(17)]],
    constant uint& shared_db_off      [[buffer(18)]],
    uint gid                          [[thread_position_in_grid]] // 0..<2048
) {
    if (gid >= 2048) return;

    // 1. Routed down projection across 8 active experts
    float routed_acc = 0.0f;
    for (uint e = 0; e < 8; ++e) {
        float score = scores[e];
        if (score <= 0.0f) continue;
        uint exp_idx = expert_indices[e];
        uint exp_byte_shift = exp_idx * 524288;
        uint exp_s_byte_shift = exp_idx * 32768;

        device const float* act_e = batched_act + e * 512;
        device const uchar* d_w_row = switch_dw + switch_dw_off + exp_byte_shift + gid * 256;
        device const uchar* d_s_row = switch_ds + switch_ds_off + exp_s_byte_shift + gid * 16;
        device const uchar* d_b_row = switch_db + switch_db_off + exp_s_byte_shift + gid * 16;

        float exp_acc = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            float ds = read_bf16_unaligned(d_s_row, g);
            float db = read_bf16_unaligned(d_b_row, g);
            device const uchar* d_ptr = d_w_row + g * 32;
            uint v_base = g * 64;

            for (uint u = 0; u < 8; ++u) {
                uint d_val = uint(d_ptr[u * 4 + 0])
                           | (uint(d_ptr[u * 4 + 1]) << 8)
                           | (uint(d_ptr[u * 4 + 2]) << 16)
                           | (uint(d_ptr[u * 4 + 3]) << 24);
                uint vi = v_base + u * 8;
                for (uint k = 0; k < 8; ++k) {
                    float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                    exp_acc += dw * act_e[vi + k];
                }
            }
        }
        routed_acc += exp_acc * score;
    }

    // 2. Shared expert down projection
    device const uchar* sh_w_row = shared_dw + shared_dw_off + gid * 256;
    device const uchar* sh_s_row = shared_ds + shared_ds_off + gid * 16;
    device const uchar* sh_b_row = shared_db + shared_db_off + gid * 16;

    float sh_down_acc = 0.0f;
    for (uint g = 0; g < 8; ++g) {
        float ds = read_bf16_unaligned(sh_s_row, g);
        float db = read_bf16_unaligned(sh_b_row, g);
        device const uchar* d_ptr = sh_w_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint d_val = uint(d_ptr[u * 4 + 0])
                       | (uint(d_ptr[u * 4 + 1]) << 8)
                       | (uint(d_ptr[u * 4 + 2]) << 16)
                       | (uint(d_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                sh_down_acc += dw * shared_act[vi + k];
            }
        }
    }

    // 3. Shared gating sigmoid: shared_out = sh_down_acc * sigmoid(shared_coeff[0])
    float coeff_val = shared_coeff[0];
    float sig_coeff = 1.0f / (1.0f + exp(-coeff_val));
    float shared_val = sh_down_acc * sig_coeff;

    // 4. Final residual add: cur_out = attn_res + routed_acc + shared_val
    cur_out[gid] = attn_res[gid] + routed_acc + shared_val;
}

// -----------------------------------------------------------------------------
// STAGED SELECTED-EXPERT STREAMING MOE KERNELS (Fixed Reusable Arena)
// -----------------------------------------------------------------------------

kernel void qwen_shared_gate_up_silu_q4(
    device const uchar* shared_gw     [[buffer(0)]],
    device const uchar* shared_gs     [[buffer(1)]],
    device const uchar* shared_gb     [[buffer(2)]],
    device const uchar* shared_uw     [[buffer(3)]],
    device const uchar* shared_us     [[buffer(4)]],
    device const uchar* shared_ub     [[buffer(5)]],
    device const float* vec           [[buffer(6)]],
    device float* shared_act          [[buffer(7)]],
    constant uint& shared_gw_off      [[buffer(8)]],
    constant uint& shared_gs_off      [[buffer(9)]],
    constant uint& shared_gb_off      [[buffer(10)]],
    constant uint& shared_uw_off      [[buffer(11)]],
    constant uint& shared_us_off      [[buffer(12)]],
    constant uint& shared_ub_off      [[buffer(13)]],
    uint row                          [[thread_position_in_grid]]
) {
    if (row >= 512) return;
    device const uchar* gw_row = shared_gw + shared_gw_off + row * 1024;
    device const uchar* gs_row = shared_gs + shared_gs_off + row * 64;
    device const uchar* gb_row = shared_gb + shared_gb_off + row * 64;
    device const uchar* uw_row = shared_uw + shared_uw_off + row * 1024;
    device const uchar* us_row = shared_us + shared_us_off + row * 64;
    device const uchar* ub_row = shared_ub + shared_ub_off + row * 64;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint g = 0; g < 32; ++g) {
        float gs = read_bf16_unaligned(gs_row, g);
        float gb = read_bf16_unaligned(gb_row, g);
        float us = read_bf16_unaligned(us_row, g);
        float ub = read_bf16_unaligned(ub_row, g);
        device const uchar* g_ptr = gw_row + g * 32;
        device const uchar* u_ptr = uw_row + g * 32;
        uint v_base = g * 64;
        for (uint u = 0; u < 8; ++u) {
            uint g_val = uint(g_ptr[u * 4 + 0])
                       | (uint(g_ptr[u * 4 + 1]) << 8)
                       | (uint(g_ptr[u * 4 + 2]) << 16)
                       | (uint(g_ptr[u * 4 + 3]) << 24);
            uint u_val = uint(u_ptr[u * 4 + 0])
                       | (uint(u_ptr[u * 4 + 1]) << 8)
                       | (uint(u_ptr[u * 4 + 2]) << 16)
                       | (uint(u_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float value = vec[vi + k];
                gate_acc += (float((g_val >> (k * 4)) & 0x0F) * gs + gb) * value;
                up_acc += (float((u_val >> (k * 4)) & 0x0F) * us + ub) * value;
            }
        }
    }
    shared_act[row] = (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
}

kernel void qwen_staged_moe_gate_up_silu_q4(
    device const uchar* staged_arena   [[buffer(0)]], // [8 * 1769472 bytes]
    device const uchar* shared_gw     [[buffer(1)]],
    device const uchar* shared_gs     [[buffer(2)]],
    device const uchar* shared_gb     [[buffer(3)]],
    device const uchar* shared_uw     [[buffer(4)]],
    device const uchar* shared_us     [[buffer(5)]],
    device const uchar* shared_ub     [[buffer(6)]],
    device const float* vec           [[buffer(7)]], // [2048]
    device float* batched_act         [[buffer(8)]], // [8 * 512]
    device float* shared_act          [[buffer(9)]], // [512]
    constant uint& shared_gw_off      [[buffer(10)]],
    constant uint& shared_gs_off      [[buffer(11)]],
    constant uint& shared_gb_off      [[buffer(12)]],
    constant uint& shared_uw_off      [[buffer(13)]],
    constant uint& shared_us_off      [[buffer(14)]],
    constant uint& shared_ub_off      [[buffer(15)]],
    uint gid                          [[thread_position_in_grid]] // 0..<4608
) {
    if (gid >= 4608) return;

    device const uchar* gw_row;
    device const uchar* gs_row;
    device const uchar* gb_row;
    device const uchar* uw_row;
    device const uchar* us_row;
    device const uchar* ub_row;

    if (gid < 4096) {
        uint e = gid / 512;
        uint r = gid % 512;
        device const uchar* exp_base = staged_arena + e * 1769472;

        gw_row = exp_base + 0 + r * 1024;
        gs_row = exp_base + 524288 + r * 64;
        gb_row = exp_base + 557056 + r * 64;
        uw_row = exp_base + 589824 + r * 1024;
        us_row = exp_base + 1114112 + r * 64;
        ub_row = exp_base + 1146880 + r * 64;
    } else {
        uint r = gid - 4096;
        gw_row = shared_gw + shared_gw_off + r * 1024;
        gs_row = shared_gs + shared_gs_off + r * 64;
        gb_row = shared_gb + shared_gb_off + r * 64;
        uw_row = shared_uw + shared_uw_off + r * 1024;
        us_row = shared_us + shared_us_off + r * 64;
        ub_row = shared_ub + shared_ub_off + r * 64;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint g = 0; g < 32; ++g) {
        float gs = read_bf16_unaligned(gs_row, g);
        float gb = read_bf16_unaligned(gb_row, g);
        float us = read_bf16_unaligned(us_row, g);
        float ub = read_bf16_unaligned(ub_row, g);

        device const uchar* g_ptr = gw_row + g * 32;
        device const uchar* u_ptr = uw_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint g_val = uint(g_ptr[u * 4 + 0])
                       | (uint(g_ptr[u * 4 + 1]) << 8)
                       | (uint(g_ptr[u * 4 + 2]) << 16)
                       | (uint(g_ptr[u * 4 + 3]) << 24);
            uint u_val = uint(u_ptr[u * 4 + 0])
                       | (uint(u_ptr[u * 4 + 1]) << 8)
                       | (uint(u_ptr[u * 4 + 2]) << 16)
                       | (uint(u_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;

            for (uint k = 0; k < 8; ++k) {
                float gw = float((g_val >> (k * 4)) & 0x0F) * gs + gb;
                float uw = float((u_val >> (k * 4)) & 0x0F) * us + ub;
                float v = vec[vi + k];
                gate_acc += gw * v;
                up_acc += uw * v;
            }
        }
    }

    float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    float act_val = silu_gate * up_acc;

    if (gid < 4096) {
        batched_act[gid] = act_val;
    } else {
        shared_act[gid - 4096] = act_val;
    }
}

kernel void qwen_staged_moe_down_accum_residual_q4(
    device const uchar* staged_arena   [[buffer(0)]], // [8 * 1769472 bytes]
    device const uchar* shared_dw     [[buffer(1)]],
    device const uchar* shared_ds     [[buffer(2)]],
    device const uchar* shared_db     [[buffer(3)]],
    device const float* batched_act   [[buffer(4)]], // [8 * 512]
    device const float* shared_act    [[buffer(5)]], // [512]
    device const float* scores        [[buffer(6)]], // [8]
    device const float* shared_coeff  [[buffer(7)]], // [1]
    device const float* attn_res      [[buffer(8)]], // [2048]
    device float* cur_out             [[buffer(9)]], // [2048]
    constant uint& shared_dw_off      [[buffer(10)]],
    constant uint& shared_ds_off      [[buffer(11)]],
    constant uint& shared_db_off      [[buffer(12)]],
    uint gid                          [[thread_position_in_grid]] // 0..<2048
) {
    if (gid >= 2048) return;

    // 1. Routed down projection across 8 staged experts
    float routed_acc = 0.0f;
    for (uint e = 0; e < 8; ++e) {
        float score = scores[e];
        if (score <= 0.0f) continue;
        device const uchar* exp_base = staged_arena + e * 1769472;

        device const float* act_e = batched_act + e * 512;
        device const uchar* d_w_row = exp_base + 1179648 + gid * 256;
        device const uchar* d_s_row = exp_base + 1703936 + gid * 16;
        device const uchar* d_b_row = exp_base + 1736704 + gid * 16;

        float exp_acc = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            float ds = read_bf16_unaligned(d_s_row, g);
            float db = read_bf16_unaligned(d_b_row, g);
            device const uchar* d_ptr = d_w_row + g * 32;
            uint v_base = g * 64;

            for (uint u = 0; u < 8; ++u) {
                uint d_val = uint(d_ptr[u * 4 + 0])
                           | (uint(d_ptr[u * 4 + 1]) << 8)
                           | (uint(d_ptr[u * 4 + 2]) << 16)
                           | (uint(d_ptr[u * 4 + 3]) << 24);
                uint vi = v_base + u * 8;
                for (uint k = 0; k < 8; ++k) {
                    float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                    exp_acc += dw * act_e[vi + k];
                }
            }
        }
        routed_acc += exp_acc * score;
    }

    // 2. Shared expert down projection
    device const uchar* sh_w_row = shared_dw + shared_dw_off + gid * 256;
    device const uchar* sh_s_row = shared_ds + shared_ds_off + gid * 16;
    device const uchar* sh_b_row = shared_db + shared_db_off + gid * 16;

    float sh_down_acc = 0.0f;
    for (uint g = 0; g < 8; ++g) {
        float ds = read_bf16_unaligned(sh_s_row, g);
        float db = read_bf16_unaligned(sh_b_row, g);
        device const uchar* d_ptr = sh_w_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint d_val = uint(d_ptr[u * 4 + 0])
                       | (uint(d_ptr[u * 4 + 1]) << 8)
                       | (uint(d_ptr[u * 4 + 2]) << 16)
                       | (uint(d_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                sh_down_acc += dw * shared_act[vi + k];
            }
        }
    }

    // 3. Shared gating sigmoid: shared_out = sh_down_acc * sigmoid(shared_coeff[0])
    float coeff_val = shared_coeff[0];
    float sig_coeff = 1.0f / (1.0f + exp(-coeff_val));
    float shared_val = sh_down_acc * sig_coeff;

    // 4. Final residual add: cur_out = attn_res + routed_acc + shared_val
    cur_out[gid] = attn_res[gid] + routed_acc + shared_val;
}

kernel void qwen_cached_moe_gate_up_silu_q4(
    device const uchar* cache_arena   [[buffer(0)]], // [capacity * 1769472]
    device const uchar* shared_gw     [[buffer(1)]],
    device const uchar* shared_gs     [[buffer(2)]],
    device const uchar* shared_gb     [[buffer(3)]],
    device const uchar* shared_uw     [[buffer(4)]],
    device const uchar* shared_us     [[buffer(5)]],
    device const uchar* shared_ub     [[buffer(6)]],
    device const float* vec           [[buffer(7)]], // [2048]
    device float* batched_act         [[buffer(8)]], // [8 * 512]
    device float* shared_act          [[buffer(9)]], // [512]
    constant uint* slot_indices       [[buffer(10)]], // [8]
    constant uint& shared_gw_off      [[buffer(11)]],
    constant uint& shared_gs_off      [[buffer(12)]],
    constant uint& shared_gb_off      [[buffer(13)]],
    constant uint& shared_uw_off      [[buffer(14)]],
    constant uint& shared_us_off      [[buffer(15)]],
    constant uint& shared_ub_off      [[buffer(16)]],
    uint gid                          [[thread_position_in_grid]] // 0..<4608
) {
    if (gid >= 4608) return;

    device const uchar* gw_row;
    device const uchar* gs_row;
    device const uchar* gb_row;
    device const uchar* uw_row;
    device const uchar* us_row;
    device const uchar* ub_row;

    if (gid < 4096) {
        uint e = gid / 512;
        uint r = gid % 512;
        uint slot = slot_indices[e];
        device const uchar* exp_base = cache_arena + slot * 1769472;

        gw_row = exp_base + 0 + r * 1024;
        gs_row = exp_base + 524288 + r * 64;
        gb_row = exp_base + 557056 + r * 64;
        uw_row = exp_base + 589824 + r * 1024;
        us_row = exp_base + 1114112 + r * 64;
        ub_row = exp_base + 1146880 + r * 64;
    } else {
        uint r = gid - 4096;
        gw_row = shared_gw + shared_gw_off + r * 1024;
        gs_row = shared_gs + shared_gs_off + r * 64;
        gb_row = shared_gb + shared_gb_off + r * 64;
        uw_row = shared_uw + shared_uw_off + r * 1024;
        us_row = shared_us + shared_us_off + r * 64;
        ub_row = shared_ub + shared_ub_off + r * 64;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint g = 0; g < 32; ++g) {
        float gs = read_bf16_unaligned(gs_row, g);
        float gb = read_bf16_unaligned(gb_row, g);
        float us = read_bf16_unaligned(us_row, g);
        float ub = read_bf16_unaligned(ub_row, g);

        device const uchar* g_ptr = gw_row + g * 32;
        device const uchar* u_ptr = uw_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint g_val = uint(g_ptr[u * 4 + 0])
                       | (uint(g_ptr[u * 4 + 1]) << 8)
                       | (uint(g_ptr[u * 4 + 2]) << 16)
                       | (uint(g_ptr[u * 4 + 3]) << 24);
            uint u_val = uint(u_ptr[u * 4 + 0])
                       | (uint(u_ptr[u * 4 + 1]) << 8)
                       | (uint(u_ptr[u * 4 + 2]) << 16)
                       | (uint(u_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;

            for (uint k = 0; k < 8; ++k) {
                float gw = float((g_val >> (k * 4)) & 0x0F) * gs + gb;
                float uw = float((u_val >> (k * 4)) & 0x0F) * us + ub;
                float v = vec[vi + k];
                gate_acc += gw * v;
                up_acc += uw * v;
            }
        }
    }

    float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    float act_val = silu_gate * up_acc;

    if (gid < 4096) {
        batched_act[gid] = act_val;
    } else {
        shared_act[gid - 4096] = act_val;
    }
}

kernel void qwen_cached_moe_down_accum_residual_q4(
    device const uchar* cache_arena   [[buffer(0)]], // [capacity * 1769472]
    device const uchar* shared_dw     [[buffer(1)]],
    device const uchar* shared_ds     [[buffer(2)]],
    device const uchar* shared_db     [[buffer(3)]],
    device const float* batched_act   [[buffer(4)]], // [8 * 512]
    device const float* shared_act    [[buffer(5)]], // [512]
    device const float* scores        [[buffer(6)]], // [8]
    device const float* shared_coeff  [[buffer(7)]], // [1]
    device const float* attn_res      [[buffer(8)]], // [2048]
    device float* cur_out             [[buffer(9)]], // [2048]
    constant uint* slot_indices       [[buffer(10)]], // [8]
    constant uint& shared_dw_off      [[buffer(11)]],
    constant uint& shared_ds_off      [[buffer(12)]],
    constant uint& shared_db_off      [[buffer(13)]],
    uint gid                          [[thread_position_in_grid]] // 0..<2048
) {
    if (gid >= 2048) return;

    // 1. Routed down projection across 8 active cached experts
    float routed_acc = 0.0f;
    for (uint e = 0; e < 8; ++e) {
        float score = scores[e];
        if (score <= 0.0f) continue;
        uint slot = slot_indices[e];
        device const uchar* exp_base = cache_arena + slot * 1769472;

        device const float* act_e = batched_act + e * 512;
        device const uchar* d_w_row = exp_base + 1179648 + gid * 256;
        device const uchar* d_s_row = exp_base + 1703936 + gid * 16;
        device const uchar* d_b_row = exp_base + 1736704 + gid * 16;

        float exp_acc = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            float ds = read_bf16_unaligned(d_s_row, g);
            float db = read_bf16_unaligned(d_b_row, g);
            device const uchar* d_ptr = d_w_row + g * 32;
            uint v_base = g * 64;

            for (uint u = 0; u < 8; ++u) {
                uint d_val = uint(d_ptr[u * 4 + 0])
                           | (uint(d_ptr[u * 4 + 1]) << 8)
                           | (uint(d_ptr[u * 4 + 2]) << 16)
                           | (uint(d_ptr[u * 4 + 3]) << 24);
                uint vi = v_base + u * 8;
                for (uint k = 0; k < 8; ++k) {
                    float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                    exp_acc += dw * act_e[vi + k];
                }
            }
        }
        routed_acc += exp_acc * score;
    }

    // 2. Shared expert down projection
    device const uchar* sh_w_row = shared_dw + shared_dw_off + gid * 256;
    device const uchar* sh_s_row = shared_ds + shared_ds_off + gid * 16;
    device const uchar* sh_b_row = shared_db + shared_db_off + gid * 16;

    float sh_down_acc = 0.0f;
    for (uint g = 0; g < 8; ++g) {
        float ds = read_bf16_unaligned(sh_s_row, g);
        float db = read_bf16_unaligned(sh_b_row, g);
        device const uchar* d_ptr = sh_w_row + g * 32;
        uint v_base = g * 64;

        for (uint u = 0; u < 8; ++u) {
            uint d_val = uint(d_ptr[u * 4 + 0])
                       | (uint(d_ptr[u * 4 + 1]) << 8)
                       | (uint(d_ptr[u * 4 + 2]) << 16)
                       | (uint(d_ptr[u * 4 + 3]) << 24);
            uint vi = v_base + u * 8;
            for (uint k = 0; k < 8; ++k) {
                float dw = float((d_val >> (k * 4)) & 0x0F) * ds + db;
                sh_down_acc += dw * shared_act[vi + k];
            }
        }
    }

    // 3. Shared gating sigmoid: shared_out = sh_down_acc * sigmoid(shared_coeff[0])
    float coeff_val = shared_coeff[0];
    float sig_coeff = 1.0f / (1.0f + exp(-coeff_val));
    float shared_val = sh_down_acc * sig_coeff;

    // 4. Final residual add: cur_out = attn_res + routed_acc + shared_val
    cur_out[gid] = attn_res[gid] + routed_acc + shared_val;
}

// -----------------------------------------------------------------------------
// SIMD-COOPERATIVE Q4 DECODE KERNELS
// -----------------------------------------------------------------------------

inline float qwen_q4_row_simd_f32(
    device const uchar* raw_weight,
    device const uchar* raw_scales,
    device const uchar* raw_biases,
    device const float* vec,
    uint row,
    uint cols,
    uint w_byte_offset,
    uint s_byte_offset,
    uint b_byte_offset,
    uint lane
) {
    const uint groups = cols / 64;
    device const uchar* w_row = raw_weight + w_byte_offset + row * (cols / 2);
    device const uchar* s_row = raw_scales + s_byte_offset + row * (groups * 2);
    device const uchar* b_row = raw_biases + b_byte_offset + row * (groups * 2);
    float acc = 0.0f;
    // Process four quantization groups per iteration. Eight lanes cooperate on
    // each group, so every lane loads four packed bytes and eight contiguous
    // activations instead of reloading scale/bias for every single byte.
    const uint full_blocks = groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = uint(w_row[byte_base])
                      | (uint(w_row[byte_base + 1u]) << 8)
                      | (uint(w_row[byte_base + 2u]) << 16)
                      | (uint(w_row[byte_base + 3u]) << 24);
        const uint g = blk * 4u + (lane >> 3);
        const float scale = read_bf16_unaligned(s_row, g);
        const float bias = read_bf16_unaligned(b_row, g);
        const uint elem = byte_base * 2u;
        const float e0 = vec[elem]; const float e1 = vec[elem + 1u];
        const float e2 = vec[elem + 2u]; const float e3 = vec[elem + 3u];
        const float e4 = vec[elem + 4u]; const float e5 = vec[elem + 5u];
        const float e6 = vec[elem + 6u]; const float e7 = vec[elem + 7u];
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(scale, dot, acc);
        acc = fma(bias, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < groups; ++g) {
        const uchar packed = w_row[g * 32u + lane];
        const float x0 = vec[g * 64u + lane * 2u];
        const float x1 = vec[g * 64u + lane * 2u + 1u];
        const float dot = fma(float(packed & 0x0Fu), x0, float(packed >> 4) * x1);
        acc = fma(read_bf16_unaligned(s_row, g), dot, acc);
        acc = fma(read_bf16_unaligned(b_row, g), x0 + x1, acc);
    }
    return simd_sum(acc);
}

kernel void qwen_gemv_q4_affine_simd_f32(
    device const uchar* raw_weight  [[buffer(0)]],
    device const uchar* raw_scales  [[buffer(1)]],
    device const uchar* raw_biases  [[buffer(2)]],
    device const float* vec         [[buffer(3)]],
    device float* out               [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& w_byte_offset    [[buffer(7)]],
    constant uint& s_byte_offset    [[buffer(8)]],
    constant uint& b_byte_offset    [[buffer(9)]],
    uint tg_idx                     [[threadgroup_position_in_grid]],
    uint sg_idx                     [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= rows) return;
    const float value = qwen_q4_row_simd_f32(
        raw_weight, raw_scales, raw_biases, vec, row, cols,
        w_byte_offset, s_byte_offset, b_byte_offset, lane);
    if (lane == 0) out[row] = value;
}

kernel void qwen_fused_linear_attn_in_proj_q4_simd_f32(
    device const uchar* qkv_w     [[buffer(0)]],
    device const uchar* qkv_s     [[buffer(1)]],
    device const uchar* qkv_b     [[buffer(2)]],
    device const uchar* z_w       [[buffer(3)]],
    device const uchar* z_s       [[buffer(4)]],
    device const uchar* z_b       [[buffer(5)]],
    device const uchar* b_w       [[buffer(6)]],
    device const uchar* b_s       [[buffer(7)]],
    device const uchar* b_b       [[buffer(8)]],
    device const uchar* a_w       [[buffer(9)]],
    device const uchar* a_s       [[buffer(10)]],
    device const uchar* a_b       [[buffer(11)]],
    device const float* vec       [[buffer(12)]],
    device float* qkv_out         [[buffer(13)]],
    device float* z_out           [[buffer(14)]],
    device float* b_out           [[buffer(15)]],
    device float* a_out           [[buffer(16)]],
    constant uint& qkv_w_off      [[buffer(17)]],
    constant uint& qkv_s_off      [[buffer(18)]],
    constant uint& qkv_b_off      [[buffer(19)]],
    constant uint& z_w_off        [[buffer(20)]],
    constant uint& z_s_off        [[buffer(21)]],
    constant uint& z_b_off        [[buffer(22)]],
    constant uint& b_w_off        [[buffer(23)]],
    constant uint& b_s_off        [[buffer(24)]],
    constant uint& b_b_off        [[buffer(25)]],
    constant uint& a_w_off        [[buffer(26)]],
    constant uint& a_s_off        [[buffer(27)]],
    constant uint& a_b_off        [[buffer(28)]],
    uint tg_idx                   [[threadgroup_position_in_grid]],
    uint sg_idx                   [[simdgroup_index_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    if (global_row >= 12352) return;

    device const uchar* raw_w;
    device const uchar* raw_s;
    device const uchar* raw_b;
    device float* output;
    uint row;
    uint w_off;
    uint s_off;
    uint b_off;
    if (global_row < 8192) {
        raw_w = qkv_w; raw_s = qkv_s; raw_b = qkv_b; output = qkv_out;
        row = global_row; w_off = qkv_w_off; s_off = qkv_s_off; b_off = qkv_b_off;
    } else if (global_row < 12288) {
        raw_w = z_w; raw_s = z_s; raw_b = z_b; output = z_out;
        row = global_row - 8192; w_off = z_w_off; s_off = z_s_off; b_off = z_b_off;
    } else if (global_row < 12320) {
        raw_w = b_w; raw_s = b_s; raw_b = b_b; output = b_out;
        row = global_row - 12288; w_off = b_w_off; s_off = b_s_off; b_off = b_b_off;
    } else {
        raw_w = a_w; raw_s = a_s; raw_b = a_b; output = a_out;
        row = global_row - 12320; w_off = a_w_off; s_off = a_s_off; b_off = a_b_off;
    }
    const float value = qwen_q4_row_simd_f32(
        raw_w, raw_s, raw_b, vec, row, 2048, w_off, s_off, b_off, lane);
    if (lane == 0) output[row] = value;
}

inline float2 qwen_q4_gate_up_row_simd_f32(
    device const uchar* gate_w,
    device const uchar* gate_s,
    device const uchar* gate_b,
    device const uchar* up_w,
    device const uchar* up_s,
    device const uchar* up_b,
    device const float* vec,
    uint row,
    uint lane
) {
    device const uchar* gw_row = gate_w + row * 1024;
    device const uchar* gs_row = gate_s + row * 64;
    device const uchar* gb_row = gate_b + row * 64;
    device const uchar* uw_row = up_w + row * 1024;
    device const uchar* us_row = up_s + row * 64;
    device const uchar* ub_row = up_b + row * 64;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint blk = 0; blk < 8; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint gw4 = uint(gw_row[byte_base])
                       | (uint(gw_row[byte_base + 1u]) << 8)
                       | (uint(gw_row[byte_base + 2u]) << 16)
                       | (uint(gw_row[byte_base + 3u]) << 24);
        const uint uw4 = uint(uw_row[byte_base])
                       | (uint(uw_row[byte_base + 1u]) << 8)
                       | (uint(uw_row[byte_base + 2u]) << 16)
                       | (uint(uw_row[byte_base + 3u]) << 24);
        const uint g = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;
        const float e0 = vec[elem]; const float e1 = vec[elem + 1u];
        const float e2 = vec[elem + 2u]; const float e3 = vec[elem + 3u];
        const float e4 = vec[elem + 4u]; const float e5 = vec[elem + 5u];
        const float e6 = vec[elem + 6u]; const float e7 = vec[elem + 7u];
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float gate_dot = 0.0f;
        gate_dot = fma(float(gb0 & 0x0Fu), e0, gate_dot); gate_dot = fma(float(gb0 >> 4), e1, gate_dot);
        gate_dot = fma(float(gb1 & 0x0Fu), e2, gate_dot); gate_dot = fma(float(gb1 >> 4), e3, gate_dot);
        gate_dot = fma(float(gb2 & 0x0Fu), e4, gate_dot); gate_dot = fma(float(gb2 >> 4), e5, gate_dot);
        gate_dot = fma(float(gb3 & 0x0Fu), e6, gate_dot); gate_dot = fma(float(gb3 >> 4), e7, gate_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float up_dot = 0.0f;
        up_dot = fma(float(ub0 & 0x0Fu), e0, up_dot); up_dot = fma(float(ub0 >> 4), e1, up_dot);
        up_dot = fma(float(ub1 & 0x0Fu), e2, up_dot); up_dot = fma(float(ub1 >> 4), e3, up_dot);
        up_dot = fma(float(ub2 & 0x0Fu), e4, up_dot); up_dot = fma(float(ub2 >> 4), e5, up_dot);
        up_dot = fma(float(ub3 & 0x0Fu), e6, up_dot); up_dot = fma(float(ub3 >> 4), e7, up_dot);

        gate_acc = fma(read_bf16_unaligned(gs_row, g), gate_dot, gate_acc);
        gate_acc = fma(read_bf16_unaligned(gb_row, g), sum, gate_acc);
        up_acc = fma(read_bf16_unaligned(us_row, g), up_dot, up_acc);
        up_acc = fma(read_bf16_unaligned(ub_row, g), sum, up_acc);
    }
    return float2(simd_sum(gate_acc), simd_sum(up_acc));
}

kernel void qwen_shared_gate_up_silu_q4_simd_f32(
    device const uchar* shared_gw     [[buffer(0)]],
    device const uchar* shared_gs     [[buffer(1)]],
    device const uchar* shared_gb     [[buffer(2)]],
    device const uchar* shared_uw     [[buffer(3)]],
    device const uchar* shared_us     [[buffer(4)]],
    device const uchar* shared_ub     [[buffer(5)]],
    device const float* vec           [[buffer(6)]],
    device float* shared_act          [[buffer(7)]],
    constant uint& shared_gw_off      [[buffer(8)]],
    constant uint& shared_gs_off      [[buffer(9)]],
    constant uint& shared_gb_off      [[buffer(10)]],
    constant uint& shared_uw_off      [[buffer(11)]],
    constant uint& shared_us_off      [[buffer(12)]],
    constant uint& shared_ub_off      [[buffer(13)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint sg_idx                      [[simdgroup_index_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8 + sg_idx;
    if (row >= 512) return;
    const float2 gu = qwen_q4_gate_up_row_simd_f32(
        shared_gw + shared_gw_off, shared_gs + shared_gs_off, shared_gb + shared_gb_off,
        shared_uw + shared_uw_off, shared_us + shared_us_off, shared_ub + shared_ub_off,
        vec, row, lane);
    if (lane == 0) shared_act[row] = (gu.x / (1.0f + exp(-gu.x))) * gu.y;
}

kernel void qwen_cached_moe_gate_up_silu_q4_simd_f32(
    device const uchar* cache_arena   [[buffer(0)]],
    device const float* vec           [[buffer(1)]],
    device float* batched_act         [[buffer(2)]],
    constant uint* slot_indices       [[buffer(3)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint sg_idx                      [[simdgroup_index_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]]
) {
    const uint global_row = tg_idx * 8 + sg_idx;
    if (global_row >= 4096) return;
    const uint expert = global_row / 512;
    const uint row = global_row % 512;
    device const uchar* base = cache_arena + slot_indices[expert] * 1769472;
    const float2 gu = qwen_q4_gate_up_row_simd_f32(
        base, base + 524288, base + 557056,
        base + 589824, base + 1114112, base + 1146880,
        vec, row, lane);
    if (lane == 0) batched_act[global_row] = (gu.x / (1.0f + exp(-gu.x))) * gu.y;
}

kernel void qwen_shared_down_gate_q4_simd_f32(
    device const uchar* shared_dw     [[buffer(0)]],
    device const uchar* shared_ds     [[buffer(1)]],
    device const uchar* shared_db     [[buffer(2)]],
    device const float* shared_act    [[buffer(3)]],
    device const float* shared_coeff  [[buffer(4)]],
    device float* shared_out          [[buffer(5)]],
    constant uint& shared_dw_off      [[buffer(6)]],
    constant uint& shared_ds_off      [[buffer(7)]],
    constant uint& shared_db_off      [[buffer(8)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint sg_idx                      [[simdgroup_index_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8 + sg_idx;
    if (row >= 2048) return;
    const float value = qwen_q4_row_simd_f32(
        shared_dw, shared_ds, shared_db, shared_act, row, 512,
        shared_dw_off, shared_ds_off, shared_db_off, lane);
    if (lane == 0) {
        const float coeff = 1.0f / (1.0f + exp(-shared_coeff[0]));
        shared_out[row] = value * coeff;
    }
}

kernel void qwen_cached_moe_down_residual_q4_simd_f32(
    device const uchar* cache_arena   [[buffer(0)]],
    device const float* batched_act   [[buffer(1)]],
    device const float* scores        [[buffer(2)]],
    device const float* shared_out    [[buffer(3)]],
    device const float* attn_res      [[buffer(4)]],
    device float* cur_out             [[buffer(5)]],
    constant uint* slot_indices       [[buffer(6)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint sg_idx                       [[simdgroup_index_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    if (row >= 2048) return;
    const uint slot = slot_indices[sg_idx];
    device const uchar* base = cache_arena + slot * 1769472;
    const float value = qwen_q4_row_simd_f32(
        base + 1179648, base + 1703936, base + 1736704,
        batched_act + sg_idx * 512, row, 512, 0, 0, 0, lane);
    if (lane == 0) partial[sg_idx] = value * scores[sg_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0 && lane == 0) {
        float routed = partial[0] + partial[1] + partial[2] + partial[3]
                     + partial[4] + partial[5] + partial[6] + partial[7];
        cur_out[row] = attn_res[row] + routed + shared_out[row];
    }
}

// Subset variants let resident expert work run while cache misses are being
// filled by the CPU. `expert_positions` indexes the router's top-8 order, while
// `slot_indices` maps that position to the bounded cache arena.
kernel void qwen_cached_moe_gate_up_subset_q4_simd_f32(
    device const uchar* cache_arena       [[buffer(0)]],
    device const float* vec               [[buffer(1)]],
    device float* batched_act             [[buffer(2)]],
    constant uint* slot_indices           [[buffer(3)]],
    constant uint* expert_positions       [[buffer(4)]],
    constant uint& expert_count           [[buffer(5)]],
    uint tg_idx                           [[threadgroup_position_in_grid]],
    uint sg_idx                           [[simdgroup_index_in_threadgroup]],
    uint lane                             [[thread_index_in_simdgroup]]
) {
    const uint subset_row = tg_idx * 8 + sg_idx;
    if (subset_row >= expert_count * 512) return;
    const uint subset_expert = subset_row / 512;
    const uint row = subset_row % 512;
    const uint position = expert_positions[subset_expert];
    device const uchar* base = cache_arena + slot_indices[position] * 1769472;
    const float2 gu = qwen_q4_gate_up_row_simd_f32(
        base, base + 524288, base + 557056,
        base + 589824, base + 1114112, base + 1146880,
        vec, row, lane);
    if (lane == 0) {
        batched_act[position * 512 + row] = (gu.x / (1.0f + exp(-gu.x))) * gu.y;
    }
}

kernel void qwen_cached_moe_down_subset_q4_simd_f32(
    device const uchar* cache_arena       [[buffer(0)]],
    device const float* batched_act       [[buffer(1)]],
    device const float* scores            [[buffer(2)]],
    device float* routed_out              [[buffer(3)]],
    constant uint* slot_indices           [[buffer(4)]],
    constant uint* expert_positions       [[buffer(5)]],
    constant uint& expert_count           [[buffer(6)]],
    uint row                              [[threadgroup_position_in_grid]],
    uint sg_idx                           [[simdgroup_index_in_threadgroup]],
    uint lane                             [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    if (row >= 2048) return;
    float weighted = 0.0f;
    if (sg_idx < expert_count) {
        const uint position = expert_positions[sg_idx];
        const uint slot = slot_indices[position];
        device const uchar* base = cache_arena + slot * 1769472;
        const float value = qwen_q4_row_simd_f32(
            base + 1179648, base + 1703936, base + 1736704,
            batched_act + position * 512, row, 512, 0, 0, 0, lane);
        if (lane == 0) weighted = value * scores[position];
    }
    if (lane == 0) partial[sg_idx] = weighted;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0 && lane == 0) {
        routed_out[row] = partial[0] + partial[1] + partial[2] + partial[3]
                        + partial[4] + partial[5] + partial[6] + partial[7];
    }
}

kernel void qwen_combine_cached_moe_subsets_f32(
    device const float* hit_out       [[buffer(0)]],
    device const float* miss_out      [[buffer(1)]],
    device const float* shared_out    [[buffer(2)]],
    device const float* attn_res      [[buffer(3)]],
    device float* cur_out             [[buffer(4)]],
    constant uint& hit_count          [[buffer(5)]],
    constant uint& miss_count         [[buffer(6)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= 2048) return;
    const float hit = hit_count > 0 ? hit_out[gid] : 0.0f;
    const float miss = miss_count > 0 ? miss_out[gid] : 0.0f;
    cur_out[gid] = attn_res[gid] + shared_out[gid] + hit + miss;
}

inline float qwen_q8_row_simd_f32(
    device const uchar* raw_weight,
    device const uchar* raw_scales,
    device const uchar* raw_biases,
    device const float* vec,
    uint row,
    uint w_byte_offset,
    uint s_byte_offset,
    uint b_byte_offset,
    uint lane
) {
    device const uchar* w_row = raw_weight + w_byte_offset + row * 2048;
    device const uchar* s_row = raw_scales + s_byte_offset + row * 64;
    device const uchar* b_row = raw_biases + b_byte_offset + row * 64;
    float acc = 0.0f;
    for (uint blk = 0; blk < 8; ++blk) {
        const uint byte_base = blk * 256u + lane * 8u;
        const uint g = blk * 4u + (lane >> 3);
        const float scale = read_bf16_unaligned(s_row, g);
        const float bias = read_bf16_unaligned(b_row, g);
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const float x = vec[byte_base + i];
            dot = fma(float(w_row[byte_base + i]), x, dot);
            sum += x;
        }
        acc = fma(scale, dot, acc);
        acc = fma(bias, sum, acc);
    }
    return simd_sum(acc);
}

kernel void qwen_router_q8_logits_simd_f32(
    device const uchar* raw_weight  [[buffer(0)]],
    device const uchar* raw_scales  [[buffer(1)]],
    device const uchar* raw_biases  [[buffer(2)]],
    device const float* vec         [[buffer(3)]],
    device float* logits            [[buffer(4)]],
    constant uint& w_byte_offset    [[buffer(5)]],
    constant uint& s_byte_offset    [[buffer(6)]],
    constant uint& b_byte_offset    [[buffer(7)]],
    uint tg_idx                     [[threadgroup_position_in_grid]],
    uint sg_idx                     [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8 + sg_idx;
    if (row >= 256) return;
    const float value = qwen_q8_row_simd_f32(
        raw_weight, raw_scales, raw_biases, vec, row,
        w_byte_offset, s_byte_offset, b_byte_offset, lane);
    if (lane == 0) logits[row] = value;
}

kernel void qwen_router_top8_logits_f32(
    device const float* logits       [[buffer(0)]],
    device uint* expert_indices      [[buffer(1)]],
    device float* routing_scores     [[buffer(2)]],
    uint tid                         [[thread_position_in_threadgroup]]
) {
    threadgroup float values[256];
    values[tid] = logits[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;

    uint top_idx[8];
    float top_val[8];
    for (uint k = 0; k < 8; ++k) {
        top_idx[k] = 0;
        top_val[k] = -INFINITY;
    }
    for (uint i = 0; i < 256; ++i) {
        const float value = values[i];
        if (value > top_val[7]) {
            top_val[7] = value;
            top_idx[7] = i;
            for (int k = 6; k >= 0; --k) {
                if (top_val[k + 1] <= top_val[k]) break;
                const float swap_value = top_val[k];
                top_val[k] = top_val[k + 1];
                top_val[k + 1] = swap_value;
                const uint swap_index = top_idx[k];
                top_idx[k] = top_idx[k + 1];
                top_idx[k + 1] = swap_index;
            }
        }
    }
    const float max_value = top_val[0];
    float denominator = 0.0f;
    for (uint i = 0; i < 256; ++i) denominator += exp(values[i] - max_value);
    float top_sum = 0.0f;
    float top_prob[8];
    for (uint k = 0; k < 8; ++k) {
        top_prob[k] = exp(top_val[k] - max_value) / denominator;
        top_sum += top_prob[k];
    }
    for (uint k = 0; k < 8; ++k) {
        expert_indices[k] = top_idx[k];
        routing_scores[k] = top_prob[k] / top_sum;
    }
}
