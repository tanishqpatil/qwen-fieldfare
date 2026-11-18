#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint MoE Sigmoid Top-6 Router Kernel
//
// Mathematical Specification:
//   - 47 MoE layers (layers 2 through 48; layers 0 and 1 are dense).
//   - Hidden dimension: 2048.
//   - Total experts: 256.
//   - Top-K selected experts: 6.
//   - Resident router gate weight tensor: BF16 [256, 2048].
//   - Scoring function: sigmoid(logits) where logits = x @ W_gate^T.
//   - Top-6 selection: 6 largest sigmoid scores, ordered descending by score.
//   - Documented tie-breaking policy: If two experts have equal scores, the one
//     with the lower expert index (0 .. 255) is selected first.
//   - Weight normalization: w_k = score_k / (sum_{m=0..5} score_m + 1e-20f).
//   - Zero GPU heap allocations, zero CPU roundtrips.
// ==============================================================================

/// Router dispatch for bfloat16 activations.
/// Grid dispatch: threadsPerThreadgroup = (256, 1, 1), threadgroups = (num_tokens, 1, batch).
kernel void flint_router_forward_bf16(
    device const bfloat* in_x [[buffer(0)]],        // [batch, num_tokens, 2048]
    device const bfloat* gate_weight [[buffer(1)]], // [256, 2048] resident router gate weights
    device uint* out_experts [[buffer(2)]],         // [batch, num_tokens, 6] expert indices (0 .. 255)
    device float* out_weights [[buffer(3)]],        // [batch, num_tokens, 6] normalized routing weights
    constant uint& num_tokens [[buffer(4)]],        // Number of tokens in sequence
    uint3 tid [[thread_position_in_threadgroup]],   // (e: 0 .. 255, 0, 0)
    uint3 pos_in_grid [[threadgroup_position_in_grid]] // (t, 0, b)
) {
    uint e = tid.x; // Expert ID (0 .. 255)
    uint t = pos_in_grid.x; // Token index
    uint b = pos_in_grid.z; // Batch index

    if (e >= 256 || t >= num_tokens) {
        return;
    }

    threadgroup float shared_scores[256];

    // Pointer to token activation vector [2048]
    uint token_offset = b * (num_tokens * 2048) + t * 2048;
    device const bfloat* x_row = in_x + token_offset;

    // Pointer to expert e weight row [2048]
    device const bfloat* w_row = gate_weight + e * 2048;

    // Compute dot product x @ w_e in FP32
    float acc = 0.0f;
    for (uint i = 0; i < 2048; i += 4) {
        acc += float(x_row[i + 0]) * float(w_row[i + 0]);
        acc += float(x_row[i + 1]) * float(w_row[i + 1]);
        acc += float(x_row[i + 2]) * float(w_row[i + 2]);
        acc += float(x_row[i + 3]) * float(w_row[i + 3]);
    }

    // Apply sigmoid activation function: score = 1 / (1 + exp(-acc))
    float score = 1.0f / (1.0f + exp(-acc));
    shared_scores[e] = score;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Single-thread deterministic top-6 selection and weight normalization
    if (e == 0) {
        float top_scores[6];
        uint top_experts[6];

        for (uint k = 0; k < 6; ++k) {
            float max_val = -1e30f;
            uint best_idx = 0;
            for (uint i = 0; i < 256; ++i) {
                float val = shared_scores[i];
                if (val > max_val) {
                    max_val = val;
                    best_idx = i;
                }
            }
            top_scores[k] = max_val;
            top_experts[k] = best_idx;
            shared_scores[best_idx] = -1e30f; // Exclude selected expert from subsequent rounds
        }

        // Normalize the 6 selected scores
        float sum = 1e-20f;
        for (uint k = 0; k < 6; ++k) {
            sum += top_scores[k];
        }

        uint token_out_offset = (b * num_tokens + t) * 6;
        for (uint k = 0; k < 6; ++k) {
            out_experts[token_out_offset + k] = top_experts[k];
            out_weights[token_out_offset + k] = top_scores[k] / sum;
        }
    }
}

/// Router dispatch for float32 activations with bfloat16 resident gate weights.
kernel void flint_router_forward_f32(
    device const float* in_x [[buffer(0)]],
    device const bfloat* gate_weight [[buffer(1)]],
    device uint* out_experts [[buffer(2)]],
    device float* out_weights [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 pos_in_grid [[threadgroup_position_in_grid]]
) {
    uint e = tid.x;
    uint t = pos_in_grid.x;
    uint b = pos_in_grid.z;

    if (e >= 256 || t >= num_tokens) {
        return;
    }

    threadgroup float shared_scores[256];

    uint token_offset = b * (num_tokens * 2048) + t * 2048;
    device const float* x_row = in_x + token_offset;
    device const bfloat* w_row = gate_weight + e * 2048;

    float acc = 0.0f;
    for (uint i = 0; i < 2048; i += 4) {
        acc += x_row[i + 0] * float(w_row[i + 0]);
        acc += x_row[i + 1] * float(w_row[i + 1]);
        acc += x_row[i + 2] * float(w_row[i + 2]);
        acc += x_row[i + 3] * float(w_row[i + 3]);
    }

    float score = 1.0f / (1.0f + exp(-acc));
    shared_scores[e] = score;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (e == 0) {
        float top_scores[6];
        uint top_experts[6];

        for (uint k = 0; k < 6; ++k) {
            float max_val = -1e30f;
            uint best_idx = 0;
            for (uint i = 0; i < 256; ++i) {
                float val = shared_scores[i];
                if (val > max_val) {
                    max_val = val;
                    best_idx = i;
                }
            }
            top_scores[k] = max_val;
            top_experts[k] = best_idx;
            shared_scores[best_idx] = -1e30f;
        }

        float sum = 1e-20f;
        for (uint k = 0; k < 6; ++k) {
            sum += top_scores[k];
        }

        uint token_out_offset = (b * num_tokens + t) * 6;
        for (uint k = 0; k < 6; ++k) {
            out_experts[token_out_offset + k] = top_experts[k];
            out_weights[token_out_offset + k] = top_scores[k] / sum;
        }
    }
}
