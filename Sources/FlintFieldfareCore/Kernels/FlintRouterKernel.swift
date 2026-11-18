import Foundation
import Metal

/// Native Metal compute kernel for Flint MoE Sigmoid Top-6 Router.
///
/// Mathematical Specification:
/// - 47 MoE layers (layers 2 through 48; layers 0 and 1 are dense).
/// - Hidden dimension: 2048.
/// - Total experts: 256.
/// - Top-K selected experts: 6.
/// - Resident router gate weight tensor: BF16 `[256, 2048]` (524,288 elements = 1,048,576 bytes).
/// - Scoring function: `sigmoid(logits)` where `logits = x @ W_gate^T`.
/// - Top-6 selection: 6 largest sigmoid scores, ordered descending by score.
/// - Documented tie-breaking policy: If two experts have equal scores, the one
///   with the lower expert index (0 .. 255) is selected first.
/// - Weight normalization: `w_k = score_k / (sum_{m=0..5} score_m + 1e-20f)` (sum of 6 weights = 1.0).
/// - Zero GPU heap allocations, zero CPU roundtrips.
public final class FlintRouterKernel: @unchecked Sendable {
    public enum Dtype: String, Sendable, CaseIterable {
        case bfloat16
        case float32

        public var elementByteWidth: Int {
            switch self {
            case .bfloat16: return 2
            case .float32: return 4
            }
        }
    }

    public let context: FlintMetalContext
    private let psoBF16: MTLComputePipelineState
    private let psoF32: MTLComputePipelineState

    private static func checkedMul(_ a: Int, _ b: Int) throws -> Int {
        let (res, overflow) = a.multipliedReportingOverflow(by: b)
        if overflow {
            throw FlintMetalError.invalidParameter("Integer overflow in calculation (\(a) * \(b))")
        }
        return res
    }

    private static func checkedAdd(_ a: Int, _ b: Int) throws -> Int {
        let (res, overflow) = a.addingReportingOverflow(b)
        if overflow {
            throw FlintMetalError.invalidParameter("Integer overflow in calculation (\(a) + \(b))")
        }
        return res
    }

    private static func checkedToUInt32(_ val: Int, name: String) throws -> UInt32 {
        guard val >= 0, val <= Int(UInt32.max) else {
            throw FlintMetalError.invalidParameter("Value for \(name) (\(val)) exceeds UInt32 bounds")
        }
        return UInt32(val)
    }

    public static func loadShaderSource() -> String {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_router.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(sourcePath)
        if let diskSource = try? String(contentsOf: fileURL, encoding: .utf8), !diskSource.isEmpty {
            return diskSource
        }
        return embeddedShaderSource
    }

    public init(context: FlintMetalContext) throws {
        self.context = context
        let source = Self.loadShaderSource()
        let library = try context.makeLibrary(source: source)
        self.psoBF16 = try context.computePipelineState(functionName: "flint_router_forward_bf16", library: library)
        self.psoF32 = try context.computePipelineState(functionName: "flint_router_forward_f32", library: library)
    }

    public func encodeForward(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        gateWeight: MTLBuffer,
        gateWeightOffset: Int = 0,
        outExperts: MTLBuffer,
        outExpertsOffset: Int = 0,
        outWeights: MTLBuffer,
        outWeightsOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard batch > 0, numTokens > 0 else {
            throw FlintMetalError.invalidParameter("batch (\(batch)) and numTokens (\(numTokens)) must be positive")
        }
        guard inputOffset >= 0, gateWeightOffset >= 0, outExpertsOffset >= 0, outWeightsOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Offsets cannot be negative")
        }

        guard inputOffset % dtype.elementByteWidth == 0,
              gateWeightOffset % 2 == 0,
              outExpertsOffset % 4 == 0,
              outWeightsOffset % 4 == 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets must be aligned to element boundaries")
        }

        let hiddenDim = 2048
        let numExperts = 256
        let topK = 6

        let inBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), try Self.checkedMul(hiddenDim, dtype.elementByteWidth))
        let weightBytes = try Self.checkedMul(try Self.checkedMul(numExperts, hiddenDim), 2) // Always BF16
        let outExpBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), try Self.checkedMul(topK, 4)) // UInt32
        let outWgtBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), try Self.checkedMul(topK, 4)) // Float

        guard input.length >= (try Self.checkedAdd(inputOffset, inBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: inputOffset + inBytes)
        }
        guard gateWeight.length >= (try Self.checkedAdd(gateWeightOffset, weightBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "gateWeight", actual: gateWeight.length, required: gateWeightOffset + weightBytes)
        }
        guard outExperts.length >= (try Self.checkedAdd(outExpertsOffset, outExpBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "outExperts", actual: outExperts.length, required: outExpertsOffset + outExpBytes)
        }
        guard outWeights.length >= (try Self.checkedAdd(outWeightsOffset, outWgtBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "outWeights", actual: outWeights.length, required: outWeightsOffset + outWgtBytes)
        }

        let maxTokens = try Self.checkedMul(batch, numTokens)
        let maxInElements = try Self.checkedMul(maxTokens, hiddenDim)
        let maxOutElements = try Self.checkedMul(maxTokens, topK)
        guard maxInElements <= UInt32.max, maxOutElements <= UInt32.max else {
            throw FlintMetalError.invalidParameter("GPU index product exceeds UInt32.max")
        }

        var numTokensCopy = try Self.checkedToUInt32(numTokens, name: "numTokens")

        let pso: MTLComputePipelineState
        switch dtype {
        case .bfloat16:
            pso = psoBF16
        case .float32:
            pso = psoF32
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(gateWeight, offset: gateWeightOffset, index: 1)
        encoder.setBuffer(outExperts, offset: outExpertsOffset, index: 2)
        encoder.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
        encoder.setBytes(&numTokensCopy, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: numTokens, height: 1, depth: batch)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    @discardableResult
    public func executeForward(
        input: MTLBuffer,
        inputOffset: Int = 0,
        gateWeight: MTLBuffer,
        gateWeightOffset: Int = 0,
        outExperts: MTLBuffer,
        outExpertsOffset: Int = 0,
        outWeights: MTLBuffer,
        outWeightsOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16,
        waitUntilCompleted: Bool = false
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer() else {
            throw FlintMetalError.commandQueueUnavailable
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to create compute command encoder")
        }

        do {
            try encodeForward(
                encoder: encoder,
                input: input,
                inputOffset: inputOffset,
                gateWeight: gateWeight,
                gateWeightOffset: gateWeightOffset,
                outExperts: outExperts,
                outExpertsOffset: outExpertsOffset,
                outWeights: outWeights,
                outWeightsOffset: outWeightsOffset,
                batch: batch,
                numTokens: numTokens,
                dtype: dtype
            )
            encoder.endEncoding()
        } catch {
            encoder.endEncoding()
            throw error
        }

        cb.commit()
        if waitUntilCompleted {
            cb.waitUntilCompleted()
            if let err = cb.error {
                throw FlintMetalError.commandBufferFailed(err.localizedDescription)
            }
            return cb.gpuEndTime - cb.gpuStartTime
        }
        return 0.0
    }

    // MARK: - Embedded MSL Source
    public static let embeddedShaderSource = """
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

"""
}
