import Foundation
import Metal

/// Host Swift controller for Flint MoE Fixed Router-Order Expert Reduction kernel.
///
/// ### Architecture & Residency Invariance Contract
/// This kernel combines the contributions of Flint's six router-selected experts strictly
/// in fixed router order `k = 0, 1, 2, 3, 4, 5`. Expert output vectors are addressed by
/// logical router slot `k`, NEVER by arrival order, thread completion order, or physical expert ID.
/// Consequently, any variation in SSD/cache load timing, streaming arrival order, or cold/warm
/// cache state produces bitwise-identical mathematical output.
public final class FlintReductionKernel {

    public enum Dtype: Sendable {
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
    private let psoF32InBF16Out: MTLComputePipelineState

    public static func loadShaderSource() -> String {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_reduction.metal"
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

        self.psoBF16 = try context.computePipelineState(functionName: "flint_expert_reduction_bf16", library: library)
        self.psoF32 = try context.computePipelineState(functionName: "flint_expert_reduction_f32", library: library)
        self.psoF32InBF16Out = try context.computePipelineState(functionName: "flint_expert_reduction_f32_in_bf16_out", library: library)
    }

    // MARK: - Checked Integer Arithmetic

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

    // MARK: - Encode & Execute

    /// Encodes expert reduction into an active Metal compute command encoder.
    public func encodeReduction(
        into encoder: MTLComputeCommandEncoder,
        contributions: MTLBuffer,
        contributionsOffset: Int = 0,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        sharedExpert: MTLBuffer? = nil,
        sharedExpertOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        inDtype: Dtype = .bfloat16,
        outDtype: Dtype = .bfloat16
    ) throws {
        guard batch > 0, numTokens > 0 else {
            throw FlintMetalError.invalidParameter("batch (\(batch)) and numTokens (\(numTokens)) must be positive")
        }
        guard contributionsOffset >= 0, weightsOffset >= 0, sharedExpertOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets cannot be negative")
        }

        // Alignments
        guard contributionsOffset % inDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("contributionsOffset must be aligned to \(inDtype.elementByteWidth) bytes")
        }
        guard weightsOffset % 4 == 0 else {
            throw FlintMetalError.invalidParameter("weightsOffset must be 4-byte aligned (FP32)")
        }
        guard sharedExpertOffset % inDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("sharedExpertOffset must be aligned to \(inDtype.elementByteWidth) bytes")
        }
        guard outputOffset % outDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset must be aligned to \(outDtype.elementByteWidth) bytes")
        }

        let hiddenDim = 2048
        let topK = 6

        let contribBytes = try Self.checkedMul(try Self.checkedMul(try Self.checkedMul(batch, numTokens), topK * hiddenDim), inDtype.elementByteWidth)
        let weightsBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), topK * 4) // Always Float32
        let sharedBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), hiddenDim * inDtype.elementByteWidth)
        let outBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), hiddenDim * outDtype.elementByteWidth)

        guard contributions.length >= (try Self.checkedAdd(contributionsOffset, contribBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "contributions", actual: contributions.length, required: contributionsOffset + contribBytes)
        }
        guard weights.length >= (try Self.checkedAdd(weightsOffset, weightsBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "weights", actual: weights.length, required: weightsOffset + weightsBytes)
        }
        if let s = sharedExpert {
            guard s.length >= (try Self.checkedAdd(sharedExpertOffset, sharedBytes)) else {
                throw FlintMetalError.bufferTooSmall(name: "sharedExpert", actual: s.length, required: sharedExpertOffset + sharedBytes)
            }
        }
        guard output.length >= (try Self.checkedAdd(outputOffset, outBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + outBytes)
        }

        // Combined integer product bounds
        let maxContribElements = try Self.checkedMul(try Self.checkedMul(batch, numTokens), topK * hiddenDim)
        guard maxContribElements <= UInt32.max else {
            throw FlintMetalError.invalidParameter("Total contribution elements exceed UInt32.max")
        }

        var numTokensCopy = try Self.checkedToUInt32(numTokens, name: "numTokens")
        var hasSharedCopy: UInt32 = (sharedExpert != nil) ? 1 : 0

        let pso: MTLComputePipelineState
        switch (inDtype, outDtype) {
        case (.bfloat16, .bfloat16): pso = psoBF16
        case (.float32, .float32): pso = psoF32
        case (.float32, .bfloat16): pso = psoF32InBF16Out
        case (.bfloat16, .float32):
            throw FlintMetalError.invalidParameter("bfloat16 input with float32 output is not supported")
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(contributions, offset: contributionsOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        if let s = sharedExpert {
            encoder.setBuffer(s, offset: sharedExpertOffset, index: 2)
        } else {
            // Bind output buffer as dummy for unused optional shared_expert buffer
            encoder.setBuffer(output, offset: outputOffset, index: 2)
        }
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBytes(&numTokensCopy, length: 4, index: 4)
        encoder.setBytes(&hasSharedCopy, length: 4, index: 5)

        let gridSize = MTLSize(width: hiddenDim, height: numTokens, depth: batch)
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Convenience standalone execution with completion wait and execution timing.
    @discardableResult
    public func executeReduction(
        contributions: MTLBuffer,
        contributionsOffset: Int = 0,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        sharedExpert: MTLBuffer? = nil,
        sharedExpertOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        inDtype: Dtype = .bfloat16,
        outDtype: Dtype = .bfloat16,
        waitUntilCompleted: Bool = true
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer() else {
            throw FlintMetalError.commandQueueUnavailable
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to create compute command encoder")
        }

        do {
            try encodeReduction(
                into: encoder,
                contributions: contributions,
                contributionsOffset: contributionsOffset,
                weights: weights,
                weightsOffset: weightsOffset,
                sharedExpert: sharedExpert,
                sharedExpertOffset: sharedExpertOffset,
                output: output,
                outputOffset: outputOffset,
                batch: batch,
                numTokens: numTokens,
                inDtype: inDtype,
                outDtype: outDtype
            )
            encoder.endEncoding()
        } catch {
            encoder.endEncoding()
            throw error
        }

        cb.commit()

        if waitUntilCompleted {
            cb.waitUntilCompleted()
            return cb.gpuEndTime - cb.gpuStartTime
        }
        return 0.0
    }

    // MARK: - Embedded Metal Shading Language Source

    public static let embeddedShaderSource: String = """
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

"""
}
