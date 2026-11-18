import Foundation
import Metal

/// Native Metal compute kernel for Flint Rotary Position Embedding (RoPE).
///
/// Mathematical contract:
///   Split-half rotation on head dimension 128 (`traditional=False` in MLX / HuggingFace):
///   For d in 0..<64:
///     freq_d = base ** (-d / 64.0)
///     theta = (offset + t) * freq_d
///     y[d] = x[d] * cos(theta) - x[d + 64] * sin(theta)
///     y[d + 64] = x[d + 64] * cos(theta) + x[d] * sin(theta)
///
/// Invariants:
/// - Applied ONLY to sliding-window attention layers.
/// - Base theta is 10000.0.
/// - Head dimension is strictly 128.
/// - Zero heap allocations during command encoding.
public final class FlintRoPEKernel: @unchecked Sendable {

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

    // MARK: - Overflow-Safe Arithmetic Helpers

    private static func checkedMul(_ a: Int, _ b: Int) throws -> Int {
        let (res, overflow) = a.multipliedReportingOverflow(by: b)
        guard !overflow, res >= 0 else {
            throw FlintMetalError.invalidParameter("Integer overflow in calculation (\(a) * \(b))")
        }
        return res
    }

    private static func checkedAdd(_ a: Int, _ b: Int) throws -> Int {
        let (res, overflow) = a.addingReportingOverflow(b)
        guard !overflow, res >= 0 else {
            throw FlintMetalError.invalidParameter("Integer overflow in calculation (\(a) + \(b))")
        }
        return res
    }

    /// Loads canonical Metal shading language source from Bundle.module, falling back to embedded source.
    public static func loadShaderSource() -> String {
        #if SWIFT_PACKAGE
        if let url = Bundle.module.url(forResource: "flint_rope", withExtension: "metal", subdirectory: "Metal") ??
                     Bundle.module.url(forResource: "flint_rope", withExtension: "metal") {
            if let src = try? String(contentsOf: url, encoding: .utf8) {
                return src
            }
        }
        #endif
        return embeddedShaderSource
    }

    /// Initializes the RoPE kernel by compiling or retrieving pipeline states from the Metal context.
    public init(context: FlintMetalContext) throws {
        self.context = context

        let source = Self.loadShaderSource()
        let library = try context.makeLibrary(source: source)

        self.psoBF16 = try context.computePipelineState(functionName: "flint_rope_split_half_bf16", library: library)
        self.psoF32 = try context.computePipelineState(functionName: "flint_rope_split_half_f32", library: library)
    }

    public convenience init() throws {
        let context = try FlintMetalContext()
        try self.init(context: context)
    }

    // MARK: - Command Encoding

    /// Encodes RoPE dispatch into an active compute command encoder.
    /// Layout: `[batch, numHeads, seqLen, 128]`.
    public func encode(
        commandEncoder encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numHeads: Int,
        seqLen: Int,
        offset: Int,
        base: Float = 10000.0,
        dtype: Dtype = .bfloat16
    ) throws {
        guard batch > 0, numHeads > 0, seqLen > 0, offset >= 0 else {
            throw FlintMetalError.invalidParameter("Invalid parameters: batch=\(batch), numHeads=\(numHeads), seqLen=\(seqLen), offset=\(offset)")
        }
        guard base > 0, base.isFinite, !base.isNaN else {
            throw FlintMetalError.invalidParameter("base must be positive and finite (got \(base))")
        }
        guard inputOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Offsets cannot be negative")
        }
        guard inputOffset % dtype.elementByteWidth == 0, outputOffset % dtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("Offsets must be aligned to \(dtype.elementByteWidth) bytes")
        }

        // Validate buffer capacities
        let headDim = 128
        let totalElements = try Self.checkedMul(try Self.checkedMul(try Self.checkedMul(batch, numHeads), seqLen), headDim)
        let totalBytes = try Self.checkedMul(totalElements, dtype.elementByteWidth)

        let totalInputReq = try Self.checkedAdd(inputOffset, totalBytes)
        guard input.length >= totalInputReq else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: totalInputReq)
        }

        let totalOutputReq = try Self.checkedAdd(outputOffset, totalBytes)
        guard output.length >= totalOutputReq else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: totalOutputReq)
        }

        var offsetCopy = UInt32(offset)
        var seqLenCopy = UInt32(seqLen)
        var numHeadsCopy = UInt32(numHeads)
        var baseCopy = base

        let pso = (dtype == .bfloat16) ? psoBF16 : psoF32
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&offsetCopy, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&seqLenCopy, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&numHeadsCopy, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&baseCopy, length: MemoryLayout<Float>.size, index: 5)

        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroups = MTLSize(width: seqLen, height: numHeads, depth: batch)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// Dispatches RoPE on GPU and synchronizes execution.
    @discardableResult
    public func execute(
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numHeads: Int,
        seqLen: Int,
        offset: Int,
        base: Float = 10000.0,
        dtype: Dtype = .bfloat16,
        commandBuffer externalCmdBuffer: MTLCommandBuffer? = nil,
        waitUntilCompleted: Bool = true
    ) throws -> Double {
        let cmdBuffer = externalCmdBuffer ?? context.queue.makeCommandBuffer()
        guard let cb = cmdBuffer else {
            throw FlintMetalError.commandQueueUnavailable
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to create compute command encoder")
        }

        do {
            try encode(
                commandEncoder: encoder,
                input: input,
                inputOffset: inputOffset,
                output: output,
                outputOffset: outputOffset,
                batch: batch,
                numHeads: numHeads,
                seqLen: seqLen,
                offset: offset,
                base: base,
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

    // MARK: - Embedded Metal Shading Language Source

    public static let embeddedShaderSource = """
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

"""
}
