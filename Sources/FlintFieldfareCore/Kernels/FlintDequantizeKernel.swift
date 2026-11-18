import Foundation
import Metal

/// Host Swift controller for Flint native 3-bit and 4-bit Metal dequantization kernels.
///
/// ### Architecture & Memory Model Note
/// This kernel is a stateless compute dispatch primitive operating on caller-provided `MTLBuffer`s
/// with explicit byte offsets. It intentionally does NOT own or manage an expert cache, scratch buffer pool,
/// or memory staging lifecycle. A future reusable resident/expert scratch-buffer manager (Milestones F3/F4)
/// will coordinate preallocated scratch slices across active MoE layers, ensuring bounded working memory.
public final class FlintDequantizeKernel {

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

    public enum QuantBits: Int, Sendable {
        case three = 3
        case four = 4

        public var wordsPerGroup: Int {
            switch self {
            case .three: return 6 // 64 * 3 / 32 = 6 words
            case .four: return 8  // 64 * 4 / 32 = 8 words
            }
        }
    }

    public let context: FlintMetalContext
    private let pso4BitBF16: MTLComputePipelineState
    private let pso4BitF32: MTLComputePipelineState
    private let pso3BitBF16: MTLComputePipelineState
    private let pso3BitF32: MTLComputePipelineState

    public static func loadShaderSource() -> String {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_dequantize.metal"
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

        self.pso4BitBF16 = try context.computePipelineState(functionName: "flint_dequantize_4bit_bf16", library: library)
        self.pso4BitF32 = try context.computePipelineState(functionName: "flint_dequantize_4bit_f32", library: library)
        self.pso3BitBF16 = try context.computePipelineState(functionName: "flint_dequantize_3bit_bf16", library: library)
        self.pso3BitF32 = try context.computePipelineState(functionName: "flint_dequantize_3bit_f32", library: library)
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

    /// Encodes dequantization into an active Metal compute command encoder.
    public func encodeDequantize(
        into encoder: MTLComputeCommandEncoder,
        packedWeights: MTLBuffer,
        packedWeightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: Int,
        cols: Int,
        bits: QuantBits,
        outDtype: Dtype = .bfloat16
    ) throws {
        guard rows > 0, cols > 0 else {
            throw FlintMetalError.invalidParameter("rows (\(rows)) and cols (\(cols)) must be positive")
        }
        guard cols % 64 == 0 else {
            throw FlintMetalError.invalidParameter("cols (\(cols)) must be a multiple of group size (64)")
        }
        guard packedWeightsOffset >= 0, scalesOffset >= 0, biasesOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets cannot be negative")
        }

        // Alignments: packedWeights UInt32 (4 bytes), scales/biases BF16 (2 bytes), output (elementByteWidth)
        guard packedWeightsOffset % 4 == 0 else {
            throw FlintMetalError.invalidParameter("packedWeightsOffset must be 4-byte aligned")
        }
        guard scalesOffset % 2 == 0, biasesOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("scales and biases offsets must be 2-byte aligned")
        }
        guard outputOffset % outDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset must be aligned to \(outDtype.elementByteWidth) bytes")
        }

        let numGroups = cols / 64
        let wordsPerRow = numGroups * bits.wordsPerGroup

        let packedBytes = try Self.checkedMul(try Self.checkedMul(rows, wordsPerRow), 4)
        let scalesBytes = try Self.checkedMul(try Self.checkedMul(rows, numGroups), 2) // Always BF16 in Flint
        let biasesBytes = scalesBytes
        let outBytes = try Self.checkedMul(try Self.checkedMul(rows, cols), outDtype.elementByteWidth)

        guard packedWeights.length >= (try Self.checkedAdd(packedWeightsOffset, packedBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "packedWeights", actual: packedWeights.length, required: packedWeightsOffset + packedBytes)
        }
        guard scales.length >= (try Self.checkedAdd(scalesOffset, scalesBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "scales", actual: scales.length, required: scalesOffset + scalesBytes)
        }
        guard biases.length >= (try Self.checkedAdd(biasesOffset, biasesBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "biases", actual: biases.length, required: biasesOffset + biasesBytes)
        }
        guard output.length >= (try Self.checkedAdd(outputOffset, outBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + outBytes)
        }

        // Combined integer product checks
        let maxElements = try Self.checkedMul(rows, cols)
        guard maxElements <= UInt32.max else {
            throw FlintMetalError.invalidParameter("Matrix element count exceeds UInt32.max")
        }

        var colsCopy = try Self.checkedToUInt32(cols, name: "cols")
        var rowsCopy = try Self.checkedToUInt32(rows, name: "rows")

        let pso: MTLComputePipelineState
        switch (bits, outDtype) {
        case (.four, .bfloat16): pso = pso4BitBF16
        case (.four, .float32): pso = pso4BitF32
        case (.three, .bfloat16): pso = pso3BitBF16
        case (.three, .float32): pso = pso3BitF32
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(packedWeights, offset: packedWeightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBytes(&colsCopy, length: 4, index: 4)
        encoder.setBytes(&rowsCopy, length: 4, index: 5)

        let gridSize = MTLSize(width: cols, height: rows, depth: 1)
        let tgWidth = min(cols, 256)
        let tgSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Convenience standalone execution with completion wait and execution timing.
    @discardableResult
    public func executeDequantize(
        packedWeights: MTLBuffer,
        packedWeightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        biases: MTLBuffer,
        biasesOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: Int,
        cols: Int,
        bits: QuantBits,
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
            try encodeDequantize(
                into: encoder,
                packedWeights: packedWeights,
                packedWeightsOffset: packedWeightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                output: output,
                outputOffset: outputOffset,
                rows: rows,
                cols: cols,
                bits: bits,
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

"""
}
