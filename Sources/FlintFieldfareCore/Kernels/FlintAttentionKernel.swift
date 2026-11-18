import Foundation
import Metal

/// Physical KV cache storage layout.
public enum FlintKVStorageLayout: String, Sendable {
    /// Contiguous physical storage of tokens [retainedKVStart ..< retainedKVStart + T_kv].
    case contiguous
    /// Circular/ring buffer layout (unsupported; explicitly rejected).
    case ring
}

/// Native Metal compute kernel for Flint Attention (Grouped-Query Attention, Learned Attention Sinks, and SDPA).
///
/// Mathematical contract matching reference MLX and Hugging Face Flint implementation:
///   - 32 query heads, 4 KV heads (grouped query ratio 8:1).
///   - Head dimension is strictly 128.
///   - Attention scale is 1.0 / sqrt(128.0) ~ 0.0883883476.
///   - Per-head learned attention sinks (`softmax_offset[32]`):
///       Denom = sum_{j in attended} exp(S_{i, j}) + exp(s_h)
///       P_{i, j} = exp(S_{i, j}) / Denom
///       O_{i} = sum_{j in attended} P_{i, j} * V_{j}
///     The sink contributes no value vector.
///   - Bounded-KV Indexing Contract:
///       Decouples absolute query position from physical buffer indices.
///       Physical K/V entries [0 ..< T_kv] store absolute positions [retainedKVStart ..< retainedKVStart + T_kv].
///       Sliding window (513) attends to max(0, p - 512) ... p.
///       Attended keys intersecting retained storage are mapped to valid physical indices.
///   - Streaming online softmax formulation in threadgroup memory (zero global memory matrix allocation).
///   - Supports full causal attention (14 layers) and sliding window attention (35 layers, window = 513).
public final class FlintAttentionKernel: @unchecked Sendable {

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

    // Specialized Compute Pipeline States
    private let psoDecodeBF16: MTLComputePipelineState
    private let psoDecodeF32: MTLComputePipelineState
    private let psoDecodeF32_BF16Sinks: MTLComputePipelineState
    private let psoPrefillBF16: MTLComputePipelineState
    private let psoPrefillF32: MTLComputePipelineState
    private let psoPrefillF32_BF16Sinks: MTLComputePipelineState

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

    private static func checkedToUInt32(_ val: Int, name: String) throws -> UInt32 {
        guard let u = UInt32(exactly: val) else {
            throw FlintMetalError.invalidParameter("Value for \(name) (\(val)) exceeds UInt32 bounds")
        }
        return u
    }

    /// Loads canonical Metal shading language source from Bundle.module, falling back to embedded source.
    public static func loadShaderSource() -> String {
        #if SWIFT_PACKAGE
        if let url = Bundle.module.url(forResource: "flint_attention", withExtension: "metal", subdirectory: "Metal") ??
                     Bundle.module.url(forResource: "flint_attention", withExtension: "metal") {
            if let src = try? String(contentsOf: url, encoding: .utf8) {
                return src
            }
        }
        #endif
        return embeddedShaderSource
    }

    /// Initializes the attention kernel by compiling or retrieving pipeline states from the Metal context.
    public init(context: FlintMetalContext) throws {
        self.context = context

        let source = Self.loadShaderSource()
        let library = try context.makeLibrary(source: source)

        self.psoDecodeBF16 = try context.computePipelineState(functionName: "flint_attention_decode_bf16", library: library)
        self.psoDecodeF32 = try context.computePipelineState(functionName: "flint_attention_decode_f32", library: library)
        self.psoDecodeF32_BF16Sinks = try context.computePipelineState(functionName: "flint_attention_decode_f32_bf16sinks", library: library)
        self.psoPrefillBF16 = try context.computePipelineState(functionName: "flint_attention_prefill_bf16", library: library)
        self.psoPrefillF32 = try context.computePipelineState(functionName: "flint_attention_prefill_f32", library: library)
        self.psoPrefillF32_BF16Sinks = try context.computePipelineState(functionName: "flint_attention_prefill_f32_bf16sinks", library: library)
    }

    public convenience init() throws {
        let context = try FlintMetalContext()
        try self.init(context: context)
    }

    // MARK: - Single-Token Decode Command Encoding (T_q = 1)

    /// Encodes single-token decode attention into an active compute command encoder.
    ///
    /// - Parameters:
    ///   - encoder: Active Metal compute command encoder.
    ///   - queries: `[batch, 32, 1, 128]`
    ///   - keys: `[batch, 4, T_kv, 128]`
    ///   - values: `[batch, 4, T_kv, 128]`
    ///   - sinks: `[32]` learned scalar sinks (`softmax_offset`)
    ///   - output: `[batch, 32, 1, 128]`
    ///   - batch: Batch size (>= 1)
    ///   - T_kv: Number of physically retained key/value tokens (>= 1)
    ///   - queryAbsPos: Absolute position of the query token (defaults to `retainedKVStart + T_kv - 1`)
    ///   - retainedKVStart: Absolute position corresponding to physical index 0 (defaults to 0)
    ///   - isSliding: True for sliding-window (513), false for full causal attention
    ///   - layout: KV storage layout (contiguous supported, ring rejected)
    ///   - scale: Attention scaling factor (1 / sqrt(128))
    ///   - dtype: Activation precision (.bfloat16 or .float32)
    ///   - sinksDtype: Precision of the sinks buffer (.bfloat16 or .float32)
    public func encodeDecode(
        encoder: MTLComputeCommandEncoder,
        queries: MTLBuffer,
        queriesOffset: Int = 0,
        keys: MTLBuffer,
        keysOffset: Int = 0,
        values: MTLBuffer,
        valuesOffset: Int = 0,
        sinks: MTLBuffer,
        sinksOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        T_kv: Int,
        queryAbsPos: Int? = nil,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        layout: FlintKVStorageLayout = .contiguous,
        scale: Float = 1.0 / sqrt(128.0),
        dtype: Dtype = .bfloat16,
        sinksDtype: Dtype? = nil
    ) throws {
        let actualSinksDtype = sinksDtype ?? dtype
        guard layout == .contiguous else {
            throw FlintMetalError.invalidParameter("Ring/circular KV cache layout is not supported; use contiguous retained storage")
        }
        guard batch > 0, T_kv > 0 else {
            throw FlintMetalError.invalidParameter("batch (\(batch)) and T_kv (\(T_kv)) must be positive")
        }
        guard retainedKVStart >= 0 else {
            throw FlintMetalError.invalidParameter("retainedKVStart (\(retainedKVStart)) cannot be negative")
        }
        let qPos = queryAbsPos ?? (retainedKVStart + T_kv - 1)
        guard qPos >= 0 else {
            throw FlintMetalError.invalidParameter("queryAbsPos (\(qPos)) cannot be negative")
        }
        guard scale > 0, scale.isFinite, !scale.isNaN else {
            throw FlintMetalError.invalidParameter("scale must be positive and finite (got \(scale))")
        }
        guard queriesOffset >= 0, keysOffset >= 0, valuesOffset >= 0, sinksOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Offsets cannot be negative")
        }

        // Alignments
        guard queriesOffset % dtype.elementByteWidth == 0,
              keysOffset % dtype.elementByteWidth == 0,
              valuesOffset % dtype.elementByteWidth == 0,
              outputOffset % dtype.elementByteWidth == 0,
              sinksOffset % actualSinksDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets must be aligned to element boundaries")
        }

        // Capacities
        let headDim = 128
        let qBytes = try Self.checkedMul(try Self.checkedMul(batch, 32), try Self.checkedMul(headDim, dtype.elementByteWidth))
        let kvBytes = try Self.checkedMul(try Self.checkedMul(try Self.checkedMul(batch, 4), T_kv), try Self.checkedMul(headDim, dtype.elementByteWidth))
        let sinksBytes = try Self.checkedMul(32, actualSinksDtype.elementByteWidth)

        guard queries.length >= (try Self.checkedAdd(queriesOffset, qBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "queries", actual: queries.length, required: queriesOffset + qBytes)
        }
        guard keys.length >= (try Self.checkedAdd(keysOffset, kvBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "keys", actual: keys.length, required: keysOffset + kvBytes)
        }
        guard values.length >= (try Self.checkedAdd(valuesOffset, kvBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "values", actual: values.length, required: valuesOffset + kvBytes)
        }
        guard sinks.length >= (try Self.checkedAdd(sinksOffset, sinksBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "sinks", actual: sinks.length, required: sinksOffset + sinksBytes)
        }
        guard output.length >= (try Self.checkedAdd(outputOffset, qBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + qBytes)
        }

        // Bounded-KV index translation
        let kStartAbs = isSliding ? max(0, qPos - 512) : 0
        let kEndAbs = qPos
        let kvEndAbs = try Self.checkedAdd(retainedKVStart, T_kv - 1)

        let effStart = max(kStartAbs, retainedKVStart)
        let effEnd = min(kEndAbs, kvEndAbs)

        let physStart: UInt32
        let physEnd: UInt32
        if effStart <= effEnd && effStart >= retainedKVStart {
            physStart = try Self.checkedToUInt32(effStart - retainedKVStart, name: "physStart")
            physEnd = try Self.checkedToUInt32(effEnd - retainedKVStart + 1, name: "physEnd")
        } else {
            physStart = 0
            physEnd = 0
        }

        var T_kv_copy = try Self.checkedToUInt32(T_kv, name: "T_kv")
        var phys_start_copy = physStart
        var phys_end_copy = physEnd
        var scale_copy = scale

        // Pipeline state selection based on activation and sink dtypes
        let pso: MTLComputePipelineState
        switch (dtype, actualSinksDtype) {
        case (.bfloat16, .bfloat16):
            pso = psoDecodeBF16
        case (.float32, .float32):
            pso = psoDecodeF32
        case (.float32, .bfloat16):
            pso = psoDecodeF32_BF16Sinks
        case (.bfloat16, .float32):
            throw FlintMetalError.invalidParameter("bfloat16 activations with float32 sinks is not supported")
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(queries, offset: queriesOffset, index: 0)
        encoder.setBuffer(keys, offset: keysOffset, index: 1)
        encoder.setBuffer(values, offset: valuesOffset, index: 2)
        encoder.setBuffer(sinks, offset: sinksOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.setBytes(&T_kv_copy, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&phys_start_copy, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&phys_end_copy, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&scale_copy, length: MemoryLayout<Float>.size, index: 8)

        let threadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 32, height: 1, depth: batch)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    // MARK: - Multi-Token Prefill Command Encoding (T_q >= 1)

    /// Encodes multi-token prefill attention into an active compute command encoder.
    public func encodePrefill(
        encoder: MTLComputeCommandEncoder,
        queries: MTLBuffer,
        queriesOffset: Int = 0,
        keys: MTLBuffer,
        keysOffset: Int = 0,
        values: MTLBuffer,
        valuesOffset: Int = 0,
        sinks: MTLBuffer,
        sinksOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        T_q: Int,
        T_kv: Int,
        cacheOffset: Int = 0,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        layout: FlintKVStorageLayout = .contiguous,
        scale: Float = 1.0 / sqrt(128.0),
        dtype: Dtype = .bfloat16,
        sinksDtype: Dtype? = nil
    ) throws {
        let actualSinksDtype = sinksDtype ?? dtype
        guard layout == .contiguous else {
            throw FlintMetalError.invalidParameter("Ring/circular KV cache layout is not supported; use contiguous retained storage")
        }
        guard batch > 0, T_q > 0, T_kv > 0 else {
            throw FlintMetalError.invalidParameter("batch (\(batch)), T_q (\(T_q)), and T_kv (\(T_kv)) must be positive")
        }
        guard cacheOffset >= 0, retainedKVStart >= 0 else {
            throw FlintMetalError.invalidParameter("cacheOffset and retainedKVStart cannot be negative")
        }
        guard scale > 0, scale.isFinite, !scale.isNaN else {
            throw FlintMetalError.invalidParameter("scale must be positive and finite (got \(scale))")
        }
        guard queriesOffset >= 0, keysOffset >= 0, valuesOffset >= 0, sinksOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Offsets cannot be negative")
        }

        // Alignments
        guard queriesOffset % dtype.elementByteWidth == 0,
              keysOffset % dtype.elementByteWidth == 0,
              valuesOffset % dtype.elementByteWidth == 0,
              outputOffset % dtype.elementByteWidth == 0,
              sinksOffset % actualSinksDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets must be aligned to element boundaries")
        }

        let headDim = 128
        let qBytes = try Self.checkedMul(try Self.checkedMul(try Self.checkedMul(batch, 32), T_q), try Self.checkedMul(headDim, dtype.elementByteWidth))
        let kvBytes = try Self.checkedMul(try Self.checkedMul(try Self.checkedMul(batch, 4), T_kv), try Self.checkedMul(headDim, dtype.elementByteWidth))
        let sinksBytes = try Self.checkedMul(32, actualSinksDtype.elementByteWidth)

        guard queries.length >= (try Self.checkedAdd(queriesOffset, qBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "queries", actual: queries.length, required: queriesOffset + qBytes)
        }
        guard keys.length >= (try Self.checkedAdd(keysOffset, kvBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "keys", actual: keys.length, required: keysOffset + kvBytes)
        }
        guard values.length >= (try Self.checkedAdd(valuesOffset, kvBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "values", actual: values.length, required: valuesOffset + kvBytes)
        }
        guard sinks.length >= (try Self.checkedAdd(sinksOffset, sinksBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "sinks", actual: sinks.length, required: sinksOffset + sinksBytes)
        }
        guard output.length >= (try Self.checkedAdd(outputOffset, qBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + qBytes)
        }

        var T_q_copy = try Self.checkedToUInt32(T_q, name: "T_q")
        var T_kv_copy = try Self.checkedToUInt32(T_kv, name: "T_kv")
        var cache_offset_copy = try Self.checkedToUInt32(cacheOffset, name: "cacheOffset")
        var kv_retained_start_copy = try Self.checkedToUInt32(retainedKVStart, name: "retainedKVStart")
        var is_sliding_copy = UInt32(isSliding ? 1 : 0)
        var scale_copy = scale

        let pso: MTLComputePipelineState
        switch (dtype, actualSinksDtype) {
        case (.bfloat16, .bfloat16):
            pso = psoPrefillBF16
        case (.float32, .float32):
            pso = psoPrefillF32
        case (.float32, .bfloat16):
            pso = psoPrefillF32_BF16Sinks
        case (.bfloat16, .float32):
            throw FlintMetalError.invalidParameter("bfloat16 activations with float32 sinks is not supported")
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(queries, offset: queriesOffset, index: 0)
        encoder.setBuffer(keys, offset: keysOffset, index: 1)
        encoder.setBuffer(values, offset: valuesOffset, index: 2)
        encoder.setBuffer(sinks, offset: sinksOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.setBytes(&T_q_copy, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&T_kv_copy, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&cache_offset_copy, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&kv_retained_start_copy, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&is_sliding_copy, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&scale_copy, length: MemoryLayout<Float>.size, index: 10)

        let threadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
        let threadgroups = MTLSize(width: T_q, height: 32, depth: batch)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    // MARK: - Execution Helpers

    @discardableResult
    public func executeDecode(
        queries: MTLBuffer,
        queriesOffset: Int = 0,
        keys: MTLBuffer,
        keysOffset: Int = 0,
        values: MTLBuffer,
        valuesOffset: Int = 0,
        sinks: MTLBuffer,
        sinksOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        T_kv: Int,
        queryAbsPos: Int? = nil,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        layout: FlintKVStorageLayout = .contiguous,
        scale: Float = 1.0 / sqrt(128.0),
        dtype: Dtype = .bfloat16,
        sinksDtype: Dtype? = nil,
        waitUntilCompleted: Bool = false
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer() else {
            throw FlintMetalError.commandQueueUnavailable
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to create compute command encoder")
        }

        do {
            try encodeDecode(
                encoder: encoder,
                queries: queries,
                queriesOffset: queriesOffset,
                keys: keys,
                keysOffset: keysOffset,
                values: values,
                valuesOffset: valuesOffset,
                sinks: sinks,
                sinksOffset: sinksOffset,
                output: output,
                outputOffset: outputOffset,
                batch: batch,
                T_kv: T_kv,
                queryAbsPos: queryAbsPos,
                retainedKVStart: retainedKVStart,
                isSliding: isSliding,
                layout: layout,
                scale: scale,
                dtype: dtype,
                sinksDtype: sinksDtype
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

    @discardableResult
    public func executePrefill(
        queries: MTLBuffer,
        queriesOffset: Int = 0,
        keys: MTLBuffer,
        keysOffset: Int = 0,
        values: MTLBuffer,
        valuesOffset: Int = 0,
        sinks: MTLBuffer,
        sinksOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        T_q: Int,
        T_kv: Int,
        cacheOffset: Int = 0,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        layout: FlintKVStorageLayout = .contiguous,
        scale: Float = 1.0 / sqrt(128.0),
        dtype: Dtype = .bfloat16,
        sinksDtype: Dtype? = nil,
        waitUntilCompleted: Bool = false
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer() else {
            throw FlintMetalError.commandQueueUnavailable
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to create compute command encoder")
        }

        do {
            try encodePrefill(
                encoder: encoder,
                queries: queries,
                queriesOffset: queriesOffset,
                keys: keys,
                keysOffset: keysOffset,
                values: values,
                valuesOffset: valuesOffset,
                sinks: sinks,
                sinksOffset: sinksOffset,
                output: output,
                outputOffset: outputOffset,
                batch: batch,
                T_q: T_q,
                T_kv: T_kv,
                cacheOffset: cacheOffset,
                retainedKVStart: retainedKVStart,
                isSliding: isSliding,
                layout: layout,
                scale: scale,
                dtype: dtype,
                sinksDtype: sinksDtype
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

"""
}
