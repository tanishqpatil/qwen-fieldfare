import Foundation
import Metal

/// Native Metal compute kernel for Flint Root Mean Square Normalization (RMSNorm).
///
/// Mathematical contract matching reference MLX and Hugging Face Flint implementation:
///   y = x * rsqrt(mean(x * x, axis=-1) + epsilon) * weight
///
/// ### Key Invariants:
/// - **Checkpoint Weights**: Weights are stored in `bfloat16` representation in resident memory.
/// - **FP32 Accumulation**: Intermediate squares, sums, means, and scale calculations evaluate
///   in full FP32 (single precision) on GPU to preserve numerical stability.
/// - **Immutable Resident Weights**: Weights are bound as `device const bfloat*` and are never mutated.
/// - **Zero Staging Memory**: Encodes dispatches directly between input and output Metal shared buffers.
/// - **Specialized Layouts**: Optimized threadgroup reductions for hidden width 2048 and head width 128.
public final class FlintRMSNormKernel: @unchecked Sendable {

    /// Supported input and output activation data types.
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
    private let pso2048_bf16: MTLComputePipelineState
    private let pso2048_f32: MTLComputePipelineState
    private let pso128_bf16: MTLComputePipelineState
    private let pso128_f32: MTLComputePipelineState
    private let psoGeneral_bf16: MTLComputePipelineState
    private let psoGeneral_f32: MTLComputePipelineState

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
        if let url = Bundle.module.url(forResource: "flint_rmsnorm", withExtension: "metal", subdirectory: "Metal") ??
                     Bundle.module.url(forResource: "flint_rmsnorm", withExtension: "metal") {
            if let src = try? String(contentsOf: url, encoding: .utf8) {
                return src
            }
        }
        #endif
        return embeddedShaderSource
    }

    /// Initializes the RMSNorm kernel by compiling or retrieving pipeline states from the Metal context.
    public init(context: FlintMetalContext) throws {
        self.context = context

        // Prefer canonical packaged Metal shader resource from Bundle.module, falling back to embedded source
        let source = Self.loadShaderSource()
        let library = try context.makeLibrary(source: source)

        self.pso2048_bf16 = try context.computePipelineState(functionName: "flint_rmsnorm_2048_bf16", library: library)
        self.pso2048_f32 = try context.computePipelineState(functionName: "flint_rmsnorm_2048_f32", library: library)
        self.pso128_bf16 = try context.computePipelineState(functionName: "flint_rmsnorm_128_bf16", library: library)
        self.pso128_f32 = try context.computePipelineState(functionName: "flint_rmsnorm_128_f32", library: library)
        self.psoGeneral_bf16 = try context.computePipelineState(functionName: "flint_rmsnorm_general_bf16", library: library)
        self.psoGeneral_f32 = try context.computePipelineState(functionName: "flint_rmsnorm_general_f32", library: library)
    }

    /// Convenience initializer using the default system Metal context.
    public convenience init() throws {
        let context = try FlintMetalContext()
        try self.init(context: context)
    }

    // MARK: - Direct Command Encoder Encoding

    /// Encodes RMSNorm dispatch into an active compute command encoder.
    ///
    /// - Parameters:
    ///   - encoder: Active Metal compute command encoder.
    ///   - input: Metal buffer containing input activations.
    ///   - inputOffset: Byte offset into input buffer.
    ///   - weight: Metal buffer containing BF16 resident weights (length = dimension * 2 bytes).
    ///   - weightOffset: Byte offset into weight buffer.
    ///   - output: Metal buffer destination for normalized activations.
    ///   - outputOffset: Byte offset into output buffer.
    ///   - rows: Number of independent rows/vectors to normalize (tokens or heads, >= 1).
    ///   - dimension: Feature dimension (e.g. 2048 for hidden state, 128 for Q/K heads).
    ///   - epsilon: Additive constant for numerical stability (typically 1e-6 for Flint).
    ///   - dtype: Data type of activations (.bfloat16 or .float32).
    public func encode(
        commandEncoder encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: MTLBuffer,
        weightOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: Int,
        dimension: Int,
        epsilon: Float,
        dtype: Dtype = .bfloat16
    ) throws {
        // Strict boundary and parameter validation
        guard rows > 0 else {
            throw FlintMetalError.invalidParameter("rows must be positive (got \(rows))")
        }
        guard dimension > 0 else {
            throw FlintMetalError.invalidParameter("dimension must be positive (got \(dimension))")
        }
        guard epsilon > 0, epsilon.isFinite, !epsilon.isNaN else {
            throw FlintMetalError.invalidParameter("epsilon must be positive and finite (got \(epsilon))")
        }
        guard inputOffset >= 0, weightOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Offsets cannot be negative")
        }

        // Validate byte alignment for pointers
        guard inputOffset % dtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("inputOffset (\(inputOffset)) is not aligned to \(dtype.elementByteWidth) bytes")
        }
        guard outputOffset % dtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset (\(outputOffset)) is not aligned to \(dtype.elementByteWidth) bytes")
        }
        guard weightOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("weightOffset (\(weightOffset)) is not aligned to 2 bytes (BF16)")
        }

        // Overflow-checked buffer capacity bounds validation
        let bytesPerRow = try Self.checkedMul(dimension, dtype.elementByteWidth)
        let totalInputRowsBytes = try Self.checkedMul(rows, bytesPerRow)
        let totalInputRequired = try Self.checkedAdd(inputOffset, totalInputRowsBytes)
        guard input.length >= totalInputRequired else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: totalInputRequired)
        }

        // Checkpoint weights are always BF16 (2 bytes per element)
        let totalWeightBytes = try Self.checkedMul(dimension, 2)
        let totalWeightRequired = try Self.checkedAdd(weightOffset, totalWeightBytes)
        guard weight.length >= totalWeightRequired else {
            throw FlintMetalError.bufferTooSmall(name: "weight", actual: weight.length, required: totalWeightRequired)
        }

        let totalOutputRowsBytes = try Self.checkedMul(rows, bytesPerRow)
        let totalOutputRequired = try Self.checkedAdd(outputOffset, totalOutputRowsBytes)
        guard output.length >= totalOutputRequired else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: totalOutputRequired)
        }

        // Select specialized pipeline and threadgroup configuration
        var epsCopy = epsilon
        var dimCopy = UInt32(dimension)

        if dimension == 2048 {
            let pso = (dtype == .bfloat16) ? pso2048_bf16 : pso2048_f32
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(input, offset: inputOffset, index: 0)
            encoder.setBuffer(weight, offset: weightOffset, index: 1)
            encoder.setBuffer(output, offset: outputOffset, index: 2)
            encoder.setBytes(&epsCopy, length: MemoryLayout<Float>.size, index: 3)

            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: rows, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        } else if dimension == 128 {
            let pso = (dtype == .bfloat16) ? pso128_bf16 : pso128_f32
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(input, offset: inputOffset, index: 0)
            encoder.setBuffer(weight, offset: weightOffset, index: 1)
            encoder.setBuffer(output, offset: outputOffset, index: 2)
            encoder.setBytes(&epsCopy, length: MemoryLayout<Float>.size, index: 3)

            let threadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
            let threadgroups = MTLSize(width: rows, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

        } else {
            let pso = (dtype == .bfloat16) ? psoGeneral_bf16 : psoGeneral_f32
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(input, offset: inputOffset, index: 0)
            encoder.setBuffer(weight, offset: weightOffset, index: 1)
            encoder.setBuffer(output, offset: outputOffset, index: 2)
            encoder.setBytes(&epsCopy, length: MemoryLayout<Float>.size, index: 3)
            encoder.setBytes(&dimCopy, length: MemoryLayout<UInt32>.size, index: 4)

            // General kernel uses a fixed 256-thread reduction with bounds checking
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: rows, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }

    // MARK: - Standalone Execution & Benchmarking Helper

    /// Dispatches RMSNorm on GPU and optionally measures execution duration.
    ///
    /// - Returns: Elapsed execution time in seconds (measured via synchronized GPU command buffer timestamps).
    @discardableResult
    public func execute(
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: MTLBuffer,
        weightOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: Int,
        dimension: Int,
        epsilon: Float,
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
                weight: weight,
                weightOffset: weightOffset,
                output: output,
                outputOffset: outputOffset,
                rows: rows,
                dimension: dimension,
                epsilon: epsilon,
                dtype: dtype
            )
            encoder.endEncoding()
        } catch {
            encoder.endEncoding()
            throw error
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        cb.commit()

        if waitUntilCompleted {
            cb.waitUntilCompleted()
            if let error = cb.error {
                throw FlintMetalError.commandBufferFailed(error.localizedDescription)
            }
            // Use hardware GPU timestamps if available, otherwise wall-clock duration
            if cb.gpuEndTime > cb.gpuStartTime && cb.gpuStartTime > 0 {
                return cb.gpuEndTime - cb.gpuStartTime
            }
        }
        return CFAbsoluteTimeGetCurrent() - startTime
    }

    // MARK: - Direct Resident Weight Integration

    /// Executes RMSNorm consuming an authoritative resident weight tensor directly without memory copies.
    public func forward(
        input: MTLBuffer,
        inputOffset: Int = 0,
        weightTensor: FlintResidentTensor,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: Int,
        epsilon: Float = 1e-6,
        dtype: Dtype = .bfloat16,
        waitUntilCompleted: Bool = true
    ) throws {
        guard let metalWeight = weightTensor.buffer.metalBuffer else {
            throw FlintResidentError.backendMismatch(expected: "metalShared", actual: weightTensor.buffer.backend.rawValue)
        }
        let dimension = weightTensor.shape.last ?? 0
        try execute(
            input: input,
            inputOffset: inputOffset,
            weight: metalWeight,
            weightOffset: 0,
            output: output,
            outputOffset: outputOffset,
            rows: rows,
            dimension: dimension,
            epsilon: epsilon,
            dtype: dtype,
            waitUntilCompleted: waitUntilCompleted
        )
    }

    // MARK: - Embedded Metal Shading Language Source

    public static let embeddedShaderSource = """
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

"""
}
