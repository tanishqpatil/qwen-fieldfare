import Foundation
import Metal

/// Host Swift controller for Flint Native Dense and Shared Expert SwiGLU MLP.
///
/// ### Architecture & Memory Contract
/// - Dense MLP (Layers 0, 1): hiddenDim = 2048, intermediateDim = 6144.
/// - Shared Expert MLP (Layers 2..48): hiddenDim = 2048, intermediateDim = 1024.
/// - Packed 4-bit affine weights with group size 64 are dequantized into a bounded
///   reusable 24 MiB scratch buffer sequentially (zero permanent expansion of weights).
/// - Matrix-vector projections evaluate with FP32 SIMD-coalesced reduction.
/// - Steady-state forward dispatches make strictly zero heap or GPU memory allocations.
public final class FlintMLPKernel {

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

    /// Holds buffers for all 5 intermediate/final stages of SwiGLU MLP forward evaluation.
    public struct Intermediates {
        public let gateProj: MTLBuffer
        public let upProj: MTLBuffer
        public let siluGate: MTLBuffer
        public let gatedProduct: MTLBuffer
        public let downProj: MTLBuffer
    }

    public let context: FlintMetalContext
    public let dequantKernel: FlintDequantizeKernel

    private let psoGemmBF16: MTLComputePipelineState
    private let psoGemmF32: MTLComputePipelineState
    private let psoGemmF32InBF16Out: MTLComputePipelineState
    private let psoSiluBF16: MTLComputePipelineState
    private let psoSiluF32: MTLComputePipelineState
    private let psoMulBF16: MTLComputePipelineState
    private let psoMulF32: MTLComputePipelineState
    private let psoSwigluBF16: MTLComputePipelineState
    private let psoSwigluF32: MTLComputePipelineState

    // Reusable bounded scratch buffers
    // Max intermediate dimension: 6144 (Dense layer) * 2048 * 4 bytes = 48 MiB (accommodates FP32)
    public let maxWeightScratchBytes: Int = 6144 * 2048 * 4
    public let maxActivationScratchBytes: Int = 64 * 6144 * 4 // up to 64 tokens in FP32

    public let weightScratch: MTLBuffer
    public let gateScratch: MTLBuffer
    public let upScratch: MTLBuffer
    public let actScratch: MTLBuffer

    public static func loadShaderSource() -> String {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_mlp.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(sourcePath)
        if let diskSource = try? String(contentsOf: fileURL, encoding: .utf8), !diskSource.isEmpty {
            return diskSource
        }
        return embeddedShaderSource
    }

    public init(context: FlintMetalContext) throws {
        self.context = context
        self.dequantKernel = try FlintDequantizeKernel(context: context)

        let source = Self.loadShaderSource()
        let library = try context.makeLibrary(source: source)

        self.psoGemmBF16 = try context.computePipelineState(functionName: "flint_gemm_bf16", library: library)
        self.psoGemmF32 = try context.computePipelineState(functionName: "flint_gemm_f32", library: library)
        self.psoGemmF32InBF16Out = try context.computePipelineState(functionName: "flint_gemm_f32_in_bf16_out", library: library)
        self.psoSiluBF16 = try context.computePipelineState(functionName: "flint_silu_bf16", library: library)
        self.psoSiluF32 = try context.computePipelineState(functionName: "flint_silu_f32", library: library)
        self.psoMulBF16 = try context.computePipelineState(functionName: "flint_elementwise_mul_bf16", library: library)
        self.psoMulF32 = try context.computePipelineState(functionName: "flint_elementwise_mul_f32", library: library)
        self.psoSwigluBF16 = try context.computePipelineState(functionName: "flint_swiglu_bf16", library: library)
        self.psoSwigluF32 = try context.computePipelineState(functionName: "flint_swiglu_f32", library: library)

        self.weightScratch = try context.makeBuffer(byteCount: maxWeightScratchBytes)
        self.gateScratch = try context.makeBuffer(byteCount: maxActivationScratchBytes)
        self.upScratch = try context.makeBuffer(byteCount: maxActivationScratchBytes)
        self.actScratch = try context.makeBuffer(byteCount: maxActivationScratchBytes)
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

    // MARK: - Encode Compute Primitives

    /// Encodes GEMM: Y = X * W^T
    /// - in_x: [batch, numTokens, K]
    /// - weight: [M, K]
    /// - out_y: [batch, numTokens, M]
    public func encodeGEMM(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: MTLBuffer,
        weightOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        M: Int,
        K: Int,
        batch: Int = 1,
        numTokens: Int,
        inDtype: Dtype = .bfloat16,
        outDtype: Dtype = .bfloat16
    ) throws {
        guard batch > 0, numTokens > 0, M > 0, K > 0 else {
            throw FlintMetalError.invalidParameter("batch, numTokens, M, and K must be positive")
        }
        guard inputOffset >= 0, weightOffset >= 0, outputOffset >= 0 else {
            throw FlintMetalError.invalidParameter("Buffer offsets cannot be negative")
        }
        guard inputOffset % inDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("inputOffset must be aligned to \(inDtype.elementByteWidth) bytes")
        }
        guard weightOffset % inDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("weightOffset must be aligned to \(inDtype.elementByteWidth) bytes")
        }
        guard outputOffset % outDtype.elementByteWidth == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset must be aligned to \(outDtype.elementByteWidth) bytes")
        }

        let inBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), try Self.checkedMul(K, inDtype.elementByteWidth))
        let wBytes = try Self.checkedMul(M, try Self.checkedMul(K, inDtype.elementByteWidth))
        let outBytes = try Self.checkedMul(try Self.checkedMul(batch, numTokens), try Self.checkedMul(M, outDtype.elementByteWidth))

        guard input.length >= (try Self.checkedAdd(inputOffset, inBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: inputOffset + inBytes)
        }
        guard weight.length >= (try Self.checkedAdd(weightOffset, wBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "weight", actual: weight.length, required: weightOffset + wBytes)
        }
        guard output.length >= (try Self.checkedAdd(outputOffset, outBytes)) else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + outBytes)
        }

        var mCopy = try Self.checkedToUInt32(M, name: "M")
        var kCopy = try Self.checkedToUInt32(K, name: "K")
        var tokensCopy = try Self.checkedToUInt32(numTokens, name: "numTokens")

        let pso: MTLComputePipelineState
        switch (inDtype, outDtype) {
        case (.bfloat16, .bfloat16): pso = psoGemmBF16
        case (.float32, .float32): pso = psoGemmF32
        case (.float32, .bfloat16): pso = psoGemmF32InBF16Out
        case (.bfloat16, .float32):
            throw FlintMetalError.invalidParameter("bfloat16 input with float32 output is not supported for GEMM")
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBytes(&mCopy, length: 4, index: 3)
        encoder.setBytes(&kCopy, length: 4, index: 4)
        encoder.setBytes(&tokensCopy, length: 4, index: 5)

        // 3D dispatch: width is M * 32 (each SIMD group of 32 threads computes 1 output scalar m)
        let totalThreadsX = try Self.checkedMul(M, 32)
        let gridSize = MTLSize(width: totalThreadsX, height: numTokens, depth: batch)
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Encodes pointwise SiLU activation: y = x / (1 + exp(-x))
    public func encodeSiLU(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        totalElements: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard totalElements > 0 else {
            throw FlintMetalError.invalidParameter("totalElements must be positive")
        }
        let byteCount = try Self.checkedMul(totalElements, dtype.elementByteWidth)
        guard input.length >= inputOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: inputOffset + byteCount)
        }
        guard output.length >= outputOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + byteCount)
        }

        var countCopy = try Self.checkedToUInt32(totalElements, name: "totalElements")
        let pso = (dtype == .bfloat16) ? psoSiluBF16 : psoSiluF32

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&countCopy, length: 4, index: 2)

        let gridSize = MTLSize(width: totalElements, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, totalElements), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Encodes elementwise multiplication: y = a * b
    public func encodeMul(
        into encoder: MTLComputeCommandEncoder,
        inputA: MTLBuffer,
        offsetA: Int = 0,
        inputB: MTLBuffer,
        offsetB: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        totalElements: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard totalElements > 0 else {
            throw FlintMetalError.invalidParameter("totalElements must be positive")
        }
        let byteCount = try Self.checkedMul(totalElements, dtype.elementByteWidth)
        guard inputA.length >= offsetA + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "inputA", actual: inputA.length, required: offsetA + byteCount)
        }
        guard inputB.length >= offsetB + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "inputB", actual: inputB.length, required: offsetB + byteCount)
        }
        guard output.length >= outputOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + byteCount)
        }

        var countCopy = try Self.checkedToUInt32(totalElements, name: "totalElements")
        let pso = (dtype == .bfloat16) ? psoMulBF16 : psoMulF32

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inputA, offset: offsetA, index: 0)
        encoder.setBuffer(inputB, offset: offsetB, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBytes(&countCopy, length: 4, index: 3)

        let gridSize = MTLSize(width: totalElements, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, totalElements), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Encodes fused SwiGLU activation: y = SiLU(gate) * up
    public func encodeSwiGLU(
        into encoder: MTLComputeCommandEncoder,
        gate: MTLBuffer,
        gateOffset: Int = 0,
        up: MTLBuffer,
        upOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        totalElements: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard totalElements > 0 else {
            throw FlintMetalError.invalidParameter("totalElements must be positive")
        }
        let byteCount = try Self.checkedMul(totalElements, dtype.elementByteWidth)
        guard gate.length >= gateOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "gate", actual: gate.length, required: gateOffset + byteCount)
        }
        guard up.length >= upOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "up", actual: up.length, required: upOffset + byteCount)
        }
        guard output.length >= outputOffset + byteCount else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + byteCount)
        }

        var countCopy = try Self.checkedToUInt32(totalElements, name: "totalElements")
        let pso = (dtype == .bfloat16) ? psoSwigluBF16 : psoSwigluF32

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(up, offset: upOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBytes(&countCopy, length: 4, index: 3)

        let gridSize = MTLSize(width: totalElements, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, totalElements), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    // MARK: - Full SwiGLU Forward Pass (Zero Heap Allocation)

    /// Executes full SwiGLU MLP forward pass with bounded scratch reuse.
    ///
    /// Pipeline:
    /// 1. Dequantize gateWeight -> weightScratch
    /// 2. GEMM input @ gateWeight^T -> gateScratch
    /// 3. Dequantize upWeight -> weightScratch (reusing scratch)
    /// 4. GEMM input @ upWeight^T -> upScratch
    /// 5. Fused SwiGLU silu(gateScratch) * upScratch -> actScratch
    /// 6. Dequantize downWeight -> weightScratch (reusing scratch)
    /// 7. GEMM actScratch @ downWeight^T -> output
    public func encodeForwardMLP(
        into encoder: MTLComputeCommandEncoder,
        gateWeight: MTLBuffer, gateScales: MTLBuffer, gateBiases: MTLBuffer,
        upWeight: MTLBuffer, upScales: MTLBuffer, upBiases: MTLBuffer,
        downWeight: MTLBuffer, downScales: MTLBuffer, downBiases: MTLBuffer,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        hiddenDim: Int = 2048,
        intermediateDim: Int,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard batch > 0, numTokens > 0 else {
            throw FlintMetalError.invalidParameter("batch and numTokens must be positive")
        }
        let totalActElements = try Self.checkedMul(try Self.checkedMul(batch, numTokens), intermediateDim)
        let actBytes = try Self.checkedMul(totalActElements, dtype.elementByteWidth)

        guard gateScratch.length >= actBytes else {
            throw FlintMetalError.bufferTooSmall(name: "gateScratch", actual: gateScratch.length, required: actBytes)
        }
        guard upScratch.length >= actBytes else {
            throw FlintMetalError.bufferTooSmall(name: "upScratch", actual: upScratch.length, required: actBytes)
        }
        guard actScratch.length >= actBytes else {
            throw FlintMetalError.bufferTooSmall(name: "actScratch", actual: actScratch.length, required: actBytes)
        }

        // 1. Gate Projection
        try dequantKernel.encodeDequantize(
            into: encoder,
            packedWeights: gateWeight,
            scales: gateScales,
            biases: gateBiases,
            output: weightScratch,
            rows: intermediateDim,
            cols: hiddenDim,
            bits: .four,
            outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
        )
        try encodeGEMM(
            into: encoder,
            input: input,
            inputOffset: inputOffset,
            weight: weightScratch,
            output: gateScratch,
            M: intermediateDim,
            K: hiddenDim,
            batch: batch,
            numTokens: numTokens,
            inDtype: dtype,
            outDtype: dtype
        )

        // 2. Up Projection (reusing weightScratch)
        try dequantKernel.encodeDequantize(
            into: encoder,
            packedWeights: upWeight,
            scales: upScales,
            biases: upBiases,
            output: weightScratch,
            rows: intermediateDim,
            cols: hiddenDim,
            bits: .four,
            outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
        )
        try encodeGEMM(
            into: encoder,
            input: input,
            inputOffset: inputOffset,
            weight: weightScratch,
            output: upScratch,
            M: intermediateDim,
            K: hiddenDim,
            batch: batch,
            numTokens: numTokens,
            inDtype: dtype,
            outDtype: dtype
        )

        // 3. Fused SwiGLU Activation: SiLU(gate) * up
        try encodeSwiGLU(
            into: encoder,
            gate: gateScratch,
            up: upScratch,
            output: actScratch,
            totalElements: totalActElements,
            dtype: dtype
        )

        // 4. Down Projection (reusing weightScratch)
        try dequantKernel.encodeDequantize(
            into: encoder,
            packedWeights: downWeight,
            scales: downScales,
            biases: downBiases,
            output: weightScratch,
            rows: hiddenDim,
            cols: intermediateDim,
            bits: .four,
            outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
        )
        try encodeGEMM(
            into: encoder,
            input: actScratch,
            weight: weightScratch,
            output: output,
            outputOffset: outputOffset,
            M: hiddenDim,
            K: intermediateDim,
            batch: batch,
            numTokens: numTokens,
            inDtype: dtype,
            outDtype: dtype
        )
    }

    /// Convenience wrapper for Dense MLP (layers 0 and 1).
    public func encodeForwardDenseMLP(
        into encoder: MTLComputeCommandEncoder,
        weights: FlintDenseMLPWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard let gw = weights.gateWeight.metalBuffer,
              let gs = weights.gateScales.metalBuffer,
              let gb = weights.gateBiases.metalBuffer,
              let uw = weights.upWeight.metalBuffer,
              let us = weights.upScales.metalBuffer,
              let ub = weights.upBiases.metalBuffer,
              let dw = weights.downWeight.metalBuffer,
              let ds = weights.downScales.metalBuffer,
              let db = weights.downBiases.metalBuffer else {
            throw FlintMetalError.invalidParameter("Dense MLP weights must be resident in Metal shared memory")
        }

        try encodeForwardMLP(
            into: encoder,
            gateWeight: gw, gateScales: gs, gateBiases: gb,
            upWeight: uw, upScales: us, upBiases: ub,
            downWeight: dw, downScales: ds, downBiases: db,
            input: input, inputOffset: inputOffset,
            output: output, outputOffset: outputOffset,
            hiddenDim: 2048,
            intermediateDim: 6144,
            batch: batch,
            numTokens: numTokens,
            dtype: dtype
        )
    }

    /// Convenience wrapper for Shared Expert MLP (layers 2..48).
    public func encodeForwardSharedExpertMLP(
        into encoder: MTLComputeCommandEncoder,
        weights: FlintMoEResidentWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16
    ) throws {
        guard let gw = weights.sharedGateWeight.metalBuffer,
              let gs = weights.sharedGateScales.metalBuffer,
              let gb = weights.sharedGateBiases.metalBuffer,
              let uw = weights.sharedUpWeight.metalBuffer,
              let us = weights.sharedUpScales.metalBuffer,
              let ub = weights.sharedUpBiases.metalBuffer,
              let dw = weights.sharedDownWeight.metalBuffer,
              let ds = weights.sharedDownScales.metalBuffer,
              let db = weights.sharedDownBiases.metalBuffer else {
            throw FlintMetalError.invalidParameter("Shared expert weights must be resident in Metal shared memory")
        }

        try encodeForwardMLP(
            into: encoder,
            gateWeight: gw, gateScales: gs, gateBiases: gb,
            upWeight: uw, upScales: us, upBiases: ub,
            downWeight: dw, downScales: ds, downBiases: db,
            input: input, inputOffset: inputOffset,
            output: output, outputOffset: outputOffset,
            hiddenDim: 2048,
            intermediateDim: 1024,
            batch: batch,
            numTokens: numTokens,
            dtype: dtype
        )
    }

    /// Standalone forward execution with execution timing.
    @discardableResult
    public func executeForwardDenseMLP(
        weights: FlintDenseMLPWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16,
        waitUntilCompleted: Bool = true
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder")
        }

        do {
            try encodeForwardDenseMLP(
                into: encoder,
                weights: weights,
                input: input,
                inputOffset: inputOffset,
                output: output,
                outputOffset: outputOffset,
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
            return cb.gpuEndTime - cb.gpuStartTime
        }
        return 0.0
    }

    /// Standalone forward execution for shared expert.
    @discardableResult
    public func executeForwardSharedExpertMLP(
        weights: FlintMoEResidentWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16,
        waitUntilCompleted: Bool = true
    ) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder")
        }

        do {
            try encodeForwardSharedExpertMLP(
                into: encoder,
                weights: weights,
                input: input,
                inputOffset: inputOffset,
                output: output,
                outputOffset: outputOffset,
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
            return cb.gpuEndTime - cb.gpuStartTime
        }
        return 0.0
    }

    // MARK: - Diagnostic Intermediate Inspection

    /// Executes step-by-step forward pass allocating dedicated buffers for all intermediate tensors.
    public func executeForwardWithIntermediates(
        gateWeight: MTLBuffer, gateScales: MTLBuffer, gateBiases: MTLBuffer,
        upWeight: MTLBuffer, upScales: MTLBuffer, upBiases: MTLBuffer,
        downWeight: MTLBuffer, downScales: MTLBuffer, downBiases: MTLBuffer,
        input: MTLBuffer,
        hiddenDim: Int = 2048,
        intermediateDim: Int,
        batch: Int = 1,
        numTokens: Int,
        dtype: Dtype = .bfloat16
    ) throws -> Intermediates {
        let totalActElements = batch * numTokens * intermediateDim
        let actBytes = totalActElements * dtype.elementByteWidth
        let outBytes = batch * numTokens * hiddenDim * dtype.elementByteWidth

        let gateBuf = try context.makeBuffer(byteCount: actBytes)
        let upBuf = try context.makeBuffer(byteCount: actBytes)
        let siluBuf = try context.makeBuffer(byteCount: actBytes)
        let productBuf = try context.makeBuffer(byteCount: actBytes)
        let downBuf = try context.makeBuffer(byteCount: outBytes)

        guard let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder")
        }

        do {
            // 1. Gate Proj
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: gateWeight, scales: gateScales, biases: gateBiases,
                output: weightScratch,
                rows: intermediateDim, cols: hiddenDim, bits: .four,
                outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
            )
            try encodeGEMM(into: encoder, input: input, weight: weightScratch, output: gateBuf, M: intermediateDim, K: hiddenDim, batch: batch, numTokens: numTokens, inDtype: dtype, outDtype: dtype)

            // 2. Up Proj
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: upWeight, scales: upScales, biases: upBiases,
                output: weightScratch,
                rows: intermediateDim, cols: hiddenDim, bits: .four,
                outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
            )
            try encodeGEMM(into: encoder, input: input, weight: weightScratch, output: upBuf, M: intermediateDim, K: hiddenDim, batch: batch, numTokens: numTokens, inDtype: dtype, outDtype: dtype)

            // 3. SiLU(gate)
            try encodeSiLU(into: encoder, input: gateBuf, output: siluBuf, totalElements: totalActElements, dtype: dtype)

            // 4. Gated product: silu(gate) * up
            try encodeMul(into: encoder, inputA: siluBuf, inputB: upBuf, output: productBuf, totalElements: totalActElements, dtype: dtype)

            // 5. Down Proj
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: downWeight, scales: downScales, biases: downBiases,
                output: weightScratch,
                rows: hiddenDim, cols: intermediateDim, bits: .four,
                outDtype: (dtype == .bfloat16) ? .bfloat16 : .float32
            )
            try encodeGEMM(into: encoder, input: productBuf, weight: weightScratch, output: downBuf, M: hiddenDim, K: intermediateDim, batch: batch, numTokens: numTokens, inDtype: dtype, outDtype: dtype)

            encoder.endEncoding()
        } catch {
            encoder.endEncoding()
            throw error
        }

        cb.commit()
        cb.waitUntilCompleted()

        return Intermediates(
            gateProj: gateBuf,
            upProj: upBuf,
            siluGate: siluBuf,
            gatedProduct: productBuf,
            downProj: downBuf
        )
    }

    // MARK: - Embedded Metal Shading Language Source

    public static let embeddedShaderSource: String = """
#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Native Dense and Shared Expert SwiGLU MLP Metal Compute Kernels
//
// Mathematical Specification:
//   - Dense MLP (Layers 0, 1): hidden_dim = 2048, intermediate_dim = 6144
//   - Shared Expert MLP (Layers 2..48): hidden_dim = 2048, intermediate_dim = 1024
//   - gate = gate_proj(x)       [batch, num_tokens, intermediate_dim]
//   - up = up_proj(x)           [batch, num_tokens, intermediate_dim]
//   - activated = SiLU(gate) * up
//   - out = down_proj(activated) [batch, num_tokens, hidden_dim]
//
// Accumulation Contract:
//   - Matrix-vector projections evaluate with FP32 SIMD-coalesced reduction.
//   - SiLU evaluates in FP32: silu(g) = g / (1.0f + exp(-g)).
//   - Product silu(g) * u evaluates in FP32 prior to casting.
//   - Zero data races, zero atomic operations, zero GPU heap allocations.
// ==============================================================================

// MARK: - Matrix-Vector Projections (GEMM: Y = X * W^T)

/// GEMM for bfloat16 activations and bfloat16 weights -> bfloat16 output.
/// Dispatch: 3D grid (M * 32, num_tokens, batch), threadsPerThreadgroup: (256, 1, 1).
/// Each SIMD group of 32 threads computes one output scalar Y[b, t, m] with 100% coalesced memory loads.
kernel void flint_gemm_bf16(
    device const bfloat* in_x [[buffer(0)]],         // [batch, num_tokens, K]
    device const bfloat* weight [[buffer(1)]],       // [M, K]
    device bfloat* out_y [[buffer(2)]],              // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const bfloat* x_vec = in_x + (b * num_tokens + t) * K;
    device const bfloat* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += float(x_vec[k]) * float(w_row[k]);
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = bfloat(total);
    }
}

/// GEMM for float32 activations and float32 weights -> float32 output.
kernel void flint_gemm_f32(
    device const float* in_x [[buffer(0)]],          // [batch, num_tokens, K]
    device const float* weight [[buffer(1)]],        // [M, K]
    device float* out_y [[buffer(2)]],               // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const float* x_vec = in_x + (b * num_tokens + t) * K;
    device const float* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += x_vec[k] * w_row[k];
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = total;
    }
}

/// GEMM for float32 activations and float32 weights -> bfloat16 output cast.
kernel void flint_gemm_f32_in_bf16_out(
    device const float* in_x [[buffer(0)]],          // [batch, num_tokens, K]
    device const float* weight [[buffer(1)]],        // [M, K]
    device bfloat* out_y [[buffer(2)]],              // [batch, num_tokens, M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint global_lane = pos.x;
    uint m = global_lane / 32;
    uint lane = global_lane % 32;
    uint t = pos.y;
    uint b = pos.z;

    if (m >= M || t >= num_tokens) {
        return;
    }

    device const float* x_vec = in_x + (b * num_tokens + t) * K;
    device const float* w_row = weight + m * K;

    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        acc += x_vec[k] * w_row[k];
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        out_y[(b * num_tokens + t) * M + m] = bfloat(total);
    }
}

// MARK: - Pointwise Activations & Operations

/// SiLU activation: y = x / (1.0 + exp(-x))
kernel void flint_silu_bf16(
    device const bfloat* in_x [[buffer(0)]],
    device bfloat* out_y [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float x = float(in_x[pos]);
    float silu_x = x / (1.0f + exp(-x));
    out_y[pos] = bfloat(silu_x);
}

kernel void flint_silu_f32(
    device const float* in_x [[buffer(0)]],
    device float* out_y [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float x = in_x[pos];
    float silu_x = x / (1.0f + exp(-x));
    out_y[pos] = silu_x;
}

/// Elementwise multiplication: y = a * b
kernel void flint_elementwise_mul_bf16(
    device const bfloat* in_a [[buffer(0)]],
    device const bfloat* in_b [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float a = float(in_a[pos]);
    float b = float(in_b[pos]);
    out_y[pos] = bfloat(a * b);
}

kernel void flint_elementwise_mul_f32(
    device const float* in_a [[buffer(0)]],
    device const float* in_b [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    out_y[pos] = in_a[pos] * in_b[pos];
}

/// Fused SwiGLU activation: y = SiLU(gate) * up
kernel void flint_swiglu_bf16(
    device const bfloat* in_gate [[buffer(0)]],
    device const bfloat* in_up [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float g = float(in_gate[pos]);
    float u = float(in_up[pos]);
    float silu_g = g / (1.0f + exp(-g));
    out_y[pos] = bfloat(silu_g * u);
}

kernel void flint_swiglu_f32(
    device const float* in_gate [[buffer(0)]],
    device const float* in_up [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    float g = in_gate[pos];
    float u = in_up[pos];
    float silu_g = g / (1.0f + exp(-g));
    out_y[pos] = silu_g * u;
}

"""
}
