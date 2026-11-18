import Foundation
import Metal
import QuartzCore

/// Runner for end-to-end Flint Transformer Decoder Layers on Apple Silicon Metal.
///
/// Implements full layer forward execution for:
/// - Layer 0 (Dense): Full causal attention without RoPE, learned sinks, Q/K per-head RMSNorm(128),
///   input RMSNorm(2048), post-attention RMSNorm(2048), Dense SwiGLU MLP (6144), and residuals.
/// - Layer 2 (MoE): Sliding-window attention (window 513) with RoPE, learned sinks, 6 routed experts (3-bit)
///   plus 1 shared expert (4-bit), fixed router-order reduction, and residuals.
public final class FlintDecoderLayerRunner: @unchecked Sendable {

    /// Timing and execution breakdown for Layer 0 forward.
    public struct Layer0TimingBreakdown: Sendable {
        public let totalGpuElapsedUs: Double
        public let totalWallClockUs: Double
        public let commandBuffersCommitted: Int
        public let gpuWaits: Int
    }

    /// Holds all 10 captured intermediate/final stages of a Layer 0 forward pass for numerical verification.
    public struct Layer0Intermediates {
        public let input: MTLBuffer
        public let norm1: MTLBuffer
        public let qRaw: MTLBuffer
        public let kRaw: MTLBuffer
        public let vRaw: MTLBuffer
        public let qNorm: MTLBuffer
        public let kNorm: MTLBuffer
        public let attnOut: MTLBuffer
        public let attnFlat: MTLBuffer
        public let attnProj: MTLBuffer
        public let res1: MTLBuffer
        public let norm2: MTLBuffer
        public let mlpOut: MTLBuffer
        public let finalOut: MTLBuffer
        public let timing: Layer0TimingBreakdown
    }

    /// Timing and execution breakdown for Layer 2 MoE forward.
    public struct Layer2TimingBreakdown: Sendable {
        public let cb1AttentionNormRouterUs: Double
        public let diskWeightReadUs: Double
        public let cb2RoutedExpertsUs: Double
        public let cb3SharedReductionRes2Us: Double
        public let totalGpuElapsedUs: Double
        public let totalWallClockUs: Double
        public let commandBuffersCommitted: Int
        public let gpuWaits: Int
        public let routedBytesRead: Int
    }

    /// Holds all 14 captured intermediate/final stages of a Layer 2 MoE forward pass for numerical verification.
    public struct Layer2Intermediates {
        public let input: MTLBuffer
        public let norm1: MTLBuffer
        public let qRaw: MTLBuffer
        public let kRaw: MTLBuffer
        public let vRaw: MTLBuffer
        public let qNorm: MTLBuffer
        public let kNorm: MTLBuffer
        public let qRope: MTLBuffer
        public let kRope: MTLBuffer
        public let attnOut: MTLBuffer
        public let attnFlat: MTLBuffer
        public let attnProj: MTLBuffer
        public let res1: MTLBuffer
        public let norm2: MTLBuffer
        public let routerExperts: [Int32]
        public let routerWeights: [Float]
        public let routedContributions: MTLBuffer
        public let sharedOut: MTLBuffer
        public let moeOut: MTLBuffer
        public let finalOut: MTLBuffer
        public let timing: Layer2TimingBreakdown
    }

    /// Temporary Metal buffers for loading 3-bit weights of 1 routed expert.
    public struct ExpertTemporaryBuffers: @unchecked Sendable {
        public let gateWeight: MTLBuffer
        public let gateScales: MTLBuffer
        public let gateBiases: MTLBuffer
        public let upWeight: MTLBuffer
        public let upScales: MTLBuffer
        public let upBiases: MTLBuffer
        public let downWeight: MTLBuffer
        public let downScales: MTLBuffer
        public let downBiases: MTLBuffer
    }

    public let context: FlintMetalContext
    public let normKernel: FlintRMSNormKernel
    public let attentionKernel: FlintAttentionKernel
    public let dequantKernel: FlintDequantizeKernel
    public let mlpKernel: FlintMLPKernel
    public let ropeKernel: FlintRoPEKernel
    public let routerKernel: FlintRouterKernel
    public let reductionKernel: FlintReductionKernel

    // Preallocated temporary expert buffers (6 sets, ~7.875 MiB total)
    public let expertBuffers: [ExpertTemporaryBuffers]

    // Preallocated layer activation buffers (for up to 16 tokens, ~1.5 MiB total)
    public let norm1Buf: MTLBuffer
    public let qRawBuf: MTLBuffer
    public let kRawBuf: MTLBuffer
    public let vRawBuf: MTLBuffer
    public let qNormBuf: MTLBuffer
    public let kNormBuf: MTLBuffer
    public let qTransposedBuf: MTLBuffer
    public let kTransposedBuf: MTLBuffer
    public let vTransposedBuf: MTLBuffer
    public let qRopeBuf: MTLBuffer
    public let kRopeBuf: MTLBuffer
    public let attnOutBuf: MTLBuffer
    public let attnFlatBuf: MTLBuffer
    public let attnProjBuf: MTLBuffer
    public let res1Buf: MTLBuffer
    public let norm2Buf: MTLBuffer
    public let mlpOutBuf: MTLBuffer
    public let routerExpertsBuf: MTLBuffer
    public let routerWeightsBuf: MTLBuffer
    public let routedContributionsBuf: MTLBuffer
    public let sharedOutBuf: MTLBuffer
    public let moeOutBuf: MTLBuffer
    public let finalOutBuf: MTLBuffer

    private let psoTranspose0213BF16: MTLComputePipelineState
    private let psoTranspose0213BackBF16: MTLComputePipelineState
    private let psoResidualAddBF16: MTLComputePipelineState

    public static func loadLayerShaderSource() -> String {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_layer.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(sourcePath)
        if let diskSource = try? String(contentsOf: fileURL, encoding: .utf8), !diskSource.isEmpty {
            return diskSource
        }
        return embeddedLayerShaderSource
    }

    public init(context: FlintMetalContext, mlpKernel: FlintMLPKernel? = nil) throws {
        self.context = context
        self.normKernel = try FlintRMSNormKernel(context: context)
        self.attentionKernel = try FlintAttentionKernel(context: context)
        self.dequantKernel = try FlintDequantizeKernel(context: context)
        self.mlpKernel = try mlpKernel ?? FlintMLPKernel(context: context)
        self.ropeKernel = try FlintRoPEKernel(context: context)
        self.routerKernel = try FlintRouterKernel(context: context)
        self.reductionKernel = try FlintReductionKernel(context: context)

        // Preallocate 6 sets of temporary expert buffers (~7.875 MiB total)
        var expList = [ExpertTemporaryBuffers]()
        for _ in 0..<6 {
            expList.append(ExpertTemporaryBuffers(
                gateWeight: try context.makeBuffer(byteCount: 393_216),
                gateScales: try context.makeBuffer(byteCount: 32_768),
                gateBiases: try context.makeBuffer(byteCount: 32_768),
                upWeight: try context.makeBuffer(byteCount: 393_216),
                upScales: try context.makeBuffer(byteCount: 32_768),
                upBiases: try context.makeBuffer(byteCount: 32_768),
                downWeight: try context.makeBuffer(byteCount: 393_216),
                downScales: try context.makeBuffer(byteCount: 32_768),
                downBiases: try context.makeBuffer(byteCount: 32_768)
            ))
        }
        self.expertBuffers = expList

        // Preallocate layer activation buffers for up to 16 tokens (~1.5 MiB total)
        let maxT = 16
        self.norm1Buf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.qRawBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.kRawBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.vRawBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.qNormBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.kNormBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.qTransposedBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.kTransposedBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.vTransposedBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.qRopeBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.kRopeBuf = try context.makeBuffer(byteCount: maxT * 512 * 2)
        self.attnOutBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.attnFlatBuf = try context.makeBuffer(byteCount: maxT * 4096 * 2)
        self.attnProjBuf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.res1Buf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.norm2Buf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.mlpOutBuf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.routerExpertsBuf = try context.makeBuffer(byteCount: maxT * 6 * 4)
        self.routerWeightsBuf = try context.makeBuffer(byteCount: maxT * 6 * 4)
        self.routedContributionsBuf = try context.makeBuffer(byteCount: maxT * 6 * 2048 * 2)
        self.sharedOutBuf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.moeOutBuf = try context.makeBuffer(byteCount: maxT * 2048 * 2)
        self.finalOutBuf = try context.makeBuffer(byteCount: maxT * 2048 * 2)

        let layerSource = Self.loadLayerShaderSource()
        let layerLibrary = try context.makeLibrary(source: layerSource)

        self.psoTranspose0213BF16 = try context.computePipelineState(functionName: "flint_transpose_0213_bf16", library: layerLibrary)
        self.psoTranspose0213BackBF16 = try context.computePipelineState(functionName: "flint_transpose_0213_back_bf16", library: layerLibrary)
        self.psoResidualAddBF16 = try context.computePipelineState(functionName: "flint_residual_add_bf16", library: layerLibrary)
    }

    // MARK: - Reshaping & Residual Primitives

    /// Transposes [batch, T, H, D] -> [batch, H, T, D] in bfloat16.
    public func encodeTranspose0213(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int,
        T: Int,
        H: Int,
        D: Int
    ) throws {
        var bCopy = UInt32(batch)
        var tCopy = UInt32(T)
        var hCopy = UInt32(H)
        var dCopy = UInt32(D)

        encoder.setComputePipelineState(psoTranspose0213BF16)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&bCopy, length: 4, index: 2)
        encoder.setBytes(&tCopy, length: 4, index: 3)
        encoder.setBytes(&hCopy, length: 4, index: 4)
        encoder.setBytes(&dCopy, length: 4, index: 5)

        let gridSize = MTLSize(width: D, height: T, depth: batch * H)
        let tgSize = MTLSize(width: min(32, D), height: min(8, T), depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Transposes [batch, H, T, D] -> [batch, T, H, D] in bfloat16.
    public func encodeTranspose0213Back(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int,
        H: Int,
        T: Int,
        D: Int
    ) throws {
        var bCopy = UInt32(batch)
        var hCopy = UInt32(H)
        var tCopy = UInt32(T)
        var dCopy = UInt32(D)

        encoder.setComputePipelineState(psoTranspose0213BackBF16)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&bCopy, length: 4, index: 2)
        encoder.setBytes(&hCopy, length: 4, index: 3)
        encoder.setBytes(&tCopy, length: 4, index: 4)
        encoder.setBytes(&dCopy, length: 4, index: 5)

        let gridSize = MTLSize(width: D, height: H, depth: batch * T)
        let tgSize = MTLSize(width: min(32, D), height: min(8, H), depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    /// Residual addition: y = a + b
    public func encodeResidualAdd(
        into encoder: MTLComputeCommandEncoder,
        inputA: MTLBuffer,
        offsetA: Int = 0,
        inputB: MTLBuffer,
        offsetB: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        totalElements: Int
    ) throws {
        var countCopy = UInt32(totalElements)

        encoder.setComputePipelineState(psoResidualAddBF16)
        encoder.setBuffer(inputA, offset: offsetA, index: 0)
        encoder.setBuffer(inputB, offset: offsetB, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBytes(&countCopy, length: 4, index: 3)

        let gridSize = MTLSize(width: totalElements, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, totalElements), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    // MARK: - Layer 0 Full Dense Forward Pass

    /// Executes Layer 0 forward pass with capture of all 10 intermediate stages.
    public func executeLayer0ForwardWithIntermediates(
        weights: FlintResidentWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int
    ) throws -> Layer0Intermediates {
        let tWallStart = CACurrentMediaTime()
        guard batch > 0, numTokens > 0 else {
            throw FlintMetalError.invalidParameter("batch (\(batch)) and numTokens (\(numTokens)) must be positive")
        }
        let (totalTokens, ofTokens) = batch.multipliedReportingOverflow(by: numTokens)
        guard !ofTokens else {
            throw FlintMetalError.invalidParameter("batch * numTokens overflowed integer bounds")
        }
        guard totalTokens <= 16 else {
            throw FlintMetalError.invalidParameter("Total tokens (\(totalTokens)) exceeds preallocated capacity (16)")
        }
        guard inputOffset >= 0, inputOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("inputOffset (\(inputOffset)) must be non-negative and 2-byte aligned for bfloat16")
        }
        let (reqInTokens, of1) = totalTokens.multipliedReportingOverflow(by: 2048)
        let (requiredInputBytes, of2) = reqInTokens.multipliedReportingOverflow(by: 2)
        guard !of1, !of2 else {
            throw FlintMetalError.invalidParameter("Input byte size calculation overflowed")
        }
        guard input.length >= inputOffset + requiredInputBytes else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: inputOffset + requiredInputBytes)
        }

        guard let attnWeights = weights.attentionWeights(forLayer: 0) else {
            throw FlintSafetensorsError.tensorNotFound(name: "model.layers.0 attention weights")
        }
        guard let denseWeights = weights.denseMLPWeights(forLayer: 0) else {
            throw FlintSafetensorsError.tensorNotFound(name: "model.layers.0 dense MLP weights")
        }

        try mlpKernel.acquireInFlight()
        defer { mlpKernel.releaseInFlight() }

        let hiddenDim = 2048
        let qDim = 32 * 128 // 4096
        let kvDim = 4 * 128 // 512
        let elementBytes = 2 // bfloat16

        let hiddenBytes = totalTokens * hiddenDim * elementBytes
        let qBytes = totalTokens * qDim * elementBytes
        let kvBytes = totalTokens * kvDim * elementBytes

        // Use preallocated buffers if within capacity (<= 16 tokens)
        let norm1Buf = (totalTokens <= 16) ? self.norm1Buf : try context.makeBuffer(byteCount: hiddenBytes)
        let qRawBuf = (totalTokens <= 16) ? self.qRawBuf : try context.makeBuffer(byteCount: qBytes)
        let kRawBuf = (totalTokens <= 16) ? self.kRawBuf : try context.makeBuffer(byteCount: kvBytes)
        let vRawBuf = (totalTokens <= 16) ? self.vRawBuf : try context.makeBuffer(byteCount: kvBytes)
        let qNormBuf = (totalTokens <= 16) ? self.qNormBuf : try context.makeBuffer(byteCount: qBytes)
        let kNormBuf = (totalTokens <= 16) ? self.kNormBuf : try context.makeBuffer(byteCount: kvBytes)
        let qTransposedBuf = (totalTokens <= 16) ? self.qTransposedBuf : try context.makeBuffer(byteCount: qBytes)
        let kTransposedBuf = (totalTokens <= 16) ? self.kTransposedBuf : try context.makeBuffer(byteCount: kvBytes)
        let vTransposedBuf = (totalTokens <= 16) ? self.vTransposedBuf : try context.makeBuffer(byteCount: kvBytes)
        let attnOutBuf = (totalTokens <= 16) ? self.attnOutBuf : try context.makeBuffer(byteCount: qBytes)
        let attnFlatBuf = (totalTokens <= 16) ? self.attnFlatBuf : try context.makeBuffer(byteCount: qBytes)
        let attnProjBuf = (totalTokens <= 16) ? self.attnProjBuf : try context.makeBuffer(byteCount: hiddenBytes)
        let res1Buf = (totalTokens <= 16) ? self.res1Buf : try context.makeBuffer(byteCount: hiddenBytes)
        let norm2Buf = (totalTokens <= 16) ? self.norm2Buf : try context.makeBuffer(byteCount: hiddenBytes)
        let mlpOutBuf = (totalTokens <= 16) ? self.mlpOutBuf : try context.makeBuffer(byteCount: hiddenBytes)
        let finalOutBuf = (totalTokens <= 16) ? self.finalOutBuf : try context.makeBuffer(byteCount: hiddenBytes)

        guard let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder")
        }

        do {
            // Stage 2: Input RMSNorm(2048)
            try normKernel.encode(
                commandEncoder: encoder,
                input: input, inputOffset: inputOffset,
                weight: attnWeights.inputLayerNorm.metalBuffer!,
                output: norm1Buf,
                rows: batch * numTokens,
                dimension: hiddenDim,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 3: Q/K/V Projections
            // Q Projection: 2048 -> 4096
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: attnWeights.qWeight.metalBuffer!,
                scales: attnWeights.qScales.metalBuffer!,
                biases: attnWeights.qBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: qDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: encoder,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: qRawBuf,
                M: qDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // K Projection: 2048 -> 512
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: attnWeights.kWeight.metalBuffer!,
                scales: attnWeights.kScales.metalBuffer!,
                biases: attnWeights.kBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: kvDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: encoder,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: kRawBuf,
                M: kvDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // V Projection: 2048 -> 512
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: attnWeights.vWeight.metalBuffer!,
                scales: attnWeights.vScales.metalBuffer!,
                biases: attnWeights.vBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: kvDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: encoder,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: vRawBuf,
                M: kvDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 4: Q/K Per-Head RMSNorm(128)
            try normKernel.encode(
                commandEncoder: encoder,
                input: qRawBuf,
                weight: attnWeights.qNorm.metalBuffer!,
                output: qNormBuf,
                rows: batch * numTokens * 32,
                dimension: 128,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            try normKernel.encode(
                commandEncoder: encoder,
                input: kRawBuf,
                weight: attnWeights.kNorm.metalBuffer!,
                output: kNormBuf,
                rows: batch * numTokens * 4,
                dimension: 128,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Layout preparation for attention:
            let qInputForAttn: MTLBuffer
            let kInputForAttn: MTLBuffer
            let vInputForAttn: MTLBuffer

            if numTokens > 1 {
                try encodeTranspose0213(into: encoder, input: qNormBuf, output: qTransposedBuf, batch: batch, T: numTokens, H: 32, D: 128)
                try encodeTranspose0213(into: encoder, input: kNormBuf, output: kTransposedBuf, batch: batch, T: numTokens, H: 4, D: 128)
                try encodeTranspose0213(into: encoder, input: vRawBuf, output: vTransposedBuf, batch: batch, T: numTokens, H: 4, D: 128)
                encoder.memoryBarrier(scope: .buffers)
                qInputForAttn = qTransposedBuf
                kInputForAttn = kTransposedBuf
                vInputForAttn = vTransposedBuf
            } else {
                qInputForAttn = qNormBuf
                kInputForAttn = kNormBuf
                vInputForAttn = vRawBuf
            }

            // Stage 5: Scaled Dot-Product Attention with Learned Sinks (No RoPE)
            if numTokens == 1 {
                try attentionKernel.encodeDecode(
                    encoder: encoder,
                    queries: qInputForAttn,
                    keys: kInputForAttn,
                    values: vInputForAttn,
                    sinks: attnWeights.softmaxOffset.metalBuffer!,
                    output: attnOutBuf,
                    batch: batch,
                    T_kv: 1,
                    isSliding: false,
                    dtype: .bfloat16
                )
            } else {
                try attentionKernel.encodePrefill(
                    encoder: encoder,
                    queries: qInputForAttn,
                    keys: kInputForAttn,
                    values: vInputForAttn,
                    sinks: attnWeights.softmaxOffset.metalBuffer!,
                    output: attnOutBuf,
                    batch: batch,
                    T_q: numTokens,
                    T_kv: numTokens,
                    isSliding: false,
                    dtype: .bfloat16
                )
            }
            encoder.memoryBarrier(scope: .buffers)

            // Transpose back if multi-token
            let attnForProj: MTLBuffer
            if numTokens > 1 {
                try encodeTranspose0213Back(into: encoder, input: attnOutBuf, output: attnFlatBuf, batch: batch, H: 32, T: numTokens, D: 128)
                encoder.memoryBarrier(scope: .buffers)
                attnForProj = attnFlatBuf
            } else {
                attnForProj = attnOutBuf
            }

            // Stage 6: Attention Output Projection (4096 -> 2048)
            try dequantKernel.encodeDequantize(
                into: encoder,
                packedWeights: attnWeights.oWeight.metalBuffer!,
                scales: attnWeights.oScales.metalBuffer!,
                biases: attnWeights.oBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: hiddenDim, cols: qDim,
                bits: .four, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: encoder,
                input: attnForProj, weight: mlpKernel.weightScratch, output: attnProjBuf,
                M: hiddenDim, K: qDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 7: First Residual State (res1 = x + attnProj)
            try encodeResidualAdd(
                into: encoder,
                inputA: input, offsetA: inputOffset,
                inputB: attnProjBuf,
                output: res1Buf,
                totalElements: batch * numTokens * hiddenDim
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 8: Post-Attention RMSNorm(2048)
            try normKernel.encode(
                commandEncoder: encoder,
                input: res1Buf,
                weight: attnWeights.postAttentionLayerNorm.metalBuffer!,
                output: norm2Buf,
                rows: batch * numTokens,
                dimension: hiddenDim,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 9: Dense SwiGLU MLP (2048 -> 6144 -> 2048)
            try mlpKernel.encodeForwardDenseMLP(
                into: encoder,
                weights: denseWeights,
                input: norm2Buf,
                output: mlpOutBuf,
                batch: batch,
                numTokens: numTokens,
                dtype: .bfloat16
            )
            encoder.memoryBarrier(scope: .buffers)

            // Stage 10: Final Residual State (finalOut = res1 + mlpOut)
            try encodeResidualAdd(
                into: encoder,
                inputA: res1Buf,
                inputB: mlpOutBuf,
                output: finalOutBuf,
                totalElements: batch * numTokens * hiddenDim
            )

            encoder.endEncoding()
        } catch {
            encoder.endEncoding()
            throw error
        }

        let tGpuStart = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        let tGpuEnd = CACurrentMediaTime()
        let tWallEnd = CACurrentMediaTime()

        let timing = Layer0TimingBreakdown(
            totalGpuElapsedUs: (tGpuEnd - tGpuStart) * 1_000_000.0,
            totalWallClockUs: (tWallEnd - tWallStart) * 1_000_000.0,
            commandBuffersCommitted: 1,
            gpuWaits: 1
        )

        return Layer0Intermediates(
            input: input,
            norm1: norm1Buf,
            qRaw: qRawBuf,
            kRaw: kRawBuf,
            vRaw: vRawBuf,
            qNorm: qNormBuf,
            kNorm: kNormBuf,
            attnOut: attnOutBuf,
            attnFlat: (numTokens > 1) ? attnFlatBuf : attnOutBuf,
            attnProj: attnProjBuf,
            res1: res1Buf,
            norm2: norm2Buf,
            mlpOut: mlpOutBuf,
            finalOut: finalOutBuf,
            timing: timing
        )
    }

    /// Full Layer 0 forward execution. Synchronously blocks until GPU completion.
    @discardableResult
    public func executeLayer0Forward(
        weights: FlintResidentWeights,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int
    ) throws -> Double {
        guard outputOffset >= 0, outputOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset (\(outputOffset)) must be non-negative and 2-byte aligned for bfloat16")
        }
        let (totalTokens, ofTokens) = batch.multipliedReportingOverflow(by: numTokens)
        guard !ofTokens else {
            throw FlintMetalError.invalidParameter("batch * numTokens overflowed integer bounds")
        }
        let (reqTokens, of1) = totalTokens.multipliedReportingOverflow(by: 2048)
        let (totalBytes, of2) = reqTokens.multipliedReportingOverflow(by: 2)
        guard !of1, !of2 else {
            throw FlintMetalError.invalidParameter("Output byte size calculation overflowed")
        }
        guard output.length >= outputOffset + totalBytes else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + totalBytes)
        }
        // Aliasing validation:
        if input.gpuAddress == output.gpuAddress {
            let inStart = inputOffset
            let inEnd = inputOffset + totalBytes
            let outStart = outputOffset
            let outEnd = outputOffset + totalBytes
            if max(inStart, outStart) < min(inEnd, outEnd) {
                throw FlintMetalError.invalidParameter("Input and output buffer ranges overlap in the same Metal buffer (in: [\(inStart)..<\(inEnd)], out: [\(outStart)..<\(outEnd)])")
            }
        }

        let start = CACurrentMediaTime()
        let inter = try executeLayer0ForwardWithIntermediates(
            weights: weights,
            input: input,
            inputOffset: inputOffset,
            batch: batch,
            numTokens: numTokens
        )

        let src = inter.finalOut.contents()
        let dst = output.contents().advanced(by: outputOffset)
        memcpy(dst, src, totalBytes)
        let elapsed = CACurrentMediaTime() - start
        return elapsed
    }

    // MARK: - Layer 2 MoE Forward Pass

    /// Executes Layer 2 MoE forward pass with capture of all 14 intermediate stages.
    public func executeLayer2ForwardWithIntermediates(
        weights: FlintResidentWeights,
        shard1File: FlintSafetensorsFile,
        input: MTLBuffer,
        inputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        cacheOffset: Int = 0
    ) throws -> Layer2Intermediates {
        let tWallStart = CACurrentMediaTime()
        guard batch == 1 else {
            throw FlintMetalError.unsupportedConfiguration("Layer 2 MoE currently supports batch == 1 only; received batch = \(batch)")
        }
        guard cacheOffset == 0 else {
            throw FlintMetalError.unsupportedConfiguration("Incremental KV caching with cacheOffset != 0 is unsupported in F3-01; single-layer forward only supports isolated sequence execution (cacheOffset == 0, received \(cacheOffset))")
        }
        guard numTokens > 0, numTokens <= 16 else {
            throw FlintMetalError.invalidParameter("numTokens (\(numTokens)) must be positive and <= 16")
        }
        guard inputOffset >= 0, inputOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("inputOffset (\(inputOffset)) must be non-negative and 2-byte aligned for bfloat16")
        }
        let (reqInTokens, of1) = numTokens.multipliedReportingOverflow(by: 2048)
        let (requiredInputBytes, of2) = reqInTokens.multipliedReportingOverflow(by: 2)
        guard !of1, !of2 else {
            throw FlintMetalError.invalidParameter("Input byte size calculation overflowed")
        }
        guard input.length >= inputOffset + requiredInputBytes else {
            throw FlintMetalError.bufferTooSmall(name: "input", actual: input.length, required: inputOffset + requiredInputBytes)
        }

        guard let attnWeights = weights.attentionWeights(forLayer: 2) else {
            throw FlintSafetensorsError.tensorNotFound(name: "model.layers.2 attention weights")
        }
        guard let moeWeights = weights.moeResidentWeights(forLayer: 2) else {
            throw FlintSafetensorsError.tensorNotFound(name: "model.layers.2 MoE resident weights")
        }

        try mlpKernel.acquireInFlight()
        defer { mlpKernel.releaseInFlight() }

        let hiddenDim = 2048
        let qDim = 32 * 128 // 4096
        let kvDim = 4 * 128 // 512

        // --- COMMAND BUFFER 1: Attention + Post-Attention Norm + Router ---
        guard let cb1 = context.queue.makeCommandBuffer(),
              let enc1 = cb1.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder for Layer 2 attention")
        }

        do {
            // Stage 2: Input RMSNorm(2048)
            try normKernel.encode(
                commandEncoder: enc1,
                input: input, inputOffset: inputOffset,
                weight: attnWeights.inputLayerNorm.metalBuffer!,
                output: norm1Buf,
                rows: batch * numTokens,
                dimension: hiddenDim,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 3: Q/K/V Projections
            // Q Projection: 2048 -> 4096
            try dequantKernel.encodeDequantize(
                into: enc1,
                packedWeights: attnWeights.qWeight.metalBuffer!,
                scales: attnWeights.qScales.metalBuffer!,
                biases: attnWeights.qBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: qDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: enc1,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: qRawBuf,
                M: qDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // K Projection: 2048 -> 512
            try dequantKernel.encodeDequantize(
                into: enc1,
                packedWeights: attnWeights.kWeight.metalBuffer!,
                scales: attnWeights.kScales.metalBuffer!,
                biases: attnWeights.kBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: kvDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: enc1,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: kRawBuf,
                M: kvDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // V Projection: 2048 -> 512
            try dequantKernel.encodeDequantize(
                into: enc1,
                packedWeights: attnWeights.vWeight.metalBuffer!,
                scales: attnWeights.vScales.metalBuffer!,
                biases: attnWeights.vBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: kvDim, cols: hiddenDim,
                bits: .four, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: enc1,
                input: norm1Buf, weight: mlpKernel.weightScratch, output: vRawBuf,
                M: kvDim, K: hiddenDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 4: Q/K Per-Head RMSNorm(128)
            try normKernel.encode(
                commandEncoder: enc1,
                input: qRawBuf,
                weight: attnWeights.qNorm.metalBuffer!,
                output: qNormBuf,
                rows: batch * numTokens * 32,
                dimension: 128,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            try normKernel.encode(
                commandEncoder: enc1,
                input: kRawBuf,
                weight: attnWeights.kNorm.metalBuffer!,
                output: kNormBuf,
                rows: batch * numTokens * 4,
                dimension: 128,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Transpose [batch, numTokens, H, D] -> [batch, H, numTokens, D]
            try encodeTranspose0213(into: enc1, input: qNormBuf, output: qTransposedBuf, batch: batch, T: numTokens, H: 32, D: 128)
            try encodeTranspose0213(into: enc1, input: kNormBuf, output: kTransposedBuf, batch: batch, T: numTokens, H: 4, D: 128)
            try encodeTranspose0213(into: enc1, input: vRawBuf, output: vTransposedBuf, batch: batch, T: numTokens, H: 4, D: 128)
            enc1.memoryBarrier(scope: .buffers)

            // Stage 5: RoPE on Q and K (split-half, base 10000.0)
            try ropeKernel.encode(
                commandEncoder: enc1,
                input: qTransposedBuf,
                output: qRopeBuf,
                batch: batch,
                numHeads: 32,
                seqLen: numTokens,
                offset: cacheOffset,
                base: 10000.0,
                dtype: .bfloat16
            )
            try ropeKernel.encode(
                commandEncoder: enc1,
                input: kTransposedBuf,
                output: kRopeBuf,
                batch: batch,
                numHeads: 4,
                seqLen: numTokens,
                offset: cacheOffset,
                base: 10000.0,
                dtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 6: Scaled Dot-Product Attention (sliding window 513) with Learned Sinks
            if numTokens == 1 {
                try attentionKernel.encodeDecode(
                    encoder: enc1,
                    queries: qRopeBuf,
                    keys: kRopeBuf,
                    values: vTransposedBuf,
                    sinks: attnWeights.softmaxOffset.metalBuffer!,
                    output: attnOutBuf,
                    batch: batch,
                    T_kv: 1,
                    isSliding: true,
                    dtype: .bfloat16
                )
            } else {
                try attentionKernel.encodePrefill(
                    encoder: enc1,
                    queries: qRopeBuf,
                    keys: kRopeBuf,
                    values: vTransposedBuf,
                    sinks: attnWeights.softmaxOffset.metalBuffer!,
                    output: attnOutBuf,
                    batch: batch,
                    T_q: numTokens,
                    T_kv: numTokens,
                    isSliding: true,
                    dtype: .bfloat16
                )
            }
            enc1.memoryBarrier(scope: .buffers)

            // Stage 7: Attention Output Projection (4096 -> 2048)
            try encodeTranspose0213Back(into: enc1, input: attnOutBuf, output: attnFlatBuf, batch: batch, H: 32, T: numTokens, D: 128)
            enc1.memoryBarrier(scope: .buffers)

            try dequantKernel.encodeDequantize(
                into: enc1,
                packedWeights: attnWeights.oWeight.metalBuffer!,
                scales: attnWeights.oScales.metalBuffer!,
                biases: attnWeights.oBiases.metalBuffer!,
                output: mlpKernel.weightScratch,
                rows: hiddenDim, cols: qDim,
                bits: .four, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)
            try mlpKernel.encodeGEMM(
                into: enc1,
                input: attnFlatBuf, weight: mlpKernel.weightScratch, output: attnProjBuf,
                M: hiddenDim, K: qDim, batch: batch, numTokens: numTokens,
                inDtype: .bfloat16, outDtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 8: First Residual (res1 = x + attnProj)
            try encodeResidualAdd(
                into: enc1,
                inputA: input, offsetA: inputOffset,
                inputB: attnProjBuf,
                output: res1Buf,
                totalElements: batch * numTokens * hiddenDim
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 9: Post-Attention RMSNorm(2048)
            try normKernel.encode(
                commandEncoder: enc1,
                input: res1Buf,
                weight: attnWeights.postAttentionLayerNorm.metalBuffer!,
                output: norm2Buf,
                rows: batch * numTokens,
                dimension: hiddenDim,
                epsilon: 1e-6,
                dtype: .bfloat16
            )
            enc1.memoryBarrier(scope: .buffers)

            // Stage 10: Router Forward
            try routerKernel.encodeForward(
                encoder: enc1,
                input: norm2Buf,
                gateWeight: moeWeights.routerGate.metalBuffer!,
                outExperts: routerExpertsBuf,
                outWeights: routerWeightsBuf,
                batch: batch,
                numTokens: numTokens,
                dtype: .bfloat16
            )

            enc1.endEncoding()
        } catch {
            enc1.endEncoding()
            throw error
        }

        let tCb1Start = CACurrentMediaTime()
        cb1.commit()
        cb1.waitUntilCompleted()
        let tCb1End = CACurrentMediaTime()
        let cb1DurationUs = (tCb1End - tCb1Start) * 1_000_000.0

        // Read router output on CPU:
        let expertIdsPtr = routerExpertsBuf.contents().bindMemory(to: Int32.self, capacity: batch * numTokens * 6)
        let weightsPtr = routerWeightsBuf.contents().bindMemory(to: Float.self, capacity: batch * numTokens * 6)

        var routerExpertsList = [Int32]()
        var routerWeightsList = [Float]()
        for i in 0..<(batch * numTokens * 6) {
            routerExpertsList.append(expertIdsPtr[i])
            routerWeightsList.append(weightsPtr[i])
        }

        // --- COMMAND BUFFER 2: Routed Experts per token ---
        var totalDiskReadDurationUs: Double = 0.0
        var totalCb2DurationUs: Double = 0.0
        var totalBytesRead = 0

        for t in 0..<numTokens {
            guard let cbToken = context.queue.makeCommandBuffer(),
                  let encToken = cbToken.makeComputeCommandEncoder() else {
                throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder for token \(t)")
            }

            do {
                // Load 3-bit slices for all 6 experts of token t
                let tDiskStart = CACurrentMediaTime()
                for k in 0..<6 {
                    let expertId = Int(expertIdsPtr[t * 6 + k])
                    guard expertId >= 0 && expertId < 256 else {
                        throw FlintMetalError.invalidParameter("Invalid expert ID \(expertId)")
                    }
                    let expBuf = expertBuffers[k]

                    // Gate
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.gate_proj.weight",
                        byteRange: (expertId * 393216)..<((expertId + 1) * 393216),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.gateWeight.contents(), count: 393216)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.gate_proj.scales",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.gateScales.contents(), count: 32768)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.gate_proj.biases",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.gateBiases.contents(), count: 32768)
                    )

                    // Up
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.up_proj.weight",
                        byteRange: (expertId * 393216)..<((expertId + 1) * 393216),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.upWeight.contents(), count: 393216)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.up_proj.scales",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.upScales.contents(), count: 32768)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.up_proj.biases",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.upBiases.contents(), count: 32768)
                    )

                    // Down
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.down_proj.weight",
                        byteRange: (expertId * 393216)..<((expertId + 1) * 393216),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.downWeight.contents(), count: 393216)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.down_proj.scales",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.downScales.contents(), count: 32768)
                    )
                    try shard1File.readTensorSlice(
                        "model.layers.2.mlp.switch_mlp.down_proj.biases",
                        byteRange: (expertId * 32768)..<((expertId + 1) * 32768),
                        into: UnsafeMutableRawBufferPointer(start: expBuf.downBiases.contents(), count: 32768)
                    )
                    totalBytesRead += 1_376_256

                    // Encode expert execution:
                    // 1. Gate dequant + GEMM (512 x 2048)
                    try dequantKernel.encodeDequantize(
                        into: encToken,
                        packedWeights: expBuf.gateWeight,
                        scales: expBuf.gateScales,
                        biases: expBuf.gateBiases,
                        output: mlpKernel.weightScratch,
                        rows: 512, cols: 2048,
                        bits: .three, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)
                    try mlpKernel.encodeGEMM(
                        into: encToken,
                        input: norm2Buf, inputOffset: t * 2048 * 2,
                        weight: mlpKernel.weightScratch,
                        output: mlpKernel.gateScratch,
                        M: 512, K: 2048, batch: 1, numTokens: 1,
                        inDtype: .bfloat16, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)

                    // 2. Up dequant + GEMM (512 x 2048)
                    try dequantKernel.encodeDequantize(
                        into: encToken,
                        packedWeights: expBuf.upWeight,
                        scales: expBuf.upScales,
                        biases: expBuf.upBiases,
                        output: mlpKernel.weightScratch,
                        rows: 512, cols: 2048,
                        bits: .three, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)
                    try mlpKernel.encodeGEMM(
                        into: encToken,
                        input: norm2Buf, inputOffset: t * 2048 * 2,
                        weight: mlpKernel.weightScratch,
                        output: mlpKernel.upScratch,
                        M: 512, K: 2048, batch: 1, numTokens: 1,
                        inDtype: .bfloat16, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)

                    // 3. SwiGLU: actScratch = silu(gateScratch) * upScratch
                    try mlpKernel.encodeSwiGLU(
                        into: encToken,
                        gate: mlpKernel.gateScratch,
                        up: mlpKernel.upScratch,
                        output: mlpKernel.actScratch,
                        totalElements: 512,
                        dtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)

                    // 4. Down dequant + GEMM (2048 x 512) -> routedContributionsBuf
                    try dequantKernel.encodeDequantize(
                        into: encToken,
                        packedWeights: expBuf.downWeight,
                        scales: expBuf.downScales,
                        biases: expBuf.downBiases,
                        output: mlpKernel.weightScratch,
                        rows: 2048, cols: 512,
                        bits: .three, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)
                    try mlpKernel.encodeGEMM(
                        into: encToken,
                        input: mlpKernel.actScratch,
                        weight: mlpKernel.weightScratch,
                        output: routedContributionsBuf, outputOffset: (t * 6 + k) * 2048 * 2,
                        M: 2048, K: 512, batch: 1, numTokens: 1,
                        inDtype: .bfloat16, outDtype: .bfloat16
                    )
                    encToken.memoryBarrier(scope: .buffers)
                }
                let tDiskEnd = CACurrentMediaTime()
                totalDiskReadDurationUs += (tDiskEnd - tDiskStart) * 1_000_000.0

                encToken.endEncoding()
            } catch {
                encToken.endEncoding()
                throw error
            }

            let tCb2Start = CACurrentMediaTime()
            cbToken.commit()
            cbToken.waitUntilCompleted()
            let tCb2End = CACurrentMediaTime()
            totalCb2DurationUs += (tCb2End - tCb2Start) * 1_000_000.0
        }

        // --- COMMAND BUFFER 3: Shared Expert + Reduction + Residual 2 ---
        guard let cbShared = context.queue.makeCommandBuffer(),
              let encShared = cbShared.makeComputeCommandEncoder() else {
            throw FlintMetalError.pipelineCreationFailed("Failed to make command encoder for Layer 2 shared/reduction")
        }

        do {
            // Stage 12: Shared Expert Forward (4-bit, 2048 -> 1024 -> 2048)
            try mlpKernel.encodeForwardSharedExpertMLP(
                into: encShared,
                weights: moeWeights,
                input: norm2Buf,
                output: sharedOutBuf,
                batch: batch,
                numTokens: numTokens,
                dtype: .bfloat16
            )
            encShared.memoryBarrier(scope: .buffers)

            // Stage 13: MoE Reduction: out = sum_{k=0..5}(weights[k] * routed[k]) + shared
            try reductionKernel.encodeReduction(
                into: encShared,
                contributions: routedContributionsBuf,
                weights: routerWeightsBuf,
                sharedExpert: sharedOutBuf,
                output: moeOutBuf,
                batch: batch,
                numTokens: numTokens,
                inDtype: .bfloat16,
                outDtype: .bfloat16
            )
            encShared.memoryBarrier(scope: .buffers)

            // Stage 14: Final Residual: finalOut = res1 + moeOut
            try encodeResidualAdd(
                into: encShared,
                inputA: res1Buf,
                inputB: moeOutBuf,
                output: finalOutBuf,
                totalElements: batch * numTokens * hiddenDim
            )

            encShared.endEncoding()
        } catch {
            encShared.endEncoding()
            throw error
        }

        let tCb3Start = CACurrentMediaTime()
        cbShared.commit()
        cbShared.waitUntilCompleted()
        let tCb3End = CACurrentMediaTime()
        let cb3DurationUs = (tCb3End - tCb3Start) * 1_000_000.0

        let tWallEnd = CACurrentMediaTime()
        let totalGpuElapsedUs = cb1DurationUs + totalCb2DurationUs + cb3DurationUs
        let totalWallClockUs = (tWallEnd - tWallStart) * 1_000_000.0

        let timing = Layer2TimingBreakdown(
            cb1AttentionNormRouterUs: cb1DurationUs,
            diskWeightReadUs: totalDiskReadDurationUs,
            cb2RoutedExpertsUs: totalCb2DurationUs,
            cb3SharedReductionRes2Us: cb3DurationUs,
            totalGpuElapsedUs: totalGpuElapsedUs,
            totalWallClockUs: totalWallClockUs,
            commandBuffersCommitted: 2 + numTokens,
            gpuWaits: 2 + numTokens,
            routedBytesRead: totalBytesRead
        )

        return Layer2Intermediates(
            input: input,
            norm1: norm1Buf,
            qRaw: qRawBuf,
            kRaw: kRawBuf,
            vRaw: vRawBuf,
            qNorm: qNormBuf,
            kNorm: kNormBuf,
            qRope: qRopeBuf,
            kRope: kRopeBuf,
            attnOut: attnOutBuf,
            attnFlat: attnFlatBuf,
            attnProj: attnProjBuf,
            res1: res1Buf,
            norm2: norm2Buf,
            routerExperts: routerExpertsList,
            routerWeights: routerWeightsList,
            routedContributions: routedContributionsBuf,
            sharedOut: sharedOutBuf,
            moeOut: moeOutBuf,
            finalOut: finalOutBuf,
            timing: timing
        )
    }

    /// Full Layer 2 MoE forward execution. Synchronously blocks until GPU completion.
    @discardableResult
    public func executeLayer2Forward(
        weights: FlintResidentWeights,
        shard1File: FlintSafetensorsFile,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        batch: Int = 1,
        numTokens: Int,
        cacheOffset: Int = 0
    ) throws -> Double {
        guard outputOffset >= 0, outputOffset % 2 == 0 else {
            throw FlintMetalError.invalidParameter("outputOffset (\(outputOffset)) must be non-negative and 2-byte aligned for bfloat16")
        }
        guard batch == 1 else {
            throw FlintMetalError.unsupportedConfiguration("Layer 2 MoE currently supports batch == 1 only; received batch = \(batch)")
        }
        let (reqTokens, of1) = numTokens.multipliedReportingOverflow(by: 2048)
        let (totalBytes, of2) = reqTokens.multipliedReportingOverflow(by: 2)
        guard !of1, !of2 else {
            throw FlintMetalError.invalidParameter("Output byte size calculation overflowed")
        }
        guard output.length >= outputOffset + totalBytes else {
            throw FlintMetalError.bufferTooSmall(name: "output", actual: output.length, required: outputOffset + totalBytes)
        }
        // Aliasing validation:
        if input.gpuAddress == output.gpuAddress {
            let inStart = inputOffset
            let inEnd = inputOffset + totalBytes
            let outStart = outputOffset
            let outEnd = outputOffset + totalBytes
            if max(inStart, outStart) < min(inEnd, outEnd) {
                throw FlintMetalError.invalidParameter("Input and output buffer ranges overlap in the same Metal buffer (in: [\(inStart)..<\(inEnd)], out: [\(outStart)..<\(outEnd)])")
            }
        }

        let start = CACurrentMediaTime()
        let inter = try executeLayer2ForwardWithIntermediates(
            weights: weights,
            shard1File: shard1File,
            input: input,
            inputOffset: inputOffset,
            batch: batch,
            numTokens: numTokens,
            cacheOffset: cacheOffset
        )

        let src = inter.finalOut.contents()
        let dst = output.contents().advanced(by: outputOffset)
        memcpy(dst, src, totalBytes)
        let elapsed = CACurrentMediaTime() - start
        return elapsed
    }

    // MARK: - Embedded Layer Shading Language Source

    public static let embeddedLayerShaderSource: String = """
#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Flint Decoder Layer Metal Compute Kernels (F3-01)
//
// Operations:
//   - Attention head transpositions: [batch, T, H, D] <-> [batch, H, T, D]
//   - Residual additions: y = a + b
// ==============================================================================

/// Transposes [batch, T, H, D] -> [batch, H, T, D] for bfloat16.
/// Grid dispatch: (D, T, batch * H)
kernel void flint_transpose_0213_bf16(
    device const bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint t = pos.y;
    uint bh = pos.z;
    uint b = bh / H;
    uint h = bh % H;

    if (d >= D || t >= T || h >= H || b >= batch) return;

    uint in_idx = (b * T + t) * (H * D) + h * D + d;
    uint out_idx = (b * H + h) * (T * D) + t * D + d;
    out[out_idx] = in[in_idx];
}

/// Transposes [batch, H, T, D] -> [batch, T, H, D] for bfloat16.
/// Grid dispatch: (D, H, batch * T)
kernel void flint_transpose_0213_back_bf16(
    device const bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint d = pos.x;
    uint h = pos.y;
    uint bt = pos.z;
    uint b = bt / T;
    uint t = bt % T;

    if (d >= D || h >= H || t >= T || b >= batch) return;

    uint in_idx = (b * H + h) * (T * D) + t * D + d;
    uint out_idx = (b * T + t) * (H * D) + h * D + d;
    out[out_idx] = in[in_idx];
}

/// Elementwise residual addition: y = a + b
kernel void flint_residual_add_bf16(
    device const bfloat* in_a [[buffer(0)]],
    device const bfloat* in_b [[buffer(1)]],
    device bfloat* out_y [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    if (pos >= total_elements) return;
    out_y[pos] = bfloat(float(in_a[pos]) + float(in_b[pos]));
}

"""
}
