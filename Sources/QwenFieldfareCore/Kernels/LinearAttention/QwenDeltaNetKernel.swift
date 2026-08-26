import Foundation
import Accelerate

public enum QwenLinearAttentionMath {
    @inline(__always)
    public static func silu(_ x: Float) -> Float {
        x / (1.0 + exp(-x))
    }

    @inline(__always)
    public static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    @inline(__always)
    public static func softplus(_ x: Float) -> Float {
        if x > 20.0 { return x }
        if x < -20.0 { return exp(x) }
        return log1p(exp(x))
    }

    public static func rmsNorm(
        input: [Float],
        weight: [Float],
        eps: Float = 1e-6
    ) -> [Float] {
        let d = input.count
        guard d > 0, weight.count == d else { return input }
        var sumSq: Float = 0.0
        vDSP_svesq(input, 1, &sumSq, vDSP_Length(d))
        let inv = 1.0 / sqrt(sumSq / Float(d) + eps)
        var out = [Float](repeating: 0, count: d)
        for i in 0..<d {
            out[i] = input[i] * inv * (1.0 + weight[i])
        }
        return out
    }

    public static func conv1dStep(
        inToken: [Float],
        state: inout [[Float]], // [dim, 4]
        weights: [[Float]]      // [dim, 4]
    ) -> [Float] {
        let dim = inToken.count
        var out = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            state[i][0] = state[i][1]
            state[i][1] = state[i][2]
            state[i][2] = state[i][3]
            state[i][3] = inToken[i]

            let acc = state[i][0] * weights[i][0]
                    + state[i][1] * weights[i][1]
                    + state[i][2] * weights[i][2]
                    + state[i][3] * weights[i][3]
            out[i] = silu(acc)
        }
        return out
    }

    public static func l2Norm(_ vec: [Float], eps: Float = 1e-6) -> [Float] {
        let d = vec.count
        var sumSq: Float = 0.0
        vDSP_svesq(vec, 1, &sumSq, vDSP_Length(d))
        let inv = 1.0 / sqrt(sumSq + eps)
        var out = [Float](repeating: 0, count: d)
        for i in 0..<d {
            out[i] = vec[i] * inv
        }
        return out
    }

    public static func deltaNetRecurrentStep(
        q: [[Float]],           // [32 heads, 128]
        k: [[Float]],           // [32 heads, 128]
        v: [[Float]],           // [32 heads, 128]
        a: [Float],             // [32 heads]
        b: [Float],             // [32 heads]
        aLog: [Float],          // [32 heads]
        dtBias: [Float],        // [32 heads]
        state: inout [[[Float]]] // [32 heads, 128, 128]
    ) -> [Float] {
        let numHeads = 32
        let headDim = 128
        var out = [Float](repeating: 0, count: numHeads * headDim)

        for h in 0..<numHeads {
            let betaH = sigmoid(b[h])
            let gH = exp(-exp(aLog[h]) * softplus(a[h] + dtBias[h]))
            let qH = q[h]
            let kH = k[h]
            let vH = v[h]

            // 1. Decay state: state = state * gH
            for r in 0..<headDim {
                for c in 0..<headDim {
                    state[h][r][c] *= gH
                }
            }

            // 2. kv_mem = sum(state * k)
            // 3. delta = (v - kv_mem) * betaH
            // 4. state += k (x) delta
            for r in 0..<headDim {
                var kvMemR: Float = 0.0
                for c in 0..<headDim {
                    kvMemR += state[h][r][c] * kH[c]
                }
                let deltaR = (vH[r] - kvMemR) * betaH
                for c in 0..<headDim {
                    state[h][r][c] += kH[c] * deltaR
                }
            }

            // 5. core_attn = sum(state * q)
            for r in 0..<headDim {
                var coreR: Float = 0.0
                for c in 0..<headDim {
                    coreR += state[h][r][c] * qH[c]
                }
                out[h * headDim + r] = coreR
            }
        }
        return out
    }

    public static func gatedRMSNorm(
        coreAttn: [Float],  // [32 * 128 = 4096]
        normWeight: [Float], // [128]
        z: [Float],          // [4096]
        eps: Float = 1e-6
    ) -> [Float] {
        let numHeads = 32
        let headDim = 128
        var out = [Float](repeating: 0, count: numHeads * headDim)

        for h in 0..<numHeads {
            let offset = h * headDim
            var sumSq: Float = 0.0
            for r in 0..<headDim {
                let val = coreAttn[offset + r]
                sumSq += val * val
            }
            let inv = 1.0 / sqrt(sumSq / Float(headDim) + eps)
            for r in 0..<headDim {
                let idx = offset + r
                out[idx] = coreAttn[idx] * inv * normWeight[r] * silu(z[idx])
            }
        }
        return out
    }

    public static func matVecMul(
        weight: [Float], // row-major [rows, cols]
        vec: [Float],    // [cols]
        rows: Int,
        cols: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        vDSP_mmul(
            weight, 1,
            vec, 1,
            &out, 1,
            vDSP_Length(rows), 1, vDSP_Length(cols)
        )
        return out
    }

    public struct StageOutputs: Sendable {
        public let normIn: [Float]
        public let conv1dOut: [Float]
        public let coreAttn: [Float]
        public let gatedNorm: [Float]
        public let outProj: [Float]
        public let attnResidual: [Float]
    }

    public static func linearAttentionLayer0Forward(
        inputHidden: [Float], // [2048]
        inputRMSNormWeight: [Float], // [2048]
        inProjQKVWeight: [Float], // [8192, 2048]
        inProjZWeight: [Float],   // [4096, 2048]
        inProjBWeight: [Float],   // [32, 2048]
        inProjAWeight: [Float],   // [32, 2048]
        conv1dWeight: [[Float]],  // [8192, 4]
        dtBias: [Float],          // [32]
        aLog: [Float],            // [32]
        normWeight: [Float],      // [128]
        outProjWeight: [Float],   // [2048, 4096]
        convState: inout [[Float]],  // [8192, 4]
        ssmState: inout [[[Float]]]  // [32, 128, 128]
    ) -> StageOutputs {
        // 1. RMSNorm
        let normIn = rmsNorm(input: inputHidden, weight: inputRMSNormWeight)

        // 2. Projections
        let mixedQKV = matVecMul(weight: inProjQKVWeight, vec: normIn, rows: 8192, cols: 2048)
        let z = matVecMul(weight: inProjZWeight, vec: normIn, rows: 4096, cols: 2048)
        let b = matVecMul(weight: inProjBWeight, vec: normIn, rows: 32, cols: 2048)
        let a = matVecMul(weight: inProjAWeight, vec: normIn, rows: 32, cols: 2048)

        // 3. Causal Conv1D
        let conv1dOut = conv1dStep(inToken: mixedQKV, state: &convState, weights: conv1dWeight)

        // 4. Split Q, K, V + Repeat Interleave x2 for Q and K
        var q = [[Float]](repeating: [Float](repeating: 0, count: 128), count: 32)
        var k = [[Float]](repeating: [Float](repeating: 0, count: 128), count: 32)
        var v = [[Float]](repeating: [Float](repeating: 0, count: 128), count: 32)

        let qScale: Float = 1.0 / sqrt(128.0)
        for h in 0..<32 {
            let srcHead = h / 2
            let qSlice = Array(conv1dOut[(srcHead * 128)..<(srcHead * 128 + 128)])
            let kSlice = Array(conv1dOut[(2048 + srcHead * 128)..<(2048 + srcHead * 128 + 128)])
            let vSlice = Array(conv1dOut[(4096 + h * 128)..<(4096 + h * 128 + 128)])

            let qNorm = l2Norm(qSlice)
            q[h] = qNorm.map { $0 * qScale }
            k[h] = l2Norm(kSlice)
            v[h] = vSlice
        }

        // 5. DeltaNet Recurrent Step
        let coreAttn = deltaNetRecurrentStep(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: &ssmState
        )

        // 6. Gated RMSNorm
        let gatedNorm = gatedRMSNorm(coreAttn: coreAttn, normWeight: normWeight, z: z)

        // 7. Output Projection
        let outProj = matVecMul(weight: outProjWeight, vec: gatedNorm, rows: 2048, cols: 4096)

        // 8. Residual
        var attnResidual = [Float](repeating: 0, count: 2048)
        for i in 0..<2048 {
            attnResidual[i] = inputHidden[i] + outProj[i]
        }

        return StageOutputs(
            normIn: normIn,
            conv1dOut: conv1dOut,
            coreAttn: coreAttn,
            gatedNorm: gatedNorm,
            outProj: outProj,
            attnResidual: attnResidual
        )
    }
}

public final class QwenLayerScratchBuffers: @unchecked Sendable {
    public let inBuf: MTLBuffer
    public let normInBuf: MTLBuffer
    public let attnOutBuf: MTLBuffer
    public let attnResBuf: MTLBuffer
    public let normMLPBuf: MTLBuffer
    public let routerLogitsBuf: MTLBuffer
    public let gateUpF32Buf: MTLBuffer
    public let actF32Buf: MTLBuffer
    public let expertOutBuf: MTLBuffer
    public let moeAccumBuf: MTLBuffer
    public let shGateBuf: MTLBuffer
    public let shUpBuf: MTLBuffer
    public let shActBuf: MTLBuffer
    public let shDownBuf: MTLBuffer
    public let shCoeffBuf: MTLBuffer
    public let sharedOutBuf: MTLBuffer
    public let completeOutBuf: MTLBuffer

    // Linear Attention
    public let qkvOutBuf: MTLBuffer
    public let zOutBuf: MTLBuffer
    public let bOutBuf: MTLBuffer
    public let aOutBuf: MTLBuffer
    public let convOutBuf: MTLBuffer
    public let qBuf: MTLBuffer
    public let kBuf: MTLBuffer
    public let vBuf: MTLBuffer
    public let coreAttnBuf: MTLBuffer
    public let gatedNormBuf: MTLBuffer

    // Full Attention
    public let qChunkBuf: MTLBuffer
    public let fullQBuf: MTLBuffer
    public let gateBuf: MTLBuffer
    public let fullKBuf: MTLBuffer
    public let fullVBuf: MTLBuffer
    public let attnHeadOutBuf: MTLBuffer
    public let cosBuf: MTLBuffer
    public let sinBuf: MTLBuffer

    public init(device: MTLDevice) throws {
        guard let inBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let normInBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let attnOutBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let attnResBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let normMLPBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let routerLogitsBuf = device.makeBuffer(length: 256 * 4, options: .storageModeShared),
              let gateUpF32Buf = device.makeBuffer(length: 1024 * 4, options: .storageModeShared),
              let actF32Buf = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let expertOutBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let moeAccumBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let shGateBuf = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shUpBuf = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shActBuf = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shDownBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let shCoeffBuf = device.makeBuffer(length: 4, options: .storageModeShared),
              let sharedOutBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let completeOutBuf = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let qkvOutBuf = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let zOutBuf = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let bOutBuf = device.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let aOutBuf = device.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let convOutBuf = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let qBuf = device.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let kBuf = device.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let vBuf = device.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let coreAttnBuf = device.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let gatedNormBuf = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let qChunkBuf = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let fullQBuf = device.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared),
              let gateBuf = device.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared),
              let fullKBuf = device.makeBuffer(length: 2 * 256 * 4, options: .storageModeShared),
              let fullVBuf = device.makeBuffer(length: 2 * 256 * 4, options: .storageModeShared),
              let attnHeadOutBuf = device.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared),
              let cosBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
              let sinBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)
        else {
            throw NSError(domain: "QwenLayerScratchBuffers", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate layer scratch buffers"])
        }

        self.inBuf = inBuf
        self.normInBuf = normInBuf
        self.attnOutBuf = attnOutBuf
        self.attnResBuf = attnResBuf
        self.normMLPBuf = normMLPBuf
        self.routerLogitsBuf = routerLogitsBuf
        self.gateUpF32Buf = gateUpF32Buf
        self.actF32Buf = actF32Buf
        self.expertOutBuf = expertOutBuf
        self.moeAccumBuf = moeAccumBuf
        self.shGateBuf = shGateBuf
        self.shUpBuf = shUpBuf
        self.shActBuf = shActBuf
        self.shDownBuf = shDownBuf
        self.shCoeffBuf = shCoeffBuf
        self.sharedOutBuf = sharedOutBuf
        self.completeOutBuf = completeOutBuf
        self.qkvOutBuf = qkvOutBuf
        self.zOutBuf = zOutBuf
        self.bOutBuf = bOutBuf
        self.aOutBuf = aOutBuf
        self.convOutBuf = convOutBuf
        self.qBuf = qBuf
        self.kBuf = kBuf
        self.vBuf = vBuf
        self.coreAttnBuf = coreAttnBuf
        self.gatedNormBuf = gatedNormBuf
        self.qChunkBuf = qChunkBuf
        self.fullQBuf = fullQBuf
        self.gateBuf = gateBuf
        self.fullKBuf = fullKBuf
        self.fullVBuf = fullVBuf
        self.attnHeadOutBuf = attnHeadOutBuf
        self.cosBuf = cosBuf
        self.sinBuf = sinBuf
    }
}

public final class QwenMetalDeltaNetRunner: @unchecked Sendable {
    public let context: MetalContext
    public let scratch: QwenLayerScratchBuffers

    private let rmsnormPipe: MTLComputePipelineState
    private let gemvPipe: MTLComputePipelineState
    private let conv1dPipe: MTLComputePipelineState
    private let prepQKVPipe: MTLComputePipelineState
    private let deltanetPipe: MTLComputePipelineState
    private let gatedNormPipe: MTLComputePipelineState
    private let addPipe: MTLComputePipelineState
    private let fusedResidualNormPipe: MTLComputePipelineState
    private let gemvBF16Pipe: MTLComputePipelineState
    private let gateUpSiluPipe: MTLComputePipelineState
    private let siluMulPipe: MTLComputePipelineState
    private let expertAccumPipe: MTLComputePipelineState
    private let sharedGatePipe: MTLComputePipelineState
    private let finalResidualPipe: MTLComputePipelineState
    private let splitQGatePipe: MTLComputePipelineState
    private let headNormPipe: MTLComputePipelineState
    private let ropePipe: MTLComputePipelineState
    private let causalAttnPipe: MTLComputePipelineState
    private let fusedExpertDownAccumPipe: MTLComputePipelineState
    private let fusedSharedGateUpSiluPipe: MTLComputePipelineState
    private let fusedSharedDownFinalResidualPipe: MTLComputePipelineState
    private let fusedGemvABPipe: MTLComputePipelineState
    private let fusedDeltaNetGatedNormPipe: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.scratch = try QwenLayerScratchBuffers(device: context.device)
        self.rmsnormPipe = try context.pipeline("qwen_rmsnorm_f32")
        self.gemvPipe = try context.pipeline("qwen_gemv_f32")
        self.conv1dPipe = try context.pipeline("qwen_conv1d_f32")
        self.prepQKVPipe = try context.pipeline("qwen_prep_qkv_f32")
        self.deltanetPipe = try context.pipeline("qwen_deltanet_recurrent_f32")
        self.gatedNormPipe = try context.pipeline("qwen_gated_rmsnorm_f32")
        self.addPipe = try context.pipeline("qwen_residual_add_f32")
        self.fusedResidualNormPipe = try context.pipeline("qwen_fused_residual_rmsnorm_f32")
        self.gemvBF16Pipe = try context.pipeline("qwen_gemv_bf16_to_f32")
        self.gateUpSiluPipe = try context.pipeline("qwen_gate_up_silu_f32")
        self.siluMulPipe = try context.pipeline("qwen_silu_mul_f32")
        self.expertAccumPipe = try context.pipeline("qwen_expert_accum_f32")
        self.sharedGatePipe = try context.pipeline("qwen_shared_expert_gate_f32")
        self.finalResidualPipe = try context.pipeline("qwen_layer_final_residual_f32")
        self.splitQGatePipe = try context.pipeline("qwen_split_q_gate_f32")
        self.headNormPipe = try context.pipeline("qwen_head_rmsnorm_f32")
        self.ropePipe = try context.pipeline("qwen_rope_partial_f32")
        self.causalAttnPipe = try context.pipeline("qwen_causal_attn_step_f32")
        self.fusedExpertDownAccumPipe = try context.pipeline("qwen_fused_expert_down_accum_bf16_to_f32")
        self.fusedSharedGateUpSiluPipe = try context.pipeline("qwen_fused_shared_gate_up_silu_f32")
        self.fusedSharedDownFinalResidualPipe = try context.pipeline("qwen_fused_shared_down_final_residual_f32")
        self.fusedGemvABPipe = try context.pipeline("qwen_fused_gemv_ab_f32")
        self.fusedDeltaNetGatedNormPipe = try context.pipeline("qwen_fused_deltanet_gated_rmsnorm_f32")
    }

    public func forwardToken(
        inputHidden: [Float],         // [2048]
        inputRMSNormWeight: [Float],  // [2048]
        inProjQKVWeight: [Float],     // [8192, 2048]
        inProjZWeight: [Float],       // [4096, 2048]
        inProjBWeight: [Float],       // [32, 2048]
        inProjAWeight: [Float],       // [32, 2048]
        conv1dFlatWeight: [Float],    // [8192 * 4]
        dtBias: [Float],              // [32]
        aLog: [Float],                // [32]
        normWeight: [Float],          // [128]
        outProjWeight: [Float],       // [2048, 4096]
        convStateBuf: MTLBuffer,      // [8192 * 4]
        ssmStateBuf: MTLBuffer        // [32 * 128 * 128]
    ) throws -> QwenLinearAttentionMath.StageOutputs {
        let dev = context.device

        // 1. Allocate input & weight buffers
        guard let inBuf = dev.makeBuffer(bytes: inputHidden, length: inputHidden.count * 4, options: .storageModeShared),
              let rmsWeightBuf = dev.makeBuffer(bytes: inputRMSNormWeight, length: inputRMSNormWeight.count * 4, options: .storageModeShared),
              let normInBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let qkvWeightBuf = dev.makeBuffer(bytes: inProjQKVWeight, length: inProjQKVWeight.count * 4, options: .storageModeShared),
              let zWeightBuf = dev.makeBuffer(bytes: inProjZWeight, length: inProjZWeight.count * 4, options: .storageModeShared),
              let bWeightBuf = dev.makeBuffer(bytes: inProjBWeight, length: inProjBWeight.count * 4, options: .storageModeShared),
              let aWeightBuf = dev.makeBuffer(bytes: inProjAWeight, length: inProjAWeight.count * 4, options: .storageModeShared),
              let qkvOutBuf = dev.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let zOutBuf = dev.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let bOutBuf = dev.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let aOutBuf = dev.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let convWeightBuf = dev.makeBuffer(bytes: conv1dFlatWeight, length: conv1dFlatWeight.count * 4, options: .storageModeShared),
              let convOutBuf = dev.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let qBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let kBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let vBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let aLogBuf = dev.makeBuffer(bytes: aLog, length: 32 * 4, options: .storageModeShared),
              let dtBiasBuf = dev.makeBuffer(bytes: dtBias, length: 32 * 4, options: .storageModeShared),
              let coreAttnBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
              let normWeightBuf = dev.makeBuffer(bytes: normWeight, length: 128 * 4, options: .storageModeShared),
              let gatedNormBuf = dev.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let outProjWeightBuf = dev.makeBuffer(bytes: outProjWeight, length: outProjWeight.count * 4, options: .storageModeShared),
              let outProjBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let attnResidualBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared)
        else {
            throw NSError(domain: "QwenMetalDeltaNetRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate Metal buffers"])
        }

        guard let cmdBuf = context.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "QwenMetalDeltaNetRunner", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command encoder"])
        }

        // 1. RMSNorm
        var dimD: UInt32 = 2048
        var eps: Float = 1e-6
        enc.setComputePipelineState(rmsnormPipe)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(rmsWeightBuf, offset: 0, index: 1)
        enc.setBuffer(normInBuf, offset: 0, index: 2)
        enc.setBytes(&dimD, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        // 2. Projections
        func encodeGEMV(weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
            var r = rows
            var c = cols
            enc.setComputePipelineState(gemvPipe)
            enc.setBuffer(weight, offset: 0, index: 0)
            enc.setBuffer(vec, offset: 0, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&r, length: 4, index: 3)
            enc.setBytes(&c, length: 4, index: 4)
            let tgSize = min(Int(rows), 64)
            let numTg = (Int(rows) + tgSize - 1) / tgSize
            enc.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        encodeGEMV(weight: qkvWeightBuf, vec: normInBuf, out: qkvOutBuf, rows: 8192, cols: 2048)
        encodeGEMV(weight: zWeightBuf, vec: normInBuf, out: zOutBuf, rows: 4096, cols: 2048)
        encodeGEMV(weight: bWeightBuf, vec: normInBuf, out: bOutBuf, rows: 32, cols: 2048)
        encodeGEMV(weight: aWeightBuf, vec: normInBuf, out: aOutBuf, rows: 32, cols: 2048)

        // 3. Conv1D
        var convDim: UInt32 = 8192
        enc.setComputePipelineState(conv1dPipe)
        enc.setBuffer(qkvOutBuf, offset: 0, index: 0)
        enc.setBuffer(convStateBuf, offset: 0, index: 1)
        enc.setBuffer(convWeightBuf, offset: 0, index: 2)
        enc.setBuffer(convOutBuf, offset: 0, index: 3)
        enc.setBytes(&convDim, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        // 4. Prep QKV + L2Norm
        enc.setComputePipelineState(prepQKVPipe)
        enc.setBuffer(convOutBuf, offset: 0, index: 0)
        enc.setBuffer(qBuf, offset: 0, index: 1)
        enc.setBuffer(kBuf, offset: 0, index: 2)
        enc.setBuffer(vBuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        // 5. DeltaNet Recurrence
        enc.setComputePipelineState(deltanetPipe)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(aOutBuf, offset: 0, index: 3)
        enc.setBuffer(bOutBuf, offset: 0, index: 4)
        enc.setBuffer(aLogBuf, offset: 0, index: 5)
        enc.setBuffer(dtBiasBuf, offset: 0, index: 6)
        enc.setBuffer(ssmStateBuf, offset: 0, index: 7)
        enc.setBuffer(coreAttnBuf, offset: 0, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: 128 / 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // 6. Gated RMSNorm
        enc.setComputePipelineState(gatedNormPipe)
        enc.setBuffer(coreAttnBuf, offset: 0, index: 0)
        enc.setBuffer(normWeightBuf, offset: 0, index: 1)
        enc.setBuffer(zOutBuf, offset: 0, index: 2)
        enc.setBuffer(gatedNormBuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        // 7. Output Projection
        encodeGEMV(weight: outProjWeightBuf, vec: gatedNormBuf, out: outProjBuf, rows: 2048, cols: 4096)

        // 8. Residual Add
        var resCount: UInt32 = 2048
        enc.setComputePipelineState(addPipe)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outProjBuf, offset: 0, index: 1)
        enc.setBuffer(attnResidualBuf, offset: 0, index: 2)
        enc.setBytes(&resCount, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        return QwenLinearAttentionMath.StageOutputs(
            normIn: readBuffer(normInBuf, count: 2048),
            conv1dOut: readBuffer(convOutBuf, count: 8192),
            coreAttn: readBuffer(coreAttnBuf, count: 4096),
            gatedNorm: readBuffer(gatedNormBuf, count: 4096),
            outProj: readBuffer(outProjBuf, count: 2048),
            attnResidual: readBuffer(attnResidualBuf, count: 2048)
        )
    }

    public struct CompleteLayerOutputs: Sendable {
        public let attnResidual: [Float]
        public let normMLP: [Float]
        public let routerLogits: [Float]
        public let top8Indices: [Int]
        public let top8Scores: [Float]
        public let routedMoEOut: [Float]
        public let sharedExpertOut: [Float]
        public let completeLayerOut: [Float]
    }

    public func forwardCompleteLayer0Token(
        inputHidden: [Float],                // [2048]
        inputRMSNormWeight: [Float],         // [2048]
        inProjQKVWeight: [Float],            // [8192, 2048]
        inProjZWeight: [Float],              // [4096, 2048]
        inProjBWeight: [Float],              // [32, 2048]
        inProjAWeight: [Float],              // [32, 2048]
        conv1dFlatWeight: [Float],           // [8192 * 4]
        dtBias: [Float],                     // [32]
        aLog: [Float],                       // [32]
        normWeight: [Float],                 // [128]
        outProjWeight: [Float],              // [2048, 4096]
        postAttentionNormWeight: [Float],    // [2048]
        routerGateWeight: [Float],           // [256, 2048]
        sharedGateProjWeight: [Float],       // [512, 2048]
        sharedUpProjWeight: [Float],         // [512, 2048]
        sharedDownProjWeight: [Float],       // [2048, 512]
        sharedGateCoeffWeight: [Float],      // [1, 2048]
        expertsMmapPointer: UnsafeRawPointer, // Safetensors mapping base
        convStateBuf: MTLBuffer,             // [8192 * 4]
        ssmStateBuf: MTLBuffer               // [32 * 128 * 128]
    ) throws -> CompleteLayerOutputs {
        let dev = context.device

        // 1. Run Linear Attention to get attention residual
        let attnOutputs = try forwardToken(
            inputHidden: inputHidden,
            inputRMSNormWeight: inputRMSNormWeight,
            inProjQKVWeight: inProjQKVWeight,
            inProjZWeight: inProjZWeight,
            inProjBWeight: inProjBWeight,
            inProjAWeight: inProjAWeight,
            conv1dFlatWeight: conv1dFlatWeight,
            dtBias: dtBias,
            aLog: aLog,
            normWeight: normWeight,
            outProjWeight: outProjWeight,
            convStateBuf: convStateBuf,
            ssmStateBuf: ssmStateBuf
        )

        // 2. Allocate buffers for MLP stage
        guard let attnResBuf = dev.makeBuffer(bytes: attnOutputs.attnResidual, length: 2048 * 4, options: .storageModeShared),
              let postNormWeightBuf = dev.makeBuffer(bytes: postAttentionNormWeight, length: 2048 * 4, options: .storageModeShared),
              let normMLPBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let routerWeightBuf = dev.makeBuffer(bytes: routerGateWeight, length: 256 * 2048 * 4, options: .storageModeShared),
              let routerLogitsBuf = dev.makeBuffer(length: 256 * 4, options: .storageModeShared),
              let shGateWeightBuf = dev.makeBuffer(bytes: sharedGateProjWeight, length: 512 * 2048 * 4, options: .storageModeShared),
              let shUpWeightBuf = dev.makeBuffer(bytes: sharedUpProjWeight, length: 512 * 2048 * 4, options: .storageModeShared),
              let shDownWeightBuf = dev.makeBuffer(bytes: sharedDownProjWeight, length: 2048 * 512 * 4, options: .storageModeShared),
              let shCoeffWeightBuf = dev.makeBuffer(bytes: sharedGateCoeffWeight, length: 2048 * 4, options: .storageModeShared),
              let shGateBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shUpBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shActBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let shDownBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let shCoeffBuf = dev.makeBuffer(length: 4, options: .storageModeShared),
              let sharedOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let moeAccumBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let gateUpF32Buf = dev.makeBuffer(length: 1024 * 4, options: .storageModeShared),
              let actF32Buf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let expertOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let completeOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared)
        else {
            throw NSError(domain: "QwenMetalDeltaNetRunner", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate MLP Metal buffers"])
        }

        memset(moeAccumBuf.contents(), 0, 2048 * 4)

        let gemvBF16Pipe = try context.pipeline("qwen_gemv_bf16_to_f32")
        let gateUpSiluPipe = try context.pipeline("qwen_gate_up_silu_f32")
        let expertAccumPipe = try context.pipeline("qwen_expert_accum_f32")
        let sharedGatePipe = try context.pipeline("qwen_shared_expert_gate_f32")
        let finalResidualPipe = try context.pipeline("qwen_layer_final_residual_f32")

        // 3. Encode Post-Attention RMSNorm and Router GEMV
        guard let cmdBuf1 = context.queue.makeCommandBuffer(),
              let enc1 = cmdBuf1.makeComputeCommandEncoder() else {
            throw NSError(domain: "QwenMetalDeltaNetRunner", code: 4, userInfo: [NSLocalizedDescriptionKey: "Encoder failure"])
        }

        var dimD: UInt32 = 2048
        var eps: Float = 1e-6
        enc1.setComputePipelineState(rmsnormPipe)
        enc1.setBuffer(attnResBuf, offset: 0, index: 0)
        enc1.setBuffer(postNormWeightBuf, offset: 0, index: 1)
        enc1.setBuffer(normMLPBuf, offset: 0, index: 2)
        enc1.setBytes(&dimD, length: 4, index: 3)
        enc1.setBytes(&eps, length: 4, index: 4)
        enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        var rRouter: UInt32 = 256
        var cRouter: UInt32 = 2048
        enc1.setComputePipelineState(gemvPipe)
        enc1.setBuffer(routerWeightBuf, offset: 0, index: 0)
        enc1.setBuffer(normMLPBuf, offset: 0, index: 1)
        enc1.setBuffer(routerLogitsBuf, offset: 0, index: 2)
        enc1.setBytes(&rRouter, length: 4, index: 3)
        enc1.setBytes(&cRouter, length: 4, index: 4)
        enc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // 4. Extract router logits and compute top-8 routing
        let logitsPtr = routerLogitsBuf.contents().bindMemory(to: Float.self, capacity: 256)
        let routerLogits = Array(UnsafeBufferPointer(start: logitsPtr, count: 256))

        var maxLogit = routerLogits[0]
        for v in routerLogits { if v > maxLogit { maxLogit = v } }
        var expLogits = [Float](repeating: 0, count: 256)
        var sumExp: Float = 0.0
        for i in 0..<256 {
            let e = exp(routerLogits[i] - maxLogit)
            expLogits[i] = e
            sumExp += e
        }
        var probs = [Float](repeating: 0, count: 256)
        for i in 0..<256 { probs[i] = expLogits[i] / sumExp }

        var indexedProbs: [(index: Int, prob: Float)] = (0..<256).map { ($0, probs[$0]) }
        indexedProbs.sort { $0.prob > $1.prob }

        let top8Indices = (0..<8).map { indexedProbs[$0].index }
        let top8RawScores = (0..<8).map { indexedProbs[$0].prob }
        let sumTop8 = top8RawScores.reduce(0.0, +)
        let top8Scores = top8RawScores.map { $0 / sumTop8 }

        // 5. Encode Routed MoE and Shared Expert
        guard let cmdBuf2 = context.queue.makeCommandBuffer(),
              let enc2 = cmdBuf2.makeComputeCommandEncoder() else {
            throw NSError(domain: "QwenMetalDeltaNetRunner", code: 5, userInfo: [NSLocalizedDescriptionKey: "Encoder failure 2"])
        }

        func encodeGEMV2(weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
            var r = rows
            var c = cols
            enc2.setComputePipelineState(gemvPipe)
            enc2.setBuffer(weight, offset: 0, index: 0)
            enc2.setBuffer(vec, offset: 0, index: 1)
            enc2.setBuffer(out, offset: 0, index: 2)
            enc2.setBytes(&r, length: 4, index: 3)
            enc2.setBytes(&c, length: 4, index: 4)
            let tgSize = min(Int(rows), 64)
            let numTg = (Int(rows) + tgSize - 1) / tgSize
            enc2.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        func encodeGEMVBF16(weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
            var r = rows
            var c = cols
            enc2.setComputePipelineState(gemvBF16Pipe)
            enc2.setBuffer(weight, offset: 0, index: 0)
            enc2.setBuffer(vec, offset: 0, index: 1)
            enc2.setBuffer(out, offset: 0, index: 2)
            enc2.setBytes(&r, length: 4, index: 3)
            enc2.setBytes(&c, length: 4, index: 4)
            let tgSize = min(Int(rows), 64)
            let numTg = (Int(rows) + tgSize - 1) / tgSize
            enc2.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Routed Experts
        for (i, expertIdx) in top8Indices.enumerated() {
            var score = top8Scores[i]
            let recOffset = expertIdx * 6_291_456
            guard let guBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset)), length: 1024 * 2048 * 2, options: .storageModeShared, deallocator: nil),
                  let dnBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset + 4_194_304)), length: 2048 * 512 * 2, options: .storageModeShared, deallocator: nil) else {
                continue
            }

            // Gate-Up
            encodeGEMVBF16(weight: guBuf, vec: normMLPBuf, out: gateUpF32Buf, rows: 1024, cols: 2048)

            // SiLU(gate) * up
            enc2.setComputePipelineState(gateUpSiluPipe)
            enc2.setBuffer(gateUpF32Buf, offset: 0, index: 0)
            enc2.setBuffer(actF32Buf, offset: 0, index: 1)
            enc2.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // Down Proj
            encodeGEMVBF16(weight: dnBuf, vec: actF32Buf, out: expertOutBuf, rows: 2048, cols: 512)

            // Accumulate weighted output
            var d2048: UInt32 = 2048
            enc2.setComputePipelineState(expertAccumPipe)
            enc2.setBuffer(expertOutBuf, offset: 0, index: 0)
            enc2.setBuffer(moeAccumBuf, offset: 0, index: 1)
            enc2.setBytes(&score, length: 4, index: 2)
            enc2.setBytes(&d2048, length: 4, index: 3)
            enc2.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }

        // Shared Expert
        encodeGEMV2(weight: shGateWeightBuf, vec: normMLPBuf, out: shGateBuf, rows: 512, cols: 2048)
        encodeGEMV2(weight: shUpWeightBuf, vec: normMLPBuf, out: shUpBuf, rows: 512, cols: 2048)

        let siluMulPipe = try context.pipeline("qwen_silu_mul_f32")
        var d512: UInt32 = 512
        enc2.setComputePipelineState(siluMulPipe)
        enc2.setBuffer(shGateBuf, offset: 0, index: 0)
        enc2.setBuffer(shUpBuf, offset: 0, index: 1)
        enc2.setBuffer(shActBuf, offset: 0, index: 2)
        enc2.setBytes(&d512, length: 4, index: 3)
        enc2.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        encodeGEMV2(weight: shDownWeightBuf, vec: shActBuf, out: shDownBuf, rows: 2048, cols: 512)
        encodeGEMV2(weight: shCoeffWeightBuf, vec: normMLPBuf, out: shCoeffBuf, rows: 1, cols: 2048)

        var d2048: UInt32 = 2048
        enc2.setComputePipelineState(sharedGatePipe)
        enc2.setBuffer(shDownBuf, offset: 0, index: 0)
        enc2.setBuffer(shCoeffBuf, offset: 0, index: 1)
        enc2.setBuffer(sharedOutBuf, offset: 0, index: 2)
        enc2.setBytes(&d2048, length: 4, index: 3)
        enc2.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        // Final Residual: completeOut = attnResidual + moeOut + sharedOut
        enc2.setComputePipelineState(finalResidualPipe)
        enc2.setBuffer(attnResBuf, offset: 0, index: 0)
        enc2.setBuffer(moeAccumBuf, offset: 0, index: 1)
        enc2.setBuffer(sharedOutBuf, offset: 0, index: 2)
        enc2.setBuffer(completeOutBuf, offset: 0, index: 3)
        enc2.setBytes(&d2048, length: 4, index: 4)
        enc2.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        func readBuf(_ buf: MTLBuffer, count: Int) -> [Float] {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        return CompleteLayerOutputs(
            attnResidual: attnOutputs.attnResidual,
            normMLP: readBuf(normMLPBuf, count: 2048),
            routerLogits: routerLogits,
            top8Indices: top8Indices,
            top8Scores: top8Scores,
            routedMoEOut: readBuf(moeAccumBuf, count: 2048),
            sharedExpertOut: readBuf(sharedOutBuf, count: 2048),
            completeLayerOut: readBuf(completeOutBuf, count: 2048)
        )
    }

    public func forwardLayerToken(
        weights: QwenLayerWeights,
        state: QwenLayerState,
        inputHidden: [Float],
        tokenPosition: Int,
        maxSeqLen: Int = 64,
        expertsMmapPointer: UnsafeRawPointer
    ) throws -> (attnOut: [Float], complete: CompleteLayerOutputs) {
        let metalWeights = try QwenLayerMetalWeights(weights: weights, device: context.device)
        return try forwardLayerToken(
            weights: metalWeights,
            state: state,
            inputHidden: inputHidden,
            tokenPosition: tokenPosition,
            maxSeqLen: maxSeqLen,
            expertsMmapPointer: expertsMmapPointer
        )
    }

    public func forwardLayerToken(
        weights: QwenLayerMetalWeights,
        state: QwenLayerState,
        inputHidden: [Float],
        tokenPosition: Int,
        maxSeqLen: Int = 64,
        expertsMmapPointer: UnsafeRawPointer
    ) throws -> (attnOut: [Float], complete: CompleteLayerOutputs) {
        try autoreleasepool {
            let dev = context.device
            let sc = scratch

            _ = inputHidden.withUnsafeBytes { ptr in
                memcpy(sc.inBuf.contents(), ptr.baseAddress!, 2048 * 4)
            }

            func encodeGEMV(encoder: MTLComputeCommandEncoder, weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
                var r = rows
                var c = cols
                encoder.setComputePipelineState(gemvPipe)
                encoder.setBuffer(weight, offset: 0, index: 0)
                encoder.setBuffer(vec, offset: 0, index: 1)
                encoder.setBuffer(out, offset: 0, index: 2)
                encoder.setBytes(&r, length: 4, index: 3)
                encoder.setBytes(&c, length: 4, index: 4)
                let tgSize = min(Int(rows), 64)
                let numTg = (Int(rows) + tgSize - 1) / tgSize
                encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            func encodeGEMVBF16(encoder: MTLComputeCommandEncoder, weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
                var r = rows
                var c = cols
                encoder.setComputePipelineState(gemvBF16Pipe)
                encoder.setBuffer(weight, offset: 0, index: 0)
                encoder.setBuffer(vec, offset: 0, index: 1)
                encoder.setBuffer(out, offset: 0, index: 2)
                encoder.setBytes(&r, length: 4, index: 3)
                encoder.setBytes(&c, length: 4, index: 4)
                let tgSize = min(Int(rows), 64)
                let numTg = (Int(rows) + tgSize - 1) / tgSize
                encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            var dim2048: UInt32 = 2048
            var eps: Float = 1e-6

            guard let cmdBuf1 = context.queue.makeCommandBuffer(),
                  let enc1 = cmdBuf1.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 11, userInfo: [NSLocalizedDescriptionKey: "Encoder failure 1"])
            }

            // 1. Input RMSNorm
            enc1.setComputePipelineState(rmsnormPipe)
            enc1.setBuffer(sc.inBuf, offset: 0, index: 0)
            enc1.setBuffer(weights.inputRMSNormWeightBuf, offset: 0, index: 1)
            enc1.setBuffer(sc.normInBuf, offset: 0, index: 2)
            enc1.setBytes(&dim2048, length: 4, index: 3)
            enc1.setBytes(&eps, length: 4, index: 4)
            enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // 2. Attention
            if weights.kind == .linearAttention {
                guard let qkvWBuf = weights.inProjQKVWeightBuf,
                      let zWBuf = weights.inProjZWeightBuf,
                      let bWBuf = weights.inProjBWeightBuf,
                      let aWBuf = weights.inProjAWeightBuf,
                      let convWBuf = weights.conv1dFlatWeightBuf,
                      let dtBiasBuf = weights.dtBiasBuf,
                      let aLogBuf = weights.aLogBuf,
                      let normWeightBuf = weights.linearNormWeightBuf,
                      let linOutProjWBuf = weights.linearOutProjWeightBuf,
                      let convStBuf = state.convStateBuf,
                      let ssmStBuf = state.ssmStateBuf
                else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 12, userInfo: [NSLocalizedDescriptionKey: "Linear weights missing"])
                }

                encodeGEMV(encoder: enc1, weight: qkvWBuf, vec: sc.normInBuf, out: sc.qkvOutBuf, rows: 8192, cols: 2048)
                encodeGEMV(encoder: enc1, weight: zWBuf, vec: sc.normInBuf, out: sc.zOutBuf, rows: 4096, cols: 2048)

                var r32: UInt32 = 32
                var c2048: UInt32 = 2048
                enc1.setComputePipelineState(fusedGemvABPipe)
                enc1.setBuffer(bWBuf, offset: 0, index: 0)
                enc1.setBuffer(aWBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.normInBuf, offset: 0, index: 2)
                enc1.setBuffer(sc.bOutBuf, offset: 0, index: 3)
                enc1.setBuffer(sc.aOutBuf, offset: 0, index: 4)
                enc1.setBytes(&r32, length: 4, index: 5)
                enc1.setBytes(&c2048, length: 4, index: 6)
                enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                var convDim: UInt32 = 8192
                enc1.setComputePipelineState(conv1dPipe)
                enc1.setBuffer(sc.qkvOutBuf, offset: 0, index: 0)
                enc1.setBuffer(convStBuf, offset: 0, index: 1)
                enc1.setBuffer(convWBuf, offset: 0, index: 2)
                enc1.setBuffer(sc.convOutBuf, offset: 0, index: 3)
                enc1.setBytes(&convDim, length: 4, index: 4)
                enc1.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                enc1.setComputePipelineState(prepQKVPipe)
                enc1.setBuffer(sc.convOutBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.qBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.kBuf, offset: 0, index: 2)
                enc1.setBuffer(sc.vBuf, offset: 0, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                // Fused DeltaNet Recurrence + Gated RMSNorm
                enc1.setComputePipelineState(fusedDeltaNetGatedNormPipe)
                enc1.setBuffer(sc.qBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.kBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.vBuf, offset: 0, index: 2)
                enc1.setBuffer(sc.aOutBuf, offset: 0, index: 3)
                enc1.setBuffer(sc.bOutBuf, offset: 0, index: 4)
                enc1.setBuffer(aLogBuf, offset: 0, index: 5)
                enc1.setBuffer(dtBiasBuf, offset: 0, index: 6)
                enc1.setBuffer(ssmStBuf, offset: 0, index: 7)
                enc1.setBuffer(normWeightBuf, offset: 0, index: 8)
                enc1.setBuffer(sc.zOutBuf, offset: 0, index: 9)
                enc1.setBuffer(sc.gatedNormBuf, offset: 0, index: 10)
                enc1.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

                encodeGEMV(encoder: enc1, weight: linOutProjWBuf, vec: sc.gatedNormBuf, out: sc.attnOutBuf, rows: 2048, cols: 4096)
            } else {
                guard let qPWBuf = weights.qProjWeightBuf,
                      let kPWBuf = weights.kProjWeightBuf,
                      let vPWBuf = weights.vProjWeightBuf,
                      let qNWBuf = weights.qNormWeightBuf,
                      let kNWBuf = weights.kNormWeightBuf,
                      let oPWBuf = weights.oProjWeightBuf,
                      let kCBuf = state.kCacheBuf,
                      let vCBuf = state.vCacheBuf
                else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 13, userInfo: [NSLocalizedDescriptionKey: "Full attention weights missing"])
                }

                encodeGEMV(encoder: enc1, weight: qPWBuf, vec: sc.normInBuf, out: sc.qChunkBuf, rows: 8192, cols: 2048)
                encodeGEMV(encoder: enc1, weight: kPWBuf, vec: sc.normInBuf, out: sc.fullKBuf, rows: 512, cols: 2048)
                encodeGEMV(encoder: enc1, weight: vPWBuf, vec: sc.normInBuf, out: sc.fullVBuf, rows: 512, cols: 2048)

                enc1.setComputePipelineState(splitQGatePipe)
                enc1.setBuffer(sc.qChunkBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.fullQBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.gateBuf, offset: 0, index: 2)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var numQHeads: UInt32 = 16
                var numKHeads: UInt32 = 2
                var headDim: UInt32 = 256
                enc1.setComputePipelineState(headNormPipe)
                enc1.setBuffer(sc.fullQBuf, offset: 0, index: 0)
                enc1.setBuffer(qNWBuf, offset: 0, index: 1)
                enc1.setBytes(&numQHeads, length: 4, index: 2)
                enc1.setBytes(&headDim, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc1.setComputePipelineState(headNormPipe)
                enc1.setBuffer(sc.fullKBuf, offset: 0, index: 0)
                enc1.setBuffer(kNWBuf, offset: 0, index: 1)
                enc1.setBytes(&numKHeads, length: 4, index: 2)
                enc1.setBytes(&headDim, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                let theta: Float = 10000000.0
                let cosPtr = sc.cosBuf.contents().bindMemory(to: Float.self, capacity: 64)
                let sinPtr = sc.sinBuf.contents().bindMemory(to: Float.self, capacity: 64)
                for i in 0..<32 {
                    let freq = 1.0 / pow(theta, Float(2 * i) / 64.0)
                    let angle = Float(tokenPosition) * freq
                    let c = cos(angle)
                    let s = sin(angle)
                    cosPtr[i] = c; cosPtr[32 + i] = c
                    sinPtr[i] = s; sinPtr[32 + i] = s
                }

                enc1.setComputePipelineState(ropePipe)
                enc1.setBuffer(sc.fullQBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.cosBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.sinBuf, offset: 0, index: 2)
                enc1.setBytes(&numQHeads, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc1.setComputePipelineState(ropePipe)
                enc1.setBuffer(sc.fullKBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.cosBuf, offset: 0, index: 1)
                enc1.setBuffer(sc.sinBuf, offset: 0, index: 2)
                enc1.setBytes(&numKHeads, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc1.endEncoding()
                cmdBuf1.commit()
                cmdBuf1.waitUntilCompleted()

                for kh in 0..<2 {
                    let kCacheOffset = kh * maxSeqLen * 256 + tokenPosition * 256
                    memcpy(kCBuf.contents().advanced(by: kCacheOffset * 4), sc.fullKBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                    let vCacheOffset = kh * maxSeqLen * 256 + tokenPosition * 256
                    memcpy(vCBuf.contents().advanced(by: vCacheOffset * 4), sc.fullVBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                }

                guard let cmdBuf1b = context.queue.makeCommandBuffer(),
                      let enc1b = cmdBuf1b.makeComputeCommandEncoder() else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 14, userInfo: nil)
                }

                var curPosU = UInt32(tokenPosition)
                var maxSeqU = UInt32(maxSeqLen)
                enc1b.setComputePipelineState(causalAttnPipe)
                enc1b.setBuffer(sc.fullQBuf, offset: 0, index: 0)
                enc1b.setBuffer(kCBuf, offset: 0, index: 1)
                enc1b.setBuffer(vCBuf, offset: 0, index: 2)
                enc1b.setBuffer(sc.gateBuf, offset: 0, index: 3)
                enc1b.setBuffer(sc.attnHeadOutBuf, offset: 0, index: 4)
                enc1b.setBytes(&curPosU, length: 4, index: 5)
                enc1b.setBytes(&maxSeqU, length: 4, index: 6)
                enc1b.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encodeGEMV(encoder: enc1b, weight: oPWBuf, vec: sc.attnHeadOutBuf, out: sc.attnOutBuf, rows: 2048, cols: 4096)

                // 3. Fused Residual Add + Post-Attention RMSNorm
                enc1b.setComputePipelineState(fusedResidualNormPipe)
                enc1b.setBuffer(sc.inBuf, offset: 0, index: 0)
                enc1b.setBuffer(sc.attnOutBuf, offset: 0, index: 1)
                enc1b.setBuffer(weights.postAttentionNormWeightBuf, offset: 0, index: 2)
                enc1b.setBuffer(sc.attnResBuf, offset: 0, index: 3)
                enc1b.setBuffer(sc.normMLPBuf, offset: 0, index: 4)
                enc1b.setBytes(&dim2048, length: 4, index: 5)
                enc1b.setBytes(&eps, length: 4, index: 6)
                enc1b.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // 4. Router GEMV
                encodeGEMV(encoder: enc1b, weight: weights.routerGateWeightBuf, vec: sc.normMLPBuf, out: sc.routerLogitsBuf, rows: 256, cols: 2048)

                enc1b.endEncoding()
                cmdBuf1b.commit()
                cmdBuf1b.waitUntilCompleted()
            }

            if weights.kind == .linearAttention {
                // 3. Fused Residual Add + Post-Attention RMSNorm
                enc1.setComputePipelineState(fusedResidualNormPipe)
                enc1.setBuffer(sc.inBuf, offset: 0, index: 0)
                enc1.setBuffer(sc.attnOutBuf, offset: 0, index: 1)
                enc1.setBuffer(weights.postAttentionNormWeightBuf, offset: 0, index: 2)
                enc1.setBuffer(sc.attnResBuf, offset: 0, index: 3)
                enc1.setBuffer(sc.normMLPBuf, offset: 0, index: 4)
                enc1.setBytes(&dim2048, length: 4, index: 5)
                enc1.setBytes(&eps, length: 4, index: 6)
                enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // 4. Router GEMV
                encodeGEMV(encoder: enc1, weight: weights.routerGateWeightBuf, vec: sc.normMLPBuf, out: sc.routerLogitsBuf, rows: 256, cols: 2048)

                enc1.endEncoding()
                cmdBuf1.commit()
                cmdBuf1.waitUntilCompleted()
            }

            // 5. Router Top-8 Selection & Softmax (CPU)
            let logitsPtr = sc.routerLogitsBuf.contents().bindMemory(to: Float.self, capacity: 256)
            let routerLogits = Array(UnsafeBufferPointer(start: logitsPtr, count: 256))

            var maxLogit = routerLogits[0]
            for v in routerLogits { if v > maxLogit { maxLogit = v } }
            var expLogits = [Float](repeating: 0, count: 256)
            var sumExp: Float = 0.0
            for i in 0..<256 {
                let e = exp(routerLogits[i] - maxLogit)
                expLogits[i] = e
                sumExp += e
            }
            var probs = [Float](repeating: 0, count: 256)
            for i in 0..<256 { probs[i] = expLogits[i] / sumExp }

            var indexedProbs: [(index: Int, prob: Float)] = (0..<256).map { ($0, probs[$0]) }
            indexedProbs.sort { $0.prob > $1.prob }

            let top8Indices = (0..<8).map { indexedProbs[$0].index }
            let top8RawScores = (0..<8).map { indexedProbs[$0].prob }
            let sumTop8 = top8RawScores.reduce(0.0, +)
            let top8Scores = top8RawScores.map { $0 / sumTop8 }

            // Phase 2: MoE + Shared Expert + Final Layer Residual (FUSED PIPELINE)
            memset(sc.moeAccumBuf.contents(), 0, 2048 * 4)

            guard let cmdBuf2 = context.queue.makeCommandBuffer(),
                  let enc2 = cmdBuf2.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 15, userInfo: [NSLocalizedDescriptionKey: "Encoder failure 2"])
            }

            var r2048: UInt32 = 2048
            var c512: UInt32 = 512

            for (i, expertIdx) in top8Indices.enumerated() {
                var score = top8Scores[i]
                let recOffset = (weights.layerIndex * 256 + expertIdx) * 6_291_456
                guard let guBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset)), length: 1024 * 2048 * 2, options: .storageModeShared, deallocator: nil),
                      let dnBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset + 4_194_304)), length: 2048 * 512 * 2, options: .storageModeShared, deallocator: nil) else {
                    continue
                }

                // 1. Gate-Up GEMV
                encodeGEMVBF16(encoder: enc2, weight: guBuf, vec: sc.normMLPBuf, out: sc.gateUpF32Buf, rows: 1024, cols: 2048)

                // 2. SiLU(gate) * up
                enc2.setComputePipelineState(gateUpSiluPipe)
                enc2.setBuffer(sc.gateUpF32Buf, offset: 0, index: 0)
                enc2.setBuffer(sc.actF32Buf, offset: 0, index: 1)
                enc2.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // 3. Fused Down GEMV + in-place weighted accumulation into moeAccumBuf
                enc2.setComputePipelineState(fusedExpertDownAccumPipe)
                enc2.setBuffer(dnBuf, offset: 0, index: 0)
                enc2.setBuffer(sc.actF32Buf, offset: 0, index: 1)
                enc2.setBuffer(sc.moeAccumBuf, offset: 0, index: 2)
                enc2.setBytes(&score, length: 4, index: 3)
                enc2.setBytes(&r2048, length: 4, index: 4)
                enc2.setBytes(&c512, length: 4, index: 5)
                enc2.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }

            // Shared Expert: Fused Gate-Up + SiLU
            var r512: UInt32 = 512
            var c2048: UInt32 = 2048
            enc2.setComputePipelineState(fusedSharedGateUpSiluPipe)
            enc2.setBuffer(weights.sharedGateProjWeightBuf, offset: 0, index: 0)
            enc2.setBuffer(weights.sharedUpProjWeightBuf, offset: 0, index: 1)
            enc2.setBuffer(sc.normMLPBuf, offset: 0, index: 2)
            enc2.setBuffer(sc.shActBuf, offset: 0, index: 3)
            enc2.setBytes(&r512, length: 4, index: 4)
            enc2.setBytes(&c2048, length: 4, index: 5)
            enc2.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // Shared Gate Coeff GEMV
            encodeGEMV(encoder: enc2, weight: weights.sharedGateCoeffWeightBuf, vec: sc.normMLPBuf, out: sc.shCoeffBuf, rows: 1, cols: 2048)

            // Fused Shared Down GEMV + Gating + Final Residual Add
            enc2.setComputePipelineState(fusedSharedDownFinalResidualPipe)
            enc2.setBuffer(weights.sharedDownProjWeightBuf, offset: 0, index: 0)
            enc2.setBuffer(sc.shActBuf, offset: 0, index: 1)
            enc2.setBuffer(sc.shCoeffBuf, offset: 0, index: 2)
            enc2.setBuffer(sc.attnResBuf, offset: 0, index: 3)
            enc2.setBuffer(sc.moeAccumBuf, offset: 0, index: 4)
            enc2.setBuffer(sc.completeOutBuf, offset: 0, index: 5)
            enc2.setBytes(&r2048, length: 4, index: 6)
            enc2.setBytes(&c512, length: 4, index: 7)
            enc2.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()

            func readBuf(_ buf: MTLBuffer, count: Int) -> [Float] {
                let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
                return Array(UnsafeBufferPointer(start: ptr, count: count))
            }

            return (
                attnOut: readBuf(sc.attnOutBuf, count: 2048),
                complete: CompleteLayerOutputs(
                    attnResidual: readBuf(sc.attnResBuf, count: 2048),
                    normMLP: readBuf(sc.normMLPBuf, count: 2048),
                    routerLogits: routerLogits,
                    top8Indices: top8Indices,
                    top8Scores: top8Scores,
                    routedMoEOut: readBuf(sc.moeAccumBuf, count: 2048),
                    sharedExpertOut: readBuf(sc.sharedOutBuf, count: 2048),
                    completeLayerOut: readBuf(sc.completeOutBuf, count: 2048)
                )
            )
        }
    }

    public func forwardLayerTokenProfiled(
        weights: QwenLayerMetalWeights,
        state: QwenLayerState,
        inputHidden: [Float],
        tokenPosition: Int,
        maxSeqLen: Int = 64,
        expertsMmapPointer: UnsafeRawPointer
    ) throws -> (attnOut: [Float], complete: CompleteLayerOutputs, timings: QwenLayerComponentTimings) {
        try autoreleasepool {
            var timings = QwenLayerComponentTimings()
            let dev = context.device

            guard let inBuf = dev.makeBuffer(bytes: inputHidden, length: 2048 * 4, options: .storageModeShared),
                  let normInBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let attnOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let attnResBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let normMLPBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let routerLogitsBuf = dev.makeBuffer(length: 256 * 4, options: .storageModeShared),
                  let gateUpF32Buf = dev.makeBuffer(length: 1024 * 4, options: .storageModeShared),
                  let actF32Buf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
                  let expertOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let moeAccumBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let shGateBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
                  let shUpBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
                  let shActBuf = dev.makeBuffer(length: 512 * 4, options: .storageModeShared),
                  let shDownBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let shCoeffBuf = dev.makeBuffer(length: 1 * 4, options: .storageModeShared),
                  let sharedOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared),
                  let completeOutBuf = dev.makeBuffer(length: 2048 * 4, options: .storageModeShared)
            else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 10, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failure"])
            }

            memset(moeAccumBuf.contents(), 0, 2048 * 4)

            let rmsnormPipe = try context.pipeline("qwen_rmsnorm_f32")
            let gemvPipe = try context.pipeline("qwen_gemv_f32")
            let addPipe = try context.pipeline("qwen_residual_add_f32")
            let gemvBF16Pipe = try context.pipeline("qwen_gemv_bf16_to_f32")
            let gateUpSiluPipe = try context.pipeline("qwen_gate_up_silu_f32")
            let siluMulPipe = try context.pipeline("qwen_silu_mul_f32")
            let expertAccumPipe = try context.pipeline("qwen_expert_accum_f32")
            let sharedGatePipe = try context.pipeline("qwen_shared_expert_gate_f32")
            let finalResidualPipe = try context.pipeline("qwen_layer_final_residual_f32")

            func encodeGEMV(encoder: MTLComputeCommandEncoder, weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
                var r = rows
                var c = cols
                encoder.setComputePipelineState(gemvPipe)
                encoder.setBuffer(weight, offset: 0, index: 0)
                encoder.setBuffer(vec, offset: 0, index: 1)
                encoder.setBuffer(out, offset: 0, index: 2)
                encoder.setBytes(&r, length: 4, index: 3)
                encoder.setBytes(&c, length: 4, index: 4)
                let tgSize = min(Int(rows), 64)
                let numTg = (Int(rows) + tgSize - 1) / tgSize
                encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            func encodeGEMVBF16(encoder: MTLComputeCommandEncoder, weight: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
                var r = rows
                var c = cols
                encoder.setComputePipelineState(gemvBF16Pipe)
                encoder.setBuffer(weight, offset: 0, index: 0)
                encoder.setBuffer(vec, offset: 0, index: 1)
                encoder.setBuffer(out, offset: 0, index: 2)
                encoder.setBytes(&r, length: 4, index: 3)
                encoder.setBytes(&c, length: 4, index: 4)
                let tgSize = min(Int(rows), 64)
                let numTg = (Int(rows) + tgSize - 1) / tgSize
                encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            var dim2048: UInt32 = 2048
            var eps: Float = 1e-6

            // Stage 1: Input RMSNorm
            let t0Norm = CFAbsoluteTimeGetCurrent()
            guard let cmdBufNorm1 = context.queue.makeCommandBuffer(),
                  let encNorm1 = cmdBufNorm1.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 11, userInfo: nil)
            }
            encNorm1.setComputePipelineState(rmsnormPipe)
            encNorm1.setBuffer(inBuf, offset: 0, index: 0)
            encNorm1.setBuffer(weights.inputRMSNormWeightBuf, offset: 0, index: 1)
            encNorm1.setBuffer(normInBuf, offset: 0, index: 2)
            encNorm1.setBytes(&dim2048, length: 4, index: 3)
            encNorm1.setBytes(&eps, length: 4, index: 4)
            encNorm1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encNorm1.endEncoding()
            cmdBufNorm1.commit()
            cmdBufNorm1.waitUntilCompleted()
            timings.inputNormMs = (CFAbsoluteTimeGetCurrent() - t0Norm) * 1000.0

            // Stage 2: Attention
            let t0Attn = CFAbsoluteTimeGetCurrent()
            guard let cmdBufAttn = context.queue.makeCommandBuffer(),
                  let encAttn = cmdBufAttn.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 12, userInfo: nil)
            }

            if weights.kind == .linearAttention {
                guard let qkvWBuf = weights.inProjQKVWeightBuf,
                      let zWBuf = weights.inProjZWeightBuf,
                      let bWBuf = weights.inProjBWeightBuf,
                      let aWBuf = weights.inProjAWeightBuf,
                      let convWBuf = weights.conv1dFlatWeightBuf,
                      let dtBiasBuf = weights.dtBiasBuf,
                      let aLogBuf = weights.aLogBuf,
                      let normWeightBuf = weights.linearNormWeightBuf,
                      let linOutProjWBuf = weights.linearOutProjWeightBuf,
                      let convStBuf = state.convStateBuf,
                      let ssmStBuf = state.ssmStateBuf,
                      let qkvOutBuf = dev.makeBuffer(length: 8192 * 4, options: .storageModeShared),
                      let zOutBuf = dev.makeBuffer(length: 4096 * 4, options: .storageModeShared),
                      let bOutBuf = dev.makeBuffer(length: 32 * 4, options: .storageModeShared),
                      let aOutBuf = dev.makeBuffer(length: 32 * 4, options: .storageModeShared),
                      let convOutBuf = dev.makeBuffer(length: 8192 * 4, options: .storageModeShared),
                      let qBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
                      let kBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
                      let vBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
                      let coreAttnBuf = dev.makeBuffer(length: 32 * 128 * 4, options: .storageModeShared),
                      let gatedNormBuf = dev.makeBuffer(length: 4096 * 4, options: .storageModeShared)
                else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 13, userInfo: nil)
                }

                encodeGEMV(encoder: encAttn, weight: qkvWBuf, vec: normInBuf, out: qkvOutBuf, rows: 8192, cols: 2048)
                encodeGEMV(encoder: encAttn, weight: zWBuf, vec: normInBuf, out: zOutBuf, rows: 4096, cols: 2048)
                encodeGEMV(encoder: encAttn, weight: bWBuf, vec: normInBuf, out: bOutBuf, rows: 32, cols: 2048)
                encodeGEMV(encoder: encAttn, weight: aWBuf, vec: normInBuf, out: aOutBuf, rows: 32, cols: 2048)

                var convDim: UInt32 = 8192
                encAttn.setComputePipelineState(conv1dPipe)
                encAttn.setBuffer(qkvOutBuf, offset: 0, index: 0)
                encAttn.setBuffer(convStBuf, offset: 0, index: 1)
                encAttn.setBuffer(convWBuf, offset: 0, index: 2)
                encAttn.setBuffer(convOutBuf, offset: 0, index: 3)
                encAttn.setBytes(&convDim, length: 4, index: 4)
                encAttn.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encAttn.setComputePipelineState(prepQKVPipe)
                encAttn.setBuffer(convOutBuf, offset: 0, index: 0)
                encAttn.setBuffer(qBuf, offset: 0, index: 1)
                encAttn.setBuffer(kBuf, offset: 0, index: 2)
                encAttn.setBuffer(vBuf, offset: 0, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encAttn.setComputePipelineState(deltanetPipe)
                encAttn.setBuffer(qBuf, offset: 0, index: 0)
                encAttn.setBuffer(kBuf, offset: 0, index: 1)
                encAttn.setBuffer(vBuf, offset: 0, index: 2)
                encAttn.setBuffer(aOutBuf, offset: 0, index: 3)
                encAttn.setBuffer(bOutBuf, offset: 0, index: 4)
                encAttn.setBuffer(aLogBuf, offset: 0, index: 5)
                encAttn.setBuffer(dtBiasBuf, offset: 0, index: 6)
                encAttn.setBuffer(ssmStBuf, offset: 0, index: 7)
                encAttn.setBuffer(coreAttnBuf, offset: 0, index: 8)
                encAttn.dispatchThreadgroups(MTLSize(width: 128 / 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                encAttn.setComputePipelineState(gatedNormPipe)
                encAttn.setBuffer(coreAttnBuf, offset: 0, index: 0)
                encAttn.setBuffer(normWeightBuf, offset: 0, index: 1)
                encAttn.setBuffer(zOutBuf, offset: 0, index: 2)
                encAttn.setBuffer(gatedNormBuf, offset: 0, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encodeGEMV(encoder: encAttn, weight: linOutProjWBuf, vec: gatedNormBuf, out: attnOutBuf, rows: 2048, cols: 4096)

                encAttn.endEncoding()
                cmdBufAttn.commit()
                cmdBufAttn.waitUntilCompleted()
            } else {
                guard let qPWBuf = weights.qProjWeightBuf,
                      let kPWBuf = weights.kProjWeightBuf,
                      let vPWBuf = weights.vProjWeightBuf,
                      let qNWBuf = weights.qNormWeightBuf,
                      let kNWBuf = weights.kNormWeightBuf,
                      let oPWBuf = weights.oProjWeightBuf,
                      let kCBuf = state.kCacheBuf,
                      let vCBuf = state.vCacheBuf,
                      let qChunkBuf = dev.makeBuffer(length: 8192 * 4, options: .storageModeShared),
                      let qBuf = dev.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared),
                      let gateBuf = dev.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared),
                      let kBuf = dev.makeBuffer(length: 2 * 256 * 4, options: .storageModeShared),
                      let vBuf = dev.makeBuffer(length: 2 * 256 * 4, options: .storageModeShared),
                      let attnHeadOutBuf = dev.makeBuffer(length: 16 * 256 * 4, options: .storageModeShared)
                else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 14, userInfo: nil)
                }

                let splitQGatePipe = try context.pipeline("qwen_split_q_gate_f32")
                let headNormPipe = try context.pipeline("qwen_head_rmsnorm_f32")
                let ropePipe = try context.pipeline("qwen_rope_partial_f32")
                let causalAttnPipe = try context.pipeline("qwen_causal_attn_step_f32")

                encodeGEMV(encoder: encAttn, weight: qPWBuf, vec: normInBuf, out: qChunkBuf, rows: 8192, cols: 2048)
                encodeGEMV(encoder: encAttn, weight: kPWBuf, vec: normInBuf, out: kBuf, rows: 512, cols: 2048)
                encodeGEMV(encoder: encAttn, weight: vPWBuf, vec: normInBuf, out: vBuf, rows: 512, cols: 2048)

                encAttn.setComputePipelineState(splitQGatePipe)
                encAttn.setBuffer(qChunkBuf, offset: 0, index: 0)
                encAttn.setBuffer(qBuf, offset: 0, index: 1)
                encAttn.setBuffer(gateBuf, offset: 0, index: 2)
                encAttn.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var numQHeads: UInt32 = 16
                var numKHeads: UInt32 = 2
                var headDim: UInt32 = 256
                encAttn.setComputePipelineState(headNormPipe)
                encAttn.setBuffer(qBuf, offset: 0, index: 0)
                encAttn.setBuffer(qNWBuf, offset: 0, index: 1)
                encAttn.setBytes(&numQHeads, length: 4, index: 2)
                encAttn.setBytes(&headDim, length: 4, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encAttn.setComputePipelineState(headNormPipe)
                encAttn.setBuffer(kBuf, offset: 0, index: 0)
                encAttn.setBuffer(kNWBuf, offset: 0, index: 1)
                encAttn.setBytes(&numKHeads, length: 4, index: 2)
                encAttn.setBytes(&headDim, length: 4, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var cosArr = [Float](repeating: 0, count: 64)
                var sinArr = [Float](repeating: 0, count: 64)
                let theta: Float = 10000000.0
                for i in 0..<32 {
                    let freq = 1.0 / pow(theta, Float(2 * i) / 64.0)
                    let angle = Float(tokenPosition) * freq
                    let c = cos(angle)
                    let s = sin(angle)
                    cosArr[i] = c; cosArr[32 + i] = c
                    sinArr[i] = s; sinArr[32 + i] = s
                }
                guard let cosBuf = dev.makeBuffer(bytes: cosArr, length: 64 * 4, options: .storageModeShared),
                      let sinBuf = dev.makeBuffer(bytes: sinArr, length: 64 * 4, options: .storageModeShared) else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 15, userInfo: nil)
                }

                encAttn.setComputePipelineState(ropePipe)
                encAttn.setBuffer(qBuf, offset: 0, index: 0)
                encAttn.setBuffer(cosBuf, offset: 0, index: 1)
                encAttn.setBuffer(sinBuf, offset: 0, index: 2)
                encAttn.setBytes(&numQHeads, length: 4, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encAttn.setComputePipelineState(ropePipe)
                encAttn.setBuffer(kBuf, offset: 0, index: 0)
                encAttn.setBuffer(cosBuf, offset: 0, index: 1)
                encAttn.setBuffer(sinBuf, offset: 0, index: 2)
                encAttn.setBytes(&numKHeads, length: 4, index: 3)
                encAttn.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encAttn.endEncoding()
                cmdBufAttn.commit()
                cmdBufAttn.waitUntilCompleted()

                for kh in 0..<2 {
                    let kCacheOffset = kh * maxSeqLen * 256 + tokenPosition * 256
                    memcpy(kCBuf.contents().advanced(by: kCacheOffset * 4), kBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                    let vCacheOffset = kh * maxSeqLen * 256 + tokenPosition * 256
                    memcpy(vCBuf.contents().advanced(by: vCacheOffset * 4), vBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                }

                guard let cmdBufCausal = context.queue.makeCommandBuffer(),
                      let encCausal = cmdBufCausal.makeComputeCommandEncoder() else {
                    throw NSError(domain: "QwenMetalDeltaNetRunner", code: 16, userInfo: nil)
                }
                var curPosU = UInt32(tokenPosition)
                var maxSeqU = UInt32(maxSeqLen)
                encCausal.setComputePipelineState(causalAttnPipe)
                encCausal.setBuffer(qBuf, offset: 0, index: 0)
                encCausal.setBuffer(kCBuf, offset: 0, index: 1)
                encCausal.setBuffer(vCBuf, offset: 0, index: 2)
                encCausal.setBuffer(gateBuf, offset: 0, index: 3)
                encCausal.setBuffer(attnHeadOutBuf, offset: 0, index: 4)
                encCausal.setBytes(&curPosU, length: 4, index: 5)
                encCausal.setBytes(&maxSeqU, length: 4, index: 6)
                encCausal.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encodeGEMV(encoder: encCausal, weight: oPWBuf, vec: attnHeadOutBuf, out: attnOutBuf, rows: 2048, cols: 4096)
                encCausal.endEncoding()
                cmdBufCausal.commit()
                cmdBufCausal.waitUntilCompleted()
            }
            timings.attentionMs = (CFAbsoluteTimeGetCurrent() - t0Attn) * 1000.0

            // Stage 3: Attention Residual
            let t0AttnRes = CFAbsoluteTimeGetCurrent()
            guard let cmdBufRes1 = context.queue.makeCommandBuffer(),
                  let encRes1 = cmdBufRes1.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 17, userInfo: nil)
            }
            encRes1.setComputePipelineState(addPipe)
            encRes1.setBuffer(inBuf, offset: 0, index: 0)
            encRes1.setBuffer(attnOutBuf, offset: 0, index: 1)
            encRes1.setBuffer(attnResBuf, offset: 0, index: 2)
            encRes1.setBytes(&dim2048, length: 4, index: 3)
            encRes1.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encRes1.endEncoding()
            cmdBufRes1.commit()
            cmdBufRes1.waitUntilCompleted()
            timings.attnResidualMs = (CFAbsoluteTimeGetCurrent() - t0AttnRes) * 1000.0

            // Stage 4: Post-Attention Norm
            let t0PostNorm = CFAbsoluteTimeGetCurrent()
            guard let cmdBufPNorm = context.queue.makeCommandBuffer(),
                  let encPNorm = cmdBufPNorm.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 18, userInfo: nil)
            }
            encPNorm.setComputePipelineState(rmsnormPipe)
            encPNorm.setBuffer(attnResBuf, offset: 0, index: 0)
            encPNorm.setBuffer(weights.postAttentionNormWeightBuf, offset: 0, index: 1)
            encPNorm.setBuffer(normMLPBuf, offset: 0, index: 2)
            encPNorm.setBytes(&dim2048, length: 4, index: 3)
            encPNorm.setBytes(&eps, length: 4, index: 4)
            encPNorm.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encPNorm.endEncoding()
            cmdBufPNorm.commit()
            cmdBufPNorm.waitUntilCompleted()
            timings.postNormMs = (CFAbsoluteTimeGetCurrent() - t0PostNorm) * 1000.0

            // Stage 5: Router GEMV & Top-8 Selection
            let t0Router = CFAbsoluteTimeGetCurrent()
            guard let cmdBufRouter = context.queue.makeCommandBuffer(),
                  let encRouter = cmdBufRouter.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 19, userInfo: nil)
            }
            encodeGEMV(encoder: encRouter, weight: weights.routerGateWeightBuf, vec: normMLPBuf, out: routerLogitsBuf, rows: 256, cols: 2048)
            encRouter.endEncoding()
            cmdBufRouter.commit()
            cmdBufRouter.waitUntilCompleted()

            let logitsPtr = routerLogitsBuf.contents().bindMemory(to: Float.self, capacity: 256)
            let routerLogits = Array(UnsafeBufferPointer(start: logitsPtr, count: 256))

            var maxLogit = routerLogits[0]
            for v in routerLogits { if v > maxLogit { maxLogit = v } }
            var expLogits = [Float](repeating: 0, count: 256)
            var sumExp: Float = 0.0
            for i in 0..<256 {
                let e = exp(routerLogits[i] - maxLogit)
                expLogits[i] = e
                sumExp += e
            }
            var probs = [Float](repeating: 0, count: 256)
            for i in 0..<256 { probs[i] = expLogits[i] / sumExp }

            var indexedProbs: [(index: Int, prob: Float)] = (0..<256).map { ($0, probs[$0]) }
            indexedProbs.sort { $0.prob > $1.prob }

            let top8Indices = (0..<8).map { indexedProbs[$0].index }
            let top8RawScores = (0..<8).map { indexedProbs[$0].prob }
            let sumTop8 = top8RawScores.reduce(0.0, +)
            let top8Scores = top8RawScores.map { $0 / sumTop8 }
            timings.routerMs = (CFAbsoluteTimeGetCurrent() - t0Router) * 1000.0

            // Stage 6: Routed MoE
            let t0MoE = CFAbsoluteTimeGetCurrent()
            guard let cmdBufMoE = context.queue.makeCommandBuffer(),
                  let encMoE = cmdBufMoE.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 20, userInfo: nil)
            }

            for (i, expertIdx) in top8Indices.enumerated() {
                var score = top8Scores[i]
                let recOffset = (weights.layerIndex * 256 + expertIdx) * 6_291_456
                guard let guBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset)), length: 1024 * 2048 * 2, options: .storageModeShared, deallocator: nil),
                      let dnBuf = dev.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: expertsMmapPointer.advanced(by: recOffset + 4_194_304)), length: 2048 * 512 * 2, options: .storageModeShared, deallocator: nil) else {
                    continue
                }

                encodeGEMVBF16(encoder: encMoE, weight: guBuf, vec: normMLPBuf, out: gateUpF32Buf, rows: 1024, cols: 2048)

                encMoE.setComputePipelineState(gateUpSiluPipe)
                encMoE.setBuffer(gateUpF32Buf, offset: 0, index: 0)
                encMoE.setBuffer(actF32Buf, offset: 0, index: 1)
                encMoE.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encodeGEMVBF16(encoder: encMoE, weight: dnBuf, vec: actF32Buf, out: expertOutBuf, rows: 2048, cols: 512)

                encMoE.setComputePipelineState(expertAccumPipe)
                encMoE.setBuffer(expertOutBuf, offset: 0, index: 0)
                encMoE.setBuffer(moeAccumBuf, offset: 0, index: 1)
                encMoE.setBytes(&score, length: 4, index: 2)
                encMoE.setBytes(&dim2048, length: 4, index: 3)
                encMoE.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            encMoE.endEncoding()
            cmdBufMoE.commit()
            cmdBufMoE.waitUntilCompleted()
            timings.routedMoEMs = (CFAbsoluteTimeGetCurrent() - t0MoE) * 1000.0

            // Stage 7: Shared Expert
            let t0Shared = CFAbsoluteTimeGetCurrent()
            guard let cmdBufShared = context.queue.makeCommandBuffer(),
                  let encShared = cmdBufShared.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 21, userInfo: nil)
            }
            encodeGEMV(encoder: encShared, weight: weights.sharedGateProjWeightBuf, vec: normMLPBuf, out: shGateBuf, rows: 512, cols: 2048)
            encodeGEMV(encoder: encShared, weight: weights.sharedUpProjWeightBuf, vec: normMLPBuf, out: shUpBuf, rows: 512, cols: 2048)

            var d512: UInt32 = 512
            encShared.setComputePipelineState(siluMulPipe)
            encShared.setBuffer(shGateBuf, offset: 0, index: 0)
            encShared.setBuffer(shUpBuf, offset: 0, index: 1)
            encShared.setBuffer(shActBuf, offset: 0, index: 2)
            encShared.setBytes(&d512, length: 4, index: 3)
            encShared.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            encodeGEMV(encoder: encShared, weight: weights.sharedDownProjWeightBuf, vec: shActBuf, out: shDownBuf, rows: 2048, cols: 512)
            encodeGEMV(encoder: encShared, weight: weights.sharedGateCoeffWeightBuf, vec: normMLPBuf, out: shCoeffBuf, rows: 1, cols: 2048)

            encShared.setComputePipelineState(sharedGatePipe)
            encShared.setBuffer(shDownBuf, offset: 0, index: 0)
            encShared.setBuffer(shCoeffBuf, offset: 0, index: 1)
            encShared.setBuffer(sharedOutBuf, offset: 0, index: 2)
            encShared.setBytes(&dim2048, length: 4, index: 3)
            encShared.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encShared.endEncoding()
            cmdBufShared.commit()
            cmdBufShared.waitUntilCompleted()
            timings.sharedExpertMs = (CFAbsoluteTimeGetCurrent() - t0Shared) * 1000.0

            // Stage 8: Final Residual
            let t0FinalRes = CFAbsoluteTimeGetCurrent()
            guard let cmdBufFinalRes = context.queue.makeCommandBuffer(),
                  let encFinalRes = cmdBufFinalRes.makeComputeCommandEncoder() else {
                throw NSError(domain: "QwenMetalDeltaNetRunner", code: 22, userInfo: nil)
            }
            encFinalRes.setComputePipelineState(finalResidualPipe)
            encFinalRes.setBuffer(attnResBuf, offset: 0, index: 0)
            encFinalRes.setBuffer(moeAccumBuf, offset: 0, index: 1)
            encFinalRes.setBuffer(sharedOutBuf, offset: 0, index: 2)
            encFinalRes.setBuffer(completeOutBuf, offset: 0, index: 3)
            encFinalRes.setBytes(&dim2048, length: 4, index: 4)
            encFinalRes.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encFinalRes.endEncoding()
            cmdBufFinalRes.commit()
            cmdBufFinalRes.waitUntilCompleted()
            timings.finalResidualMs = (CFAbsoluteTimeGetCurrent() - t0FinalRes) * 1000.0

            timings.totalLayerMs = timings.inputNormMs + timings.attentionMs + timings.attnResidualMs + timings.postNormMs + timings.routerMs + timings.routedMoEMs + timings.sharedExpertMs + timings.finalResidualMs

            func readBuf(_ buf: MTLBuffer, count: Int) -> [Float] {
                let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
                return Array(UnsafeBufferPointer(start: ptr, count: count))
            }

            return (
                attnOut: readBuf(attnOutBuf, count: 2048),
                complete: CompleteLayerOutputs(
                    attnResidual: readBuf(attnResBuf, count: 2048),
                    normMLP: readBuf(normMLPBuf, count: 2048),
                    routerLogits: routerLogits,
                    top8Indices: top8Indices,
                    top8Scores: top8Scores,
                    routedMoEOut: readBuf(moeAccumBuf, count: 2048),
                    sharedExpertOut: readBuf(sharedOutBuf, count: 2048),
                    completeLayerOut: readBuf(completeOutBuf, count: 2048)
                ),
                timings: timings
            )
        }
    }
}

public struct QwenLayerComponentTimings: Sendable {
    public var inputNormMs: Double = 0
    public var attentionMs: Double = 0
    public var attnResidualMs: Double = 0
    public var postNormMs: Double = 0
    public var routerMs: Double = 0
    public var routedMoEMs: Double = 0
    public var sharedExpertMs: Double = 0
    public var finalResidualMs: Double = 0
    public var totalLayerMs: Double = 0

    public init() {}
}

public final class QwenLayerMetalWeights: @unchecked Sendable {
    public let layerIndex: Int
    public let kind: QwenLayerKind
    public let inputRMSNormWeightBuf: MTLBuffer
    public let postAttentionNormWeightBuf: MTLBuffer
    public let routerGateWeightBuf: MTLBuffer
    public let sharedGateProjWeightBuf: MTLBuffer
    public let sharedUpProjWeightBuf: MTLBuffer
    public let sharedDownProjWeightBuf: MTLBuffer
    public let sharedGateCoeffWeightBuf: MTLBuffer

    // Linear Attention
    public let inProjQKVWeightBuf: MTLBuffer?
    public let inProjZWeightBuf: MTLBuffer?
    public let inProjBWeightBuf: MTLBuffer?
    public let inProjAWeightBuf: MTLBuffer?
    public let conv1dFlatWeightBuf: MTLBuffer?
    public let dtBiasBuf: MTLBuffer?
    public let aLogBuf: MTLBuffer?
    public let linearNormWeightBuf: MTLBuffer?
    public let linearOutProjWeightBuf: MTLBuffer?

    // Full Attention
    public let qProjWeightBuf: MTLBuffer?
    public let kProjWeightBuf: MTLBuffer?
    public let vProjWeightBuf: MTLBuffer?
    public let qNormWeightBuf: MTLBuffer?
    public let kNormWeightBuf: MTLBuffer?
    public let oProjWeightBuf: MTLBuffer?

    public init(weights: QwenLayerWeights, device: MTLDevice) throws {
        self.layerIndex = weights.layerIndex
        self.kind = weights.kind

        guard let inNormBuf = device.makeBuffer(bytes: weights.inputRMSNormWeight, length: 2048 * 4, options: .storageModeShared),
              let postNormBuf = device.makeBuffer(bytes: weights.postAttentionNormWeight, length: 2048 * 4, options: .storageModeShared),
              let routerBuf = device.makeBuffer(bytes: weights.routerGateWeight, length: 256 * 2048 * 4, options: .storageModeShared),
              let shGateBuf = device.makeBuffer(bytes: weights.sharedGateProjWeight, length: 512 * 2048 * 4, options: .storageModeShared),
              let shUpBuf = device.makeBuffer(bytes: weights.sharedUpProjWeight, length: 512 * 2048 * 4, options: .storageModeShared),
              let shDownBuf = device.makeBuffer(bytes: weights.sharedDownProjWeight, length: 2048 * 512 * 4, options: .storageModeShared),
              let shCoeffBuf = device.makeBuffer(bytes: weights.sharedGateCoeffWeight, length: 2048 * 4, options: .storageModeShared)
        else {
            throw NSError(domain: "QwenLayerMetalWeights", code: 1, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }

        self.inputRMSNormWeightBuf = inNormBuf
        self.postAttentionNormWeightBuf = postNormBuf
        self.routerGateWeightBuf = routerBuf
        self.sharedGateProjWeightBuf = shGateBuf
        self.sharedUpProjWeightBuf = shUpBuf
        self.sharedDownProjWeightBuf = shDownBuf
        self.sharedGateCoeffWeightBuf = shCoeffBuf

        if weights.kind == .linearAttention {
            guard let qkv = weights.inProjQKVWeight,
                  let z = weights.inProjZWeight,
                  let b = weights.inProjBWeight,
                  let a = weights.inProjAWeight,
                  let conv = weights.conv1dFlatWeight,
                  let dtB = weights.dtBias,
                  let aL = weights.aLog,
                  let lNorm = weights.linearNormWeight,
                  let lOut = weights.linearOutProjWeight,
                  let qkvBuf = device.makeBuffer(bytes: qkv, length: 8192 * 2048 * 4, options: .storageModeShared),
                  let zBuf = device.makeBuffer(bytes: z, length: 4096 * 2048 * 4, options: .storageModeShared),
                  let bBuf = device.makeBuffer(bytes: b, length: 32 * 2048 * 4, options: .storageModeShared),
                  let aBuf = device.makeBuffer(bytes: a, length: 32 * 2048 * 4, options: .storageModeShared),
                  let convBuf = device.makeBuffer(bytes: conv, length: 8192 * 4 * 4, options: .storageModeShared),
                  let dtBBuf = device.makeBuffer(bytes: dtB, length: 32 * 4, options: .storageModeShared),
                  let aLBuf = device.makeBuffer(bytes: aL, length: 32 * 4, options: .storageModeShared),
                  let lNormBuf = device.makeBuffer(bytes: lNorm, length: 128 * 4, options: .storageModeShared),
                  let lOutBuf = device.makeBuffer(bytes: lOut, length: 2048 * 4096 * 4, options: .storageModeShared)
            else {
                throw NSError(domain: "QwenLayerMetalWeights", code: 2, userInfo: [NSLocalizedDescriptionKey: "Linear buffer allocation failed"])
            }
            self.inProjQKVWeightBuf = qkvBuf
            self.inProjZWeightBuf = zBuf
            self.inProjBWeightBuf = bBuf
            self.inProjAWeightBuf = aBuf
            self.conv1dFlatWeightBuf = convBuf
            self.dtBiasBuf = dtBBuf
            self.aLogBuf = aLBuf
            self.linearNormWeightBuf = lNormBuf
            self.linearOutProjWeightBuf = lOutBuf

            self.qProjWeightBuf = nil
            self.kProjWeightBuf = nil
            self.vProjWeightBuf = nil
            self.qNormWeightBuf = nil
            self.kNormWeightBuf = nil
            self.oProjWeightBuf = nil
        } else {
            guard let qP = weights.qProjWeight,
                  let kP = weights.kProjWeight,
                  let vP = weights.vProjWeight,
                  let qN = weights.qNormWeight,
                  let kN = weights.kNormWeight,
                  let oP = weights.oProjWeight,
                  let qPBuf = device.makeBuffer(bytes: qP, length: 8192 * 2048 * 4, options: .storageModeShared),
                  let kPBuf = device.makeBuffer(bytes: kP, length: 512 * 2048 * 4, options: .storageModeShared),
                  let vPBuf = device.makeBuffer(bytes: vP, length: 512 * 2048 * 4, options: .storageModeShared),
                  let qNBuf = device.makeBuffer(bytes: qN, length: 256 * 4, options: .storageModeShared),
                  let kNBuf = device.makeBuffer(bytes: kN, length: 256 * 4, options: .storageModeShared),
                  let oPBuf = device.makeBuffer(bytes: oP, length: 2048 * 4096 * 4, options: .storageModeShared)
            else {
                throw NSError(domain: "QwenLayerMetalWeights", code: 3, userInfo: [NSLocalizedDescriptionKey: "Full attention buffer allocation failed"])
            }
            self.inProjQKVWeightBuf = nil
            self.inProjZWeightBuf = nil
            self.inProjBWeightBuf = nil
            self.inProjAWeightBuf = nil
            self.conv1dFlatWeightBuf = nil
            self.dtBiasBuf = nil
            self.aLogBuf = nil
            self.linearNormWeightBuf = nil
            self.linearOutProjWeightBuf = nil

            self.qProjWeightBuf = qPBuf
            self.kProjWeightBuf = kPBuf
            self.vProjWeightBuf = vPBuf
            self.qNormWeightBuf = qNBuf
            self.kNormWeightBuf = kNBuf
            self.oProjWeightBuf = oPBuf
        }
    }
}

public final class QwenLayerState: @unchecked Sendable {
    public let kind: QwenLayerKind
    public let convStateBuf: MTLBuffer?
    public let ssmStateBuf: MTLBuffer?
    public let kCacheBuf: MTLBuffer?
    public let vCacheBuf: MTLBuffer?

    public init(kind: QwenLayerKind, device: MTLDevice, maxSeqLen: Int = 64) throws {
        self.kind = kind
        if kind == .linearAttention {
            self.convStateBuf = device.makeBuffer(length: 8192 * 4 * 4, options: .storageModeShared)
            self.ssmStateBuf = device.makeBuffer(length: 32 * 128 * 128 * 4, options: .storageModeShared)
            self.kCacheBuf = nil
            self.vCacheBuf = nil
            memset(self.convStateBuf!.contents(), 0, 8192 * 4 * 4)
            memset(self.ssmStateBuf!.contents(), 0, 32 * 128 * 128 * 4)
        } else {
            self.convStateBuf = nil
            self.ssmStateBuf = nil
            self.kCacheBuf = device.makeBuffer(length: 2 * maxSeqLen * 256 * 4, options: .storageModeShared)
            self.vCacheBuf = device.makeBuffer(length: 2 * maxSeqLen * 256 * 4, options: .storageModeShared)
            memset(self.kCacheBuf!.contents(), 0, 2 * maxSeqLen * 256 * 4)
            memset(self.vCacheBuf!.contents(), 0, 2 * maxSeqLen * 256 * 4)
        }
    }
}
