import Foundation
import Metal
import Accelerate
import Tokenizers

public final class DenseLayerArena: @unchecked Sendable {
    // Exact end of shared_expert.down_proj.biases. The former 24 MiB round-up
    // left 2,709,376 unused bytes in each of 41 allocated arenas.
    public static let totalByteLength: Int = 22_456_448

    public let device: MTLDevice
    public let buffer: MTLBuffer

    // Fixed Sub-Offsets
    public static let qkvWOff: Int = 0
    public static let qkvSOff: Int = 8_388_608
    public static let qkvBOff: Int = 8_912_896
    public static let zWOff: Int = 9_437_184
    public static let zSOff: Int = 13_631_488
    public static let zBOff: Int = 13_893_632
    public static let bWOff: Int = 14_155_776
    public static let bSOff: Int = 14_188_544
    public static let bBOff: Int = 14_190_592
    public static let aWOff: Int = 14_192_640
    public static let aSOff: Int = 14_225_408
    public static let aBOff: Int = 14_227_456
    public static let linOutWOff: Int = 14_229_504
    public static let linOutSOff: Int = 18_423_808
    public static let linOutBOff: Int = 18_685_952
    public static let kProjWOff: Int = 18_948_096
    public static let kProjSOff: Int = 19_472_384
    public static let kProjBOff: Int = 19_505_152
    public static let vProjWOff: Int = 19_537_920
    public static let vProjSOff: Int = 20_062_208
    public static let vProjBOff: Int = 20_094_976
    public static let routerWOff: Int = 20_127_744
    public static let routerSOff: Int = 20_652_032
    public static let routerBOff: Int = 20_668_416
    public static let shCoeffWOff: Int = 20_684_800
    public static let shCoeffSOff: Int = 20_686_848
    public static let shCoeffBOff: Int = 20_686_912
    public static let shGateWOff: Int = 20_686_976
    public static let shGateSOff: Int = 21_211_264
    public static let shGateBOff: Int = 21_244_032
    public static let shUpWOff: Int = 21_276_800
    public static let shUpSOff: Int = 21_801_088
    public static let shUpBOff: Int = 21_833_856
    public static let shDownWOff: Int = 21_866_624
    public static let shDownSOff: Int = 22_390_912
    public static let shDownBOff: Int = 22_423_680

    public init(device: MTLDevice) throws {
        self.device = device
        guard let buf = device.makeBuffer(length: Self.totalByteLength, options: .storageModeShared) else {
            throw NSError(domain: "DenseLayerArena", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate DenseLayerArena buffer"])
        }
        self.buffer = buf
    }
}

public final class QwenQ4ActiveExpertStreamingRunner: @unchecked Sendable {
    public enum LayerKind: Sendable {
        case linearAttention
        case fullAttention
    }

    public struct StepProfile: Sendable {
        public let step: Int
        public let tokenId: Int32
        public let totalStepMs: Double

        // Non-overlapping CPU breakdown (summing to totalStepMs within ~1%)
        public let cpuSetupMs: Double
        public let cbEncodeMs: Double
        public let cbCommitWaitMs: Double
        public let cacheLookupMs: Double
        public let cacheMissFetchMs: Double
        public let cpuArgmaxMs: Double

        // Component breakdown
        public let denseStagingMs: Double
        public let routerMs: Double
        public let expertStagingMs: Double
        public let attentionMs: Double
        public let deltanetMs: Double
        public let stagedMoEMs: Double
        public let normsResidualsMs: Double
        public let lmHeadMs: Double
        public let attentionRouterGPUMs: Double
        public let sharedHitGPUMs: Double
        public let moeTailGPUMs: Double
        public let totalGPUMs: Double
        public let bytesStaged: Int
    }

    private struct MmapShard {
        let fd: Int32
        let ptr: UnsafeRawPointer
        let size: Int
        let headerLen: Int
        let headerJson: [String: Any]
    }

    private struct RawSlice: Sendable {
        let shardName: String
        let fileOffset: Int
        let size: Int
    }

    private struct SendableMutableRawPointer: @unchecked Sendable {
        let value: UnsafeMutableRawPointer
    }

    private struct LayerDensePointers: Sendable {
        let kind: LayerKind
        // Linear / Full Attention
        let qkvW: RawSlice?
        let qkvS: RawSlice?
        let qkvB: RawSlice?
        let zW: RawSlice?
        let zS: RawSlice?
        let zB: RawSlice?
        let bW: RawSlice?
        let bS: RawSlice?
        let bB: RawSlice?
        let aW: RawSlice?
        let aS: RawSlice?
        let aB: RawSlice?
        let outW: RawSlice
        let outS: RawSlice
        let outB: RawSlice
        let kW: RawSlice?
        let kS: RawSlice?
        let kB: RawSlice?
        let vW: RawSlice?
        let vS: RawSlice?
        let vB: RawSlice?
        // Router & Shared
        let routerW: RawSlice
        let routerS: RawSlice
        let routerB: RawSlice
        let shCoeffW: RawSlice
        let shCoeffS: RawSlice
        let shCoeffB: RawSlice
        let shGateW: RawSlice
        let shGateS: RawSlice
        let shGateB: RawSlice
        let shUpW: RawSlice
        let shUpS: RawSlice
        let shUpB: RawSlice
        let shDownW: RawSlice
        let shDownS: RawSlice
        let shDownB: RawSlice
        // MoE Switch slices for 256 experts
        let switchGateW: RawSlice
        let switchGateS: RawSlice
        let switchGateB: RawSlice
        let switchUpW: RawSlice
        let switchUpS: RawSlice
        let switchUpB: RawSlice
        let switchDownW: RawSlice
        let switchDownS: RawSlice
        let switchDownB: RawSlice
    }

    private struct StaticLayerConstants {
        let kind: LayerKind
        let inNormBuf: MTLBuffer
        let postNormBuf: MTLBuffer
        let conv1dWBuf: MTLBuffer?
        let dtBiasBuf: MTLBuffer?
        let aLogBuf: MTLBuffer?
        let linNormWBuf: MTLBuffer?
        let qNormWBuf: MTLBuffer?
        let kNormWBuf: MTLBuffer?
    }

    private let metalContext: MetalContext
    private let device: MTLDevice
    private var shards: [String: MmapShard] = [:]
    private var weightMap: [String: String] = [:]

    private var layerPointers: [LayerDensePointers] = []
    private var staticConstants: [StaticLayerConstants] = []
    public private(set) var layerKinds: [LayerKind] = []
    private var packedExpertStore: QwenQ4ExpertStore?

    // Arenas & Persistent Expert Cache
    private let denseArena: DenseLayerArena?
    public let expertCache: Q4BoundedExpertCache
    public let usePersistentExpertCache: Bool
    public let keepDenseWeightsResident: Bool
    private let enableGPUProfiling: Bool
    private var residentDenseArenas: [DenseLayerArena] = []
    private var perLayerRecentExperts: [Int: [UInt32]] = [:]

    // Permanent Resident Tensors
    private var embedW: RawSlice!
    private var embedS: RawSlice!
    private var embedB: RawSlice!
    private var lmHeadWBuf: MTLBuffer!
    private var lmHeadSBuf: MTLBuffer!
    private var lmHeadBBuf: MTLBuffer!
    private var lmHeadWOff: Int = 0
    private var lmHeadSOff: Int = 0
    private var lmHeadBOff: Int = 0
    private var finalNormWBuf: MTLBuffer!

    // State Buffers
    private var convStates: [MTLBuffer] = []
    private var ssmStates: [MTLBuffer] = []
    private var kCaches: [MTLBuffer] = []
    private var vCaches: [MTLBuffer] = []
    public let maxSeqLen: Int

    // Scratch Buffers
    private let hiddenBuf0: MTLBuffer
    private let hiddenBuf1: MTLBuffer
    private let normInBuf: MTLBuffer
    private let qkvOutBuf: MTLBuffer
    private let zOutBuf: MTLBuffer
    private let bOutBuf: MTLBuffer
    private let aOutBuf: MTLBuffer
    private let convOutBuf: MTLBuffer
    private let qBuf: MTLBuffer
    private let kBuf: MTLBuffer
    private let vBuf: MTLBuffer
    private let coreAttnBuf: MTLBuffer
    private let gatedNormBuf: MTLBuffer
    private let qChunkBuf: MTLBuffer
    private let fullGateBuf: MTLBuffer
    private let attnHeadOutBuf: MTLBuffer
    private let qHalfBuf: MTLBuffer
    private let attentionHalfOutBuf: MTLBuffer
    private let cosBuf: MTLBuffer
    private let sinBuf: MTLBuffer
    private let attnOutBuf: MTLBuffer
    private let attnResBuf: MTLBuffer
    private let normMLPBuf: MTLBuffer
    private let batchedActBuf: MTLBuffer
    private let shActBuf: MTLBuffer
    private let sharedOutBuf: MTLBuffer
    private let routedHitOutBuf: MTLBuffer
    private let routedMissOutBuf: MTLBuffer
    private let expertIndicesBuf: MTLBuffer
    private let activeSlotIndicesBuf: MTLBuffer
    private let hitExpertPositionsBuf: MTLBuffer
    private let missExpertPositionsBuf: MTLBuffer
    private let routerLogitsBuf: MTLBuffer
    private let scoresBuf: MTLBuffer
    private let attentionScoresBuf: MTLBuffer
    private let shCoeffBuf: MTLBuffer
    private let finalNormOutBuf: MTLBuffer
    private let logitsOutBuf: MTLBuffer

    // Metal Pipelines
    private let q4Pipe: MTLComputePipelineState
    private let q8Pipe: MTLComputePipelineState
    private let rmsnormPipe: MTLComputePipelineState
    private let conv1dPipe: MTLComputePipelineState
    private let prepQKVPipe: MTLComputePipelineState
    private let deltanetPipe: MTLComputePipelineState
    private let gatedNormPipe: MTLComputePipelineState
    private let fusedDeltanetStepPipe: MTLComputePipelineState
    private let addPipe: MTLComputePipelineState
    private let splitQGatePipe: MTLComputePipelineState
    private let headNormPipe: MTLComputePipelineState
    private let ropePipe: MTLComputePipelineState
    private let updateKVCachePipe: MTLComputePipelineState
    private let updateKVCacheHalfPipe: MTLComputePipelineState
    private let floatToHalfPipe: MTLComputePipelineState
    private let applyAttentionGatePipe: MTLComputePipelineState
    private let causalAttnPipe: MTLComputePipelineState
    private let splitAttention: Attention
    private let fusedRouterPipe: MTLComputePipelineState
    private let fusedLinInProjPipe: MTLComputePipelineState
    private let stagedMoeGateUpSiluPipe: MTLComputePipelineState
    private let stagedMoeDownAccumResidualPipe: MTLComputePipelineState
    private let sharedGateUpSiluPipe: MTLComputePipelineState
    private let cachedMoeGateUpSiluPipe: MTLComputePipelineState
    private let cachedMoeDownAccumResidualPipe: MTLComputePipelineState
    private let q4SimdPipe: MTLComputePipelineState
    private let fusedLinInProjSimdPipe: MTLComputePipelineState
    private let sharedGateUpSiluSimdPipe: MTLComputePipelineState
    private let sharedDownGateSimdPipe: MTLComputePipelineState
    private let cachedMoeGateUpSiluSimdPipe: MTLComputePipelineState
    private let cachedMoeDownResidualSimdPipe: MTLComputePipelineState
    private let cachedMoeGateUpSubsetSimdPipe: MTLComputePipelineState
    private let cachedMoeDownSubsetSimdPipe: MTLComputePipelineState
    private let combineMoeSubsetsPipe: MTLComputePipelineState
    private let routerQ8LogitsSimdPipe: MTLComputePipelineState
    private let routerTop8Pipe: MTLComputePipelineState

    public var latestProfile: StepProfile?

    public convenience init(
        modelPath: String,
        maxSeqLen: Int = 2048,
        expertCacheCapacity: Int = 1024,
        usePersistentExpertCache: Bool = true,
        keepDenseWeightsResident: Bool = true
    ) throws {
        try self.init(
            modelPath: modelPath,
            maxSeqLen: maxSeqLen,
            expertCacheCapacity: expertCacheCapacity,
            usePersistentExpertCache: usePersistentExpertCache,
            keepDenseWeightsResident: keepDenseWeightsResident,
            usePackedExpertStore: false,
            recordExpertAccessTrace: false,
            enableGPUProfiling: false)
    }

    public init(
        modelPath: String,
        maxSeqLen: Int,
        expertCacheCapacity: Int,
        usePersistentExpertCache: Bool,
        keepDenseWeightsResident: Bool,
        usePackedExpertStore: Bool,
        recordExpertAccessTrace: Bool,
        enableGPUProfiling: Bool
    ) throws {
        self.maxSeqLen = maxSeqLen
        self.usePersistentExpertCache = usePersistentExpertCache
        self.keepDenseWeightsResident = keepDenseWeightsResident
        self.enableGPUProfiling = enableGPUProfiling
        self.metalContext = try MetalContext()
        self.device = metalContext.device
        self.splitAttention = try Attention(context: metalContext)

        // 1. Allocate Fixed Reusable Arenas & Cache
        self.denseArena = keepDenseWeightsResident ? nil : try DenseLayerArena(device: device)
        self.expertCache = try Q4BoundedExpertCache(
            device: device,
            capacity: expertCacheCapacity,
            recordsAccessTrace: recordExpertAccessTrace)

        if keepDenseWeightsResident {
            for _ in 0..<40 {
                let arena = try DenseLayerArena(device: device)
                residentDenseArenas.append(arena)
            }
        }

        // 2. Allocate Scratch Buffers (~10 MB total)
        guard let h0 = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let h1 = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let ni = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let qkvo = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let zo = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let bo = device.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let ao = device.makeBuffer(length: 32 * 4, options: .storageModeShared),
              let co = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let qb = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let kb = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let vb = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let cab = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let gnb = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let qcb = device.makeBuffer(length: 8192 * 4, options: .storageModeShared),
              let fgb = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let ahob = device.makeBuffer(length: 4096 * 4, options: .storageModeShared),
              let qhb = device.makeBuffer(length: 4096 * 2, options: .storageModeShared),
              let ahhb = device.makeBuffer(length: 4096 * 2, options: .storageModeShared),
              let cosB = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
              let sinB = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
              let aob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let arb = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let nmb = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let bab = device.makeBuffer(length: 8 * 512 * 4, options: .storageModeShared),
              let sab = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let sob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let rhob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let rmob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let eib = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let asib = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let hepb = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let mepb = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let rlb = device.makeBuffer(length: 256 * 4, options: .storageModeShared),
              let scb = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let attsb = device.makeBuffer(length: 16 * maxSeqLen * 4, options: .storageModeShared),
              let shcb = device.makeBuffer(length: 1 * 4, options: .storageModeShared),
              let fnob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let lob = device.makeBuffer(length: 248320 * 4, options: .storageModeShared) else {
            throw QwenQ4RunnerError.metalBufferAllocationFailed("Failed to allocate scratch buffers")
        }

        self.hiddenBuf0 = h0; self.hiddenBuf1 = h1; self.normInBuf = ni
        self.qkvOutBuf = qkvo; self.zOutBuf = zo; self.bOutBuf = bo; self.aOutBuf = ao
        self.convOutBuf = co; self.qBuf = qb; self.kBuf = kb; self.vBuf = vb
        self.coreAttnBuf = cab; self.gatedNormBuf = gnb; self.qChunkBuf = qcb; self.fullGateBuf = fgb
        self.attnHeadOutBuf = ahob; self.qHalfBuf = qhb; self.attentionHalfOutBuf = ahhb
        self.cosBuf = cosB; self.sinBuf = sinB
        self.attnOutBuf = aob; self.attnResBuf = arb; self.normMLPBuf = nmb
        self.batchedActBuf = bab; self.shActBuf = sab; self.sharedOutBuf = sob
        self.routedHitOutBuf = rhob; self.routedMissOutBuf = rmob
        self.expertIndicesBuf = eib; self.activeSlotIndicesBuf = asib
        self.hitExpertPositionsBuf = hepb; self.missExpertPositionsBuf = mepb
        self.routerLogitsBuf = rlb
        self.scoresBuf = scb; self.attentionScoresBuf = attsb; self.shCoeffBuf = shcb
        self.finalNormOutBuf = fnob; self.logitsOutBuf = lob

        // 3. Compile Pipelines
        self.q4Pipe = try metalContext.pipeline("qwen_gemv_q4_affine")
        self.q8Pipe = try metalContext.pipeline("qwen_gemv_q8_affine")
        self.rmsnormPipe = try metalContext.pipeline("qwen_rmsnorm_f32")
        self.conv1dPipe = try metalContext.pipeline("qwen_conv1d_f32")
        self.prepQKVPipe = try metalContext.pipeline("qwen_prep_qkv_f32")
        self.deltanetPipe = try metalContext.pipeline("qwen_deltanet_recurrent_f32")
        self.gatedNormPipe = try metalContext.pipeline("qwen_gated_rmsnorm_f32")
        self.fusedDeltanetStepPipe = try metalContext.pipeline("qwen_fused_deltanet_step_f32")
        self.addPipe = try metalContext.pipeline("qwen_residual_add_f32")
        self.splitQGatePipe = try metalContext.pipeline("qwen_split_q_gate_f32")
        self.headNormPipe = try metalContext.pipeline("qwen_head_rmsnorm_f32")
        self.ropePipe = try metalContext.pipeline("qwen_rope_partial_f32")
        self.updateKVCachePipe = try metalContext.pipeline("qwen_update_kv_cache_f32")
        self.updateKVCacheHalfPipe = try metalContext.pipeline("qwen_update_kv_cache_f16")
        self.floatToHalfPipe = try metalContext.pipeline("qwen_f32_to_f16")
        self.applyAttentionGatePipe = try metalContext.pipeline("qwen_apply_attention_gate_f16_f32")
        self.causalAttnPipe = try metalContext.pipeline("qwen_causal_attn_step_scratch_f32")
        self.fusedRouterPipe = try metalContext.pipeline("qwen_fused_router_top8_q8")
        self.fusedLinInProjPipe = try metalContext.pipeline("qwen_fused_linear_attn_in_proj_q4")
        self.stagedMoeGateUpSiluPipe = try metalContext.pipeline("qwen_staged_moe_gate_up_silu_q4")
        self.stagedMoeDownAccumResidualPipe = try metalContext.pipeline("qwen_staged_moe_down_accum_residual_q4")
        self.sharedGateUpSiluPipe = try metalContext.pipeline("qwen_shared_gate_up_silu_q4")
        self.cachedMoeGateUpSiluPipe = try metalContext.pipeline("qwen_cached_moe_gate_up_silu_q4")
        self.cachedMoeDownAccumResidualPipe = try metalContext.pipeline("qwen_cached_moe_down_accum_residual_q4")
        self.q4SimdPipe = try metalContext.pipeline("qwen_gemv_q4_affine_simd_f32")
        self.fusedLinInProjSimdPipe = try metalContext.pipeline("qwen_fused_linear_attn_in_proj_q4_simd_f32")
        self.sharedGateUpSiluSimdPipe = try metalContext.pipeline("qwen_shared_gate_up_silu_q4_simd_f32")
        self.sharedDownGateSimdPipe = try metalContext.pipeline("qwen_shared_down_gate_q4_simd_f32")
        self.cachedMoeGateUpSiluSimdPipe = try metalContext.pipeline("qwen_cached_moe_gate_up_silu_q4_simd_f32")
        self.cachedMoeDownResidualSimdPipe = try metalContext.pipeline("qwen_cached_moe_down_residual_q4_simd_f32")
        self.cachedMoeGateUpSubsetSimdPipe = try metalContext.pipeline("qwen_cached_moe_gate_up_subset_q4_simd_f32")
        self.cachedMoeDownSubsetSimdPipe = try metalContext.pipeline("qwen_cached_moe_down_subset_q4_simd_f32")
        self.combineMoeSubsetsPipe = try metalContext.pipeline("qwen_combine_cached_moe_subsets_f32")
        self.routerQ8LogitsSimdPipe = try metalContext.pipeline("qwen_router_q8_logits_simd_f32")
        self.routerTop8Pipe = try metalContext.pipeline("qwen_router_top8_logits_f32")

        // 4. Load Shards & Metadata
        try loadModelMetadata(modelPath: modelPath)
        if usePackedExpertStore {
            self.packedExpertStore = try QwenQ4ExpertStore.openIfPresent(modelPath: modelPath)
        }
        try initStates()

        if keepDenseWeightsResident {
            for l in 0..<40 {
                stageDenseWeights(forLayer: l, targetBuffer: residentDenseArenas[l].buffer)
            }
        }
    }

    deinit {
        for (_, s) in shards {
            munmap(UnsafeMutableRawPointer(mutating: s.ptr), s.size)
            close(s.fd)
        }
    }

    private func getRawSlice(_ name: String) throws -> RawSlice {
        guard let shardName = weightMap[name],
              let shard = shards[shardName],
              let dict = shard.headerJson[name] as? [String: Any],
              let offs = dict["data_offsets"] as? [Int] else {
            throw QwenQ4RunnerError.missingTensor("Missing tensor \(name)")
        }
        let fileOffset = 8 + shard.headerLen + offs[0]
        let size = offs[1] - offs[0]
        return RawSlice(shardName: shardName, fileOffset: fileOffset, size: size)
    }

    private func decodeBF16Tensor(_ slice: RawSlice) -> [Float] {
        let shard = shards[slice.shardName]!
        let count = slice.size / 2
        var out = [Float](repeating: 0, count: count)
        let u16Ptr = shard.ptr.advanced(by: slice.fileOffset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            let u32Val = UInt32(u16Ptr[i]) << 16
            out[i] = Float(bitPattern: u32Val)
        }
        return out
    }

    private func makeZeroCopyBuffer(slice: RawSlice) throws -> (buffer: MTLBuffer, offset: Int) {
        guard let shard = shards[slice.shardName] else {
            throw QwenQ4RunnerError.missingTensor("Shard \(slice.shardName) not found")
        }
        let pageSize = Int(getpagesize())
        let pageOffset = slice.fileOffset % pageSize
        let alignedOffset = slice.fileOffset - pageOffset
        let alignedLength = ((slice.size + pageOffset + pageSize - 1) / pageSize) * pageSize
        let alignedPtr = shard.ptr.advanced(by: alignedOffset)
        guard let buf = device.makeBuffer(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: alignedPtr),
            length: alignedLength,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw QwenQ4RunnerError.metalBufferAllocationFailed("Failed to make zero-copy buffer")
        }
        return (buf, pageOffset)
    }

    private func loadModelMetadata(modelPath: String) throws {
        let indexPath = "\(modelPath)/model.safetensors.index.json"
        let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        guard let indexJson = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let wm = indexJson["weight_map"] as? [String: String] else {
            throw QwenQ4RunnerError.invalidIndexJSON("Failed to parse weight_map")
        }
        self.weightMap = wm

        let shardNames = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"]
        for sName in shardNames {
            let sPath = "\(modelPath)/\(sName)"
            let sFd = open(sPath, O_RDONLY)
            guard sFd >= 0 else { throw QwenQ4RunnerError.failedToOpenShard("Failed to open \(sPath)") }
            var sStat = stat()
            fstat(sFd, &sStat)
            let sSize = Int(sStat.st_size)
            guard let sPtr = mmap(nil, sSize, PROT_READ, MAP_SHARED, sFd, 0), sPtr != MAP_FAILED else {
                close(sFd)
                throw QwenQ4RunnerError.failedToMmapShard("Failed to mmap \(sPath)")
            }
            madvise(UnsafeMutableRawPointer(mutating: sPtr), sSize, MADV_RANDOM)

            let hLen = Int(sPtr.load(as: UInt64.self))
            let hData = Data(bytes: sPtr.advanced(by: 8), count: hLen)
            guard let hJson = try JSONSerialization.jsonObject(with: hData) as? [String: Any] else {
                throw QwenQ4RunnerError.invalidIndexJSON("Invalid JSON header in \(sName)")
            }
            shards[sName] = MmapShard(fd: sFd, ptr: sPtr, size: sSize, headerLen: hLen, headerJson: hJson)
        }

        // Embedding & LM Head slices
        self.embedW = try getRawSlice("language_model.model.embed_tokens.weight")
        self.embedS = try getRawSlice("language_model.model.embed_tokens.scales")
        self.embedB = try getRawSlice("language_model.model.embed_tokens.biases")

        let lmW = try getRawSlice("language_model.lm_head.weight")
        let lmS = try getRawSlice("language_model.lm_head.scales")
        let lmB = try getRawSlice("language_model.lm_head.biases")
        let (lmWB, lmWO) = try makeZeroCopyBuffer(slice: lmW)
        let (lmSB, lmSO) = try makeZeroCopyBuffer(slice: lmS)
        let (lmBB, lmBO) = try makeZeroCopyBuffer(slice: lmB)
        self.lmHeadWBuf = lmWB; self.lmHeadSBuf = lmSB; self.lmHeadBBuf = lmBB
        self.lmHeadWOff = lmWO; self.lmHeadSOff = lmSO; self.lmHeadBOff = lmBO

        let finalNormWeight = decodeBF16Tensor(try getRawSlice("language_model.model.norm.weight"))
        self.finalNormWBuf = device.makeBuffer(bytes: finalNormWeight, length: 2048 * 4, options: .storageModeShared)!

        // All 40 layers metadata
        for l in 0..<40 {
            let kind: LayerKind = (l % 4 == 3) ? .fullAttention : .linearAttention
            layerKinds.append(kind)

            let inNormW = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).input_layernorm.weight"))
            let postNormW = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).post_attention_layernorm.weight"))
            let inNormBuf = device.makeBuffer(bytes: inNormW, length: 2048 * 4, options: .storageModeShared)!
            let postNormBuf = device.makeBuffer(bytes: postNormW, length: 2048 * 4, options: .storageModeShared)!

            if kind == .linearAttention {
                let convW = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).linear_attn.conv1d.weight"))
                let dtB = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).linear_attn.dt_bias"))
                let aL = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).linear_attn.A_log"))
                let linN = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).linear_attn.norm.weight"))

                let cBuf = device.makeBuffer(bytes: convW, length: 8192 * 4 * 4, options: .storageModeShared)!
                let dtBuf = device.makeBuffer(bytes: dtB, length: 32 * 4, options: .storageModeShared)!
                let aLBuf = device.makeBuffer(bytes: aL, length: 32 * 4, options: .storageModeShared)!
                let lnBuf = device.makeBuffer(bytes: linN, length: 128 * 4, options: .storageModeShared)!

                staticConstants.append(StaticLayerConstants(
                    kind: kind, inNormBuf: inNormBuf, postNormBuf: postNormBuf,
                    conv1dWBuf: cBuf, dtBiasBuf: dtBuf, aLogBuf: aLBuf, linNormWBuf: lnBuf,
                    qNormWBuf: nil, kNormWBuf: nil
                ))

                let qkvW = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_qkv.weight")
                let qkvS = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_qkv.scales")
                let qkvB = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_qkv.biases")
                let zW = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_z.weight")
                let zS = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_z.scales")
                let zB = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_z.biases")
                let bW = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_b.weight")
                let bS = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_b.scales")
                let bB = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_b.biases")
                let aW = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_a.weight")
                let aS = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_a.scales")
                let aB = try getRawSlice("language_model.model.layers.\(l).linear_attn.in_proj_a.biases")
                let outW = try getRawSlice("language_model.model.layers.\(l).linear_attn.out_proj.weight")
                let outS = try getRawSlice("language_model.model.layers.\(l).linear_attn.out_proj.scales")
                let outB = try getRawSlice("language_model.model.layers.\(l).linear_attn.out_proj.biases")

                let rtrW = try getRawSlice("language_model.model.layers.\(l).mlp.gate.weight")
                let rtrS = try getRawSlice("language_model.model.layers.\(l).mlp.gate.scales")
                let rtrB = try getRawSlice("language_model.model.layers.\(l).mlp.gate.biases")
                let shCoeffW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.weight")
                let shCoeffS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.scales")
                let shCoeffB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.biases")
                let shGateW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.weight")
                let shGateS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.scales")
                let shGateB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.biases")
                let shUpW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.weight")
                let shUpS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.scales")
                let shUpB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.biases")
                let shDownW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.weight")
                let shDownS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.scales")
                let shDownB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.biases")

                let swGateW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.weight")
                let swGateS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.scales")
                let swGateB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.biases")
                let swUpW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.weight")
                let swUpS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.scales")
                let swUpB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.biases")
                let swDownW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.weight")
                let swDownS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.scales")
                let swDownB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.biases")

                layerPointers.append(LayerDensePointers(
                    kind: kind,
                    qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                    zW: zW, zS: zS, zB: zB,
                    bW: bW, bS: bS, bB: bB,
                    aW: aW, aS: aS, aB: aB,
                    outW: outW, outS: outS, outB: outB,
                    kW: nil, kS: nil, kB: nil, vW: nil, vS: nil, vB: nil,
                    routerW: rtrW, routerS: rtrS, routerB: rtrB,
                    shCoeffW: shCoeffW, shCoeffS: shCoeffS, shCoeffB: shCoeffB,
                    shGateW: shGateW, shGateS: shGateS, shGateB: shGateB,
                    shUpW: shUpW, shUpS: shUpS, shUpB: shUpB,
                    shDownW: shDownW, shDownS: shDownS, shDownB: shDownB,
                    switchGateW: swGateW, switchGateS: swGateS, switchGateB: swGateB,
                    switchUpW: swUpW, switchUpS: swUpS, switchUpB: swUpB,
                    switchDownW: swDownW, switchDownS: swDownS, switchDownB: swDownB
                ))
            } else {
                let qN = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).self_attn.q_norm.weight"))
                let kN = decodeBF16Tensor(try getRawSlice("language_model.model.layers.\(l).self_attn.k_norm.weight"))
                let qNBuf = device.makeBuffer(bytes: qN, length: 256 * 4, options: .storageModeShared)!
                let kNBuf = device.makeBuffer(bytes: kN, length: 256 * 4, options: .storageModeShared)!

                staticConstants.append(StaticLayerConstants(
                    kind: kind, inNormBuf: inNormBuf, postNormBuf: postNormBuf,
                    conv1dWBuf: nil, dtBiasBuf: nil, aLogBuf: nil, linNormWBuf: nil,
                    qNormWBuf: qNBuf, kNormWBuf: kNBuf
                ))

                let qW = try getRawSlice("language_model.model.layers.\(l).self_attn.q_proj.weight")
                let qS = try getRawSlice("language_model.model.layers.\(l).self_attn.q_proj.scales")
                let qB = try getRawSlice("language_model.model.layers.\(l).self_attn.q_proj.biases")
                let kW = try getRawSlice("language_model.model.layers.\(l).self_attn.k_proj.weight")
                let kS = try getRawSlice("language_model.model.layers.\(l).self_attn.k_proj.scales")
                let kB = try getRawSlice("language_model.model.layers.\(l).self_attn.k_proj.biases")
                let vW = try getRawSlice("language_model.model.layers.\(l).self_attn.v_proj.weight")
                let vS = try getRawSlice("language_model.model.layers.\(l).self_attn.v_proj.scales")
                let vB = try getRawSlice("language_model.model.layers.\(l).self_attn.v_proj.biases")
                let outW = try getRawSlice("language_model.model.layers.\(l).self_attn.o_proj.weight")
                let outS = try getRawSlice("language_model.model.layers.\(l).self_attn.o_proj.scales")
                let outB = try getRawSlice("language_model.model.layers.\(l).self_attn.o_proj.biases")

                let rtrW = try getRawSlice("language_model.model.layers.\(l).mlp.gate.weight")
                let rtrS = try getRawSlice("language_model.model.layers.\(l).mlp.gate.scales")
                let rtrB = try getRawSlice("language_model.model.layers.\(l).mlp.gate.biases")
                let shCoeffW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.weight")
                let shCoeffS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.scales")
                let shCoeffB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert_gate.biases")
                let shGateW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.weight")
                let shGateS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.scales")
                let shGateB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.gate_proj.biases")
                let shUpW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.weight")
                let shUpS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.scales")
                let shUpB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.up_proj.biases")
                let shDownW = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.weight")
                let shDownS = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.scales")
                let shDownB = try getRawSlice("language_model.model.layers.\(l).mlp.shared_expert.down_proj.biases")

                let swGateW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.weight")
                let swGateS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.scales")
                let swGateB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.gate_proj.biases")
                let swUpW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.weight")
                let swUpS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.scales")
                let swUpB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.up_proj.biases")
                let swDownW = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.weight")
                let swDownS = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.scales")
                let swDownB = try getRawSlice("language_model.model.layers.\(l).mlp.switch_mlp.down_proj.biases")

                layerPointers.append(LayerDensePointers(
                    kind: kind,
                    qkvW: qW, qkvS: qS, qkvB: qB,
                    zW: nil, zS: nil, zB: nil,
                    bW: nil, bS: nil, bB: nil,
                    aW: nil, aS: nil, aB: nil,
                    outW: outW, outS: outS, outB: outB,
                    kW: kW, kS: kS, kB: kB, vW: vW, vS: vS, vB: vB,
                    routerW: rtrW, routerS: rtrS, routerB: rtrB,
                    shCoeffW: shCoeffW, shCoeffS: shCoeffS, shCoeffB: shCoeffB,
                    shGateW: shGateW, shGateS: shGateS, shGateB: shGateB,
                    shUpW: shUpW, shUpS: shUpS, shUpB: shUpB,
                    shDownW: shDownW, shDownS: shDownS, shDownB: shDownB,
                    switchGateW: swGateW, switchGateS: swGateS, switchGateB: swGateB,
                    switchUpW: swUpW, switchUpS: swUpS, switchUpB: swUpB,
                    switchDownW: swDownW, switchDownS: swDownS, switchDownB: swDownB
                ))
            }
        }
    }

    private func initStates() throws {
        func buffer(_ length: Int) throws -> MTLBuffer {
            guard let value = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw NSError(domain: "QwenFieldfare", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Unable to allocate model state buffer (\(length) bytes)"])
            }
            return value
        }
        convStates.removeAll(); ssmStates.removeAll(); kCaches.removeAll(); vCaches.removeAll()
        for l in 0..<40 {
            if layerKinds[l] == .linearAttention {
                let cBuf = try buffer(8192 * 4 * 4)
                let sBuf = try buffer(32 * 128 * 128 * 4)
                memset(cBuf.contents(), 0, 8192 * 4 * 4)
                memset(sBuf.contents(), 0, 32 * 128 * 128 * 4)
                convStates.append(cBuf)
                ssmStates.append(sBuf)
                kCaches.append(try buffer(4))
                vCaches.append(try buffer(4))
            } else {
                let kvByteLength = 2 * maxSeqLen * 256 * MemoryLayout<Float16>.size
                let kBuf = try buffer(kvByteLength)
                let vBuf = try buffer(kvByteLength)
                memset(kBuf.contents(), 0, kvByteLength)
                memset(vBuf.contents(), 0, kvByteLength)
                kCaches.append(kBuf)
                vCaches.append(vBuf)
                convStates.append(try buffer(4))
                ssmStates.append(try buffer(4))
            }
        }
    }

    public func resetState() {
        for l in 0..<40 {
            if layerKinds[l] == .linearAttention {
                memset(convStates[l].contents(), 0, 8192 * 4 * 4)
                memset(ssmStates[l].contents(), 0, 32 * 128 * 128 * 4)
            } else {
                let kvByteLength = 2 * maxSeqLen * 256 * MemoryLayout<Float16>.size
                memset(kCaches[l].contents(), 0, kvByteLength)
                memset(vCaches[l].contents(), 0, kvByteLength)
            }
        }
        perLayerRecentExperts.removeAll()
    }

    private func stageDenseWeights(forLayer l: Int, targetBuffer: MTLBuffer? = nil) {
        let p = layerPointers[l]
        guard let destinationBuffer = targetBuffer ?? denseArena?.buffer else {
            preconditionFailure("Dense staging requires a resident or reusable arena")
        }
        let dst = destinationBuffer.contents()

        func copySlice(_ slice: RawSlice?, toOffset: Int) {
            guard let sl = slice else { return }
            let src = shards[sl.shardName]!.ptr.advanced(by: sl.fileOffset)
            memcpy(dst.advanced(by: toOffset), src, sl.size)
        }

        copySlice(p.qkvW, toOffset: DenseLayerArena.qkvWOff)
        copySlice(p.qkvS, toOffset: DenseLayerArena.qkvSOff)
        copySlice(p.qkvB, toOffset: DenseLayerArena.qkvBOff)

        if p.kind == .linearAttention {
            copySlice(p.zW, toOffset: DenseLayerArena.zWOff)
            copySlice(p.zS, toOffset: DenseLayerArena.zSOff)
            copySlice(p.zB, toOffset: DenseLayerArena.zBOff)
            copySlice(p.bW, toOffset: DenseLayerArena.bWOff)
            copySlice(p.bS, toOffset: DenseLayerArena.bSOff)
            copySlice(p.bB, toOffset: DenseLayerArena.bBOff)
            copySlice(p.aW, toOffset: DenseLayerArena.aWOff)
            copySlice(p.aS, toOffset: DenseLayerArena.aSOff)
            copySlice(p.aB, toOffset: DenseLayerArena.aBOff)
        } else {
            copySlice(p.kW, toOffset: DenseLayerArena.kProjWOff)
            copySlice(p.kS, toOffset: DenseLayerArena.kProjSOff)
            copySlice(p.kB, toOffset: DenseLayerArena.kProjBOff)
            copySlice(p.vW, toOffset: DenseLayerArena.vProjWOff)
            copySlice(p.vS, toOffset: DenseLayerArena.vProjSOff)
            copySlice(p.vB, toOffset: DenseLayerArena.vProjBOff)
        }

        copySlice(p.outW, toOffset: DenseLayerArena.linOutWOff)
        copySlice(p.outS, toOffset: DenseLayerArena.linOutSOff)
        copySlice(p.outB, toOffset: DenseLayerArena.linOutBOff)

        copySlice(p.routerW, toOffset: DenseLayerArena.routerWOff)
        copySlice(p.routerS, toOffset: DenseLayerArena.routerSOff)
        copySlice(p.routerB, toOffset: DenseLayerArena.routerBOff)

        copySlice(p.shCoeffW, toOffset: DenseLayerArena.shCoeffWOff)
        copySlice(p.shCoeffS, toOffset: DenseLayerArena.shCoeffSOff)
        copySlice(p.shCoeffB, toOffset: DenseLayerArena.shCoeffBOff)

        copySlice(p.shGateW, toOffset: DenseLayerArena.shGateWOff)
        copySlice(p.shGateS, toOffset: DenseLayerArena.shGateSOff)
        copySlice(p.shGateB, toOffset: DenseLayerArena.shGateBOff)

        copySlice(p.shUpW, toOffset: DenseLayerArena.shUpWOff)
        copySlice(p.shUpS, toOffset: DenseLayerArena.shUpSOff)
        copySlice(p.shUpB, toOffset: DenseLayerArena.shUpBOff)

        copySlice(p.shDownW, toOffset: DenseLayerArena.shDownWOff)
        copySlice(p.shDownS, toOffset: DenseLayerArena.shDownSOff)
        copySlice(p.shDownB, toOffset: DenseLayerArena.shDownBOff)
    }

    private static func preadFully(
        fileDescriptor: Int32,
        destination: UnsafeMutableRawPointer,
        count: Int,
        fileOffset: Int
    ) -> Bool {
        var completed = 0
        while completed < count {
            let result = Darwin.pread(
                fileDescriptor,
                destination.advanced(by: completed),
                count - completed,
                off_t(fileOffset + completed))
            if result < 0 && errno == EINTR { continue }
            guard result > 0 else { return false }
            completed += result
        }
        return true
    }

    private func fetchExpertWithPread(
        layer: Int,
        expert: Int,
        destination: UnsafeMutableRawPointer
    ) {
        if packedExpertStore?.readExpert(
            layer: layer, expert: expert, destination: destination) == true {
            return
        }

        let pointers = layerPointers[layer]
        let sendableDestination = SendableMutableRawPointer(value: destination)

        @Sendable func read(_ slice: RawSlice, expertStride: Int, count: Int, destinationOffset: Int) {
            let shard = shards[slice.shardName]!
            let sourceOffset = slice.fileOffset + expert * expertStride
            let target = sendableDestination.value.advanced(by: destinationOffset)
            if !Self.preadFully(
                fileDescriptor: shard.fd,
                destination: target,
                count: count,
                fileOffset: sourceOffset)
            {
                // Preserve correctness if an explicit read is interrupted by
                // an unexpected filesystem condition; telemetry will still
                // classify the request as a cache miss.
                memcpy(target, shard.ptr.advanced(by: sourceOffset), count)
            }
        }

        // Gate, up, and down occupy independent source ranges and destinations.
        // Reading the three projections concurrently removes the nine-read
        // serial tail that dominates layers with only one or two misses.
        DispatchQueue.concurrentPerform(iterations: 3) { projection in
            switch projection {
            case 0:
                read(pointers.switchGateW, expertStride: 524_288, count: 524_288, destinationOffset: 0)
                read(pointers.switchGateS, expertStride: 32_768, count: 32_768, destinationOffset: 524_288)
                read(pointers.switchGateB, expertStride: 32_768, count: 32_768, destinationOffset: 557_056)
            case 1:
                read(pointers.switchUpW, expertStride: 524_288, count: 524_288, destinationOffset: 589_824)
                read(pointers.switchUpS, expertStride: 32_768, count: 32_768, destinationOffset: 1_114_112)
                read(pointers.switchUpB, expertStride: 32_768, count: 32_768, destinationOffset: 1_146_880)
            default:
                read(pointers.switchDownW, expertStride: 524_288, count: 524_288, destinationOffset: 1_179_648)
                read(pointers.switchDownS, expertStride: 32_768, count: 32_768, destinationOffset: 1_703_936)
                read(pointers.switchDownB, expertStride: 32_768, count: 32_768, destinationOffset: 1_736_704)
            }
        }
    }

    public func expertStorageMetrics() -> QwenQ4ExpertStorageMetrics {
        packedExpertStore?.metricsSnapshot()
            ?? QwenQ4ExpertStorageMetrics(
                kind: "raw-safetensors",
                packedReads: 0,
                packedPreadOperations: 0,
                packedReadFailures: 0)
    }

    private func dequantizeEmbedding(tokenId: Int, outBuf: MTLBuffer) {
        let rowU32Offset = tokenId * 256
        let rowGroupOffset = tokenId * 32
        let wShard = shards[embedW.shardName]!.ptr.advanced(by: embedW.fileOffset)
        let sShard = shards[embedS.shardName]!.ptr.advanced(by: embedS.fileOffset)
        let bShard = shards[embedB.shardName]!.ptr.advanced(by: embedB.fileOffset)

        let wPtr = wShard.assumingMemoryBound(to: UInt32.self)
        let sPtr = sShard.assumingMemoryBound(to: UInt16.self)
        let bPtr = bShard.assumingMemoryBound(to: UInt16.self)
        let outPtr = outBuf.contents().assumingMemoryBound(to: Float.self)

        for g in 0..<32 {
            let sF = Float(bitPattern: UInt32(sPtr[rowGroupOffset + g]) << 16)
            let bF = Float(bitPattern: UInt32(bPtr[rowGroupOffset + g]) << 16)
            for j in 0..<8 {
                let u = wPtr[rowU32Offset + g * 8 + j]
                for n in 0..<8 {
                    let nibble = Float((u >> (n * 4)) & 0x0F)
                    outPtr[g * 64 + j * 8 + n] = nibble * sF + bF
                }
            }
        }
    }

    public func forwardToken(
        tokenId: Int,
        tokenPos: Int,
        generationConfiguration: QwenGenerationConfiguration? = nil,
        seenTokenIDs: Set<Int32> = [],
        samplePosition: Int = 0
    ) throws -> Int32 {
        let tStepStart = CFAbsoluteTimeGetCurrent()
dequantizeEmbedding(tokenId: tokenId, outBuf: hiddenBuf0)

        var dim2048: UInt32 = 2048
        var eps: Float = 1e-6

        var totalCpuSetupMs: Double = 0.0
        var totalCbEncodeMs: Double = 0.0
        var totalCbWaitMs: Double = 0.0
        var totalCacheLookupMs: Double = 0.0
        var totalCacheMissFetchMs: Double = 0.0
        let totalDenseStagingMs: Double = 0.0
        var totalRouterMs: Double = 0.0
        var totalExpertStagingMs: Double = 0.0
        var totalAttnMs: Double = 0.0
        let totalDeltanetMs: Double = 0.0
        var totalStagedMoEMs: Double = 0.0
        let totalNormResMs: Double = 0.0
        var lmHeadMs: Double = 0.0
        var totalBytesStaged: Int = 0
        var attentionRouterGPUMs: Double = 0.0
        var profiledSharedHitCommands: [MTLCommandBuffer] = []
        var profiledMoETailCommands: [MTLCommandBuffer] = []

        let tSetupEnd = CFAbsoluteTimeGetCurrent()
        totalCpuSetupMs += (tSetupEnd - tStepStart) * 1000.0

        for l in 0..<40 {
            let curInBuf = (l % 2 == 0) ? hiddenBuf0 : hiddenBuf1
            let curOutBuf = (l % 2 == 0) ? hiddenBuf1 : hiddenBuf0
            let statics = staticConstants[l]
            let curDenseBuf: MTLBuffer = keepDenseWeightsResident
                ? residentDenseArenas[l].buffer
                : denseArena!.buffer

            // COMMAND BUFFER 1: Attention + PostNorm + Router + SharedCoeff
            let tEnc1Start = CFAbsoluteTimeGetCurrent()
            guard let cmdAttnRouter = metalContext.queue.makeCommandBuffer(),
                  let initialEncoder = cmdAttnRouter.makeComputeCommandEncoder() else { fatalError() }
            var enc1 = initialEncoder

            if statics.kind == .linearAttention {
                // RMSNorm
                enc1.setComputePipelineState(rmsnormPipe)
                enc1.setBuffer(curInBuf, offset: 0, index: 0)
                enc1.setBuffer(statics.inNormBuf, offset: 0, index: 1)
                enc1.setBuffer(normInBuf, offset: 0, index: 2)
                enc1.setBytes(&dim2048, length: 4, index: 3)
                enc1.setBytes(&eps, length: 4, index: 4)
                enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // Fused Lin In Proj
                var qkvWOff = UInt32(DenseLayerArena.qkvWOff); var qkvSOff = UInt32(DenseLayerArena.qkvSOff); var qkvBOff = UInt32(DenseLayerArena.qkvBOff)
                var zWOff = UInt32(DenseLayerArena.zWOff); var zSOff = UInt32(DenseLayerArena.zSOff); var zBOff = UInt32(DenseLayerArena.zBOff)
                var bWOff = UInt32(DenseLayerArena.bWOff); var bSOff = UInt32(DenseLayerArena.bSOff); var bBOff = UInt32(DenseLayerArena.bBOff)
                var aWOff = UInt32(DenseLayerArena.aWOff); var aSOff = UInt32(DenseLayerArena.aSOff); var aBOff = UInt32(DenseLayerArena.aBOff)

                enc1.setComputePipelineState(fusedLinInProjSimdPipe)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 3); enc1.setBuffer(curDenseBuf, offset: 0, index: 4); enc1.setBuffer(curDenseBuf, offset: 0, index: 5)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 6); enc1.setBuffer(curDenseBuf, offset: 0, index: 7); enc1.setBuffer(curDenseBuf, offset: 0, index: 8)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 9); enc1.setBuffer(curDenseBuf, offset: 0, index: 10); enc1.setBuffer(curDenseBuf, offset: 0, index: 11)
                enc1.setBuffer(normInBuf, offset: 0, index: 12); enc1.setBuffer(qkvOutBuf, offset: 0, index: 13)
                enc1.setBuffer(zOutBuf, offset: 0, index: 14); enc1.setBuffer(bOutBuf, offset: 0, index: 15); enc1.setBuffer(aOutBuf, offset: 0, index: 16)
                enc1.setBytes(&qkvWOff, length: 4, index: 17); enc1.setBytes(&qkvSOff, length: 4, index: 18); enc1.setBytes(&qkvBOff, length: 4, index: 19)
                enc1.setBytes(&zWOff, length: 4, index: 20); enc1.setBytes(&zSOff, length: 4, index: 21); enc1.setBytes(&zBOff, length: 4, index: 22)
                enc1.setBytes(&bWOff, length: 4, index: 23); enc1.setBytes(&bSOff, length: 4, index: 24); enc1.setBytes(&bBOff, length: 4, index: 25)
                enc1.setBytes(&aWOff, length: 4, index: 26); enc1.setBytes(&aSOff, length: 4, index: 27); enc1.setBytes(&aBOff, length: 4, index: 28)
                enc1.dispatchThreadgroups(MTLSize(width: (12352 + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                // Conv1D
                var convDim: UInt32 = 8192
                enc1.setComputePipelineState(conv1dPipe)
                enc1.setBuffer(qkvOutBuf, offset: 0, index: 0)
                enc1.setBuffer(convStates[l], offset: 0, index: 1)
                enc1.setBuffer(statics.conv1dWBuf!, offset: 0, index: 2)
                enc1.setBuffer(convOutBuf, offset: 0, index: 3)
                enc1.setBytes(&convDim, length: 4, index: 4)
                enc1.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // Fused DeltaNet Step
                enc1.setComputePipelineState(fusedDeltanetStepPipe)
                enc1.setBuffer(convOutBuf, offset: 0, index: 0)
                enc1.setBuffer(zOutBuf, offset: 0, index: 1)
                enc1.setBuffer(aOutBuf, offset: 0, index: 2)
                enc1.setBuffer(bOutBuf, offset: 0, index: 3)
                enc1.setBuffer(statics.aLogBuf!, offset: 0, index: 4)
                enc1.setBuffer(statics.dtBiasBuf!, offset: 0, index: 5)
                enc1.setBuffer(ssmStates[l], offset: 0, index: 6)
                enc1.setBuffer(statics.linNormWBuf!, offset: 0, index: 7)
                enc1.setBuffer(gatedNormBuf, offset: 0, index: 8)
                enc1.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

                // Out Proj
                var linOutRows: UInt32 = 2048; var linOutCols: UInt32 = 4096
                var linOutWOff = UInt32(DenseLayerArena.linOutWOff); var linOutSOff = UInt32(DenseLayerArena.linOutSOff); var linOutBOff = UInt32(DenseLayerArena.linOutBOff)
                enc1.setComputePipelineState(q4SimdPipe)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(gatedNormBuf, offset: 0, index: 3); enc1.setBuffer(attnOutBuf, offset: 0, index: 4)
                enc1.setBytes(&linOutRows, length: 4, index: 5); enc1.setBytes(&linOutCols, length: 4, index: 6)
                enc1.setBytes(&linOutWOff, length: 4, index: 7); enc1.setBytes(&linOutSOff, length: 4, index: 8); enc1.setBytes(&linOutBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 2048 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                // Residual Add -> attnResBuf
                enc1.setComputePipelineState(addPipe)
                enc1.setBuffer(curInBuf, offset: 0, index: 0)
                enc1.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc1.setBuffer(attnResBuf, offset: 0, index: 2)
                enc1.setBytes(&dim2048, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            } else {
                // Full Attention: RMSNorm -> QKV Proj -> Split -> HeadNorm -> RoPE -> Update KV Cache -> CausalAttn -> Out Proj -> Residual Add
                enc1.setComputePipelineState(rmsnormPipe)
                enc1.setBuffer(curInBuf, offset: 0, index: 0)
                enc1.setBuffer(statics.inNormBuf, offset: 0, index: 1)
                enc1.setBuffer(normInBuf, offset: 0, index: 2)
                enc1.setBytes(&dim2048, length: 4, index: 3)
                enc1.setBytes(&eps, length: 4, index: 4)
                enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                var qRows: UInt32 = 8192; var kvRows: UInt32 = 512; var inCols: UInt32 = 2048
                var qWOff = UInt32(DenseLayerArena.qkvWOff); var qSOff = UInt32(DenseLayerArena.qkvSOff); var qBOff = UInt32(DenseLayerArena.qkvBOff)
                var kWOff = UInt32(DenseLayerArena.kProjWOff); var kSOff = UInt32(DenseLayerArena.kProjSOff); var kBOff = UInt32(DenseLayerArena.kProjBOff)
                var vWOff = UInt32(DenseLayerArena.vProjWOff); var vSOff = UInt32(DenseLayerArena.vProjSOff); var vBOff = UInt32(DenseLayerArena.vProjBOff)

                // Q
                enc1.setComputePipelineState(q4SimdPipe)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(qChunkBuf, offset: 0, index: 4)
                enc1.setBytes(&qRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&qWOff, length: 4, index: 7); enc1.setBytes(&qSOff, length: 4, index: 8); enc1.setBytes(&qBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 8192 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // K
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(kBuf, offset: 0, index: 4)
                enc1.setBytes(&kvRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&kWOff, length: 4, index: 7); enc1.setBytes(&kSOff, length: 4, index: 8); enc1.setBytes(&kBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 512 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // V
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(vBuf, offset: 0, index: 4)
                enc1.setBytes(&kvRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&vWOff, length: 4, index: 7); enc1.setBytes(&vSOff, length: 4, index: 8); enc1.setBytes(&vBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 512 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                enc1.setComputePipelineState(splitQGatePipe)
                enc1.setBuffer(qChunkBuf, offset: 0, index: 0); enc1.setBuffer(qBuf, offset: 0, index: 1); enc1.setBuffer(fullGateBuf, offset: 0, index: 2)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var numQHeads: UInt32 = 16; var numKHeads: UInt32 = 2; var headDim: UInt32 = 256
                enc1.setComputePipelineState(headNormPipe)
                enc1.setBuffer(qBuf, offset: 0, index: 0); enc1.setBuffer(statics.qNormWBuf!, offset: 0, index: 1)
                enc1.setBytes(&numQHeads, length: 4, index: 2); enc1.setBytes(&headDim, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc1.setBuffer(kBuf, offset: 0, index: 0); enc1.setBuffer(statics.kNormWBuf!, offset: 0, index: 1)
                enc1.setBytes(&numKHeads, length: 4, index: 2); enc1.setBytes(&headDim, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var cosArr = [Float](repeating: 0, count: 64); var sinArr = [Float](repeating: 0, count: 64)
                let theta: Float = 10000000.0
                for i in 0..<32 {
                    let freq = 1.0 / pow(theta, Float(2 * i) / 64.0)
                    let angle = Float(tokenPos) * freq
                    let c = cos(angle); let s = sin(angle)
                    cosArr[i] = c; cosArr[32 + i] = c
                    sinArr[i] = s; sinArr[32 + i] = s
                }
                memcpy(cosBuf.contents(), cosArr, 64 * 4); memcpy(sinBuf.contents(), sinArr, 64 * 4)

                enc1.setComputePipelineState(ropePipe)
                enc1.setBuffer(qBuf, offset: 0, index: 0); enc1.setBuffer(cosBuf, offset: 0, index: 1); enc1.setBuffer(sinBuf, offset: 0, index: 2)
                enc1.setBytes(&numQHeads, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc1.setBuffer(kBuf, offset: 0, index: 0); enc1.setBuffer(cosBuf, offset: 0, index: 1); enc1.setBuffer(sinBuf, offset: 0, index: 2)
                enc1.setBytes(&numKHeads, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                // Store token-major FP16 KV and convert Q for the shared exact
                // split-K/V attention path.
                var curPosU = UInt32(tokenPos); var maxSeqU = UInt32(maxSeqLen)
                enc1.setComputePipelineState(updateKVCacheHalfPipe)
                enc1.setBuffer(kBuf, offset: 0, index: 0); enc1.setBuffer(vBuf, offset: 0, index: 1)
                enc1.setBuffer(kCaches[l], offset: 0, index: 2); enc1.setBuffer(vCaches[l], offset: 0, index: 3)
                enc1.setBytes(&curPosU, length: 4, index: 4); enc1.setBytes(&maxSeqU, length: 4, index: 5)
                enc1.dispatchThreadgroups(MTLSize(width: 512 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                var qElementCount: UInt32 = 16 * 256
                enc1.setComputePipelineState(floatToHalfPipe)
                enc1.setBuffer(qBuf, offset: 0, index: 0)
                enc1.setBuffer(qHalfBuf, offset: 0, index: 1)
                enc1.setBytes(&qElementCount, length: 4, index: 2)
                enc1.dispatchThreadgroups(
                    MTLSize(width: (Int(qElementCount) + 63) / 64, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                enc1.endEncoding()
                splitAttention.encodeFull(
                    commandBuffer: cmdAttnRouter,
                    q: qHalfBuf,
                    k: kCaches[l],
                    v: vCaches[l],
                    out: attentionHalfOutBuf,
                    headDim: 256,
                    numQHeads: 16,
                    numKVHeads: 2,
                    seqLen: curPosU + 1,
                    scale: 1.0 / 16.0)
                guard let continuationEncoder = cmdAttnRouter.makeComputeCommandEncoder() else { fatalError() }
                enc1 = continuationEncoder

                enc1.setComputePipelineState(applyAttentionGatePipe)
                enc1.setBuffer(attentionHalfOutBuf, offset: 0, index: 0)
                enc1.setBuffer(fullGateBuf, offset: 0, index: 1)
                enc1.setBuffer(attnHeadOutBuf, offset: 0, index: 2)
                enc1.setBytes(&qElementCount, length: 4, index: 3)
                enc1.dispatchThreadgroups(
                    MTLSize(width: (Int(qElementCount) + 63) / 64, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // Full Out Proj
                var fullOutRows: UInt32 = 2048; var fullOutCols: UInt32 = 4096
                var fullOutWOff = UInt32(DenseLayerArena.linOutWOff); var fullOutSOff = UInt32(DenseLayerArena.linOutSOff); var fullOutBOff = UInt32(DenseLayerArena.linOutBOff)
                enc1.setComputePipelineState(q4SimdPipe)
                enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
                enc1.setBuffer(attnHeadOutBuf, offset: 0, index: 3); enc1.setBuffer(attnOutBuf, offset: 0, index: 4)
                enc1.setBytes(&fullOutRows, length: 4, index: 5); enc1.setBytes(&fullOutCols, length: 4, index: 6)
                enc1.setBytes(&fullOutWOff, length: 4, index: 7); enc1.setBytes(&fullOutSOff, length: 4, index: 8); enc1.setBytes(&fullOutBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 2048 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                // Residual Add
                enc1.setComputePipelineState(addPipe)
                enc1.setBuffer(curInBuf, offset: 0, index: 0); enc1.setBuffer(attnOutBuf, offset: 0, index: 1); enc1.setBuffer(attnResBuf, offset: 0, index: 2)
                enc1.setBytes(&dim2048, length: 4, index: 3)
                enc1.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }

            // Post-Attention RMSNorm
            enc1.setComputePipelineState(rmsnormPipe)
            enc1.setBuffer(attnResBuf, offset: 0, index: 0)
            enc1.setBuffer(statics.postNormBuf, offset: 0, index: 1)
            enc1.setBuffer(normMLPBuf, offset: 0, index: 2)
            enc1.setBytes(&dim2048, length: 4, index: 3)
            enc1.setBytes(&eps, length: 4, index: 4)
            enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // SIMD-cooperative router GEMV followed by an in-buffer top-8.
            var rwOff = UInt32(DenseLayerArena.routerWOff); var rsOff = UInt32(DenseLayerArena.routerSOff); var rbOff = UInt32(DenseLayerArena.routerBOff)
            enc1.setComputePipelineState(routerQ8LogitsSimdPipe)
            enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
            enc1.setBuffer(normMLPBuf, offset: 0, index: 3); enc1.setBuffer(routerLogitsBuf, offset: 0, index: 4)
            enc1.setBytes(&rwOff, length: 4, index: 5); enc1.setBytes(&rsOff, length: 4, index: 6); enc1.setBytes(&rbOff, length: 4, index: 7)
            enc1.dispatchThreadgroups(MTLSize(width: 256 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            enc1.setComputePipelineState(routerTop8Pipe)
            enc1.setBuffer(routerLogitsBuf, offset: 0, index: 0)
            enc1.setBuffer(expertIndicesBuf, offset: 0, index: 1)
            enc1.setBuffer(scoresBuf, offset: 0, index: 2)
            enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // Shared Coeff
            var shcRows: UInt32 = 1; var shcCols: UInt32 = 2048
            var shcWOff = UInt32(DenseLayerArena.shCoeffWOff); var shcSOff = UInt32(DenseLayerArena.shCoeffSOff); var shcBOff = UInt32(DenseLayerArena.shCoeffBOff)
            enc1.setComputePipelineState(q8Pipe)
            enc1.setBuffer(curDenseBuf, offset: 0, index: 0); enc1.setBuffer(curDenseBuf, offset: 0, index: 1); enc1.setBuffer(curDenseBuf, offset: 0, index: 2)
            enc1.setBuffer(normMLPBuf, offset: 0, index: 3); enc1.setBuffer(shCoeffBuf, offset: 0, index: 4)
            enc1.setBytes(&shcRows, length: 4, index: 5); enc1.setBytes(&shcCols, length: 4, index: 6)
            enc1.setBytes(&shcWOff, length: 4, index: 7); enc1.setBytes(&shcSOff, length: 4, index: 8); enc1.setBytes(&shcBOff, length: 4, index: 9)
            enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            enc1.endEncoding()
            let tEnc1End = CFAbsoluteTimeGetCurrent()
            totalCbEncodeMs += (tEnc1End - tEnc1Start) * 1000.0

            let tWait1Start = CFAbsoluteTimeGetCurrent()
            cmdAttnRouter.commit()
            cmdAttnRouter.waitUntilCompleted()
            let tWait1End = CFAbsoluteTimeGetCurrent()
            let cb1Wait = (tWait1End - tWait1Start) * 1000.0
            totalCbWaitMs += cb1Wait
            totalAttnMs += cb1Wait * 0.70
            totalRouterMs += cb1Wait * 0.30
            if enableGPUProfiling {
                attentionRouterGPUMs += Self.gpuDurationMs(cmdAttnRouter)
            }

            // CPU slot planning starts the shared and hit-only GPU work before
            // servicing cache misses, so both overlap the explicit reads.
            let expPtr = expertIndicesBuf.contents().assumingMemoryBound(to: UInt32.self)
            let selectedIds = (0..<8).map { expPtr[$0] }
            perLayerRecentExperts[l] = selectedIds

            let tCacheLookupStart = CFAbsoluteTimeGetCurrent()
            let prevBytes = expertCache.bytesFetched
            var hitCount: UInt32 = 0
            var missPositions: [UInt32] = []
            var layerHitCommand: MTLCommandBuffer?
            let activeSlots = expertCache.acquireSlotsParallel(
                layer: l,
                expertIndices: selectedIds,
                afterPlanning: { plannedSlots, plannedMissPositions in
                    memcpy(self.activeSlotIndicesBuf.contents(), plannedSlots, plannedSlots.count * 4)
                    missPositions = plannedMissPositions
                    let misses = Set(plannedMissPositions)
                    let hitPositions = plannedSlots.indices.compactMap {
                        misses.contains(UInt32($0)) ? nil : UInt32($0)
                    }
                    hitCount = UInt32(hitPositions.count)
                    if hitCount > 0 {
                        memcpy(self.hitExpertPositionsBuf.contents(), hitPositions, hitPositions.count * 4)
                    }

                    guard let hitCommand = self.metalContext.queue.makeCommandBuffer(),
                          let hitEncoder = hitCommand.makeComputeCommandEncoder() else { fatalError() }

                    var shGWOff = UInt32(DenseLayerArena.shGateWOff)
                    var shGSOff = UInt32(DenseLayerArena.shGateSOff)
                    var shGBOff = UInt32(DenseLayerArena.shGateBOff)
                    var shUWOff = UInt32(DenseLayerArena.shUpWOff)
                    var shUSOff = UInt32(DenseLayerArena.shUpSOff)
                    var shUBOff = UInt32(DenseLayerArena.shUpBOff)
                    hitEncoder.setComputePipelineState(self.sharedGateUpSiluSimdPipe)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 0)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 1)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 2)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 3)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 4)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 5)
                    hitEncoder.setBuffer(self.normMLPBuf, offset: 0, index: 6)
                    hitEncoder.setBuffer(self.shActBuf, offset: 0, index: 7)
                    hitEncoder.setBytes(&shGWOff, length: 4, index: 8)
                    hitEncoder.setBytes(&shGSOff, length: 4, index: 9)
                    hitEncoder.setBytes(&shGBOff, length: 4, index: 10)
                    hitEncoder.setBytes(&shUWOff, length: 4, index: 11)
                    hitEncoder.setBytes(&shUSOff, length: 4, index: 12)
                    hitEncoder.setBytes(&shUBOff, length: 4, index: 13)
                    hitEncoder.dispatchThreadgroups(
                        MTLSize(width: 512 / 8, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                    var shDWOff = UInt32(DenseLayerArena.shDownWOff)
                    var shDSOff = UInt32(DenseLayerArena.shDownSOff)
                    var shDBOff = UInt32(DenseLayerArena.shDownBOff)
                    hitEncoder.setComputePipelineState(self.sharedDownGateSimdPipe)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 0)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 1)
                    hitEncoder.setBuffer(curDenseBuf, offset: 0, index: 2)
                    hitEncoder.setBuffer(self.shActBuf, offset: 0, index: 3)
                    hitEncoder.setBuffer(self.shCoeffBuf, offset: 0, index: 4)
                    hitEncoder.setBuffer(self.sharedOutBuf, offset: 0, index: 5)
                    hitEncoder.setBytes(&shDWOff, length: 4, index: 6)
                    hitEncoder.setBytes(&shDSOff, length: 4, index: 7)
                    hitEncoder.setBytes(&shDBOff, length: 4, index: 8)
                    hitEncoder.dispatchThreadgroups(
                        MTLSize(width: 2048 / 8, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                    // Hit slots cannot be victims for the pending reads. Queue
                    // their routed work in this same command buffer.
                    if hitCount > 0 {
                    hitEncoder.setComputePipelineState(self.cachedMoeGateUpSubsetSimdPipe)
                    hitEncoder.setBuffer(self.expertCache.cacheBuffer, offset: 0, index: 0)
                    hitEncoder.setBuffer(self.normMLPBuf, offset: 0, index: 1)
                    hitEncoder.setBuffer(self.batchedActBuf, offset: 0, index: 2)
                    hitEncoder.setBuffer(self.activeSlotIndicesBuf, offset: 0, index: 3)
                    hitEncoder.setBuffer(self.hitExpertPositionsBuf, offset: 0, index: 4)
                    var count = hitCount
                    hitEncoder.setBytes(&count, length: 4, index: 5)
                    hitEncoder.dispatchThreadgroups(
                        MTLSize(width: (Int(count) * 512 + 7) / 8, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                    hitEncoder.setComputePipelineState(self.cachedMoeDownSubsetSimdPipe)
                    hitEncoder.setBuffer(self.expertCache.cacheBuffer, offset: 0, index: 0)
                    hitEncoder.setBuffer(self.batchedActBuf, offset: 0, index: 1)
                    hitEncoder.setBuffer(self.scoresBuf, offset: 0, index: 2)
                    hitEncoder.setBuffer(self.routedHitOutBuf, offset: 0, index: 3)
                    hitEncoder.setBuffer(self.activeSlotIndicesBuf, offset: 0, index: 4)
                    hitEncoder.setBuffer(self.hitExpertPositionsBuf, offset: 0, index: 5)
                    hitEncoder.setBytes(&count, length: 4, index: 6)
                    hitEncoder.dispatchThreadgroups(
                        MTLSize(width: 2048, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                    hitEncoder.endEncoding()
                    hitCommand.commit()
                    layerHitCommand = hitCommand
                }) {
                    self.fetchExpertWithPread(layer: $0, expert: $1, destination: $2)
                }
            if enableGPUProfiling, let layerHitCommand {
                profiledSharedHitCommands.append(layerHitCommand)
            }
            let tCacheLookupEnd = CFAbsoluteTimeGetCurrent()
            let acquisitionMs = (tCacheLookupEnd - tCacheLookupStart) * 1_000.0
            let fetchedBytes = expertCache.bytesFetched - prevBytes
            if fetchedBytes == 0 { totalCacheLookupMs += acquisitionMs }
            else { totalCacheMissFetchMs += acquisitionMs }
            totalExpertStagingMs += acquisitionMs
            totalBytesStaged += fetchedBytes
            memcpy(activeSlotIndicesBuf.contents(), activeSlots, 8 * 4)
            let missCount = UInt32(missPositions.count)
            if missCount > 0 {
                memcpy(missExpertPositionsBuf.contents(), missPositions, missPositions.count * 4)
            }

            // COMMAND BUFFER 2: MoE (and on layer 39: Final RMSNorm + LM Head)
            let tEnc2Start = CFAbsoluteTimeGetCurrent()
            guard let cmdMoE = metalContext.queue.makeCommandBuffer(),
                  let enc2 = cmdMoE.makeComputeCommandEncoder() else { fatalError() }

            if missCount > 0 {
                var count = missCount
                enc2.setComputePipelineState(cachedMoeGateUpSubsetSimdPipe)
                enc2.setBuffer(expertCache.cacheBuffer, offset: 0, index: 0)
                enc2.setBuffer(normMLPBuf, offset: 0, index: 1)
                enc2.setBuffer(batchedActBuf, offset: 0, index: 2)
                enc2.setBuffer(activeSlotIndicesBuf, offset: 0, index: 3)
                enc2.setBuffer(missExpertPositionsBuf, offset: 0, index: 4)
                enc2.setBytes(&count, length: 4, index: 5)
                enc2.dispatchThreadgroups(
                    MTLSize(width: (Int(count) * 512 + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                enc2.setComputePipelineState(cachedMoeDownSubsetSimdPipe)
                enc2.setBuffer(expertCache.cacheBuffer, offset: 0, index: 0)
                enc2.setBuffer(batchedActBuf, offset: 0, index: 1)
                enc2.setBuffer(scoresBuf, offset: 0, index: 2)
                enc2.setBuffer(routedMissOutBuf, offset: 0, index: 3)
                enc2.setBuffer(activeSlotIndicesBuf, offset: 0, index: 4)
                enc2.setBuffer(missExpertPositionsBuf, offset: 0, index: 5)
                enc2.setBytes(&count, length: 4, index: 6)
                enc2.dispatchThreadgroups(
                    MTLSize(width: 2048, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            var mutableHitCount = hitCount
            var mutableMissCount = missCount
            enc2.setComputePipelineState(combineMoeSubsetsPipe)
            enc2.setBuffer(routedHitOutBuf, offset: 0, index: 0)
            enc2.setBuffer(routedMissOutBuf, offset: 0, index: 1)
            enc2.setBuffer(sharedOutBuf, offset: 0, index: 2)
            enc2.setBuffer(attnResBuf, offset: 0, index: 3)
            enc2.setBuffer(curOutBuf, offset: 0, index: 4)
            enc2.setBytes(&mutableHitCount, length: 4, index: 5)
            enc2.setBytes(&mutableMissCount, length: 4, index: 6)
            enc2.dispatchThreadgroups(
                MTLSize(width: 2048 / 64, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            if l == 39 {
                // Final RMSNorm
                enc2.setComputePipelineState(rmsnormPipe)
                enc2.setBuffer(curOutBuf, offset: 0, index: 0)
                enc2.setBuffer(finalNormWBuf, offset: 0, index: 1)
                enc2.setBuffer(finalNormOutBuf, offset: 0, index: 2)
                enc2.setBytes(&dim2048, length: 4, index: 3); enc2.setBytes(&eps, length: 4, index: 4)
                enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // LM Head GEMV
                var lmRows: UInt32 = 248320; var lmCols: UInt32 = 2048
                var lmWOff = UInt32(lmHeadWOff); var lmSOff = UInt32(lmHeadSOff); var lmBOff = UInt32(lmHeadBOff)
                enc2.setComputePipelineState(q4SimdPipe)
                enc2.setBuffer(lmHeadWBuf, offset: 0, index: 0); enc2.setBuffer(lmHeadSBuf, offset: 0, index: 1); enc2.setBuffer(lmHeadBBuf, offset: 0, index: 2)
                enc2.setBuffer(finalNormOutBuf, offset: 0, index: 3); enc2.setBuffer(logitsOutBuf, offset: 0, index: 4)
                enc2.setBytes(&lmRows, length: 4, index: 5); enc2.setBytes(&lmCols, length: 4, index: 6)
                enc2.setBytes(&lmWOff, length: 4, index: 7); enc2.setBytes(&lmSOff, length: 4, index: 8); enc2.setBytes(&lmBOff, length: 4, index: 9)
                enc2.dispatchThreadgroups(MTLSize(width: (248320 + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            enc2.endEncoding()
            let tEnc2End = CFAbsoluteTimeGetCurrent()
            totalCbEncodeMs += (tEnc2End - tEnc2Start) * 1000.0

            let tWait2Start = CFAbsoluteTimeGetCurrent()
            cmdMoE.commit()
            if enableGPUProfiling {
                profiledMoETailCommands.append(cmdMoE)
            }
            // The next layer is ordered behind this command buffer on the
            // same queue. Only the final layer needs an immediate CPU wait for
            // logits; earlier MoE work overlaps the host's next-layer encode.
            if l == 39 {
                cmdMoE.waitUntilCompleted()
            }
            let tWait2End = CFAbsoluteTimeGetCurrent()
            let cb2Wait = (tWait2End - tWait2Start) * 1000.0
            totalCbWaitMs += cb2Wait
            if l == 39 {
                totalStagedMoEMs += max(0.0, cb2Wait - 3.0)
                lmHeadMs = 3.0
            }
        }

        let tArgmaxStart = CFAbsoluteTimeGetCurrent()
        let logitsPtr = logitsOutBuf.contents().assumingMemoryBound(to: Float.self)
        let selectedToken: Int32
        if let generationConfiguration {
            selectedToken = QwenLogitSelector.select(
                logits: UnsafePointer(logitsPtr),
                count: 248_320,
                seenTokenIDs: seenTokenIDs,
                configuration: generationConfiguration,
                samplePosition: samplePosition
            )
        } else {
            var maxLogit: Float = 0.0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(logitsPtr, 1, &maxLogit, &maxIdx, 248_320)
            selectedToken = Int32(maxIdx)
        }
        let tArgmaxEnd = CFAbsoluteTimeGetCurrent()
        let cpuArgmaxMs = (tArgmaxEnd - tArgmaxStart) * 1000.0

        let sharedHitGPUMs = enableGPUProfiling
            ? profiledSharedHitCommands.reduce(0.0) { $0 + Self.gpuDurationMs($1) }
            : 0.0
        let moeTailGPUMs = enableGPUProfiling
            ? profiledMoETailCommands.reduce(0.0) { $0 + Self.gpuDurationMs($1) }
            : 0.0
        let totalGPUMs = attentionRouterGPUMs + sharedHitGPUMs + moeTailGPUMs

        let tStepEnd = CFAbsoluteTimeGetCurrent()
        let totalStepMs = (tStepEnd - tStepStart) * 1000.0

        self.latestProfile = StepProfile(
            step: tokenPos,
            tokenId: selectedToken,
            totalStepMs: totalStepMs,
            cpuSetupMs: totalCpuSetupMs,
            cbEncodeMs: totalCbEncodeMs,
            cbCommitWaitMs: totalCbWaitMs,
            cacheLookupMs: totalCacheLookupMs,
            cacheMissFetchMs: totalCacheMissFetchMs,
            cpuArgmaxMs: cpuArgmaxMs,
            denseStagingMs: totalDenseStagingMs,
            routerMs: totalRouterMs,
            expertStagingMs: totalExpertStagingMs,
            attentionMs: totalAttnMs,
            deltanetMs: totalDeltanetMs,
            stagedMoEMs: totalStagedMoEMs,
            normsResidualsMs: totalNormResMs,
            lmHeadMs: lmHeadMs,
            attentionRouterGPUMs: attentionRouterGPUMs,
            sharedHitGPUMs: sharedHitGPUMs,
            moeTailGPUMs: moeTailGPUMs,
            totalGPUMs: totalGPUMs,
            bytesStaged: totalBytesStaged
        )

        return selectedToken
    }

    private static func gpuDurationMs(_ commandBuffer: MTLCommandBuffer) -> Double {
        let start = commandBuffer.gpuStartTime
        let end = commandBuffer.gpuEndTime
        guard start > 0, end >= start else { return 0.0 }
        return (end - start) * 1_000.0
    }

    func debugTopLogits(limit: Int = 5) -> [(tokenID: Int32, logit: Float)] {
        guard limit > 0 else { return [] }
        let logits = logitsOutBuf.contents().assumingMemoryBound(to: Float.self)
        var candidates: [(tokenID: Int32, logit: Float)] = []
        candidates.reserveCapacity(limit)
        for tokenID in 0..<248_320 {
            let candidate = (tokenID: Int32(tokenID), logit: logits[tokenID])
            if candidates.count < limit {
                candidates.append(candidate)
                if candidates.count == limit {
                    candidates.sort { $0.logit > $1.logit }
                }
            } else if candidate.logit > candidates[limit - 1].logit {
                candidates[limit - 1] = candidate
                candidates.sort { $0.logit > $1.logit }
            }
        }
        return candidates
    }

    public func generate(
        promptTokens: [Int32],
        maxNewTokens: Int = 512,
        onPrefillComplete: (@Sendable (Double) -> Void)? = nil,
        onToken: (@Sendable (Int32) -> Void)? = nil
    ) async throws -> [Int32] {
        try await generateResult(
            promptTokens: promptTokens,
            configuration: .greedy(maxNewTokens: maxNewTokens),
            onPrefillComplete: onPrefillComplete,
            onToken: onToken
        ).tokens
    }

    public func generateResult(
        promptTokens: [Int32],
        configuration: QwenGenerationConfiguration = QwenGenerationConfiguration(),
        onPrefillComplete: (@Sendable (Double) -> Void)? = nil,
        onToken: (@Sendable (Int32) -> Void)? = nil
    ) async throws -> QwenGenerationResult {
        try configuration.validate()
        guard !promptTokens.isEmpty else {
            return QwenGenerationResult(tokens: [], finishReason: .maxNewTokens)
        }
        guard promptTokens.count < maxSeqLen else {
            throw QwenGenerationConfigurationError.promptExceedsContext(
                promptTokens: promptTokens.count,
                contextLength: maxSeqLen
            )
        }
        resetState()

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var lastTokenId: Int32 = promptTokens[0]
        for (pos, token) in promptTokens.enumerated() {
            try Task.checkCancellation()
            let isLastPromptToken = pos == promptTokens.count - 1
            let nextId = try forwardToken(
                tokenId: Int(token),
                tokenPos: pos,
                generationConfiguration: isLastPromptToken ? configuration : nil,
                seenTokenIDs: isLastPromptToken ? Set(promptTokens) : [],
                samplePosition: 0
            )
            if isLastPromptToken {
                lastTokenId = nextId
            }
        }
        onPrefillComplete?((CFAbsoluteTimeGetCurrent() - prefillStart) * 1_000.0)

        return try QwenGenerationLoop.run(
            initialToken: lastTokenId,
            promptTokens: promptTokens,
            maxSequenceLength: maxSeqLen,
            configuration: configuration,
            checkCancellation: { try Task.checkCancellation() },
            onToken: { onToken?($0) },
            nextToken: { token, position, seenTokenIDs, samplePosition in
                try self.forwardToken(
                    tokenId: Int(token),
                    tokenPos: position,
                    generationConfiguration: configuration,
                    seenTokenIDs: seenTokenIDs,
                    samplePosition: samplePosition
                )
            }
        )
    }

    public struct SubcomponentProfileReport: Sendable {
        public let linInNormMs: Double
        public let linInProjMs: Double
        public let linConv1dMs: Double
        public let linFusedDeltanetMs: Double
        public let linOutProjMs: Double
        public let linResAddMs: Double
        public let totalLinAttnMs: Double

        public let fullInNormMs: Double
        public let fullQKVProjMs: Double
        public let fullSplitRopeNormMs: Double
        public let fullCausalAttnMs: Double
        public let fullOutProjMs: Double
        public let fullResAddMs: Double
        public let totalFullAttnMs: Double

        public let totalAttentionMs: Double
    }

    public func profileSubcomponents(iterations: Int = 10) -> SubcomponentProfileReport {
        var dim2048: UInt32 = 2048; var eps: Float = 1e-6
        var convDim: UInt32 = 8192
        var linOutRows: UInt32 = 2048; var linOutCols: UInt32 = 4096
        var linOutWOff = UInt32(DenseLayerArena.linOutWOff); var linOutSOff = UInt32(DenseLayerArena.linOutSOff); var linOutBOff = UInt32(DenseLayerArena.linOutBOff)
        var qkvWOff = UInt32(DenseLayerArena.qkvWOff); var qkvSOff = UInt32(DenseLayerArena.qkvSOff); var qkvBOff = UInt32(DenseLayerArena.qkvBOff)
        var zWOff = UInt32(DenseLayerArena.zWOff); var zSOff = UInt32(DenseLayerArena.zSOff); var zBOff = UInt32(DenseLayerArena.zBOff)
        var bWOff = UInt32(DenseLayerArena.bWOff); var bSOff = UInt32(DenseLayerArena.bSOff); var bBOff = UInt32(DenseLayerArena.bBOff)
        var aWOff = UInt32(DenseLayerArena.aWOff); var aSOff = UInt32(DenseLayerArena.aSOff); var aBOff = UInt32(DenseLayerArena.aBOff)

        // 1. Profile 30 Linear Layers
        var linInNormTimes: [Double] = []
        var linInProjTimes: [Double] = []
        var linConv1dTimes: [Double] = []
        var linFusedDeltanetTimes: [Double] = []
        var linOutProjTimes: [Double] = []
        var linResAddTimes: [Double] = []

        let linearIndices = (0..<40).filter { layerKinds[$0] == .linearAttention }
        let fullIndices = (0..<40).filter { layerKinds[$0] == .fullAttention }

        for _ in 0..<iterations {
            // (a) Linear In-Norm
            guard let cmd0 = metalContext.queue.makeCommandBuffer(),
                  let enc0 = cmd0.makeComputeCommandEncoder() else { fatalError() }
            enc0.setComputePipelineState(rmsnormPipe)
            for l in linearIndices {
                enc0.setBuffer(hiddenBuf0, offset: 0, index: 0)
                enc0.setBuffer(staticConstants[l].inNormBuf, offset: 0, index: 1)
                enc0.setBuffer(normInBuf, offset: 0, index: 2)
                enc0.setBytes(&dim2048, length: 4, index: 3)
                enc0.setBytes(&eps, length: 4, index: 4)
                enc0.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc0.endEncoding(); cmd0.commit(); cmd0.waitUntilCompleted()
            linInNormTimes.append(Self.gpuDurationMs(cmd0))

            // (b) Linear In-Proj (12352 rows x 2048 cols)
            guard let cmd1 = metalContext.queue.makeCommandBuffer(),
                  let enc1 = cmd1.makeComputeCommandEncoder() else { fatalError() }
            enc1.setComputePipelineState(fusedLinInProjSimdPipe)
            for l in linearIndices {
                let dBuf = keepDenseWeightsResident ? residentDenseArenas[l].buffer : denseArena!.buffer
                enc1.setBuffer(dBuf, offset: 0, index: 0); enc1.setBuffer(dBuf, offset: 0, index: 1); enc1.setBuffer(dBuf, offset: 0, index: 2)
                enc1.setBuffer(dBuf, offset: 0, index: 3); enc1.setBuffer(dBuf, offset: 0, index: 4); enc1.setBuffer(dBuf, offset: 0, index: 5)
                enc1.setBuffer(dBuf, offset: 0, index: 6); enc1.setBuffer(dBuf, offset: 0, index: 7); enc1.setBuffer(dBuf, offset: 0, index: 8)
                enc1.setBuffer(dBuf, offset: 0, index: 9); enc1.setBuffer(dBuf, offset: 0, index: 10); enc1.setBuffer(dBuf, offset: 0, index: 11)
                enc1.setBuffer(normInBuf, offset: 0, index: 12); enc1.setBuffer(qkvOutBuf, offset: 0, index: 13)
                enc1.setBuffer(zOutBuf, offset: 0, index: 14); enc1.setBuffer(bOutBuf, offset: 0, index: 15); enc1.setBuffer(aOutBuf, offset: 0, index: 16)
                enc1.setBytes(&qkvWOff, length: 4, index: 17); enc1.setBytes(&qkvSOff, length: 4, index: 18); enc1.setBytes(&qkvBOff, length: 4, index: 19)
                enc1.setBytes(&zWOff, length: 4, index: 20); enc1.setBytes(&zSOff, length: 4, index: 21); enc1.setBytes(&zBOff, length: 4, index: 22)
                enc1.setBytes(&bWOff, length: 4, index: 23); enc1.setBytes(&bSOff, length: 4, index: 24); enc1.setBytes(&bBOff, length: 4, index: 25)
                enc1.setBytes(&aWOff, length: 4, index: 26); enc1.setBytes(&aSOff, length: 4, index: 27); enc1.setBytes(&aBOff, length: 4, index: 28)
                enc1.dispatchThreadgroups(MTLSize(width: (12352 + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc1.endEncoding(); cmd1.commit(); cmd1.waitUntilCompleted()
            linInProjTimes.append(Self.gpuDurationMs(cmd1))

            // (c) Linear Conv1D (8192 dims)
            guard let cmd2 = metalContext.queue.makeCommandBuffer(),
                  let enc2 = cmd2.makeComputeCommandEncoder() else { fatalError() }
            enc2.setComputePipelineState(conv1dPipe)
            for l in linearIndices {
                enc2.setBuffer(qkvOutBuf, offset: 0, index: 0)
                enc2.setBuffer(convStates[l], offset: 0, index: 1)
                enc2.setBuffer(staticConstants[l].conv1dWBuf!, offset: 0, index: 2)
                enc2.setBuffer(convOutBuf, offset: 0, index: 3)
                enc2.setBytes(&convDim, length: 4, index: 4)
                enc2.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc2.endEncoding(); cmd2.commit(); cmd2.waitUntilCompleted()
            linConv1dTimes.append(Self.gpuDurationMs(cmd2))

            // (d) Fused DeltaNet Step (32 heads x 128 rows)
            guard let cmd3 = metalContext.queue.makeCommandBuffer(),
                  let enc3 = cmd3.makeComputeCommandEncoder() else { fatalError() }
            enc3.setComputePipelineState(fusedDeltanetStepPipe)
            for l in linearIndices {
                enc3.setBuffer(convOutBuf, offset: 0, index: 0)
                enc3.setBuffer(zOutBuf, offset: 0, index: 1)
                enc3.setBuffer(aOutBuf, offset: 0, index: 2)
                enc3.setBuffer(bOutBuf, offset: 0, index: 3)
                enc3.setBuffer(staticConstants[l].aLogBuf!, offset: 0, index: 4)
                enc3.setBuffer(staticConstants[l].dtBiasBuf!, offset: 0, index: 5)
                enc3.setBuffer(ssmStates[l], offset: 0, index: 6)
                enc3.setBuffer(staticConstants[l].linNormWBuf!, offset: 0, index: 7)
                enc3.setBuffer(gatedNormBuf, offset: 0, index: 8)
                enc3.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            enc3.endEncoding(); cmd3.commit(); cmd3.waitUntilCompleted()
            linFusedDeltanetTimes.append(Self.gpuDurationMs(cmd3))

            // (e) Linear Out-Proj (2048 rows x 4096 cols)
            guard let cmd4 = metalContext.queue.makeCommandBuffer(),
                  let enc4 = cmd4.makeComputeCommandEncoder() else { fatalError() }
            enc4.setComputePipelineState(q4SimdPipe)
            for l in linearIndices {
                let dBuf = keepDenseWeightsResident ? residentDenseArenas[l].buffer : denseArena!.buffer
                enc4.setBuffer(dBuf, offset: 0, index: 0); enc4.setBuffer(dBuf, offset: 0, index: 1); enc4.setBuffer(dBuf, offset: 0, index: 2)
                enc4.setBuffer(gatedNormBuf, offset: 0, index: 3); enc4.setBuffer(attnOutBuf, offset: 0, index: 4)
                enc4.setBytes(&linOutRows, length: 4, index: 5); enc4.setBytes(&linOutCols, length: 4, index: 6)
                enc4.setBytes(&linOutWOff, length: 4, index: 7); enc4.setBytes(&linOutSOff, length: 4, index: 8); enc4.setBytes(&linOutBOff, length: 4, index: 9)
                enc4.dispatchThreadgroups(MTLSize(width: 2048 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc4.endEncoding(); cmd4.commit(); cmd4.waitUntilCompleted()
            linOutProjTimes.append(Self.gpuDurationMs(cmd4))

            // (f) Linear Residual Add
            guard let cmd5 = metalContext.queue.makeCommandBuffer(),
                  let enc5 = cmd5.makeComputeCommandEncoder() else { fatalError() }
            enc5.setComputePipelineState(addPipe)
            for _ in linearIndices {
                enc5.setBuffer(hiddenBuf0, offset: 0, index: 0)
                enc5.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc5.setBuffer(attnResBuf, offset: 0, index: 2)
                enc5.setBytes(&dim2048, length: 4, index: 3)
                enc5.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc5.endEncoding(); cmd5.commit(); cmd5.waitUntilCompleted()
            linResAddTimes.append(Self.gpuDurationMs(cmd5))
        }

        // 2. Profile 10 Full Attention Layers
        var fullInNormTimes: [Double] = []
        var fullQKVProjTimes: [Double] = []
        var fullSplitRopeNormTimes: [Double] = []
        var fullCausalAttnTimes: [Double] = []
        var fullOutProjTimes: [Double] = []
        var fullResAddTimes: [Double] = []

        var qRows: UInt32 = 8192; var kvRows: UInt32 = 512; var inCols: UInt32 = 2048
        var qWOff = UInt32(DenseLayerArena.qkvWOff); var qSOff = UInt32(DenseLayerArena.qkvSOff); var qBOff = UInt32(DenseLayerArena.qkvBOff)
        var kWOff = UInt32(DenseLayerArena.kProjWOff); var kSOff = UInt32(DenseLayerArena.kProjSOff); var kBOff = UInt32(DenseLayerArena.kProjBOff)
        var vWOff = UInt32(DenseLayerArena.vProjWOff); var vSOff = UInt32(DenseLayerArena.vProjSOff); var vBOff = UInt32(DenseLayerArena.vProjBOff)
        var numQHeads: UInt32 = 16; var numKVHeads: UInt32 = 2; var headDim: UInt32 = 256
        var currPosU32: UInt32 = 4; var maxSeqU32: UInt32 = UInt32(maxSeqLen)
        var fullOutRows: UInt32 = 2048; var fullOutCols: UInt32 = 4096
        var fullOutWOff = UInt32(DenseLayerArena.linOutWOff); var fullOutSOff = UInt32(DenseLayerArena.linOutSOff); var fullOutBOff = UInt32(DenseLayerArena.linOutBOff)

        for _ in 0..<iterations {
            // (a) Full In-Norm
            guard let cmd0 = metalContext.queue.makeCommandBuffer(),
                  let enc0 = cmd0.makeComputeCommandEncoder() else { fatalError() }
            enc0.setComputePipelineState(rmsnormPipe)
            for l in fullIndices {
                enc0.setBuffer(hiddenBuf0, offset: 0, index: 0)
                enc0.setBuffer(staticConstants[l].inNormBuf, offset: 0, index: 1)
                enc0.setBuffer(normInBuf, offset: 0, index: 2)
                enc0.setBytes(&dim2048, length: 4, index: 3)
                enc0.setBytes(&eps, length: 4, index: 4)
                enc0.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc0.endEncoding(); cmd0.commit(); cmd0.waitUntilCompleted()
            fullInNormTimes.append(Self.gpuDurationMs(cmd0))

            // (b) Full Q/K/V Proj
            guard let cmd1 = metalContext.queue.makeCommandBuffer(),
                  let enc1 = cmd1.makeComputeCommandEncoder() else { fatalError() }
            enc1.setComputePipelineState(q4SimdPipe)
            for l in fullIndices {
                let dBuf = keepDenseWeightsResident ? residentDenseArenas[l].buffer : denseArena!.buffer
                // Q
                enc1.setBuffer(dBuf, offset: 0, index: 0); enc1.setBuffer(dBuf, offset: 0, index: 1); enc1.setBuffer(dBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(qChunkBuf, offset: 0, index: 4)
                enc1.setBytes(&qRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&qWOff, length: 4, index: 7); enc1.setBytes(&qSOff, length: 4, index: 8); enc1.setBytes(&qBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 8192 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // K
                enc1.setBuffer(dBuf, offset: 0, index: 0); enc1.setBuffer(dBuf, offset: 0, index: 1); enc1.setBuffer(dBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(kBuf, offset: 0, index: 4)
                enc1.setBytes(&kvRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&kWOff, length: 4, index: 7); enc1.setBytes(&kSOff, length: 4, index: 8); enc1.setBytes(&kBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 512 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // V
                enc1.setBuffer(dBuf, offset: 0, index: 0); enc1.setBuffer(dBuf, offset: 0, index: 1); enc1.setBuffer(dBuf, offset: 0, index: 2)
                enc1.setBuffer(normInBuf, offset: 0, index: 3); enc1.setBuffer(vBuf, offset: 0, index: 4)
                enc1.setBytes(&kvRows, length: 4, index: 5); enc1.setBytes(&inCols, length: 4, index: 6)
                enc1.setBytes(&vWOff, length: 4, index: 7); enc1.setBytes(&vSOff, length: 4, index: 8); enc1.setBytes(&vBOff, length: 4, index: 9)
                enc1.dispatchThreadgroups(MTLSize(width: 512 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc1.endEncoding(); cmd1.commit(); cmd1.waitUntilCompleted()
            fullQKVProjTimes.append(Self.gpuDurationMs(cmd1))

            // (c) Full Split / RoPE / Norm
            guard let cmd2 = metalContext.queue.makeCommandBuffer(),
                  let enc2 = cmd2.makeComputeCommandEncoder() else { fatalError() }
            for l in fullIndices {
                enc2.setComputePipelineState(splitQGatePipe)
                enc2.setBuffer(qChunkBuf, offset: 0, index: 0); enc2.setBuffer(qBuf, offset: 0, index: 1); enc2.setBuffer(fullGateBuf, offset: 0, index: 2)
                enc2.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc2.setComputePipelineState(headNormPipe)
                enc2.setBuffer(qBuf, offset: 0, index: 0); enc2.setBuffer(staticConstants[l].qNormWBuf!, offset: 0, index: 1)
                enc2.setBytes(&numQHeads, length: 4, index: 2); enc2.setBytes(&headDim, length: 4, index: 3)
                enc2.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc2.setBuffer(kBuf, offset: 0, index: 0); enc2.setBuffer(staticConstants[l].kNormWBuf!, offset: 0, index: 1)
                enc2.setBytes(&numKVHeads, length: 4, index: 2); enc2.setBytes(&headDim, length: 4, index: 3)
                enc2.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc2.setComputePipelineState(ropePipe)
                enc2.setBuffer(qBuf, offset: 0, index: 0); enc2.setBuffer(cosBuf, offset: 0, index: 1); enc2.setBuffer(sinBuf, offset: 0, index: 2)
                enc2.setBytes(&numQHeads, length: 4, index: 3)
                enc2.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                enc2.setBuffer(kBuf, offset: 0, index: 0); enc2.setBuffer(cosBuf, offset: 0, index: 1); enc2.setBuffer(sinBuf, offset: 0, index: 2)
                enc2.setBytes(&numKVHeads, length: 4, index: 3)
                enc2.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            enc2.endEncoding(); cmd2.commit(); cmd2.waitUntilCompleted()
            fullSplitRopeNormTimes.append(Self.gpuDurationMs(cmd2))

            // (d) Full Causal Softmax Attention Step
            var causalGPUTimeMs = 0.0
            for l in fullIndices {
                guard let cmd3 = metalContext.queue.makeCommandBuffer(),
                      let prep = cmd3.makeComputeCommandEncoder() else { fatalError() }
                prep.setComputePipelineState(updateKVCacheHalfPipe)
                prep.setBuffer(kBuf, offset: 0, index: 0); prep.setBuffer(vBuf, offset: 0, index: 1)
                prep.setBuffer(kCaches[l], offset: 0, index: 2); prep.setBuffer(vCaches[l], offset: 0, index: 3)
                prep.setBytes(&currPosU32, length: 4, index: 4); prep.setBytes(&maxSeqU32, length: 4, index: 5)
                prep.dispatchThreadgroups(MTLSize(width: 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                var qCount: UInt32 = 4096
                prep.setComputePipelineState(floatToHalfPipe)
                prep.setBuffer(qBuf, offset: 0, index: 0); prep.setBuffer(qHalfBuf, offset: 0, index: 1)
                prep.setBytes(&qCount, length: 4, index: 2)
                prep.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                prep.endEncoding()

                splitAttention.encodeFull(
                    commandBuffer: cmd3, q: qHalfBuf, k: kCaches[l], v: vCaches[l],
                    out: attentionHalfOutBuf, headDim: 256, numQHeads: 16,
                    numKVHeads: 2, seqLen: currPosU32 + 1, scale: 1.0 / 16.0)
                guard let gate = cmd3.makeComputeCommandEncoder() else { fatalError() }
                gate.setComputePipelineState(applyAttentionGatePipe)
                gate.setBuffer(attentionHalfOutBuf, offset: 0, index: 0)
                gate.setBuffer(fullGateBuf, offset: 0, index: 1)
                gate.setBuffer(attnHeadOutBuf, offset: 0, index: 2)
                gate.setBytes(&qCount, length: 4, index: 3)
                gate.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                gate.endEncoding(); cmd3.commit(); cmd3.waitUntilCompleted()
                causalGPUTimeMs += Self.gpuDurationMs(cmd3)
            }
            fullCausalAttnTimes.append(causalGPUTimeMs)

            // (e) Full Out-Proj (2048 rows x 4096 cols)
            guard let cmd4 = metalContext.queue.makeCommandBuffer(),
                  let enc4 = cmd4.makeComputeCommandEncoder() else { fatalError() }
            enc4.setComputePipelineState(q4SimdPipe)
            for l in fullIndices {
                let dBuf = keepDenseWeightsResident ? residentDenseArenas[l].buffer : denseArena!.buffer
                enc4.setBuffer(dBuf, offset: 0, index: 0); enc4.setBuffer(dBuf, offset: 0, index: 1); enc4.setBuffer(dBuf, offset: 0, index: 2)
                enc4.setBuffer(attnHeadOutBuf, offset: 0, index: 3); enc4.setBuffer(attnOutBuf, offset: 0, index: 4)
                enc4.setBytes(&fullOutRows, length: 4, index: 5); enc4.setBytes(&fullOutCols, length: 4, index: 6)
                enc4.setBytes(&fullOutWOff, length: 4, index: 7); enc4.setBytes(&fullOutSOff, length: 4, index: 8); enc4.setBytes(&fullOutBOff, length: 4, index: 9)
                enc4.dispatchThreadgroups(MTLSize(width: 2048 / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc4.endEncoding(); cmd4.commit(); cmd4.waitUntilCompleted()
            fullOutProjTimes.append(Self.gpuDurationMs(cmd4))

            // (f) Full Residual Add
            guard let cmd5 = metalContext.queue.makeCommandBuffer(),
                  let enc5 = cmd5.makeComputeCommandEncoder() else { fatalError() }
            enc5.setComputePipelineState(addPipe)
            for _ in fullIndices {
                enc5.setBuffer(hiddenBuf0, offset: 0, index: 0)
                enc5.setBuffer(attnOutBuf, offset: 0, index: 1)
                enc5.setBuffer(attnResBuf, offset: 0, index: 2)
                enc5.setBytes(&dim2048, length: 4, index: 3)
                enc5.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc5.endEncoding(); cmd5.commit(); cmd5.waitUntilCompleted()
            fullResAddTimes.append(Self.gpuDurationMs(cmd5))
        }

        func avg(_ arr: [Double]) -> Double { arr.reduce(0, +) / Double(arr.count) }

        let linInNormAvg = avg(linInNormTimes)
        let linInProjAvg = avg(linInProjTimes)
        let linConv1dAvg = avg(linConv1dTimes)
        let linFusedDeltanetAvg = avg(linFusedDeltanetTimes)
        let linOutProjAvg = avg(linOutProjTimes)
        let linResAddAvg = avg(linResAddTimes)
        let totalLin = linInNormAvg + linInProjAvg + linConv1dAvg + linFusedDeltanetAvg + linOutProjAvg + linResAddAvg

        let fullInNormAvg = avg(fullInNormTimes)
        let fullQKVProjAvg = avg(fullQKVProjTimes)
        let fullSplitRopeNormAvg = avg(fullSplitRopeNormTimes)
        let fullCausalAttnAvg = avg(fullCausalAttnTimes)
        let fullOutProjAvg = avg(fullOutProjTimes)
        let fullResAddAvg = avg(fullResAddTimes)
        let totalFull = fullInNormAvg + fullQKVProjAvg + fullSplitRopeNormAvg + fullCausalAttnAvg + fullOutProjAvg + fullResAddAvg

        return SubcomponentProfileReport(
            linInNormMs: linInNormAvg,
            linInProjMs: linInProjAvg,
            linConv1dMs: linConv1dAvg,
            linFusedDeltanetMs: linFusedDeltanetAvg,
            linOutProjMs: linOutProjAvg,
            linResAddMs: linResAddAvg,
            totalLinAttnMs: totalLin,
            fullInNormMs: fullInNormAvg,
            fullQKVProjMs: fullQKVProjAvg,
            fullSplitRopeNormMs: fullSplitRopeNormAvg,
            fullCausalAttnMs: fullCausalAttnAvg,
            fullOutProjMs: fullOutProjAvg,
            fullResAddMs: fullResAddAvg,
            totalFullAttnMs: totalFull,
            totalAttentionMs: totalLin + totalFull
        )
    }
}
