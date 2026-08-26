import Foundation
import Metal
import Tokenizers

public struct QwenTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    public static func load(from folder: URL) async throws -> QwenTokenizer {
        let tok = try await AutoTokenizer.from(modelFolder: folder)
        return QwenTokenizer(tokenizer: tok)
    }

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    public func decode(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map(Int.init))
    }
}

public enum QwenQ4RunnerError: Error, CustomStringConvertible {
    case failedToOpenShard(String)
    case failedToMmapShard(String)
    case invalidIndexJSON(String)
    case missingTensor(String)
    case metalBufferAllocationFailed(String)
    case pipelineCompilationFailed(String)
    case memoryBudgetExceeded(String)

    public var description: String {
        switch self {
        case .failedToOpenShard(let s): return "Failed to open shard: \(s)"
        case .failedToMmapShard(let s): return "Failed to mmap shard: \(s)"
        case .invalidIndexJSON(let s): return "Invalid index JSON: \(s)"
        case .missingTensor(let s): return "Missing tensor in model: \(s)"
        case .metalBufferAllocationFailed(let s): return "Metal buffer allocation failed: \(s)"
        case .pipelineCompilationFailed(let s): return "Pipeline compilation failed: \(s)"
        case .memoryBudgetExceeded(let s): return "Memory budget exceeded: \(s)"
        }
    }
}

public final class QwenQ4ChatRunner: @unchecked Sendable {
    public enum LayerKind {
        case linearAttention
        case fullAttention
    }

    public enum RuntimeMode: String, Sendable, CaseIterable {
        case activeStreaming = "Active Expert Streaming — Fast Low Memory"
        case fastResident = "Fast Resident Q4"
        case memorySaver = "Memory Saver Streamed Q4"
    }

    private struct MmapShard {
        let fd: Int32
        let ptr: UnsafeRawPointer
        let size: Int
        let headerLen: Int
        let headerJson: [String: Any]
    }

    private struct TensorSlice {
        let shardName: String
        let fileOffset: Int
        let size: Int
        let shape: [Int]
        let dtype: String
    }

    private struct QTriplet {
        let wBuf: MTLBuffer
        let wOff: Int
        let sBuf: MTLBuffer
        let sOff: Int
        let bBuf: MTLBuffer
        let bOff: Int
    }

    private struct QuantizedLayerWeights {
        let kind: LayerKind
        let inputNormWeightBuf: MTLBuffer
        let postNormWeightBuf: MTLBuffer
        // Linear
        let qkv: QTriplet?
        let z: QTriplet?
        let b: QTriplet?
        let a: QTriplet?
        let conv1dWBuf: MTLBuffer?
        let dtBiasBuf: MTLBuffer?
        let aLogBuf: MTLBuffer?
        let linNormWBuf: MTLBuffer?
        let linOutProj: QTriplet?
        // Full
        let qProj: QTriplet?
        let kProj: QTriplet?
        let vProj: QTriplet?
        let qNormWBuf: MTLBuffer?
        let kNormWBuf: MTLBuffer?
        let fullOutProj: QTriplet?
        // MoE
        let router: QTriplet
        let switchGate: QTriplet
        let switchUp: QTriplet
        let switchDown: QTriplet
        let sharedGate: QTriplet
        let sharedUp: QTriplet
        let sharedDown: QTriplet
        let sharedCoeff: QTriplet
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

    private final class LayerWeightCache: @unchecked Sendable {
        let capacity: Int
        private var cache: [Int: QuantizedLayerWeights] = [:]
        private var accessOrder: [Int] = []
        var creations: Int = 0
        var evictions: Int = 0
        var hits: Int = 0
        var misses: Int = 0

        init(capacity: Int) {
            self.capacity = max(1, capacity)
        }

        func get(layerIndex: Int, loader: (Int) throws -> QuantizedLayerWeights) throws -> QuantizedLayerWeights {
            if let existing = cache[layerIndex] {
                hits += 1
                if let idx = accessOrder.firstIndex(of: layerIndex) {
                    accessOrder.remove(at: idx)
                    accessOrder.append(layerIndex)
                }
                return existing
            }

            misses += 1
            creations += 1

            while cache.count >= capacity && !accessOrder.isEmpty {
                let oldest = accessOrder.removeFirst()
                cache.removeValue(forKey: oldest)
                evictions += 1
            }

            let newWeights = try loader(layerIndex)
            cache[layerIndex] = newWeights
            accessOrder.append(layerIndex)
            return newWeights
        }

        func clear() {
            cache.removeAll()
            accessOrder.removeAll()
        }
    }

    public let mode: RuntimeMode
    public var memoryBudgetGiB: Double = 5.0
    public var hardLimitGiB: Double = 6.0
    public var streamedLayerResidency: Bool { mode == .memorySaver }
    public let layerCacheCapacity: Int
    private let layerCache: LayerWeightCache

    public var cacheCreations: Int { layerCache.creations }
    public var cacheEvictions: Int { layerCache.evictions }
    public var cacheHits: Int { layerCache.hits }
    public var cacheMisses: Int { layerCache.misses }

    private let metalContext: MetalContext
    private let device: MTLDevice
    private var shards: [String: MmapShard] = [:]
    private var weightMap: [String: String] = [:]

    private var embedTriplet: QTriplet!
    private var lmHeadTriplet: QTriplet!
    private var finalNormWBuf: MTLBuffer!
    private var staticLayerConstants: [StaticLayerConstants] = []
    public private(set) var layerKinds: [LayerKind] = []

    // State buffers
    private var convStates: [MTLBuffer] = []
    private var ssmStates: [MTLBuffer] = []
    private var kCaches: [MTLBuffer] = []
    private var vCaches: [MTLBuffer] = []
    public let maxSeqLen: Int

    // Scratch buffers
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
    private let cosBuf: MTLBuffer
    private let sinBuf: MTLBuffer
    private let attnOutBuf: MTLBuffer
    private let attnResBuf: MTLBuffer
    private let normMLPBuf: MTLBuffer
    private let batchedActBuf: MTLBuffer
    private let shActBuf: MTLBuffer
    private let expertIndicesBuf: MTLBuffer
    private let scoresBuf: MTLBuffer
    private let attentionScoresBuf: MTLBuffer
    private let shCoeffBuf: MTLBuffer
    private let finalNormOutBuf: MTLBuffer
    private let logitsOutBuf: MTLBuffer

    // Metal pipelines
    private let q4Pipe: MTLComputePipelineState
    private let q8Pipe: MTLComputePipelineState
    private let rmsnormPipe: MTLComputePipelineState
    private let conv1dPipe: MTLComputePipelineState
    private let prepQKVPipe: MTLComputePipelineState
    private let deltanetPipe: MTLComputePipelineState
    private let gatedNormPipe: MTLComputePipelineState
    private let addPipe: MTLComputePipelineState
    private let splitQGatePipe: MTLComputePipelineState
    private let headNormPipe: MTLComputePipelineState
    private let ropePipe: MTLComputePipelineState
    private let causalAttnPipe: MTLComputePipelineState
    private let fusedRouterPipe: MTLComputePipelineState
    private let fusedLinInProjPipe: MTLComputePipelineState
    private let fusedMoeGateUpSiluPipe: MTLComputePipelineState
    private let fusedMoeDownAccumResidualPipe: MTLComputePipelineState

    public convenience init(modelPath: String, maxSeqLen: Int = 2048, mode: RuntimeMode = .memorySaver) throws {
        let cap = (mode == .fastResident) ? 40 : 4
        let budget = (mode == .fastResident) ? 20.0 : 5.0
        let hardLim = (mode == .fastResident) ? 24.0 : 6.0
        try self.init(modelPath: modelPath, maxSeqLen: maxSeqLen, mode: mode, memoryBudgetGiB: budget, hardLimitGiB: hardLim, layerCacheCapacity: cap)
    }

    public init(modelPath: String, maxSeqLen: Int = 2048, mode: RuntimeMode = .memorySaver, memoryBudgetGiB: Double = 5.0, hardLimitGiB: Double = 6.0, layerCacheCapacity: Int = 4) throws {
        self.mode = mode
        self.maxSeqLen = maxSeqLen
        self.memoryBudgetGiB = memoryBudgetGiB
        self.hardLimitGiB = hardLimitGiB
        self.layerCacheCapacity = layerCacheCapacity
        self.layerCache = LayerWeightCache(capacity: layerCacheCapacity)
        self.metalContext = try MetalContext()
        self.device = metalContext.device

        // Scratch buffers allocation (only ~10 MB total)
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
              let cosB = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
              let sinB = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
              let aob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let arb = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let nmb = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let bab = device.makeBuffer(length: 8 * 512 * 4, options: .storageModeShared),
              let sab = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let eib = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let scb = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let attsb = device.makeBuffer(length: 16 * maxSeqLen * 4, options: .storageModeShared),
              let shcb = device.makeBuffer(length: 1 * 4, options: .storageModeShared),
              let fnob = device.makeBuffer(length: 2048 * 4, options: .storageModeShared),
              let lob = device.makeBuffer(length: 248320 * 4, options: .storageModeShared) else {
            throw QwenQ4RunnerError.metalBufferAllocationFailed("Failed to allocate scratch buffers")
        }

        self.hiddenBuf0 = h0
        self.hiddenBuf1 = h1
        self.normInBuf = ni
        self.qkvOutBuf = qkvo
        self.zOutBuf = zo
        self.bOutBuf = bo
        self.aOutBuf = ao
        self.convOutBuf = co
        self.qBuf = qb
        self.kBuf = kb
        self.vBuf = vb
        self.coreAttnBuf = cab
        self.gatedNormBuf = gnb
        self.qChunkBuf = qcb
        self.fullGateBuf = fgb
        self.attnHeadOutBuf = ahob
        self.cosBuf = cosB
        self.sinBuf = sinB
        self.attnOutBuf = aob
        self.attnResBuf = arb
        self.normMLPBuf = nmb
        self.batchedActBuf = bab
        self.shActBuf = sab
        self.expertIndicesBuf = eib
        self.scoresBuf = scb
        self.attentionScoresBuf = attsb
        self.shCoeffBuf = shcb
        self.finalNormOutBuf = fnob
        self.logitsOutBuf = lob

        // Compile Metal Pipelines
        self.q4Pipe = try metalContext.pipeline("qwen_gemv_q4_affine")
        self.q8Pipe = try metalContext.pipeline("qwen_gemv_q8_affine")
        self.rmsnormPipe = try metalContext.pipeline("qwen_rmsnorm_f32")
        self.conv1dPipe = try metalContext.pipeline("qwen_conv1d_f32")
        self.prepQKVPipe = try metalContext.pipeline("qwen_prep_qkv_f32")
        self.deltanetPipe = try metalContext.pipeline("qwen_deltanet_recurrent_f32")
        self.gatedNormPipe = try metalContext.pipeline("qwen_gated_rmsnorm_f32")
        self.addPipe = try metalContext.pipeline("qwen_residual_add_f32")
        self.splitQGatePipe = try metalContext.pipeline("qwen_split_q_gate_f32")
        self.headNormPipe = try metalContext.pipeline("qwen_head_rmsnorm_f32")
        self.ropePipe = try metalContext.pipeline("qwen_rope_partial_f32")
        self.causalAttnPipe = try metalContext.pipeline("qwen_causal_attn_step_scratch_f32")
        self.fusedRouterPipe = try metalContext.pipeline("qwen_fused_router_top8_q8")
        self.fusedLinInProjPipe = try metalContext.pipeline("qwen_fused_linear_attn_in_proj_q4")
        self.fusedMoeGateUpSiluPipe = try metalContext.pipeline("qwen_fused_moe_gate_up_silu_q4")
        self.fusedMoeDownAccumResidualPipe = try metalContext.pipeline("qwen_fused_moe_down_accum_residual_q4")

        // Load shards and index
        try loadSafetensors(modelPath: modelPath)
        initStates()

        // If Fast Resident mode, pre-populate all 40 layers into cache upfront
        if mode == .fastResident {
            for l in 0..<40 {
                _ = try layerCache.get(layerIndex: l, loader: { try self.loadLayerWeights(layerIndex: $0) })
            }
        }
    }

    deinit {
        for (_, s) in shards {
            munmap(UnsafeMutableRawPointer(mutating: s.ptr), s.size)
            close(s.fd)
        }
    }

    private func getTensorSlice(_ name: String) throws -> TensorSlice {
        guard let shardName = weightMap[name],
              let shard = shards[shardName],
              let dict = shard.headerJson[name] as? [String: Any],
              let offs = dict["data_offsets"] as? [Int],
              let shape = dict["shape"] as? [Int],
              let dtype = dict["dtype"] as? String else {
            throw QwenQ4RunnerError.missingTensor("Missing tensor \(name)")
        }
        let fileOffset = 8 + shard.headerLen + offs[0]
        let size = offs[1] - offs[0]
        return TensorSlice(shardName: shardName, fileOffset: fileOffset, size: size, shape: shape, dtype: dtype)
    }

    private func makeZeroCopyBuffer(slice: TensorSlice) throws -> (buffer: MTLBuffer, offset: Int) {
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

    private func decodeBF16Tensor(_ slice: TensorSlice) -> [Float] {
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

    private func getQTriplet(prefix: String) throws -> QTriplet {
        let wSlice = try getTensorSlice("\(prefix).weight")
        let sSlice = try getTensorSlice("\(prefix).scales")
        let bSlice = try getTensorSlice("\(prefix).biases")
        let (wB, wO) = try makeZeroCopyBuffer(slice: wSlice)
        let (sB, sO) = try makeZeroCopyBuffer(slice: sSlice)
        let (bB, bO) = try makeZeroCopyBuffer(slice: bSlice)
        return QTriplet(wBuf: wB, wOff: wO, sBuf: sB, sOff: sO, bBuf: bB, bOff: bO)
    }

    private func loadSafetensors(modelPath: String) throws {
        let indexPath = "\(modelPath)/model.safetensors.index.json"
        let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        guard let indexJson = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let wm = indexJson["weight_map"] as? [String: String] else {
            throw QwenQ4RunnerError.invalidIndexJSON("Failed to parse weight_map in index JSON")
        }
        self.weightMap = wm

        let shardNames = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"]
        for sName in shardNames {
            let sPath = "\(modelPath)/\(sName)"
            let sFd = open(sPath, O_RDONLY)
            guard sFd >= 0 else {
                throw QwenQ4RunnerError.failedToOpenShard("Failed to open \(sPath)")
            }
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

        // Embed tokens, LM head, and final norm
        self.embedTriplet = try getQTriplet(prefix: "language_model.model.embed_tokens")
        self.lmHeadTriplet = try getQTriplet(prefix: "language_model.lm_head")
        let finalNormWeight = decodeBF16Tensor(try getTensorSlice("language_model.model.norm.weight"))
        guard let fNormBuf = device.makeBuffer(bytes: finalNormWeight, length: 2048 * 4, options: .storageModeShared) else {
            throw QwenQ4RunnerError.metalBufferAllocationFailed("Failed to create finalNorm buffer")
        }
        self.finalNormWBuf = fNormBuf

        // Initialize static lightweight constants for all 40 layers
        for l in 0..<40 {
            let kind: LayerKind = (l % 4 == 3) ? .fullAttention : .linearAttention
            layerKinds.append(kind)

            let inNormW = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).input_layernorm.weight"))
            let postNormW = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).post_attention_layernorm.weight"))
            let inNormBuf = device.makeBuffer(bytes: inNormW, length: 2048 * 4, options: .storageModeShared)!
            let postNormBuf = device.makeBuffer(bytes: postNormW, length: 2048 * 4, options: .storageModeShared)!

            if kind == .linearAttention {
                let convW = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).linear_attn.conv1d.weight"))
                let dtB = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).linear_attn.dt_bias"))
                let aL = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).linear_attn.A_log"))
                let linN = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).linear_attn.norm.weight"))

                let cBuf = device.makeBuffer(bytes: convW, length: 8192 * 4 * 4, options: .storageModeShared)!
                let dtBuf = device.makeBuffer(bytes: dtB, length: 32 * 4, options: .storageModeShared)!
                let aLBuf = device.makeBuffer(bytes: aL, length: 32 * 4, options: .storageModeShared)!
                let lnBuf = device.makeBuffer(bytes: linN, length: 128 * 4, options: .storageModeShared)!

                staticLayerConstants.append(StaticLayerConstants(
                    kind: kind, inNormBuf: inNormBuf, postNormBuf: postNormBuf,
                    conv1dWBuf: cBuf, dtBiasBuf: dtBuf, aLogBuf: aLBuf, linNormWBuf: lnBuf,
                    qNormWBuf: nil, kNormWBuf: nil
                ))
            } else {
                let qN = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).self_attn.q_norm.weight"))
                let kN = decodeBF16Tensor(try getTensorSlice("language_model.model.layers.\(l).self_attn.k_norm.weight"))

                let qNBuf = device.makeBuffer(bytes: qN, length: 256 * 4, options: .storageModeShared)!
                let kNBuf = device.makeBuffer(bytes: kN, length: 256 * 4, options: .storageModeShared)!

                staticLayerConstants.append(StaticLayerConstants(
                    kind: kind, inNormBuf: inNormBuf, postNormBuf: postNormBuf,
                    conv1dWBuf: nil, dtBiasBuf: nil, aLogBuf: nil, linNormWBuf: nil,
                    qNormWBuf: qNBuf, kNormWBuf: kNBuf
                ))
            }
        }
    }

    private func loadLayerWeights(layerIndex l: Int) throws -> QuantizedLayerWeights {
        let kind = layerKinds[l]
        let statics = staticLayerConstants[l]

        let router = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.gate")
        let swGate = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.switch_mlp.gate_proj")
        let swUp = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.switch_mlp.up_proj")
        let swDown = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.switch_mlp.down_proj")
        let shGate = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.shared_expert.gate_proj")
        let shUp = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.shared_expert.up_proj")
        let shDown = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.shared_expert.down_proj")
        let shCoeff = try getQTriplet(prefix: "language_model.model.layers.\(l).mlp.shared_expert_gate")

        if kind == .linearAttention {
            let qkv = try getQTriplet(prefix: "language_model.model.layers.\(l).linear_attn.in_proj_qkv")
            let z = try getQTriplet(prefix: "language_model.model.layers.\(l).linear_attn.in_proj_z")
            let b = try getQTriplet(prefix: "language_model.model.layers.\(l).linear_attn.in_proj_b")
            let a = try getQTriplet(prefix: "language_model.model.layers.\(l).linear_attn.in_proj_a")
            let linOut = try getQTriplet(prefix: "language_model.model.layers.\(l).linear_attn.out_proj")

            return QuantizedLayerWeights(
                kind: kind, inputNormWeightBuf: statics.inNormBuf, postNormWeightBuf: statics.postNormBuf,
                qkv: qkv, z: z, b: b, a: a, conv1dWBuf: statics.conv1dWBuf, dtBiasBuf: statics.dtBiasBuf, aLogBuf: statics.aLogBuf, linNormWBuf: statics.linNormWBuf, linOutProj: linOut,
                qProj: nil, kProj: nil, vProj: nil, qNormWBuf: nil, kNormWBuf: nil, fullOutProj: nil,
                router: router, switchGate: swGate, switchUp: swUp, switchDown: swDown, sharedGate: shGate, sharedUp: shUp, sharedDown: shDown, sharedCoeff: shCoeff
            )
        } else {
            let qProj = try getQTriplet(prefix: "language_model.model.layers.\(l).self_attn.q_proj")
            let kProj = try getQTriplet(prefix: "language_model.model.layers.\(l).self_attn.k_proj")
            let vProj = try getQTriplet(prefix: "language_model.model.layers.\(l).self_attn.v_proj")
            let fullOut = try getQTriplet(prefix: "language_model.model.layers.\(l).self_attn.o_proj")

            return QuantizedLayerWeights(
                kind: kind, inputNormWeightBuf: statics.inNormBuf, postNormWeightBuf: statics.postNormBuf,
                qkv: nil, z: nil, b: nil, a: nil, conv1dWBuf: nil, dtBiasBuf: nil, aLogBuf: nil, linNormWBuf: nil, linOutProj: nil,
                qProj: qProj, kProj: kProj, vProj: vProj, qNormWBuf: statics.qNormWBuf, kNormWBuf: statics.kNormWBuf, fullOutProj: fullOut,
                router: router, switchGate: swGate, switchUp: swUp, switchDown: swDown, sharedGate: shGate, sharedUp: shUp, sharedDown: shDown, sharedCoeff: shCoeff
            )
        }
    }

    private func initStates() {
        convStates.removeAll()
        ssmStates.removeAll()
        kCaches.removeAll()
        vCaches.removeAll()

        for l in 0..<40 {
            if layerKinds[l] == .linearAttention {
                let cBuf = device.makeBuffer(length: 8192 * 4 * 4, options: .storageModeShared)!
                let sBuf = device.makeBuffer(length: 32 * 128 * 128 * 4, options: .storageModeShared)!
                memset(cBuf.contents(), 0, 8192 * 4 * 4)
                memset(sBuf.contents(), 0, 32 * 128 * 128 * 4)
                convStates.append(cBuf)
                ssmStates.append(sBuf)
                kCaches.append(device.makeBuffer(length: 4, options: .storageModeShared)!)
                vCaches.append(device.makeBuffer(length: 4, options: .storageModeShared)!)
            } else {
                let kBuf = device.makeBuffer(length: 2 * maxSeqLen * 256 * 4, options: .storageModeShared)!
                let vBuf = device.makeBuffer(length: 2 * maxSeqLen * 256 * 4, options: .storageModeShared)!
                memset(kBuf.contents(), 0, 2 * maxSeqLen * 256 * 4)
                memset(vBuf.contents(), 0, 2 * maxSeqLen * 256 * 4)
                kCaches.append(kBuf)
                vCaches.append(vBuf)
                convStates.append(device.makeBuffer(length: 4, options: .storageModeShared)!)
                ssmStates.append(device.makeBuffer(length: 4, options: .storageModeShared)!)
            }
        }
    }

    public func resetState() {
        for l in 0..<40 {
            if layerKinds[l] == .linearAttention {
                memset(convStates[l].contents(), 0, 8192 * 4 * 4)
                memset(ssmStates[l].contents(), 0, 32 * 128 * 128 * 4)
            } else {
                memset(kCaches[l].contents(), 0, 2 * maxSeqLen * 256 * 4)
                memset(vCaches[l].contents(), 0, 2 * maxSeqLen * 256 * 4)
            }
        }
    }

    private func encodeQ4GEMV(encoder: MTLComputeCommandEncoder, triplet: QTriplet, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
        var r = rows
        var c = cols
        var wOff = UInt32(triplet.wOff)
        var sOff = UInt32(triplet.sOff)
        var bOff = UInt32(triplet.bOff)
        encoder.setComputePipelineState(q4Pipe)
        encoder.setBuffer(triplet.wBuf, offset: 0, index: 0)
        encoder.setBuffer(triplet.sBuf, offset: 0, index: 1)
        encoder.setBuffer(triplet.bBuf, offset: 0, index: 2)
        encoder.setBuffer(vec, offset: 0, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBytes(&r, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&wOff, length: 4, index: 7)
        encoder.setBytes(&sOff, length: 4, index: 8)
        encoder.setBytes(&bOff, length: 4, index: 9)
        let tgSize = min(Int(rows), 64)
        let numTg = (Int(rows) + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    private func encodeQ8GEMV(encoder: MTLComputeCommandEncoder, triplet: QTriplet, vec: MTLBuffer, out: MTLBuffer, rows: UInt32, cols: UInt32) {
        var r = rows
        var c = cols
        var wOff = UInt32(triplet.wOff)
        var sOff = UInt32(triplet.sOff)
        var bOff = UInt32(triplet.bOff)
        encoder.setComputePipelineState(q8Pipe)
        encoder.setBuffer(triplet.wBuf, offset: 0, index: 0)
        encoder.setBuffer(triplet.sBuf, offset: 0, index: 1)
        encoder.setBuffer(triplet.bBuf, offset: 0, index: 2)
        encoder.setBuffer(vec, offset: 0, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBytes(&r, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&wOff, length: 4, index: 7)
        encoder.setBytes(&sOff, length: 4, index: 8)
        encoder.setBytes(&bOff, length: 4, index: 9)
        let tgSize = min(Int(rows), 64)
        let numTg = (Int(rows) + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(MTLSize(width: numTg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    private func dequantizeEmbedding(tokenId: Int, outBuf: MTLBuffer) {
        let rowU32Offset = tokenId * 256
        let rowGroupOffset = tokenId * 32
        let wPtr = embedTriplet.wBuf.contents().advanced(by: embedTriplet.wOff).assumingMemoryBound(to: UInt32.self)
        let sPtr = embedTriplet.sBuf.contents().advanced(by: embedTriplet.sOff).assumingMemoryBound(to: UInt16.self)
        let bPtr = embedTriplet.bBuf.contents().advanced(by: embedTriplet.bOff).assumingMemoryBound(to: UInt16.self)
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
        dequantizeEmbedding(tokenId: tokenId, outBuf: hiddenBuf0)

        var dim2048: UInt32 = 2048
        var eps: Float = 1e-6

        for l in 0..<40 {
            let curInBuf = (l % 2 == 0) ? hiddenBuf0 : hiddenBuf1
            let curOutBuf = (l % 2 == 0) ? hiddenBuf1 : hiddenBuf0

            // Bounded Layer-Weight Cache Lookup
            let layer = try layerCache.get(layerIndex: l, loader: { try self.loadLayerWeights(layerIndex: $0) })

            if layer.kind == .linearAttention {
                guard let cmdLin = metalContext.queue.makeCommandBuffer(),
                      let encLin = cmdLin.makeComputeCommandEncoder() else { fatalError() }

                // 1. Input RMSNorm
                encLin.setComputePipelineState(rmsnormPipe)
                encLin.setBuffer(curInBuf, offset: 0, index: 0)
                encLin.setBuffer(layer.inputNormWeightBuf, offset: 0, index: 1)
                encLin.setBuffer(normInBuf, offset: 0, index: 2)
                encLin.setBytes(&dim2048, length: 4, index: 3)
                encLin.setBytes(&eps, length: 4, index: 4)
                encLin.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // 2. Fused Input Projections
                var qkvWOff = UInt32(layer.qkv!.wOff)
                var qkvSOff = UInt32(layer.qkv!.sOff)
                var qkvBOff = UInt32(layer.qkv!.bOff)
                var zWOff = UInt32(layer.z!.wOff)
                var zSOff = UInt32(layer.z!.sOff)
                var zBOff = UInt32(layer.z!.bOff)
                var bWOff = UInt32(layer.b!.wOff)
                var bSOff = UInt32(layer.b!.sOff)
                var bBOff = UInt32(layer.b!.bOff)
                var aWOff = UInt32(layer.a!.wOff)
                var aSOff = UInt32(layer.a!.sOff)
                var aBOff = UInt32(layer.a!.bOff)

                encLin.setComputePipelineState(fusedLinInProjPipe)
                encLin.setBuffer(layer.qkv!.wBuf, offset: 0, index: 0)
                encLin.setBuffer(layer.qkv!.sBuf, offset: 0, index: 1)
                encLin.setBuffer(layer.qkv!.bBuf, offset: 0, index: 2)
                encLin.setBuffer(layer.z!.wBuf, offset: 0, index: 3)
                encLin.setBuffer(layer.z!.sBuf, offset: 0, index: 4)
                encLin.setBuffer(layer.z!.bBuf, offset: 0, index: 5)
                encLin.setBuffer(layer.b!.wBuf, offset: 0, index: 6)
                encLin.setBuffer(layer.b!.sBuf, offset: 0, index: 7)
                encLin.setBuffer(layer.b!.bBuf, offset: 0, index: 8)
                encLin.setBuffer(layer.a!.wBuf, offset: 0, index: 9)
                encLin.setBuffer(layer.a!.sBuf, offset: 0, index: 10)
                encLin.setBuffer(layer.a!.bBuf, offset: 0, index: 11)
                encLin.setBuffer(normInBuf, offset: 0, index: 12)
                encLin.setBuffer(qkvOutBuf, offset: 0, index: 13)
                encLin.setBuffer(zOutBuf, offset: 0, index: 14)
                encLin.setBuffer(bOutBuf, offset: 0, index: 15)
                encLin.setBuffer(aOutBuf, offset: 0, index: 16)
                encLin.setBytes(&qkvWOff, length: 4, index: 17)
                encLin.setBytes(&qkvSOff, length: 4, index: 18)
                encLin.setBytes(&qkvBOff, length: 4, index: 19)
                encLin.setBytes(&zWOff, length: 4, index: 20)
                encLin.setBytes(&zSOff, length: 4, index: 21)
                encLin.setBytes(&zBOff, length: 4, index: 22)
                encLin.setBytes(&bWOff, length: 4, index: 23)
                encLin.setBytes(&bSOff, length: 4, index: 24)
                encLin.setBytes(&bBOff, length: 4, index: 25)
                encLin.setBytes(&aWOff, length: 4, index: 26)
                encLin.setBytes(&aSOff, length: 4, index: 27)
                encLin.setBytes(&aBOff, length: 4, index: 28)
                encLin.dispatchThreadgroups(MTLSize(width: (12352 + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                // 3. DeltaNet Recurrence
                var convDim: UInt32 = 8192
                encLin.setComputePipelineState(conv1dPipe)
                encLin.setBuffer(qkvOutBuf, offset: 0, index: 0)
                encLin.setBuffer(convStates[l], offset: 0, index: 1)
                encLin.setBuffer(layer.conv1dWBuf!, offset: 0, index: 2)
                encLin.setBuffer(convOutBuf, offset: 0, index: 3)
                encLin.setBytes(&convDim, length: 4, index: 4)
                encLin.dispatchThreadgroups(MTLSize(width: 8192 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encLin.setComputePipelineState(prepQKVPipe)
                encLin.setBuffer(convOutBuf, offset: 0, index: 0)
                encLin.setBuffer(qBuf, offset: 0, index: 1)
                encLin.setBuffer(kBuf, offset: 0, index: 2)
                encLin.setBuffer(vBuf, offset: 0, index: 3)
                encLin.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encLin.setComputePipelineState(deltanetPipe)
                encLin.setBuffer(qBuf, offset: 0, index: 0)
                encLin.setBuffer(kBuf, offset: 0, index: 1)
                encLin.setBuffer(vBuf, offset: 0, index: 2)
                encLin.setBuffer(aOutBuf, offset: 0, index: 3)
                encLin.setBuffer(bOutBuf, offset: 0, index: 4)
                encLin.setBuffer(layer.aLogBuf!, offset: 0, index: 5)
                encLin.setBuffer(layer.dtBiasBuf!, offset: 0, index: 6)
                encLin.setBuffer(ssmStates[l], offset: 0, index: 7)
                encLin.setBuffer(coreAttnBuf, offset: 0, index: 8)
                encLin.dispatchThreadgroups(MTLSize(width: 128 / 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                encLin.setComputePipelineState(gatedNormPipe)
                encLin.setBuffer(coreAttnBuf, offset: 0, index: 0)
                encLin.setBuffer(layer.linNormWBuf!, offset: 0, index: 1)
                encLin.setBuffer(zOutBuf, offset: 0, index: 2)
                encLin.setBuffer(gatedNormBuf, offset: 0, index: 3)
                encLin.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encodeQ4GEMV(encoder: encLin, triplet: layer.linOutProj!, vec: gatedNormBuf, out: attnOutBuf, rows: 2048, cols: 4096)

                encLin.setComputePipelineState(addPipe)
                encLin.setBuffer(curInBuf, offset: 0, index: 0)
                encLin.setBuffer(attnOutBuf, offset: 0, index: 1)
                encLin.setBuffer(attnResBuf, offset: 0, index: 2)
                encLin.setBytes(&dim2048, length: 4, index: 3)
                encLin.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encLin.endEncoding()
                cmdLin.commit()
                cmdLin.waitUntilCompleted()
            } else {
                // Full Attention
                guard let cmdFull = metalContext.queue.makeCommandBuffer(),
                      let encFull = cmdFull.makeComputeCommandEncoder() else { fatalError() }

                // 1. Input RMSNorm
                encFull.setComputePipelineState(rmsnormPipe)
                encFull.setBuffer(curInBuf, offset: 0, index: 0)
                encFull.setBuffer(layer.inputNormWeightBuf, offset: 0, index: 1)
                encFull.setBuffer(normInBuf, offset: 0, index: 2)
                encFull.setBytes(&dim2048, length: 4, index: 3)
                encFull.setBytes(&eps, length: 4, index: 4)
                encFull.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encodeQ4GEMV(encoder: encFull, triplet: layer.qProj!, vec: normInBuf, out: qChunkBuf, rows: 8192, cols: 2048)
                encodeQ4GEMV(encoder: encFull, triplet: layer.kProj!, vec: normInBuf, out: kBuf, rows: 512, cols: 2048)
                encodeQ4GEMV(encoder: encFull, triplet: layer.vProj!, vec: normInBuf, out: vBuf, rows: 512, cols: 2048)

                encFull.setComputePipelineState(splitQGatePipe)
                encFull.setBuffer(qChunkBuf, offset: 0, index: 0)
                encFull.setBuffer(qBuf, offset: 0, index: 1)
                encFull.setBuffer(fullGateBuf, offset: 0, index: 2)
                encFull.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var numQHeads: UInt32 = 16
                var numKHeads: UInt32 = 2
                var headDim: UInt32 = 256
                encFull.setComputePipelineState(headNormPipe)
                encFull.setBuffer(qBuf, offset: 0, index: 0)
                encFull.setBuffer(layer.qNormWBuf!, offset: 0, index: 1)
                encFull.setBytes(&numQHeads, length: 4, index: 2)
                encFull.setBytes(&headDim, length: 4, index: 3)
                encFull.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encFull.setComputePipelineState(headNormPipe)
                encFull.setBuffer(kBuf, offset: 0, index: 0)
                encFull.setBuffer(layer.kNormWBuf!, offset: 0, index: 1)
                encFull.setBytes(&numKHeads, length: 4, index: 2)
                encFull.setBytes(&headDim, length: 4, index: 3)
                encFull.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                var cosArr = [Float](repeating: 0, count: 64)
                var sinArr = [Float](repeating: 0, count: 64)
                let theta: Float = 10000000.0
                for i in 0..<32 {
                    let freq = 1.0 / pow(theta, Float(2 * i) / 64.0)
                    let angle = Float(tokenPos) * freq
                    let c = cos(angle)
                    let s = sin(angle)
                    cosArr[i] = c; cosArr[32 + i] = c
                    sinArr[i] = s; sinArr[32 + i] = s
                }
                memcpy(cosBuf.contents(), cosArr, 64 * 4)
                memcpy(sinBuf.contents(), sinArr, 64 * 4)

                encFull.setComputePipelineState(ropePipe)
                encFull.setBuffer(qBuf, offset: 0, index: 0)
                encFull.setBuffer(cosBuf, offset: 0, index: 1)
                encFull.setBuffer(sinBuf, offset: 0, index: 2)
                encFull.setBytes(&numQHeads, length: 4, index: 3)
                encFull.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encFull.setComputePipelineState(ropePipe)
                encFull.setBuffer(kBuf, offset: 0, index: 0)
                encFull.setBuffer(cosBuf, offset: 0, index: 1)
                encFull.setBuffer(sinBuf, offset: 0, index: 2)
                encFull.setBytes(&numKHeads, length: 4, index: 3)
                encFull.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encFull.endEncoding()
                cmdFull.commit()
                cmdFull.waitUntilCompleted()

                for kh in 0..<2 {
                    let kOffset = kh * maxSeqLen * 256 + tokenPos * 256
                    memcpy(kCaches[l].contents().advanced(by: kOffset * 4), kBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                    let vOffset = kh * maxSeqLen * 256 + tokenPos * 256
                    memcpy(vCaches[l].contents().advanced(by: vOffset * 4), vBuf.contents().advanced(by: kh * 256 * 4), 256 * 4)
                }

                guard let cmdCausal = metalContext.queue.makeCommandBuffer(),
                      let encCausal = cmdCausal.makeComputeCommandEncoder() else { fatalError() }
                var curPosU = UInt32(tokenPos)
                var maxSeqU = UInt32(maxSeqLen)
                encCausal.setComputePipelineState(causalAttnPipe)
                encCausal.setBuffer(qBuf, offset: 0, index: 0)
                encCausal.setBuffer(kCaches[l], offset: 0, index: 1)
                encCausal.setBuffer(vCaches[l], offset: 0, index: 2)
                encCausal.setBuffer(fullGateBuf, offset: 0, index: 3)
                encCausal.setBuffer(attnHeadOutBuf, offset: 0, index: 4)
                encCausal.setBytes(&curPosU, length: 4, index: 5)
                encCausal.setBytes(&maxSeqU, length: 4, index: 6)
                encCausal.setBuffer(attentionScoresBuf, offset: 0, index: 7)
                encCausal.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

                encodeQ4GEMV(encoder: encCausal, triplet: layer.fullOutProj!, vec: attnHeadOutBuf, out: attnOutBuf, rows: 2048, cols: 4096)

                encCausal.setComputePipelineState(addPipe)
                encCausal.setBuffer(curInBuf, offset: 0, index: 0)
                encCausal.setBuffer(attnOutBuf, offset: 0, index: 1)
                encCausal.setBuffer(attnResBuf, offset: 0, index: 2)
                encCausal.setBytes(&dim2048, length: 4, index: 3)
                encCausal.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

                encCausal.endEncoding()
                cmdCausal.commit()
                cmdCausal.waitUntilCompleted()
            }

            // MoE Stage
            guard let cmdMoE = metalContext.queue.makeCommandBuffer(),
                  let encMoE = cmdMoE.makeComputeCommandEncoder() else { fatalError() }

            // 1. Post-attention RMSNorm
            encMoE.setComputePipelineState(rmsnormPipe)
            encMoE.setBuffer(attnResBuf, offset: 0, index: 0)
            encMoE.setBuffer(layer.postNormWeightBuf, offset: 0, index: 1)
            encMoE.setBuffer(normMLPBuf, offset: 0, index: 2)
            encMoE.setBytes(&dim2048, length: 4, index: 3)
            encMoE.setBytes(&eps, length: 4, index: 4)
            encMoE.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // 2. Fused Router
            var rwOff = UInt32(layer.router.wOff)
            var rsOff = UInt32(layer.router.sOff)
            var rbOff = UInt32(layer.router.bOff)
            encMoE.setComputePipelineState(fusedRouterPipe)
            encMoE.setBuffer(layer.router.wBuf, offset: 0, index: 0)
            encMoE.setBuffer(layer.router.sBuf, offset: 0, index: 1)
            encMoE.setBuffer(layer.router.bBuf, offset: 0, index: 2)
            encMoE.setBuffer(normMLPBuf, offset: 0, index: 3)
            encMoE.setBuffer(expertIndicesBuf, offset: 0, index: 4)
            encMoE.setBuffer(scoresBuf, offset: 0, index: 5)
            encMoE.setBytes(&rwOff, length: 4, index: 6)
            encMoE.setBytes(&rsOff, length: 4, index: 7)
            encMoE.setBytes(&rbOff, length: 4, index: 8)
            encMoE.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // 3. Shared Expert Coefficient
            encodeQ8GEMV(encoder: encMoE, triplet: layer.sharedCoeff, vec: normMLPBuf, out: shCoeffBuf, rows: 1, cols: 2048)

            // 4. Fused Gate-Up-SiLU
            var switchGWOff = UInt32(layer.switchGate.wOff)
            var switchGSOff = UInt32(layer.switchGate.sOff)
            var switchGBOff = UInt32(layer.switchGate.bOff)
            var switchUWOff = UInt32(layer.switchUp.wOff)
            var switchUSOff = UInt32(layer.switchUp.sOff)
            var switchUBOff = UInt32(layer.switchUp.bOff)
            var sharedGWOff = UInt32(layer.sharedGate.wOff)
            var sharedGSOff = UInt32(layer.sharedGate.sOff)
            var sharedGBOff = UInt32(layer.sharedGate.bOff)
            var sharedUWOff = UInt32(layer.sharedUp.wOff)
            var sharedUSOff = UInt32(layer.sharedUp.sOff)
            var sharedUBOff = UInt32(layer.sharedUp.bOff)

            encMoE.setComputePipelineState(fusedMoeGateUpSiluPipe)
            encMoE.setBuffer(layer.switchGate.wBuf, offset: 0, index: 0)
            encMoE.setBuffer(layer.switchGate.sBuf, offset: 0, index: 1)
            encMoE.setBuffer(layer.switchGate.bBuf, offset: 0, index: 2)
            encMoE.setBuffer(layer.switchUp.wBuf, offset: 0, index: 3)
            encMoE.setBuffer(layer.switchUp.sBuf, offset: 0, index: 4)
            encMoE.setBuffer(layer.switchUp.bBuf, offset: 0, index: 5)
            encMoE.setBuffer(layer.sharedGate.wBuf, offset: 0, index: 6)
            encMoE.setBuffer(layer.sharedGate.sBuf, offset: 0, index: 7)
            encMoE.setBuffer(layer.sharedGate.bBuf, offset: 0, index: 8)
            encMoE.setBuffer(layer.sharedUp.wBuf, offset: 0, index: 9)
            encMoE.setBuffer(layer.sharedUp.sBuf, offset: 0, index: 10)
            encMoE.setBuffer(layer.sharedUp.bBuf, offset: 0, index: 11)
            encMoE.setBuffer(normMLPBuf, offset: 0, index: 12)
            encMoE.setBuffer(batchedActBuf, offset: 0, index: 13)
            encMoE.setBuffer(shActBuf, offset: 0, index: 14)
            encMoE.setBuffer(expertIndicesBuf, offset: 0, index: 15)
            encMoE.setBytes(&switchGWOff, length: 4, index: 16)
            encMoE.setBytes(&switchGSOff, length: 4, index: 17)
            encMoE.setBytes(&switchGBOff, length: 4, index: 18)
            encMoE.setBytes(&switchUWOff, length: 4, index: 19)
            encMoE.setBytes(&switchUSOff, length: 4, index: 20)
            encMoE.setBytes(&switchUBOff, length: 4, index: 21)
            encMoE.setBytes(&sharedGWOff, length: 4, index: 22)
            encMoE.setBytes(&sharedGSOff, length: 4, index: 23)
            encMoE.setBytes(&sharedGBOff, length: 4, index: 24)
            encMoE.setBytes(&sharedUWOff, length: 4, index: 25)
            encMoE.setBytes(&sharedUSOff, length: 4, index: 26)
            encMoE.setBytes(&sharedUBOff, length: 4, index: 27)
            encMoE.dispatchThreadgroups(MTLSize(width: (4608 + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            // 5. Fused Down-Accum & Residual
            var switchDWOff = UInt32(layer.switchDown.wOff)
            var switchDSOff = UInt32(layer.switchDown.sOff)
            var switchDBOff = UInt32(layer.switchDown.bOff)
            var sharedDWOff = UInt32(layer.sharedDown.wOff)
            var sharedDSOff = UInt32(layer.sharedDown.sOff)
            var sharedDBOff = UInt32(layer.sharedDown.bOff)

            encMoE.setComputePipelineState(fusedMoeDownAccumResidualPipe)
            encMoE.setBuffer(layer.switchDown.wBuf, offset: 0, index: 0)
            encMoE.setBuffer(layer.switchDown.sBuf, offset: 0, index: 1)
            encMoE.setBuffer(layer.switchDown.bBuf, offset: 0, index: 2)
            encMoE.setBuffer(layer.sharedDown.wBuf, offset: 0, index: 3)
            encMoE.setBuffer(layer.sharedDown.sBuf, offset: 0, index: 4)
            encMoE.setBuffer(layer.sharedDown.bBuf, offset: 0, index: 5)
            encMoE.setBuffer(batchedActBuf, offset: 0, index: 6)
            encMoE.setBuffer(shActBuf, offset: 0, index: 7)
            encMoE.setBuffer(expertIndicesBuf, offset: 0, index: 8)
            encMoE.setBuffer(scoresBuf, offset: 0, index: 9)
            encMoE.setBuffer(shCoeffBuf, offset: 0, index: 10)
            encMoE.setBuffer(attnResBuf, offset: 0, index: 11)
            encMoE.setBuffer(curOutBuf, offset: 0, index: 12)
            encMoE.setBytes(&switchDWOff, length: 4, index: 13)
            encMoE.setBytes(&switchDSOff, length: 4, index: 14)
            encMoE.setBytes(&switchDBOff, length: 4, index: 15)
            encMoE.setBytes(&sharedDWOff, length: 4, index: 16)
            encMoE.setBytes(&sharedDSOff, length: 4, index: 17)
            encMoE.setBytes(&sharedDBOff, length: 4, index: 18)
            encMoE.dispatchThreadgroups(MTLSize(width: 2048 / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            encMoE.endEncoding()
            cmdMoE.commit()
            cmdMoE.waitUntilCompleted()
        }

        let finalHBuf = hiddenBuf0

        // Final RMSNorm & LM Head
        guard let cmdLM = metalContext.queue.makeCommandBuffer(),
              let encLM = cmdLM.makeComputeCommandEncoder() else { fatalError() }
        encLM.setComputePipelineState(rmsnormPipe)
        encLM.setBuffer(finalHBuf, offset: 0, index: 0)
        encLM.setBuffer(finalNormWBuf, offset: 0, index: 1)
        encLM.setBuffer(finalNormOutBuf, offset: 0, index: 2)
        encLM.setBytes(&dim2048, length: 4, index: 3)
        encLM.setBytes(&eps, length: 4, index: 4)
        encLM.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        encodeQ4GEMV(encoder: encLM, triplet: lmHeadTriplet, vec: finalNormOutBuf, out: logitsOutBuf, rows: 248320, cols: 2048)
        encLM.endEncoding()
        cmdLM.commit()
        cmdLM.waitUntilCompleted()

        let logitsPtr = logitsOutBuf.contents().assumingMemoryBound(to: Float.self)
        if let generationConfiguration {
            return QwenLogitSelector.select(
                logits: UnsafePointer(logitsPtr),
                count: 248_320,
                seenTokenIDs: seenTokenIDs,
                configuration: generationConfiguration,
                samplePosition: samplePosition
            )
        }
        var maxLogit: Float = -Float.infinity
        var maxIdx: Int32 = 0
        for index in 0..<248_320 where logitsPtr[index] > maxLogit {
            maxLogit = logitsPtr[index]
            maxIdx = Int32(index)
        }
        return maxIdx
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

        // 1. Prefill all prompt tokens
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
}
