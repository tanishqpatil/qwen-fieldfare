import Foundation

public struct QwenTensorInfo: Sendable {
    public let shape: [Int]
    public let dtype: String
    public let dataOffsets: [Int]
}

public final class QwenResidentWeights: @unchecked Sendable {
    public let checkpointDirectory: String
    public let weightMap: [String: String]
    public let numLayers: Int
    private var shardHeaders: [String: (headerSize: Int, tensors: [String: QwenTensorInfo])] = [:]
    private let lock = NSLock()

    public init(checkpointDirectory: String, weightMap: [String: String] = [:], numLayers: Int = 40) {
        self.checkpointDirectory = checkpointDirectory
        self.weightMap = weightMap
        self.numLayers = numLayers
    }

    public func getTensorInfo(name: String) throws -> (shardPath: String, headerSize: Int, info: QwenTensorInfo) {
        guard let shardFile = weightMap[name] else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "Tensor \(name) not found in weight_map"])
        }
        let shardPath = (checkpointDirectory as NSString).appendingPathComponent(shardFile)

        lock.lock()
        defer { lock.unlock() }

        if let cached = shardHeaders[shardFile], let info = cached.tensors[name] {
            return (shardPath, cached.headerSize, info)
        }

        // Read shard header
        guard let fileHandle = FileHandle(forReadingAtPath: shardPath) else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot open shard \(shardPath)"])
        }
        defer { try? fileHandle.close() }

        guard let lenData = try? fileHandle.read(upToCount: 8), lenData.count == 8 else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 5, userInfo: [NSLocalizedDescriptionKey: "Cannot read header length from \(shardPath)"])
        }
        let rawHeaderLen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        let fileSize = try fileHandle.seekToEnd()
        guard rawHeaderLen <= 100_000_000, fileSize >= 8,
              rawHeaderLen <= fileSize - 8, let headerLen = Int(exactly: rawHeaderLen) else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Invalid or oversized Safetensors header in \(shardPath)"])
        }
        try fileHandle.seek(toOffset: 8)

        guard let headerData = try? fileHandle.read(upToCount: headerLen), headerData.count == headerLen else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 6, userInfo: [NSLocalizedDescriptionKey: "Cannot read JSON header from \(shardPath)"])
        }

        guard let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 7, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON header in \(shardPath)"])
        }

        var tensors: [String: QwenTensorInfo] = [:]
        for (k, v) in json {
            if k == "__metadata__" { continue }
            guard let dict = v as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shape = dict["shape"] as? [Int],
                  let offs = dict["data_offsets"] as? [Int], offs.count == 2 else { continue }
            guard offs[0] >= 0, offs[1] >= offs[0],
                  UInt64(offs[1]) <= fileSize - 8 - rawHeaderLen,
                  shape.allSatisfy({ $0 >= 0 }) else {
                throw NSError(domain: "QwenResidentCoreLoader", code: 7,
                    userInfo: [NSLocalizedDescriptionKey: "Invalid tensor bounds for \(k) in \(shardPath)"])
            }
            tensors[k] = QwenTensorInfo(shape: shape, dtype: dtype, dataOffsets: offs)
        }

        shardHeaders[shardFile] = (headerLen, tensors)
        guard let info = tensors[name] else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 8, userInfo: [NSLocalizedDescriptionKey: "Tensor \(name) not found in shard header \(shardFile)"])
        }
        return (shardPath, headerLen, info)
    }

    public func loadTensor(name: String) throws -> [Float] {
        let (shardPath, headerLen, info) = try getTensorInfo(name: name)
        guard let fileHandle = FileHandle(forReadingAtPath: shardPath) else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 9, userInfo: [NSLocalizedDescriptionKey: "Cannot open shard \(shardPath)"])
        }
        defer { try? fileHandle.close() }

        let startOffset = UInt64(8 + headerLen + info.dataOffsets[0])
        let byteCount = info.dataOffsets[1] - info.dataOffsets[0]

        try fileHandle.seek(toOffset: startOffset)
        guard let rawData = try fileHandle.read(upToCount: byteCount), rawData.count == byteCount else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to read \(byteCount) bytes for \(name)"])
        }

        if info.dtype == "BF16" {
            let elemCount = byteCount / 2
            var out = [Float](repeating: 0, count: elemCount)
            rawData.withUnsafeBytes { rawPtr in
                let u16Ptr = rawPtr.bindMemory(to: UInt16.self)
                for i in 0..<elemCount {
                    let u32Val = UInt32(u16Ptr[i]) << 16
                    out[i] = Float(bitPattern: u32Val)
                }
            }
            return out
        } else if info.dtype == "F32" {
            let elemCount = byteCount / 4
            var out = [Float](repeating: 0, count: elemCount)
            rawData.withUnsafeBytes { rawPtr in
                let fPtr = rawPtr.bindMemory(to: Float.self)
                for i in 0..<elemCount {
                    out[i] = fPtr[i]
                }
            }
            return out
        } else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 11, userInfo: [NSLocalizedDescriptionKey: "Unsupported dtype \(info.dtype) for \(name)"])
        }
    }

    public func loadEmbedTokensRows(tokenIds: [Int], hiddenSize: Int = 2048) throws -> [Float] {
        let name = "model.language_model.embed_tokens.weight"
        let (shardPath, headerLen, info) = try getTensorInfo(name: name)
        guard let fileHandle = FileHandle(forReadingAtPath: shardPath) else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 12, userInfo: [NSLocalizedDescriptionKey: "Cannot open shard \(shardPath)"])
        }
        defer { try? fileHandle.close() }

        let bytesPerRow = hiddenSize * (info.dtype == "BF16" ? 2 : 4)
        var out = [Float](repeating: 0, count: tokenIds.count * hiddenSize)

        for (idx, tok) in tokenIds.enumerated() {
            let rowOffset = UInt64(8 + headerLen + info.dataOffsets[0] + tok * bytesPerRow)
            try fileHandle.seek(toOffset: rowOffset)
            guard let rawData = try fileHandle.read(upToCount: bytesPerRow), rawData.count == bytesPerRow else {
                throw NSError(domain: "QwenResidentCoreLoader", code: 13, userInfo: [NSLocalizedDescriptionKey: "Failed to read embedding row for token \(tok)"])
            }
            if info.dtype == "BF16" {
                rawData.withUnsafeBytes { rawPtr in
                    let u16Ptr = rawPtr.bindMemory(to: UInt16.self)
                    for i in 0..<hiddenSize {
                        let u32Val = UInt32(u16Ptr[i]) << 16
                        out[idx * hiddenSize + i] = Float(bitPattern: u32Val)
                    }
                }
            } else {
                rawData.withUnsafeBytes { rawPtr in
                    let fPtr = rawPtr.bindMemory(to: Float.self)
                    for i in 0..<hiddenSize {
                        out[idx * hiddenSize + i] = fPtr[i]
                    }
                }
            }
        }
        return out
    }

    public func loadLayerWeights(layerIndex l: Int, kind: QwenLayerKind) throws -> QwenLayerWeights {
        let prefix = "model.language_model.layers.\(l)."
        let inputRMSNormWeight = try loadTensor(name: prefix + "input_layernorm.weight")
        let postAttentionNormWeight = try loadTensor(name: prefix + "post_attention_layernorm.weight")
        let routerGateWeight = try loadTensor(name: prefix + "mlp.gate.weight")
        let sharedGateProjWeight = try loadTensor(name: prefix + "mlp.shared_expert.gate_proj.weight")
        let sharedUpProjWeight = try loadTensor(name: prefix + "mlp.shared_expert.up_proj.weight")
        let sharedDownProjWeight = try loadTensor(name: prefix + "mlp.shared_expert.down_proj.weight")
        let sharedGateCoeffWeight = try loadTensor(name: prefix + "mlp.shared_expert_gate.weight")

        if kind == .linearAttention {
            let inProjQKVWeight = try loadTensor(name: prefix + "linear_attn.in_proj_qkv.weight")
            let inProjZWeight = try loadTensor(name: prefix + "linear_attn.in_proj_z.weight")
            let inProjBWeight = try loadTensor(name: prefix + "linear_attn.in_proj_b.weight")
            let inProjAWeight = try loadTensor(name: prefix + "linear_attn.in_proj_a.weight")
            let conv1dFlatWeight = try loadTensor(name: prefix + "linear_attn.conv1d.weight")
            let dtBias = try loadTensor(name: prefix + "linear_attn.dt_bias")
            let aLog = try loadTensor(name: prefix + "linear_attn.A_log")
            let linearNormWeight = try loadTensor(name: prefix + "linear_attn.norm.weight")
            let linearOutProjWeight = try loadTensor(name: prefix + "linear_attn.out_proj.weight")

            return QwenLayerWeights(
                layerIndex: l,
                kind: .linearAttention,
                inputRMSNormWeight: inputRMSNormWeight,
                postAttentionNormWeight: postAttentionNormWeight,
                routerGateWeight: routerGateWeight,
                sharedGateProjWeight: sharedGateProjWeight,
                sharedUpProjWeight: sharedUpProjWeight,
                sharedDownProjWeight: sharedDownProjWeight,
                sharedGateCoeffWeight: sharedGateCoeffWeight,
                inProjQKVWeight: inProjQKVWeight,
                inProjZWeight: inProjZWeight,
                inProjBWeight: inProjBWeight,
                inProjAWeight: inProjAWeight,
                conv1dFlatWeight: conv1dFlatWeight,
                dtBias: dtBias,
                aLog: aLog,
                linearNormWeight: linearNormWeight,
                linearOutProjWeight: linearOutProjWeight,
                qProjWeight: nil,
                kProjWeight: nil,
                vProjWeight: nil,
                qNormWeight: nil,
                kNormWeight: nil,
                oProjWeight: nil
            )
        } else {
            let qProjWeight = try loadTensor(name: prefix + "self_attn.q_proj.weight")
            let kProjWeight = try loadTensor(name: prefix + "self_attn.k_proj.weight")
            let vProjWeight = try loadTensor(name: prefix + "self_attn.v_proj.weight")
            let qNormWeight = try loadTensor(name: prefix + "self_attn.q_norm.weight")
            let kNormWeight = try loadTensor(name: prefix + "self_attn.k_norm.weight")
            let oProjWeight = try loadTensor(name: prefix + "self_attn.o_proj.weight")

            return QwenLayerWeights(
                layerIndex: l,
                kind: .fullAttention,
                inputRMSNormWeight: inputRMSNormWeight,
                postAttentionNormWeight: postAttentionNormWeight,
                routerGateWeight: routerGateWeight,
                sharedGateProjWeight: sharedGateProjWeight,
                sharedUpProjWeight: sharedUpProjWeight,
                sharedDownProjWeight: sharedDownProjWeight,
                sharedGateCoeffWeight: sharedGateCoeffWeight,
                inProjQKVWeight: nil,
                inProjZWeight: nil,
                inProjBWeight: nil,
                inProjAWeight: nil,
                conv1dFlatWeight: nil,
                dtBias: nil,
                aLog: nil,
                linearNormWeight: nil,
                linearOutProjWeight: nil,
                qProjWeight: qProjWeight,
                kProjWeight: kProjWeight,
                vProjWeight: vProjWeight,
                qNormWeight: qNormWeight,
                kNormWeight: kNormWeight,
                oProjWeight: oProjWeight
            )
        }
    }
}

public enum QwenLayerKind: Sendable {
    case linearAttention
    case fullAttention
}

public struct QwenLayerWeights: Sendable {
    public let layerIndex: Int
    public let kind: QwenLayerKind
    public let inputRMSNormWeight: [Float]
    public let postAttentionNormWeight: [Float]
    public let routerGateWeight: [Float]
    public let sharedGateProjWeight: [Float]
    public let sharedUpProjWeight: [Float]
    public let sharedDownProjWeight: [Float]
    public let sharedGateCoeffWeight: [Float]

    // Linear Attention
    public let inProjQKVWeight: [Float]?
    public let inProjZWeight: [Float]?
    public let inProjBWeight: [Float]?
    public let inProjAWeight: [Float]?
    public let conv1dFlatWeight: [Float]?
    public let dtBias: [Float]?
    public let aLog: [Float]?
    public let linearNormWeight: [Float]?
    public let linearOutProjWeight: [Float]?

    // Full Attention
    public let qProjWeight: [Float]?
    public let kProjWeight: [Float]?
    public let vProjWeight: [Float]?
    public let qNormWeight: [Float]?
    public let kNormWeight: [Float]?
    public let oProjWeight: [Float]?
}

public enum QwenResidentCoreLoader {
    public static func load(from checkpointDirectory: String) throws -> QwenResidentWeights {
        let indexPath = (checkpointDirectory as NSString).appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: indexPath)) else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing index.json"])
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String] else {
            throw NSError(domain: "QwenResidentCoreLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid weight_map in index.json"])
        }
        return QwenResidentWeights(checkpointDirectory: checkpointDirectory, weightMap: weightMap, numLayers: 40)
    }
}
