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
    private struct Shard {
        let path: String
        let file: QwenSafetensorsFile
    }
    private var shards: [String: Shard] = [:]
    private let canonicalDirectory: URL
    private let lock = NSLock()

    public init(checkpointDirectory: String, weightMap: [String: String] = [:], numLayers: Int = 40) {
        self.checkpointDirectory = checkpointDirectory
        self.weightMap = weightMap
        self.numLayers = numLayers
        self.canonicalDirectory = URL(fileURLWithPath: (checkpointDirectory as NSString).expandingTildeInPath,
                                      isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
    }

    // A serialized borrow: body must not retain the file or re-enter this reader.
    // Cache keeps validated descriptors, not merely metadata for later path opens.
    func withTensor<T>(name: String, _ body: (QwenSafetensorsFile, String, QwenTensorInfo) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        guard let shardName = weightMap[name] else {
            throw QwenSafetensorsError.invalidTensor(name: name, reason: "not found in weight_map")
        }
        do {
            let shard: Shard
            if let cached = shards[shardName] {
                shard = cached
            } else {
                guard !shardName.isEmpty, !shardName.utf8.contains(0),
                      shardName == (shardName as NSString).lastPathComponent else {
                    throw QwenModelDirectoryError.invalidIndex("unsafe shard name \(shardName)")
                }
                let url = canonicalDirectory.appendingPathComponent(shardName).resolvingSymlinksInPath()
                guard url.deletingLastPathComponent().path == canonicalDirectory.path else {
                    throw QwenModelDirectoryError.invalidIndex("shard resolves outside model directory: \(shardName)")
                }
                shard = Shard(path: url.path, file: try QwenSafetensorsFile(resolvedURL: url))
                shards[shardName] = shard
            }
            try shard.file.verifyUnchanged()
            guard let info = shard.file.header.tensors[name] else {
                throw QwenSafetensorsError.invalidTensor(name: name, reason: "not found in shard header")
            }
            return try body(shard.file, shard.path, info)
        } catch {
            // Fail closed, release ownership, and require explicit caller retry.
            // A retry opens/revalidates a fresh descriptor, never stale offsets.
            shards.removeValue(forKey: shardName)
            throw error
        }
    }

    public func getTensorInfo(name: String) throws -> (shardPath: String, headerSize: Int, info: QwenTensorInfo) {
        try withTensor(name: name) { file, path, info in
            (path, file.header.headerByteCount, info)
        }
    }

    public func loadTensor(name: String) throws -> [Float] {
        try withTensor(name: name) { file, _, info in
            let width = try Self.floatWidth(info.dtype, name: name)
            let range = info.dataOffsets[0]..<info.dataOffsets[1]
            let count = try Self.checkedOutputCount(range.count / width, name: name)
            let bytes = try file.readPayload(range)
            var output = [Float](repeating: 0, count: count)
            Self.decode(bytes, dtype: info.dtype, into: &output, at: 0)
            return output
        }
    }

    public func loadEmbedTokensRows(tokenIds: [Int], hiddenSize: Int = 2048) throws -> [Float] {
        let name = "model.language_model.embed_tokens.weight"
        return try withTensor(name: name) { file, _, info in
            let width = try Self.floatWidth(info.dtype, name: name)
            guard hiddenSize > 0, info.shape.count == 2, info.shape[1] == hiddenSize,
                  tokenIds.allSatisfy({ $0 >= 0 && $0 < info.shape[0] }) else {
                throw QwenSafetensorsError.invalidTensor(name: name, reason: "invalid embedding shape, hidden size or token index")
            }
            let rowBytes = hiddenSize.multipliedReportingOverflow(by: width)
            let count = tokenIds.count.multipliedReportingOverflow(by: hiddenSize)
            guard !rowBytes.overflow, !count.overflow else {
                throw QwenSafetensorsError.arithmeticOverflow(name: name)
            }
            let outputCount = try Self.checkedOutputCount(count.partialValue, name: name)
            var output = [Float](repeating: 0, count: outputCount)
            for (index, token) in tokenIds.enumerated() {
                // Validated 2D shape/extent and token bounds prove these fit,
                // including the final row ending exactly at the tensor bound.
                let start = info.dataOffsets[0] + token * rowBytes.partialValue
                let bytes = try file.readPayload(start..<(start + rowBytes.partialValue))
                Self.decode(bytes, dtype: info.dtype, into: &output, at: index * hiddenSize)
            }
            return output
        }
    }

    private static func floatWidth(_ dtype: String, name: String) throws -> Int {
        switch dtype {
        case "BF16": return 2
        case "F32": return 4
        default: throw QwenSafetensorsError.unsupportedDtype(name: name, dtype: dtype)
        }
    }

    private static func checkedOutputCount(_ count: Int, name: String) throws -> Int {
        guard count >= 0, !count.multipliedReportingOverflow(by: MemoryLayout<Float>.stride).overflow else {
            throw QwenSafetensorsError.arithmeticOverflow(name: name)
        }
        return count
    }

    private static func decode(_ bytes: Data, dtype: String, into output: inout [Float], at start: Int) {
        bytes.withUnsafeBytes { raw in
            if dtype == "BF16" {
                for index in 0..<(bytes.count / 2) {
                    let bits = raw.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self).littleEndian
                    output[start + index] = Float(bitPattern: UInt32(bits) << 16)
                }
            } else {
                for index in 0..<(bytes.count / 4) {
                    let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian
                    output[start + index] = Float(bitPattern: bits)
                }
            }
        }
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
