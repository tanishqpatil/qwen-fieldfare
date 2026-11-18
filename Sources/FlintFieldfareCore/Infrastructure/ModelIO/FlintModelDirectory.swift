import Darwin
import Foundation

public enum FlintModelDirectoryError: Error, LocalizedError, Equatable {
    case notDirectory(String)
    case missingFile(String)
    case invalidIndex(String)
    case invalidMetadata(String)

    public var errorDescription: String? {
        switch self {
        case .notDirectory(let path): return "The selected Flint model path is not a directory: \(path)"
        case .missingFile(let name): return "The selected Flint model directory is missing \(name)."
        case .invalidIndex(let detail): return "The Flint model Safetensors index is invalid: \(detail)"
        case .invalidMetadata(let detail): return "The Flint model configuration/metadata is invalid: \(detail)"
        }
    }
}

public struct FlintConfig: Equatable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let noRopeLayerIndices: [Int]
    public let mlpOnlyLayers: [Int]
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let routerScoreFunction: String
    public let bosTokenId: Int
    public let eosTokenIds: [Int]
    public let routedExpertBits: Int
    public let embeddingBits: Int
    public let groupSize: Int

    public static let expectedFullAttentionIndices: Set<Int> = [
        0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48
    ]

    public var fullAttentionLayerIndices: [Int] {
        layerTypes.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil }
    }

    public var slidingAttentionLayerIndices: [Int] {
        layerTypes.enumerated().compactMap { $0.element == "sliding_attention" ? $0.offset : nil }
    }

    public var moeLayerIndices: [Int] {
        Array(2..<numHiddenLayers)
    }
}

public enum FlintModelDirectory {
    static let maximumMetadataBytes = 16 * 1_024 * 1_024 // 16 MiB bound for metadata files
    public static let requiredMetadataFiles = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja"
    ]

    public struct ValidatedDirectory: Sendable {
        public let canonicalPath: String
        public let config: FlintConfig
        public let weightMap: [String: String]
        public let shards: [String]
    }

    @discardableResult
    public static func validate(path: String, fileManager: FileManager = .default) throws -> ValidatedDirectory {
        let expanded = (path as NSString).expandingTildeInPath
        let canonical = URL(fileURLWithPath: expanded, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath().path
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: canonical, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw FlintModelDirectoryError.notDirectory(canonical)
        }

        var metadataURLs: [String: URL] = [:]
        for name in requiredMetadataFiles {
            let url = URL(fileURLWithPath: canonical, isDirectory: true)
                .appendingPathComponent(name).resolvingSymlinksInPath()
            guard url.deletingLastPathComponent().path == canonical else {
                throw FlintModelDirectoryError.invalidMetadata("\(name) resolves outside model directory")
            }
            guard fileManager.fileExists(atPath: url.path) else {
                throw FlintModelDirectoryError.missingFile(name)
            }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                throw FlintModelDirectoryError.invalidMetadata("\(name) is not a regular file")
            }
            metadataURLs[name] = url
        }

        // Validate config.json
        guard let configURL = metadataURLs["config.json"] else {
            throw FlintModelDirectoryError.missingFile("config.json")
        }
        let configData = try readFileBounded(at: configURL)
        let config = try parseAndValidateConfig(data: configData)

        // Validate model.safetensors.index.json
        guard let indexURL = metadataURLs["model.safetensors.index.json"] else {
            throw FlintModelDirectoryError.missingFile("model.safetensors.index.json")
        }
        let indexData = try readFileBounded(at: indexURL)
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: indexData)
        } catch {
            throw FlintModelDirectoryError.invalidIndex("malformed JSON")
        }
        guard let root = object as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String], !weightMap.isEmpty else {
            throw FlintModelDirectoryError.invalidIndex("weight_map is missing or empty")
        }

        let shardNames = Set(weightMap.values).sorted()
        for name in shardNames {
            guard !name.isEmpty, name == (name as NSString).lastPathComponent else {
                throw FlintModelDirectoryError.invalidIndex("unsafe shard name \(name)")
            }
            let shard = URL(fileURLWithPath: canonical, isDirectory: true)
                .appendingPathComponent(name).resolvingSymlinksInPath()
            guard shard.deletingLastPathComponent().path == canonical else {
                throw FlintModelDirectoryError.invalidIndex("shard resolves outside model directory: \(name)")
            }
            var shardIsDir: ObjCBool = false
            guard fileManager.fileExists(atPath: shard.path, isDirectory: &shardIsDir) else {
                throw FlintModelDirectoryError.missingFile(name)
            }
            guard !shardIsDir.boolValue,
                  (try? shard.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                throw FlintModelDirectoryError.invalidIndex("shard is not a regular file: \(name)")
            }
        }

        // Validate that critical representative tensors are present in weight_map
        let requiredTensorSamples = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.softmax_offset",
            "model.layers.2.mlp.gate.weight",
            "model.layers.2.mlp.shared_expert.gate_proj.weight",
            "model.layers.2.mlp.switch_mlp.gate_proj.weight",
            "model.layers.48.mlp.switch_mlp.down_proj.weight"
        ]
        for tensor in requiredTensorSamples {
            guard weightMap[tensor] != nil else {
                throw FlintModelDirectoryError.invalidIndex("critical tensor \(tensor) missing from weight_map")
            }
        }

        return ValidatedDirectory(
            canonicalPath: canonical,
            config: config,
            weightMap: weightMap,
            shards: shardNames
        )
    }

    private static func readFileBounded(at url: URL) throws -> Data {
        let fd = open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            throw FlintModelDirectoryError.missingFile(url.lastPathComponent)
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) is not a regular file")
        }
        guard info.st_size > 0 else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) is empty")
        }
        guard info.st_size <= off_t(maximumMetadataBytes) else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) exceeds maximum metadata bound")
        }
        let length = Int(info.st_size)
        var buffer = [UInt8](repeating: 0, count: length)
        var totalRead = 0
        while totalRead < length {
            let bytes = pread(fd, &buffer[totalRead], length - totalRead, off_t(totalRead))
            guard bytes > 0 else {
                throw FlintModelDirectoryError.invalidMetadata("short read on \(url.lastPathComponent)")
            }
            totalRead += bytes
        }
        return Data(buffer)
    }

    public static func parseAndValidateConfig(data: Data) throws -> FlintConfig {
        let jsonObject: Any
        do {
            jsonObject = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw FlintModelDirectoryError.invalidMetadata("malformed config.json")
        }
        guard let dict = jsonObject as? [String: Any] else {
            throw FlintModelDirectoryError.invalidMetadata("config.json root must be a JSON object")
        }

        // Architecture check
        guard let architectures = dict["architectures"] as? [String],
              architectures.contains("FlintForCausalLM") else {
            throw FlintModelDirectoryError.invalidMetadata("architectures must include FlintForCausalLM")
        }

        // Model type check
        if let modelType = dict["model_type"] as? String {
            guard modelType == "flint" else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported model_type: \(modelType)")
            }
        }

        // Mandatory dimensions - NO SILENT FALLBACKS
        guard let hiddenSize = dict["hidden_size"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: hidden_size")
        }
        guard hiddenSize == 2048 else {
            throw FlintModelDirectoryError.invalidMetadata("hidden_size must be 2048, got \(hiddenSize)")
        }

        guard let intermediateSize = dict["intermediate_size"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: intermediate_size")
        }
        guard intermediateSize == 6144 else {
            throw FlintModelDirectoryError.invalidMetadata("intermediate_size must be 6144, got \(intermediateSize)")
        }

        guard let numHiddenLayers = dict["num_hidden_layers"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: num_hidden_layers")
        }
        guard numHiddenLayers == 49 else {
            throw FlintModelDirectoryError.invalidMetadata("num_hidden_layers must be 49, got \(numHiddenLayers)")
        }

        guard let numAttentionHeads = dict["num_attention_heads"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: num_attention_heads")
        }
        guard numAttentionHeads == 32 else {
            throw FlintModelDirectoryError.invalidMetadata("num_attention_heads must be 32, got \(numAttentionHeads)")
        }

        guard let numKeyValueHeads = dict["num_key_value_heads"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: num_key_value_heads")
        }
        guard numKeyValueHeads == 4 else {
            throw FlintModelDirectoryError.invalidMetadata("num_key_value_heads must be 4, got \(numKeyValueHeads)")
        }

        guard let headDim = dict["head_dim"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: head_dim")
        }
        guard headDim == 128 else {
            throw FlintModelDirectoryError.invalidMetadata("head_dim must be 128, got \(headDim)")
        }

        guard let vocabSize = dict["vocab_size"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: vocab_size")
        }
        guard vocabSize == 150144 else {
            throw FlintModelDirectoryError.invalidMetadata("vocab_size must be 150144, got \(vocabSize)")
        }

        guard let eps = dict["rms_norm_eps"] as? Double, eps > 0 else {
            throw FlintModelDirectoryError.invalidMetadata("missing or non-positive rms_norm_eps")
        }
        let rmsNormEps = Float(eps)

        let maxPosEmbeddings = dict["max_position_embeddings"] as? Int ?? 262144
        guard maxPosEmbeddings >= 4096 else {
            throw FlintModelDirectoryError.invalidMetadata("max_position_embeddings too small: \(maxPosEmbeddings)")
        }

        let ropeThetaVal = dict["rope_theta"] as? Double ?? 10000.0
        guard ropeThetaVal > 0 else {
            throw FlintModelDirectoryError.invalidMetadata("rope_theta must be positive")
        }
        let ropeTheta = Float(ropeThetaVal)

        guard let slidingWindow = dict["sliding_window"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: sliding_window")
        }
        guard slidingWindow == 513 else {
            throw FlintModelDirectoryError.invalidMetadata("sliding_window must be 513, got \(slidingWindow)")
        }

        if let hiddenAct = dict["hidden_act"] as? String {
            guard hiddenAct == "silu" else {
                throw FlintModelDirectoryError.invalidMetadata("hidden_act must be silu, got \(hiddenAct)")
            }
        }

        // Layer types validation: exactly 49 entries, exact full-attention arrangement
        guard let layerTypes = dict["layer_types"] as? [String] else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: layer_types")
        }
        guard layerTypes.count == numHiddenLayers else {
            throw FlintModelDirectoryError.invalidMetadata("layer_types count must be \(numHiddenLayers), got \(layerTypes.count)")
        }
        for (idx, type) in layerTypes.enumerated() {
            guard type == "full_attention" || type == "sliding_attention" else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported layer type at index \(idx): \(type)")
            }
        }
        let fullIndices = Set(layerTypes.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil })
        guard fullIndices == FlintConfig.expectedFullAttentionIndices else {
            throw FlintModelDirectoryError.invalidMetadata("layer_types full_attention arrangement mismatch: expected \(FlintConfig.expectedFullAttentionIndices.sorted()), got \(fullIndices.sorted())")
        }

        // No-RoPE layer indices validation
        guard let noRopeIndices = dict["no_rope_layer_indices"] as? [Int] else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: no_rope_layer_indices")
        }
        guard Set(noRopeIndices) == FlintConfig.expectedFullAttentionIndices else {
            throw FlintModelDirectoryError.invalidMetadata("no_rope_layer_indices must exactly match the 14 full-attention layers")
        }

        // Dense MLP layers validation
        guard let mlpOnlyLayers = dict["mlp_only_layers"] as? [Int] else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: mlp_only_layers")
        }
        guard mlpOnlyLayers == [0, 1] else {
            throw FlintModelDirectoryError.invalidMetadata("mlp_only_layers must be [0, 1], got \(mlpOnlyLayers)")
        }

        // MoE parameters
        guard let numExperts = dict["num_experts"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: num_experts")
        }
        guard numExperts == 256 else {
            throw FlintModelDirectoryError.invalidMetadata("num_experts must be 256, got \(numExperts)")
        }

        guard let numExpertsPerTok = dict["num_experts_per_tok"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: num_experts_per_tok")
        }
        guard numExpertsPerTok == 6 else {
            throw FlintModelDirectoryError.invalidMetadata("num_experts_per_tok must be 6, got \(numExpertsPerTok)")
        }

        guard let moeIntermediateSize = dict["moe_intermediate_size"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: moe_intermediate_size")
        }
        guard moeIntermediateSize == 512 else {
            throw FlintModelDirectoryError.invalidMetadata("moe_intermediate_size must be 512, got \(moeIntermediateSize)")
        }

        guard let sharedExpertIntermediateSize = dict["shared_expert_intermediate_size"] as? Int else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: shared_expert_intermediate_size")
        }
        guard sharedExpertIntermediateSize == 1024 else {
            throw FlintModelDirectoryError.invalidMetadata("shared_expert_intermediate_size must be 1024, got \(sharedExpertIntermediateSize)")
        }

        let routerScoreFunction = dict["router_score_function"] as? String ?? dict["scoring_func"] as? String
        guard let rScoreFn = routerScoreFunction else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory router scoring function")
        }
        guard rScoreFn == "sigmoid" else {
            throw FlintModelDirectoryError.invalidMetadata("router score function must be sigmoid, got \(rScoreFn)")
        }

        if let normTopk = dict["norm_topk_prob"] as? Bool {
            guard normTopk == true else {
                throw FlintModelDirectoryError.invalidMetadata("norm_topk_prob must be true")
            }
        }

        let bosTokenId = dict["bos_token_id"] as? Int ?? 150000
        let eosTokenIds: [Int]
        if let singleEos = dict["eos_token_id"] as? Int {
            eosTokenIds = [singleEos]
        } else if let multiEos = dict["eos_token_id"] as? [Int] {
            eosTokenIds = multiEos
        } else {
            eosTokenIds = [150001, 150003, 150005, 150018]
        }

        // Comprehensive Quantization Validation
        guard let quant = dict["quantization"] as? [String: Any] else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory quantization metadata")
        }
        guard let baseGroupSize = quant["group_size"] as? Int, baseGroupSize == 64 else {
            throw FlintModelDirectoryError.invalidMetadata("quantization group_size must be 64")
        }
        guard let baseBits = quant["bits"] as? Int, baseBits == 4 else {
            throw FlintModelDirectoryError.invalidMetadata("base quantization bits must be 4")
        }

        var embeddingBits = 8
        if let embedQuant = quant["model.embed_tokens"] as? [String: Any] {
            guard let eb = embedQuant["bits"] as? Int, eb == 8 else {
                throw FlintModelDirectoryError.invalidMetadata("embedding bits must be 8")
            }
            embeddingBits = eb
        }

        // Validate EVERY required routed-expert projection across all 47 MoE layers
        var detectedRoutedBits: Int? = nil
        let projections = ["gate_proj", "up_proj", "down_proj"]
        for layerIdx in 2..<numHiddenLayers {
            for proj in projections {
                let key = "model.layers.\(layerIdx).mlp.switch_mlp.\(proj)"
                if let projQuant = quant[key] as? [String: Any] {
                    guard let g = projQuant["group_size"] as? Int, g == 64 else {
                        throw FlintModelDirectoryError.invalidMetadata("expert projection \(key) group_size must be 64")
                    }
                    guard let b = projQuant["bits"] as? Int, b == 3 || b == 4 else {
                        throw FlintModelDirectoryError.invalidMetadata("expert projection \(key) bits must be 3 or 4, got \(String(describing: projQuant["bits"]))")
                    }
                    if let current = detectedRoutedBits {
                        guard current == b else {
                            throw FlintModelDirectoryError.invalidMetadata("inconsistent expert quantization: \(key) has bits=\(b) while other experts have bits=\(current)")
                        }
                    } else {
                        detectedRoutedBits = b
                    }
                }
            }
        }
        let routedBits = detectedRoutedBits ?? baseBits

        return FlintConfig(
            modelType: dict["model_type"] as? String ?? "flint",
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            vocabSize: vocabSize,
            rmsNormEps: rmsNormEps,
            maxPositionEmbeddings: maxPosEmbeddings,
            ropeTheta: ropeTheta,
            slidingWindow: slidingWindow,
            layerTypes: layerTypes,
            noRopeLayerIndices: noRopeIndices,
            mlpOnlyLayers: mlpOnlyLayers,
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            moeIntermediateSize: moeIntermediateSize,
            sharedExpertIntermediateSize: sharedExpertIntermediateSize,
            routerScoreFunction: rScoreFn,
            bosTokenId: bosTokenId,
            eosTokenIds: eosTokenIds,
            routedExpertBits: routedBits,
            embeddingBits: embeddingBits,
            groupSize: baseGroupSize
        )
    }
}
