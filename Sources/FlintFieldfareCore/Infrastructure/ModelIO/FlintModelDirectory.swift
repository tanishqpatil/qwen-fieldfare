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
        public let configData: Data
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
        guard (try? JSONSerialization.jsonObject(with: configData)) is [String: Any] else {
            throw FlintModelDirectoryError.invalidMetadata("config.json root must be a JSON object")
        }
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
            configData: configData,
            weightMap: weightMap,
            shards: shardNames
        )
    }

    /// Opens all Safetensors shard files declared by the validated directory.
    public static func openShards(canonicalPath: String, shardNames: [String]) throws -> [String: FlintSafetensorsFile] {
        var shardFiles: [String: FlintSafetensorsFile] = [:]
        for name in shardNames {
            let url = URL(fileURLWithPath: canonicalPath, isDirectory: true).appendingPathComponent(name)
            shardFiles[name] = try FlintSafetensorsFile(url: url)
        }
        return shardFiles
    }

    /// Cross-validates metadata and physical Safetensors headers to prevent contradictory or corrupt layouts.
    public static func crossValidate(
        config: FlintConfig,
        configData: Data,
        shardFiles: [String: FlintSafetensorsFile],
        weightMap: [String: String]
    ) throws {
        guard let rawConfigDict = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any] else {
            throw FlintSafetensorsError.malformedHeader("configData is not a valid JSON dictionary")
        }
        // 1. Cross-check mlx_export metadata consistency
        if let mlxExport = rawConfigDict["mlx_export"] as? [String: Any] {
            if let mlxEmbedBits = mlxExport["embedding_bits"] as? Int {
                guard mlxEmbedBits == config.embeddingBits else {
                    throw FlintSafetensorsError.metadataConflict(
                        "mlx_export.embedding_bits (\(mlxEmbedBits)) conflicts with config embeddingBits (\(config.embeddingBits))"
                    )
                }
                if let quant = (rawConfigDict["quantization"] ?? rawConfigDict["quantization_config"]) as? [String: Any],
                   let embedQuant = quant["model.embed_tokens"] as? [String: Any],
                   let quantEmbedBits = embedQuant["bits"] as? Int {
                    guard quantEmbedBits == mlxEmbedBits else {
                        throw FlintSafetensorsError.metadataConflict(
                            "mlx_export.embedding_bits (\(mlxEmbedBits)) conflicts with quantization.model.embed_tokens.bits (\(quantEmbedBits))"
                        )
                    }
                }
            }

            if let mlxRoutedBits = mlxExport["routed_expert_bits"] as? Int {
                guard mlxRoutedBits == config.routedExpertBits else {
                    throw FlintSafetensorsError.metadataConflict(
                        "mlx_export.routed_expert_bits (\(mlxRoutedBits)) conflicts with detected routedExpertBits (\(config.routedExpertBits))"
                    )
                }
            }
        }

        // 2. Cross-check that every tensor in weight_map exists in its assigned shard
        for (tensorName, shardName) in weightMap {
            guard let shardFile = shardFiles[shardName] else {
                throw FlintSafetensorsError.tensorNotFound(name: "shard '\(shardName)' not open for tensor '\(tensorName)'")
            }
            guard shardFile.header.tensors[tensorName] != nil else {
                throw FlintSafetensorsError.tensorNotFound(name: "\(tensorName) missing from shard \(shardName)")
            }
        }

        // 3. Verify physical layout of representative tensors against config
        func verifyTensor(name: String, expectedShape: [Int], expectedDtype: String) throws {
            guard let shardName = weightMap[name], let shardFile = shardFiles[shardName] else {
                throw FlintSafetensorsError.tensorNotFound(name: name)
            }
            guard let tensorInfo = shardFile.header.tensors[name] else {
                throw FlintSafetensorsError.tensorNotFound(name: "\(name) in shard \(shardName)")
            }
            guard tensorInfo.shape == expectedShape else {
                throw FlintSafetensorsError.metadataConflict(
                    "physical tensor '\(name)' shape \(tensorInfo.shape) does not match expected shape \(expectedShape)"
                )
            }
            guard tensorInfo.dtype == expectedDtype else {
                throw FlintSafetensorsError.metadataConflict(
                    "physical tensor '\(name)' dtype \(tensorInfo.dtype) does not match expected dtype \(expectedDtype)"
                )
            }
        }

        // Embedding:
        if config.embeddingBits == 8 {
            try verifyTensor(name: "model.embed_tokens.weight", expectedShape: [150144, 512], expectedDtype: "U32")
        } else if config.embeddingBits == 4 {
            try verifyTensor(name: "model.embed_tokens.weight", expectedShape: [150144, 256], expectedDtype: "U32")
        }
        try verifyTensor(name: "model.embed_tokens.scales", expectedShape: [150144, 32], expectedDtype: "BF16")
        try verifyTensor(name: "model.embed_tokens.biases", expectedShape: [150144, 32], expectedDtype: "BF16")

        // LM Head:
        try verifyTensor(name: "lm_head.weight", expectedShape: [150144, 256], expectedDtype: "U32")
        try verifyTensor(name: "lm_head.scales", expectedShape: [150144, 32], expectedDtype: "BF16")
        try verifyTensor(name: "lm_head.biases", expectedShape: [150144, 32], expectedDtype: "BF16")

        // Norms & Attention:
        try verifyTensor(name: "model.norm.weight", expectedShape: [2048], expectedDtype: "BF16")
        try verifyTensor(name: "model.layers.0.input_layernorm.weight", expectedShape: [2048], expectedDtype: "BF16")
        try verifyTensor(name: "model.layers.0.self_attn.q_proj.weight", expectedShape: [4096, 256], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.0.self_attn.q_norm.weight", expectedShape: [128], expectedDtype: "BF16")
        try verifyTensor(name: "model.layers.0.self_attn.softmax_offset", expectedShape: [32], expectedDtype: "BF16")

        // Dense MLP (Layer 0):
        try verifyTensor(name: "model.layers.0.mlp.gate_proj.weight", expectedShape: [6144, 256], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.0.mlp.down_proj.weight", expectedShape: [2048, 768], expectedDtype: "U32")

        // MoE Router & Shared Expert (Layer 2):
        try verifyTensor(name: "model.layers.2.mlp.gate.weight", expectedShape: [256, 2048], expectedDtype: "BF16")
        try verifyTensor(name: "model.layers.2.mlp.shared_expert.gate_proj.weight", expectedShape: [1024, 256], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.2.mlp.shared_expert.down_proj.weight", expectedShape: [2048, 128], expectedDtype: "U32")

        // MoE Routed Experts (Layer 2 & Layer 48):
        let gateUpU32Words = config.routedExpertBits == 3 ? 192 : 256
        let downU32Words = config.routedExpertBits == 3 ? 48 : 64

        try verifyTensor(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", expectedShape: [256, 512, gateUpU32Words], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.2.mlp.switch_mlp.up_proj.weight", expectedShape: [256, 512, gateUpU32Words], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.2.mlp.switch_mlp.down_proj.weight", expectedShape: [256, 2048, downU32Words], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.2.mlp.switch_mlp.gate_proj.scales", expectedShape: [256, 512, 32], expectedDtype: "BF16")
        try verifyTensor(name: "model.layers.2.mlp.switch_mlp.down_proj.scales", expectedShape: [256, 2048, 8], expectedDtype: "BF16")

        try verifyTensor(name: "model.layers.48.mlp.switch_mlp.gate_proj.weight", expectedShape: [256, 512, gateUpU32Words], expectedDtype: "U32")
        try verifyTensor(name: "model.layers.48.mlp.switch_mlp.down_proj.weight", expectedShape: [256, 2048, downU32Words], expectedDtype: "U32")
    }

    private static func readFileBounded(at url: URL) throws -> Data {
        let fd = open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            throw FlintModelDirectoryError.missingFile(url.lastPathComponent)
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) is not a regular file")
        }
        guard info.st_size > 0 else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) is empty")
        }
        guard info.st_size <= off_t(maximumMetadataBytes) else {
            throw FlintModelDirectoryError.invalidMetadata("\(url.lastPathComponent) exceeds maximum metadata bound (\(info.st_size) > \(maximumMetadataBytes) bytes)")
        }
        let length = Int(info.st_size)
        var data = Data(count: length)
        var totalRead = 0
        try data.withUnsafeMutableBytes { (rawBuffer: UnsafeMutableRawBufferPointer) throws in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw FlintModelDirectoryError.invalidMetadata("failed to obtain memory buffer for \(url.lastPathComponent)")
            }
            while totalRead < length {
                let bytes = pread(fd, baseAddress.advanced(by: totalRead), length - totalRead, off_t(totalRead))
                if bytes < 0 {
                    let err = errno
                    throw FlintModelDirectoryError.invalidMetadata("read failed on \(url.lastPathComponent) with errno \(err): \(String(cString: strerror(err)))")
                }
                guard bytes > 0 else {
                    throw FlintModelDirectoryError.invalidMetadata("unexpected EOF / short read on \(url.lastPathComponent): read \(totalRead) of \(length) bytes")
                }
                totalRead += bytes
            }
        }
        return data
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

        // Model type check: mandatory, must be "flint"
        guard let modelType = dict["model_type"] as? String else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: model_type")
        }
        guard modelType == "flint" else {
            throw FlintModelDirectoryError.invalidMetadata("unsupported model_type: \(modelType) (expected flint)")
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

        // No-RoPE layer indices validation: exactly 14 unique indices matching the full-attention layers
        guard let noRopeIndices = dict["no_rope_layer_indices"] as? [Int] else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: no_rope_layer_indices")
        }
        guard noRopeIndices.count == 14 else {
            throw FlintModelDirectoryError.invalidMetadata("no_rope_layer_indices must contain exactly 14 layer indices, got \(noRopeIndices.count)")
        }
        guard Set(noRopeIndices).count == 14 else {
            throw FlintModelDirectoryError.invalidMetadata("no_rope_layer_indices contains duplicate layer indices: \(noRopeIndices)")
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

        guard let normTopk = dict["norm_topk_prob"] as? Bool else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory field: norm_topk_prob")
        }
        guard normTopk == true else {
            throw FlintModelDirectoryError.invalidMetadata("norm_topk_prob must be true for Flint router normalization")
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
        let quantDict = (dict["quantization"] ?? dict["quantization_config"]) as? [String: Any]
        guard let quant = quantDict else {
            throw FlintModelDirectoryError.invalidMetadata("missing mandatory quantization metadata (quantization / quantization_config)")
        }
        guard let baseGroupSize = quant["group_size"] as? Int, baseGroupSize == 64 else {
            throw FlintModelDirectoryError.invalidMetadata("quantization group_size must be 64")
        }
        guard let baseBits = quant["bits"] as? Int, baseBits == 4 else {
            throw FlintModelDirectoryError.invalidMetadata("base quantization bits must be 4")
        }

        // Embedding quantization:
        // Supported: INT8 (standard MLX/Flint) or INT4 (uniform).
        // If explicitly in quant["model.embed_tokens"], validate group_size and bits.
        // Otherwise, inspect mlx_export["embedding_bits"] or default to baseBits.
        let embeddingBits: Int
        if let embedVal = quant["model.embed_tokens"] {
            guard let embedQuant = embedVal as? [String: Any] else {
                throw FlintModelDirectoryError.invalidMetadata("model.embed_tokens in quantization must be a dictionary")
            }
            guard let eb = embedQuant["bits"] as? Int, eb == 8 || eb == 4 else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported model.embed_tokens bits: \(String(describing: embedQuant["bits"])) (expected 8 or 4)")
            }
            guard let eg = embedQuant["group_size"] as? Int, eg == 64 else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported model.embed_tokens group_size: \(String(describing: embedQuant["group_size"])) (expected 64)")
            }
            embeddingBits = eb
        } else if let mlxExport = dict["mlx_export"] as? [String: Any],
                  let eb = mlxExport["embedding_bits"] as? Int {
            guard eb == 8 || eb == 4 else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported mlx_export embedding_bits: \(eb) (expected 8 or 4)")
            }
            embeddingBits = eb
        } else {
            // Unquantized or uniform 4-bit embedding: explicit fallback to baseBits (4)
            embeddingBits = baseBits
        }

        // Validate routed expert quantization:
        // Case A: Official INT4 checkpoint (0 switch_mlp overrides, inherits base 4-bit / group 64)
        // Case B: Mixed 3/4-bit checkpoint (ALL 141 projections must explicitly exist, have group_size 64, and match bits == 3 or 4)
        // Any partial count (1..140), extraneous keys, or inconsistent bit depth/group size is rejected.
        let projections = ["gate_proj", "up_proj", "down_proj"]
        var expectedSwitchMlpKeys: [String] = []
        for layerIdx in 2..<numHiddenLayers {
            for proj in projections {
                expectedSwitchMlpKeys.append("model.layers.\(layerIdx).mlp.switch_mlp.\(proj)")
            }
        }
        let expectedKeySet = Set(expectedSwitchMlpKeys)
        precondition(expectedSwitchMlpKeys.count == 141, "Invariant: Flint has 47 MoE layers * 3 projections = 141")

        var switchMlpOverrides: [String: (bits: Int, groupSize: Int)] = [:]
        for (key, val) in quant {
            guard key.contains("switch_mlp") else { continue }
            guard let projDict = val as? [String: Any] else {
                throw FlintModelDirectoryError.invalidMetadata("quantization entry for \(key) must be a dictionary")
            }
            guard let b = projDict["bits"] as? Int else {
                throw FlintModelDirectoryError.invalidMetadata("missing bits in quantization entry for \(key)")
            }
            guard let g = projDict["group_size"] as? Int else {
                throw FlintModelDirectoryError.invalidMetadata("missing group_size in quantization entry for \(key)")
            }
            switchMlpOverrides[key] = (bits: b, groupSize: g)
        }

        let routedBits: Int
        if switchMlpOverrides.isEmpty {
            // Case A: Official INT4 checkpoint
            routedBits = baseBits
        } else if switchMlpOverrides.count == 141 {
            // Case B: Mixed 3/4-bit checkpoint
            let foundKeys = Set(switchMlpOverrides.keys)
            guard foundKeys == expectedKeySet else {
                let missing = expectedKeySet.subtracting(foundKeys)
                let extra = foundKeys.subtracting(expectedKeySet)
                throw FlintModelDirectoryError.invalidMetadata("switch_mlp overrides key mismatch: \(missing.count) missing, \(extra.count) unexpected extra")
            }

            guard let firstOverride = switchMlpOverrides[expectedSwitchMlpKeys[0]] else {
                throw FlintModelDirectoryError.invalidMetadata("missing expected switch_mlp override")
            }
            guard firstOverride.bits == 3 || firstOverride.bits == 4 else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported routed expert bits: \(firstOverride.bits) (expected 3 or 4)")
            }
            guard firstOverride.groupSize == 64 else {
                throw FlintModelDirectoryError.invalidMetadata("unsupported routed expert group_size: \(firstOverride.groupSize) (expected 64)")
            }

            for key in expectedSwitchMlpKeys {
                let override = switchMlpOverrides[key]!
                guard override.bits == firstOverride.bits else {
                    throw FlintModelDirectoryError.invalidMetadata("inconsistent expert quantization: \(key) has bits=\(override.bits), expected \(firstOverride.bits)")
                }
                guard override.groupSize == 64 else {
                    throw FlintModelDirectoryError.invalidMetadata("inconsistent expert quantization: \(key) has group_size=\(override.groupSize), expected 64")
                }
            }
            routedBits = firstOverride.bits
        } else {
            // Incomplete or invalid mixed overrides (e.g. 1..140 overrides or extraneous projections)
            let foundKeys = Set(switchMlpOverrides.keys)
            let missing = expectedKeySet.subtracting(foundKeys)
            let extra = foundKeys.subtracting(expectedKeySet)
            throw FlintModelDirectoryError.invalidMetadata("incomplete or invalid mixed quantization: found \(switchMlpOverrides.count) switch_mlp overrides, expected 0 (INT4) or 141 (Mixed). Missing: \(missing.count), Extra: \(extra.count)")
        }

        return FlintConfig(
            modelType: modelType,
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
