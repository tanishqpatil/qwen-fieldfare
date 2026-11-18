import Foundation
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintModelDirectoryTests {
    private static func makeValidConfigDict() -> [String: Any] {
        var layerTypes = [String]()
        let fullIndices = FlintConfig.expectedFullAttentionIndices
        for i in 0..<49 {
            layerTypes.append(fullIndices.contains(i) ? "full_attention" : "sliding_attention")
        }

        var quant: [String: Any] = [
            "group_size": 64,
            "bits": 4,
            "model.embed_tokens": ["group_size": 64, "bits": 8]
        ]
        // Explicit 3-bit overrides across all 47 MoE layers
        for layer in 2..<49 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                quant["model.layers.\(layer).mlp.switch_mlp.\(proj)"] = [
                    "group_size": 64,
                    "bits": 3
                ]
            }
        }

        return [
            "architectures": ["FlintForCausalLM"],
            "model_type": "flint",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 49,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 150144,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "rope_theta": 10000.0,
            "sliding_window": 513,
            "hidden_act": "silu",
            "num_experts": 256,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 1024,
            "router_score_function": "sigmoid",
            "norm_topk_prob": true,
            "layer_types": layerTypes,
            "no_rope_layer_indices": Array(fullIndices).sorted(),
            "mlp_only_layers": [0, 1],
            "quantization": quant
        ]
    }

    @Test func rejectsNonDirectory() {
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.validate(path: "/nonexistent-directory-\(UUID().uuidString)")
        }
    }

    @Test func rejectsDirectoryMissingMetadataFiles() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        #expect(throws: FlintModelDirectoryError.missingFile("config.json")) {
            try FlintModelDirectory.validate(path: tempDir.path)
        }
    }

    @Test func validatesCompleteValidConfiguration() throws {
        let dict = Self.makeValidConfigDict()
        let data = try JSONSerialization.data(withJSONObject: dict)
        let config = try FlintModelDirectory.parseAndValidateConfig(data: data)

        #expect(config.hiddenSize == 2048)
        #expect(config.numHiddenLayers == 49)
        #expect(config.fullAttentionLayerIndices.count == 14)
        #expect(config.slidingAttentionLayerIndices.count == 35)
        #expect(config.noRopeLayerIndices.count == 14)
        #expect(config.moeLayerIndices.count == 47)
        #expect(config.numExperts == 256)
        #expect(config.numExpertsPerTok == 6)
        #expect(config.routedExpertBits == 3)
        #expect(config.embeddingBits == 8)
        #expect(config.groupSize == 64)
    }

    @Test func rejectsMissingMandatoryFields() throws {
        let mandatoryKeys = [
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "sliding_window", "num_experts", "num_experts_per_tok",
            "moe_intermediate_size", "shared_expert_intermediate_size",
            "layer_types", "no_rope_layer_indices", "mlp_only_layers", "quantization"
        ]

        for key in mandatoryKeys {
            var dict = Self.makeValidConfigDict()
            dict.removeValue(forKey: key)
            let data = try JSONSerialization.data(withJSONObject: dict)
            #expect(throws: FlintModelDirectoryError.self) {
                try FlintModelDirectory.parseAndValidateConfig(data: data)
            }
        }
    }

    @Test func rejectsUnsupportedLayerTypes() throws {
        var dict = Self.makeValidConfigDict()
        var types = dict["layer_types"] as! [String]
        types[2] = "gated_deltanet" // Unsupported architecture injection
        dict["layer_types"] = types

        let data = try JSONSerialization.data(withJSONObject: dict)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data)
        }
    }

    @Test func rejectsMismatchedFullAttentionArrangement() throws {
        var dict = Self.makeValidConfigDict()
        var types = dict["layer_types"] as! [String]
        // Swap layer 1 (full) with layer 2 (sliding)
        types[1] = "sliding_attention"
        types[2] = "full_attention"
        dict["layer_types"] = types

        let data = try JSONSerialization.data(withJSONObject: dict)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data)
        }
    }

    @Test func rejectsIncorrectExpertCountsAndDimensions() throws {
        // Wrong num_experts
        var dict1 = Self.makeValidConfigDict()
        dict1["num_experts"] = 128
        let data1 = try JSONSerialization.data(withJSONObject: dict1)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data1)
        }

        // Wrong num_experts_per_tok
        var dict2 = Self.makeValidConfigDict()
        dict2["num_experts_per_tok"] = 8
        let data2 = try JSONSerialization.data(withJSONObject: dict2)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data2)
        }

        // Wrong moe_intermediate_size
        var dict3 = Self.makeValidConfigDict()
        dict3["moe_intermediate_size"] = 1024
        let data3 = try JSONSerialization.data(withJSONObject: dict3)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data3)
        }
    }

    @Test func rejectsInconsistentExpertQuantizationOverrides() throws {
        var dict = Self.makeValidConfigDict()
        var quant = dict["quantization"] as! [String: Any]
        // Make layer 10 up_proj 4-bit while others are 3-bit
        quant["model.layers.10.mlp.switch_mlp.up_proj"] = [
            "group_size": 64,
            "bits": 4
        ]
        dict["quantization"] = quant

        let data = try JSONSerialization.data(withJSONObject: dict)
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: data)
        }
    }

    @Test func rejectsInvalidTensorIndexShardNames() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-test-shards-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Write required metadata files
        let validConfig = try JSONSerialization.data(withJSONObject: Self.makeValidConfigDict())
        try validConfig.write(to: tempDir.appendingPathComponent("config.json"))
        try "{}".data(using: .utf8)!.write(to: tempDir.appendingPathComponent("tokenizer.json"))
        try "{}".data(using: .utf8)!.write(to: tempDir.appendingPathComponent("tokenizer_config.json"))
        try "{{ prompt }}".data(using: .utf8)!.write(to: tempDir.appendingPathComponent("chat_template.jinja"))

        // Unsafe shard name attempting to escape directory
        let maliciousIndex: [String: Any] = [
            "weight_map": [
                "model.embed_tokens.weight": "../../../etc/passwd"
            ]
        ]
        let indexData = try JSONSerialization.data(withJSONObject: maliciousIndex)
        try indexData.write(to: tempDir.appendingPathComponent("model.safetensors.index.json"))

        #expect(throws: FlintModelDirectoryError.invalidIndex("unsafe shard name ../../../etc/passwd")) {
            try FlintModelDirectory.validate(path: tempDir.path)
        }
    }

    @Test func validatesLiveCheckpointWithConfigurableEnvironment() throws {
        let envPath = ProcessInfo.processInfo.environment["FLINT_TEST_MODEL_DIR"]
        let defaultPath = "/Users/tanishq/models/flint-mixed-3-4bit"
        let candidatePath = envPath ?? defaultPath

        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: candidatePath, isDirectory: &isDir) && isDir.boolValue

        if !exists {
            Issue.record("Live Flint checkpoint not found at \(candidatePath). Set FLINT_TEST_MODEL_DIR to validate.")
            return
        }

        let validated = try FlintModelDirectory.validate(path: candidatePath)
        #expect(validated.config.hiddenSize == 2048)
        #expect(validated.config.numHiddenLayers == 49)
        #expect(validated.config.fullAttentionLayerIndices.count == 14)
        #expect(validated.config.slidingAttentionLayerIndices.count == 35)
        #expect(validated.config.routedExpertBits == 3)
        #expect(validated.shards.count == 4)
        #expect(validated.weightMap["model.embed_tokens.weight"] != nil)
        #expect(validated.weightMap["lm_head.weight"] != nil)
        #expect(validated.weightMap["model.layers.48.mlp.switch_mlp.down_proj.weight"] != nil)
    }
}
