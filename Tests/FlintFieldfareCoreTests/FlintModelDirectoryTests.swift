import Foundation
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintModelDirectoryTests {
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

    @Test func rejectsInvalidConfigurationValues() throws {
        let validJson = """
        {
            "architectures": ["FlintForCausalLM"],
            "hidden_size": 1024,
            "intermediate_size": 6144,
            "num_hidden_layers": 49,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 150144,
            "layer_types": \(Array(repeating: "sliding_attention", count: 49).description)
        }
        """
        #expect(throws: FlintModelDirectoryError.self) {
            try FlintModelDirectory.parseAndValidateConfig(data: Data(validJson.utf8))
        }
    }

    @Test func validatesConfigWith14FullAttentionLayers() throws {
        var layerTypes = [String]()
        let fullIndices: Set<Int> = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        for i in 0..<49 {
            layerTypes.append(fullIndices.contains(i) ? "full_attention" : "sliding_attention")
        }

        let json = """
        {
            "architectures": ["FlintForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 49,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 150144,
            "sliding_window": 513,
            "num_experts": 256,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 1024,
            "router_score_function": "sigmoid",
            "layer_types": \(layerTypes.description),
            "no_rope_layer_indices": \(Array(fullIndices).sorted().description),
            "mlp_only_layers": [0, 1],
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "model.layers.2.mlp.switch_mlp.gate_proj": { "bits": 3 }
            }
        }
        """
        let config = try FlintModelDirectory.parseAndValidateConfig(data: Data(json.utf8))
        #expect(config.hiddenSize == 2048)
        #expect(config.numHiddenLayers == 49)
        #expect(config.fullAttentionLayerIndices.count == 14)
        #expect(config.slidingAttentionLayerIndices.count == 35)
        #expect(config.noRopeLayerIndices.count == 14)
        #expect(config.moeLayerIndices.count == 47)
        #expect(config.numExperts == 256)
        #expect(config.numExpertsPerTok == 6)
        #expect(config.routedExpertBits == 3)
    }

    @Test func validatesRealLocalCheckpointIfPresent() throws {
        let localModelPath = "/Users/tanishq/models/flint-mixed-3-4bit"
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: localModelPath, isDirectory: &isDir), isDir.boolValue else {
            return
        }

        let validated = try FlintModelDirectory.validate(path: localModelPath)
        #expect(validated.config.hiddenSize == 2048)
        #expect(validated.config.numHiddenLayers == 49)
        #expect(validated.config.fullAttentionLayerIndices.count == 14)
        #expect(validated.config.slidingAttentionLayerIndices.count == 35)
        #expect(validated.config.routedExpertBits == 3)
        #expect(validated.shards.count == 4)
        #expect(validated.weightMap["model.embed_tokens.weight"] != nil)
        #expect(validated.weightMap["lm_head.weight"] != nil)
    }
}
