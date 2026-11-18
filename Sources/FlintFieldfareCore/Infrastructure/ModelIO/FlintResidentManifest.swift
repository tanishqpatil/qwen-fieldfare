import Foundation

/// Categories of resident weights for the Recursive Flint 40B-A2.5B architecture.
public enum FlintResidentCategory: String, CaseIterable, Sendable {
    case embeddings
    case lmHead
    case outputNorm
    case layerNorms
    case attentionProjections
    case attentionSinksAndQKNorms
    case denseMLP
    case moeRouters
    case sharedExperts

    public var isPerLayer: Bool {
        switch self {
        case .embeddings, .lmHead, .outputNorm:
            return false
        default:
            return true
        }
    }
}

/// Specification for a single resident tensor.
public struct FlintResidentTensorSpec: Equatable, Sendable {
    public let name: String
    public let category: FlintResidentCategory
    public let layerIndex: Int?
    public let shape: [Int]
    public let dtype: String
    public let byteWidth: Int
    public let expectedByteCount: Int

    public init(
        name: String,
        category: FlintResidentCategory,
        layerIndex: Int? = nil,
        shape: [Int],
        dtype: String,
        byteWidth: Int
    ) {
        self.name = name
        self.category = category
        self.layerIndex = layerIndex
        self.shape = shape
        self.dtype = dtype
        self.byteWidth = byteWidth
        self.expectedByteCount = shape.reduce(1, *) * byteWidth
    }
}

/// Programmatic manifest of all 1,328 permanently resident tensors in Recursive Flint 40B-A2.5B.
/// Enforces complete layer coverage (all 49 layers), dense MLP layers 0 & 1, MoE routers and shared experts
/// in layers 2..48, attention sinks/norms, and strictly zero routed expert tensors (switch_mlp).
public struct FlintResidentManifest: Sendable {
    public static let expectedResidentTensorCount = 1_328
    public static let expectedResidentTotalBytes = 1_278_418_496 // 1.1906 GiB

    public let specs: [String: FlintResidentTensorSpec]
    public let orderedNames: [String]
    public let expectedTotalByteCount: Int
    public let expectedCategoryByteCounts: [FlintResidentCategory: Int]

    public init(specs: [String: FlintResidentTensorSpec], orderedNames: [String]) {
        self.specs = specs
        self.orderedNames = orderedNames
        self.expectedTotalByteCount = specs.values.reduce(0) { $0 + $1.expectedByteCount }

        var categoryTotals = [FlintResidentCategory: Int]()
        for cat in FlintResidentCategory.allCases {
            categoryTotals[cat] = 0
        }
        for spec in specs.values {
            categoryTotals[spec.category, default: 0] += spec.expectedByteCount
        }
        self.expectedCategoryByteCounts = categoryTotals
    }

    /// Constructs the authoritative resident manifest from the validated model configuration.
    public static func build(config: FlintConfig) throws -> FlintResidentManifest {
        // Enforce architectural invariants
        guard config.numHiddenLayers == 49 else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported numHiddenLayers: \(config.numHiddenLayers) (expected 49)")
        }
        guard config.mlpOnlyLayers == [0, 1] else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported mlpOnlyLayers: \(config.mlpOnlyLayers) (expected [0, 1])")
        }
        guard config.numExperts == 256 else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported numExperts: \(config.numExperts) (expected 256)")
        }
        guard config.numExpertsPerTok == 6 else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported numExpertsPerTok: \(config.numExpertsPerTok) (expected 6)")
        }
        guard config.moeIntermediateSize == 512 else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported moeIntermediateSize: \(config.moeIntermediateSize) (expected 512)")
        }
        guard config.sharedExpertIntermediateSize == 1024 else {
            throw FlintSafetensorsError.invalidMetadata("Unsupported sharedExpertIntermediateSize: \(config.sharedExpertIntermediateSize) (expected 1024)")
        }

        var specs = [String: FlintResidentTensorSpec]()
        var orderedNames = [String]()

        func add(_ spec: FlintResidentTensorSpec) {
            specs[spec.name] = spec
            orderedNames.append(spec.name)
        }

        // 1. Global Embeddings (3 tensors) - Derived from validated config.embeddingBits
        let embeddingCols: Int
        switch config.embeddingBits {
        case 8:
            // 8-bit quantized: 1 byte per weight, 4 weights per U32 -> hiddenSize / 4 = 512
            embeddingCols = config.hiddenSize / 4
        case 4:
            // 4-bit quantized: 0.5 bytes per weight, 8 weights per U32 -> hiddenSize / 8 = 256
            embeddingCols = config.hiddenSize / 8
        default:
            throw FlintSafetensorsError.unsupportedDtype(
                name: "model.embed_tokens",
                dtype: "\(config.embeddingBits)-bit embedding quantization"
            )
        }

        add(FlintResidentTensorSpec(
            name: "model.embed_tokens.weight",
            category: .embeddings,
            shape: [config.vocabSize, embeddingCols],
            dtype: "U32",
            byteWidth: 4
        ))
        add(FlintResidentTensorSpec(
            name: "model.embed_tokens.scales",
            category: .embeddings,
            shape: [config.vocabSize, config.hiddenSize / config.groupSize],
            dtype: "BF16",
            byteWidth: 2
        ))
        add(FlintResidentTensorSpec(
            name: "model.embed_tokens.biases",
            category: .embeddings,
            shape: [config.vocabSize, config.hiddenSize / config.groupSize],
            dtype: "BF16",
            byteWidth: 2
        ))

        // 2. LM Head (3 tensors)
        add(FlintResidentTensorSpec(
            name: "lm_head.weight",
            category: .lmHead,
            shape: [150144, 256],
            dtype: "U32",
            byteWidth: 4
        ))
        add(FlintResidentTensorSpec(
            name: "lm_head.scales",
            category: .lmHead,
            shape: [150144, 32],
            dtype: "BF16",
            byteWidth: 2
        ))
        add(FlintResidentTensorSpec(
            name: "lm_head.biases",
            category: .lmHead,
            shape: [150144, 32],
            dtype: "BF16",
            byteWidth: 2
        ))

        // 3. Final Output Norm (1 tensor)
        add(FlintResidentTensorSpec(
            name: "model.norm.weight",
            category: .outputNorm,
            shape: [2048],
            dtype: "BF16",
            byteWidth: 2
        ))

        // 4. Per-Layer Attention & Layer Norms across ALL 49 layers (0..<49)
        for layer in 0..<config.numHiddenLayers {
            // Layer Norms (2 tensors per layer)
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).input_layernorm.weight",
                category: .layerNorms,
                layerIndex: layer,
                shape: [2048],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).post_attention_layernorm.weight",
                category: .layerNorms,
                layerIndex: layer,
                shape: [2048],
                dtype: "BF16",
                byteWidth: 2
            ))

            // Attention Projections: q, k, v, o (12 tensors per layer)
            // q_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.q_proj.weight",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [4096, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.q_proj.scales",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [4096, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.q_proj.biases",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [4096, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // k_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.k_proj.weight",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.k_proj.scales",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.k_proj.biases",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // v_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.v_proj.weight",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.v_proj.scales",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.v_proj.biases",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [512, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // o_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.o_proj.weight",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [2048, 512],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.o_proj.scales",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [2048, 64],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.o_proj.biases",
                category: .attentionProjections,
                layerIndex: layer,
                shape: [2048, 64],
                dtype: "BF16",
                byteWidth: 2
            ))

            // Attention Sinks and Q/K Norms (3 tensors per layer)
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.q_norm.weight",
                category: .attentionSinksAndQKNorms,
                layerIndex: layer,
                shape: [128],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.k_norm.weight",
                category: .attentionSinksAndQKNorms,
                layerIndex: layer,
                shape: [128],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).self_attn.softmax_offset",
                category: .attentionSinksAndQKNorms,
                layerIndex: layer,
                shape: [32],
                dtype: "BF16",
                byteWidth: 2
            ))
        }

        // 5. Dense Feed-Forward in Layers 0 and 1 (9 tensors per layer * 2 = 18 tensors)
        for layer in config.mlpOnlyLayers {
            // gate_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.gate_proj.weight",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.gate_proj.scales",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.gate_proj.biases",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // up_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.up_proj.weight",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.up_proj.scales",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.up_proj.biases",
                category: .denseMLP,
                layerIndex: layer,
                shape: [6144, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // down_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.down_proj.weight",
                category: .denseMLP,
                layerIndex: layer,
                shape: [2048, 768],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.down_proj.scales",
                category: .denseMLP,
                layerIndex: layer,
                shape: [2048, 96],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.down_proj.biases",
                category: .denseMLP,
                layerIndex: layer,
                shape: [2048, 96],
                dtype: "BF16",
                byteWidth: 2
            ))
        }

        // 6. MoE Router Gates & Shared Experts in Layers 2..48 (10 tensors per layer * 47 = 470 tensors)
        for layer in config.moeLayerIndices {
            // Router Gate (1 tensor per MoE layer)
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.gate.weight",
                category: .moeRouters,
                layerIndex: layer,
                shape: [256, 2048],
                dtype: "BF16",
                byteWidth: 2
            ))

            // Shared Expert (9 tensors per MoE layer)
            // gate_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.gate_proj.weight",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.gate_proj.scales",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.gate_proj.biases",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // up_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.up_proj.weight",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 256],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.up_proj.scales",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 32],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.up_proj.biases",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [1024, 32],
                dtype: "BF16",
                byteWidth: 2
            ))

            // down_proj
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.down_proj.weight",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [2048, 128],
                dtype: "U32",
                byteWidth: 4
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.down_proj.scales",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [2048, 16],
                dtype: "BF16",
                byteWidth: 2
            ))
            add(FlintResidentTensorSpec(
                name: "model.layers.\(layer).mlp.shared_expert.down_proj.biases",
                category: .sharedExperts,
                layerIndex: layer,
                shape: [2048, 16],
                dtype: "BF16",
                byteWidth: 2
            ))
        }

        guard specs.count == expectedResidentTensorCount else {
            throw FlintSafetensorsError.invalidMetadata(
                "Resident manifest tensor count \(specs.count) != expected \(expectedResidentTensorCount)"
            )
        }

        return FlintResidentManifest(specs: specs, orderedNames: orderedNames)
    }

    /// Cross-checks the manifest against physical Safetensors shard headers to ensure every tensor exists,
    /// has matching shape, dtype, and byte size, and verifies that ZERO routed expert tensors are present.
    public func validateAgainstShards(
        _ shardFiles: [String: FlintSafetensorsFile],
        weightMap: [String: String]
    ) throws {
        for (name, spec) in specs {
            guard let shardName = weightMap[name] else {
                throw FlintSafetensorsError.tensorNotFound(name: "Manifest tensor '\(name)' missing from weight_map")
            }
            guard let shard = shardFiles[shardName] else {
                throw FlintModelDirectoryError.missingFile("Referenced shard file '\(shardName)' not open")
            }
            guard let tensorInfo = shard.header.tensors[name] else {
                throw FlintSafetensorsError.tensorNotFound(name: "Tensor '\(name)' declared in \(shardName) not found in header")
            }

            guard tensorInfo.shape == spec.shape else {
                throw FlintSafetensorsError.invalidTensor(
                    name: name,
                    reason: "Physical shape \(tensorInfo.shape) does not match manifest shape \(spec.shape)"
                )
            }
            guard tensorInfo.dtype == spec.dtype else {
                throw FlintSafetensorsError.invalidTensor(
                    name: name,
                    reason: "Physical dtype \(tensorInfo.dtype) does not match manifest dtype \(spec.dtype)"
                )
            }
            guard tensorInfo.byteCount == spec.expectedByteCount else {
                throw FlintSafetensorsError.invalidTensor(
                    name: name,
                    reason: "Physical byte count \(tensorInfo.byteCount) does not match expected \(spec.expectedByteCount)"
                )
            }
            // Strict invariant: no routed expert weights in manifest
            guard !name.contains("switch_mlp") else {
                throw FlintSafetensorsError.invalidTensor(
                    name: name,
                    reason: "Routed expert tensor (switch_mlp) must never be included in resident manifest"
                )
            }
        }
    }
}
