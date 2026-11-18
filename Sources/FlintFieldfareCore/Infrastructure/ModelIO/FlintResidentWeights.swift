import Foundation
import Metal

/// A single loaded resident tensor combining metadata with its authoritative memory buffer.
public struct FlintResidentTensor: Sendable {
    public let spec: FlintResidentTensorSpec
    public let buffer: FlintResidentBuffer

    public var name: String { spec.name }
    public var category: FlintResidentCategory { spec.category }
    public var layerIndex: Int? { spec.layerIndex }
    public var shape: [Int] { spec.shape }
    public var dtype: String { spec.dtype }
    public var byteWidth: Int { spec.byteWidth }
    public var byteCount: Int { buffer.byteCount }
    public var readOnlyPointer: UnsafeRawBufferPointer { buffer.readOnlyPointer }
    public var baseAddress: UnsafeRawPointer? { buffer.baseAddress }
    public var metalBuffer: MTLBuffer? { buffer.metalBuffer }

    public init(spec: FlintResidentTensorSpec, buffer: FlintResidentBuffer) {
        self.spec = spec
        self.buffer = buffer
    }
}

/// Structured weights for a single layer's multi-head attention block.
public struct FlintLayerAttentionWeights: Sendable {
    public let layerIndex: Int
    public let inputLayerNorm: FlintResidentTensor
    public let postAttentionLayerNorm: FlintResidentTensor

    public let qWeight: FlintResidentTensor
    public let qScales: FlintResidentTensor
    public let qBiases: FlintResidentTensor

    public let kWeight: FlintResidentTensor
    public let kScales: FlintResidentTensor
    public let kBiases: FlintResidentTensor

    public let vWeight: FlintResidentTensor
    public let vScales: FlintResidentTensor
    public let vBiases: FlintResidentTensor

    public let oWeight: FlintResidentTensor
    public let oScales: FlintResidentTensor
    public let oBiases: FlintResidentTensor

    public let qNorm: FlintResidentTensor
    public let kNorm: FlintResidentTensor
    public let softmaxOffset: FlintResidentTensor
}

/// Structured weights for a dense MLP block (layers 0 and 1).
public struct FlintDenseMLPWeights: Sendable {
    public let layerIndex: Int

    public let gateWeight: FlintResidentTensor
    public let gateScales: FlintResidentTensor
    public let gateBiases: FlintResidentTensor

    public let upWeight: FlintResidentTensor
    public let upScales: FlintResidentTensor
    public let upBiases: FlintResidentTensor

    public let downWeight: FlintResidentTensor
    public let downScales: FlintResidentTensor
    public let downBiases: FlintResidentTensor
}

/// Structured resident weights for an MoE layer (layers 2..48).
/// Contains the router gate and shared expert weights.
/// Strictly excludes routed experts (which are streamed dynamically).
public struct FlintMoEResidentWeights: Sendable {
    public let layerIndex: Int
    public let routerGate: FlintResidentTensor

    public let sharedGateWeight: FlintResidentTensor
    public let sharedGateScales: FlintResidentTensor
    public let sharedGateBiases: FlintResidentTensor

    public let sharedUpWeight: FlintResidentTensor
    public let sharedUpScales: FlintResidentTensor
    public let sharedUpBiases: FlintResidentTensor

    public let sharedDownWeight: FlintResidentTensor
    public let sharedDownScales: FlintResidentTensor
    public let sharedDownBiases: FlintResidentTensor
}

/// Structured global embeddings.
public struct FlintEmbeddingWeights: Sendable {
    public let weight: FlintResidentTensor
    public let scales: FlintResidentTensor
    public let biases: FlintResidentTensor
}

/// Structured LM head projections.
public struct FlintLMHeadWeights: Sendable {
    public let weight: FlintResidentTensor
    public let scales: FlintResidentTensor
    public let biases: FlintResidentTensor
}

/// Immutable, thread-safe container owning the single authoritative memory allocations
/// for all permanently resident weights of Recursive Flint 40B-A2.5B.
public final class FlintResidentWeights: @unchecked Sendable {
    public let tensors: [String: FlintResidentTensor]
    public let totalByteCount: Int
    public let tensorCount: Int
    public let categoryByteCounts: [FlintResidentCategory: Int]

    public let embeddings: FlintEmbeddingWeights
    public let lmHead: FlintLMHeadWeights
    public let outputNorm: FlintResidentTensor

    private let layerAttentionMap: [Int: FlintLayerAttentionWeights]
    private let denseMLPMap: [Int: FlintDenseMLPWeights]
    private let moeResidentMap: [Int: FlintMoEResidentWeights]

    public init(tensors: [String: FlintResidentTensor], manifest: FlintResidentManifest) throws {
        self.tensors = tensors
        self.totalByteCount = tensors.values.reduce(0) { $0 + $1.byteCount }
        self.tensorCount = tensors.count

        var categoryTotals = [FlintResidentCategory: Int]()
        for cat in FlintResidentCategory.allCases {
            categoryTotals[cat] = 0
        }
        for tensor in tensors.values {
            categoryTotals[tensor.category, default: 0] += tensor.byteCount
        }
        self.categoryByteCounts = categoryTotals

        func get(_ name: String) throws -> FlintResidentTensor {
            guard let t = tensors[name] else {
                throw FlintSafetensorsError.tensorNotFound(name: name)
            }
            return t
        }

        // Global embeddings
        self.embeddings = FlintEmbeddingWeights(
            weight: try get("model.embed_tokens.weight"),
            scales: try get("model.embed_tokens.scales"),
            biases: try get("model.embed_tokens.biases")
        )

        // LM head
        self.lmHead = FlintLMHeadWeights(
            weight: try get("lm_head.weight"),
            scales: try get("lm_head.scales"),
            biases: try get("lm_head.biases")
        )

        // Output norm
        self.outputNorm = try get("model.norm.weight")

        // Layer attention (all 49 layers)
        var attMap = [Int: FlintLayerAttentionWeights]()
        for layer in 0..<49 {
            attMap[layer] = FlintLayerAttentionWeights(
                layerIndex: layer,
                inputLayerNorm: try get("model.layers.\(layer).input_layernorm.weight"),
                postAttentionLayerNorm: try get("model.layers.\(layer).post_attention_layernorm.weight"),
                qWeight: try get("model.layers.\(layer).self_attn.q_proj.weight"),
                qScales: try get("model.layers.\(layer).self_attn.q_proj.scales"),
                qBiases: try get("model.layers.\(layer).self_attn.q_proj.biases"),
                kWeight: try get("model.layers.\(layer).self_attn.k_proj.weight"),
                kScales: try get("model.layers.\(layer).self_attn.k_proj.scales"),
                kBiases: try get("model.layers.\(layer).self_attn.k_proj.biases"),
                vWeight: try get("model.layers.\(layer).self_attn.v_proj.weight"),
                vScales: try get("model.layers.\(layer).self_attn.v_proj.scales"),
                vBiases: try get("model.layers.\(layer).self_attn.v_proj.biases"),
                oWeight: try get("model.layers.\(layer).self_attn.o_proj.weight"),
                oScales: try get("model.layers.\(layer).self_attn.o_proj.scales"),
                oBiases: try get("model.layers.\(layer).self_attn.o_proj.biases"),
                qNorm: try get("model.layers.\(layer).self_attn.q_norm.weight"),
                kNorm: try get("model.layers.\(layer).self_attn.k_norm.weight"),
                softmaxOffset: try get("model.layers.\(layer).self_attn.softmax_offset")
            )
        }
        self.layerAttentionMap = attMap

        // Dense MLPs (layers 0 & 1)
        var dMLPMap = [Int: FlintDenseMLPWeights]()
        for layer in [0, 1] {
            dMLPMap[layer] = FlintDenseMLPWeights(
                layerIndex: layer,
                gateWeight: try get("model.layers.\(layer).mlp.gate_proj.weight"),
                gateScales: try get("model.layers.\(layer).mlp.gate_proj.scales"),
                gateBiases: try get("model.layers.\(layer).mlp.gate_proj.biases"),
                upWeight: try get("model.layers.\(layer).mlp.up_proj.weight"),
                upScales: try get("model.layers.\(layer).mlp.up_proj.scales"),
                upBiases: try get("model.layers.\(layer).mlp.up_proj.biases"),
                downWeight: try get("model.layers.\(layer).mlp.down_proj.weight"),
                downScales: try get("model.layers.\(layer).mlp.down_proj.scales"),
                downBiases: try get("model.layers.\(layer).mlp.down_proj.biases")
            )
        }
        self.denseMLPMap = dMLPMap

        // MoE resident (layers 2..48)
        var moeMap = [Int: FlintMoEResidentWeights]()
        for layer in 2..<49 {
            moeMap[layer] = FlintMoEResidentWeights(
                layerIndex: layer,
                routerGate: try get("model.layers.\(layer).mlp.gate.weight"),
                sharedGateWeight: try get("model.layers.\(layer).mlp.shared_expert.gate_proj.weight"),
                sharedGateScales: try get("model.layers.\(layer).mlp.shared_expert.gate_proj.scales"),
                sharedGateBiases: try get("model.layers.\(layer).mlp.shared_expert.gate_proj.biases"),
                sharedUpWeight: try get("model.layers.\(layer).mlp.shared_expert.up_proj.weight"),
                sharedUpScales: try get("model.layers.\(layer).mlp.shared_expert.up_proj.scales"),
                sharedUpBiases: try get("model.layers.\(layer).mlp.shared_expert.up_proj.biases"),
                sharedDownWeight: try get("model.layers.\(layer).mlp.shared_expert.down_proj.weight"),
                sharedDownScales: try get("model.layers.\(layer).mlp.shared_expert.down_proj.scales"),
                sharedDownBiases: try get("model.layers.\(layer).mlp.shared_expert.down_proj.biases")
            )
        }
        self.moeResidentMap = moeMap
    }

    public subscript(name: String) -> FlintResidentTensor? {
        tensors[name]
    }

    public func require(_ name: String) throws -> FlintResidentTensor {
        guard let t = tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        return t
    }

    public func attentionWeights(forLayer layer: Int) -> FlintLayerAttentionWeights? {
        layerAttentionMap[layer]
    }

    public func denseMLPWeights(forLayer layer: Int) -> FlintDenseMLPWeights? {
        denseMLPMap[layer]
    }

    public func moeResidentWeights(forLayer layer: Int) -> FlintMoEResidentWeights? {
        moeResidentMap[layer]
    }
}
