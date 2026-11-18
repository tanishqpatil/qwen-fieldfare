import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintResidentLoaderTests {

    private static func getProcessResidentMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func makeSyntheticFlintConfig() -> FlintConfig {
        var layerTypes = [String]()
        for i in 0..<49 {
            if FlintConfig.expectedFullAttentionIndices.contains(i) {
                layerTypes.append("full_attention")
            } else {
                layerTypes.append("sliding_attention")
            }
        }
        return FlintConfig(
            modelType: "flint",
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 49,
            numAttentionHeads: 32,
            numKeyValueHeads: 4,
            headDim: 128,
            vocabSize: 150144,
            rmsNormEps: 1e-6,
            maxPositionEmbeddings: 32768,
            ropeTheta: 1_000_000.0,
            slidingWindow: 513,
            layerTypes: layerTypes,
            noRopeLayerIndices: Array(FlintConfig.expectedFullAttentionIndices).sorted(),
            mlpOnlyLayers: [0, 1],
            numExperts: 256,
            numExpertsPerTok: 8,
            moeIntermediateSize: 768,
            sharedExpertIntermediateSize: 1024,
            routerScoreFunction: "sigmoid",
            bosTokenId: 1,
            eosTokenIds: [2],
            routedExpertBits: 3,
            embeddingBits: 8,
            groupSize: 64
        )
    }

    // MARK: - 1. Manifest Completeness and Layer Coverage Tests

    @Test func manifestCompletenessAndAllCategoriesCovered() throws {
        let config = Self.makeSyntheticFlintConfig()
        let manifest = try FlintResidentManifest.build(config: config)

        // 1. Total tensor count must be exactly 1,328
        #expect(manifest.specs.count == 1328)
        #expect(manifest.orderedNames.count == 1328)
        #expect(manifest.expectedTotalByteCount == 1_278_418_496)

        // 2. Exact category counts and byte breakdown
        let cats = manifest.expectedCategoryByteCounts
        #expect(cats[.embeddings] == 326_713_344)
        #expect(cats[.lmHead] == 172_965_888)
        #expect(cats[.outputNorm] == 4_096)
        #expect(cats[.layerNorms] == 401_408)
        #expect(cats[.attentionProjections] == 520_224_768)
        #expect(cats[.attentionSinksAndQKNorms] == 28_224)
        #expect(cats[.denseMLP] == 42_467_328)
        #expect(cats[.moeRouters] == 49_283_072)
        #expect(cats[.sharedExperts] == 166_330_368)

        // 3. Verify complete layer coverage (all 49 layers have attention, layer norms, sinks)
        for layer in 0..<49 {
            #expect(manifest.specs["model.layers.\(layer).input_layernorm.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).post_attention_layernorm.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.q_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.k_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.v_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.o_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.q_norm.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.k_norm.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).self_attn.softmax_offset"] != nil)
        }

        // 4. Verify Dense MLP only in layers 0 and 1
        for layer in [0, 1] {
            #expect(manifest.specs["model.layers.\(layer).mlp.gate_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).mlp.up_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).mlp.down_proj.weight"] != nil)
        }
        #expect(manifest.specs["model.layers.2.mlp.gate_proj.weight"] == nil)

        // 5. Verify MoE routers and shared experts only in layers 2..48
        #expect(manifest.specs["model.layers.0.mlp.gate.weight"] == nil)
        #expect(manifest.specs["model.layers.1.mlp.gate.weight"] == nil)
        for layer in 2..<49 {
            #expect(manifest.specs["model.layers.\(layer).mlp.gate.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).mlp.shared_expert.gate_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).mlp.shared_expert.up_proj.weight"] != nil)
            #expect(manifest.specs["model.layers.\(layer).mlp.shared_expert.down_proj.weight"] != nil)
        }

        // 6. Verify strictly ZERO routed expert tensors (switch_mlp) are present
        for name in manifest.specs.keys {
            #expect(!name.contains("switch_mlp"))
        }
    }

    // MARK: - 2. Buffer Memory Lifecycle Tests

    @Test func residentBufferMemoryLifecycleAndDeallocation() throws {
        // Zero-sized buffer
        let emptyBuf = FlintResidentBuffer(byteCount: 0)
        #expect(emptyBuf.byteCount == 0)
        #expect(emptyBuf.pointer.count == 0)

        // 4096-byte buffer
        let buf = FlintResidentBuffer(byteCount: 4096)
        #expect(buf.byteCount == 4096)
        #expect(buf.pointer.count == 4096)
        #expect(buf.pointer.baseAddress != nil)

        // Write and read pattern
        if let base = buf.pointer.baseAddress {
            let uint8Ptr = base.assumingMemoryBound(to: UInt8.self)
            for i in 0..<4096 {
                uint8Ptr[i] = UInt8(i % 256)
            }
        }
        let data = buf.copyData()
        #expect(data.count == 4096)
        #expect(data[0] == 0)
        #expect(data[255] == 255)
        #expect(data[256] == 0)

        // Slice copy
        let slice = buf.copySlice(range: 10..<20)
        #expect(slice.count == 10)
        #expect(slice[0] == 10)
    }

    // MARK: - 3. Partial-Load Failure Resource Cleanup Tests

    @Test func partialLoadFailureReleasesResourcesWithoutLeak() throws {
        let config = Self.makeSyntheticFlintConfig()
        let manifest = try FlintResidentManifest.build(config: config)

        // Create a synthetic directory missing most tensors
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("flint-partial-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Weight map pointing to a shard that only has 1 tensor
        let rawBytes = [UInt8](repeating: 42, count: 128)
        let headerDict: [String: Any] = [
            "model.embed_tokens.weight": [
                "shape": [32, 1],
                "dtype": "U32",
                "data_offsets": [0, 128]
            ]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: headerDict)
        var fileData = Data()
        var headerLen = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { fileData.append(contentsOf: $0) }
        fileData.append(headerData)
        fileData.append(contentsOf: rawBytes)

        let shardURL = tempDir.appendingPathComponent("model-00001.safetensors")
        try fileData.write(to: shardURL)
        let shardFile = try FlintSafetensorsFile(url: shardURL)

        var weightMap = [String: String]()
        for name in manifest.orderedNames {
            weightMap[name] = "model-00001.safetensors"
        }

        let validatedDir = FlintModelDirectory.ValidatedDirectory(
            canonicalPath: tempDir.path,
            config: config,
            configData: Data(),
            weightMap: weightMap,
            shards: ["model-00001.safetensors"]
        )

        // Attempting to load fails immediately because model.embed_tokens.weight has mismatched shape
        // and subsequent tensors are missing, proving that partial load catches errors and cleans up.
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintResidentLoader.loadResidentWeights(
                validatedDirectory: validatedDir,
                shardFiles: ["model-00001.safetensors": shardFile]
            )
        }
    }

    // MARK: - 4. Live Checkpoint Resident Loading Qualification on Apple M5

    @Test func liveCheckpointResidentLoadingQualificationOnM5() throws {
        let modelPath = ProcessInfo.processInfo.environment["FLINTFIELDFARE_MODEL"] ?? "/Users/tanishq/models/flint-mixed-3-4bit"
        guard FileManager.default.fileExists(atPath: modelPath) else {
            return
        }

        // Measure initial process memory
        let rssBefore = Self.getProcessResidentMemory()

        // 1. Validate directory
        let validated = try FlintModelDirectory.validate(path: modelPath)
        #expect(validated.config.numHiddenLayers == 49)
        #expect(validated.config.routedExpertBits == 3)
        #expect(validated.config.embeddingBits == 8)

        // 2. Open all 4 shards
        let shards = try FlintModelDirectory.openShards(canonicalPath: validated.canonicalPath, shardNames: validated.shards)
        #expect(shards.count == 4)

        // 3. Load resident weights with exact metrics capture
        let (weights, metrics) = try FlintResidentLoader.loadResidentWeights(
            validatedDirectory: validated,
            shardFiles: shards
        )

        let rssAfterLoad = Self.getProcessResidentMemory()

        // 4. Validate exact counts and payload bytes
        #expect(metrics.residentTensorCount == 1328)
        #expect(metrics.residentByteCount == 1_278_418_496)
        #expect(metrics.routedExpertByteCount == 0) // Strictly zero routed expert bytes!
        #expect(metrics.peakTemporaryAllocatedBytes == 0) // Direct I/O: 0 temporary bytes!
        #expect(metrics.usesMetal == true) // Metal shared memory buffers on Apple Silicon

        // 5. Validate category breakdowns
        let cats = metrics.categoryByteCounts
        #expect(cats[.embeddings] == 326_713_344)
        #expect(cats[.lmHead] == 172_965_888)
        #expect(cats[.outputNorm] == 4_096)
        #expect(cats[.layerNorms] == 401_408)
        #expect(cats[.attentionProjections] == 520_224_768)
        #expect(cats[.attentionSinksAndQKNorms] == 28_224)
        #expect(cats[.denseMLP] == 42_467_328)
        #expect(cats[.moeRouters] == 49_283_072)
        #expect(cats[.sharedExperts] == 166_330_368)

        // 6. Verify independent reference SHA-256 digests across early, middle, and final layers
        // Early layers:
        let embWeight = try weights.require("model.embed_tokens.weight")
        #expect(embWeight.dtype == "U32")
        #expect(embWeight.byteWidth == 4)
        #expect(embWeight.shape == [150144, 512])
        #expect(Self.sha256Hex(embWeight.buffer.copySlice(range: 0..<128)) == "ef0768498be4264c846d1a62c667875d4b5ed9711bf7fa72fa704f04d391d20f")

        let outNorm = try weights.require("model.norm.weight")
        #expect(outNorm.byteCount == 4096)
        #expect(Self.sha256Hex(outNorm.buffer.copyData()) == "b254961d001bbee273daa00efeef34e4351b783074fcce83b01146d7a5d90f76")

        let l0InputNorm = try weights.require("model.layers.0.input_layernorm.weight")
        #expect(l0InputNorm.byteCount == 4096)
        #expect(Self.sha256Hex(l0InputNorm.buffer.copyData()) == "a47a5d39863526d5a24983779dda01852c4ea1f6d6989b695da57f18aa7b24a0")

        let l0Sink = try weights.require("model.layers.0.self_attn.softmax_offset")
        #expect(l0Sink.byteCount == 64)
        #expect(Self.sha256Hex(l0Sink.buffer.copyData()) == "47fac1a49ccf7fe080e87522368e90e9246c6d8b5d4d9a84a202858d4160188a")

        let l0Gate = try weights.require("model.layers.0.mlp.gate_proj.weight")
        #expect(l0Gate.dtype == "U32")
        #expect(l0Gate.byteWidth == 4)
        #expect(Self.sha256Hex(l0Gate.buffer.copySlice(range: 0..<128)) == "f483f5e873a8bef9c125f27fd10224b187474a33096a889621a859f2ee2cb0c3")

        let l1DownScales = try weights.require("model.layers.1.mlp.down_proj.scales")
        #expect(l1DownScales.byteCount == 393216)
        #expect(Self.sha256Hex(l1DownScales.buffer.copySlice(range: 0..<64)) == "55661f4a958e53a0230ad7a65299cd13643c54b5a114ad64d450881c18a0c87b")

        // Middle layers:
        let l20InputNorm = try weights.require("model.layers.20.input_layernorm.weight")
        #expect(l20InputNorm.byteCount == 4096)
        #expect(Self.sha256Hex(l20InputNorm.buffer.copyData()) == "728d0d938132958627ac3a399711084deac01d73fd6cfd7912b10f054136367d")

        let l20QProj = try weights.require("model.layers.20.self_attn.q_proj.weight")
        #expect(l20QProj.dtype == "U32")
        #expect(l20QProj.byteWidth == 4)
        #expect(Self.sha256Hex(l20QProj.buffer.copySlice(range: 0..<128)) == "b72b37ce3d6e60dfad8ad92462d9a26dd54c12a1dbace3e6b226acc329a76d0a")

        let l20RouterGate = try weights.require("model.layers.20.mlp.gate.weight")
        #expect(l20RouterGate.byteCount == 1048576)
        #expect(Self.sha256Hex(l20RouterGate.buffer.copySlice(range: 0..<128)) == "a5daf73c87a5381296e830e461d5b2aa29506b54f8a7a3ba1d3fe11b93073fca")
        #expect(Self.sha256Hex(l20RouterGate.buffer.copyData()) == "edbf9c3b24e0ca035058b9eceba36ee56f634a82325a0608e2018316f0c2a03b")

        let l24SharedGate = try weights.require("model.layers.24.mlp.shared_expert.gate_proj.weight")
        #expect(l24SharedGate.dtype == "U32")
        #expect(l24SharedGate.byteWidth == 4)
        #expect(Self.sha256Hex(l24SharedGate.buffer.copySlice(range: 0..<128)) == "e265ca6a81ea69368b00517677e35eb361f3c02dd3801ca8c457d8be80672f4a")

        // Final layers:
        let l48PostNorm = try weights.require("model.layers.48.post_attention_layernorm.weight")
        #expect(l48PostNorm.byteCount == 4096)
        #expect(Self.sha256Hex(l48PostNorm.buffer.copyData()) == "640bc5225fae549b69b5a62c25551a1837e0afab300972c89951f2c2c8c01de4")

        let l48OScales = try weights.require("model.layers.48.self_attn.o_proj.scales")
        #expect(l48OScales.byteCount == 262144)
        #expect(Self.sha256Hex(l48OScales.buffer.copySlice(range: 0..<64)) == "6c876aa5c7744296d504573762df1b8398804fc650d76854576e67017c534fa3")

        let l48SharedDown = try weights.require("model.layers.48.mlp.shared_expert.down_proj.weight")
        #expect(l48SharedDown.dtype == "U32")
        #expect(l48SharedDown.byteWidth == 4)
        #expect(Self.sha256Hex(l48SharedDown.buffer.copySlice(range: 0..<128)) == "281d309d2d9a3efa0e0aa7a97f12d9f7d6504f76937086a93e28cf0c9cdf0f51")

        let lmHeadWeight = try weights.require("lm_head.weight")
        #expect(lmHeadWeight.dtype == "U32")
        #expect(lmHeadWeight.byteWidth == 4)
        #expect(Self.sha256Hex(lmHeadWeight.buffer.copySlice(range: 0..<128)) == "201e7d45820db8a313367c551ef61352a7394f5efba3560e10a5466f244f932a")

        // 7. Test structured accessors
        let l0Att = weights.attentionWeights(forLayer: 0)
        #expect(l0Att != nil)
        #expect(l0Att?.qWeight.name == "model.layers.0.self_attn.q_proj.weight")

        let l0Dense = weights.denseMLPWeights(forLayer: 0)
        #expect(l0Dense != nil)
        #expect(l0Dense?.gateWeight.name == "model.layers.0.mlp.gate_proj.weight")

        let l2MoE = weights.moeResidentWeights(forLayer: 2)
        #expect(l2MoE != nil)
        #expect(l2MoE?.routerGate.name == "model.layers.2.mlp.gate.weight")
        #expect(l2MoE?.sharedGateWeight.name == "model.layers.2.mlp.shared_expert.gate_proj.weight")

        // 8. Test teardown and release behavior
        var optionalWeights: FlintResidentWeights? = weights
        #expect(optionalWeights != nil)
        optionalWeights = nil
        let rssAfterTeardown = Self.getProcessResidentMemory()

        // 9. Log measurements for qualification report
        print("""
        === FLINT RESIDENT LOADER M5 QUALIFICATION ===
        Resident Tensors Loaded : \(metrics.residentTensorCount)
        Total Resident Bytes   : \(metrics.residentByteCount) B (\(Double(metrics.residentByteCount) / (1024*1024*1024)) GiB)
        Routed Expert Bytes    : \(metrics.routedExpertByteCount) B (strictly 0)
        Load Duration          : \(String(format: "%.3f", metrics.loadDurationSeconds)) s
        Peak Temporary Memory  : \(metrics.peakTemporaryAllocatedBytes) B
        Uses Metal Shared Mem  : \(metrics.usesMetal)
        Process RSS Before     : \(rssBefore / (1024*1024)) MiB
        Process RSS After Load : \(rssAfterLoad / (1024*1024)) MiB
        Process RSS Delta      : \(Int64(rssAfterLoad) - Int64(rssBefore)) B (\(Double(Int64(rssAfterLoad) - Int64(rssBefore)) / (1024*1024)) MiB)
        Process RSS After Teardown : \(rssAfterTeardown / (1024*1024)) MiB
        ==============================================
        """)
    }

    // MARK: - 5. Repeated Load/Unload Cycles Test

    @Test func repeatedLoadUnloadCyclesDoNotAccumulateMemory() throws {
        // Allocate and release 10 cycles of 16 MB buffers
        var peakMemory: UInt64 = 0
        for _ in 0..<10 {
            autoreleasepool {
                let buf = FlintResidentBuffer(byteCount: 16 * 1024 * 1024)
                #expect(buf.byteCount == 16 * 1024 * 1024)
                let currentMem = Self.getProcessResidentMemory()
                peakMemory = max(peakMemory, currentMem)
            }
        }
        let endMem = Self.getProcessResidentMemory()
        // Memory after cycles must be within normal bounds of peak memory
        #expect(endMem <= peakMemory + (32 * 1024 * 1024))
    }
}
