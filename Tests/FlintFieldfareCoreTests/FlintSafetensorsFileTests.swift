import CryptoKit
import Foundation
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintSafetensorsFileTests {
    private static func makeSyntheticShard(
        tensors: [String: (shape: [Int], dtype: String, offsets: [Int], data: [UInt8])],
        metadata: [String: String]? = nil,
        headerBytePadding: Int = 0
    ) throws -> (url: URL, totalBytes: Int) {
        var headerDict: [String: Any] = [:]
        if let metadata = metadata {
            headerDict["__metadata__"] = metadata
        }

        var maxOffset = 0
        for (name, t) in tensors {
            headerDict[name] = [
                "shape": t.shape,
                "dtype": t.dtype,
                "data_offsets": t.offsets
            ]
            if t.offsets[1] > maxOffset {
                maxOffset = t.offsets[1]
            }
        }

        var headerJson = try JSONSerialization.data(withJSONObject: headerDict)
        if headerBytePadding > 0 {
            headerJson.append(Data(repeating: 0x20, count: headerBytePadding)) // space padding
        }

        var fileData = Data()
        var headerLenLE = UInt64(headerJson.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { fileData.append(contentsOf: $0) }
        fileData.append(headerJson)

        // Write payload data at exact offsets
        var payloadData = Data(repeating: 0, count: maxOffset)
        for (_, t) in tensors {
            let start = t.offsets[0]
            let count = t.offsets[1] - t.offsets[0]
            if count > 0 && !t.data.isEmpty {
                payloadData.replaceSubrange(start..<(start + min(count, t.data.count)), with: t.data.prefix(count))
            }
        }
        fileData.append(payloadData)

        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-shard-\(UUID().uuidString).safetensors")
        try fileData.write(to: tempFile)
        return (tempFile, fileData.count)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Header Length Tests

    @Test func headerLengthDecodesCorrectly() throws {
        var prefix = Data()
        var len = UInt64(1024).littleEndian
        withUnsafeBytes(of: &len) { prefix.append(contentsOf: $0) }
        let decoded = try FlintSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: 2048)
        #expect(decoded == 1024)
    }

    @Test func headerLengthRejectsShortPrefix() {
        let prefix = Data([0x01, 0x02, 0x03])
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: 100)
        }
    }

    @Test func headerLengthRejectsZeroLength() {
        let prefix = Data(repeating: 0, count: 8)
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: 100)
        }
    }

    @Test func headerLengthRejectsExceedingFileByteCount() {
        var prefix = Data()
        var len = UInt64(500).littleEndian
        withUnsafeBytes(of: &len) { prefix.append(contentsOf: $0) }
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: 200) // 500 > 200 - 8
        }
    }

    @Test func headerLengthRejectsExceedingMaxBound() {
        var prefix = Data()
        var len = UInt64(FlintSafetensorsHeader.maximumHeaderBytes + 1).littleEndian
        withUnsafeBytes(of: &len) { prefix.append(contentsOf: $0) }
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: 200_000_000)
        }
    }

    // MARK: - Header Parsing and Validation Tests

    @Test func parsesValidHeaderAndMetadata() throws {
        let testData: [UInt8] = [1, 2, 3, 4, 5, 6, 7, 8]
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "layer.0.bias": (shape: [4], dtype: "BF16", offsets: [0, 8], data: testData)
            ],
            metadata: ["format": "pt", "author": "RecursiveSI"]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)
        #expect(shard.header.metadata["format"] == "pt")
        #expect(shard.header.metadata["author"] == "RecursiveSI")
        #expect(shard.header.tensors["layer.0.bias"] != nil)
        let t = shard.header.tensors["layer.0.bias"]!
        #expect(t.shape == [4])
        #expect(t.dtype == "BF16")
        #expect(t.byteWidth == 2)
        #expect(t.byteCount == 8)
        #expect(t.elementCount == 4)

        let readData = try shard.readTensorPayload("layer.0.bias")
        #expect([UInt8](readData) == testData)
    }

    @Test func rejectsHeaderWithNulBytes() throws {
        let headerData = Data([0x7B, 0x00, 0x7D]) // "{\0}"
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: headerData, fileByteCount: 100)
        }
    }

    @Test func rejectsNonStringMetadataValues() throws {
        let jsonStr = """
        {
            "__metadata__": {"version": 123},
            "tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        }
        """
        let data = jsonStr.data(using: .utf8)!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 100)
        }
    }

    @Test func rejectsUnsupportedDtype() throws {
        let jsonStr = """
        {
            "tensor": {"dtype": "COMPLEX128", "shape": [2], "data_offsets": [0, 16]}
        }
        """
        let data = jsonStr.data(using: .utf8)!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 100)
        }
    }

    @Test func rejectsInvertedOffsets() throws {
        let jsonStr = """
        {
            "tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [10, 4]}
        }
        """
        let data = jsonStr.data(using: .utf8)!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 100)
        }
    }

    @Test func rejectsOffsetsExceedingShardPayload() throws {
        let jsonStr = """
        {
            "tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        }
        """
        let data = jsonStr.data(using: .utf8)!
        // If fileByteCount is 8 + headerLen (i.e. payload size = 0), offset 4 exceeds payload!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 8 + data.count)
        }
    }

    @Test func rejectsShapeByteSizeMismatchWithOffsets() throws {
        let jsonStr = """
        {
            "tensor": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}
        }
        """ // 4 elements of BF16 (2 bytes each) = 8 bytes, but offsets declare 4 bytes
        let data = jsonStr.data(using: .utf8)!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 100)
        }
    }

    @Test func rejectsOverlappingTensorRanges() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "tensor_a": (shape: [4], dtype: "BF16", offsets: [0, 8], data: [0, 1, 2, 3, 4, 5, 6, 7]),
                "tensor_b": (shape: [4], dtype: "BF16", offsets: [4, 12], data: [8, 9, 10, 11, 12, 13, 14, 15]) // Overlaps 4..<8
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsArithmeticOverflowInShapeDimensions() throws {
        let jsonStr = """
        {
            "tensor": {"dtype": "BF16", "shape": [9223372036854775807, 2], "data_offsets": [0, 4]}
        }
        """
        let data = jsonStr.data(using: .utf8)!
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsHeader.parse(data: data, fileByteCount: 100)
        }
    }

    // MARK: - Resource Ownership & File Descriptor Safety

    @Test func rejectsNonRegularFile() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsFile(url: tempDir)
        }
    }

    @Test func rejectsNonExistentFile() {
        let nonExistent = URL(fileURLWithPath: "/tmp/non-existent-\(UUID().uuidString).safetensors")
        #expect(throws: FlintSafetensorsError.self) {
            try FlintSafetensorsFile(url: nonExistent)
        }
    }

    @Test func detectsFileModificationWhileOpen() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "tensor_a": (shape: [4], dtype: "BF16", offsets: [0, 8], data: [0, 1, 2, 3, 4, 5, 6, 7])
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)
        try shard.verifyUnchanged()

        // Mutate the backing file by truncating or writing to it
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([0xFF, 0xFF]))
        try handle.close()

        #expect(throws: FlintSafetensorsError.self) {
            try shard.verifyUnchanged()
        }
    }

    // MARK: - Bounded Range & Slice Reading

    @Test func readsTensorSliceAndRejectsOutOfBounds() throws {
        let rawBytes: [UInt8] = Array(0..<32)
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "test.tensor": (shape: [16], dtype: "BF16", offsets: [0, 32], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)

        // Read exact sub-slice: bytes 4..<12
        let sliceData = try shard.readTensorSlice("test.tensor", byteRange: 4..<12)
        #expect([UInt8](sliceData) == Array(rawBytes[4..<12]))

        // Empty slice returns empty Data
        let emptyData = try shard.readTensorSlice("test.tensor", byteRange: 8..<8)
        #expect(emptyData.isEmpty)

        // Out of bounds: upper bound > 32
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorSlice("test.tensor", byteRange: 0..<33)
        }

        // Out of bounds: negative lower bound
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorSlice("test.tensor", byteRange: -1..<10)
        }

        // Non-existent tensor
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorSlice("non.existent.tensor", byteRange: 0..<4)
        }
    }

    @Test func concurrentReadsDoNotRace() async throws {
        let rawBytes: [UInt8] = (0..<1024).map { UInt8($0 % 256) }
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "big.tensor": (shape: [512], dtype: "BF16", offsets: [0, 1024], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)

        // Spawn 10 concurrent asynchronous tasks reading disjoint slices
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                let start = i * 100
                let range = start..<(start + 100)
                group.addTask {
                    let slice = try shard.readTensorSlice("big.tensor", byteRange: range)
                    #expect([UInt8](slice) == Array(rawBytes[range]))
                }
            }
            try await group.waitForAll()
        }
    }

    // MARK: - Metadata Consistency & Cross-Validation Tests

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

    @Test func crossValidationRejectsContradictoryEmbeddingBits() throws {
        var dict = Self.makeValidConfigDict()
        // Contradictory: config embedding is 8-bit, but mlx_export declares 4-bit!
        dict["mlx_export"] = [
            "embedding_bits": 4,
            "routed_expert_bits": 3
        ]
        let configData = try JSONSerialization.data(withJSONObject: dict)
        let config = try FlintModelDirectory.parseAndValidateConfig(data: configData)

        let (url, _) = try Self.makeSyntheticShard(tensors: [:])
        defer { try? FileManager.default.removeItem(at: url) }
        let shardFile = try FlintSafetensorsFile(url: url)

        #expect(throws: FlintSafetensorsError.self) {
            try FlintModelDirectory.crossValidate(
                config: config,
                configData: configData,
                shardFiles: ["shard.safetensors": shardFile],
                weightMap: [:]
            )
        }
    }

    @Test func crossValidationRejectsContradictoryRoutedExpertBits() throws {
        var dict = Self.makeValidConfigDict()
        // Contradictory: config routed_expert_bits is 3, but mlx_export declares 4!
        dict["mlx_export"] = [
            "embedding_bits": 8,
            "routed_expert_bits": 4
        ]
        let configData = try JSONSerialization.data(withJSONObject: dict)
        let config = try FlintModelDirectory.parseAndValidateConfig(data: configData)

        let (url, _) = try Self.makeSyntheticShard(tensors: [:])
        defer { try? FileManager.default.removeItem(at: url) }
        let shardFile = try FlintSafetensorsFile(url: url)

        #expect(throws: FlintSafetensorsError.self) {
            try FlintModelDirectory.crossValidate(
                config: config,
                configData: configData,
                shardFiles: ["shard.safetensors": shardFile],
                weightMap: [:]
            )
        }
    }

    @Test func crossValidationRejectsPhysicalTensorShapeMismatch() throws {
        let dict = Self.makeValidConfigDict()
        let configData = try JSONSerialization.data(withJSONObject: dict)
        let config = try FlintModelDirectory.parseAndValidateConfig(data: configData)

        // Shard with a tensor having wrong shape: embed_tokens shape [10, 10] instead of [150144, 512]
        let (url, _) = try Self.makeSyntheticShard(tensors: [
            "model.embed_tokens.weight": (shape: [10, 10], dtype: "U32", offsets: [0, 400], data: [UInt8](repeating: 0, count: 400))
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let shardFile = try FlintSafetensorsFile(url: url)

        #expect(throws: FlintSafetensorsError.self) {
            try FlintModelDirectory.crossValidate(
                config: config,
                configData: configData,
                shardFiles: ["model.safetensors": shardFile],
                weightMap: ["model.embed_tokens.weight": "model.safetensors"]
            )
        }
    }

    // MARK: - Live Checkpoint & Independent SHA-256 Validation Tests

    @Test func validatesLiveCheckpointPhysicalInventoryAndIndependentBytes() throws {
        let envPath = ProcessInfo.processInfo.environment["FLINT_TEST_MODEL_DIR"]
        let defaultPath = "/Users/tanishq/models/flint-mixed-3-4bit"
        let candidatePath = envPath ?? defaultPath

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidatePath, isDirectory: &isDir), isDir.boolValue else {
            Issue.record("Live Flint checkpoint not found at \(candidatePath). Set FLINT_TEST_MODEL_DIR to validate.")
            return
        }

        let validated = try FlintModelDirectory.validate(path: candidatePath)
        #expect(validated.shards.count == 4)

        // Open all 4 physical shards
        let shardFiles = try FlintModelDirectory.openShards(
            canonicalPath: validated.canonicalPath,
            shardNames: validated.shards
        )
        #expect(shardFiles.count == 4)

        // 1. Shard inspection: verify total tensor count matches 1,751
        var totalTensors = 0
        for (_, shard) in shardFiles {
            totalTensors += shard.header.tensors.count
        }
        #expect(totalTensors == 1751)

        // 2. Cross-validate physical tensor headers against config metadata
        try FlintModelDirectory.crossValidate(
            config: validated.config,
            configData: validated.configData,
            shardFiles: shardFiles,
            weightMap: validated.weightMap
        )

        // Helper to read tensor slice from whichever shard contains it
        func readSlice(name: String, byteRange: Range<Int>) throws -> Data {
            guard let shardName = validated.weightMap[name], let shard = shardFiles[shardName] else {
                throw FlintSafetensorsError.tensorNotFound(name: name)
            }
            return try shard.readTensorSlice(name, byteRange: byteRange)
        }

        // 3. Independent Reference Verification (matched against Python / MLX reference)

        // Item 1: Small metadata-backed tensor: model.layers.2.self_attn.softmax_offset (64 bytes)
        let soData = try readSlice(name: "model.layers.2.self_attn.softmax_offset", byteRange: 0..<64)
        #expect(soData.count == 64)
        #expect(Self.sha256Hex(soData) == "e55d79b0bd25bc2d585bc4bf6706ae3d6d62e2561c9b0ec8f9918041f6404404")

        // Item 2: Attention norm slice: model.layers.0.input_layernorm.weight (128 bytes)
        let inData = try readSlice(name: "model.layers.0.input_layernorm.weight", byteRange: 0..<128)
        #expect(inData.count == 128)
        #expect(Self.sha256Hex(inData) == "f33f5b184b24b4e62c37c757ee2d90396173e9ef2ef28af130b266ca021cc4a5")

        // Item 3: INT8 embedding payload range: model.embed_tokens.weight (256 bytes)
        let embData = try readSlice(name: "model.embed_tokens.weight", byteRange: 0..<256)
        #expect(embData.count == 256)
        #expect(Self.sha256Hex(embData) == "c5b8b890109befaf2efd23d462ece554855d2c73d1c14301a8c62afff75bed71")

        // Item 4: First routed-expert record: layer 2, expert 0 gate_proj.weight (393,216 bytes)
        let exp0Data = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<393216)
        #expect(exp0Data.count == 393216)
        #expect(Self.sha256Hex(exp0Data) == "a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820")

        // Item 5: Middle routed-expert record: layer 20, expert 128 gate_proj.weight (393,216 bytes)
        let exp128Offset = 128 * 393216
        let exp128Data = try readSlice(name: "model.layers.20.mlp.switch_mlp.gate_proj.weight", byteRange: exp128Offset..<(exp128Offset + 393216))
        #expect(exp128Data.count == 393216)
        #expect(Self.sha256Hex(exp128Data) == "a3fcc07e846e6c78e5a57c7e89870e97bbb56afe2a0ddb7979043c127acbd678")

        // Item 6: Last routed-expert record: layer 48, expert 255 down_proj.weight (393,216 bytes)
        let exp255Offset = 255 * 393216
        let exp255Data = try readSlice(name: "model.layers.48.mlp.switch_mlp.down_proj.weight", byteRange: exp255Offset..<(exp255Offset + 393216))
        #expect(exp255Data.count == 393216)
        #expect(Self.sha256Hex(exp255Data) == "2e823253c72b37404f447e5656a08995746460f39b417474eeed19078d9fd15b")

        // Item 7: Quantization scales: model.layers.2.mlp.switch_mlp.gate_proj.scales (expert 0, 64 bytes)
        let scData = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.scales", byteRange: 0..<64)
        #expect(scData.count == 64)
        #expect(Self.sha256Hex(scData) == "7cc18d29413197ee959987c119bbb6e0ed834af8af8107cb3e9e937d9ee267d7")

        // Item 8: Quantization biases: model.layers.2.mlp.switch_mlp.gate_proj.biases (expert 0, 64 bytes)
        let biData = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.biases", byteRange: 0..<64)
        #expect(biData.count == 64)
        #expect(Self.sha256Hex(biData) == "4629cee3a9d13041dafcf227f7ac56e5c2388a2619ed197595e176a1fd1bab2d")

        // Item 9: Out-of-bounds range access throws
        let totalGateBytes = 256 * 393216
        #expect(throws: FlintSafetensorsError.self) {
            _ = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<(totalGateBytes + 1))
        }
    }
}
