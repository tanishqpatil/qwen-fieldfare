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

    // MARK: - Header Parsing & Validation Tests

    @Test func parsesValidHeaderAndMetadata() throws {
        let (url, totalBytes) = try Self.makeSyntheticShard(
            tensors: [
                "weight": (shape: [2, 2], dtype: "BF16", offsets: [0, 8], data: [1, 2, 3, 4, 5, 6, 7, 8])
            ],
            metadata: ["format": "pt", "author": "recursive"]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)
        #expect(shard.header.metadata["format"] == "pt")
        #expect(shard.header.metadata["author"] == "recursive")
        #expect(shard.header.tensors.count == 1)

        let info = try #require(shard.header.tensors["weight"])
        #expect(info.name == "weight")
        #expect(info.shape == [2, 2])
        #expect(info.dtype == "BF16")
        #expect(info.byteWidth == 2)
        #expect(info.elementCount == 4)
        #expect(info.byteCount == 8)
        #expect(info.dataOffsets == [0, 8])
        #expect(shard.header.fileByteCount == totalBytes)
    }

    @Test func rejectsNonStringMetadataValues() throws {
        let jsonStr = #"{"__metadata__":{"num":123},"t":{"shape":[1],"dtype":"BF16","data_offsets":[0,2]}}"#
        let headerData = jsonStr.data(using: .utf8)!
        var fileData = Data()
        var headerLenLE = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { fileData.append(contentsOf: $0) }
        fileData.append(headerData)
        fileData.append(Data([0, 0]))

        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-meta-\(UUID().uuidString).safetensors")
        try fileData.write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: tempFile)
        }
    }

    @Test func rejectsHeaderWithNulBytes() throws {
        var headerData = #"{"t":{"shape":[1],"dtype":"BF16","data_offsets":[0,2]}}"#.data(using: .utf8)!
        headerData.append(0x00) // NUL byte injection

        var fileData = Data()
        var headerLenLE = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { fileData.append(contentsOf: $0) }
        fileData.append(headerData)
        fileData.append(Data([0, 0]))

        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-nul-\(UUID().uuidString).safetensors")
        try fileData.write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: tempFile)
        }
    }

    @Test func rejectsUnsupportedDtype() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "bad.dtype": (shape: [2], dtype: "COMPLEX128", offsets: [0, 16], data: [UInt8](repeating: 0, count: 16))
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsArithmeticOverflowInShapeDimensions() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "overflow": (shape: [Int.max, 2], dtype: "BF16", offsets: [0, 4], data: [0, 0, 0, 0])
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func safeTensorDescriptorConstructionEnforcesAllInvariants() throws {
        // [Int.max, Int.max, 0] must not crash unchecked reduce
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsHeader.validatedElementCount(shape: [Int.max, Int.max, 0], name: "overflow")
        }

        // 1. Arithmetic overflow in shape element count throws without silent fallback to 0
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "overflow", shape: [Int.max, Int.max, 0], dtype: "BF16", byteWidth: 2, dataOffsets: [0, 0])
        }

        // 2. Negative dimensions throw
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "negDim", shape: [-1, 10], dtype: "BF16", byteWidth: 2, dataOffsets: [0, 20])
        }

        // 3. Invalid dataOffsets element count throws
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "badOffsetsCount", shape: [10], dtype: "BF16", byteWidth: 2, dataOffsets: [0])
        }
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "badOffsetsCount3", shape: [10], dtype: "BF16", byteWidth: 2, dataOffsets: [0, 10, 20])
        }

        // 4. Negative offsets throw
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "negOffset", shape: [10], dtype: "BF16", byteWidth: 2, dataOffsets: [-5, 15])
        }

        // 5. Reversed offsets throw
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "reversedOffset", shape: [10], dtype: "BF16", byteWidth: 2, dataOffsets: [20, 10])
        }

        // 6. Invalid byteWidth throws
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "zeroByteWidth", shape: [10], dtype: "BF16", byteWidth: 0, dataOffsets: [0, 0])
        }

        // 7. Shape byte size mismatch with dataOffsets throws
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintTensorInfo(name: "byteSizeMismatch", shape: [10], dtype: "BF16", byteWidth: 2, dataOffsets: [0, 10]) // expected 20 bytes, got 10
        }

        // 8. Scalar tensor [] has 1 element
        let scalarCount = try FlintSafetensorsHeader.validatedElementCount(shape: [], name: "scalar")
        #expect(scalarCount == 1)

        // 9. Valid zero-dim [10, 0, 5] has 0 elements
        let zeroCount = try FlintSafetensorsHeader.validatedElementCount(shape: [10, 0, 5], name: "zero")
        #expect(zeroCount == 0)

        // 10. Valid descriptor construction succeeds and sets all properties correctly
        let validInfo = try FlintTensorInfo(name: "valid", shape: [4, 4], dtype: "BF16", byteWidth: 2, dataOffsets: [0, 32])
        #expect(validInfo.name == "valid")
        #expect(validInfo.shape == [4, 4])
        #expect(validInfo.dtype == "BF16")
        #expect(validInfo.byteWidth == 2)
        #expect(validInfo.dataOffsets == [0, 32])
        #expect(validInfo.byteCount == 32)
        #expect(validInfo.elementCount == 16)
    }

    @Test func rejectsInvertedOffsets() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "inverted": (shape: [2], dtype: "BF16", offsets: [8, 4], data: [0, 0, 0, 0])
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsOffsetsExceedingShardPayload() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "oob": (shape: [2], dtype: "BF16", offsets: [0, 100], data: [0, 0, 0, 0]) // actual file only has 4 bytes
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsShapeByteSizeMismatchWithOffsets() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "mismatch": (shape: [4, 4], dtype: "BF16", offsets: [0, 8], data: [UInt8](repeating: 0, count: 8)) // expected 32 bytes, got 8
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsOverlappingTensorRanges() throws {
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "t1": (shape: [8], dtype: "BF16", offsets: [0, 16], data: [UInt8](repeating: 1, count: 16)),
                "t2": (shape: [8], dtype: "BF16", offsets: [8, 24], data: [UInt8](repeating: 2, count: 16)) // overlaps 8..<16
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsNonExistentFile() {
        let url = URL(fileURLWithPath: "/tmp/non-existent-\(UUID().uuidString).safetensors")
        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: url)
        }
    }

    @Test func rejectsNonRegularFile() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: tempDir)
        }
    }

    @Test func symlinkRejectedByONoFollow() throws {
        let (url, _) = try Self.makeSyntheticShard(tensors: [:])
        defer { try? FileManager.default.removeItem(at: url) }

        let symlinkURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-symlink-\(UUID().uuidString).safetensors")
        try FileManager.default.createSymbolicLink(at: symlinkURL, withDestinationURL: url)
        defer { try? FileManager.default.removeItem(at: symlinkURL) }

        #expect(throws: FlintSafetensorsError.self) {
            _ = try FlintSafetensorsFile(url: symlinkURL)
        }
    }

    @Test func fileReplacementDetectedOnDisk() throws {
        let (url, _) = try Self.makeSyntheticShard(tensors: [:])
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)
        try shard.verifyUnchanged()

        // Unlink and recreate a new file at the exact same path
        try FileManager.default.removeItem(at: url)
        let (newURL, _) = try Self.makeSyntheticShard(tensors: [:])
        defer { try? FileManager.default.removeItem(at: newURL) }
        try FileManager.default.moveItem(at: newURL, to: url)

        #expect(throws: FlintSafetensorsError.self) {
            try shard.verifyUnchanged()
        }
    }

    // MARK: - Bounded Reading & Buffer Safety Tests

    @Test func readsTensorSliceAndRejectsOutOfBounds() throws {
        let rawBytes: [UInt8] = (0..<32).map { UInt8($0) }
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "test.tensor": (shape: [16], dtype: "BF16", offsets: [0, 32], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let shard = try FlintSafetensorsFile(url: url)
        let payload = try shard.readTensorPayload("test.tensor")
        #expect([UInt8](payload) == rawBytes)

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

    @Test func bufferCapacityHandlingExactOversizedZeroAndSmall() throws {
        let rawBytes: [UInt8] = (0..<1024).map { UInt8($0 % 256) }
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "test.buf": (shape: [512], dtype: "BF16", offsets: [0, 1024], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }
        let shard = try FlintSafetensorsFile(url: url)

        // 1. Exact-sized destination buffer (1024 bytes)
        var exactDest = [UInt8](repeating: 0, count: 1024)
        try exactDest.withUnsafeMutableBytes { buf in
            try shard.readPayload(into: buf, range: 0..<1024)
        }
        #expect(exactDest == rawBytes)

        // 2. Oversized destination buffer (2048 bytes)
        var overDest = [UInt8](repeating: 0xEE, count: 2048)
        try overDest.withUnsafeMutableBytes { buf in
            try shard.readPayload(into: buf, range: 0..<1024)
        }
        #expect(Array(overDest.prefix(1024)) == rawBytes)
        #expect(overDest[1024] == 0xEE)

        // 3. Zero-length destination buffer with empty range (range.count == 0)
        var emptyDest = [UInt8]()
        try emptyDest.withUnsafeMutableBytes { buf in
            try shard.readPayload(into: buf, range: 100..<100)
        }

        // 4. Smaller-than-requested destination buffer (512 bytes for 1024 range)
        var smallDest = [UInt8](repeating: 0, count: 512)
        #expect(throws: FlintSafetensorsError.self) {
            try smallDest.withUnsafeMutableBytes { buf in
                try shard.readPayload(into: buf, range: 0..<1024)
            }
        }

        // 5. Direct tensor buffer reads
        var tensorDest = [UInt8](repeating: 0, count: 1024)
        try tensorDest.withUnsafeMutableBytes { buf in
            try shard.readTensorPayload("test.buf", into: buf)
        }
        #expect(tensorDest == rawBytes)

        var sliceDest = [UInt8](repeating: 0, count: 64)
        try sliceDest.withUnsafeMutableBytes { buf in
            try shard.readTensorSlice("test.buf", byteRange: 128..<192, into: buf)
        }
        #expect(sliceDest == Array(rawBytes[128..<192]))
    }

    @Test func readTensorChunksStreamsInBoundedChunksWithoutFullAllocation() throws {
        let rawBytes: [UInt8] = (0..<65536).map { UInt8($0 % 256) }
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "large.tensor": (shape: [32768], dtype: "BF16", offsets: [0, 65536], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }
        let shard = try FlintSafetensorsFile(url: url)

        // Valid bounded chunked streaming succeeds and preserves exact bytes
        var accumulated = [UInt8]()
        var lastOffset = 0
        try shard.readTensorChunks("large.tensor", chunkSize: 4096) { chunk, offset in
            #expect(offset == lastOffset)
            accumulated.append(contentsOf: chunk)
            lastOffset += chunk.count
        }
        #expect(accumulated == rawBytes)

        // Safeguard A: chunkSize exceeding default maxChunkSize (16 MiB) is rejected
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorChunks("large.tensor", chunkSize: 32 * 1024 * 1024) { _, _ in }
        }

        // Negative or zero chunk size is rejected
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorChunks("large.tensor", chunkSize: 0) { _, _ in }
        }
        #expect(throws: FlintSafetensorsError.self) {
            try shard.readTensorChunks("large.tensor", chunkSize: -1) { _, _ in }
        }

        // Custom tight maxChunkSize clamps and rejects requests exceeding limit
        let tightShard = try FlintSafetensorsFile(url: url, maxChunkSize: 8192)
        #expect(tightShard.maxChunkSize == 8192)
        #expect(throws: FlintSafetensorsError.self) {
            try tightShard.readTensorChunks("large.tensor", chunkSize: 16384) { _, _ in }
        }
    }

    @Test func dataReturningReadsEnforceAllocationLimit() throws {
        let rawBytes: [UInt8] = (0..<2048).map { UInt8($0 % 256) }
        let (url, _) = try Self.makeSyntheticShard(
            tensors: [
                "medium.tensor": (shape: [1024], dtype: "BF16", offsets: [0, 2048], data: rawBytes)
            ]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        // Open with a tight limit of 1024 bytes
        let cappedShard = try FlintSafetensorsFile(url: url, maxAllocatedBytes: 1024)
        // 512 bytes succeeds
        let slice = try cappedShard.readTensorSlice("medium.tensor", byteRange: 0..<512)
        #expect(slice.count == 512)

        // 2048 bytes exceeds 1024 limit and throws requestedAllocationExceedsLimit
        #expect(throws: FlintSafetensorsError.self) {
            _ = try cappedShard.readTensorPayload("medium.tensor")
        }

        // Caller-buffer reading works even when payload exceeds maxAllocatedBytes
        var callerBuf = [UInt8](repeating: 0, count: 2048)
        try callerBuf.withUnsafeMutableBytes { buf in
            try cappedShard.readTensorPayload("medium.tensor", into: buf)
        }
        #expect(callerBuf == rawBytes)
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

            // Verify physical zero-gap and zero-trailing-bytes invariant on live shard
            let ranges = shard.header.tensors.values.map { $0.dataOffsets }.sorted { $0[0] < $1[0] }
            #expect(ranges.first?[0] == 0)
            for i in 1..<ranges.count {
                #expect(ranges[i - 1][1] == ranges[i][0])
            }
            let payloadBytes = shard.header.fileByteCount - shard.header.payloadOffset
            #expect(ranges.last?[1] == payloadBytes)
        }
        #expect(totalTensors == 1751)

        // 2. Cross-validate physical tensor headers against config metadata (bijection check)
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

        // Helper to read tensor into buffer
        func readIntoBuffer(name: String, byteRange: Range<Int>, destination: UnsafeMutableRawBufferPointer) throws {
            guard let shardName = validated.weightMap[name], let shard = shardFiles[shardName] else {
                throw FlintSafetensorsError.tensorNotFound(name: name)
            }
            try shard.readTensorSlice(name, byteRange: byteRange, into: destination)
        }

        // 3. Independent Reference Verification (matched against Python / MLX reference)

        // Item 1: Small metadata-backed tensor: model.layers.2.self_attn.softmax_offset (64 bytes)
        let soData = try readSlice(name: "model.layers.2.self_attn.softmax_offset", byteRange: 0..<64)
        #expect(soData.count == 64)
        #expect(Self.sha256Hex(soData) == "e55d79b0bd25bc2d585bc4bf6706ae3d6d62e2561c9b0ec8f9918041f6404404")

        // Item 2: Attention norm: model.layers.0.input_layernorm.weight
        // Item 2a: Prefix slice (128 bytes)
        let inDataSlice = try readSlice(name: "model.layers.0.input_layernorm.weight", byteRange: 0..<128)
        #expect(inDataSlice.count == 128)
        #expect(Self.sha256Hex(inDataSlice) == "f33f5b184b24b4e62c37c757ee2d90396173e9ef2ef28af130b266ca021cc4a5")

        // Item 2b: Full tensor (4,096 bytes)
        let inDataFull = try readSlice(name: "model.layers.0.input_layernorm.weight", byteRange: 0..<4096)
        #expect(inDataFull.count == 4096)
        #expect(Self.sha256Hex(inDataFull) == "a47a5d39863526d5a24983779dda01852c4ea1f6d6989b695da57f18aa7b24a0")

        // Item 3: INT8 embedding payload range: model.embed_tokens.weight (256 bytes)
        let embData = try readSlice(name: "model.embed_tokens.weight", byteRange: 0..<256)
        #expect(embData.count == 256)
        #expect(Self.sha256Hex(embData) == "c5b8b890109befaf2efd23d462ece554855d2c73d1c14301a8c62afff75bed71")

        // Item 4: First routed-expert record: layer 2, expert 0 gate_proj.weight (393,216 bytes)
        let exp0Data = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<393216)
        #expect(exp0Data.count == 393216)
        #expect(Self.sha256Hex(exp0Data) == "a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820")

        // Item 4b: Test buffer-based read into caller-allocated memory for expert 0
        var exp0Buf = [UInt8](repeating: 0, count: 393216)
        try exp0Buf.withUnsafeMutableBytes { buf in
            try readIntoBuffer(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<393216, destination: buf)
        }
        #expect(Self.sha256Hex(Data(exp0Buf)) == "a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820")

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

        // Item 7: Quantization scales: model.layers.2.mlp.switch_mlp.gate_proj.scales
        // Item 7a: Prefix slice (64 bytes)
        let scDataSlice = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.scales", byteRange: 0..<64)
        #expect(scDataSlice.count == 64)
        #expect(Self.sha256Hex(scDataSlice) == "7cc18d29413197ee959987c119bbb6e0ed834af8af8107cb3e9e937d9ee267d7")

        // Item 7b: Expert 0 full scales (32,768 bytes)
        let scDataFull = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.scales", byteRange: 0..<32768)
        #expect(scDataFull.count == 32768)
        #expect(Self.sha256Hex(scDataFull) == "91243a6f1cd6e9489624c85e9780297562c8f94ea9aec390d02eae69b774cb40")

        // Item 8: Quantization biases: model.layers.2.mlp.switch_mlp.gate_proj.biases
        // Item 8a: Prefix slice (64 bytes)
        let biDataSlice = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.biases", byteRange: 0..<64)
        #expect(biDataSlice.count == 64)
        #expect(Self.sha256Hex(biDataSlice) == "4629cee3a9d13041dafcf227f7ac56e5c2388a2619ed197595e176a1fd1bab2d")

        // Item 8b: Expert 0 full biases (32,768 bytes)
        let biDataFull = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.biases", byteRange: 0..<32768)
        #expect(biDataFull.count == 32768)
        #expect(Self.sha256Hex(biDataFull) == "86e7916072ec4e247330d049e2712cce082873727610bfaa9007f74256f4373e")

        // Item 9: Out-of-bounds range access throws
        let totalGateBytes = 256 * 393216
        #expect(throws: FlintSafetensorsError.self) {
            _ = try readSlice(name: "model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<(totalGateBytes + 1))
        }
    }
}
