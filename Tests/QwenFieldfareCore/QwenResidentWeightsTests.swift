import Darwin
import Foundation
import Testing
@testable import QwenFieldfareCore

@Suite struct QwenResidentWeightsTests {
    private let embedding = "model.language_model.embed_tokens.weight"
    private let header = #"{"padding":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"norm":{"dtype":"BF16","shape":[2],"data_offsets":[3,7]},"model.language_model.embed_tokens.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[7,31]},"empty":{"dtype":"BF16","shape":[0],"data_offsets":[31,31]}}"#

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-resident-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func payload() -> Data {
        var data = Data([0xaa,0xbb,0xcc])
        for bits: UInt16 in [0x3f80,0xc000] {
            var value = bits.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        for value: Float in [1,-2,3.5,4,-5,6] {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func write(in root: URL) throws {
        let bytes = Data(header.utf8)
        var length = UInt64(bytes.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(bytes); data.append(payload())
        try data.write(to: root.appendingPathComponent("shard.safetensors"))
    }

    private func reader(_ root: URL, shard: String = "shard.safetensors") -> QwenResidentWeights {
        QwenResidentWeights(checkpointDirectory: root.path,
                            weightMap: ["norm":shard, embedding:shard, "padding":shard, "empty":shard, "absent":shard])
    }

    @Test func readsCorrectDataRelativeOffsetsAndFloatValues() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try write(in: root)
        let weights = reader(root)
        #expect(try weights.loadTensor(name: "norm") == [1,-2])
        #expect(try weights.loadTensor(name: embedding) == [1,-2,3.5,4,-5,6])
        #expect(try weights.loadTensor(name: "empty").isEmpty)
        let info = try weights.getTensorInfo(name: "norm")
        #expect(info.headerSize == header.utf8.count)
        #expect(info.info.dataOffsets == [3,7])
        #expect(info.shardPath == root.appendingPathComponent("shard.safetensors").resolvingSymlinksInPath().path)
        #expect(try weights.loadEmbedTokensRows(tokenIds: [2,0,2], hiddenSize: 2) == [-5,6,1,-2,-5,6])
        #expect(try weights.loadEmbedTokensRows(tokenIds: [], hiddenSize: 2).isEmpty)
    }

    @Test func rejectsInvalidRowsShapesAndUnsupportedDecodeTypes() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try write(in: root); let weights = reader(root)
        for token in [-1,Int.min,3,Int.max] {
            #expect(throws: QwenSafetensorsError.self) { try weights.loadEmbedTokensRows(tokenIds: [token], hiddenSize: 2) }
        }
        for hidden in [0,-1,1,Int.max] {
            #expect(throws: QwenSafetensorsError.self) { try weights.loadEmbedTokensRows(tokenIds: [0], hiddenSize: hidden) }
        }
        #expect(throws: QwenSafetensorsError.unsupportedDtype(name: "padding", dtype: "U8")) {
            try weights.loadTensor(name: "padding")
        }
        #expect(throws: QwenSafetensorsError.self) { try weights.getTensorInfo(name: "not-mapped") }
        #expect(throws: QwenSafetensorsError.self) { try weights.getTensorInfo(name: "absent") }
        #expect(try weights.loadTensor(name: "norm") == [1,-2])
    }

    @Test func rejectsEmbeddingRowByteOverflowWithoutAllocation() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let json = Data("{\"\(embedding)\":{\"dtype\":\"BF16\",\"shape\":[0,\(Int.max)],\"data_offsets\":[0,0]}}".utf8)
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8); data.append(json)
        try data.write(to: root.appendingPathComponent("shard.safetensors"))
        #expect(throws: QwenSafetensorsError.arithmeticOverflow(name: embedding)) {
            try reader(root).loadEmbedTokensRows(tokenIds: [], hiddenSize: Int.max)
        }
    }

    @Test func supportsInternalLinksAndPinsTargetWhenLinkIsRetargeted() throws {
        let root = try directory(); let outside = try directory()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        try write(in: root); try write(in: outside)
        let link = root.appendingPathComponent("link.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root.appendingPathComponent("shard.safetensors"))
        let weights = reader(root, shard: "link.safetensors")
        let descriptor = try weights.withTensor(name: "norm") { file, _, _ in file.descriptor }
        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside.appendingPathComponent("shard.safetensors"))
        // Cached ownership reads the original internal target, never follows the new link.
        #expect(try weights.loadTensor(name: "norm") == [1,-2])
        #expect(try weights.withTensor(name: "norm") { file, _, _ in file.descriptor } == descriptor)
        #expect(throws: QwenModelDirectoryError.self) { try reader(root, shard: "link.safetensors").loadTensor(name: "norm") }
    }

    @Test(arguments: ["", "../escape.safetensors","/tmp/escape.safetensors","..",".","shard.safetensors\0extra"])
    func rejectsUnsafeShardNames(shard: String) throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: QwenModelDirectoryError.self) { try reader(root, shard: shard).loadTensor(name: "norm") }
    }

    @Test func rejectsFIFOsDirectoriesAndMissingFiles() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let shard = root.appendingPathComponent("shard.safetensors")
        try #require(mkfifo(shard.path, 0o600) == 0)
        #expect(throws: QwenSafetensorsError.notRegularFile) { try reader(root).loadTensor(name: "norm") }
        try FileManager.default.removeItem(at: shard)
        try FileManager.default.createDirectory(at: shard, withIntermediateDirectories: false)
        #expect(throws: QwenSafetensorsError.notRegularFile) { try reader(root).loadTensor(name: "norm") }
        try FileManager.default.removeItem(at: shard)
        #expect(throws: QwenSafetensorsError.posix(operation: "open", code: ENOENT)) { try reader(root).loadTensor(name: "norm") }
    }

    @Test func rejectsMalformedHeadersAndCanRetryAfterRepair() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let weights = reader(root); let path = root.appendingPathComponent("shard.safetensors")
        var oversized = UInt64.max.littleEndian
        for invalid in [Data([1]),Data(bytes: &oversized, count: 8)] {
            try invalid.write(to: path)
            #expect(throws: QwenSafetensorsError.self) { try weights.loadTensor(name: "norm") }
        }
        let malformed = Data(#"{"norm":{"dtype":"BF16","shape":[1],"data_offsets":[0,9]}}"#.utf8)
        var length = UInt64(malformed.count).littleEndian
        var data = Data(bytes: &length, count: 8); data.append(malformed); data.append(Data([0,0]))
        try data.write(to: path)
        #expect(throws: QwenSafetensorsError.self) { try weights.loadTensor(name: "norm") }
        try write(in: root)
        #expect(try weights.loadTensor(name: "norm") == [1,-2])
    }

    @Test func mutationInvalidatesAndReleasesDescriptorBeforeExplicitRetry() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try write(in: root); let weights = reader(root)
        let fd = try weights.withTensor(name: "norm") { file, _, _ in file.descriptor }
        let writer = try FileHandle(forWritingTo: root.appendingPathComponent("shard.safetensors"))
        try writer.truncate(atOffset: 0); try writer.close()
        #expect(throws: QwenSafetensorsError.self) { try weights.loadTensor(name: "norm") }
        #expect(fcntl(fd, F_GETFD) == -1)
        #expect(errno == EBADF)
        try write(in: root)
        #expect(try weights.loadTensor(name: "norm") == [1,-2])
    }

    @Test func replacedPathNeverSilentlySuppliesReplacementBytes() throws {
        let root = try directory(); let replacementRoot = try directory()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: replacementRoot) }
        try write(in: root); try write(in: replacementRoot)
        let weights = reader(root); _ = try weights.getTensorInfo(name: "norm")
        let replacement = replacementRoot.appendingPathComponent("shard.safetensors")
        let writer = try FileHandle(forWritingTo: replacement)
        try writer.seek(toOffset: UInt64(8 + header.utf8.count + 3))
        try writer.write(contentsOf: Data([0,0,0,0])); try writer.close()
        try #require(rename(replacement.path, root.appendingPathComponent("shard.safetensors").path) == 0)
        do {
            #expect(try weights.loadTensor(name: "norm") == [1,-2])
        } catch {
            // Namespace replacement may change the pinned inode's ctime.
            // Either old-descriptor data or a controlled error is acceptable;
            // reopening the replacement silently is not.
            #expect(error is QwenSafetensorsError)
        }
    }

    @Test func readerDeinitReleasesOwnedDescriptors() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try write(in: root)
        var weights: QwenResidentWeights? = reader(root)
        let descriptor = try weights!.withTensor(name: "norm") { file, _, _ in file.descriptor }
        weights = nil
        #expect(fcntl(descriptor, F_GETFD) == -1)
        #expect(errno == EBADF)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWENFIELDFARE_READER_CHECKPOINT"] != nil))
    func localCheckpointHeadersAndSmallTensorIntegration() throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWENFIELDFARE_READER_CHECKPOINT"])
        let weights = try QwenResidentCoreLoader.load(from: path)
        for shard in Set(weights.weightMap.values).sorted() {
            let name = try #require(weights.weightMap.keys.sorted().first { weights.weightMap[$0] == shard })
            let metadata = try weights.getTensorInfo(name: name)
            #expect(metadata.headerSize <= QwenSafetensorsHeader.maximumHeaderBytes)
            print("Validated resident shard \(shard): header \(metadata.headerSize) bytes")
        }
        let name = "language_model.model.norm.weight"
        #expect(try weights.getTensorInfo(name: name).info.shape == [2048])
        let values = try weights.loadTensor(name: name)
        #expect(values.count == 2048)
        #expect(values.allSatisfy { $0.isFinite })
        print("Resident integration: all shard headers plus one 2048-element norm; no full-model residency")
    }
}
