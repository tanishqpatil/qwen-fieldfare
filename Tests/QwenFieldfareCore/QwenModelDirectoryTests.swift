import Foundation
import Testing
@testable import QwenFieldfareCore

@Suite struct QwenModelDirectoryTests {
    @Test func validatesCompleteDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("{}", named: "config.json", in: root)
        try write("{}", named: "tokenizer.json", in: root)
        try write("{}", named: "tokenizer_config.json", in: root)
        try write("shard", named: "model-00001-of-00001.safetensors", in: root)
        try write(#"{"weight_map":{"tensor":"model-00001-of-00001.safetensors"}}"#,
                  named: "model.safetensors.index.json", in: root)
        #expect(try QwenModelDirectory.validate(path: root.path) == root.standardizedFileURL.path)
    }

    @Test func rejectsMissingReferencedShard() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("{}", named: "config.json", in: root)
        try write("{}", named: "tokenizer.json", in: root)
        try write("{}", named: "tokenizer_config.json", in: root)
        try write(#"{"weight_map":{"tensor":"missing.safetensors"}}"#,
                  named: "model.safetensors.index.json", in: root)
        #expect(throws: QwenModelDirectoryError.missingFile("missing.safetensors")) {
            try QwenModelDirectory.validate(path: root.path)
        }
    }

    @Test func rejectsShardSymlinkOutsideModelDirectory() throws {
        let root = try temporaryDirectory()
        let outside = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            try write("{}", named: name, in: root)
        }
        try write("shard", named: "external.safetensors", in: outside)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("shard.safetensors"),
            withDestinationURL: outside.appendingPathComponent("external.safetensors"))
        try write(#"{"weight_map":{"tensor":"shard.safetensors"}}"#,
                  named: "model.safetensors.index.json", in: root)
        #expect(throws: QwenModelDirectoryError.self) {
            try QwenModelDirectory.validate(path: root.path)
        }
    }

    @Test func rejectsOversizedHeaderAndOutOfBoundsTensor() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let shard = root.appendingPathComponent("shard.safetensors")
        let loader = QwenResidentWeights(checkpointDirectory: root.path,
                                         weightMap: ["tensor": "shard.safetensors"])
        var length = UInt64(100_000_001).littleEndian
        try Data(bytes: &length, count: 8).write(to: shard)
        #expect(throws: (any Error).self) { try loader.getTensorInfo(name: "tensor") }

        let header = Data(#"{"tensor":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.utf8)
        length = UInt64(header.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(header)
        try data.write(to: shard)
        #expect(throws: (any Error).self) { try loader.getTensorInfo(name: "tensor") }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-model-directory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ text: String, named name: String, in root: URL) throws {
        try Data(text.utf8).write(to: root.appendingPathComponent(name))
    }
}
