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
