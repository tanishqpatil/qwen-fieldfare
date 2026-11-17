import Darwin
import Foundation
import Testing
@testable import QwenFieldfareCore

@Suite struct QwenSafetensorsHeaderTests {
    private func prefix(_ length: UInt64) -> Data {
        var value = length.littleEndian
        return Data(bytes: &value, count: 8)
    }

    private func parse(_ text: String, payloadBytes: Int = 0) throws -> QwenSafetensorsHeader {
        let data = Data(text.utf8)
        return try QwenSafetensorsHeader.parse(data, fileByteCount: 8 + data.count + payloadBytes)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-header-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ header: String, payload: Data = Data(), at url: URL) throws {
        let json = Data(header.utf8)
        var bytes = prefix(UInt64(json.count)); bytes.append(json); bytes.append(payload)
        try bytes.write(to: url)
    }

    @Test func validQuantizedMetadataAndOwnedDescriptor() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("shard.safetensors")
        let text = #"{"__metadata__":{"format":"mlx"},"weight":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]},"scales":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]},"norm":{"dtype":"F32","shape":[],"data_offsets":[20,24]}}   "#
        try write(text, payload: Data(repeating: 0, count: 24), at: path)
        var fd: Int32 = -1
        do {
            let file = try QwenSafetensorsFile(resolvedURL: path)
            fd = file.descriptor
            #expect(file.header.headerByteCount == text.utf8.count)
            #expect(file.header.payloadOffset == 8 + text.utf8.count)
            #expect(file.header.tensors["weight"]?.shape == [2,2])
            #expect(file.header.tensors["norm"]?.dtype == "F32")
            try file.verifyUnchanged()
            // Header validation uses positional reads and preserves FD position.
            #expect(lseek(fd, 0, SEEK_CUR) == 0)
        }
        #expect(fcntl(fd, F_GETFD) == -1)
        #expect(errno == EBADF)
    }

    @Test func headerLengthBoundariesAreCheckedBeforeAllocation() throws {
        let limit = QwenSafetensorsHeader.maximumHeaderBytes
        #expect(try QwenSafetensorsHeader.headerLength(prefix: prefix(1), fileByteCount: 9) == 1)
        #expect(try QwenSafetensorsHeader.headerLength(prefix: prefix(UInt64(limit)), fileByteCount: limit + 8) == limit)
        for invalid in [UInt64(0), UInt64(limit + 1), UInt64(Int.max), UInt64.max] {
            #expect(throws: QwenSafetensorsError.invalidHeaderLength) {
                try QwenSafetensorsHeader.headerLength(prefix: prefix(invalid), fileByteCount: Int.max)
            }
        }
        #expect(throws: QwenSafetensorsError.shortRead(expected: 8, actual: 7)) {
            try QwenSafetensorsHeader.headerLength(prefix: Data(repeating: 0, count: 7), fileByteCount: 8)
        }
        #expect(throws: QwenSafetensorsError.shortRead(expected: 5, actual: 4)) {
            try QwenSafetensorsHeader.headerLength(prefix: prefix(5), fileByteCount: 12)
        }
        // Handwritten bytes must be interpreted as LE, regardless of alignment.
        #expect(try QwenSafetensorsHeader.headerLength(prefix: Data([0,1,0,0,0,0,0,0]), fileByteCount: 264) == 256)
    }

    @Test(arguments: 0..<8) func rejectsTruncatedPrefix(bytes: Int) throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("short")
        try Data(repeating: 0, count: bytes).write(to: path)
        #expect(throws: QwenSafetensorsError.invalidFileSize) { try QwenSafetensorsFile(resolvedURL: path) }
    }

    @Test func rejectsTruncatedAndOversizedHeaderFiles() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("bad")
        var bytes = prefix(20); bytes.append(Data("{}".utf8)); try bytes.write(to: path)
        #expect(throws: QwenSafetensorsError.shortRead(expected: 20, actual: 2)) { try QwenSafetensorsFile(resolvedURL: path) }
        try prefix(100_000_001).write(to: path)
        #expect(throws: QwenSafetensorsError.invalidHeaderLength) { try QwenSafetensorsFile(resolvedURL: path) }
    }

    @Test(arguments: ["{", "[]", "null", "123", "\"text\""])
    func rejectsMalformedOrNonObjectJSON(text: String) {
        #expect(throws: QwenSafetensorsError.malformedHeader) { try parse(text) }
    }

    @Test func rejectsInvalidUTF8() {
        for bytes in [Data([0xff,0xfe,0xfd]), Data([0x7b,0,0x7d,0]), Data([0x7b,0x7d,0])] {
            #expect(throws: QwenSafetensorsError.malformedHeader) {
                try QwenSafetensorsHeader.parse(bytes, fileByteCount: 8 + bytes.count)
            }
        }
    }

    @Test(arguments: [
        #"{"t":null}"#, #"{"t":{"dtype":"BF16","shape":[],"data_offsets":[0]}}"#,
        #"{"t":{"dtype":42,"shape":[1],"data_offsets":[0,2]}}"#,
        #"{"t":{"dtype":"BF16","shape":"1","data_offsets":[0,2]}}"#,
        #"{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2,4]}}"#
    ]) func rejectsMissingOrMalformedTensorFields(text: String) {
        #expect(throws: QwenSafetensorsError.self) { try parse(text, payloadBytes: 4) }
    }

    @Test(arguments: ["true", "false", "-1", "1.0", "1e0", "1.5", "1e30", "null", "\"1\""])
    func rejectsNonIntegerDimensionsAndOffsets(value: String) {
        for field in ["shape", "data_offsets"] {
            let shape = field == "shape" ? "[\(value)]" : "[1]"
            let offsets = field == "data_offsets" ? "[\(value),2]" : "[0,2]"
            let text = "{\"t\":{\"dtype\":\"BF16\",\"shape\":\(shape),\"data_offsets\":\(offsets)}}"
            #expect(throws: QwenSafetensorsError.self) { try parse(text, payloadBytes: 2) }
        }
    }

    @Test(arguments: ["F8_UNKNOWN", "Q4", ""])
    func rejectsUnsupportedDtypes(dtype: String) {
        #expect(throws: QwenSafetensorsError.unsupportedDtype(name: "t", dtype: dtype)) {
            try parse("{\"t\":{\"dtype\":\"\(dtype)\",\"shape\":[0],\"data_offsets\":[0,0]}}")
        }
    }

    @Test func validatesScalarEmptyAndFixedWidthTypes() throws {
        for (dtype, width) in [("BOOL",1),("I8",1),("U8",1),("I16",2),("U16",2),("F16",2),("BF16",2),("I32",4),("U32",4),("F32",4),("I64",8),("U64",8),("F64",8)] {
            let text = "{\"t\":{\"dtype\":\"\(dtype)\",\"shape\":[],\"data_offsets\":[0,\(width)]}}"
            #expect(try parse(text, payloadBytes: width).tensors["t"]?.shape == [])
        }
        #expect(try parse("{\"t\":{\"dtype\":\"F64\",\"shape\":[\(Int.max),\(Int.max),0],\"data_offsets\":[0,0]}}").tensors["t"]?.shape.last == 0)
        #expect(try parse("{}").tensors.isEmpty)
    }

    @Test func rejectsShapeAndByteSizeOverflow() {
        for (dtype, shape) in [("U8","\(Int.max),2"),("F64","\(Int.max)"),("BF16","18446744073709551615")] {
            #expect(throws: QwenSafetensorsError.arithmeticOverflow(name: "t")) {
                try parse("{\"t\":{\"dtype\":\"\(dtype)\",\"shape\":[\(shape)],\"data_offsets\":[0,0]}}")
            }
        }
    }

    @Test(arguments: ["[2,1]", "[0,3]", "[0,9223372036854775807]", "[0,18446744073709551615]", "[0,1]"])
    func rejectsInvalidOrMismatchedPayloadRanges(offsets: String) {
        #expect(throws: QwenSafetensorsError.self) {
            try parse("{\"t\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":\(offsets)}}", payloadBytes: 2)
        }
    }

    @Test func validatesPayloadBoundaryAndRejectsOverlap() throws {
        let valid = #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#
        let header = try parse(valid, payloadBytes: 4)
        #expect(header.payloadOffset + header.tensors["b"]!.dataOffsets[1] == header.fileByteCount)
        let overlap = valid.replacingOccurrences(of: "[2,4]", with: "[1,3]")
        #expect(throws: QwenSafetensorsError.overlappingTensors) { try parse(overlap, payloadBytes: 4) }
        // This helper is not a canonical packing checker: gaps/trailing bytes are allowed.
        #expect(try parse(valid, payloadBytes: 8).fileByteCount > header.fileByteCount)
    }

    @Test func validatesLargestRepresentableAbsoluteFileBoundary() throws {
        func text(end: Int) -> String {
            "{\"t\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[\(end - 1),\(end)]}}"
        }
        let template = text(end: Int.max)
        let payloadOffset = template.utf8.count + 8
        let headerText = text(end: Int.max - payloadOffset)
        try #require(headerText.utf8.count == template.utf8.count)
        // A virtual size tests metadata arithmetic without creating/reading an
        // enormous payload or claiming that such a model allocation is allowed.
        let header = try QwenSafetensorsHeader.parse(Data(headerText.utf8), fileByteCount: Int.max)
        #expect(header.payloadOffset + header.tensors["t"]!.dataOffsets[1] == Int.max)
    }

    @Test func rejectsInvalidMetadataAndFileSize() {
        #expect(throws: QwenSafetensorsError.invalidMetadata) { try parse(#"{"__metadata__":{"bad":1}}"#) }
        #expect(throws: QwenSafetensorsError.invalidFileSize) {
            try QwenSafetensorsHeader.parse(Data("{}".utf8), fileByteCount: Int.min)
        }
    }

    @Test func rejectsUnsafeFileKindsAndRawLeafLinks() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: QwenSafetensorsError.notRegularFile) { try QwenSafetensorsFile(resolvedURL: root) }
        let fifo = root.appendingPathComponent("fifo")
        try #require(mkfifo(fifo.path, 0o600) == 0)
        #expect(throws: QwenSafetensorsError.notRegularFile) { try QwenSafetensorsFile(resolvedURL: fifo) }
        let target = root.appendingPathComponent("target"); try write("{}", at: target)
        let link = root.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        #expect(throws: QwenSafetensorsError.self) { try QwenSafetensorsFile(resolvedURL: link) }
        // Root containment belongs to caller; resolved legitimate links can be opened.
        #expect(try QwenSafetensorsFile(resolvedURL: link.resolvingSymlinksInPath()).header.tensors.isEmpty)
        #expect(throws: QwenSafetensorsError.posix(operation: "open", code: ENOENT)) {
            try QwenSafetensorsFile(resolvedURL: root.appendingPathComponent("missing"))
        }
    }

    @Test func detectsInPlaceMutation() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("shard"); try write("{}", at: path)
        let file = try QwenSafetensorsFile(resolvedURL: path)
        let writer = try FileHandle(forWritingTo: path); defer { try? writer.close() }
        try writer.seekToEnd(); try writer.write(contentsOf: Data([1]))
        #expect(throws: QwenSafetensorsError.fileChanged) { try file.verifyUnchanged() }
    }

    @Test func descriptorPinsValidatedObjectAfterPathReplacement() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("shard")
        try write(#"{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, payload: Data([7,8]), at: path)
        let original = try QwenSafetensorsFile(resolvedURL: path)
        let replacement = root.appendingPathComponent("replacement"); try write("{}", at: replacement)
        try #require(rename(replacement.path, path.path) == 0)
        #expect(try QwenSafetensorsFile(resolvedURL: path).header.tensors.isEmpty)
        var value: UInt8 = 0
        #expect(pread(original.descriptor, &value, 1, off_t(original.header.payloadOffset)) == 1)
        #expect(value == 7)
    }
}
