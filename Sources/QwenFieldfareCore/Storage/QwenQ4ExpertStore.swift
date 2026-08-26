import CryptoKit
import Darwin
import Foundation

public struct QwenQ4ExpertStorageMetrics: Sendable, Equatable {
    public let kind: String
    public let packedReads: Int
    public let packedPreadOperations: Int
    public let packedReadFailures: Int

    public init(
        kind: String,
        packedReads: Int,
        packedPreadOperations: Int,
        packedReadFailures: Int
    ) {
        self.kind = kind
        self.packedReads = packedReads
        self.packedPreadOperations = packedPreadOperations
        self.packedReadFailures = packedReadFailures
    }
}

public struct QwenQ4ExpertStoreManifest: Codable, Sendable, Equatable {
    public struct Layer: Codable, Sendable, Equatable {
        public let layerIndex: Int
        public let filename: String
        public let fileSize: UInt64
        public let payloadSHA256: String

        public init(layerIndex: Int, filename: String, fileSize: UInt64, payloadSHA256: String) {
            self.layerIndex = layerIndex
            self.filename = filename
            self.fileSize = fileSize
            self.payloadSHA256 = payloadSHA256
        }
    }

    public let schemaVersion: Int
    public let sourceIndexSHA256: String
    public let layers: [Layer]

    public init(schemaVersion: Int, sourceIndexSHA256: String, layers: [Layer]) {
        self.schemaVersion = schemaVersion
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.layers = layers
    }
}

public struct QwenQ4ExpertLayerHeader: Sendable, Equatable {
    public static let magic: [UInt8] = Array("Q4EXP02\0".utf8)
    public static let schemaVersion: UInt32 = 2
    public static let byteLength = 4_096

    public let layerIndex: Int
    public let expertsPerLayer: Int
    public let expertByteLength: Int
    public let payloadByteLength: UInt64
    public let payloadSHA256: String
    public let sourceIndexSHA256: String

    public init(
        layerIndex: Int,
        expertsPerLayer: Int,
        expertByteLength: Int,
        payloadByteLength: UInt64,
        payloadSHA256: String,
        sourceIndexSHA256: String
    ) {
        self.layerIndex = layerIndex
        self.expertsPerLayer = expertsPerLayer
        self.expertByteLength = expertByteLength
        self.payloadByteLength = payloadByteLength
        self.payloadSHA256 = payloadSHA256
        self.sourceIndexSHA256 = sourceIndexSHA256
    }

    public func encode() throws -> Data {
        guard Self.isSHA256(payloadSHA256), Self.isSHA256(sourceIndexSHA256) else {
            throw QwenQ4ExpertStoreError.invalidHeader("SHA-256 fields must be 64 lowercase hexadecimal characters")
        }
        guard layerIndex >= 0,
              expertsPerLayer > 0,
              expertByteLength > 0,
              layerIndex <= Int(UInt32.max),
              expertsPerLayer <= Int(UInt32.max),
              expertByteLength <= Int(UInt32.max) else {
            throw QwenQ4ExpertStoreError.invalidHeader("numeric field is out of range")
        }

        var bytes = [UInt8](repeating: 0, count: Self.byteLength)
        bytes.replaceSubrange(0..<8, with: Self.magic)
        Self.writeUInt32(Self.schemaVersion, to: &bytes, at: 8)
        Self.writeUInt32(UInt32(Self.byteLength), to: &bytes, at: 12)
        Self.writeUInt32(UInt32(layerIndex), to: &bytes, at: 16)
        Self.writeUInt32(UInt32(expertsPerLayer), to: &bytes, at: 20)
        Self.writeUInt32(UInt32(expertByteLength), to: &bytes, at: 24)
        Self.writeUInt64(payloadByteLength, to: &bytes, at: 32)
        bytes.replaceSubrange(40..<104, with: payloadSHA256.utf8)
        bytes.replaceSubrange(104..<168, with: sourceIndexSHA256.utf8)
        return Data(bytes)
    }

    public static func decode(_ data: Data) throws -> Self {
        guard data.count == byteLength else {
            throw QwenQ4ExpertStoreError.invalidHeader("expected \(byteLength) bytes, got \(data.count)")
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<8]) == magic else {
            throw QwenQ4ExpertStoreError.invalidHeader("magic mismatch")
        }
        guard readUInt32(bytes, at: 8) == schemaVersion else {
            throw QwenQ4ExpertStoreError.invalidHeader("unsupported schema version")
        }
        guard readUInt32(bytes, at: 12) == UInt32(byteLength) else {
            throw QwenQ4ExpertStoreError.invalidHeader("header byte length mismatch")
        }
        guard let payloadHash = String(bytes: bytes[40..<104], encoding: .utf8),
              let sourceHash = String(bytes: bytes[104..<168], encoding: .utf8),
              isSHA256(payloadHash), isSHA256(sourceHash) else {
            throw QwenQ4ExpertStoreError.invalidHeader("invalid SHA-256 field")
        }
        return Self(
            layerIndex: Int(readUInt32(bytes, at: 16)),
            expertsPerLayer: Int(readUInt32(bytes, at: 20)),
            expertByteLength: Int(readUInt32(bytes, at: 24)),
            payloadByteLength: readUInt64(bytes, at: 32),
            payloadSHA256: payloadHash,
            sourceIndexSHA256: sourceHash)
    }

    package static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func writeUInt32(_ value: UInt32, to bytes: inout [UInt8], at offset: Int) {
        let little = value.littleEndian
        withUnsafeBytes(of: little) { bytes.replaceSubrange(offset..<(offset + 4), with: $0) }
    }

    private static func writeUInt64(_ value: UInt64, to bytes: inout [UInt8], at offset: Int) {
        let little = value.littleEndian
        withUnsafeBytes(of: little) { bytes.replaceSubrange(offset..<(offset + 8), with: $0) }
    }

    private static func readUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        bytes[offset..<(offset + 4)].enumerated().reduce(0) {
            $0 | (UInt32($1.element) << UInt32($1.offset * 8))
        }
    }

    private static func readUInt64(_ bytes: [UInt8], at offset: Int) -> UInt64 {
        bytes[offset..<(offset + 8)].enumerated().reduce(0) {
            $0 | (UInt64($1.element) << UInt64($1.offset * 8))
        }
    }
}

public enum QwenQ4ExpertStoreError: Error, LocalizedError {
    case invalidManifest(String)
    case invalidHeader(String)
    case sourceMismatch(expected: String, actual: String)
    case posix(operation: String, path: String, code: Int32)
    case shortRead(path: String, expected: Int, actual: Int)
    case partialStore(String)

    public var errorDescription: String? {
        switch self {
        case .invalidManifest(let detail): return "Invalid Qwen Q4 expert-store manifest: \(detail)"
        case .invalidHeader(let detail): return "Invalid Qwen Q4 expert-store header: \(detail)"
        case .sourceMismatch(let expected, let actual):
            return "Qwen Q4 expert store belongs to a different source index (expected \(expected), got \(actual))"
        case .posix(let operation, let path, let code):
            return "\(operation) failed for \(path): errno \(code)"
        case .shortRead(let path, let expected, let actual):
            return "Short read from \(path): expected \(expected) bytes, got \(actual)"
        case .partialStore(let detail): return "Incomplete Qwen Q4 expert store: \(detail)"
        }
    }
}

public final class QwenQ4ExpertStore: @unchecked Sendable {
    public static let directoryName = "qwen_q4_experts"
    public static let manifestFilename = "manifest.json"
    public static let layerCount = 40
    public static let expertsPerLayer = 256
    public static let expertByteLength = 1_769_472
    public static let payloadByteLength = UInt64(expertsPerLayer * expertByteLength)
    public static let layerFileByteLength = UInt64(QwenQ4ExpertLayerHeader.byteLength) + payloadByteLength

    public let directoryPath: String
    public let manifest: QwenQ4ExpertStoreManifest
    private let layerFileDescriptors: [Int32]
    private let counterLock = NSLock()
    private var packedReads = 0
    private var packedPreadOperations = 0
    private var packedReadFailures = 0

    public static func defaultDirectory(modelPath: String) -> String {
        (modelPath as NSString).appendingPathComponent(directoryName)
    }

    public static func layerFilename(_ layer: Int) -> String {
        String(format: "layer_%02d.q4exp", layer)
    }

    public static func openIfPresent(
        modelPath: String,
        directoryPath: String? = nil
    ) throws -> QwenQ4ExpertStore? {
        let directory = directoryPath ?? defaultDirectory(modelPath: modelPath)
        let manifestPath = (directory as NSString).appendingPathComponent(manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestPath) else { return nil }
        return try QwenQ4ExpertStore(modelPath: modelPath, directoryPath: directory)
    }

    public init(modelPath: String, directoryPath: String? = nil) throws {
        let directory = directoryPath ?? Self.defaultDirectory(modelPath: modelPath)
        self.directoryPath = directory
        let manifestPath = (directory as NSString).appendingPathComponent(Self.manifestFilename)
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath), options: .mappedIfSafe)
        let decoded: QwenQ4ExpertStoreManifest
        do {
            decoded = try JSONDecoder().decode(QwenQ4ExpertStoreManifest.self, from: manifestData)
        } catch {
            throw QwenQ4ExpertStoreError.invalidManifest(error.localizedDescription)
        }
        try Self.validateManifest(decoded)

        let sourceIndex = (modelPath as NSString).appendingPathComponent("model.safetensors.index.json")
        let actualSourceHash = try Sha256Verifier.hashFile(at: URL(fileURLWithPath: sourceIndex))
        guard actualSourceHash == decoded.sourceIndexSHA256 else {
            throw QwenQ4ExpertStoreError.sourceMismatch(
                expected: decoded.sourceIndexSHA256, actual: actualSourceHash)
        }

        var descriptors: [Int32] = []
        descriptors.reserveCapacity(Self.layerCount)
        do {
            for layer in decoded.layers.sorted(by: { $0.layerIndex < $1.layerIndex }) {
                let path = (directory as NSString).appendingPathComponent(layer.filename)
                let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
                guard fd >= 0 else {
                    throw QwenQ4ExpertStoreError.posix(operation: "open", path: path, code: errno)
                }
                descriptors.append(fd)
                var info = stat()
                guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
                    throw QwenQ4ExpertStoreError.posix(operation: "fstat", path: path, code: errno == 0 ? EINVAL : errno)
                }
                guard UInt64(info.st_size) == layer.fileSize,
                      layer.fileSize == Self.layerFileByteLength else {
                    throw QwenQ4ExpertStoreError.invalidManifest("unexpected size for \(layer.filename)")
                }
                var headerData = Data(count: QwenQ4ExpertLayerHeader.byteLength)
                let got = headerData.withUnsafeMutableBytes {
                    Self.preadFully(fd: fd, destination: $0.baseAddress!, count: $0.count, offset: 0)
                }
                guard got == QwenQ4ExpertLayerHeader.byteLength else {
                    throw QwenQ4ExpertStoreError.shortRead(
                        path: path, expected: QwenQ4ExpertLayerHeader.byteLength, actual: got)
                }
                let header = try QwenQ4ExpertLayerHeader.decode(headerData)
                guard header.layerIndex == layer.layerIndex,
                      header.expertsPerLayer == Self.expertsPerLayer,
                      header.expertByteLength == Self.expertByteLength,
                      header.payloadByteLength == Self.payloadByteLength,
                      header.payloadSHA256 == layer.payloadSHA256,
                      header.sourceIndexSHA256 == decoded.sourceIndexSHA256 else {
                    throw QwenQ4ExpertStoreError.invalidHeader("metadata mismatch in \(layer.filename)")
                }
            }
        } catch {
            for fd in descriptors { Darwin.close(fd) }
            throw error
        }
        self.manifest = decoded
        self.layerFileDescriptors = descriptors
    }

    deinit {
        for fd in layerFileDescriptors { Darwin.close(fd) }
    }

    public func readExpert(
        layer: Int,
        expert: Int,
        destination: UnsafeMutableRawPointer
    ) -> Bool {
        guard (0..<Self.layerCount).contains(layer),
              (0..<Self.expertsPerLayer).contains(expert) else {
            recordRead(success: false)
            return false
        }
        let offset = UInt64(QwenQ4ExpertLayerHeader.byteLength)
            + UInt64(expert * Self.expertByteLength)
        let got = Self.preadFully(
            fd: layerFileDescriptors[layer],
            destination: destination,
            count: Self.expertByteLength,
            offset: offset)
        let success = got == Self.expertByteLength
        recordRead(success: success, preadOperations: 1)
        return success
    }

    public func metricsSnapshot() -> QwenQ4ExpertStorageMetrics {
        counterLock.lock()
        defer { counterLock.unlock() }
        return QwenQ4ExpertStorageMetrics(
            kind: "qwen-q4-packed-v2",
            packedReads: packedReads,
            packedPreadOperations: packedPreadOperations,
            packedReadFailures: packedReadFailures)
    }

    private func recordRead(success: Bool, preadOperations: Int = 0) {
        counterLock.lock()
        packedPreadOperations += preadOperations
        if success { packedReads += 1 } else { packedReadFailures += 1 }
        counterLock.unlock()
    }

    package static func validateManifest(_ manifest: QwenQ4ExpertStoreManifest) throws {
        guard manifest.schemaVersion == Int(QwenQ4ExpertLayerHeader.schemaVersion) else {
            throw QwenQ4ExpertStoreError.invalidManifest("unsupported schema version \(manifest.schemaVersion)")
        }
        guard QwenQ4ExpertLayerHeader.isSHA256(manifest.sourceIndexSHA256) else {
            throw QwenQ4ExpertStoreError.invalidManifest("invalid source index SHA-256")
        }
        guard manifest.layers.count == layerCount else {
            throw QwenQ4ExpertStoreError.invalidManifest("expected \(layerCount) layers, got \(manifest.layers.count)")
        }
        let sorted = manifest.layers.sorted(by: { $0.layerIndex < $1.layerIndex })
        for (expected, layer) in sorted.enumerated() {
            guard layer.layerIndex == expected,
                  layer.filename == layerFilename(expected),
                  layer.fileSize == layerFileByteLength,
                  QwenQ4ExpertLayerHeader.isSHA256(layer.payloadSHA256) else {
                throw QwenQ4ExpertStoreError.invalidManifest("invalid layer entry \(expected)")
            }
        }
    }

    package static func preadFully(
        fd: Int32,
        destination: UnsafeMutableRawPointer,
        count: Int,
        offset: UInt64
    ) -> Int {
        var completed = 0
        while completed < count {
            let result = Darwin.pread(
                fd,
                destination.advanced(by: completed),
                count - completed,
                off_t(offset + UInt64(completed)))
            if result < 0, errno == EINTR { continue }
            guard result > 0 else { return completed }
            completed += result
        }
        return completed
    }
}
