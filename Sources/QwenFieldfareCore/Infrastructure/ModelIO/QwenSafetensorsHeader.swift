import CoreFoundation
import Darwin
import Foundation

/// Typed errors for metadata validation and bounded file/payload reads.
enum QwenSafetensorsError: Error, LocalizedError, Equatable {
    case posix(operation: String, code: Int32)
    case notRegularFile
    case invalidFileSize
    case invalidHeaderLength
    case shortRead(expected: Int, actual: Int)
    case malformedHeader
    case invalidMetadata
    case invalidTensor(name: String, reason: String)
    case unsupportedDtype(name: String, dtype: String)
    case arithmeticOverflow(name: String)
    case overlappingTensors
    case fileChanged

    var errorDescription: String? {
        switch self {
        case .posix(let operation, let code): return "Safetensors \(operation) failed (errno \(code))."
        case .notRegularFile: return "Safetensors input must be a regular file."
        case .invalidFileSize: return "Safetensors file size is invalid."
        case .invalidHeaderLength: return "Safetensors header length is invalid or exceeds 100,000,000 bytes."
        case .shortRead(let expected, let actual): return "Safetensors read was incomplete: expected \(expected), got \(actual) bytes."
        case .malformedHeader: return "Safetensors header must be a JSON object."
        case .invalidMetadata: return "Safetensors __metadata__ must contain string values."
        case .invalidTensor(let name, let reason): return "Invalid Safetensors tensor \(name): \(reason)."
        case .unsupportedDtype(let name, let dtype): return "Unsupported Safetensors dtype \(dtype) for \(name)."
        case .arithmeticOverflow(let name): return "Safetensors size arithmetic overflow for \(name)."
        case .overlappingTensors: return "Safetensors tensor payload ranges overlap."
        case .fileChanged: return "Safetensors file changed during validation."
        }
    }
}

/// Shared parser used by CPU resident and active-streaming Safetensors readers.
struct QwenSafetensorsHeader {
    // Preserve the resident loader's existing decimal-byte limit.
    static let maximumHeaderBytes = 100_000_000
    let headerByteCount: Int
    let payloadOffset: Int
    let fileByteCount: Int
    let tensors: [String: QwenTensorInfo]

    static func headerLength(prefix: Data, fileByteCount: Int) throws -> Int {
        guard fileByteCount >= 8 else { throw QwenSafetensorsError.invalidFileSize }
        guard prefix.count == 8 else {
            throw QwenSafetensorsError.shortRead(expected: 8, actual: prefix.count)
        }
        // Bytewise LE decode avoids both alignment assumptions and host endianness.
        let raw = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard raw > 0, raw <= UInt64(maximumHeaderBytes),
              let length = Int(exactly: raw) else { throw QwenSafetensorsError.invalidHeaderLength }
        guard length <= fileByteCount - 8 else {
            throw QwenSafetensorsError.shortRead(expected: length, actual: fileByteCount - 8)
        }
        return length
    }

    static func parse(_ data: Data, fileByteCount: Int) throws -> Self {
        guard !data.isEmpty, data.count <= maximumHeaderBytes else {
            throw QwenSafetensorsError.invalidHeaderLength
        }
        let (payloadOffset, overflow) = data.count.addingReportingOverflow(8)
        guard !overflow else { throw QwenSafetensorsError.arithmeticOverflow(name: "header") }
        guard fileByteCount >= payloadOffset else { throw QwenSafetensorsError.invalidFileSize }
        // JSONSerialization auto-detects UTF-16/32. Safetensors headers are UTF-8;
        // raw NUL is never valid JSON and would otherwise enable that detection.
        guard !data.contains(0), String(data: data, encoding: .utf8) != nil else {
            throw QwenSafetensorsError.malformedHeader
        }
        let object: Any
        do { object = try JSONSerialization.jsonObject(with: data) }
        catch { throw QwenSafetensorsError.malformedHeader }
        guard let entries = object as? [String: Any] else { throw QwenSafetensorsError.malformedHeader }
        var tensors: [String: QwenTensorInfo] = [:]
        var ranges: [Range<Int>] = []
        for (name, value) in entries {
            if name == "__metadata__" {
                guard value is [String: String] else { throw QwenSafetensorsError.invalidMetadata }
                continue
            }
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                  let rawShape = entry["shape"] as? [Any],
                  let rawOffsets = entry["data_offsets"] as? [Any], rawOffsets.count == 2 else {
                throw QwenSafetensorsError.invalidTensor(name: name, reason: "missing or malformed fields")
            }
            let width: Int
            switch dtype {
            case "BOOL", "I8", "U8": width = 1
            case "I16", "U16", "F16", "BF16": width = 2
            case "I32", "U32", "F32": width = 4
            case "I64", "U64", "F64": width = 8
            default: throw QwenSafetensorsError.unsupportedDtype(name: name, dtype: dtype)
            }
            let shape = try rawShape.map { try nonnegativeInteger($0, name: name) }
            let offsets = try rawOffsets.map { try nonnegativeInteger($0, name: name) }
            guard offsets[0] <= offsets[1], offsets[1] <= fileByteCount - payloadOffset else {
                throw QwenSafetensorsError.invalidTensor(name: name, reason: "payload range is out of bounds")
            }
            // Validate every dimension, but an empty tensor has zero elements
            // even if dimensions preceding its zero would otherwise overflow.
            var elements = shape.contains(0) ? 0 : 1
            if elements != 0 {
                for dimension in shape {
                    let result = elements.multipliedReportingOverflow(by: dimension)
                    guard !result.overflow else { throw QwenSafetensorsError.arithmeticOverflow(name: name) }
                    elements = result.partialValue
                }
            }
            let bytes = elements.multipliedReportingOverflow(by: width)
            guard !bytes.overflow else { throw QwenSafetensorsError.arithmeticOverflow(name: name) }
            guard bytes.partialValue == offsets[1] - offsets[0] else {
                throw QwenSafetensorsError.invalidTensor(name: name, reason: "shape/dtype byte size does not match payload range")
            }
            // This addition is safe because end <= fileByteCount - payloadOffset.
            if offsets[0] < offsets[1] { ranges.append(offsets[0]..<offsets[1]) }
            tensors[name] = QwenTensorInfo(shape: shape, dtype: dtype, dataOffsets: offsets)
        }
        ranges.sort { $0.lowerBound < $1.lowerBound }
        for index in ranges.indices.dropFirst() {
            guard ranges[index - 1].upperBound <= ranges[index].lowerBound else {
                throw QwenSafetensorsError.overlappingTensors
            }
        }
        return Self(headerByteCount: data.count, payloadOffset: payloadOffset,
                    fileByteCount: fileByteCount, tensors: tensors)
    }

    private static func nonnegativeInteger(_ value: Any, name: String) throws -> Int {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              !["f", "d"].contains(String(cString: number.objCType)),
              let unsigned = UInt64(number.stringValue) else {
            throw QwenSafetensorsError.invalidTensor(name: name, reason: "expected a nonnegative integer")
        }
        guard let result = Int(exactly: unsigned) else {
            throw QwenSafetensorsError.arithmeticOverflow(name: name)
        }
        return result
    }
}

/// Owns the validated descriptor. Future readers must retain this object and use
/// that descriptor, not reopen a path using previously validated offsets.
/// Callers resolve legitimate links and enforce model-root containment FIRST.
/// NOFOLLOW rejects a substituted leaf link, NONBLOCK prevents FIFO-open hangs.
final class QwenSafetensorsFile {
    let descriptor: Int32 // Borrowed by consumers; only this owner closes it.
    let header: QwenSafetensorsHeader
    private let identity: Identity

    private struct Identity: Equatable {
        let device: dev_t
        let inode: ino_t
        let size: Int
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int
        let changedNanoseconds: Int

        init(descriptor: Int32) throws {
            var info = stat()
            guard fstat(descriptor, &info) == 0 else {
                throw QwenSafetensorsError.posix(operation: "fstat", code: errno)
            }
            guard info.st_mode & S_IFMT == S_IFREG else { throw QwenSafetensorsError.notRegularFile }
            guard info.st_size >= 8, let size = Int(exactly: info.st_size) else {
                throw QwenSafetensorsError.invalidFileSize
            }
            self.device = info.st_dev; self.inode = info.st_ino; self.size = size
            self.modifiedSeconds = info.st_mtimespec.tv_sec
            self.modifiedNanoseconds = info.st_mtimespec.tv_nsec
            self.changedSeconds = info.st_ctimespec.tv_sec
            self.changedNanoseconds = info.st_ctimespec.tv_nsec
        }
    }

    init(resolvedURL: URL) throws {
        let fd = open(resolvedURL.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw QwenSafetensorsError.posix(operation: "open", code: errno) }
        do {
            let identity = try Identity(descriptor: fd)
            let prefix = try Self.read(fd: fd, count: 8, offset: 0)
            let length = try QwenSafetensorsHeader.headerLength(prefix: prefix, fileByteCount: identity.size)
            let bytes = try Self.read(fd: fd, count: length, offset: 8)
            let header = try QwenSafetensorsHeader.parse(bytes, fileByteCount: identity.size)
            guard identity == (try Identity(descriptor: fd)) else { throw QwenSafetensorsError.fileChanged }
            self.descriptor = fd; self.header = header; self.identity = identity
        } catch {
            close(fd)
            throw error
        }
    }

    /// Detects size/mtime/ctime changes to the same opened object. Does not lock
    /// out external writes or guarantee against mutations after this check.
    func verifyUnchanged() throws {
        guard identity == (try Identity(descriptor: descriptor)) else { throw QwenSafetensorsError.fileChanged }
    }

    /// Range is relative to tensor-data start, never the beginning of the file.
    /// This is a bounded selected-range read, not a complete-shard read.
    func readPayload(_ range: Range<Int>) throws -> Data {
        let payloadBytes = header.fileByteCount - header.payloadOffset
        guard range.lowerBound >= 0, range.upperBound <= payloadBytes else {
            throw QwenSafetensorsError.invalidTensor(name: "payload", reason: "read range is out of bounds")
        }
        try verifyUnchanged()
        // The bound above proves both addition and offset+count fit Int/off_t.
        let offset = header.payloadOffset + range.lowerBound
        let bytes = try Self.read(fd: descriptor, count: range.count, offset: UInt64(offset))
        try verifyUnchanged()
        return bytes
    }

    deinit { close(descriptor) }

    private static func read(fd: Int32, count: Int, offset: UInt64) throws -> Data {
        // Header calls fit 100,000,008; payload calls are bounded by validated
        // file size <= Int.max. The helper's UInt64/off_t arithmetic is safe.
        guard count > 0 else { return Data() }
        var bytes = Data(count: count)
        let got = bytes.withUnsafeMutableBytes {
            QwenQ4ExpertStore.preadFully(fd: fd, destination: $0.baseAddress!, count: count, offset: offset)
        }
        guard got == count else { throw QwenSafetensorsError.shortRead(expected: count, actual: got) }
        return bytes
    }
}
