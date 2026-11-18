import CoreFoundation
import Darwin
import Foundation

/// Errors that can occur during Safetensors file reading and validation.
public enum FlintSafetensorsError: Error, LocalizedError, Equatable {
    case posix(operation: String, code: Int32)
    case notRegularFile(String)
    case invalidFileSize(String)
    case invalidHeaderLength(String)
    case shortRead(expected: Int, actual: Int)
    case malformedHeader(String)
    case invalidMetadata(String)
    case invalidTensor(name: String, reason: String)
    case unsupportedDtype(name: String, dtype: String)
    case arithmeticOverflow(name: String)
    case overlappingTensors(String)
    case fileChanged(String)
    case tensorNotFound(name: String)
    case outOfBoundsRange(requested: Range<Int>, available: Range<Int>)
    case metadataConflict(String)
    case bufferCapacityInsufficient(required: Int, available: Int)
    case requestedAllocationExceedsLimit(requested: Int, limit: Int)

    public var errorDescription: String? {
        switch self {
        case .posix(let op, let code):
            return "Safetensors \(op) failed with errno \(code): \(String(cString: strerror(code)))"
        case .notRegularFile(let path):
            return "Safetensors input is not a regular file: \(path)"
        case .invalidFileSize(let path):
            return "Safetensors file size is invalid: \(path)"
        case .invalidHeaderLength(let detail):
            return "Safetensors header length is invalid: \(detail)"
        case .shortRead(let exp, let act):
            return "Safetensors short read: expected \(exp) bytes, got \(act) bytes"
        case .malformedHeader(let detail):
            return "Safetensors header JSON is malformed: \(detail)"
        case .invalidMetadata(let detail):
            return "Safetensors __metadata__ is invalid: \(detail)"
        case .invalidTensor(let name, let reason):
            return "Safetensors tensor '\(name)' is invalid: \(reason)"
        case .unsupportedDtype(let name, let dtype):
            return "Unsupported Safetensors dtype '\(dtype)' for tensor '\(name)'"
        case .arithmeticOverflow(let name):
            return "Arithmetic overflow computing bounds for tensor '\(name)'"
        case .overlappingTensors(let detail):
            return "Safetensors payload ranges overlap: \(detail)"
        case .fileChanged(let path):
            return "Safetensors file changed or mutated while open: \(path)"
        case .tensorNotFound(let name):
            return "Tensor '\(name)' not found in Safetensors header"
        case .outOfBoundsRange(let req, let avail):
            return "Requested byte range \(req) is out of bounds for available range \(avail)"
        case .metadataConflict(let detail):
            return "Contradictory metadata between configuration and Safetensors checkpoint: \(detail)"
        case .bufferCapacityInsufficient(let req, let avail):
            return "Destination buffer capacity (\(avail) bytes) is insufficient for requested read size (\(req) bytes)"
        case .requestedAllocationExceedsLimit(let req, let lim):
            return "Requested Data allocation of \(req) bytes exceeds maximum allowed limit of \(lim) bytes"
        }
    }
}

/// Metadata and physical location descriptor for a single tensor within a Safetensors shard.
public struct FlintTensorInfo: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: String
    public let byteWidth: Int
    public let dataOffsets: [Int] // [start, end] relative to shard payload start (immediately following header)
    public let byteCount: Int     // dataOffsets[1] - dataOffsets[0]
    public let elementCount: Int

    public init(
        name: String,
        shape: [Int],
        dtype: String,
        byteWidth: Int,
        dataOffsets: [Int],
        elementCount: Int
    ) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.byteWidth = byteWidth
        self.dataOffsets = dataOffsets
        self.byteCount = dataOffsets.count >= 2 ? (dataOffsets[1] - dataOffsets[0]) : 0
        self.elementCount = elementCount
    }

    public init(name: String, shape: [Int], dtype: String, byteWidth: Int, dataOffsets: [Int]) {
        let count = (try? FlintSafetensorsHeader.validatedElementCount(shape: shape, name: name)) ?? 0
        self.init(
            name: name,
            shape: shape,
            dtype: dtype,
            byteWidth: byteWidth,
            dataOffsets: dataOffsets,
            elementCount: count
        )
    }
}

/// Parsed header descriptor of a Safetensors file.
public struct FlintSafetensorsHeader: Equatable, Sendable {
    public static let maximumHeaderBytes = 100_000_000 // 100 MB bound
    public let headerByteCount: Int
    public let payloadOffset: Int
    public let fileByteCount: Int
    public let metadata: [String: String]
    public let tensors: [String: FlintTensorInfo]

    public static func headerLength(prefix: Data, fileByteCount: Int) throws -> Int {
        guard fileByteCount >= 8 else {
            throw FlintSafetensorsError.invalidFileSize("file byte count \(fileByteCount) < 8")
        }
        guard prefix.count == 8 else {
            throw FlintSafetensorsError.shortRead(expected: 8, actual: prefix.count)
        }
        // Little-endian 64-bit integer decode avoiding host alignment issues
        let raw = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard raw > 0 else {
            throw FlintSafetensorsError.invalidHeaderLength("header length is 0")
        }
        guard raw <= UInt64(maximumHeaderBytes) else {
            throw FlintSafetensorsError.invalidHeaderLength("header length \(raw) exceeds bound of \(maximumHeaderBytes) bytes")
        }
        guard let length = Int(exactly: raw) else {
            throw FlintSafetensorsError.arithmeticOverflow(name: "headerLength")
        }
        guard length <= fileByteCount - 8 else {
            throw FlintSafetensorsError.invalidHeaderLength("header length \(length) exceeds remaining file size \(fileByteCount - 8)")
        }
        return length
    }

    /// Safely computes the total element count of a shape tensor without risk of integer overflow.
    /// Traps are strictly prevented for malformed shapes such as [Int.max, Int.max, 0].
    public static func validatedElementCount(shape: [Int], name: String) throws -> Int {
        var product = 1
        var hasZero = false
        for dim in shape {
            guard dim >= 0 else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "dimension \(dim) must be non-negative")
            }
            if dim == 0 {
                hasZero = true
            } else {
                let (newProduct, overflow) = product.multipliedReportingOverflow(by: dim)
                guard !overflow else {
                    throw FlintSafetensorsError.arithmeticOverflow(name: "\(name) element count")
                }
                product = newProduct
            }
        }
        return hasZero ? 0 : product
    }

    public static func parse(data: Data, fileByteCount: Int) throws -> FlintSafetensorsHeader {
        guard !data.isEmpty, data.count <= maximumHeaderBytes else {
            throw FlintSafetensorsError.invalidHeaderLength("header data count \(data.count) invalid")
        }
        let (payloadOffset, overflow) = data.count.addingReportingOverflow(8)
        guard !overflow else {
            throw FlintSafetensorsError.arithmeticOverflow(name: "payloadOffset")
        }
        guard fileByteCount >= payloadOffset else {
            throw FlintSafetensorsError.invalidFileSize("fileByteCount \(fileByteCount) < payloadOffset \(payloadOffset)")
        }

        // Strict UTF-8 check; raw NUL byte is never valid JSON
        guard !data.contains(0), String(data: data, encoding: .utf8) != nil else {
            throw FlintSafetensorsError.malformedHeader("header data contains NUL or is not valid UTF-8")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw FlintSafetensorsError.malformedHeader(error.localizedDescription)
        }
        guard let entries = object as? [String: Any] else {
            throw FlintSafetensorsError.malformedHeader("root must be a dictionary")
        }

        var metadata: [String: String] = [:]
        var tensors: [String: FlintTensorInfo] = [:]
        var ranges: [(name: String, range: Range<Int>)] = []

        for (name, value) in entries {
            if name == "__metadata__" {
                if let metaDict = value as? [String: String] {
                    metadata = metaDict
                } else if let metaAny = value as? [String: Any] {
                    // Safetensors allows string metadata; ensure all values are strings
                    for (mk, mv) in metaAny {
                        guard let str = mv as? String else {
                            throw FlintSafetensorsError.invalidMetadata("value for key '\(mk)' must be string")
                        }
                        metadata[mk] = str
                    }
                } else {
                    throw FlintSafetensorsError.invalidMetadata("__metadata__ must be a dictionary")
                }
                continue
            }

            guard let entry = value as? [String: Any] else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "entry must be a dictionary")
            }
            guard let dtype = entry["dtype"] as? String else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "missing 'dtype'")
            }
            guard let rawShape = entry["shape"] as? [Any] else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "missing 'shape'")
            }
            guard let rawOffsets = entry["data_offsets"] as? [Any], rawOffsets.count == 2 else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "missing or invalid 'data_offsets' (must have 2 elements)")
            }

            let byteWidth: Int
            switch dtype {
            case "BOOL", "I8", "U8":
                byteWidth = 1
            case "I16", "U16", "F16", "BF16":
                byteWidth = 2
            case "I32", "U32", "F32":
                byteWidth = 4
            case "I64", "U64", "F64":
                byteWidth = 8
            default:
                throw FlintSafetensorsError.unsupportedDtype(name: name, dtype: dtype)
            }

            let shape = try rawShape.map { try parseNonnegativeInt($0, name: name, field: "shape") }
            let offsets = try rawOffsets.map { try parseNonnegativeInt($0, name: name, field: "data_offsets") }

            guard offsets[0] <= offsets[1] else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "data_offsets start (\(offsets[0])) > end (\(offsets[1]))")
            }
            let maxPayload = fileByteCount - payloadOffset
            guard offsets[1] <= maxPayload else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "data_offsets end (\(offsets[1])) exceeds shard payload size (\(maxPayload))")
            }

            let elements = try validatedElementCount(shape: shape, name: name)

            let (expectedBytes, bytesOverflow) = elements.multipliedReportingOverflow(by: byteWidth)
            guard !bytesOverflow else {
                throw FlintSafetensorsError.arithmeticOverflow(name: "\(name) byte size")
            }

            let actualBytes = offsets[1] - offsets[0]
            guard expectedBytes == actualBytes else {
                throw FlintSafetensorsError.invalidTensor(
                    name: name,
                    reason: "shape/dtype byte size (\(expectedBytes)) != payload range (\(actualBytes))"
                )
            }

            if offsets[0] < offsets[1] {
                ranges.append((name: name, range: offsets[0]..<offsets[1]))
            }

            tensors[name] = FlintTensorInfo(
                name: name,
                shape: shape,
                dtype: dtype,
                byteWidth: byteWidth,
                dataOffsets: offsets,
                elementCount: elements
            )
        }

        // Sort ranges and verify zero overlapping tensor allocations
        if ranges.count >= 2 {
            ranges.sort { $0.range.lowerBound < $1.range.lowerBound }
            for i in 1..<ranges.count {
                let prev = ranges[i - 1]
                let curr = ranges[i]
                guard prev.range.upperBound <= curr.range.lowerBound else {
                    throw FlintSafetensorsError.overlappingTensors(
                        "Tensor '\(prev.name)' range \(prev.range) overlaps with '\(curr.name)' range \(curr.range)"
                    )
                }
            }
        }

        return FlintSafetensorsHeader(
            headerByteCount: data.count,
            payloadOffset: payloadOffset,
            fileByteCount: fileByteCount,
            metadata: metadata,
            tensors: tensors
        )
    }

    private static func parseNonnegativeInt(_ value: Any, name: String, field: String) throws -> Int {
        if let num = value as? Int {
            guard num >= 0 else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "\(field) must be non-negative, got \(num)")
            }
            return num
        }
        if let num = value as? NSNumber {
            guard CFGetTypeID(num) != CFBooleanGetTypeID() else {
                throw FlintSafetensorsError.invalidTensor(name: name, reason: "\(field) cannot be boolean")
            }
            let unsigned = num.uint64Value
            guard let result = Int(exactly: unsigned), result >= 0 else {
                throw FlintSafetensorsError.arithmeticOverflow(name: "\(name) \(field)")
            }
            return result
        }
        throw FlintSafetensorsError.invalidTensor(name: name, reason: "\(field) must be an integer, got \(type(of: value))")
    }
}

/// An open, validated Safetensors shard owning a POSIX file descriptor.
/// Thread-safe for concurrent read operations via pread.
public final class FlintSafetensorsFile: @unchecked Sendable {
    public static let defaultMaxAllocatedBytes = 64 * 1024 * 1024 // 64 MiB default cap for Data allocations

    public let path: String
    internal let descriptor: Int32
    public let header: FlintSafetensorsHeader
    public let maxAllocatedBytes: Int
    private let identity: Identity

    public struct Identity: Equatable, Sendable {
        public let device: dev_t
        public let inode: ino_t
        public let size: Int
        public let modifiedSeconds: Int
        public let modifiedNanoseconds: Int
        public let changedSeconds: Int
        public let changedNanoseconds: Int

        public init(descriptor: Int32, path: String) throws {
            var info = stat()
            guard fstat(descriptor, &info) == 0 else {
                throw FlintSafetensorsError.posix(operation: "fstat", code: errno)
            }
            guard (info.st_mode & S_IFMT) == S_IFREG else {
                throw FlintSafetensorsError.notRegularFile(path)
            }
            guard info.st_size >= 8, let size = Int(exactly: info.st_size) else {
                throw FlintSafetensorsError.invalidFileSize(path)
            }
            self.device = info.st_dev
            self.inode = info.st_ino
            self.size = size
            self.modifiedSeconds = info.st_mtimespec.tv_sec
            self.modifiedNanoseconds = info.st_mtimespec.tv_nsec
            self.changedSeconds = info.st_ctimespec.tv_sec
            self.changedNanoseconds = info.st_ctimespec.tv_nsec
        }
    }

    public init(url: URL, maxAllocatedBytes: Int = defaultMaxAllocatedBytes) throws {
        let filePath = url.path
        let fd = open(filePath, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            if errno == ELOOP {
                throw FlintSafetensorsError.notRegularFile("Symlink rejected by O_NOFOLLOW: \(filePath)")
            }
            throw FlintSafetensorsError.posix(operation: "open(\(filePath))", code: errno)
        }

        do {
            let identity = try Identity(descriptor: fd, path: filePath)
            var prefixData = Data(count: 8)
            try Self.preadExact(fd: fd, destination: &prefixData, offset: 0, path: filePath)

            let headerLen = try FlintSafetensorsHeader.headerLength(prefix: prefixData, fileByteCount: identity.size)
            var headerData = Data(count: headerLen)
            try Self.preadExact(fd: fd, destination: &headerData, offset: 8, path: filePath)

            let header = try FlintSafetensorsHeader.parse(data: headerData, fileByteCount: identity.size)

            // Re-verify that file wasn't modified while reading header
            let currentIdentity = try Identity(descriptor: fd, path: filePath)
            guard identity == currentIdentity else {
                throw FlintSafetensorsError.fileChanged(filePath)
            }

            self.path = filePath
            self.descriptor = fd
            self.header = header
            self.maxAllocatedBytes = maxAllocatedBytes
            self.identity = identity
        } catch {
            close(fd)
            throw error
        }
    }

    deinit {
        close(descriptor)
    }

    /// Verifies that the open file descriptor has not undergone size or timestamp changes,
    /// and verifies that the file on disk has not been unlinked or replaced with another inode.
    public func verifyUnchanged() throws {
        // 1. Open descriptor check
        let current = try Identity(descriptor: descriptor, path: path)
        guard identity == current else {
            throw FlintSafetensorsError.fileChanged("Open file descriptor identity modified for \(path)")
        }

        // 2. On-disk path check
        var pathInfo = stat()
        guard stat(path, &pathInfo) == 0 else {
            throw FlintSafetensorsError.fileChanged("File at \(path) is no longer accessible on disk (errno \(errno))")
        }
        guard pathInfo.st_dev == identity.device && pathInfo.st_ino == identity.inode else {
            throw FlintSafetensorsError.fileChanged("File at \(path) was replaced on disk (inode \(identity.inode) -> \(pathInfo.st_ino))")
        }
    }

    /// Reads the entire physical byte payload of a named tensor into Data.
    /// Allocation is capped by maxAllocatedBytes.
    public func readTensorPayload(_ name: String) throws -> Data {
        guard let info = header.tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        return try readPayloadSlice(range: info.dataOffsets[0]..<info.dataOffsets[1])
    }

    /// Reads a sub-range within a named tensor's payload into Data.
    /// byteRange is relative to the start of the tensor's data.
    /// Allocation is capped by maxAllocatedBytes.
    public func readTensorSlice(_ name: String, byteRange: Range<Int>) throws -> Data {
        guard let info = header.tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        guard byteRange.lowerBound >= 0, byteRange.upperBound <= info.byteCount else {
            throw FlintSafetensorsError.outOfBoundsRange(
                requested: byteRange,
                available: 0..<info.byteCount
            )
        }
        if byteRange.isEmpty {
            return Data()
        }
        let physicalStart = info.dataOffsets[0] + byteRange.lowerBound
        let physicalEnd = info.dataOffsets[0] + byteRange.upperBound
        return try readPayloadSlice(range: physicalStart..<physicalEnd)
    }

    /// Reads a byte range relative to the shard's payload start (i.e. immediately after header) into Data.
    /// Allocation is capped by maxAllocatedBytes.
    public func readPayloadSlice(range: Range<Int>) throws -> Data {
        let payloadBytes = header.fileByteCount - header.payloadOffset
        guard range.lowerBound >= 0, range.upperBound <= payloadBytes else {
            throw FlintSafetensorsError.outOfBoundsRange(
                requested: range,
                available: 0..<payloadBytes
            )
        }
        if range.isEmpty {
            return Data()
        }
        guard range.count <= maxAllocatedBytes else {
            throw FlintSafetensorsError.requestedAllocationExceedsLimit(
                requested: range.count,
                limit: maxAllocatedBytes
            )
        }

        try verifyUnchanged()
        let fileOffset = header.payloadOffset + range.lowerBound
        var data = Data(count: range.count)
        try Self.preadExact(fd: descriptor, destination: &data, offset: off_t(fileOffset), path: path)
        try verifyUnchanged()
        return data
    }

    /// Reads a byte range relative to the shard's payload start into a caller-allocated buffer.
    /// Destination capacity is strictly verified against requested range.count to prevent buffer overflow.
    public func readPayload(into buffer: UnsafeMutableRawBufferPointer, range: Range<Int>) throws {
        let payloadBytes = header.fileByteCount - header.payloadOffset
        guard range.lowerBound >= 0, range.upperBound <= payloadBytes else {
            throw FlintSafetensorsError.outOfBoundsRange(
                requested: range,
                available: 0..<payloadBytes
            )
        }
        guard buffer.count >= range.count else {
            throw FlintSafetensorsError.bufferCapacityInsufficient(
                required: range.count,
                available: buffer.count
            )
        }
        if range.isEmpty {
            return
        }
        guard let baseAddress = buffer.baseAddress else {
            throw FlintSafetensorsError.bufferCapacityInsufficient(
                required: range.count,
                available: 0
            )
        }

        try verifyUnchanged()
        let fileOffset = header.payloadOffset + range.lowerBound
        var totalRead = 0
        while totalRead < range.count {
            let bytes = pread(
                descriptor,
                baseAddress.advanced(by: totalRead),
                range.count - totalRead,
                off_t(fileOffset + totalRead)
            )
            if bytes < 0 {
                let err = errno
                if err == EINTR { continue }
                throw FlintSafetensorsError.posix(operation: "pread", code: err)
            }
            guard bytes > 0 else {
                throw FlintSafetensorsError.shortRead(expected: range.count, actual: totalRead)
            }
            totalRead += bytes
        }
        try verifyUnchanged()
    }

    /// Reads the entire physical byte payload of a named tensor into a caller-allocated buffer.
    public func readTensorPayload(_ name: String, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard let info = header.tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        try readPayload(into: buffer, range: info.dataOffsets[0]..<info.dataOffsets[1])
    }

    /// Reads a sub-range within a named tensor's payload into a caller-allocated buffer.
    public func readTensorSlice(_ name: String, byteRange: Range<Int>, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard let info = header.tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        guard byteRange.lowerBound >= 0, byteRange.upperBound <= info.byteCount else {
            throw FlintSafetensorsError.outOfBoundsRange(
                requested: byteRange,
                available: 0..<info.byteCount
            )
        }
        let physicalStart = info.dataOffsets[0] + byteRange.lowerBound
        let physicalEnd = info.dataOffsets[0] + byteRange.upperBound
        try readPayload(into: buffer, range: physicalStart..<physicalEnd)
    }

    /// Reads a named tensor in bounded chunks without loading the entire tensor into memory.
    public func readTensorChunks(
        _ name: String,
        chunkSize: Int = 1024 * 1024,
        handler: (_ chunk: UnsafeRawBufferPointer, _ offsetInTensor: Int) throws -> Void
    ) throws {
        guard let info = header.tensors[name] else {
            throw FlintSafetensorsError.tensorNotFound(name: name)
        }
        guard chunkSize > 0 else {
            throw FlintSafetensorsError.bufferCapacityInsufficient(required: 1, available: chunkSize)
        }
        let totalBytes = info.byteCount
        if totalBytes == 0 {
            try handler(UnsafeRawBufferPointer(start: nil, count: 0), 0)
            return
        }

        let actualChunk = min(chunkSize, totalBytes)
        var chunkBuffer = Data(count: actualChunk)
        var offset = 0
        while offset < totalBytes {
            let thisRead = min(actualChunk, totalBytes - offset)
            let subRange = offset..<(offset + thisRead)
            try chunkBuffer.withUnsafeMutableBytes { rawBuf in
                let sliceBuf = UnsafeMutableRawBufferPointer(rebasing: rawBuf[0..<thisRead])
                try readTensorSlice(name, byteRange: subRange, into: sliceBuf)
                try handler(UnsafeRawBufferPointer(sliceBuf), offset)
            }
            offset += thisRead
        }
    }

    private static func preadExact(fd: Int32, destination: inout Data, offset: off_t, path: String) throws {
        let count = destination.count
        guard count > 0 else { return }
        var totalRead = 0
        try destination.withUnsafeMutableBytes { (rawBuffer: UnsafeMutableRawBufferPointer) throws in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw FlintSafetensorsError.invalidMetadata("buffer allocation failed for \(path)")
            }
            while totalRead < count {
                let bytes = pread(fd, baseAddress.advanced(by: totalRead), count - totalRead, offset + off_t(totalRead))
                if bytes < 0 {
                    let err = errno
                    if err == EINTR { continue }
                    throw FlintSafetensorsError.posix(operation: "pread", code: err)
                }
                guard bytes > 0 else {
                    throw FlintSafetensorsError.shortRead(expected: count, actual: totalRead)
                }
                totalRead += bytes
            }
        }
    }
}
