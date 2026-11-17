import Darwin
import Foundation
import Metal

/// Whole-file, read-only mapping of the *validated descriptor*. ARC is the only
/// release mechanism: a buffer lease must never be invalidated by runner unload.
/// Sendable permits immutable address/metadata sharing, not concurrent file writes.
/// Checkpoints must remain immutable until the last CPU/GPU lease is released.
final class QwenMappedSafetensorsFile: @unchecked Sendable {
    let file: QwenSafetensorsFile
    let ptr: UnsafeRawPointer
    let mappedByteCount: Int
    private let unmap: (UnsafeMutableRawPointer, Int) -> Void
    var fd: Int32 { file.descriptor }

    static func roundedLength(_ count: Int, pageSize: Int) throws -> Int {
        guard count > 0, pageSize > 0 else { throw QwenSafetensorsError.invalidFileSize }
        let extra = (pageSize - count % pageSize) % pageSize
        let result = count.addingReportingOverflow(extra)
        guard !result.overflow else { throw QwenSafetensorsError.arithmeticOverflow(name: "mapping") }
        return result.partialValue
    }

    /// Injection is internal, for deterministic mmap/partial-init failure tests.
    init(file: QwenSafetensorsFile,
         map: (Int32, Int) -> UnsafeMutableRawPointer? = {
             mmap(nil, $1, PROT_READ, MAP_SHARED, $0, 0)
         },
         unmap: @escaping (UnsafeMutableRawPointer, Int) -> Void = { _ = munmap($0, $1) }) throws {
        let length = try Self.roundedLength(file.header.fileByteCount, pageSize: Int(getpagesize()))
        try file.verifyUnchanged()
        guard let pointer = map(file.descriptor, length), pointer != MAP_FAILED else {
            throw QwenSafetensorsError.posix(operation: "mmap", code: errno)
        }
        do { try file.verifyUnchanged() } catch {
            unmap(pointer, length)
            throw error
        }
        self.file = file; self.ptr = UnsafeRawPointer(pointer)
        self.mappedByteCount = length; self.unmap = unmap
        _ = madvise(pointer, length, MADV_RANDOM)
    }

    deinit { unmap(UnsafeMutableRawPointer(mutating: ptr), mappedByteCount) }

    /// Canonical internal links are supported; the final component is opened
    /// NOFOLLOW by QwenSafetensorsFile. Parent-directory replacement remains a
    /// trusted-local-filesystem assumption, as with the resident reader.
    static func openShards(root: URL, names: Set<String>,
                           makeMapping: (QwenSafetensorsFile) throws -> QwenMappedSafetensorsFile = {
                               try QwenMappedSafetensorsFile(file: $0)
                           }) throws -> [String: QwenMappedSafetensorsFile] {
        let root = root.resolvingSymlinksInPath().standardizedFileURL
        var result: [String: QwenMappedSafetensorsFile] = [:]
        for name in names.sorted() {
            guard !name.isEmpty, !name.contains("\0"), !name.contains("\\"),
                  name != ".", name != "..", name == (name as NSString).lastPathComponent else {
                throw QwenModelDirectoryError.invalidIndex("unsafe shard name \(name)")
            }
            let url = root.appendingPathComponent(name).resolvingSymlinksInPath().standardizedFileURL
            guard url.deletingLastPathComponent().path == root.path else {
                throw QwenModelDirectoryError.invalidIndex("shard resolves outside model directory: \(name)")
            }
            result[name] = try makeMapping(QwenSafetensorsFile(resolvedURL: url))
        }
        // Local owners unwind automatically if any later shard failed.
        return result
    }

    func tensorRange(_ name: String) throws -> Range<Int> {
        guard let info = file.header.tensors[name] else {
            throw QwenSafetensorsError.invalidTensor(name: name, reason: "missing tensor")
        }
        // Shared parser proves payloadOffset + each endpoint <= fileByteCount.
        return (file.header.payloadOffset + info.dataOffsets[0])..<(file.header.payloadOffset + info.dataOffsets[1])
    }

    /// Caller supplies a live destination of at least range.count bytes. Expert
    /// paths use this bounded pread rather than a dangerous mapped-read fallback.
    func read(into destination: UnsafeMutableRawPointer, range: Range<Int>) throws {
        guard range.lowerBound >= file.header.payloadOffset,
              range.upperBound <= file.header.fileByteCount else {
            throw QwenSafetensorsError.invalidTensor(name: "pread", reason: "range exceeds payload")
        }
        if range.isEmpty { return }
        let got = QwenQ4ExpertStore.preadFully(fd: fd, destination: destination,
                                              count: range.count, offset: UInt64(range.lowerBound))
        guard got == range.count else { throw QwenSafetensorsError.shortRead(expected: range.count, actual: got) }
    }

    func bufferWindow(_ range: Range<Int>) throws -> (offset: Int, length: Int, tensorOffset: Int) {
        guard range.lowerBound >= file.header.payloadOffset,
              range.upperBound <= file.header.fileByteCount, !range.isEmpty else {
            throw QwenSafetensorsError.invalidTensor(name: "mapping", reason: "invalid mapped range")
        }
        let page = Int(getpagesize())
        let offset = range.lowerBound - range.lowerBound % page
        let length = try Self.roundedLength(range.upperBound - offset, pageSize: page)
        guard length <= mappedByteCount - offset else {
            throw QwenSafetensorsError.invalidTensor(name: "mapping", reason: "buffer exceeds mapping")
        }
        return (offset, length, range.lowerBound - offset)
    }

    func makeBuffer(device: MTLDevice, range: Range<Int>) throws -> (buffer: MTLBuffer, offset: Int) {
        try file.verifyUnchanged()
        let window = try bufferWindow(range)
        guard let buffer = device.makeBuffer(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.advanced(by: window.offset)),
            length: window.length, options: .storageModeShared,
            deallocator: { [self] _, _ in withExtendedLifetime(self) {} }) else {
            throw QwenQ4RunnerError.metalBufferAllocationFailed("Failed to make mapped tensor buffer")
        }
        // No cycle: owner has no reference back to the buffer. Default Metal
        // command buffers retain the buffer (and therefore this lease) to completion.
        return (buffer, window.tensorOffset)
    }
}
