import Darwin
import Foundation
import Metal
import Testing
@testable import QwenFieldfareCore

@Suite(.serialized) struct QwenMappedSafetensorsTests {
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-mapped-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func write(_ root: URL, name: String = "shard", payload: Data = Data([11,22,33,44])) throws -> URL {
        let header = Data(#"{"pad":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"tensor":{"dtype":"U8","shape":[3],"data_offsets":[1,4]}}"#.utf8)
        var count = UInt64(header.count).littleEndian
        var data = Data(bytes: &count, count: 8); data.append(header); data.append(payload)
        let url = root.appendingPathComponent(name); try data.write(to: url); return url
    }
    private func mapped(_ url: URL) throws -> QwenMappedSafetensorsFile {
        try QwenMappedSafetensorsFile(file: QwenSafetensorsFile(resolvedURL: url))
    }

    @Test func mappedBytesAndDataRelativeOffsets() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let owner = try mapped(write(root))
        let range = try owner.tensorRange("tensor")
        #expect(range.lowerBound == owner.file.header.payloadOffset + 1)
        #expect(range.upperBound == owner.file.header.fileByteCount)
        #expect(Data(bytes: owner.ptr.advanced(by: range.lowerBound), count: range.count) == Data([22,33,44]))
        let window = try owner.bufferWindow(range)
        #expect(window.offset % Int(getpagesize()) == 0)
        #expect(window.length % Int(getpagesize()) == 0)
        #expect(window.offset + window.tensorOffset == range.lowerBound)
        #expect(window.offset + window.length <= owner.mappedByteCount)
        #expect(throws: QwenSafetensorsError.self) { try owner.tensorRange("missing") }
    }

    @Test func tensorCrossesPageAndEndsAtFileBoundary() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let page = Int(getpagesize()), start = page - 1 - 264
        let json = "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[\(start),\(start+3)]}}"
        var length = UInt64(256).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(Data(json.utf8)); data.append(Data(repeating: 32, count: 256-json.utf8.count))
        data.append(Data(repeating: 0, count: start)); data.append(Data([1,2,3]))
        let url = root.appendingPathComponent("crossing"); try data.write(to: url)
        let owner = try mapped(url); let range = try owner.tensorRange("tensor")
        #expect(range == (page-1)..<(page+2))
        let window = try owner.bufferWindow(range)
        #expect(window.offset == 0 && window.length == page*2 && window.tensorOffset == page-1)
        #expect(Data(bytes: owner.ptr.advanced(by: range.lowerBound), count: 3) == Data([1,2,3]))
    }

    @Test func unlinkAndInPlaceMutationAreObservedWithoutReopening() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); let owner = try mapped(url)
        let writer = open(url.path, O_WRONLY); #expect(writer >= 0)
        var value: UInt8 = 55
        #expect(pwrite(writer, &value, 1, off_t(owner.file.header.payloadOffset)) == 1)
        close(writer)
        #expect(throws: QwenSafetensorsError.fileChanged) { try owner.file.verifyUnchanged() }
        try FileManager.default.removeItem(at: url)
        _ = try write(root, payload: Data([99,98,97,96]))
        #expect(throws: QwenSafetensorsError.fileChanged) { try owner.file.verifyUnchanged() }
        // Descriptor still refers to the unlinked object; never read replacement.
        var byte: UInt8 = 0
        #expect(pread(owner.fd, &byte, 1, off_t(owner.file.header.payloadOffset)) == 1)
        #expect(byte == 55)
    }

    @Test func checkedPageLengthsAndInvalidRanges() throws {
        #expect(try QwenMappedSafetensorsFile.roundedLength(4096, pageSize: 4096) == 4096)
        #expect(try QwenMappedSafetensorsFile.roundedLength(4097, pageSize: 4096) == 8192)
        for count in [0,-1,Int.max] {
            #expect(throws: QwenSafetensorsError.self) { try QwenMappedSafetensorsFile.roundedLength(count, pageSize: 4096) }
        }
        #expect(throws: QwenSafetensorsError.self) { try QwenMappedSafetensorsFile.roundedLength(1, pageSize: 0) }
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let owner = try mapped(write(root)); let end = owner.file.header.fileByteCount
        for range in [-1..<1,0..<1,end..<end,end..<(end+1),0..<Int.max] {
            #expect(throws: QwenSafetensorsError.self) { try owner.bufferWindow(range) }
        }
    }

    @Test func rejectsMalformedTruncatedOversizedAndOverflowHeaders() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("bad")
        for bytes in [Data(), Data([1,2,3]), Data(repeating: 255, count: 8)] {
            try bytes.write(to: url)
            #expect(throws: QwenSafetensorsError.self) { try mapped(url) }
        }
        for header in ["{", #"{"t":{"dtype":"U8","shape":[9223372036854775807,2],"data_offsets":[0,0]}}"#,
                       #"{"t":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
                       #"{"t":{"dtype":"F128","shape":[0],"data_offsets":[0,0]}}"#] {
            var n = UInt64(header.utf8.count).littleEndian
            var bytes = Data(bytes: &n, count: 8); bytes.append(Data(header.utf8))
            try bytes.write(to: url)
            #expect(throws: QwenSafetensorsError.self) { try mapped(url) }
        }
    }

    @Test func supportsInternalLinksRejectsEscapesDirectoriesAndFifo() throws {
        let root = try directory(); let outside = try directory()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        let target = try write(root); let external = try write(outside)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: target)
        #expect(try QwenMappedSafetensorsFile.openShards(root: root, names: ["link"]).count == 1)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("escape"), withDestinationURL: external)
        #expect(mkfifo(root.appendingPathComponent("fifo").path, 0o600) == 0)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("dir"), withIntermediateDirectories: false)
        for name in ["escape","fifo","dir","../shard","/tmp/shard","a\\b","a\0b"] {
            #expect(throws: (any Error).self) { try QwenMappedSafetensorsFile.openShards(root: root, names: [name]) }
        }
    }

    @Test func neverReopensReplacedPathBetweenValidationAndMapping() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); let file = try QwenSafetensorsFile(resolvedURL: url)
        try FileManager.default.moveItem(at: url, to: root.appendingPathComponent("original"))
        _ = try write(root, payload: Data([99,98,97,96]))
        // Rename can change ctime. Either fail closed, or map original descriptor;
        // replacement bytes must never be accepted under the old header.
        do {
            let owner = try QwenMappedSafetensorsFile(file: file)
            let range = try owner.tensorRange("tensor")
            #expect(owner.ptr.load(fromByteOffset: range.lowerBound, as: UInt8.self) == 22)
        } catch let error as QwenSafetensorsError { #expect(error == .fileChanged) }
    }

    @Test func mappingFailureReleasesDescriptor() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var descriptor: Int32 = -1
        #expect(throws: QwenSafetensorsError.posix(operation: "mmap", code: ENOMEM)) {
            let file = try QwenSafetensorsFile(resolvedURL: url); descriptor = file.descriptor
            _ = try QwenMappedSafetensorsFile(file: file, map: { _,_ in errno = ENOMEM; return MAP_FAILED })
        }
        #expect(fcntl(descriptor, F_GETFD) == -1 && errno == EBADF)
    }

    @Test func mutationDuringMappingUnmapsAndThrows() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var unmapped = 0
        #expect(throws: QwenSafetensorsError.fileChanged) {
            _ = try QwenMappedSafetensorsFile(file: QwenSafetensorsFile(resolvedURL: url), map: { fd,length in
                let pointer = mmap(nil,length,PROT_READ,MAP_SHARED,fd,0)
                // No mapped access after truncation, so no intentional SIGBUS.
                #expect(truncate(url.path, 8) == 0)
                return pointer
            }, unmap: { pointer,length in unmapped += 1; #expect(munmap(pointer,length) == 0) })
        }
        #expect(unmapped == 1)
    }

    @Test func failedPartialShardSetReleasesEarlierOwners() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(root, name: "a"); _ = try write(root, name: "b")
        var descriptors: [Int32] = []; var unmapped = 0
        #expect(throws: QwenSafetensorsError.fileChanged) {
            _ = try QwenMappedSafetensorsFile.openShards(root: root, names: ["a","b"], makeMapping: { file in
                descriptors.append(file.descriptor)
                if descriptors.count == 2 { throw QwenSafetensorsError.fileChanged }
                return try QwenMappedSafetensorsFile(file: file, unmap: { p,n in unmapped += 1; _ = munmap(p,n) })
            })
        }
        #expect(unmapped == 1)
        for fd in descriptors { #expect(fcntl(fd,F_GETFD) == -1 && errno == EBADF) }
    }

    @Test func repeatedLoadUnloadAndNoUsableOwnerAfterCleanup() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var unmapped = 0
        for _ in 0..<24 {
            var owner: QwenMappedSafetensorsFile? = try QwenMappedSafetensorsFile(
                file: QwenSafetensorsFile(resolvedURL: url), unmap: { p,n in unmapped += 1; _ = munmap(p,n) })
            weak let lease = owner; let fd = try #require(owner).fd
            owner = nil
            #expect(lease == nil) // No close/unmap API allows a surviving owner to use stale pointers.
            #expect(fcntl(fd,F_GETFD) == -1 && errno == EBADF)
        }
        #expect(unmapped == 24)
    }

    @Test func observedTruncationFailsBeforeMappedAccess() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); let owner = try mapped(url)
        #expect(truncate(url.path, 8) == 0)
        #expect(throws: QwenSafetensorsError.fileChanged) { try owner.file.verifyUnchanged() }
        // Deliberately do NOT dereference the stale mapping. Concurrent access
        // during truncation is outside the immutable-file contract.
    }

    @Test func metalLeaseSurvivesOwnerReleaseUntilCommandCompletion() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var unmapped = 0; var descriptor: Int32 = -1
        try autoreleasepool {
            var owner: QwenMappedSafetensorsFile? = try QwenMappedSafetensorsFile(
                file: QwenSafetensorsFile(resolvedURL: url), unmap: { p,n in unmapped += 1; _ = munmap(p,n) })
            weak let lease = owner; descriptor = try #require(owner).fd
            let range = try #require(owner).tensorRange("tensor")
            var bufferLease: (buffer: MTLBuffer, offset: Int)? = try #require(owner).makeBuffer(device: device, range: range)
            owner = nil
            #expect(lease != nil && unmapped == 0)
            let target = try #require(device.makeBuffer(length: 3, options: .storageModeShared))
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeBlitCommandEncoder())
            encoder.copy(from: try #require(bufferLease).buffer, sourceOffset: try #require(bufferLease).offset,
                         to: target, destinationOffset: 0, size: 3)
            encoder.endEncoding()
            bufferLease = nil // The command must now retain the buffer/owner.
            #expect(lease != nil && unmapped == 0)
            command.commit(); command.waitUntilCompleted()
            #expect(command.status == .completed)
            #expect(Data(bytes: target.contents(), count: 3) == Data([22,33,44]))
            // Metal may release resources immediately on completion; no lease
            // is required after GPU access has ended. Pool exit must release all.
        }
        #expect(unmapped == 1)
        #expect(fcntl(descriptor,F_GETFD) == -1 && errno == EBADF)
    }

    @Test func failedInitializationAfterMetalBufferReleasesLease() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var unmapped = 0; var descriptor: Int32 = -1
        _ = autoreleasepool {
            #expect(throws: QwenSafetensorsError.fileChanged) {
                let owner = try QwenMappedSafetensorsFile(file: QwenSafetensorsFile(resolvedURL: url),
                    unmap: { p,n in unmapped += 1; _ = munmap(p,n) })
                descriptor = owner.fd
                let lease = try owner.makeBuffer(device: device, range: owner.tensorRange("tensor"))
                withExtendedLifetime(lease) {}
                throw QwenSafetensorsError.fileChanged
            }
        }
        #expect(unmapped == 1)
        #expect(fcntl(descriptor,F_GETFD) == -1 && errno == EBADF)
    }

    @Test func expertPreadRejectsInvalidRangesAndShortReadsWithoutMappedFallback() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); let owner = try mapped(url)
        let range = try owner.tensorRange("tensor")
        var bytes = [UInt8](repeating: 0, count: 3)
        try bytes.withUnsafeMutableBytes { try owner.read(into: $0.baseAddress!, range: range) }
        #expect(bytes == [22,33,44])
        #expect(throws: QwenSafetensorsError.self) {
            try bytes.withUnsafeMutableBytes { try owner.read(into: $0.baseAddress!, range: 0..<1) }
        }
        #expect(truncate(url.path, 8) == 0)
        #expect(throws: QwenSafetensorsError.shortRead(expected: 3, actual: 0)) {
            try bytes.withUnsafeMutableBytes { try owner.read(into: $0.baseAddress!, range: range) }
        }
    }

    @Test func rejectsUnsafeFixedConsumerExtents() throws {
        let name = "language_model.model.norm.weight"
        for offsets in [[0,2], [0,4098], [-1,4095], [Int.min,Int.max], []] {
            #expect(throws: QwenSafetensorsError.self) {
                try QwenQ4ActiveExpertStreamingRunner.validateMappedTensor(name: name,
                    info: QwenTensorInfo(shape: [2048], dtype: "BF16", dataOffsets: offsets))
            }
        }
        try QwenQ4ActiveExpertStreamingRunner.validateMappedTensor(name: name,
            info: QwenTensorInfo(shape: [2048], dtype: "BF16", dataOffsets: [0,4096]))
        #expect(throws: QwenSafetensorsError.self) {
            try QwenQ4ActiveExpertStreamingRunner.validateMappedTensor(name: name,
                info: QwenTensorInfo(shape: [2048], dtype: "U16", dataOffsets: [0,4096]))
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWENFIELDFARE_READER_CHECKPOINT"] != nil))
    func localCheckpointMappingIntegration() throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWENFIELDFARE_READER_CHECKPOINT"])
        let root = URL(fileURLWithPath: path)
        let names = Set((1...4).map { String(format: "model-%05d-of-00004.safetensors", $0) })
        let device = try #require(MTLCreateSystemDefaultDevice())
        try autoreleasepool {
            let owners = try QwenMappedSafetensorsFile.openShards(root: root, names: names)
            #expect(owners.count == 4)
            for name in names.sorted() {
                let owner = try #require(owners[name])
                print("S1-01c mapped \(name): header=\(owner.file.header.headerByteCount) bytes, file=\(owner.file.header.fileByteCount) bytes")
                for (tensor,info) in owner.file.header.tensors where tensor.hasPrefix("language_model.") {
                    try QwenQ4ActiveExpertStreamingRunner.validateMappedTensor(name: tensor, info: info)
                }
                if owner.file.header.tensors["language_model.lm_head.weight"] != nil {
                    let range = try owner.tensorRange("language_model.lm_head.weight")
                    let lease = try owner.makeBuffer(device: device, range: range)
                    #expect(lease.buffer.length >= range.count)
                }
                try owner.file.verifyUnchanged()
            }
        }
    }
}
