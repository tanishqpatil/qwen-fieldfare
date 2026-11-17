import Darwin
import Foundation
import Metal
import Testing
@testable import QwenFieldfareCore

@Suite(.serialized) struct QwenFallbackReaderTests {
    private let tensor = "language_model.model.norm.weight"
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-fallback-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func write(_ root: URL, header: String? = nil, payload: Data? = nil) throws -> URL {
        let text = header ?? "{\"pad\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},\"\(tensor)\":{\"dtype\":\"BF16\",\"shape\":[2048],\"data_offsets\":[3,4099]}}"
        var n = UInt64(text.utf8.count).littleEndian
        var bytes = Data(bytes: &n, count: 8); bytes.append(Data(text.utf8))
        bytes.append(payload ?? (Data([11,22,33,0x80,0x3f]) + Data(repeating: 0, count: 4094)))
        let url = root.appendingPathComponent("shard.safetensors"); try bytes.write(to: url)
        let index: [String:Any] = ["weight_map":[tensor:"shard.safetensors"]]
        try JSONSerialization.data(withJSONObject: index).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return url
    }
    private func open(_ root: URL, name: String = "shard.safetensors") throws -> [String:QwenMappedSafetensorsFile] {
        try QwenMappedSafetensorsFile.openShards(root: root, names: [name])
    }
    private func slice(_ owners: [String:QwenMappedSafetensorsFile], name: String = "shard.safetensors") throws -> QwenQ4ChatRunner.TensorSlice {
        try QwenQ4ChatRunner.validatedTensorSlice(tensor, weightMap: [tensor:name], shards: owners)
    }

    @Test func validFallbackSliceUsesDataRegionAndExactEnd() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(root); let owners = try open(root)
        let owner = try #require(owners["shard.safetensors"]); let tensor = try slice(owners)
        #expect(tensor.size == 4096)
        #expect(tensor.fileOffset == owner.file.header.payloadOffset + 3)
        #expect(tensor.fileOffset + tensor.size == owner.file.header.fileByteCount)
        #expect(owner.ptr.loadUnaligned(fromByteOffset: tensor.fileOffset, as: UInt16.self) == 0x3f80)
    }

    @Test func rejectsMalformedMetadataAndTruncationBeforeMapping() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        for header in ["{", #"{"__metadata__":{"key":false}}"#,
                       #"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0]}}"#,
                       #"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
                       #"{"x":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,0]}}"#,
                       #"{"x":{"dtype":"F128","shape":[0],"data_offsets":[0,0]}}"#] {
            _ = try write(root, header: header, payload: Data())
            #expect(throws: QwenSafetensorsError.self) { try open(root) }
        }
        let url = try write(root)
        for count in [0,7,8] {
            #expect(truncate(url.path, off_t(count)) == 0)
            #expect(throws: QwenSafetensorsError.self) { try open(root) }
        }
        #expect(QwenSafetensorsHeader.maximumHeaderBytes == 100_000_000)
        var tooLarge = UInt64(100_000_001).littleEndian
        try Data(bytes: &tooLarge, count: 8).write(to: url)
        #expect(throws: QwenSafetensorsError.invalidHeaderLength) { try open(root) }
    }

    @Test func rejectsWrongConsumerDtypeAndByteExtentRecoverably() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        for (dtype,count) in [("U16",2048),("BF16",1),("BF16",2049)] {
            let json = "{\"\(tensor)\":{\"dtype\":\"\(dtype)\",\"shape\":[\(count)],\"data_offsets\":[0,\(count*2)]}}"
            _ = try write(root, header: json, payload: Data(repeating: 0, count: count*2))
            let owners = try open(root)
            #expect(throws: QwenSafetensorsError.self) { try slice(owners) }
        }
        _ = try write(root)
        #expect(try slice(open(root)).size == 4096)
        #expect(throws: QwenQ4RunnerError.self) {
            try QwenQ4ChatRunner.validatedTensorSlice("missing", weightMap: [:], shards: [:])
        }
    }

    @Test func preservesInternalLinksRejectsUnsafePathsAndFileTypes() throws {
        let root = try directory(); let outside = try directory()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        let url = try write(root); let external = try write(outside)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: url)
        #expect(try slice(open(root,name: "link"), name: "link").size == 4096)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("escape"), withDestinationURL: external)
        #expect(mkfifo(root.appendingPathComponent("fifo").path, 0o600) == 0)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("dir"), withIntermediateDirectories: false)
        for name in ["escape","fifo","dir","../shard.safetensors"] {
            #expect(throws: (any Error).self) { try open(root,name: name) }
        }
    }

    @Test func replacementAndMutationNeverRedirectCachedSlice() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); let owners = try open(root)
        let before = try slice(owners)
        try FileManager.default.moveItem(at: url, to: root.appendingPathComponent("original"))
        _ = try write(root, payload: Data(repeating: 0xff, count: 4099))
        // Rename changes ctime on this host. If unchanged, the retained handle
        // must still read original bytes; replacement data is never authorized.
        do {
            let after = try slice(owners)
            #expect(after.fileOffset == before.fileOffset)
            let owner = try #require(owners[after.shardName])
            #expect(owner.ptr.loadUnaligned(fromByteOffset: after.fileOffset, as: UInt16.self) == 0x3f80)
        } catch let error as QwenSafetensorsError { #expect(error == .fileChanged) }
        let fresh = try open(root)
        #expect(truncate(url.path, 8) == 0)
        #expect(throws: QwenSafetensorsError.fileChanged) { try slice(fresh) }
        // Never touch the truncated mapping; immutable checkpoint contract.
    }

    @Test func fallbackSliceMetalConsumerOwnsMappingUntilReleased() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try write(root); let device = try #require(MTLCreateSystemDefaultDevice())
        var fd: Int32 = -1
        try autoreleasepool {
            var owners = try open(root)
            let tensor = try slice(owners)
            fd = try #require(owners[tensor.shardName]).fd
            let (buffer,offset) = try #require(owners[tensor.shardName]).makeBuffer(
                device: device, range: tensor.fileOffset..<(tensor.fileOffset+tensor.size))
            weak let lease = owners[tensor.shardName]
            owners.removeAll()
            #expect(lease != nil)
            #expect(buffer.contents().loadUnaligned(fromByteOffset: offset, as: UInt16.self) == 0x3f80)
        }
        #expect(fcntl(fd,F_GETFD) == -1 && errno == EBADF)
    }

    @Test func failedFallbackInitializationReleasesDescriptorsAcrossRetries() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = try write(root); var identity = stat(); #expect(stat(url.path,&identity) == 0)
        func matchingDescriptors() -> Int {
            (0..<1024).reduce(0) { count,fd in
                var info = stat()
                return count + ((fstat(Int32(fd),&info) == 0 && info.st_dev == identity.st_dev && info.st_ino == identity.st_ino) ? 1 : 0)
            }
        }
        for _ in 0..<3 {
            // Missing embedding fails after the shard has been mapped, before
            // state/cache population. This exercises the real fallback init.
            #expect(throws: QwenQ4RunnerError.self) {
                try autoreleasepool { try QwenQ4ChatRunner(modelPath: root.path, maxSeqLen: 8) }
            }
            #expect(matchingDescriptors() == 0)
        }
    }

    @Test func contextAllocationArithmeticRejectsOverflowBeforeAllocation() throws {
        for context in [0,-1,Int.min,Int.max,Int.max/2048+1] {
            #expect(throws: QwenQ4RunnerError.self) { try QwenQ4ChatRunner.contextAllocationLengths(context) }
        }
        for context in [1,8,2048,8192,65536,Int.max/2048] {
            let lengths = try QwenQ4ChatRunner.contextAllocationLengths(context)
            #expect(lengths.scores == context*64 && lengths.kv == context*2048)
        }
        #expect(throws: QwenQ4RunnerError.self) { try QwenQ4ChatRunner(modelPath: "/missing", maxSeqLen: Int.max) }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWENFIELDFARE_FALLBACK_CHECKPOINT"] != nil))
    func localFallbackCheckpointIntegration() throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWENFIELDFARE_FALLBACK_CHECKPOINT"])
        let device = try #require(MTLCreateSystemDefaultDevice())
        func footprint() -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let result = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
                }
            }
            return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
        }
        print("Fallback pre-load footprint=\(footprint()), Metal=\(device.currentAllocatedSize) bytes")
        try autoreleasepool {
            let runner = try QwenQ4ChatRunner(modelPath: path, maxSeqLen: 8, mode: .memorySaver,
                memoryBudgetGiB: 5, hardLimitGiB: 6, layerCacheCapacity: 1)
            #expect(runner.layerKinds.count == 40 && runner.cacheCreations == 0)
            print("Fallback loaded footprint=\(footprint()), Metal=\(device.currentAllocatedSize) bytes")
            for (token,position) in [(-1,0),(248320,0),(0,-1),(0,8),(Int.max,Int.max)] {
                #expect(throws: QwenQ4RunnerError.self) { try runner.forwardToken(tokenId: token, tokenPos: position) }
            }
            // Public fixture's first prompt token, two independent reset forwards.
            // This is a bounded IO/state smoke test, not teacher-forced parity.
            let first = try runner.forwardToken(tokenId: 248045, tokenPos: 0)
            runner.resetState()
            let second = try runner.forwardToken(tokenId: 248045, tokenPos: 0)
            #expect(first == second && first >= 0 && first < 248320)
            #expect(runner.cacheCreations == 80 && runner.cacheEvictions == 79)
            print("Fallback predictions=\(first),\(second); forward calls=2, input tokens/call=1; context=8, layer cache=1")
            print("Fallback cache creations=\(runner.cacheCreations), evictions=\(runner.cacheEvictions), hits=\(runner.cacheHits), misses=\(runner.cacheMisses)")
            print("Fallback after forwards footprint=\(footprint()), Metal=\(device.currentAllocatedSize) bytes")
        }
        print("Fallback after unload footprint=\(footprint()), Metal=\(device.currentAllocatedSize) bytes")
    }
}
