import Darwin
import Foundation
import Testing
@testable import QwenFieldfareCore

@Suite(.serialized) struct QwenInferenceLeaseTests {
    @Test func exclusiveDescriptorsAndARCRelease() throws {
        let path=FileManager.default.temporaryDirectory.appendingPathComponent("lease-\(UUID())").path
        defer {unlink(path)}
        var first:QwenInferenceLease?=try QwenInferenceLease(path:path)
        weak let weakOwner=first
        #expect(throws:QwenInferenceLeaseError.inUse) {try QwenInferenceLease(path:path)}
        first=nil;#expect(weakOwner == nil)
        let second=try QwenInferenceLease(path:path);withExtendedLifetime(second) {}
    }
    @Test func rejectsUnsafeLockFilesWithoutBlocking() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("lease-files-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let fifo=root.appendingPathComponent("fifo");#expect(mkfifo(fifo.path,0o600) == 0)
        let file=root.appendingPathComponent("file");try Data().write(to:file);#expect(chmod(file.path,0o644) == 0)
        let link=root.appendingPathComponent("link");try FileManager.default.createSymbolicLink(at:link,withDestinationURL:file)
        for path in [fifo.path,root.path,file.path,link.path] {#expect(throws:QwenInferenceLeaseError.self) {try QwenInferenceLease(path:path)}}
    }
    @Test func bothRunnerEntryPointsRejectBeforeMetalOrModelIO() throws {
        let owner=try QwenInferenceLease();defer {withExtendedLifetime(owner) {}}
        #expect(throws:QwenInferenceLeaseError.inUse) {try QwenQ4ActiveExpertStreamingRunner(modelPath:"/missing",maxSeqLen:8,expertCacheCapacity:320)}
        #expect(throws:QwenInferenceLeaseError.inUse) {try QwenQ4ChatRunner(modelPath:"/missing",maxSeqLen:8)}
    }
}
