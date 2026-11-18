import Foundation
import Testing
@testable import FlintFieldfareCore

@Suite(.serialized)
struct FlintInferenceLeaseTests {
    @Test func leaseAcquisitionAndMutualExclusion() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let lockPath = tempDir.appendingPathComponent("flint-test-lease-\(UUID().uuidString).lock").path
        defer { try? FileManager.default.removeItem(atPath: lockPath) }

        var first: FlintInferenceLease? = try FlintInferenceLease(path: lockPath)
        #expect(first != nil)

        // Second attempt at the same path must throw .inUse
        #expect(throws: FlintInferenceLeaseError.inUse) {
            try FlintInferenceLease(path: lockPath)
        }

        // Releasing first lease allows second acquisition
        first = nil
        let second = try FlintInferenceLease(path: lockPath)
        withExtendedLifetime(second) {}
    }

    @Test func rejectsUnsafeFilePaths() throws {
        let missingParent = "/nonexistent-dir-\(UUID().uuidString)/lock.lock"
        #expect(throws: FlintInferenceLeaseError.unavailable) {
            try FlintInferenceLease(path: missingParent)
        }
    }
}
