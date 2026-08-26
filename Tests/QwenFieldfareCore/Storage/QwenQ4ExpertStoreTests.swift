import Foundation
import Testing
@testable import QwenFieldfareCore

@Suite
struct QwenQ4ExpertStoreTests {
    private let payloadHash = String(repeating: "a", count: 64)
    private let sourceHash = String(repeating: "b", count: 64)

    @Test
    func layerHeaderRoundTripsWithoutNativeAlignmentAssumptions() throws {
        let value = QwenQ4ExpertLayerHeader(
            layerIndex: 17,
            expertsPerLayer: QwenQ4ExpertStore.expertsPerLayer,
            expertByteLength: QwenQ4ExpertStore.expertByteLength,
            payloadByteLength: QwenQ4ExpertStore.payloadByteLength,
            payloadSHA256: payloadHash,
            sourceIndexSHA256: sourceHash)

        let encoded = try value.encode()

        #expect(encoded.count == QwenQ4ExpertLayerHeader.byteLength)
        #expect(try QwenQ4ExpertLayerHeader.decode(encoded) == value)
    }

    @Test
    func layerHeaderRejectsCorruptMagicAndDigest() throws {
        let value = QwenQ4ExpertLayerHeader(
            layerIndex: 0,
            expertsPerLayer: QwenQ4ExpertStore.expertsPerLayer,
            expertByteLength: QwenQ4ExpertStore.expertByteLength,
            payloadByteLength: QwenQ4ExpertStore.payloadByteLength,
            payloadSHA256: payloadHash,
            sourceIndexSHA256: sourceHash)
        var encoded = try value.encode()
        encoded[0] ^= 0xff
        #expect(throws: QwenQ4ExpertStoreError.self) {
            try QwenQ4ExpertLayerHeader.decode(encoded)
        }

        let invalidDigest = QwenQ4ExpertLayerHeader(
            layerIndex: 0,
            expertsPerLayer: QwenQ4ExpertStore.expertsPerLayer,
            expertByteLength: QwenQ4ExpertStore.expertByteLength,
            payloadByteLength: QwenQ4ExpertStore.payloadByteLength,
            payloadSHA256: String(repeating: "z", count: 64),
            sourceIndexSHA256: sourceHash)
        #expect(throws: QwenQ4ExpertStoreError.self) {
            try invalidDigest.encode()
        }
    }

    @Test
    func manifestRequiresEveryCanonicalLayerExactlyOnce() throws {
        let validLayers = (0..<QwenQ4ExpertStore.layerCount).map {
            QwenQ4ExpertStoreManifest.Layer(
                layerIndex: $0,
                filename: QwenQ4ExpertStore.layerFilename($0),
                fileSize: QwenQ4ExpertStore.layerFileByteLength,
                payloadSHA256: payloadHash)
        }
        let valid = QwenQ4ExpertStoreManifest(
            schemaVersion: Int(QwenQ4ExpertLayerHeader.schemaVersion),
            sourceIndexSHA256: sourceHash,
            layers: validLayers)
        try QwenQ4ExpertStore.validateManifest(valid)

        let missing = QwenQ4ExpertStoreManifest(
            schemaVersion: valid.schemaVersion,
            sourceIndexSHA256: sourceHash,
            layers: Array(validLayers.dropLast()))
        #expect(throws: QwenQ4ExpertStoreError.self) {
            try QwenQ4ExpertStore.validateManifest(missing)
        }

        var duplicateLayers = validLayers
        duplicateLayers[39] = validLayers[38]
        let duplicate = QwenQ4ExpertStoreManifest(
            schemaVersion: valid.schemaVersion,
            sourceIndexSHA256: sourceHash,
            layers: duplicateLayers)
        #expect(throws: QwenQ4ExpertStoreError.self) {
            try QwenQ4ExpertStore.validateManifest(duplicate)
        }
    }
}
