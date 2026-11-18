import Darwin
import Foundation
import Metal

/// Loader responsible for populating permanently resident weights for Recursive Flint 40B-A2.5B.
/// Direct zero-copy I/O: transfers bytes directly from Safetensors file descriptors into
/// single-copy Metal shared buffers (or aligned memory), achieving zero peak temporary staging allocation.
public enum FlintResidentLoader {

    /// Detailed metrics captured during resident weight loading.
    public struct LoadMetrics: Sendable {
        public let residentTensorCount: Int
        public let residentByteCount: Int
        public let routedExpertByteCount: Int
        public let loadDurationSeconds: Double
        public let peakTemporaryAllocatedBytes: Int
        public let categoryByteCounts: [FlintResidentCategory: Int]
        public let usesMetal: Bool
    }

    /// Loads the complete set of 1,328 permanently resident tensors from open Safetensors shards.
    ///
    /// - Parameters:
    ///   - validatedDirectory: Validated Flint model directory.
    ///   - shardFiles: Open, verified Safetensors shard files.
    ///   - device: Metal device to allocate shared unified memory buffers (defaults to system default device).
    /// - Returns: Fully populated, immutable FlintResidentWeights and measurement metrics.
    public static func loadResidentWeights(
        validatedDirectory: FlintModelDirectory.ValidatedDirectory,
        shardFiles: [String: FlintSafetensorsFile],
        device: MTLDevice? = MTLCreateSystemDefaultDevice()
    ) throws -> (weights: FlintResidentWeights, metrics: LoadMetrics) {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Build and validate the complete resident manifest against shard headers
        let manifest = try FlintResidentManifest.build(config: validatedDirectory.config)
        try manifest.validateAgainstShards(shardFiles, weightMap: validatedDirectory.weightMap)

        var loadedTensors = [String: FlintResidentTensor]()
        loadedTensors.reserveCapacity(manifest.specs.count)

        var loadedRoutedBytes = 0
        var usesMetal = false

        do {
            // 2. Iterate in declared manifest order and load each tensor directly into its buffer
            for name in manifest.orderedNames {
                guard let spec = manifest.specs[name] else {
                    throw FlintSafetensorsError.tensorNotFound(name: "Manifest spec missing for \(name)")
                }

                // Strict invariant: verify no switch_mlp tensor is loaded
                if name.contains("switch_mlp") {
                    loadedRoutedBytes += spec.expectedByteCount
                    throw FlintSafetensorsError.invalidTensor(
                        name: name,
                        reason: "Routed expert tensor unexpectedly encountered during resident loading"
                    )
                }

                guard let shardName = validatedDirectory.weightMap[name] else {
                    throw FlintSafetensorsError.tensorNotFound(name: "Tensor \(name) not in weight map")
                }
                guard let shard = shardFiles[shardName] else {
                    throw FlintModelDirectoryError.missingFile("Shard file \(shardName) not open")
                }

                // Allocate single authoritative buffer
                let buffer = FlintResidentBuffer(byteCount: spec.expectedByteCount, device: device)
                if buffer.usesMetal {
                    usesMetal = true
                }

                // Direct I/O from shard file descriptor into allocated buffer memory (zero temporary staging)
                try shard.readTensorPayload(name, into: buffer.pointer)

                let residentTensor = FlintResidentTensor(spec: spec, buffer: buffer)
                loadedTensors[name] = residentTensor
            }

            let weights = try FlintResidentWeights(tensors: loadedTensors, manifest: manifest)
            let duration = CFAbsoluteTimeGetCurrent() - startTime

            let metrics = LoadMetrics(
                residentTensorCount: weights.tensorCount,
                residentByteCount: weights.totalByteCount,
                routedExpertByteCount: loadedRoutedBytes,
                loadDurationSeconds: duration,
                peakTemporaryAllocatedBytes: 0, // Direct I/O: 0 temporary staging bytes allocated
                categoryByteCounts: weights.categoryByteCounts,
                usesMetal: usesMetal
            )

            return (weights, metrics)
        } catch {
            // Partial-failure cleanup: immediately drop all allocated buffers so ARC frees memory
            loadedTensors.removeAll(keepingCapacity: false)
            throw error
        }
    }
}
