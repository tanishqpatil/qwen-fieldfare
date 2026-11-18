import Darwin
import Foundation
import Metal

/// Loader responsible for populating permanently resident weights for Recursive Flint 40B-A2.5B.
/// Direct zero-copy I/O: transfers bytes directly from Safetensors file descriptors into
/// single-copy Metal shared buffers (or aligned memory), achieving zero peak temporary staging allocation.
public enum FlintResidentLoader {

    /// Summary of the storage backend used for resident weights.
    public enum BackendSummary: String, Sendable {
        case allMetal
        case cpuOnly
    }

    /// Internal failure-injection seam for testing partial-load cleanup and error recovery.
    internal struct FailureInjection: Sendable {
        let failAfterLoadingTensorCount: Int
        let errorToThrow: Error
        let onBufferAllocated: (@Sendable (FlintResidentBuffer) -> Void)?

        init(
            failAfterLoadingTensorCount: Int,
            errorToThrow: Error,
            onBufferAllocated: (@Sendable (FlintResidentBuffer) -> Void)? = nil
        ) {
            self.failAfterLoadingTensorCount = failAfterLoadingTensorCount
            self.errorToThrow = errorToThrow
            self.onBufferAllocated = onBufferAllocated
        }
    }

    /// Detailed metrics captured during resident weight loading.
    public struct LoadMetrics: Sendable {
        public let residentTensorCount: Int
        public let residentByteCount: Int
        public let routedExpertByteCount: Int
        public let loadDurationSeconds: Double
        public let peakTemporaryAllocatedBytes: Int
        public let categoryByteCounts: [FlintResidentCategory: Int]
        public let backendSummary: BackendSummary

        public var usesMetal: Bool {
            backendSummary == .allMetal
        }
    }

    /// Loads the complete set of 1,328 permanently resident tensors from open Safetensors shards.
    ///
    /// - Parameters:
    ///   - validatedDirectory: Validated Flint model directory.
    ///   - shardFiles: Open, verified Safetensors shard files.
    ///   - storageMode: Mode governing buffer allocation (.gpuRequired or .cpuOnly).
    ///   - device: Metal device to allocate shared unified memory buffers (defaults to system default device).
    /// - Returns: Fully populated, immutable FlintResidentWeights and measurement metrics.
    public static func loadResidentWeights(
        validatedDirectory: FlintModelDirectory.ValidatedDirectory,
        shardFiles: [String: FlintSafetensorsFile],
        storageMode: FlintStorageMode = .gpuRequired,
        device: MTLDevice? = MTLCreateSystemDefaultDevice()
    ) throws -> (weights: FlintResidentWeights, metrics: LoadMetrics) {
        try loadResidentWeightsInternal(
            validatedDirectory: validatedDirectory,
            shardFiles: shardFiles,
            storageMode: storageMode,
            device: device,
            metalBufferAllocator: nil,
            failureInjection: nil
        )
    }

    /// Internal loading implementation supporting test seams for failure injection and allocator overrides.
    internal static func loadResidentWeightsInternal(
        validatedDirectory: FlintModelDirectory.ValidatedDirectory,
        shardFiles: [String: FlintSafetensorsFile],
        storageMode: FlintStorageMode = .gpuRequired,
        device: MTLDevice? = MTLCreateSystemDefaultDevice(),
        metalBufferAllocator: ((Int) -> MTLBuffer?)? = nil,
        failureInjection: FailureInjection? = nil
    ) throws -> (weights: FlintResidentWeights, metrics: LoadMetrics) {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Build and validate the complete resident manifest against shard headers
        let manifest = try FlintResidentManifest.build(config: validatedDirectory.config)
        try manifest.validateAgainstShards(shardFiles, weightMap: validatedDirectory.weightMap)

        var loadedTensors = [String: FlintResidentTensor]()
        loadedTensors.reserveCapacity(manifest.specs.count)

        var loadedRoutedBytes = 0

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

                // Allocate single authoritative buffer with explicit storage mode
                let buffer = try FlintResidentBuffer(
                    byteCount: spec.expectedByteCount,
                    storageMode: storageMode,
                    device: device,
                    metalBufferAllocator: metalBufferAllocator
                )

                // Direct I/O from shard file descriptor into allocated buffer memory (zero temporary staging)
                try shard.readTensorPayload(name, into: buffer.mutablePointer)

                let residentTensor = FlintResidentTensor(spec: spec, buffer: buffer)
                loadedTensors[name] = residentTensor

                // Test hook: inform observer and inject failure if requested
                if let fi = failureInjection {
                    fi.onBufferAllocated?(buffer)
                    if loadedTensors.count >= fi.failAfterLoadingTensorCount {
                        throw fi.errorToThrow
                    }
                }
            }

            // Verify backend consistency: no silent mixing of GPU and CPU backends
            let expectedBackend: FlintResidentBuffer.Backend = (storageMode == .gpuRequired) ? .metalShared : .cpuAlignedHeap
            for (tName, tensor) in loadedTensors {
                if tensor.buffer.byteCount > 0 && tensor.buffer.backend != expectedBackend {
                    throw FlintResidentError.backendMismatch(
                        expected: expectedBackend.rawValue,
                        actual: "\(tName) has \(tensor.buffer.backend.rawValue)"
                    )
                }
            }

            let weights = try FlintResidentWeights(tensors: loadedTensors, manifest: manifest)
            let duration = CFAbsoluteTimeGetCurrent() - startTime

            let backendSummary: BackendSummary = (storageMode == .gpuRequired) ? .allMetal : .cpuOnly

            let metrics = LoadMetrics(
                residentTensorCount: weights.tensorCount,
                residentByteCount: weights.totalByteCount,
                routedExpertByteCount: loadedRoutedBytes,
                loadDurationSeconds: duration,
                peakTemporaryAllocatedBytes: 0, // Direct zero-copy I/O: 0 temporary staging bytes allocated
                categoryByteCounts: weights.categoryByteCounts,
                backendSummary: backendSummary
            )

            return (weights, metrics)
        } catch {
            // Partial-failure cleanup: immediately drop all allocated buffers so ARC and deinit free memory
            loadedTensors.removeAll(keepingCapacity: false)
            throw error
        }
    }
}
