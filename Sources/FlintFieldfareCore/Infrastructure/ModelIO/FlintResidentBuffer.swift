import Darwin
import Foundation
import Metal

/// Explicit storage modes for resident weight buffers.
public enum FlintStorageMode: String, Sendable, CaseIterable {
    /// Strict GPU-required allocation: allocates page-aligned Metal shared memory
    /// (MTLBuffer with .storageModeShared). Fails explicitly if Metal is unavailable
    /// or if allocation fails. Never silently falls back to CPU heap.
    case gpuRequired

    /// Dedicated CPU-only allocation: allocates 64-byte aligned native heap memory.
    /// Used intentionally for unit testing and non-GPU environments.
    case cpuOnly
}

/// Errors specific to resident weight allocations and storage backends.
public enum FlintResidentError: Error, LocalizedError, Equatable {
    case invalidByteCount(Int)
    case metalDeviceUnavailable
    case metalAllocationFailed(byteCount: Int)
    case invalidMetalBuffer(reason: String)
    case backendMismatch(expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .invalidByteCount(let bytes):
            return "Invalid resident buffer byte count: \(bytes) (must be non-negative)"
        case .metalDeviceUnavailable:
            return "Metal device is unavailable for GPU-required resident weight allocation"
        case .metalAllocationFailed(let bytes):
            return "Failed to allocate Metal shared buffer of \(bytes) bytes (.storageModeShared)"
        case .invalidMetalBuffer(let reason):
            return "Invalid Metal buffer allocated: \(reason)"
        case .backendMismatch(let expected, let actual):
            return "Resident storage backend mismatch: expected \(expected), got \(actual)"
        }
    }
}

/// A single authoritative memory allocation for a resident model weight tensor.
///
/// ### Concurrency and Immutability Model:
/// - **Controlled Loading Phase**: Writes to the buffer occur strictly during initial shard loading
///   via the internal `mutablePointer` using POSIX `pread`.
/// - **Post-Loading Immutability**: Once returned inside `FlintResidentWeights`, the buffer is
///   strictly immutable. External consumers access data exclusively through `readOnlyPointer`
///   or pass `metalBuffer` to Metal compute command encoders.
/// - **Thread Safety**: Concurrent inference threads can safely read `readOnlyPointer` and dispatch
///   concurrent GPU command buffers referencing `metalBuffer` without locks or data races.
/// - **Lifecycle**: Metal shared buffers are managed by ARC; CPU heap allocations are freed on deinit.
public final class FlintResidentBuffer: @unchecked Sendable {
    public enum Backend: String, Sendable {
        case metalShared
        case cpuAlignedHeap
    }

    public let byteCount: Int
    public let backend: Backend
    public let metalBuffer: MTLBuffer?
    internal let mutablePointer: UnsafeMutableRawBufferPointer

    /// Public read-only pointer for host CPU access, preventing external mutation.
    public var readOnlyPointer: UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(mutablePointer)
    }

    public var baseAddress: UnsafeRawPointer? {
        readOnlyPointer.baseAddress
    }

    public var usesMetal: Bool {
        backend == .metalShared
    }

    /// Initializes a resident weight buffer with explicit storage mode.
    ///
    /// - Parameters:
    ///   - byteCount: Non-negative byte size of the tensor.
    ///   - storageMode: Mode governing memory allocation (`.gpuRequired` or `.cpuOnly`).
    ///   - device: Optional Metal device (defaults to system default device in `.gpuRequired` mode).
    public convenience init(
        byteCount: Int,
        storageMode: FlintStorageMode = .gpuRequired,
        device: MTLDevice? = nil
    ) throws {
        try self.init(
            byteCount: byteCount,
            storageMode: storageMode,
            device: device,
            metalBufferAllocator: nil
        )
    }

    /// Internal initializer with test seam for injecting custom or failing Metal buffer allocations.
    internal init(
        byteCount: Int,
        storageMode: FlintStorageMode = .gpuRequired,
        device: MTLDevice? = nil,
        metalBufferAllocator: ((Int) -> MTLBuffer?)?
    ) throws {
        guard byteCount >= 0 else {
            throw FlintResidentError.invalidByteCount(byteCount)
        }
        self.byteCount = byteCount

        if byteCount == 0 {
            self.backend = (storageMode == .gpuRequired) ? .metalShared : .cpuAlignedHeap
            self.metalBuffer = nil
            self.mutablePointer = UnsafeMutableRawBufferPointer(start: nil, count: 0)
            return
        }

        switch storageMode {
        case .gpuRequired:
            let dev = device ?? MTLCreateSystemDefaultDevice()
            guard let metalDevice = dev else {
                throw FlintResidentError.metalDeviceUnavailable
            }
            let allocatedBuf: MTLBuffer?
            if let customAllocator = metalBufferAllocator {
                allocatedBuf = customAllocator(byteCount)
            } else {
                allocatedBuf = metalDevice.makeBuffer(length: byteCount, options: .storageModeShared)
            }
            guard let buf = allocatedBuf else {
                throw FlintResidentError.metalAllocationFailed(byteCount: byteCount)
            }
            // Enforce allocation safety invariants before binding memory pointers
            guard buf.length >= byteCount else {
                throw FlintResidentError.invalidMetalBuffer(
                    reason: "Allocated Metal buffer length \(buf.length) B is less than requested \(byteCount) B"
                )
            }
            guard buf.storageMode == .shared else {
                throw FlintResidentError.invalidMetalBuffer(
                    reason: "Allocated Metal buffer storageMode must be .storageModeShared (got \(buf.storageMode.rawValue))"
                )
            }
            let contents = buf.contents()
            self.metalBuffer = buf
            self.backend = .metalShared
            self.mutablePointer = UnsafeMutableRawBufferPointer(start: contents, count: byteCount)

        case .cpuOnly:
            let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
            self.metalBuffer = nil
            self.backend = .cpuAlignedHeap
            self.mutablePointer = UnsafeMutableRawBufferPointer(start: rawPtr, count: byteCount)
        }
    }

    deinit {
        if backend == .cpuAlignedHeap, let base = mutablePointer.baseAddress {
            base.deallocate()
        }
        // Metal shared buffer is automatically released by ARC
    }

    /// Copies bytes out into Swift Data for hash verification or testing.
    public func copyData() -> Data {
        guard let base = readOnlyPointer.baseAddress, byteCount > 0 else { return Data() }
        return Data(bytes: base, count: byteCount)
    }

    /// Reads a slice into Swift Data.
    public func copySlice(range: Range<Int>) -> Data {
        guard range.lowerBound >= 0, range.upperBound <= byteCount, !range.isEmpty else { return Data() }
        guard let base = readOnlyPointer.baseAddress else { return Data() }
        return Data(bytes: base.advanced(by: range.lowerBound), count: range.count)
    }
}
