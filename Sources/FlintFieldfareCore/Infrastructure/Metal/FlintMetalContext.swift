import Foundation
import Metal

/// Errors specific to Metal device, queue, and compute pipeline operations.
public enum FlintMetalError: Error, LocalizedError, Equatable {
    case deviceUnavailable
    case commandQueueUnavailable
    case libraryCompilationFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case commandBufferFailed(String)
    case bufferTooSmall(name: String, actual: Int, required: Int)
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .deviceUnavailable:
            return "Metal device is unavailable on this system"
        case .commandQueueUnavailable:
            return "Failed to create Metal command queue"
        case .libraryCompilationFailed(let reason):
            return "Metal library compilation failed: \(reason)"
        case .functionNotFound(let name):
            return "Metal kernel function not found: \(name)"
        case .pipelineCreationFailed(let reason):
            return "Failed to create compute pipeline state: \(reason)"
        case .commandBufferFailed(let reason):
            return "Metal command buffer execution failed: \(reason)"
        case .bufferTooSmall(let name, let actual, let required):
            return "Buffer '\(name)' length (\(actual) B) is smaller than required (\(required) B)"
        case .invalidParameter(let reason):
            return "Invalid kernel parameter: \(reason)"
        }
    }
}

/// Single owner of the `MTLDevice`, default `MTLCommandQueue`, and compiled shader libraries.
/// Thread-safe (`@unchecked Sendable`): Metal device, queue, and library are immutable;
/// pipeline state cache is synchronized with an internal lock.
public final class FlintMetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let lock = NSLock()
    private var pipelineCache = [String: MTLComputePipelineState]()

    /// Initializes a context using the specified Metal device or default system device.
    public init(device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw FlintMetalError.deviceUnavailable
        }
        guard let q = dev.makeCommandQueue() else {
            throw FlintMetalError.commandQueueUnavailable
        }
        self.device = dev
        self.queue = q
    }

    /// Compiles a Metal library from source code string.
    public func makeLibrary(source: String) throws -> MTLLibrary {
        do {
            let options = MTLCompileOptions()
            options.languageVersion = .version3_1
            return try device.makeLibrary(source: source, options: options)
        } catch {
            // Fall back without languageVersion restriction if 3.1 is not recognized
            do {
                return try device.makeLibrary(source: source, options: nil)
            } catch let innerError {
                throw FlintMetalError.libraryCompilationFailed(innerError.localizedDescription)
            }
        }
    }

    /// Retrieves or compiles a compute pipeline state for the given function name.
    public func computePipelineState(functionName: String, library: MTLLibrary) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let cached = pipelineCache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw FlintMetalError.functionNotFound(functionName)
        }

        do {
            let pso = try device.makeComputePipelineState(function: function)
            pipelineCache[functionName] = pso
            return pso
        } catch {
            throw FlintMetalError.pipelineCreationFailed("Failed to build pipeline for \(functionName): \(error.localizedDescription)")
        }
    }

    /// Allocates an immutable or mutable shared memory buffer.
    public func makeBuffer(byteCount: Int) throws -> MTLBuffer {
        guard byteCount >= 0 else {
            throw FlintMetalError.invalidParameter("byteCount cannot be negative: \(byteCount)")
        }
        if byteCount == 0 {
            // Metal doesn't allow 0-length buffers; allocate minimal 16 bytes for empty handles
            guard let buf = device.makeBuffer(length: 16, options: .storageModeShared) else {
                throw FlintMetalError.pipelineCreationFailed("Failed to allocate empty Metal buffer")
            }
            return buf
        }
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw FlintMetalError.pipelineCreationFailed("Failed to allocate Metal buffer of \(byteCount) bytes")
        }
        return buf
    }

    /// Allocates a shared memory buffer initialized with raw bytes.
    public func makeBuffer(bytes: UnsafeRawPointer, byteCount: Int) throws -> MTLBuffer {
        let buf = try makeBuffer(byteCount: byteCount)
        if byteCount > 0 {
            memcpy(buf.contents(), bytes, byteCount)
        }
        return buf
    }
}
