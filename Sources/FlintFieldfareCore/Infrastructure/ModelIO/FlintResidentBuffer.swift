import Darwin
import Foundation
import Metal

/// A single authoritative memory allocation for a resident model weight tensor.
/// Uses a Metal shared buffer (MTLBuffer with .storageModeShared) on Apple Silicon,
/// enabling zero-copy CPU-to-GPU access with page alignment. Falls back safely to
/// aligned native memory allocation when Metal is unavailable (e.g. in headless testing).
public final class FlintResidentBuffer: @unchecked Sendable {
    public let byteCount: Int
    public let pointer: UnsafeMutableRawBufferPointer
    public let metalBuffer: MTLBuffer?
    public let usesMetal: Bool

    public init(byteCount: Int, device: MTLDevice? = nil) {
        self.byteCount = byteCount
        if byteCount == 0 {
            self.pointer = UnsafeMutableRawBufferPointer(start: nil, count: 0)
            self.metalBuffer = nil
            self.usesMetal = false
            return
        }

        if let metalDevice = device ?? MTLCreateSystemDefaultDevice() {
            // Allocate page-aligned Metal shared buffer
            guard let buf = metalDevice.makeBuffer(length: byteCount, options: .storageModeShared) else {
                // Fallback to aligned heap allocation if Metal allocation fails
                let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
                self.pointer = UnsafeMutableRawBufferPointer(start: rawPtr, count: byteCount)
                self.metalBuffer = nil
                self.usesMetal = false
                return
            }
            self.metalBuffer = buf
            self.pointer = UnsafeMutableRawBufferPointer(start: buf.contents(), count: byteCount)
            self.usesMetal = true
        } else {
            // Native 64-byte aligned allocation for SIMD/CPU access
            let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
            self.pointer = UnsafeMutableRawBufferPointer(start: rawPtr, count: byteCount)
            self.metalBuffer = nil
            self.usesMetal = false
        }
    }

    deinit {
        if !usesMetal, let base = pointer.baseAddress {
            base.deallocate()
        }
        // Metal buffer is automatically freed by ARC
    }

    /// Copies bytes out into Swift Data for hash verification or testing.
    public func copyData() -> Data {
        guard let base = pointer.baseAddress, byteCount > 0 else { return Data() }
        return Data(bytes: base, count: byteCount)
    }

    /// Reads a slice into Swift Data.
    public func copySlice(range: Range<Int>) -> Data {
        guard range.lowerBound >= 0, range.upperBound <= byteCount, !range.isEmpty else { return Data() }
        guard let base = pointer.baseAddress else { return Data() }
        return Data(bytes: base.advanced(by: range.lowerBound), count: range.count)
    }
}
