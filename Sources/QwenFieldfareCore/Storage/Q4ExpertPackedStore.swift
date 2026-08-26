import Foundation
import Metal

public struct Q4ExpertPackedHeader: Sendable {
    public static let magic = "Q4EXP01\0"
    public static let expertByteLength: Int = 1_769_472 // 3 * 524288 + 6 * 32768
    public static let headerSize: Int = 4096

    public let numLayers: Int
    public let numExpertsPerLayer: Int

    public init(numLayers: Int, numExpertsPerLayer: Int = 256) {
        self.numLayers = numLayers
        self.numExpertsPerLayer = numExpertsPerLayer
    }

    public func offsetFor(layer: Int, expert: Int) -> Int {
        Self.headerSize + (layer * numExpertsPerLayer + expert) * Self.expertByteLength
    }
}

public final class Q4ExpertPackedStore: @unchecked Sendable {
    public let header: Q4ExpertPackedHeader
    public let filePath: String
    private let fd: Int32
    public let mmapPtr: UnsafeRawPointer
    public let fileSize: Int

    public init(filePath: String) throws {
        self.filePath = filePath
        let f = open(filePath, O_RDONLY)
        guard f >= 0 else {
            throw NSError(domain: "Q4ExpertPackedStore", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to open \(filePath)"])
        }
        self.fd = f
        var st = stat()
        fstat(f, &st)
        self.fileSize = Int(st.st_size)

        guard let ptr = mmap(nil, fileSize, PROT_READ, MAP_SHARED, f, 0), ptr != MAP_FAILED else {
            close(f)
            throw NSError(domain: "Q4ExpertPackedStore", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to mmap \(filePath)"])
        }
        let rawPtr = UnsafeRawPointer(ptr)
        self.mmapPtr = rawPtr

        let magicStr = String(bytes: UnsafeRawBufferPointer(start: rawPtr, count: 8), encoding: .utf8) ?? ""
        guard magicStr == Q4ExpertPackedHeader.magic else {
            munmap(ptr, fileSize)
            close(f)
            throw NSError(domain: "Q4ExpertPackedStore", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid magic header: \(magicStr)"])
        }

        let numLayers = Int(rawPtr.advanced(by: 8).load(as: UInt32.self))
        let numExperts = Int(rawPtr.advanced(by: 12).load(as: UInt32.self))
        self.header = Q4ExpertPackedHeader(numLayers: numLayers, numExpertsPerLayer: numExperts)
        madvise(UnsafeMutableRawPointer(mutating: ptr), fileSize, MADV_RANDOM)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: mmapPtr), fileSize)
        close(fd)
    }

    public func expertPointer(layer: Int, expert: Int) -> UnsafeRawPointer {
        let offset = header.offsetFor(layer: layer, expert: expert)
        return mmapPtr.advanced(by: offset)
    }

    /// Packs layer expert slices from raw Safetensors pointers into a contiguous .q4exp file.
    public static func createPackedFile(
        layer: Int,
        numExperts: Int = 256,
        outputPath: String,
        gateW: UnsafeRawPointer, gateS: UnsafeRawPointer, gateB: UnsafeRawPointer,
        upW: UnsafeRawPointer, upS: UnsafeRawPointer, upB: UnsafeRawPointer,
        downW: UnsafeRawPointer, downS: UnsafeRawPointer, downB: UnsafeRawPointer
    ) throws {
        let expertSize = Q4ExpertPackedHeader.expertByteLength
        let totalSize = Q4ExpertPackedHeader.headerSize + numExperts * expertSize

        let fd = open(outputPath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else {
            throw NSError(domain: "Q4ExpertPackedStore", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create \(outputPath)"])
        }
        ftruncate(fd, off_t(totalSize))

        guard let destPtr = mmap(nil, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0), destPtr != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "Q4ExpertPackedStore", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to mmap output file"])
        }

        // Write header
        memset(destPtr, 0, Q4ExpertPackedHeader.headerSize)
        let magicBytes = Array(Q4ExpertPackedHeader.magic.utf8)
        memcpy(destPtr, magicBytes, 8)
        destPtr.advanced(by: 8).storeBytes(of: UInt32(1), as: UInt32.self) // 1 layer in this prototype
        destPtr.advanced(by: 12).storeBytes(of: UInt32(numExperts), as: UInt32.self)
        destPtr.advanced(by: 16).storeBytes(of: UInt32(expertSize), as: UInt32.self)

        // Write contiguous expert blocks
        for e in 0..<numExperts {
            let expDest = destPtr.advanced(by: Q4ExpertPackedHeader.headerSize + e * expertSize)

            let gwSrc = gateW.advanced(by: e * 524288)
            let gsSrc = gateS.advanced(by: e * 32768)
            let gbSrc = gateB.advanced(by: e * 32768)

            let uwSrc = upW.advanced(by: e * 524288)
            let usSrc = upS.advanced(by: e * 32768)
            let ubSrc = upB.advanced(by: e * 32768)

            let dwSrc = downW.advanced(by: e * 524288)
            let dsSrc = downS.advanced(by: e * 32768)
            let dbSrc = downB.advanced(by: e * 32768)

            memcpy(expDest.advanced(by: 0), gwSrc, 524288)
            memcpy(expDest.advanced(by: 524288), gsSrc, 32768)
            memcpy(expDest.advanced(by: 557056), gbSrc, 32768)

            memcpy(expDest.advanced(by: 589824), uwSrc, 524288)
            memcpy(expDest.advanced(by: 1114112), usSrc, 32768)
            memcpy(expDest.advanced(by: 1146880), ubSrc, 32768)

            memcpy(expDest.advanced(by: 1179648), dwSrc, 524288)
            memcpy(expDest.advanced(by: 1703936), dsSrc, 32768)
            memcpy(expDest.advanced(by: 1736704), dbSrc, 32768)
        }

        msync(destPtr, totalSize, MS_SYNC)
        munmap(destPtr, totalSize)
        close(fd)
    }
}
