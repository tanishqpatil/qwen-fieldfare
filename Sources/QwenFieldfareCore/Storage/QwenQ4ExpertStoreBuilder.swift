import CryptoKit
import Darwin
import Foundation

public struct QwenQ4ExpertPackResult: Sendable, Equatable {
    public let outputDirectory: String
    public let layersWritten: Int
    public let totalBytes: UInt64
    public let sourceIndexSHA256: String
}

public struct QwenQ4ExpertVerificationResult: Sendable, Equatable {
    public let outputDirectory: String
    public let layersVerified: Int
    public let totalBytesVerified: UInt64
}

public enum QwenQ4ExpertStoreBuilder {
    private struct Checkpoint: Codable {
        let schemaVersion: Int
        let sourceIndexSHA256: String
        var completedLayers: [QwenQ4ExpertStoreManifest.Layer]
    }

    private struct Slice {
        let shardPath: String
        let fileOffset: UInt64
        let byteLength: Int
    }

    private struct LayerSlices {
        let values: [Slice]
    }

    private struct OpenShard {
        let path: String
        let fd: Int32
    }

    private static let checkpointFilename = "pack.resume.json"
    private static let partialSuffix = ".partial"
    private static let componentSuffixes = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight", "up_proj.scales", "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]
    private static let componentExpertByteLengths = [
        524_288, 32_768, 32_768,
        524_288, 32_768, 32_768,
        524_288, 32_768, 32_768,
    ]
    private static let componentRecordOffsets = [
        0, 524_288, 557_056,
        589_824, 1_114_112, 1_146_880,
        1_179_648, 1_703_936, 1_736_704,
    ]

    public static func pack(
        modelPath: String,
        outputDirectory: String? = nil,
        resume: Bool = false,
        progress: ((String) -> Void)? = nil
    ) throws -> QwenQ4ExpertPackResult {
        let output = outputDirectory ?? QwenQ4ExpertStore.defaultDirectory(modelPath: modelPath)
        let sourceIndexPath = (modelPath as NSString).appendingPathComponent("model.safetensors.index.json")
        let sourceHash = try Sha256Verifier.hashFile(at: URL(fileURLWithPath: sourceIndexPath))
        let manifestPath = (output as NSString).appendingPathComponent(QwenQ4ExpertStore.manifestFilename)
        if FileManager.default.fileExists(atPath: manifestPath) {
            let store = try QwenQ4ExpertStore(modelPath: modelPath, directoryPath: output)
            return QwenQ4ExpertPackResult(
                outputDirectory: store.directoryPath,
                layersWritten: 0,
                totalBytes: UInt64(QwenQ4ExpertStore.layerCount) * QwenQ4ExpertStore.layerFileByteLength,
                sourceIndexSHA256: sourceHash)
        }

        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: output), withIntermediateDirectories: true)
        let checkpointPath = (output as NSString).appendingPathComponent(checkpointFilename)
        var checkpoint: Checkpoint
        if FileManager.default.fileExists(atPath: checkpointPath) {
            guard resume else {
                throw QwenQ4ExpertStoreError.partialStore("use --resume to continue \(checkpointPath)")
            }
            checkpoint = try JSONDecoder().decode(
                Checkpoint.self, from: Data(contentsOf: URL(fileURLWithPath: checkpointPath)))
            guard checkpoint.schemaVersion == Int(QwenQ4ExpertLayerHeader.schemaVersion),
                  checkpoint.sourceIndexSHA256 == sourceHash else {
                throw QwenQ4ExpertStoreError.partialStore("checkpoint does not match this source model")
            }
        } else {
            checkpoint = Checkpoint(
                schemaVersion: Int(QwenQ4ExpertLayerHeader.schemaVersion),
                sourceIndexSHA256: sourceHash,
                completedLayers: [])
            try writeCheckpoint(checkpoint, path: checkpointPath, directory: output)
        }

        let layers = try discoverSlices(modelPath: modelPath, indexPath: sourceIndexPath)
        let shardPaths = Set(layers.flatMap { $0.values.map(\.shardPath) }).sorted()
        var openShards: [String: OpenShard] = [:]
        defer { for shard in openShards.values { Darwin.close(shard.fd) } }
        for path in shardPaths {
            let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard fd >= 0 else {
                throw QwenQ4ExpertStoreError.posix(operation: "open", path: path, code: errno)
            }
            _ = fcntl(fd, F_NOCACHE, 1)
            openShards[path] = OpenShard(path: path, fd: fd)
        }

        var completed = Dictionary(uniqueKeysWithValues: checkpoint.completedLayers.map { ($0.layerIndex, $0) })
        var layersWritten = 0
        for layer in 0..<QwenQ4ExpertStore.layerCount {
            let filename = QwenQ4ExpertStore.layerFilename(layer)
            let finalPath = (output as NSString).appendingPathComponent(filename)
            let partialPath = finalPath + partialSuffix

            if let entry = completed[layer] {
                progress?("Verifying completed layer \(layer + 1)/\(QwenQ4ExpertStore.layerCount)")
                try verifyLayerFile(
                    path: finalPath, expected: entry, sourceIndexSHA256: sourceHash)
                continue
            }

            if FileManager.default.fileExists(atPath: finalPath) {
                guard resume else {
                    throw QwenQ4ExpertStoreError.partialStore("unexpected completed file \(finalPath); use --resume")
                }
                progress?("Recovering completed layer \(layer + 1)/\(QwenQ4ExpertStore.layerCount)")
                let recovered = try inspectAndVerifyLayerFile(
                    path: finalPath, layer: layer, sourceIndexSHA256: sourceHash)
                completed[layer] = recovered
                checkpoint.completedLayers = completed.values.sorted(by: { $0.layerIndex < $1.layerIndex })
                try writeCheckpoint(checkpoint, path: checkpointPath, directory: output)
                continue
            }

            if FileManager.default.fileExists(atPath: partialPath) {
                guard resume else {
                    throw QwenQ4ExpertStoreError.partialStore("unfinished layer file \(partialPath); use --resume")
                }
                guard Darwin.unlink(partialPath) == 0 else {
                    throw QwenQ4ExpertStoreError.posix(operation: "unlink", path: partialPath, code: errno)
                }
                try fsyncDirectory(output)
            }

            progress?("Packing layer \(layer + 1)/\(QwenQ4ExpertStore.layerCount)")
            let entry = try packLayer(
                layer: layer,
                slices: layers[layer],
                openShards: openShards,
                sourceIndexSHA256: sourceHash,
                partialPath: partialPath,
                finalPath: finalPath)
            completed[layer] = entry
            checkpoint.completedLayers = completed.values.sorted(by: { $0.layerIndex < $1.layerIndex })
            try writeCheckpoint(checkpoint, path: checkpointPath, directory: output)
            layersWritten += 1
        }

        let manifest = QwenQ4ExpertStoreManifest(
            schemaVersion: Int(QwenQ4ExpertLayerHeader.schemaVersion),
            sourceIndexSHA256: sourceHash,
            layers: completed.values.sorted(by: { $0.layerIndex < $1.layerIndex }))
        try QwenQ4ExpertStore.validateManifest(manifest)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try atomicWrite(try encoder.encode(manifest), path: manifestPath, directory: output)
        if Darwin.unlink(checkpointPath) != 0, errno != ENOENT {
            throw QwenQ4ExpertStoreError.posix(operation: "unlink", path: checkpointPath, code: errno)
        }
        try fsyncDirectory(output)
        return QwenQ4ExpertPackResult(
            outputDirectory: output,
            layersWritten: layersWritten,
            totalBytes: UInt64(QwenQ4ExpertStore.layerCount) * QwenQ4ExpertStore.layerFileByteLength,
            sourceIndexSHA256: sourceHash)
    }

    public static func verify(
        modelPath: String,
        outputDirectory: String? = nil,
        progress: ((String) -> Void)? = nil
    ) throws -> QwenQ4ExpertVerificationResult {
        let output = outputDirectory ?? QwenQ4ExpertStore.defaultDirectory(modelPath: modelPath)
        let store = try QwenQ4ExpertStore(modelPath: modelPath, directoryPath: output)
        for layer in store.manifest.layers.sorted(by: { $0.layerIndex < $1.layerIndex }) {
            progress?("Verifying layer \(layer.layerIndex + 1)/\(QwenQ4ExpertStore.layerCount)")
            let path = (output as NSString).appendingPathComponent(layer.filename)
            try verifyLayerFile(
                path: path, expected: layer, sourceIndexSHA256: store.manifest.sourceIndexSHA256)
        }
        return QwenQ4ExpertVerificationResult(
            outputDirectory: output,
            layersVerified: QwenQ4ExpertStore.layerCount,
            totalBytesVerified: UInt64(QwenQ4ExpertStore.layerCount) * QwenQ4ExpertStore.payloadByteLength)
    }

    private static func discoverSlices(modelPath: String, indexPath: String) throws -> [LayerSlices] {
        let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        guard let root = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String] else {
            throw QwenQ4ExpertStoreError.invalidManifest("source index has no weight_map")
        }
        var neededByShard: [String: Set<String>] = [:]
        var namesByLayer: [[String]] = []
        for layer in 0..<QwenQ4ExpertStore.layerCount {
            let prefix = "language_model.model.layers.\(layer).mlp.switch_mlp."
            let names = componentSuffixes.map { prefix + $0 }
            for name in names {
                guard let shard = weightMap[name] else {
                    throw QwenQ4ExpertStoreError.invalidManifest("source index is missing \(name)")
                }
                neededByShard[shard, default: []].insert(name)
            }
            namesByLayer.append(names)
        }

        var slicesByName: [String: Slice] = [:]
        for (shardName, neededNames) in neededByShard {
            let path = (modelPath as NSString).appendingPathComponent(shardName)
            let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard fd >= 0 else {
                throw QwenQ4ExpertStoreError.posix(operation: "open", path: path, code: errno)
            }
            defer { Darwin.close(fd) }
            var lengthBytes = [UInt8](repeating: 0, count: 8)
            let lengthGot = lengthBytes.withUnsafeMutableBytes {
                QwenQ4ExpertStore.preadFully(fd: fd, destination: $0.baseAddress!, count: 8, offset: 0)
            }
            guard lengthGot == 8 else {
                throw QwenQ4ExpertStoreError.shortRead(path: path, expected: 8, actual: lengthGot)
            }
            let headerLength = lengthBytes.enumerated().reduce(UInt64(0)) {
                $0 | (UInt64($1.element) << UInt64($1.offset * 8))
            }
            guard headerLength > 0, headerLength <= 256 * 1_024 * 1_024,
                  headerLength <= UInt64(Int.max) else {
                throw QwenQ4ExpertStoreError.invalidManifest("invalid safetensors header length in \(shardName)")
            }
            var headerData = Data(count: Int(headerLength))
            let headerGot = headerData.withUnsafeMutableBytes {
                QwenQ4ExpertStore.preadFully(
                    fd: fd, destination: $0.baseAddress!, count: $0.count, offset: 8)
            }
            guard headerGot == Int(headerLength),
                  let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
                throw QwenQ4ExpertStoreError.invalidManifest("invalid safetensors header in \(shardName)")
            }
            for name in neededNames {
                guard let tensor = header[name] as? [String: Any],
                      let offsets = tensor["data_offsets"] as? [NSNumber], offsets.count == 2 else {
                    throw QwenQ4ExpertStoreError.invalidManifest("missing tensor metadata for \(name)")
                }
                let start = offsets[0].uint64Value
                let end = offsets[1].uint64Value
                guard end >= start, end - start <= UInt64(Int.max) else {
                    throw QwenQ4ExpertStoreError.invalidManifest("invalid data offsets for \(name)")
                }
                slicesByName[name] = Slice(
                    shardPath: path,
                    fileOffset: 8 + headerLength + start,
                    byteLength: Int(end - start))
            }
        }

        return try namesByLayer.map { names in
            let values = try names.enumerated().map { component, name -> Slice in
                guard let slice = slicesByName[name] else {
                    throw QwenQ4ExpertStoreError.invalidManifest("missing discovered slice for \(name)")
                }
                let expected = componentExpertByteLengths[component] * QwenQ4ExpertStore.expertsPerLayer
                guard slice.byteLength == expected else {
                    throw QwenQ4ExpertStoreError.invalidManifest(
                        "unexpected byte length for \(name): expected \(expected), got \(slice.byteLength)")
                }
                return slice
            }
            return LayerSlices(values: values)
        }
    }

    private static func packLayer(
        layer: Int,
        slices: LayerSlices,
        openShards: [String: OpenShard],
        sourceIndexSHA256: String,
        partialPath: String,
        finalPath: String
    ) throws -> QwenQ4ExpertStoreManifest.Layer {
        let fd = Darwin.open(
            partialPath, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard fd >= 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "open", path: partialPath, code: errno)
        }
        defer { Darwin.close(fd) }
        guard ftruncate(fd, off_t(QwenQ4ExpertStore.layerFileByteLength)) == 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "ftruncate", path: partialPath, code: errno)
        }
        _ = fcntl(fd, F_NOCACHE, 1)

        let record = UnsafeMutableRawBufferPointer.allocate(
            byteCount: QwenQ4ExpertStore.expertByteLength, alignment: 16_384)
        defer { record.deallocate() }
        var hasher = SHA256()
        for expert in 0..<QwenQ4ExpertStore.expertsPerLayer {
            for component in slices.values.indices {
                let slice = slices.values[component]
                let count = componentExpertByteLengths[component]
                guard let shard = openShards[slice.shardPath] else {
                    throw QwenQ4ExpertStoreError.invalidManifest("source shard was not opened")
                }
                let sourceOffset = slice.fileOffset + UInt64(expert * count)
                let target = record.baseAddress!.advanced(by: componentRecordOffsets[component])
                let got = QwenQ4ExpertStore.preadFully(
                    fd: shard.fd, destination: target, count: count, offset: sourceOffset)
                guard got == count else {
                    throw QwenQ4ExpertStoreError.shortRead(path: shard.path, expected: count, actual: got)
                }
            }
            hasher.update(bufferPointer: UnsafeRawBufferPointer(record))
            let destinationOffset = UInt64(QwenQ4ExpertLayerHeader.byteLength)
                + UInt64(expert * QwenQ4ExpertStore.expertByteLength)
            try pwriteFully(
                fd: fd,
                source: record.baseAddress!,
                count: record.count,
                offset: destinationOffset,
                path: partialPath)
        }
        let payloadHash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        let header = QwenQ4ExpertLayerHeader(
            layerIndex: layer,
            expertsPerLayer: QwenQ4ExpertStore.expertsPerLayer,
            expertByteLength: QwenQ4ExpertStore.expertByteLength,
            payloadByteLength: QwenQ4ExpertStore.payloadByteLength,
            payloadSHA256: payloadHash,
            sourceIndexSHA256: sourceIndexSHA256)
        try header.encode().withUnsafeBytes {
            try pwriteFully(fd: fd, source: $0.baseAddress!, count: $0.count, offset: 0, path: partialPath)
        }
        guard fsync(fd) == 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "fsync", path: partialPath, code: errno)
        }
        guard Darwin.rename(partialPath, finalPath) == 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "rename", path: partialPath, code: errno)
        }
        try fsyncDirectory((finalPath as NSString).deletingLastPathComponent)
        return QwenQ4ExpertStoreManifest.Layer(
            layerIndex: layer,
            filename: QwenQ4ExpertStore.layerFilename(layer),
            fileSize: QwenQ4ExpertStore.layerFileByteLength,
            payloadSHA256: payloadHash)
    }

    private static func inspectAndVerifyLayerFile(
        path: String,
        layer: Int,
        sourceIndexSHA256: String
    ) throws -> QwenQ4ExpertStoreManifest.Layer {
        let header = try readHeader(path: path)
        let expected = QwenQ4ExpertStoreManifest.Layer(
            layerIndex: layer,
            filename: QwenQ4ExpertStore.layerFilename(layer),
            fileSize: QwenQ4ExpertStore.layerFileByteLength,
            payloadSHA256: header.payloadSHA256)
        try verifyLayerFile(path: path, expected: expected, sourceIndexSHA256: sourceIndexSHA256)
        return expected
    }

    private static func verifyLayerFile(
        path: String,
        expected: QwenQ4ExpertStoreManifest.Layer,
        sourceIndexSHA256: String
    ) throws {
        let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "open", path: path, code: errno)
        }
        defer { Darwin.close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var info = stat()
        guard fstat(fd, &info) == 0, UInt64(info.st_size) == expected.fileSize else {
            throw QwenQ4ExpertStoreError.invalidHeader("file size mismatch for \(path)")
        }
        let header = try readHeader(fd: fd, path: path)
        guard header.layerIndex == expected.layerIndex,
              header.expertsPerLayer == QwenQ4ExpertStore.expertsPerLayer,
              header.expertByteLength == QwenQ4ExpertStore.expertByteLength,
              header.payloadByteLength == QwenQ4ExpertStore.payloadByteLength,
              header.payloadSHA256 == expected.payloadSHA256,
              header.sourceIndexSHA256 == sourceIndexSHA256 else {
            throw QwenQ4ExpertStoreError.invalidHeader("metadata mismatch for \(path)")
        }
        let actualHash = try hashPayload(fd: fd, path: path)
        guard actualHash == expected.payloadSHA256 else {
            throw QwenQ4ExpertStoreError.invalidHeader("payload SHA-256 mismatch for \(path)")
        }
    }

    private static func readHeader(path: String) throws -> QwenQ4ExpertLayerHeader {
        let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "open", path: path, code: errno)
        }
        defer { Darwin.close(fd) }
        return try readHeader(fd: fd, path: path)
    }

    private static func readHeader(fd: Int32, path: String) throws -> QwenQ4ExpertLayerHeader {
        var data = Data(count: QwenQ4ExpertLayerHeader.byteLength)
        let got = data.withUnsafeMutableBytes {
            QwenQ4ExpertStore.preadFully(fd: fd, destination: $0.baseAddress!, count: $0.count, offset: 0)
        }
        guard got == data.count else {
            throw QwenQ4ExpertStoreError.shortRead(path: path, expected: data.count, actual: got)
        }
        return try QwenQ4ExpertLayerHeader.decode(data)
    }

    private static func hashPayload(fd: Int32, path: String) throws -> String {
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 1 << 20, alignment: 16_384)
        defer { buffer.deallocate() }
        var hasher = SHA256()
        var offset = UInt64(QwenQ4ExpertLayerHeader.byteLength)
        var remaining = QwenQ4ExpertStore.payloadByteLength
        while remaining > 0 {
            let count = min(buffer.count, Int(remaining))
            let got = QwenQ4ExpertStore.preadFully(
                fd: fd, destination: buffer.baseAddress!, count: count, offset: offset)
            guard got == count else {
                throw QwenQ4ExpertStoreError.shortRead(path: path, expected: count, actual: got)
            }
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buffer.baseAddress, count: count))
            offset += UInt64(count)
            remaining -= UInt64(count)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func writeCheckpoint(_ checkpoint: Checkpoint, path: String, directory: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try atomicWrite(try encoder.encode(checkpoint), path: path, directory: directory)
    }

    private static func atomicWrite(_ data: Data, path: String, directory: String) throws {
        let temporary = path + ".tmp"
        if Darwin.unlink(temporary) != 0, errno != ENOENT {
            throw QwenQ4ExpertStoreError.posix(operation: "unlink", path: temporary, code: errno)
        }
        let fd = Darwin.open(
            temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard fd >= 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "open", path: temporary, code: errno)
        }
        var descriptorIsOpen = true
        defer {
            if descriptorIsOpen { Darwin.close(fd) }
        }
        do {
            try data.withUnsafeBytes {
                try pwriteFully(fd: fd, source: $0.baseAddress!, count: $0.count, offset: 0, path: temporary)
            }
            guard fsync(fd) == 0 else {
                throw QwenQ4ExpertStoreError.posix(operation: "fsync", path: temporary, code: errno)
            }
            Darwin.close(fd)
            descriptorIsOpen = false
            guard Darwin.rename(temporary, path) == 0 else {
                throw QwenQ4ExpertStoreError.posix(operation: "rename", path: temporary, code: errno)
            }
            try fsyncDirectory(directory)
        } catch {
            _ = Darwin.unlink(temporary)
            throw error
        }
    }

    private static func pwriteFully(
        fd: Int32,
        source: UnsafeRawPointer,
        count: Int,
        offset: UInt64,
        path: String
    ) throws {
        var completed = 0
        while completed < count {
            let result = Darwin.pwrite(
                fd, source.advanced(by: completed), count - completed,
                off_t(offset + UInt64(completed)))
            if result < 0, errno == EINTR { continue }
            guard result > 0 else {
                throw QwenQ4ExpertStoreError.posix(operation: "pwrite", path: path, code: errno)
            }
            completed += result
        }
    }

    private static func fsyncDirectory(_ path: String) throws {
        let fd = Darwin.open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "open directory", path: path, code: errno)
        }
        defer { Darwin.close(fd) }
        guard fsync(fd) == 0 else {
            throw QwenQ4ExpertStoreError.posix(operation: "fsync directory", path: path, code: errno)
        }
    }
}
