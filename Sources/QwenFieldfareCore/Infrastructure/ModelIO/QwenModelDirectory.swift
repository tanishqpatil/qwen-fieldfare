import Darwin
import Foundation

public enum QwenModelDirectoryError: Error, LocalizedError, Equatable {
    case notDirectory(String)
    case missingFile(String)
    case invalidIndex(String)
    case invalidMetadata(String)

    public var errorDescription: String? {
        switch self {
        case .notDirectory(let path): return "The selected model path is not a directory: \(path)"
        case .missingFile(let name): return "The selected directory is missing \(name)."
        case .invalidIndex(let detail): return "The model Safetensors index is invalid: \(detail)"
        case .invalidMetadata(let detail): return "The model metadata is invalid: \(detail)"
        }
    }
}

public enum QwenModelDirectory {
    // The index is metadata, not a weight payload. Bound both initial size and
    // bytes read in case the file grows after fstat.
    static let maximumIndexBytes = 16 * 1_024 * 1_024
    public static let requiredMetadataFiles = [
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    ]

    @discardableResult
    public static func validate(path: String, fileManager: FileManager = .default) throws -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let canonical = URL(fileURLWithPath: expanded, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath().path
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: canonical, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw QwenModelDirectoryError.notDirectory(canonical)
        }
        var metadataURLs: [String: URL] = [:]
        for name in requiredMetadataFiles {
            let url = URL(fileURLWithPath: canonical, isDirectory: true)
                .appendingPathComponent(name).resolvingSymlinksInPath()
            guard url.deletingLastPathComponent().path == canonical else {
                throw QwenModelDirectoryError.invalidMetadata("\(name) resolves outside model directory")
            }
            guard fileManager.fileExists(atPath: url.path) else {
                throw QwenModelDirectoryError.missingFile(name)
            }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                throw QwenModelDirectoryError.invalidMetadata("\(name) is not a regular file")
            }
            metadataURLs[name] = url
        }
        let indexName = "model.safetensors.index.json"
        guard let indexURL = metadataURLs[indexName] else {
            throw QwenModelDirectoryError.missingFile(indexName)
        }
        let data = try readIndex(at: indexURL)
        let object: Any
        do { object = try JSONSerialization.jsonObject(with: data) }
        catch { throw QwenModelDirectoryError.invalidIndex("malformed JSON") }
        guard let root = object as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String], !weightMap.isEmpty else {
            throw QwenModelDirectoryError.invalidIndex("weight_map is missing or empty")
        }
        for name in Set(weightMap.values).sorted() {
            guard !name.isEmpty, name == (name as NSString).lastPathComponent else {
                throw QwenModelDirectoryError.invalidIndex("unsafe shard name \(name)")
            }
            let shard = URL(fileURLWithPath: canonical, isDirectory: true)
                .appendingPathComponent(name).resolvingSymlinksInPath()
            guard shard.deletingLastPathComponent().path == canonical else {
                throw QwenModelDirectoryError.invalidIndex("shard resolves outside model directory: \(name)")
            }
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: shard.path, isDirectory: &isDirectory) else {
                throw QwenModelDirectoryError.missingFile(name)
            }
            guard !isDirectory.boolValue,
                  (try? shard.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                throw QwenModelDirectoryError.invalidIndex("shard is not a regular file: \(name)")
            }
        }
        return canonical
    }

    private static func readIndex(at url: URL) throws -> Data {
        // Resolve permitted internal links above, then refuse a substituted
        // leaf symlink. NONBLOCK prevents a substituted FIFO from hanging open.
        let descriptor = open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw QwenModelDirectoryError.invalidIndex("cannot open index")
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(descriptor, &info) == 0,
              info.st_mode & S_IFMT == S_IFREG else {
            throw QwenModelDirectoryError.invalidIndex("index is not a regular file")
        }
        guard info.st_size >= 0, info.st_size <= Int64(maximumIndexBytes) else {
            throw QwenModelDirectoryError.invalidIndex("index exceeds 16 MiB limit")
        }
        var data = Data()
        do {
            while data.count <= maximumIndexBytes {
                let count = min(64 * 1_024, maximumIndexBytes + 1 - data.count)
                guard let chunk = try handle.read(upToCount: count), !chunk.isEmpty else { break }
                data.append(chunk)
            }
        } catch {
            throw QwenModelDirectoryError.invalidIndex("cannot read index")
        }
        guard data.count <= maximumIndexBytes else {
            throw QwenModelDirectoryError.invalidIndex("index exceeds 16 MiB limit")
        }
        return data
    }
}
