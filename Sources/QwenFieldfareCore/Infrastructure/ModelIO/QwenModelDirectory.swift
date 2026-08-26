import Foundation

public enum QwenModelDirectoryError: Error, LocalizedError, Equatable {
    case notDirectory(String)
    case missingFile(String)
    case invalidIndex(String)

    public var errorDescription: String? {
        switch self {
        case .notDirectory(let path): return "The selected model path is not a directory: \(path)"
        case .missingFile(let name): return "The selected directory is missing \(name)."
        case .invalidIndex(let detail): return "The model Safetensors index is invalid: \(detail)"
        }
    }
}

public enum QwenModelDirectory {
    public static let requiredMetadataFiles = [
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    ]

    @discardableResult
    public static func validate(path: String, fileManager: FileManager = .default) throws -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let canonical = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: canonical, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw QwenModelDirectoryError.notDirectory(canonical)
        }
        for name in requiredMetadataFiles {
            guard fileManager.fileExists(atPath: (canonical as NSString).appendingPathComponent(name)) else {
                throw QwenModelDirectoryError.missingFile(name)
            }
        }
        let indexPath = (canonical as NSString).appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String], !weightMap.isEmpty else {
            throw QwenModelDirectoryError.invalidIndex("weight_map is missing or empty")
        }
        for name in Set(weightMap.values).sorted() {
            guard !name.isEmpty, name == (name as NSString).lastPathComponent else {
                throw QwenModelDirectoryError.invalidIndex("unsafe shard name \(name)")
            }
            guard fileManager.fileExists(atPath: (canonical as NSString).appendingPathComponent(name)) else {
                throw QwenModelDirectoryError.missingFile(name)
            }
        }
        return canonical
    }
}
