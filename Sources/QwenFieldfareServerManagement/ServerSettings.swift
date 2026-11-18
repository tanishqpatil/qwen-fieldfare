import Darwin
import Foundation
import QwenFieldfareServerSupport

public struct ServerSettings: Codable, Equatable, Sendable {
    public var modelPath: String
    public var port: Int = 8080
    public var context: Int = 8192
    public var memoryGiB: Int = 5
    public var queueCapacity: Int = 2
    public var prefixCacheMode: QwenPrefixCacheMode = .off
    public init(modelPath: String = "") {self.modelPath = modelPath}
    private enum CodingKeys: String,CodingKey {case modelPath,port,context,memoryGiB,queueCapacity,prefixCacheMode}
    public init(from decoder: Decoder) throws {
        let values=try decoder.container(keyedBy:CodingKeys.self)
        modelPath=try values.decode(String.self,forKey:.modelPath)
        port=try values.decodeIfPresent(Int.self,forKey:.port) ?? 8080
        context=try values.decodeIfPresent(Int.self,forKey:.context) ?? 8192
        memoryGiB=try values.decodeIfPresent(Int.self,forKey:.memoryGiB) ?? 5
        queueCapacity=try values.decodeIfPresent(Int.self,forKey:.queueCapacity) ?? 2
        prefixCacheMode=try values.decodeIfPresent(QwenPrefixCacheMode.self,forKey:.prefixCacheMode) ?? .off
    }
    public func configuration(keyFile: String) throws -> ServerConfiguration {
        let action = try ServerConfiguration.parse([
            "--model",modelPath,"--host","127.0.0.1","--port",String(port),
            "--context",String(context),"--memory-gib",String(memoryGiB),
            "--queue-capacity",String(queueCapacity),"--prefix-cache",prefixCacheMode.rawValue,"--api-key-file",keyFile
        ],environment:[:])
        guard case .serve(let config) = action else {throw ManagementError.invalid("Invalid server configuration.")}
        return config
    }
    public var validationMessage: String? {
        do {
            _ = try ServerConfiguration.parse(["--model",modelPath,"--host","127.0.0.1","--port",String(port),"--context",String(context),"--memory-gib",String(memoryGiB),"--queue-capacity",String(queueCapacity),"--prefix-cache",prefixCacheMode.rawValue],environment:["QWENFIELDFARE_API_KEY":"validation-placeholder-key"])
            return nil
        } catch {return managementMessage(error)}
    }
}
public enum ManagementError: Error, LocalizedError {
    case invalid(String), credential, connection
    public var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        case .credential: return "Credential unavailable. Unlock Keychain and retry, or import a private CLI key file."
        case .connection: return "Cannot authenticate or observe a QwenFieldfare server at this loopback address. Check its port and matching key."
        }
    }
}
func managementMessage(_ error: Error) -> String {
    if let error = error as? ManagementError {return error.localizedDescription}
    if let error = error as? APIError {return error.message}
    if let error = error as? QwenFieldfareCore.QwenInferenceLeaseError {return error.localizedDescription}
    return "Server operation failed. Check the compatible model directory, port and model ownership."
}
import QwenFieldfareCore

/// Same bounded private-file authentication parser used by the standalone CLI.
func configuration(_ settings: ServerSettings, key: String) throws -> ServerConfiguration {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-server-key-\(UUID())")
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
    defer {try? FileManager.default.removeItem(at:directory)}
    let path = directory.appendingPathComponent("key").path
    try exportPrivateKey(key,to:path)
    return try settings.configuration(keyFile:path)
}
public func exportPrivateKey(_ key: String, to path: String) throws {
    let fd = open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
    guard fd >= 0 else {throw ManagementError.invalid("Cannot create key file. Choose a new file in a private directory; existing files are never overwritten.")}
    let handle = FileHandle(fileDescriptor:fd,closeOnDealloc:true)
    do {try handle.write(contentsOf:Data((key+"\n").utf8));try handle.close()}
    catch {try? handle.close();unlink(path);throw ManagementError.credential}
}
