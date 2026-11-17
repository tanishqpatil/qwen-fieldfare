import Darwin
import Foundation
import QwenFieldfareCore

public struct ServerConfiguration: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let modelPath: String
    public let modelID: String
    public let host: String
    public let port: Int
    public let context: Int
    public let budget: QwenActiveStreamingMemoryBudget
    public let queueCapacity: Int
    public let apiKey: String
    public var description: String { "QwenFieldfareServer(host=\(host), port=\(port), context=\(context), budget=\(budget.rawValue), queue=\(queueCapacity), apiKey=<redacted>)" }
    public var debugDescription: String { description }
    public static let version = "QwenFieldfareServer S2-02 (development build)"
    public static let help = """
    Usage: QwenFieldfareServer --model PATH [options]
      --host 127.0.0.1       Loopback only (external binds rejected)
      --port 8080            TCP port 1–65535 (0 selects an ephemeral port)
      --model-id qwen-fieldfare
      --context 8192         Positive context supported by the selected memory plan
      --memory-gib 5         Active-streaming Metal target: 3, 4 or 5 GiB
      --queue-capacity 2     Waiting generations: 0–8; one active generation
      --api-key-file PATH    Owned regular file, no group/other access, max 512-byte key
      --help, -h            Print this help without loading a model
      --version            Print development version without loading a model
    Authentication: QWENFIELDFARE_API_KEY OR --api-key-file (never a key in argv).
    Readiness: authenticated GET /health after initialization; model loaded once.
    Stop: SIGINT/SIGTERM; stop admission, cancel and drain before process exit.
    App, Bench and server Q4 runners share an exclusive per-user inference lease.
    Tools execute only in clients. Strict numerical parity remains unresolved.
    """
    public enum Action: Sendable { case help, version, serve(ServerConfiguration) }
    public static func parse(_ arguments: [String], environment: [String:String]) throws -> Action {
        if arguments == ["--help"] || arguments == ["-h"] {return .help}
        if arguments == ["--version"] {return .version}
        let allowed: Set<String> = ["--model","--host","--port","--context","--memory-gib","--queue-capacity","--model-id","--api-key-file"]
        var options: [String:String] = [:];var index=0
        while index < arguments.count {
            let key=arguments[index]
            guard allowed.contains(key),options[key] == nil,index+1 < arguments.count,!arguments[index+1].hasPrefix("--") else {throw APIError(400,"Unknown, repeated or missing option; use --help")}
            options[key]=arguments[index+1];index += 2
        }
        guard let path=options["--model"],!path.isEmpty else {throw APIError(400,"Provide --model PATH; use --help")}
        let host=options["--host"] ?? "127.0.0.1"
        guard host == "127.0.0.1" else {throw APIError(400,"Only --host 127.0.0.1 is supported")}
        guard let port=Int(options["--port"] ?? "8080"),(0...65535).contains(port) else {throw APIError(400,"Port must be an integer 0–65535")}
        guard let context=Int(options["--context"] ?? "8192"),context > 0 else {throw APIError(400,"Context must be a positive integer")}
        guard let n=Int(options["--memory-gib"] ?? "5"),let budget=QwenActiveStreamingMemoryBudget(rawValue:n) else {throw APIError(400,"Memory budget must be 3, 4 or 5 GiB")}
        guard let queue=Int(options["--queue-capacity"] ?? "2"),(0...8).contains(queue) else {throw APIError(400,"Queue capacity must be 0–8")}
        let modelID=options["--model-id"] ?? "qwen-fieldfare"
        guard !modelID.isEmpty,modelID.utf8.count <= 128,modelID.unicodeScalars.allSatisfy({$0.value >= 33 && $0.value <= 126}) else {throw APIError(400,"Model ID must be 1–128 printable ASCII characters")}
        do {_ = try QwenActiveStreamingMemoryConfiguration.make(budget:budget,contextLength:context)} catch {throw APIError(400,"Context does not fit selected memory plan")}
        let key: String
        if let file=options["--api-key-file"] {
            guard environment["QWENFIELDFARE_API_KEY"] == nil else {throw APIError(400,"Choose environment key OR key file, not both")}
            key=try readKey(file)
        } else {
            guard let value=environment["QWENFIELDFARE_API_KEY"] else {throw APIError(400,"Set QWENFIELDFARE_API_KEY or use --api-key-file")};key=value
        }
        guard (16...512).contains(key.utf8.count),key.utf8.allSatisfy({(33...126).contains($0)}) else {throw APIError(400,"API key must be 16–512 printable ASCII characters without whitespace")}
        return .serve(.init(modelPath:path,modelID:modelID,host:host,port:port,context:context,budget:budget,queueCapacity:queue,apiKey:key))
    }
    private static func readKey(_ path: String) throws -> String {
        let fd=open(path,O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC)
        guard fd >= 0 else {throw APIError(400,"Cannot open API-key file")}
        let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer {try? file.close()}
        var info=stat()
        guard fstat(fd,&info) == 0,info.st_mode & S_IFMT == S_IFREG,info.st_uid == getuid(),info.st_mode & 0o077 == 0,info.st_size <= 514 else {throw APIError(400,"Key file must be owned, regular, private (0600/0400), and small")}
        guard let data=try file.read(upToCount:515),data.count <= 514,var key=String(data:data,encoding:.utf8) else {throw APIError(400,"Cannot read API-key file")}
        if key.hasSuffix("\n") {key.removeLast();if key.hasSuffix("\r") {key.removeLast()}}
        return key
    }
}
