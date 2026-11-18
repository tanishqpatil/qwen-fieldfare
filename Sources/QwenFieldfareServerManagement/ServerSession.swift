import Darwin
import Foundation
import QwenFieldfareServerSupport

public protocol ManagedServerSession: Sendable {
    var port: Int {get}
    func health() async throws -> ServerHealth
    func clearPrefixCache() async throws
    func stop() async throws
}
public extension ManagedServerSession {
    func clearPrefixCache() async throws {throw ManagementError.invalid("Prefix cache control is unavailable for this session.")}
}
public actor InProcessServerSession: ManagedServerSession {
    public nonisolated let port: Int
    private var server: LocalServer?
    public init(server: LocalServer,port: Int) {self.server = server;self.port = port}
    public func health() throws -> ServerHealth {
        guard let server else {throw ManagementError.connection};return server.snapshot
    }
    public func clearPrefixCache() throws {guard let server else {throw ManagementError.connection};server.clearPrefixCache()}
    public func stop() async throws {
        guard let owned = server else {return}
        try await owned.shutdown()
        server = nil // releases backend/model before native chat can resume
    }
    public static func start(_ config: ServerConfiguration) async throws -> any ManagedServerSession {
        try assertPortAvailable(config.port)
        let backend = try await CoreBackend(path:config.modelPath,context:config.context,budget:config.budget,prefixCacheMode:config.prefixCacheMode)
        try Task.checkCancellation()
        let server = try LocalServer(backend:backend,key:config.apiKey,model:config.modelID,queueCapacity:config.queueCapacity)
        do {return InProcessServerSession(server:server,port:try await server.start(host:config.host,port:config.port))}
        catch {try? await server.shutdown();throw error}
    }
}
// An early diagnostic, not a reservation. LocalServer's real bind is authoritative.
func assertPortAvailable(_ port: Int) throws {
    let fd = socket(AF_INET,SOCK_STREAM,0)
    guard fd >= 0 else {throw ManagementError.invalid("Cannot create loopback socket.")}
    defer {close(fd)}
    var reuse: Int32 = 1
    guard setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&reuse,socklen_t(MemoryLayout<Int32>.size)) == 0 else {throw ManagementError.invalid("Cannot configure loopback socket.")}
    var address = sockaddr_in();address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);address.sin_family = sa_family_t(AF_INET)
    address.sin_port = in_port_t(port).bigEndian;address.sin_addr.s_addr = inet_addr("127.0.0.1")
    let result = withUnsafePointer(to:&address) {$0.withMemoryRebound(to:sockaddr.self,capacity:1) {bind(fd,$0,socklen_t(MemoryLayout<sockaddr_in>.size))}}
    guard result == 0 else {throw ManagementError.invalid("Loopback port is unavailable. An independent CLI server may already be running; observe it with its matching key or choose another port.")}
}
private final class NoRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,newRequest request: URLRequest,completionHandler: @escaping (URLRequest?) -> Void) {completionHandler(nil)}
}
public enum LoopbackServerObserver {
    public static func health(port: Int,key: String) async throws -> ServerHealth {
        guard (1...65535).contains(port) else {throw ManagementError.connection}
        let config = URLSessionConfiguration.ephemeral
        config.connectionProxyDictionary = [:];config.httpShouldSetCookies = false;config.urlCache = nil
        let session = URLSession(configuration:config,delegate:NoRedirects(),delegateQueue:nil)
        defer {session.invalidateAndCancel()}
        var request = URLRequest(url:URL(string:"http://127.0.0.1:\(port)/health")!)
        request.timeoutInterval = 3;request.setValue("Bearer \(key)",forHTTPHeaderField:"Authorization")
        do {
            let (bytes,response) = try await session.bytes(for:request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {throw ManagementError.connection}
            var data = Data()
            for try await byte in bytes {guard data.count < 65536 else {throw ManagementError.connection};data.append(byte)}
            let value = try JSONDecoder().decode(ServerHealth.self,from:data)
            guard value.service == "QwenFieldfareServer",value.status == "ready" else {throw ManagementError.connection}
            return value
        } catch {throw ManagementError.connection}
    }
}
