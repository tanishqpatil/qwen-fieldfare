import Foundation
import Testing
import QwenFieldfareCore
@testable import QwenFieldfareServerSupport

private final class PrefixLifecycleBackend: GenerationBackend, @unchecked Sendable {
    enum Behavior: Equatable {case text, malformedTool, slow, failure}
    let behavior: Behavior
    private let lock=NSLock()
    private var commits=0
    private var reasons:[QwenPrefixInvalidation]=[]
    private var started=false
    init(_ behavior:Behavior = .text) {self.behavior=behavior}
    var snapshot:(Int,[QwenPrefixInvalidation],Bool) {lock.withLock {(commits,reasons,started)}}
    func commitGeneration() {lock.withLock {commits+=1}}
    func invalidatePrefixCache(_ reason:QwenPrefixInvalidation) {lock.withLock {reasons.append(reason)}}
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        lock.withLock {started=true}
        switch behavior {
        case .failure: throw APIError(400,"Context overflow")
        case .slow:
            for _ in 0..<1000 {try Task.checkCancellation();emit("x");try await Task.sleep(for:.milliseconds(10))}
        case .malformedTool: emit("<tool_call><function=lookup>")
        case .text: emit("hello")
        }
        return Completion(text:"hello",prompt:2,count:1,reason:"stop")
    }
}

@Suite(.serialized) struct PrefixLifecycleTests {
    let key="prefix-lifecycle-test-key"
    func request(_ port:Int,path:String="/v1/chat/completions",authorized:Bool=true,tools:Bool=false,stream:Bool=false) -> URLRequest {
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod="POST"
        request.setValue("Bearer \(authorized ? key : "incorrect")",forHTTPHeaderField:"Authorization")
        request.setValue("application/json",forHTTPHeaderField:"Content-Type")
        var body:[String:Any]=["model":"qwen","messages":[["role":"user","content":"fixture"]],"stream":stream]
        if tools {body["tools"]=[["type":"function","function":["name":"lookup","parameters":["type":"object","properties":[:]]]]]}
        request.httpBody=path == "/v1/cache/clear" ? wire([:]):wire(body)
        return request
    }
    @Test func clearRequiresAuthenticationAndDoesNotGenerate() async throws {
        let backend=PrefixLifecycleBackend();let server=try LocalServer(backend:backend,key:key,model:"qwen")
        let port=try await server.start(port:0)
        let (_,denied)=try await URLSession.shared.data(for:request(port,path:"/v1/cache/clear",authorized:false))
        #expect((denied as? HTTPURLResponse)?.statusCode == 401 && backend.snapshot.1.isEmpty)
        var invalid=request(port,path:"/v1/cache/clear");invalid.httpBody=wire(["unexpected":true])
        let (_,bad)=try await URLSession.shared.data(for:invalid)
        #expect((bad as? HTTPURLResponse)?.statusCode == 400 && backend.snapshot.1.isEmpty)
        let (_,cleared)=try await URLSession.shared.data(for:request(port,path:"/v1/cache/clear"))
        #expect((cleared as? HTTPURLResponse)?.statusCode == 200)
        #expect(backend.snapshot.0 == 0 && backend.snapshot.1 == [.explicitClear] && !backend.snapshot.2)
        try await server.shutdown();#expect(backend.snapshot.1.last == .shutdown)
    }
    @Test func protocolFailureAndBackendErrorNeverCommit() async throws {
        for behavior in [PrefixLifecycleBackend.Behavior.malformedTool,.failure] {
            let backend=PrefixLifecycleBackend(behavior);let server=try LocalServer(backend:backend,key:key,model:"qwen")
            let port=try await server.start(port:0)
            _=try await URLSession.shared.data(for:request(port,tools:true))
            for _ in 0..<100 where server.snapshot.requests.failed == 0 {try await Task.sleep(for:.milliseconds(5))}
            #expect(backend.snapshot.0 == 0)
            #expect(backend.snapshot.1.contains(behavior == .failure ? .rejectedRequest:.outputValidationFailed))
            try await server.shutdown()
        }
    }
    @Test func disconnectInvalidatesBeforeWorkerRelease() async throws {
        let backend=PrefixLifecycleBackend(.slow);let server=try LocalServer(backend:backend,key:key,model:"qwen")
        let port=try await server.start(port:0);let request=request(port,stream:true)
        let client=Task {let (bytes,_)=try await URLSession.shared.bytes(for:request);for try await _ in bytes.lines {try Task.checkCancellation()}}
        for _ in 0..<100 where !backend.snapshot.2 {try await Task.sleep(for:.milliseconds(5))}
        client.cancel();_=await client.result
        for _ in 0..<200 where server.snapshot.requests.cancelled == 0 {try await Task.sleep(for:.milliseconds(5))}
        #expect(backend.snapshot.0 == 0 && backend.snapshot.1.contains(.cancelled))
        try await server.shutdown()
    }
}
