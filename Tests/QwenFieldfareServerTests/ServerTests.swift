import Foundation
import Testing
@testable import QwenFieldfareServerSupport

private struct Mock: GenerationBackend {
    var failure=false
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        if failure {throw APIError(400,"Context overflow")}
        for word in ["Hello"," world"] {try Task.checkCancellation();emit(word);try await Task.sleep(for:.milliseconds(10))}
        return Completion(text:"Hello world",prompt:4,count:2,reason:"stop")
    }
}
private final class SlowMock: GenerationBackend, @unchecked Sendable {
    private let lock=NSLock();private var observed=false
    var cancelled:Bool {lock.withLock {observed}}
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        do {
            for _ in 0..<1000 {try Task.checkCancellation();emit("x");try await Task.sleep(for:.milliseconds(10))}
        } catch {lock.withLock {observed=true};throw error}
        return Completion(text:"x",prompt:1,count:1,reason:"length")
    }
}
@Suite(.serialized) struct ServerTests {
    let secret="test-secret-not-production"
    func request(_ extra:[String:Any]=[:]) throws -> ChatRequest {
        var json:[String:Any]=["model":"qwen","messages":[["role":"user","content":"hi"]]]
        json.merge(extra) {_,new in new};return try ChatRequest(data:wire(json),model:"qwen")
    }
    @Test func validation() throws {
        #expect(try request().configuration.maxNewTokens == 256)
        for invalid:[String:Any] in [["tools":[]],["temperature":true],["stream":"yes"],["max_tokens":5000],["n":2],["top_k":300],["stream_options":["include_usage":true]],["model":"wrong"],["messages":[["role":"tool","content":"no"]]],["max_tokens":2,"max_completion_tokens":3]] {
            #expect(throws:APIError.self) {try request(invalid)}
        }
        #expect(throws:APIError.self) {try ChatRequest(data:Data("{broken".utf8),model:"qwen")}
    }
    @Test func admissionQueueCancellationAndShutdown() async throws {
        let gate=Admission();let first=Cancellation();let second=Cancellation();let third=Cancellation()
        try gate.submit(first) {try? await Task.sleep(for:.seconds(30))}
        try gate.submit(second) {try? await Task.sleep(for:.seconds(30))}
        try gate.submit(third) {try? await Task.sleep(for:.seconds(30))}
        #expect(throws:APIError.self) {try gate.submit(Cancellation()) {}}
        gate.shutdown()
        for _ in 0..<100 where !gate.idle {try await Task.sleep(for:.milliseconds(10))}
        #expect(gate.idle && first.isCancelled && second.isCancelled && third.isCancelled)
        #expect(throws:APIError.self) {try gate.submit(Cancellation()) {}}
    }
    func call(_ port:Int,path:String="/v1/chat/completions",json:[String:Any]?=nil,key:String?=nil) async throws -> (Data,HTTPURLResponse) {
        var req=URLRequest(url:URL(string:"http://127.0.0.1:\(port)\(path)")!)
        req.setValue("Bearer \(key ?? secret)",forHTTPHeaderField:"authorization")
        if let json {req.httpMethod="POST";req.httpBody=wire(json);req.setValue("application/json",forHTTPHeaderField:"content-type")}
        let (data,response)=try await URLSession.shared.data(for:req)
        return (data,response as! HTTPURLResponse)
    }
    @Test func realHTTPProtocolAndSSE() async throws {
        let server=try LocalServer(backend:Mock(),key:secret,model:"qwen");let port=try await server.start(port:0)
        let health=try await call(port,path:"/health");#expect(health.1.statusCode == 200)
        let models=try await call(port,path:"/v1/models");#expect(String(decoding:models.0,as:UTF8.self).contains("qwen"))
        let denied=try await call(port,path:"/health",key:"wrong");#expect(denied.1.statusCode == 401)
        let json:[String:Any]=["model":"qwen","messages":[["role":"system","content":"hello"],["role":"user","content":"hi"]],"max_tokens":2]
        let result=try await call(port,json:json);#expect(result.1.statusCode == 200)
        #expect(String(decoding:result.0,as:UTF8.self).contains("Hello world"))
        var streaming=json;streaming["stream"]=true;streaming["stream_options"]=["include_usage":true]
        let result2=try await call(port,json:streaming)
        let text=String(decoding:result2.0,as:UTF8.self)
        #expect(text.hasSuffix("data: [DONE]\n\n"))
        for line in text.components(separatedBy:"\n\n").filter({!$0.isEmpty && !$0.contains("[DONE]")}) {
            #expect(line.hasPrefix("data: "))
            #expect((try? JSONSerialization.jsonObject(with:Data(line.dropFirst(6).utf8))) != nil)
        }
        #expect(text.contains("completion_tokens"))
        try await server.shutdown()
    }
    @Test func clientDisconnectCancelsAndWorkerCanShutdown() async throws {
        let backend=SlowMock();let server=try LocalServer(backend:backend,key:secret,model:"qwen")
        let port=try await server.start(port:0)
        var req=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/v1/chat/completions")!)
        req.httpMethod="POST";req.setValue("Bearer \(secret)",forHTTPHeaderField:"authorization");req.setValue("application/json",forHTTPHeaderField:"content-type")
        req.httpBody=wire(["model":"qwen","messages":[["role":"user","content":"hi"]],"stream":true])
        let task=Task {let (bytes,_)=try await URLSession.shared.bytes(for:req);for try await _ in bytes.lines {try Task.checkCancellation()}}
        try await Task.sleep(for:.milliseconds(100));task.cancel();_ = await task.result
        for _ in 0..<100 where !backend.cancelled {try await Task.sleep(for:.milliseconds(10))}
        #expect(backend.cancelled)
        try await server.shutdown()
    }
    @Test func initializationFailureIsRecoverable() async {
        do {_ = try await CoreBackend(path:"/definitely-missing-qwen-checkpoint",context:8,budget:.threeGiB);Issue.record("Missing model accepted")}
        catch {}
    }
    @Test func failureAndLoopbackSecurity() async throws {
        #expect(throws:APIError.self) {try LocalServer(backend:Mock(),key:"short",model:"qwen")}
        let server=try LocalServer(backend:Mock(failure:true),key:secret,model:"qwen")
        do {_ = try await server.start(host:"0.0.0.0",port:0);Issue.record("External binding accepted")} catch is APIError {}
        let port=try await server.start(port:0)
        let result=try await call(port,json:["model":"qwen","messages":[["role":"user","content":"hi"]]])
        #expect(result.1.statusCode == 400)
        try await server.shutdown()
    }
}
