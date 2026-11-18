import Foundation
import Testing
@testable import QwenFieldfareServerSupport

private struct TimingMock: GenerationBackend {
    func generate(_ request: ChatRequest,emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        try await generate(request,timing:RequestTimingRecorder(),emit:emit)
    }
    func generate(_ request: ChatRequest,timing: RequestTimingRecorder,emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        timing.prompt(formattingMs:1,tokenizationMs:2,count:7);timing.prefill(3);timing.token();emit("hello")
        timing.generatedOutput(count:1);return Completion(text:"hello",prompt:7,count:1,reason:"stop")
    }
}
@Suite(.serialized) struct TelemetryTests {
    @Test func healthCountersAndTimingsRetainNoPayload() async throws {
        let key="telemetry-test-key-only"
        let server=try LocalServer(backend:TimingMock(),key:key,model:"qwen")
        let port=try await server.start(port:0)
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod="POST";request.setValue("Bearer \(key)",forHTTPHeaderField:"Authorization");request.setValue("application/json",forHTTPHeaderField:"Content-Type")
        request.httpBody=wire(["model":"qwen","messages":[["role":"user","content":"private-conversation-sentinel"]],"stream":true])
        _=try await URLSession.shared.data(for:request)
        for _ in 0..<100 where server.snapshot.requests.completed == 0 {try await Task.sleep(for:.milliseconds(5))}
        let timing=try #require(server.snapshot.requests.lastRequest)
        #expect(timing.promptTokens == 7 && timing.outputTokens == 1 && timing.prefillMs == 3)
        #expect(timing.firstGeneratedTokenMs != nil && timing.firstVisibleSSEMs != nil && timing.overallMs != nil && timing.admissionMs != nil)
        request.setValue("wrong",forHTTPHeaderField:"Authorization");_=try await URLSession.shared.data(for:request)
        #expect(server.snapshot.requests.rejected == 1)
        let encoded=try JSONEncoder().encode(server.snapshot)
        #expect(!String(decoding:encoded,as:UTF8.self).contains("private-conversation-sentinel"))
        #expect(!String(decoding:encoded,as:UTF8.self).contains(key))
        try await server.shutdown()
    }
    @Test func queuedCancellationIsCountedWithoutExecutingWork() async throws {
        let gate=Admission();let stats=RequestStatistics();let queued=Cancellation();let timing=RequestTimingRecorder()
        try gate.submit(Cancellation()) {try? await Task.sleep(for:.seconds(30))}
        try gate.submit(queued,onCancelled:{stats.finish(timing.finish("cancelled"))}) {Issue.record("Cancelled queued work ran")}
        queued.cancel();gate.shutdown()
        for _ in 0..<100 where !gate.idle {try await Task.sleep(for:.milliseconds(5))}
        #expect(gate.idle && stats.snapshot.cancelled == 1)
    }
}

extension TelemetryTests {
    @Test func restartImmediatelyRebindsUsedPort() async throws {
        let first=try LocalServer(backend:TimingMock(),key:"restart-test-credential",model:"qwen")
        let port=try await first.start(port:0)
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/health")!)
        request.setValue("Bearer restart-test-credential",forHTTPHeaderField:"Authorization")
        _=try await URLSession.shared.data(for:request)
        try await first.shutdown()
        let second=try LocalServer(backend:TimingMock(),key:"restart-test-credential",model:"qwen")
        #expect(try await second.start(port:port) == port)
        try await second.shutdown()
    }
}

private struct ToolTimingMock: GenerationBackend {
    func generate(_ request: ChatRequest,emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        emit("<tool_call>")
        try await Task.sleep(for:.milliseconds(10))
        emit("<function=lookup>\n<parameter=item>x</parameter>\n</function>\n</tool_call>")
        return Completion(text:"",prompt:7,count:20,reason:"stop")
    }
}
extension TelemetryTests {
    @Test func bufferingDurationAppearsOnlyForValidatedCalls() async throws {
        let server=try LocalServer(backend:ToolTimingMock(),key:"buffering-test-credential",model:"qwen")
        let port=try await server.start(port:0)
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod="POST";request.setValue("Bearer buffering-test-credential",forHTTPHeaderField:"Authorization");request.setValue("application/json",forHTTPHeaderField:"Content-Type")
        request.httpBody=wire(["model":"qwen","messages":[["role":"user","content":"lookup"]],"tools":[["type":"function","function":["name":"lookup","parameters":["type":"object","properties":["item":["type":"string"]],"required":["item"]]]]],"stream":true])
        let (data,response)=try await URLSession.shared.data(for:request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200 && String(decoding:data,as:UTF8.self).contains("tool_calls"))
        for _ in 0..<100 where server.snapshot.requests.completed == 0 {try await Task.sleep(for:.milliseconds(5))}
        #expect(try #require(server.snapshot.requests.lastRequest?.parserBufferMs) >= 5)
        try await server.shutdown()
    }
}

private final class FlushTrackingMock: GenerationBackend, @unchecked Sendable {
    private let lock=NSLock();private var committed=0
    var count:Int {lock.withLock {committed}}
    func commitGeneration() {lock.withLock {committed+=1}}
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        emit("hello");return Completion(text:"hello",prompt:1,count:1,reason:"stop")
    }
}
extension TelemetryTests {
    @Test func completeFlushCommitsBeforeAnEagerClientCloses() async throws {
        let backend=FlushTrackingMock();let server=try LocalServer(backend:backend,key:"eager-close-test-key",model:"qwen")
        let port=try await server.start(port:0)
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod="POST";request.setValue("Bearer eager-close-test-key",forHTTPHeaderField:"Authorization");request.setValue("application/json",forHTTPHeaderField:"Content-Type")
        request.httpBody=wire(["model":"qwen","messages":[["role":"user","content":"hi"]]])
        for _ in 0..<40 {_=try await URLSession.shared.data(for:request)}
        #expect(backend.count == 40)
        try await server.shutdown()
    }
}
