import Foundation
import Testing
@testable import QwenFieldfareServerSupport

private struct NativeMock: GenerationBackend {
    let native: String
    var reason = "stop"
    func generate(_ request: ChatRequest, emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        for character in native {try Task.checkCancellation();emit(String(character))}
        return .init(text:native,prompt:12,count:64,reason:reason)
    }
}
@Suite(.serialized) struct ToolHTTPTests {
    let key = "tool-test-key-not-production"
    let fixture = ToolContractTests()
    func call(_ port: Int, body: Data) async throws -> (Data,Int) {
        var request = URLRequest(url:URL(string:"http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod="POST";request.httpBody=body
        request.setValue("Bearer \(key)",forHTTPHeaderField:"authorization");request.setValue("application/json",forHTTPHeaderField:"content-type")
        let (data,response)=try await URLSession.shared.data(for:request)
        return (data,(response as! HTTPURLResponse).statusCode)
    }
    func body(stream: Bool = false, choice: String = "auto") throws -> Data {
        wire(["model":"qwen","messages":[["role":"user","content":"check"]],"tools":[try JSONSerialization.jsonObject(with:Data(fixture.tool.utf8))],"stream":stream,"tool_choice":choice,"stream_options":stream ? ["include_usage":true] : [:]])
    }
    @Test func jsonAndSSEToolObjectsRoundTrip() async throws {
        let long = String(repeating:"雪 and \"quotes\" line\n",count:12)
        let text = "Checking.\n\n" + fixture.native.replacingOccurrences(of:"fieldfare",with:long) + "\n" + fixture.native
        let server=try LocalServer(backend:NativeMock(native:text),key:key,model:"qwen");let port=try await server.start(port:0)
        var object = try #require(JSONSerialization.jsonObject(with:body(stream:true)) as? [String:Any]);object.removeValue(forKey:"stream_options");object["stream"]=false
        let (data,status)=try await call(port,body:wire(object));#expect(status == 200)
        let json=try #require(JSONSerialization.jsonObject(with:data) as? [String:Any])
        let choice=try #require((json["choices"] as? [[String:Any]])?.first)
        #expect(choice["finish_reason"] as? String == "tool_calls")
        let message=try #require(choice["message"] as? [String:Any]);let calls=try #require(message["tool_calls"] as? [[String:Any]])
        #expect(calls.count == 2 && message["content"] as? String == "Checking.\n\n")
        let (stream,status2)=try await call(port,body:body(stream:true));#expect(status2 == 200)
        let textStream=String(decoding:stream,as:UTF8.self);#expect(textStream.hasSuffix("data: [DONE]\n\n"))
        var arguments: [Int:String]=[:];var names: [Int:String]=[:];var ids=Set<String>();var responseIDs=Set<String>();var finish="";var content=""
        for line in textStream.components(separatedBy:"\n\n") where line.hasPrefix("data: {") {
            let chunk=try #require(JSONSerialization.jsonObject(with:Data(line.dropFirst(6).utf8)) as? [String:Any]);responseIDs.insert(chunk["id"] as? String ?? "")
            for choice in chunk["choices"] as? [[String:Any]] ?? [] {
                if let r=choice["finish_reason"] as? String {finish=r}
                let delta=choice["delta"] as? [String:Any] ?? [:];content += delta["content"] as? String ?? ""
                for call in delta["tool_calls"] as? [[String:Any]] ?? [] {
                    let index=try #require(call["index"] as? Int)
                    if let id=call["id"] as? String {#expect(ids.insert(id).inserted)}
                    let function=try #require(call["function"] as? [String:Any])
                    if let name=function["name"] as? String {#expect(names[index] == nil);names[index]=name}
                    arguments[index,default:""] += function["arguments"] as? String ?? ""
                }
            }
        }
        #expect(responseIDs.count == 1 && ids.count == 2 && finish == "tool_calls")
        #expect(content == "Checking.\n\n" && !content.contains("<tool"))
        #expect(try ToolJSON.parse(#require(arguments[0]))["project"]?.string == long)
        #expect(try ToolJSON.parse(#require(arguments[1]))["project"]?.string == "fieldfare")
        #expect(names == [0:"get_project_status",1:"get_project_status"])
        try await server.shutdown()
    }
    @Test func invalidOutputAndDisabledCallsFailWithoutExecutableDeltas() async throws {
        for (native,reason,choice) in [(String(fixture.native.dropLast()),"stop","auto"),(fixture.native,"length","auto"),(fixture.native,"stop","none")] {
            let server=try LocalServer(backend:NativeMock(native:native,reason:reason),key:key,model:"qwen");let port=try await server.start(port:0)
            let (stream,status)=try await call(port,body:body(stream:true,choice:choice))
            #expect(status == 502)
            let text=String(decoding:stream,as:UTF8.self)
            #expect(text.contains("error") && !text.contains("\"tool_calls\"") && !text.contains("[DONE]"))
            try await server.shutdown()
        }
    }
    @Test func cancellationDoesNotCommitPartialCallAndContextBounds() throws {
        let parser=NativeToolParser(tools:[try fixture.definition()]);try parser.append(String(fixture.native.prefix(25)))
        #expect(throws:NativeToolError.self) {try parser.finish(endedNormally:false)}
        try validateContextReservation(promptTokens:100,outputTokens:20,context:120)
        for counts in [(100,21,120),(120,1,120),(Int.max,1,120),(-1,1,120)] {
            #expect(throws:APIError.self) {try validateContextReservation(promptTokens:counts.0,outputTokens:counts.1,context:counts.2)}
        }
    }
}

private final class SlowNative: GenerationBackend, @unchecked Sendable {
    let lock=NSLock();var value=false
    var cancelled:Bool {lock.withLock {value}}
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        emit("<tool_ca")
        do {try await Task.sleep(for:.seconds(30))} catch {lock.withLock {value=true};throw error}
        return .init(text:"<tool_ca",prompt:1,count:1,reason:"length")
    }
}
extension ToolHTTPTests {
    @Test func disconnectWhileNativeDelimiterPendingCancelsGeneration() async throws {
        let backend=SlowNative();let server=try LocalServer(backend:backend,key:key,model:"qwen");let port=try await server.start(port:0)
        let data=try body(stream:true)
        let task=Task {try await call(port,body:data)}
        try await Task.sleep(for:.milliseconds(100));task.cancel();_ = await task.result
        for _ in 0..<100 where !backend.cancelled {try await Task.sleep(for:.milliseconds(10))}
        #expect(backend.cancelled)
        try await server.shutdown()
    }
    @Test func toolsDisabledCanStillReturnNormalText() async throws {
        let server=try LocalServer(backend:NativeMock(native:"Hello"),key:key,model:"qwen");let port=try await server.start(port:0)
        let (data,status)=try await call(port,body:body(stream:true,choice:"none"))
        #expect(status == 200 && String(decoding:data,as:UTF8.self).contains("[DONE]"))
        #expect(!String(decoding:data,as:UTF8.self).contains("\"tool_calls\""))
        try await server.shutdown()
    }
}
