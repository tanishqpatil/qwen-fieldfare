import Darwin
import Foundation
import Testing
@testable import QwenFieldfareServerSupport

@Suite(.serialized) struct ConfigurationTests {
    let key="configuration-test-key-only"
    func config(_ args: [String] = []) throws -> ServerConfiguration {
        let action=try ServerConfiguration.parse(["--model","/test-model"]+args,environment:["QWENFIELDFARE_API_KEY":key])
        guard case .serve(let value)=action else {throw APIError(500,"Expected config")};return value
    }
    @Test func defaultsAndHelpVersionNeedNoModelOrSecret() throws {
        let c=try config();#expect(c.host == "127.0.0.1" && c.port == 8080 && c.context == 8192 && c.queueCapacity == 2 && c.budget.rawValue == 5)
        if case .help=try ServerConfiguration.parse(["--help"],environment:[:]) {} else {Issue.record("help missing")}
        if case .version=try ServerConfiguration.parse(["--version"],environment:[:]) {} else {Issue.record("version missing")}
        #expect(!String(describing:c).contains(key) && !String(reflecting:c).contains(key))
        #expect(try config(["--queue-capacity","0","--port","0","--memory-gib","3"]).queueCapacity == 0)
    }
    @Test func invalidSettingsAreRecoverableAndSecretSafe() throws {
        for args in [["--host","0.0.0.0"],["--host","localhost"],["--port","65536"],["--port","-1"],["--context","0"],["--context",String(Int.max)],["--memory-gib","6"],["--queue-capacity","9"],["--queue-capacity","-1"],["--port","12","--port","13"],["--port"],["--api-key",key],["--model-id",""]] {
            #expect(throws:APIError.self) {try config(args)}
        }
        for secret in ["short","whitespace is not safe","aaaaaaaaaaaaaaaa\r",String(repeating:"x",count:513)] {
            do {_=try ServerConfiguration.parse(["--model","path"],environment:["QWENFIELDFARE_API_KEY":secret]);Issue.record("Bad key accepted")}
            catch let error as APIError {#expect(!error.message.contains(secret))}
        }
    }
    @Test func restrictedKeyFileAndUnsafeTypes() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("key-test-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let file=root.appendingPathComponent("key");try Data((key+"\n").utf8).write(to:file);#expect(chmod(file.path,0o600) == 0)
        let action=try ServerConfiguration.parse(["--model","path","--api-key-file",file.path],environment:[:])
        if case .serve(let c)=action {#expect(c.apiKey == key)} else {Issue.record("Missing config")}
        #expect(throws:APIError.self) {try ServerConfiguration.parse(["--model","path","--api-key-file",file.path],environment:["QWENFIELDFARE_API_KEY":key])}
        #expect(chmod(file.path,0o644) == 0)
        #expect(throws:APIError.self) {try ServerConfiguration.parse(["--model","path","--api-key-file",file.path],environment:[:])}
        let fifo=root.appendingPathComponent("fifo");#expect(mkfifo(fifo.path,0o600) == 0)
        let link=root.appendingPathComponent("link");try FileManager.default.createSymbolicLink(at:link,withDestinationURL:file)
        for path in [fifo.path,link.path,root.path] {#expect(throws:APIError.self) {try ServerConfiguration.parse(["--model","path","--api-key-file",path],environment:[:])}}
    }
    @Test func configurableQueueSaturatesAtEachBound() async throws {
        for capacity in [0,1,8] {
            let gate=Admission(capacity:capacity)
            for _ in 0...capacity {try gate.submit(Cancellation()) {try? await Task.sleep(for:.seconds(30))}}
            #expect(gate.occupancy.active == 1 && gate.occupancy.queued == capacity)
            #expect(throws:APIError.self) {try gate.submit(Cancellation()) {}}
            gate.shutdown();for _ in 0..<100 where !gate.idle {try await Task.sleep(for:.milliseconds(10))};#expect(gate.idle)
        }
    }
    @Test func installedOpenCodeToolHistoryWireFixture() throws {
        let url=try #require(Bundle.module.url(forResource:"opencode-1.19.5",withExtension:"json",subdirectory:"Fixtures"))
        let request=try ChatRequest(data:Data(contentsOf:url),model:"qwen-fieldfare")
        #expect(request.stream && request.toolConversation?.tools.count == 5)
        #expect(request.toolConversation?.messages.last?.role == "tool")
    }
    @Test func openCodeNumericSchemaBoundsEnforced() throws {
        let schema=try ToolSchema(ToolJSON.parse(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"integer","minimum":-9007199254740991,"exclusiveMinimum":0,"maximum":9007199254740991}"#))
        #expect(schema.accepts(.number("1")) && !schema.accepts(.number("0")) && !schema.accepts(.number("-1")) && !schema.accepts(.number("9007199254740992")))
        let exact=try ToolSchema(ToolJSON.parse(#"{"type":"integer","maximum":9007199254740992}"#))
        #expect(exact.accepts(.number("9007199254740992")) && !exact.accepts(.number("9007199254740993")))
        let exclusive=try ToolSchema(ToolJSON.parse(#"{"type":"integer","exclusiveMaximum":9223372036854775807}"#))
        #expect(exclusive.accepts(.number("9223372036854775806")) && !exclusive.accepts(.number("9223372036854775807")))
        let huge=try ToolSchema(ToolJSON.parse(#"{"type":"number","minimum":1e299,"maximum":1e300}"#))
        #expect(huge.accepts(.number("1e300")) && !huge.accepts(.number("1e298")))
        #expect(throws:APIError.self) {try ToolSchema(ToolJSON.parse(#"{"type":"integer","minimum":"0"}"#))}
        #expect(throws:APIError.self) {try ToolSchema(ToolJSON.parse(#"{"$schema":"unknown","type":"object"}"#))}
    }
}

extension ConfigurationTests {
    @Test func prefixCacheDefaultsOffAndRejectsUnknownModes() throws {
        #expect(try config().prefixCacheMode == .off)
        #expect(try config(["--prefix-cache","exact-prefix"]).prefixCacheMode == .exactPrefix)
        #expect(throws:APIError.self) {try config(["--prefix-cache","approximate"])}
    }
    @Test func successfulFlushProtectsCommitFromExpectedConnectionClose() {
        let token=Cancellation();var commits=0
        #expect(token.completeDelivery {commits+=1})
        token.cancel()
        #expect(!token.isCancelled && commits == 1)
        #expect(!token.completeDelivery {commits+=1})
        let cancelled=Cancellation();cancelled.cancel()
        #expect(!cancelled.completeDelivery {commits+=1})
        #expect(commits == 1)
    }
}
