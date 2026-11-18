import Foundation
import QwenFieldfareCore
import NIOCore
import NIOHTTP1
import NIOPosix

public final class LocalServer: @unchecked Sendable {
    let group: MultiThreadedEventLoopGroup
    private let connectionLock=NSLock()
    private var connections=0
    func reserve() -> Bool {connectionLock.withLock {guard connections < 32 else {return false};connections += 1;return true}}
    func release() {connectionLock.withLock {connections -= 1}}
    let admission: Admission
    let statistics = RequestStatistics()
    let queueCapacity: Int
    let backend: any GenerationBackend
    let key: String
    let model: String
    var listener: Channel?
    public init(backend: any GenerationBackend,key:String,model:String,queueCapacity:Int = 2) throws {
        guard key.utf8.count >= 16, !key.contains("\n"), !model.isEmpty else {throw APIError(400,"Require a bearer secret of at least 16 bytes and model ID")}
        guard (0...8).contains(queueCapacity) else {throw APIError(400,"Queue capacity must be 0–8")}
        self.queueCapacity=queueCapacity;self.admission=Admission(capacity:queueCapacity)
        self.backend=backend;self.key=key;self.model=model
        group=MultiThreadedEventLoopGroup(numberOfThreads:2)
    }
    public var snapshot: ServerHealth {
        let occupancy=admission.occupancy
        return ServerHealth(model:model,active_generations:occupancy.active,queued_requests:occupancy.queued,queue_capacity:queueCapacity,diagnostics:backend.diagnostics,requests:statistics.snapshot)
    }
    public func start(host:String="127.0.0.1",port:Int) async throws -> Int {
        guard host == "127.0.0.1", (0...65535).contains(port) else {throw APIError(400,"Only IPv4 loopback binding is supported")}
        let channel = try await ServerBootstrap(group:group)
            .serverChannelOption(ChannelOptions.backlog,value:32)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr),value:1)
            .childChannelOption(ChannelOptions.autoRead,value:true)
            .childChannelOption(ChannelOptions.writeBufferWaterMark,value:ChannelOptions.Types.WriteBufferWaterMark(low:8192,high:16384))
            .childChannelInitializer { channel in
                guard self.reserve() else {return channel.close()}
                channel.closeFuture.whenComplete {_ in self.release()}
                return channel.pipeline.configureHTTPServerPipeline(withPipeliningAssistance:false).flatMap {
                    channel.pipeline.addHandler(HTTPHandler(server:self))
                }
            }.bind(host:host,port:port).get()
        listener=channel
        return channel.localAddress!.port!
    }
    public func clearPrefixCache() {backend.invalidatePrefixCache(.explicitClear)}
    public func shutdown() async throws {
        backend.invalidatePrefixCache(.shutdown)
        admission.shutdown()
        try await listener?.close().get()
        while !admission.idle {try await Task.sleep(for:.milliseconds(20))}
        try await group.shutdownGracefully()
    }
}
private final class HTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart
    let server: LocalServer
    var head: HTTPRequestHead?
    var body = Data()
    var requestTiming: RequestTimingRecorder?
    var busy=false
    var rejected=false
    var deadline: Scheduled<Void>?
    func handlerAdded(context:ChannelHandlerContext) {let channel=context.channel;deadline=context.eventLoop.scheduleTask(in:.seconds(15)) {channel.close(promise:nil)} }
    let cancellation=Cancellation()
    init(server:LocalServer) {self.server=server}
    func channelInactive(context:ChannelHandlerContext) { deadline?.cancel(); cancellation.cancel() }
    func errorCaught(context:ChannelHandlerContext,error:Error) {cancellation.cancel();context.close(promise:nil)}
    func channelRead(context:ChannelHandlerContext,data:NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let h):
            guard !busy, head == nil else {cancellation.cancel();context.close(promise:nil);return}
            head=h;requestTiming=RequestTimingRecorder()
            guard h.headers["authorization"] == ["Bearer \(server.key)"], h.headers["origin"].isEmpty else {rejected=true;respond(context.channel,status:401,data:APIError(401,"Unauthorized").data);return}
        case .body(var bytes):
            guard !rejected else {return}
            guard body.count + bytes.readableBytes <= 1_048_576 else {rejected=true;respond(context.channel,status:413,data:APIError(413,"Body limit exceeded").data);return}
            body.append(contentsOf:bytes.readBytes(length:bytes.readableBytes) ?? [])
        case .end:
            guard !rejected, let head else {return};busy=true;deadline?.cancel()
            let channel=context.channel
            if head.method == .GET && head.uri == "/health" {
                do {respond(channel,status:200,data:try JSONEncoder().encode(server.snapshot))}
                catch {respond(channel,status:500,data:APIError(500,"Health encoding failed").data)}
                return
            }
            if head.method == .GET && head.uri == "/v1/models" {
                respond(channel,status:200,data:wire(["object":"list","data":[["id":server.model,"object":"model","created":0,"owned_by":"local"]]]));return
            }
            if head.method == .POST && head.uri == "/v1/cache/clear" {
                guard body.isEmpty || (try? ToolJSON.parse(String(decoding:body,as:UTF8.self))) == .object([]) else {respond(channel,status:400,data:APIError(400,"Clear cache takes an empty object").data);return}
                server.clearPrefixCache();respond(channel,status:200,data:wire(["cleared":true]));return
            }
            guard head.method == .POST,head.uri == "/v1/chat/completions" else {respond(channel,status:404,data:APIError(404,"Not found").data);return}
            do {
                guard head.headers["content-type"].first?.split(separator:";").first == "application/json" else {throw APIError(415,"Expected application/json")}
                let request=try ChatRequest(data:body,model:server.model)
                body=Data()
                let timing=requestTiming ?? RequestTimingRecorder();timing.enqueue()
                try server.admission.submit(cancellation,onCancelled:{ [server] in server.backend.invalidatePrefixCache(.cancelled);server.statistics.finish(timing.finish("cancelled")) }) { [server,cancellation] in
                    timing.start()
                    var outcome="failed"
                    defer {server.statistics.finish(timing.finish(outcome))}
                    let output=Output(channel:channel,model:server.model,stream:request.stream,includeUsage:request.includeUsage,timing:timing,tools:request.toolConversation.map { $0.allowsCalls ? $0.tools : [] },cancellation:cancellation)
                    do {
                        let result=try await server.backend.generate(request,timing:timing) { output.delta($0) }
                        if !cancellation.isCancelled {
                            try output.complete(result,usage:request.includeUsage,onFlushed:{_ = cancellation.completeDelivery {server.backend.commitGeneration()}})
                            if !cancellation.isCancelled {outcome="completed"}
                            else {server.backend.invalidatePrefixCache(.cancelled);outcome="cancelled"}
                        } else {server.backend.invalidatePrefixCache(output.parserFailed ? .outputValidationFailed:.cancelled);outcome=output.parserFailed ? "failed":"cancelled"}
                    } catch {
                        let reason: QwenPrefixInvalidation
                        if error is NativeToolError || output.parserFailed {reason = .outputValidationFailed}
                        else if cancellation.isCancelled {reason = .cancelled}
                        else if let api = error as? APIError, (400..<500).contains(api.status) {reason = .rejectedRequest}
                        else {reason = .inferenceError}
                        server.backend.invalidatePrefixCache(reason)
                        if cancellation.isCancelled && !output.parserFailed {outcome="cancelled"}
                        if !cancellation.isCancelled {output.fail(error as? APIError ?? APIError(error is NativeToolError ? 502 : 500,error is NativeToolError ? "Invalid native tool output: \(error)" : "Generation failed"))}
                    }
                    _ = try? await channel.close().get()
                }
            } catch {server.backend.invalidatePrefixCache(.rejectedRequest);respond(channel,status:(error as? APIError)?.status ?? 400,data:(error as? APIError ?? APIError(400,"Invalid request")).data)}
        }
    }
    private func respond(_ channel:Channel,status:Int,data:Data) {
        if status >= 400 {server.statistics.reject()}
        var h=HTTPHeaders();h.add(name:"content-type",value:"application/json");h.add(name:"content-length",value:"\(data.count)");h.add(name:"connection",value:"close")
        channel.write(HTTPServerResponsePart.head(HTTPResponseHead(version:.http1_1,status:.init(statusCode:status),headers:h)),promise:nil)
        channel.write(HTTPServerResponsePart.body(.byteBuffer(channel.allocator.buffer(bytes:data))),promise:nil)
        channel.writeAndFlush(HTTPServerResponsePart.end(nil)).whenComplete {_ in channel.close(promise:nil)}
    }
}
/// Only called by one admitted generation job; one synchronous flush at a time
/// bounds scheduled writes. Slow/non-writable channels cancel instead of buffering.
private final class Output: @unchecked Sendable {
    let channel:Channel;let model:String;let stream:Bool;let includeUsage:Bool;let cancellation:Cancellation
    let id="chatcmpl-"+UUID().uuidString;let created=Int(Date().timeIntervalSince1970)
    var started=false
    var parser: NativeToolParser?
    var parserFailed=false
    let timing: RequestTimingRecorder
    init(channel:Channel,model:String,stream:Bool,includeUsage:Bool,timing:RequestTimingRecorder,tools:[FunctionTool]?,cancellation:Cancellation) {
        self.channel=channel;self.model=model;self.stream=stream;self.includeUsage=includeUsage;self.cancellation=cancellation;self.timing=timing
        if let tools {parser=NativeToolParser(tools:tools,emit:{ [weak self] text in self?.proseDelta(text) })}
    }
    func write(_ part:HTTPServerResponsePart,onFlushed: (@Sendable () -> Void)? = nil) throws {
        guard !cancellation.isCancelled,channel.isActive,channel.isWritable else {cancellation.cancel();throw CancellationError()}
        let channel=self.channel
        let timeout=channel.eventLoop.scheduleTask(in:.seconds(10)) {channel.close(promise:nil)}
        defer {timeout.cancel()}
        if let onFlushed {
            // Register the terminal flush callback on the event loop itself, so
            // the expected peer close cannot race ahead of the commit boundary.
            let promise=channel.eventLoop.makePromise(of:Void.self)
            channel.eventLoop.execute {
                channel.writeAndFlush(part).whenComplete {result in
                    if case .success=result {onFlushed()}
                    promise.completeWith(result)
                }
            }
            try promise.futureResult.wait()
        } else {try channel.writeAndFlush(part).wait()}
    }
    func begin() throws {
        if started {return};started=true
        var headers=HTTPHeaders();headers.add(name:"content-type",value:stream ? "text/event-stream":"application/json");headers.add(name:"connection",value:"close");headers.add(name:"cache-control",value:"no-cache")
        try write(.head(.init(version:.http1_1,status:.ok,headers:headers)))
        if stream {try send(chunk(delta:["role":"assistant","content":""]))}
    }
    func chunk(delta:[String:Any],reason:String?=nil) -> [String:Any] {
        var value:[String:Any] = ["id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[["index":0,"delta":delta,"finish_reason":reason as Any? ?? NSNull()]]]
        if includeUsage {value["usage"]=NSNull()}
        return value
    }
    func send(_ json:[String:Any]) throws {try write(.body(.byteBuffer(channel.allocator.buffer(bytes:sse(json)))))}
    func delta(_ text:String) {
        guard !parserFailed else {return}
        do {
            if let parser {try parser.append(text);if parser.isBufferingCalls {timing.buffering()}} else {proseDelta(text)}
        } catch {
            parserFailed=true
            fail(APIError(502,"Invalid native tool output: \(error)"))
            channel.close(promise:nil)
        }
    }
    func proseDelta(_ text:String) {
        guard stream else {return}
        do {try begin();try send(chunk(delta:["content":text]));if !text.isEmpty {timing.visible()}} catch {cancellation.cancel()}
    }
    func complete(_ value:Completion,usage:Bool,onFlushed: @escaping @Sendable () -> Void) throws {
        guard !parserFailed else {throw NativeToolError.closed}
        let parsed = try parser?.finish(endedNormally:value.reason == "stop")
        let calls = parsed?.calls ?? []
        if !calls.isEmpty {timing.committed()}
        let content = parsed?.content ?? value.text
        let reason = calls.isEmpty ? value.reason : "tool_calls"
        try begin()
        let counts:[String:Any]=["prompt_tokens":value.prompt,"completion_tokens":value.count,"total_tokens":value.prompt+value.count]
        if stream {
            for (index,call) in calls.enumerated() {
                try send(chunk(delta:["tool_calls":[["index":index,"id":call.id,"type":"function","function":["name":call.name,"arguments":""]]]]))
                for fragment in argumentFragments(call.arguments.encoded) {
                    try send(chunk(delta:["tool_calls":[["index":index,"function":["arguments":fragment]]]]))
                }
            }
            try send(chunk(delta:[:],reason:reason))
            if usage {try send(["id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[],"usage":counts])}
            try write(.body(.byteBuffer(channel.allocator.buffer(string:"data: [DONE]\n\n"))))
        } else {
            var message:[String:Any] = ["role":"assistant","content":content]
            if !calls.isEmpty {message["tool_calls"]=calls.map(\.wire);if content.isEmpty {message["content"]=NSNull()}}
            let data=wire(["id":id,"object":"chat.completion","created":created,"model":model,"choices":[["index":0,"message":message,"finish_reason":reason]],"usage":counts])
            try write(.body(.byteBuffer(channel.allocator.buffer(bytes:data))))
        }
        try write(.end(nil),onFlushed:onFlushed)
    }
    func fail(_ error:APIError) {
        do {
            if !started {
                var h=HTTPHeaders();h.add(name:"content-type",value:"application/json");h.add(name:"connection",value:"close")
                try write(.head(.init(version:.http1_1,status:.init(statusCode:error.status),headers:h)))
                try write(.body(.byteBuffer(channel.allocator.buffer(bytes:error.data))))
            } else {try write(.body(.byteBuffer(channel.allocator.buffer(bytes:Data("data: ".utf8)+error.data+Data("\n\n".utf8)))))}
            try write(.end(nil))
        } catch {cancellation.cancel()}
    }
}
