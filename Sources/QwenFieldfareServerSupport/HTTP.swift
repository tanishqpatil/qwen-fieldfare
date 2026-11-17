import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix

public final class LocalServer: @unchecked Sendable {
    let group: MultiThreadedEventLoopGroup
    private let connectionLock=NSLock()
    private var connections=0
    func reserve() -> Bool {connectionLock.withLock {guard connections < 32 else {return false};connections += 1;return true}}
    func release() {connectionLock.withLock {connections -= 1}}
    let admission = Admission()
    let backend: any GenerationBackend
    let key: String
    let model: String
    var listener: Channel?
    public init(backend: any GenerationBackend,key:String,model:String) throws {
        guard key.utf8.count >= 16, !key.contains("\n"), !model.isEmpty else {throw APIError(400,"Require a bearer secret of at least 16 bytes and model ID")}
        self.backend=backend;self.key=key;self.model=model
        group=MultiThreadedEventLoopGroup(numberOfThreads:2)
    }
    public func start(host:String="127.0.0.1",port:Int) async throws -> Int {
        guard host == "127.0.0.1", (0...65535).contains(port) else {throw APIError(400,"Only IPv4 loopback binding is supported")}
        let channel = try await ServerBootstrap(group:group)
            .serverChannelOption(ChannelOptions.backlog,value:32)
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
    public func shutdown() async throws {
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
            head=h
            guard h.headers["authorization"] == ["Bearer \(server.key)"], h.headers["origin"].isEmpty else {rejected=true;respond(context.channel,status:401,data:APIError(401,"Unauthorized").data);return}
        case .body(var bytes):
            guard !rejected else {return}
            guard body.count + bytes.readableBytes <= 1_048_576 else {rejected=true;respond(context.channel,status:413,data:APIError(413,"Body limit exceeded").data);return}
            body.append(contentsOf:bytes.readBytes(length:bytes.readableBytes) ?? [])
        case .end:
            guard !rejected, let head else {return};busy=true;deadline?.cancel()
            let channel=context.channel
            if head.method == .GET && head.uri == "/health" {
                respond(channel,status:200,data:wire(["status":"ready","model":server.model,"tools":false]));return
            }
            if head.method == .GET && head.uri == "/v1/models" {
                respond(channel,status:200,data:wire(["object":"list","data":[["id":server.model,"object":"model","created":0,"owned_by":"local"]]]));return
            }
            guard head.method == .POST,head.uri == "/v1/chat/completions" else {respond(channel,status:404,data:APIError(404,"Not found").data);return}
            do {
                guard head.headers["content-type"].first?.split(separator:";").first == "application/json" else {throw APIError(415,"Expected application/json")}
                let request=try ChatRequest(data:body,model:server.model)
                body=Data()
                try server.admission.submit(cancellation) { [server,cancellation] in
                    let output=Output(channel:channel,model:server.model,stream:request.stream,includeUsage:request.includeUsage,cancellation:cancellation)
                    do {
                        let result=try await server.backend.generate(request) { output.delta($0) }
                        if !cancellation.isCancelled {try output.complete(result,usage:request.includeUsage)}
                    } catch {
                        if !cancellation.isCancelled {output.fail(error as? APIError ?? APIError(500,"Generation failed"))}
                    }
                    _ = try? await channel.close().get()
                }
            } catch {respond(channel,status:(error as? APIError)?.status ?? 400,data:(error as? APIError ?? APIError(400,"Invalid request")).data)}
        }
    }
    private func respond(_ channel:Channel,status:Int,data:Data) {
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
    init(channel:Channel,model:String,stream:Bool,includeUsage:Bool,cancellation:Cancellation) {self.channel=channel;self.model=model;self.stream=stream;self.includeUsage=includeUsage;self.cancellation=cancellation}
    func write(_ part:HTTPServerResponsePart) throws {
        guard !cancellation.isCancelled,channel.isActive,channel.isWritable else {cancellation.cancel();throw CancellationError()}
        let channel=self.channel
        let timeout=channel.eventLoop.scheduleTask(in:.seconds(10)) {channel.close(promise:nil)}
        defer {timeout.cancel()}
        try channel.writeAndFlush(part).wait()
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
        guard stream else {return}
        do {try begin();try send(chunk(delta:["content":text]))} catch {cancellation.cancel()}
    }
    func complete(_ value:Completion,usage:Bool) throws {
        try begin()
        let counts:[String:Any]=["prompt_tokens":value.prompt,"completion_tokens":value.count,"total_tokens":value.prompt+value.count]
        if stream {
            try send(chunk(delta:[:],reason:value.reason))
            if usage {try send(["id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[],"usage":counts])}
            try write(.body(.byteBuffer(channel.allocator.buffer(string:"data: [DONE]\n\n"))))
        } else {
            let data=wire(["id":id,"object":"chat.completion","created":created,"model":model,"choices":[["index":0,"message":["role":"assistant","content":value.text],"finish_reason":value.reason]],"usage":counts])
            try write(.body(.byteBuffer(channel.allocator.buffer(bytes:data))))
        }
        try write(.end(nil))
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
