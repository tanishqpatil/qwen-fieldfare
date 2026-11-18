import Foundation
import CryptoKit
import Darwin
import Metal
import QwenFieldfareCore

public final class Cancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var delivered = false
    private var task: Task<Void,Never>?
    public init() {}
    public var isCancelled: Bool { lock.withLock { cancelled } }
    public func attach(_ task: Task<Void,Never>) { lock.withLock { self.task = task; if cancelled { task.cancel() } } }
    public func cancel() {lock.withLock {guard !delivered else {return};cancelled = true;task?.cancel()}}
    public func completeDelivery(_ commit: () -> Void) -> Bool {
        lock.withLock {guard !cancelled,!delivered else {return false};delivered = true;commit();return true}
    }
    public func finish() { lock.withLock { task = nil } }
}
/// Serial admission independently bounds asynchronous work; actor reentrancy
/// cannot permit simultaneous inference. Jobs retain all model resources.
public final class Admission: @unchecked Sendable {
    private let lock = NSLock()
    private var jobs: [(Cancellation,@Sendable () async -> Void,@Sendable () -> Void)] = []
    private var active: Cancellation?
    private var closed = false
    private let capacity: Int
    public init(capacity: Int = 2) {self.capacity = capacity}
    public var occupancy: (active: Int, queued: Int) {lock.withLock {(active == nil ? 0:1,jobs.count)}}
    public var idle: Bool { lock.withLock { active == nil } }
    public func submit(_ cancellation: Cancellation, onCancelled: @escaping @Sendable () -> Void = {}, work: @escaping @Sendable () async -> Void) throws {
        try lock.withLock {
            guard !closed else { throw APIError(503,"Server stopping") }
            guard active == nil || jobs.count < capacity else { throw APIError(429,"Generation queue full") }
            if active == nil { active = cancellation; start(cancellation,work,onCancelled) } else { jobs.append((cancellation,work,onCancelled)) }
        }
    }
    private func start(_ cancellation: Cancellation, _ work: @escaping @Sendable () async -> Void,_ onCancelled: @escaping @Sendable () -> Void) {
        let task = Task.detached {
            if !cancellation.isCancelled { await work() } else {onCancelled()}
            cancellation.finish()
            self.lock.withLock {
                self.active = nil
                if !self.jobs.isEmpty {
                    let next = self.jobs.removeFirst(); self.active = next.0; self.start(next.0,next.1,next.2)
                }
            }
        }
        cancellation.attach(task)
    }
    public func shutdown() { lock.withLock { closed = true; active?.cancel(); for job in jobs { job.0.cancel() } } }
}
public struct Completion: Sendable {
    public let text: String; public let prompt: Int; public let count: Int; public let reason: String
    public init(text: String,prompt: Int,count: Int,reason: String) { self.text=text;self.prompt=prompt;self.count=count;self.reason=reason }
}
public protocol GenerationBackend: Sendable {
    func commitGeneration()
    func invalidatePrefixCache(_ reason: QwenPrefixInvalidation)
    var diagnostics: BackendDiagnostics {get}
    func generate(_ request: ChatRequest, timing: RequestTimingRecorder, emit: @escaping @Sendable (String) -> Void) async throws -> Completion
    func generate(_ request: ChatRequest, emit: @escaping @Sendable (String) -> Void) async throws -> Completion
}
public extension GenerationBackend {
    func commitGeneration() {}
    func invalidatePrefixCache(_ reason: QwenPrefixInvalidation) {}
    var diagnostics: BackendDiagnostics {BackendDiagnostics()}
    func generate(_ request: ChatRequest, timing: RequestTimingRecorder, emit: @escaping @Sendable (String) -> Void) async throws -> Completion {try await generate(request,emit:emit)}
}
public final class CoreBackend: GenerationBackend, @unchecked Sendable {
    private let runner: QwenQ4ActiveExpertStreamingRunner
    private let tokenizer: QwenTokenizer
    private let context: Int
    private let toolPrompt: ToolPrompt
    private let budget: Int
    public var diagnostics: BackendDiagnostics {BackendDiagnostics(context:context,memoryBudgetGiB:budget,metalAllocatedBytes:MTLCreateSystemDefaultDevice().map {UInt64($0.currentAllocatedSize)},physicalFootprintBytes:BackendDiagnostics.currentPhysicalFootprint,prefixCache:runner.prefixCacheSnapshot)}
    public init(path: String, context: Int, budget: QwenActiveStreamingMemoryBudget,prefixCacheMode: QwenPrefixCacheMode = .off) async throws {
        let path = try QwenModelDirectory.validate(path:path)
        let plan = try QwenActiveStreamingMemoryConfiguration.make(budget:budget,contextLength:context)
        self.context = context; self.budget = budget.rawValue
        let root = URL(fileURLWithPath:path).resolvingSymlinksInPath()
        let url = root.appendingPathComponent("chat_template.jinja").resolvingSymlinksInPath()
        guard url.deletingLastPathComponent().path == root.path else {throw APIError(400,"Template resolves outside model root")}
        let fd = open(url.path,O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC)
        guard fd >= 0 else {throw APIError(400,"Cannot open checkpoint template")}
        let handle = FileHandle(fileDescriptor:fd,closeOnDealloc:true)
        defer {try? handle.close()}
        var metadata = stat()
        guard fstat(fd,&metadata) == 0, metadata.st_mode & S_IFMT == S_IFREG,
              metadata.st_size <= 262_144, let bytes = try handle.read(upToCount:262_145), bytes.count <= 262_144,
              let source = String(data:bytes,encoding:.utf8) else {throw APIError(400,"Invalid checkpoint template")}
        toolPrompt = try ToolPrompt(source:source)
        tokenizer = try await QwenTokenizer.load(from:URL(fileURLWithPath:path))
        runner = try QwenQ4ActiveExpertStreamingRunner(modelPath:path,maxSeqLen:context,expertCacheCapacity:plan.expertCacheSlots,usePersistentExpertCache:true,keepDenseWeightsResident:true)
        runner.setPrefixCacheMode(prefixCacheMode)
    }
    public func commitGeneration() {runner.commitPrefixCache()}
    public func invalidatePrefixCache(_ reason: QwenPrefixInvalidation) {runner.invalidatePrefixCache(reason)}
    public func generate(_ request: ChatRequest, emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        try await generate(request,timing:RequestTimingRecorder(),emit:emit)
    }
    public func generate(_ request: ChatRequest, timing: RequestTimingRecorder, emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        let epoch = runner.prefixCacheEpoch
        let cacheBefore=runner.expertCache.metricsSnapshot()
        var succeeded = false
        var enteredRunner = false
        defer {
            if !succeeded {runner.invalidatePrefixCache(Task.isCancelled ? .cancelled:(enteredRunner ? .inferenceError:.rejectedRequest))}
            let after=runner.expertCache.metricsSnapshot();let fetched=after.fetchedBytes-cacheBefore.fetchedBytes
            timing.replayWork(prefillTokens:enteredRunner ? runner.prefixCacheSnapshot.processedPromptTokens:0,hits:after.hits-cacheBefore.hits,misses:after.misses-cacheBefore.misses,logicalReads:fetched/Q4BoundedExpertCache.expertByteLength,fetchedBytes:fetched)
            if enteredRunner {timing.cache(runner.prefixCacheSnapshot)}
        }
        let formattingStarted = ProcessInfo.processInfo.systemUptime
        let last = request.messages.last!
        let prompt = try request.toolConversation.map {try toolPrompt.render($0)}
            ?? QwenChatTemplate.format(messages:Array(request.messages.dropLast()),appendingUser:last.content,enableThinking:false)
        let tokenizationStarted = ProcessInfo.processInfo.systemUptime
        let tokens = tokenizer.encode(prompt)
        timing.prompt(formattingMs:(tokenizationStarted-formattingStarted)*1000,tokenizationMs:(ProcessInfo.processInfo.systemUptime-tokenizationStarted)*1000,count:tokens.count)
        try validateContextReservation(promptTokens:tokens.count,outputTokens:request.configuration.maxNewTokens,context:context)
        let decoder = Decoder(tokenizer:tokenizer,emit:emit)
        let policy=ToolJSON.object([.init("format",.string(request.toolConversation == nil ? "core-text-thinking-off":"checkpoint-jinja-thinking-off")),.init("tools",.array(request.toolConversation?.tools.map(\.json) ?? [])),.init("allows",.bool(request.toolConversation?.allowsCalls ?? false))])
        let identity=SHA256.hash(data:Data(policy.encoded.utf8)).map {String(format:"%02x",$0)}.joined()
        enteredRunner = true
        let result = try await runner.generateResult(promptTokens:tokens,configuration:request.configuration,prefixIdentity:identity,prefixEpoch:epoch,onPrefillComplete:{timing.prefill($0)},onToken:{timing.token(); decoder.append($0) })
        timing.generatedOutput(count:result.tokens.count)
        try decoder.flush()
        let reason: String
        switch result.finishReason { case .endOfSequence: reason="stop"; case .contextLength,.maxNewTokens: reason="length" }
        succeeded = true
        return Completion(text:tokenizer.decode(result.tokens),prompt:tokens.count,count:result.tokens.count,reason:reason)
    }
    private final class Decoder: @unchecked Sendable {
        let tokenizer: QwenTokenizer; let emit: @Sendable (String)->Void
        var tokens:[Int32]=[]; var sent=""
        init(tokenizer:QwenTokenizer,emit:@escaping @Sendable (String)->Void) {self.tokenizer=tokenizer;self.emit=emit}
        func append(_ token:Int32) {tokens.append(token); let text=tokenizer.decode(tokens); if !text.hasSuffix("\u{FFFD}"),text.utf8.starts(with:sent.utf8) {let delta=String(decoding:text.utf8.dropFirst(sent.utf8.count),as:UTF8.self);sent=text;if !delta.isEmpty {emit(delta)}} }
        func flush() throws {let text=tokenizer.decode(tokens); guard text.utf8.starts(with:sent.utf8) else {throw APIError(500,"Incremental decoding failed")};let tail=String(decoding:text.utf8.dropFirst(sent.utf8.count),as:UTF8.self);if !tail.isEmpty {emit(tail)} }
    }
}

public func validateContextReservation(promptTokens: Int, outputTokens: Int, context: Int) throws {
    guard promptTokens >= 0, outputTokens > 0, context > 0, promptTokens < context,
          outputTokens <= context - promptTokens else {throw APIError(400,"Prompt plus output reservation exceeds context")}
}
