import Foundation
import QwenFieldfareCore

public final class Cancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var task: Task<Void,Never>?
    public init() {}
    public var isCancelled: Bool { lock.withLock { cancelled } }
    public func attach(_ task: Task<Void,Never>) { lock.withLock { self.task = task; if cancelled { task.cancel() } } }
    public func cancel() { lock.withLock { cancelled = true; task?.cancel() } }
    public func finish() { lock.withLock { task = nil } }
}
/// Serial admission independently bounds asynchronous work; actor reentrancy
/// cannot permit simultaneous inference. Jobs retain all model resources.
public final class Admission: @unchecked Sendable {
    private let lock = NSLock()
    private var jobs: [(Cancellation,@Sendable () async -> Void)] = []
    private var active: Cancellation?
    private var closed = false
    public init() {}
    public var idle: Bool { lock.withLock { active == nil } }
    public func submit(_ cancellation: Cancellation, work: @escaping @Sendable () async -> Void) throws {
        try lock.withLock {
            guard !closed else { throw APIError(503,"Server stopping") }
            guard active == nil || jobs.count < 2 else { throw APIError(429,"Generation queue full") }
            if active == nil { active = cancellation; start(cancellation,work) } else { jobs.append((cancellation,work)) }
        }
    }
    private func start(_ cancellation: Cancellation, _ work: @escaping @Sendable () async -> Void) {
        let task = Task.detached {
            if !cancellation.isCancelled { await work() }
            cancellation.finish()
            self.lock.withLock {
                self.active = nil
                if !self.jobs.isEmpty {
                    let next = self.jobs.removeFirst(); self.active = next.0; self.start(next.0,next.1)
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
    func generate(_ request: ChatRequest, emit: @escaping @Sendable (String) -> Void) async throws -> Completion
}
public final class CoreBackend: GenerationBackend, @unchecked Sendable {
    private let runner: QwenQ4ActiveExpertStreamingRunner
    private let tokenizer: QwenTokenizer
    private let context: Int
    public init(path: String, context: Int, budget: QwenActiveStreamingMemoryBudget) async throws {
        let path = try QwenModelDirectory.validate(path:path)
        let plan = try QwenActiveStreamingMemoryConfiguration.make(budget:budget,contextLength:context)
        self.context = context
        tokenizer = try await QwenTokenizer.load(from:URL(fileURLWithPath:path))
        runner = try QwenQ4ActiveExpertStreamingRunner(modelPath:path,maxSeqLen:context,expertCacheCapacity:plan.expertCacheSlots,usePersistentExpertCache:true,keepDenseWeightsResident:true)
    }
    public func generate(_ request: ChatRequest, emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        let last = request.messages.last!
        let prompt = QwenChatTemplate.format(messages:Array(request.messages.dropLast()),appendingUser:last.content,enableThinking:false)
        let tokens = tokenizer.encode(prompt)
        guard tokens.count < context, request.configuration.maxNewTokens <= context-tokens.count else { throw APIError(400,"Prompt plus output reservation exceeds context") }
        let decoder = Decoder(tokenizer:tokenizer,emit:emit)
        let result = try await runner.generateResult(promptTokens:tokens,configuration:request.configuration,onToken:{ decoder.append($0) })
        try decoder.flush()
        let reason: String
        switch result.finishReason { case .endOfSequence: reason="stop"; case .contextLength,.maxNewTokens: reason="length" }
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
