import Foundation
import QwenFieldfareCore
import Darwin
import Darwin.Mach

public struct RequestTiming: Codable, Sendable {
    public var admissionMs: Double?
    public var queueWaitMs: Double?
    public var formattingMs: Double?
    public var tokenizationMs: Double?
    public var promptTokens: Int?
    public var prefixCache: QwenPrefixCacheSnapshot?
    public var actualPrefillTokens: Int?
    public var expertCacheHits: Int?
    public var expertCacheMisses: Int?
    public var expertLogicalReads: Int?
    public var expertFetchedBytes: Int?
    public var prefillMs: Double?
    public var firstGeneratedTokenMs: Double?
    public var firstVisibleSSEMs: Double?
    public var parserBufferMs: Double?
    public var decodeAfterFirstTokenMs: Double?
    public var outputTokens: Int?
    public var overallMs: Double?
    public var outcome: String = "pending"
}
/// One bounded record, monotonic clocks, no payload or token retention.
public final class RequestTimingRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let received = ProcessInfo.processInfo.systemUptime
    private var admitted: Double?
    private var generated: Double?
    private var parserBegan: Double?
    private var value = RequestTiming()
    public init() {}
    public func enqueue() {lock.withLock {admitted = ProcessInfo.processInfo.systemUptime;value.admissionMs = (admitted! - received) * 1000}}
    public func start() {lock.withLock {value.queueWaitMs = (ProcessInfo.processInfo.systemUptime - (admitted ?? received)) * 1000}}
    public func prompt(formattingMs: Double, tokenizationMs: Double, count: Int) {
        lock.withLock {value.formattingMs = formattingMs; value.tokenizationMs = tokenizationMs; value.promptTokens = count}
    }
    public func replayWork(prefillTokens: Int,hits: Int,misses: Int,logicalReads: Int,fetchedBytes: Int) {
        lock.withLock {value.actualPrefillTokens=prefillTokens;value.expertCacheHits=hits;value.expertCacheMisses=misses;value.expertLogicalReads=logicalReads;value.expertFetchedBytes=fetchedBytes}
    }
    public func cache(_ snapshot: QwenPrefixCacheSnapshot) {lock.withLock {value.prefixCache=snapshot}}
    public func prefill(_ ms: Double) {lock.withLock {value.prefillMs = ms}}
    public func token() {
        lock.withLock {
            if generated == nil {generated = ProcessInfo.processInfo.systemUptime; value.firstGeneratedTokenMs = (generated! - received) * 1000}
        }
    }
    public func generatedOutput(count: Int) {
        lock.withLock {value.outputTokens = count; value.decodeAfterFirstTokenMs = generated.map {(ProcessInfo.processInfo.systemUptime - $0) * 1000}}
    }
    public func visible() {lock.withLock {if value.firstVisibleSSEMs == nil {value.firstVisibleSSEMs = (ProcessInfo.processInfo.systemUptime - received) * 1000}}}
    public func buffering() {lock.withLock {if parserBegan == nil {parserBegan = ProcessInfo.processInfo.systemUptime}}}
    public func committed() {lock.withLock {value.parserBufferMs = parserBegan.map {(ProcessInfo.processInfo.systemUptime - $0) * 1000}}}
    public func finish(_ outcome: String) -> RequestTiming {
        lock.withLock {value.overallMs = (ProcessInfo.processInfo.systemUptime - received) * 1000; value.outcome = outcome; return value}
    }
}
public struct ServerRequestStatistics: Codable, Sendable {
    public var completed: UInt64 = 0
    public var rejected: UInt64 = 0
    public var failed: UInt64 = 0
    public var cancelled: UInt64 = 0
    public var lastRequest: RequestTiming?
}
final class RequestStatistics: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ServerRequestStatistics()
    var snapshot: ServerRequestStatistics {lock.withLock {value}}
    private func increment(_ n: inout UInt64) {if n < UInt64.max {n += 1}}
    func reject() {lock.withLock {increment(&value.rejected)}}
    func finish(_ record: RequestTiming) {
        lock.withLock {
            switch record.outcome {
            case "completed": increment(&value.completed)
            case "cancelled": increment(&value.cancelled)
            default: increment(&value.failed)
            }
            value.lastRequest = record
        }
    }
}
public struct BackendDiagnostics: Codable, Sendable {
    public var context: Int?
    public var memoryBudgetGiB: Int?
    public var metalAllocatedBytes: UInt64?
    public var physicalFootprintBytes: UInt64?
    public var prefixCache: QwenPrefixCacheSnapshot?
    public static var currentPhysicalFootprint: UInt64? {
        var info=task_vm_info_data_t();var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
        let code=withUnsafeMutablePointer(to:&info) {$0.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)}}
        return code == KERN_SUCCESS ? info.phys_footprint:nil
    }
    public init(context: Int? = nil, memoryBudgetGiB: Int? = nil, metalAllocatedBytes: UInt64? = nil,physicalFootprintBytes: UInt64? = nil,prefixCache: QwenPrefixCacheSnapshot? = nil) {
        self.context = context; self.memoryBudgetGiB = memoryBudgetGiB; self.metalAllocatedBytes = metalAllocatedBytes;self.physicalFootprintBytes=physicalFootprintBytes;self.prefixCache=prefixCache
    }
}
public struct ServerHealth: Codable, Sendable {
    public var service = "QwenFieldfareServer"
    public var status = "ready"
    public var model: String
    public var tools = true
    public var active_generations: Int
    public var queued_requests: Int
    public var queue_capacity: Int
    public var diagnostics: BackendDiagnostics
    public var requests: ServerRequestStatistics
}
