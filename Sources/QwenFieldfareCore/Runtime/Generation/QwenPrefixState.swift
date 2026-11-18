import Foundation

public enum QwenPrefixCacheMode: String, Codable, CaseIterable, Sendable {
    case off
    case exactPrefix = "exact-prefix"
}
public enum QwenPrefixInvalidation: String, Codable, Sendable {
    case cold, disabled, explicitClear, modeChanged, reset, externalForward, externalProfiling
    case incompatibleIdentity, prefixMismatch, stateAhead, emptySuffix, invalidBoundary
    case inferenceError, cancelled, outputIncomplete, outputValidationFailed, shutdown, rejectedRequest
}
public struct QwenPrefixCacheSnapshot: Codable, Sendable {
    public let mode: QwenPrefixCacheMode
    public let valid: Bool
    public let consumedTokens: Int
    public let reusedPrefixTokens: Int
    public let processedPromptTokens: Int
    public let hits: UInt64
    public let misses: UInt64
    public let invalidations: UInt64
    public let reason: QwenPrefixInvalidation?
    public let lastInvalidationReason: QwenPrefixInvalidation?
    public let fullReplays: UInt64
    public let maximumTokens: Int
    public let pendingOutputToken: Bool
    public let comparedPrefixTokens: Int
    public let tokenStorageBytes: Int
    public let session: String
}
public enum QwenPrefixStateError: Error, LocalizedError {
    case boundary
    public var errorDescription: String? {"Inference consumed-token boundary is inconsistent; reset and retry."}
}
/// Metadata only. No GPU state copy, rewind or unbounded conversation map.
/// All forwards remain serialized by the runner's caller. The lock permits only
/// concurrent clear/mode changes and redacted snapshots, never concurrent GPU use.
final class QwenPrefixState: @unchecked Sendable {
    private let lock = NSLock()
    private let maximum: Int
    private let session = UUID().uuidString
    private var mode: QwenPrefixCacheMode = .off
    private var activeMode: QwenPrefixCacheMode = .off
    private var tokens: [Int32] = []
    private var identity = ""
    private var configuration: QwenGenerationConfiguration?
    private var pendingOutputToken = false
    private var valid = false
    private var active = false
    private var epoch: UInt64 = 0
    private var activeEpoch: UInt64 = 0
    private var eligible = false
    private var representedCount = 0
    private var reused = 0
    private var processed = 0
    private var hits: UInt64 = 0
    private var misses: UInt64 = 0
    private var fullReplays: UInt64 = 0
    private var invalidations: UInt64 = 0
    private var reason: QwenPrefixInvalidation? = .cold
    private var lastInvalidationReason: QwenPrefixInvalidation?
    private var compared = 0
    init(maximum: Int) {self.maximum = max(0,maximum)}
    private func increment(_ value: inout UInt64) {if value < UInt64.max {value += 1}}
    private func invalidateLocked(_ why: QwenPrefixInvalidation) {
        valid = false;eligible = false;reason = why;lastInvalidationReason = why;increment(&invalidations)
        // Overflow does not allow an old transaction to become valid again.
        if epoch == UInt64.max {active = false;epoch = 0} else {epoch += 1}
        if !active {tokens.removeAll(keepingCapacity:false);representedCount = 0}
    }
    var currentEpoch: UInt64 { lock.withLock { epoch } }
    func clear(_ why: QwenPrefixInvalidation) {lock.withLock {invalidateLocked(why)}}
    func setMode(_ value: QwenPrefixCacheMode) {
        lock.withLock {if mode != value {mode = value;invalidateLocked(.modeChanged)}}
    }
    func begin(
        prompt: [Int32], identity newIdentity: String,
        configuration newConfiguration: QwenGenerationConfiguration? = nil, expectedEpoch: UInt64? = nil
    ) throws -> Int {
        try lock.withLock {
            guard !active,!prompt.isEmpty,prompt.count < maximum else {invalidateLocked(.invalidBoundary);throw QwenPrefixStateError.boundary}
            let wasValid = valid
            let requestEpoch = expectedEpoch ?? epoch
            // Output ceilings do not change any consumed token's forward or the
            // sampler inputs. Include every actual sampling setting in identity.
            var sampling = newConfiguration
            sampling?.maxNewTokens = 1
            reused = 0;processed = 0;compared = 0;eligible = false;pendingOutputToken = false
            if mode == .exactPrefix {
                if valid && requestEpoch == epoch && identity == newIdentity && configuration == sampling {
                    while compared < min(tokens.count,prompt.count),tokens[compared] == prompt[compared] {compared += 1}
                    if compared == tokens.count && tokens.count < prompt.count {
                        reused = tokens.count;increment(&hits);reason = nil
                    } else {
                        reason = compared == prompt.count ? (tokens.count == prompt.count ? .emptySuffix:.stateAhead):.prefixMismatch
                        increment(&misses)
                    }
                } else {
                    if requestEpoch == epoch {
                        reason = valid ? .incompatibleIdentity : (reason ?? .cold)
                    }
                    increment(&misses)
                }
            } else {reason = .disabled}
            if reused == 0 {
                increment(&fullReplays)
                if wasValid {
                    lastInvalidationReason = reason
                    increment(&invalidations)
                }
                tokens.removeAll(keepingCapacity: mode == .exactPrefix)
            }
            valid = false;active = true;activeMode = mode;activeEpoch = requestEpoch;identity = newIdentity;configuration = sampling
            representedCount = reused
            return reused
        }
    }
    func consume(_ token: Int32,position: Int,prompt: Bool) throws {
        try lock.withLock {
            guard active,position == representedCount,representedCount < maximum else {invalidateLocked(.invalidBoundary);throw QwenPrefixStateError.boundary}
            representedCount += 1
            if prompt {processed += 1}
            if activeMode == .exactPrefix {tokens.append(token)}
        }
    }
    func finish(promptCount: Int,result: QwenGenerationResult) throws {
        try lock.withLock {
            guard active else {throw QwenPrefixStateError.boundary}
            let consumedOutput: Int
            if case .endOfSequence = result.finishReason {consumedOutput = result.tokens.count}
            else {
                pendingOutputToken = !result.tokens.isEmpty
                consumedOutput = max(0, result.tokens.count - 1)
            }
            let count = promptCount.addingReportingOverflow(consumedOutput)
            guard !count.overflow,count.partialValue == representedCount else {invalidateLocked(.invalidBoundary);throw QwenPrefixStateError.boundary}
            active = false
            if case .endOfSequence = result.finishReason,activeMode == .exactPrefix,activeEpoch == epoch {eligible = true}
            else {
                eligible = false
                tokens.removeAll(keepingCapacity: false)
                if activeEpoch == epoch {
                    reason = activeMode == .off ? .disabled : .outputIncomplete
                    if activeMode == .exactPrefix {
                        lastInvalidationReason = .outputIncomplete
                        increment(&invalidations)
                    }
                }
            }
        }
    }
    /// Called only after protocol/native-output validation and complete flush.
    func commit() {
        lock.withLock {
            guard !active,eligible,activeEpoch == epoch,mode == .exactPrefix else {return}
            valid = true;eligible = false;reason = nil
        }
    }
    func abort(_ why: QwenPrefixInvalidation) {
        lock.withLock {active = false;invalidateLocked(why);tokens.removeAll(keepingCapacity:false);representedCount = 0}
    }
    var snapshot: QwenPrefixCacheSnapshot {
        lock.withLock {
            QwenPrefixCacheSnapshot(
                mode: mode, valid: valid, consumedTokens: valid ? tokens.count : 0,
                reusedPrefixTokens: reused, processedPromptTokens: processed, hits: hits,
                misses: misses, invalidations: invalidations, reason: reason,
                lastInvalidationReason: lastInvalidationReason, fullReplays: fullReplays,
                maximumTokens: maximum, pendingOutputToken: pendingOutputToken,
                comparedPrefixTokens: compared, tokenStorageBytes: tokens.capacity * MemoryLayout<Int32>.size,
                session: session
            )
        }
    }
}
