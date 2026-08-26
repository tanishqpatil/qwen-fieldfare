import Foundation

public struct ChatMessageMetrics: Sendable, Equatable {
    public var steadyTokensPerSecond: Double?
    public var runAverageTokensPerSecond: Double?
    public var firstTokenLatencyMs: Double?
    public var generatedTokenCount: Int?
    public var prefillDurationMs: Double?
    public var cacheHitRate: Double?
    public var finishReason: String?

    public init(
        steadyTokensPerSecond: Double? = nil,
        runAverageTokensPerSecond: Double? = nil,
        firstTokenLatencyMs: Double? = nil,
        generatedTokenCount: Int? = nil,
        prefillDurationMs: Double? = nil,
        cacheHitRate: Double? = nil,
        finishReason: String? = nil
    ) {
        self.steadyTokensPerSecond = steadyTokensPerSecond
        self.runAverageTokensPerSecond = runAverageTokensPerSecond
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.generatedTokenCount = generatedTokenCount
        self.prefillDurationMs = prefillDurationMs
        self.cacheHitRate = cacheHitRate
        self.finishReason = finishReason
    }
}

public struct ChatMessage: Identifiable, Sendable, Equatable {
    public enum Role: String, Sendable, Equatable {
        case user
        case assistant
        case system
    }

    public let id: UUID
    public let role: Role
    public var content: String
    public let timestamp: Date
    public var isStreaming: Bool
    public var metrics: ChatMessageMetrics?
    public var errorMessage: String?

    public init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        timestamp: Date = Date(),
        isStreaming: Bool = false,
        metrics: ChatMessageMetrics? = nil,
        errorMessage: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.isStreaming = isStreaming
        self.metrics = metrics
        self.errorMessage = errorMessage
    }
}
