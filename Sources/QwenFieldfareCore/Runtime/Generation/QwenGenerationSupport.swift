import Foundation

public enum QwenGenerationFinishReason: Sendable, Equatable {
    case endOfSequence(tokenID: Int32)
    case maxNewTokens
    case contextLength

    public var label: String {
        switch self {
        case .endOfSequence: return "End of turn"
        case .maxNewTokens: return "Output limit"
        case .contextLength: return "Context full"
        }
    }
}

public struct QwenGenerationResult: Sendable, Equatable {
    public let tokens: [Int32]
    public let finishReason: QwenGenerationFinishReason

    public init(tokens: [Int32], finishReason: QwenGenerationFinishReason) {
        self.tokens = tokens
        self.finishReason = finishReason
    }
}

public enum QwenGenerationConfigurationError: Error, LocalizedError, Equatable {
    case invalidMaxNewTokens(Int)
    case invalidTemperature(Float)
    case invalidTopK(Int)
    case invalidTopP(Float)
    case invalidPresencePenalty(Float)
    case invalidRepetitionPenalty(Float)
    case promptExceedsContext(promptTokens: Int, contextLength: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidMaxNewTokens(let value):
            return "Qwen max-new-tokens must be positive; received \(value)."
        case .invalidTemperature(let value):
            return "Qwen temperature must be finite and nonnegative; received \(value)."
        case .invalidTopK(let value):
            return "Qwen Top-K must be between 1 and 256; received \(value)."
        case .invalidTopP(let value):
            return "Qwen Top-P must be finite, greater than zero, and at most one; received \(value)."
        case .invalidPresencePenalty(let value):
            return "Qwen presence penalty must be finite and between -2 and 2; received \(value)."
        case .invalidRepetitionPenalty(let value):
            return "Qwen repetition penalty must be finite and at least one; received \(value)."
        case .promptExceedsContext(let promptTokens, let contextLength):
            return "The Qwen prompt uses \(promptTokens) tokens, which leaves no room in the \(contextLength)-token runtime context."
        }
    }
}

/// Generation settings for the Qwen runtime. Temperature/Top-K/Top-P mirror
/// the checkpoint; presence penalty mirrors the validated Ollama renderer.
public struct QwenGenerationConfiguration: Sendable, Equatable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var presencePenalty: Float
    public var repetitionPenalty: Float
    public var seed: UInt64?

    public init(
        maxNewTokens: Int = 4_096,
        temperature: Float = 1.0,
        topK: Int = 20,
        topP: Float = 0.95,
        presencePenalty: Float = 1.5,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    public static func greedy(maxNewTokens: Int, seed: UInt64? = 42) -> Self {
        Self(
            maxNewTokens: maxNewTokens,
            temperature: 0,
            topK: 1,
            topP: 1,
            presencePenalty: 0,
            repetitionPenalty: 1,
            seed: seed
        )
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw QwenGenerationConfigurationError.invalidMaxNewTokens(maxNewTokens)
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw QwenGenerationConfigurationError.invalidTemperature(temperature)
        }
        guard (1...256).contains(topK) else {
            throw QwenGenerationConfigurationError.invalidTopK(topK)
        }
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw QwenGenerationConfigurationError.invalidTopP(topP)
        }
        guard presencePenalty.isFinite, (-2...2).contains(presencePenalty) else {
            throw QwenGenerationConfigurationError.invalidPresencePenalty(presencePenalty)
        }
        guard repetitionPenalty.isFinite, repetitionPenalty >= 1 else {
            throw QwenGenerationConfigurationError.invalidRepetitionPenalty(repetitionPenalty)
        }
    }
}

public enum QwenSpecialTokens {
    /// From the checkpoint's `generation_config.json`.
    public static let stopTokenIDs: Set<Int32> = [248_046, 248_044]
}

public enum QwenChatTemplate {
    public struct Message: Sendable, Equatable {
        public enum Role: String, Sendable {
            case system
            case user
            case assistant
        }

        public let role: Role
        public let content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }
    }

    /// Text-only form of the checkpoint's `chat_template.jinja`. When thinking
    /// is disabled, the empty think block is part of the prompt; omitting it
    /// changes the model distribution and exposes the reasoning trace.
    public static func format(
        messages: [Message],
        appendingUser userMessage: String,
        enableThinking: Bool
    ) -> String {
        var prompt = ""
        for message in messages {
            let content = message.role == .assistant
                ? visibleAssistantContent(message.content)
                : message.content
            prompt += "<|im_start|>\(message.role.rawValue)\n\(content)<|im_end|>\n"
        }
        prompt += "<|im_start|>user\n\(userMessage)<|im_end|>\n<|im_start|>assistant\n"
        prompt += enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n"
        return prompt
    }

    public static func visibleAssistantContent(_ text: String) -> String {
        guard let closeRange = text.range(of: "</think>") else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<think>")
                ? ""
                : text
        }
        return String(text[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Shared generation state machine. Keeping the stop/limit logic independent
/// of Metal makes 1K+ token behavior testable without loading the model.
enum QwenGenerationLoop {
    static func run(
        initialToken: Int32,
        promptTokens: [Int32],
        maxSequenceLength: Int,
        configuration: QwenGenerationConfiguration,
        stopTokenIDs: Set<Int32> = QwenSpecialTokens.stopTokenIDs,
        checkCancellation: () throws -> Void,
        onToken: (Int32) -> Void,
        nextToken: (_ consumedToken: Int32, _ tokenPosition: Int, _ seenTokenIDs: Set<Int32>, _ samplePosition: Int) throws -> Int32
    ) throws -> QwenGenerationResult {
        var generated: [Int32] = []
        generated.reserveCapacity(min(configuration.maxNewTokens, max(0, maxSequenceLength - promptTokens.count)))
        var seenTokenIDs = Set(promptTokens)
        var currentToken = initialToken
        var currentPosition = promptTokens.count

        while true {
            try checkCancellation()
            if stopTokenIDs.contains(currentToken) {
                return QwenGenerationResult(
                    tokens: generated,
                    finishReason: .endOfSequence(tokenID: currentToken)
                )
            }
            if currentPosition >= maxSequenceLength {
                return QwenGenerationResult(tokens: generated, finishReason: .contextLength)
            }

            generated.append(currentToken)
            onToken(currentToken)
            seenTokenIDs.insert(currentToken)

            let nextPosition = currentPosition + 1
            if nextPosition >= maxSequenceLength {
                return QwenGenerationResult(tokens: generated, finishReason: .contextLength)
            }
            if generated.count >= configuration.maxNewTokens {
                return QwenGenerationResult(tokens: generated, finishReason: .maxNewTokens)
            }

            currentToken = try nextToken(
                currentToken,
                currentPosition,
                seenTokenIDs,
                generated.count
            )
            currentPosition = nextPosition
        }
    }
}

/// CPU selector for the Qwen Float32 logits buffer. Greedy selection remains a
/// single vocabulary scan; sampled selection retains only Top-K candidates.
enum QwenLogitSelector {
    private struct Candidate {
        let tokenID: Int32
        let logit: Float
    }

    static func select(
        logits: UnsafePointer<Float>,
        count: Int,
        seenTokenIDs: Set<Int32>,
        configuration: QwenGenerationConfiguration,
        samplePosition: Int
    ) -> Int32 {
        let candidateCount = min(configuration.topK, count)
        var candidates: [Candidate] = []
        candidates.reserveCapacity(candidateCount)
        var minimumIndex = 0

        for index in 0..<count {
            let tokenID = Int32(index)
            var logit = logits[index]
            if seenTokenIDs.contains(tokenID) {
                if configuration.repetitionPenalty != 1 {
                    logit = logit >= 0
                        ? logit / configuration.repetitionPenalty
                        : logit * configuration.repetitionPenalty
                }
                logit -= configuration.presencePenalty
            }

            if candidates.count < candidateCount {
                candidates.append(Candidate(tokenID: tokenID, logit: logit))
                if candidates.count == candidateCount {
                    minimumIndex = indexOfMinimum(candidates)
                }
            } else if logit > candidates[minimumIndex].logit {
                candidates[minimumIndex] = Candidate(tokenID: tokenID, logit: logit)
                minimumIndex = indexOfMinimum(candidates)
            }
        }

        candidates.sort {
            $0.logit == $1.logit ? $0.tokenID < $1.tokenID : $0.logit > $1.logit
        }
        guard configuration.temperature > 0 else {
            return candidates.first?.tokenID ?? 0
        }

        let inverseTemperature = 1 / configuration.temperature
        let maximum = candidates.first?.logit ?? 0
        var weights = candidates.map { exp(Double(($0.logit - maximum) * inverseTemperature)) }
        let fullTotal = weights.reduce(0, +)
        let cutoff = Double(configuration.topP) * fullTotal
        var retainedCount = weights.count
        var cumulative = 0.0
        for index in weights.indices {
            cumulative += weights[index]
            if cumulative >= cutoff {
                retainedCount = index + 1
                break
            }
        }
        if retainedCount < weights.count {
            weights.removeSubrange(retainedCount...)
            candidates.removeSubrange(retainedCount...)
        }

        let retainedTotal = weights.reduce(0, +)
        var draw = uniformUnit(seed: configuration.seed, position: samplePosition) * retainedTotal
        for (candidate, weight) in zip(candidates, weights) {
            draw -= weight
            if draw <= 0 { return candidate.tokenID }
        }
        return candidates.last?.tokenID ?? 0
    }

    static func select(
        logits: [Float],
        seenTokenIDs: Set<Int32> = [],
        configuration: QwenGenerationConfiguration,
        samplePosition: Int = 0
    ) -> Int32 {
        logits.withUnsafeBufferPointer {
            select(
                logits: $0.baseAddress!,
                count: $0.count,
                seenTokenIDs: seenTokenIDs,
                configuration: configuration,
                samplePosition: samplePosition
            )
        }
    }

    private static func indexOfMinimum(_ candidates: [Candidate]) -> Int {
        var result = 0
        for index in candidates.indices.dropFirst() where candidates[index].logit < candidates[result].logit {
            result = index
        }
        return result
    }

    private static func uniformUnit(seed: UInt64?, position: Int) -> Double {
        var value: UInt64
        if let seed {
            value = seed &+ UInt64(bitPattern: Int64(position))
        } else {
            value = UInt64.random(in: UInt64.min...UInt64.max)
        }
        value &+= 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        return Double(value >> 11) / 9_007_199_254_740_992.0
    }
}
