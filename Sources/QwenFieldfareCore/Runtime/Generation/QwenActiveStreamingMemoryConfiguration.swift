import Foundation

/// User-selectable active-expert Metal allocation targets. These are binary
/// GiB, matching Metal allocation and the diagnostics shown by the app.
public enum QwenActiveStreamingMemoryBudget: Int, CaseIterable, Sendable {
    case threeGiB = 3
    case fourGiB = 4
    case fiveGiB = 5

    public var byteCount: UInt64 {
        UInt64(rawValue) * QwenActiveStreamingMemoryConfiguration.gibibyte
    }

    public var displayName: String { "\(rawValue) GiB" }
}

public enum QwenActiveStreamingMemoryConfigurationError: LocalizedError, CustomStringConvertible, Equatable {
    case invalidContextLength(Int)
    case arithmeticOverflow
    case insufficientBudget(
        budget: QwenActiveStreamingMemoryBudget,
        contextLength: Int,
        availableSlots: Int
    )

    public var description: String {
        switch self {
        case .invalidContextLength(let value):
            return "Context length must be positive (got \(value))."
        case .arithmeticOverflow:
            return "Memory-plan arithmetic overflowed."
        case .insufficientBudget(let budget, let contextLength, let availableSlots):
            return "\(budget.displayName) cannot support \(contextLength) tokens: "
                + "only \(availableSlots) expert-cache slots remain; at least "
                + "\(QwenActiveStreamingMemoryConfiguration.minimumExpertCacheSlots) are required."
        }
    }

    public var errorDescription: String? { description }
}

/// A checked, model-specific memory plan for the Qwen 3.6 35B-A3B Q4 active
/// streaming path. It is deliberately not a process-RAM cap: mapped checkpoint
/// pages and OS file-cache residency are outside Metal's allocation accounting.
public struct QwenActiveStreamingMemoryConfiguration: Equatable, Sendable {
    public static let gibibyte: UInt64 = 1_073_741_824
    public static let mebibyte: UInt64 = 1_048_576
    public static let kibibyte: UInt64 = 1_024

    public static let layerCount = 40
    public static let fullAttentionLayerCount = 10
    public static let linearAttentionLayerCount = 30
    public static let expertsPerLayer = 8
    public static let minimumExpertCacheSlots = layerCount * expertsPerLayer

    public let budget: QwenActiveStreamingMemoryBudget
    public let contextLength: Int
    public let expertCacheSlots: Int
    public let targetBytes: UInt64
    public let fixedResidentBytes: UInt64
    public let kvStateBytes: UInt64
    public let contextScratchBytes: UInt64
    public let expertCacheBytes: UInt64
    public let estimatedTotalBytes: UInt64

    /// This is an allocation plan, not a promise about physical process
    /// footprint. Keep the safety reserve explicit in diagnostics and tests.
    public static let denseResidentBytes: UInt64 =
        UInt64(layerCount * DenseLayerArena.totalByteLength)
    public static let recurrentStateBytes: UInt64 =
        UInt64(linearAttentionLayerCount * ((8_192 * 4 * 4) + (32 * 128 * 128 * 4)))
    public static let decodedConstantsBytes: UInt64 =
        UInt64(
            2_048 * 4 // final RMS norm
                + layerCount * 2 * 2_048 * 4 // layer input/post norms
                + linearAttentionLayerCount * ((8_192 * 4 * 4) + (32 * 4) + (32 * 4) + (128 * 4))
                + fullAttentionLayerCount * 2 * 256 * 4 // Q/K norms
        )
    public static let fixedScratchBytes: UInt64 = 1_341_348

    /// Zero-copy LM-head mapping still has Metal accounting and allocator
    /// overhead. This conservative reserve also leaves headroom below the user
    /// target for page alignment and small command resources. It was calibrated
    /// to retain the prior 5 GiB / 8K 2,183-slot profile.
    public static let mappedModelAndSafetyReserveBytes: UInt64 = 366_570_000

    public static let perContextKVBytes: UInt64 =
        UInt64(fullAttentionLayerCount * 2 * 2 * 256 * MemoryLayout<Float16>.size)
    public static let perContextScratchBytes: UInt64 = 16 * 4
    public static let perExpertCacheSlotBytes: UInt64 =
        UInt64(Q4BoundedExpertCache.expertByteLength)

    public static func make(
        budget: QwenActiveStreamingMemoryBudget = .fiveGiB,
        contextLength: Int
    ) throws -> QwenActiveStreamingMemoryConfiguration {
        guard contextLength > 0 else {
            throw QwenActiveStreamingMemoryConfigurationError.invalidContextLength(contextLength)
        }

        let context = UInt64(contextLength)
        let kvBytes = try multiply(perContextKVBytes, context)
        let contextScratchBytes = try multiply(perContextScratchBytes, context)
        let fixedResidentBytes = try add([
            denseResidentBytes,
            recurrentStateBytes,
            decodedConstantsBytes,
            fixedScratchBytes,
            mappedModelAndSafetyReserveBytes,
        ])
        let nonCacheBytes = try add([fixedResidentBytes, kvBytes, contextScratchBytes])
        let targetBytes = budget.byteCount
        let remainingBytes = targetBytes > nonCacheBytes ? targetBytes - nonCacheBytes : 0
        let availableSlots = Int(remainingBytes / perExpertCacheSlotBytes)
        guard availableSlots >= minimumExpertCacheSlots else {
            throw QwenActiveStreamingMemoryConfigurationError.insufficientBudget(
                budget: budget,
                contextLength: contextLength,
                availableSlots: availableSlots
            )
        }
        let cacheBytes = try multiply(UInt64(availableSlots), perExpertCacheSlotBytes)
        let estimatedTotalBytes = try add([nonCacheBytes, cacheBytes])
        return QwenActiveStreamingMemoryConfiguration(
            budget: budget,
            contextLength: contextLength,
            expertCacheSlots: availableSlots,
            targetBytes: targetBytes,
            fixedResidentBytes: fixedResidentBytes,
            kvStateBytes: kvBytes,
            contextScratchBytes: contextScratchBytes,
            expertCacheBytes: cacheBytes,
            estimatedTotalBytes: estimatedTotalBytes
        )
    }

    private static func add(_ values: [UInt64]) throws -> UInt64 {
        try values.reduce(0) { partial, value in
            let result = partial.addingReportingOverflow(value)
            guard !result.overflow else {
                throw QwenActiveStreamingMemoryConfigurationError.arithmeticOverflow
            }
            return result.partialValue
        }
    }

    private static func multiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else {
            throw QwenActiveStreamingMemoryConfigurationError.arithmeticOverflow
        }
        return result.partialValue
    }
}
