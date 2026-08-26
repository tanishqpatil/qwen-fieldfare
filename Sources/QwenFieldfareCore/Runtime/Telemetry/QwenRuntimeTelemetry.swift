import Foundation

public struct QwenLatencySummary: Sendable, Equatable {
    public let sampleCount: Int
    public let minimumMs: Double?
    public let maximumMs: Double?
    public let meanMs: Double?
    public let p50Ms: Double?
    public let p95Ms: Double?

    public static func summarize(_ samples: [Double]) -> Self {
        guard !samples.isEmpty else {
            return Self(sampleCount: 0, minimumMs: nil, maximumMs: nil,
                        meanMs: nil, p50Ms: nil, p95Ms: nil)
        }
        let sorted = samples.sorted()
        func nearestRank(_ percentile: Double) -> Double {
            let rank = max(1, Int(ceil(percentile * Double(sorted.count))))
            return sorted[min(sorted.count - 1, rank - 1)]
        }
        return Self(
            sampleCount: sorted.count,
            minimumMs: sorted.first,
            maximumMs: sorted.last,
            meanMs: sorted.reduce(0, +) / Double(sorted.count),
            p50Ms: nearestRank(0.50),
            p95Ms: nearestRank(0.95)
        )
    }
}

public struct QwenBoundedLatencyAccumulator: Sendable {
    public let capacity: Int
    private var samples: [Double] = []

    public init(capacity: Int = 4_096) {
        self.capacity = max(1, capacity)
        samples.reserveCapacity(min(self.capacity, 4_096))
    }

    public mutating func record(milliseconds: Double) {
        guard milliseconds.isFinite, milliseconds >= 0 else { return }
        if samples.count == capacity {
            samples.removeFirst()
        }
        samples.append(milliseconds)
    }

    public mutating func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    public var summary: QwenLatencySummary {
        QwenLatencySummary.summarize(samples)
    }
}

public struct QwenExpertCacheMetrics: Sendable, Equatable {
    public let capacity: Int
    public let occupancy: Int
    public let probationaryOccupancy: Int
    public let protectedOccupancy: Int
    public let hits: Int
    public let misses: Int
    public let evictions: Int
    public let fetchedBytes: Int
    public let prefetchRequests: Int
    public let usefulPrefetchHits: Int
    public let wastedPrefetches: Int

    public var requests: Int { hits + misses }
    public var hitRate: Double? {
        requests == 0 ? nil : Double(hits) / Double(requests)
    }
}

public struct QwenMemoryMetrics: Sendable, Equatable {
    public var metalAllocatedBytes: UInt64?
    public var peakMetalAllocatedBytes: UInt64?
    public var systemWiredBytes: UInt64?
    public var hostUsedBytes: UInt64?
    public var processRSSBytes: UInt64?
    public var physicalFootprintBytes: UInt64?
    public var virtualBytes: UInt64?
    public var swapUsedBytes: UInt64?
    public var swapDeltaBytes: Int64?
    public var pressure: String?

    public init() {}
}

public struct QwenRuntimeMetricsSnapshot: Sendable, Equatable {
    public var mode: String
    public var modelState: String
    public var promptTokenCount: Int
    public var generatedTokenCount: Int
    public var loadDurationMs: Double?
    public var prefillDurationMs: Double?
    public var firstTokenLatencyMs: Double?
    public var latency: QwenLatencySummary
    public var steadyTokensPerSecond: Double?
    public var runAverageTokensPerSecond: Double?
    public var cache: QwenExpertCacheMetrics?
    public var missesPerGeneratedToken: Double?
    public var fetchedBytesPerGeneratedToken: Double?
    public var stagedBytesPerGeneratedToken: Double?
    public var memory: QwenMemoryMetrics
    public var classification: String?
    public var finishReason: String?
    public var error: String?
    public var wasCancelled: Bool

    public init(mode: String, modelState: String) {
        self.mode = mode
        self.modelState = modelState
        self.promptTokenCount = 0
        self.generatedTokenCount = 0
        self.latency = .summarize([])
        self.memory = QwenMemoryMetrics()
        self.wasCancelled = false
    }
}
