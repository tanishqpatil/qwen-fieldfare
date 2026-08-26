import Testing
@testable import QwenFieldfareCore

@Suite
struct QwenRuntimeTelemetryTests {
    @Test
    func nearestRankPercentilesAndMean() {
        let summary = QwenLatencySummary.summarize([5, 1, 4, 2, 3, 100])

        #expect(summary.sampleCount == 6)
        #expect(summary.minimumMs == 1)
        #expect(summary.maximumMs == 100)
        #expect(summary.meanMs == 19.166666666666668)
        #expect(summary.p50Ms == 3)
        #expect(summary.p95Ms == 100)
    }

    @Test
    func boundedAccumulatorDropsOldestSamplesAndResets() {
        var accumulator = QwenBoundedLatencyAccumulator(capacity: 3)
        accumulator.record(milliseconds: 1)
        accumulator.record(milliseconds: 2)
        accumulator.record(milliseconds: 3)
        accumulator.record(milliseconds: 4)

        #expect(accumulator.summary.sampleCount == 3)
        #expect(accumulator.summary.minimumMs == 2)
        #expect(accumulator.summary.maximumMs == 4)

        accumulator.reset()
        #expect(accumulator.summary.sampleCount == 0)
        #expect(accumulator.summary.p95Ms == nil)
    }

    @Test
    func cacheDerivedMetricsUseDemandRequestsOnly() {
        let metrics = QwenExpertCacheMetrics(
            capacity: 1_024,
            occupancy: 900,
            probationaryOccupancy: 200,
            protectedOccupancy: 700,
            hits: 75,
            misses: 25,
            evictions: 4,
            fetchedBytes: 25 * Q4BoundedExpertCache.expertByteLength,
            prefetchRequests: 12,
            usefulPrefetchHits: 8,
            wastedPrefetches: 2
        )

        #expect(metrics.requests == 100)
        #expect(metrics.hitRate == 0.75)
    }
}
