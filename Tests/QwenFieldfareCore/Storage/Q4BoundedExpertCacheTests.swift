import Metal
import Testing
@testable import QwenFieldfareCore

@Suite
struct Q4BoundedExpertCacheTests {
    @Test
    func layerPartitionsPreventCrossLayerEviction() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try Q4BoundedExpertCache(device: device, capacity: 16, layerCount: 2)
        let first = (0..<8).map(UInt32.init)
        let second = (16..<24).map(UInt32.init)
        let replacement = (32..<40).map(UInt32.init)
        let fill: (Int, Int, UnsafeMutableRawPointer) -> Void = { _, expert, destination in
            destination.storeBytes(of: UInt8(truncatingIfNeeded: expert), as: UInt8.self)
        }

        let layer0Slots = cache.acquireSlots(layer: 0, expertIndices: first, fetchBlock: fill)
        let layer1Slots = cache.acquireSlots(layer: 1, expertIndices: second, fetchBlock: fill)
        #expect(Set(layer0Slots).isDisjoint(with: Set(layer1Slots)))

        _ = cache.acquireSlots(layer: 0, expertIndices: replacement, fetchBlock: fill)
        cache.resetMetrics()
        let layer1Again = cache.acquireSlots(layer: 1, expertIndices: second, fetchBlock: fill)
        #expect(layer1Again == layer1Slots)
        #expect(cache.hits == 8)
        #expect(cache.misses == 0)
    }

    @Test
    func parallelMissFetchWritesAssignedSlots() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try Q4BoundedExpertCache(device: device, capacity: 8, layerCount: 1)
        let experts = (40..<48).map(UInt32.init)
        let slots = cache.acquireSlotsParallel(layer: 0, expertIndices: experts) {
            $2.storeBytes(of: UInt8(truncatingIfNeeded: $1), as: UInt8.self)
        }

        for (index, slot) in slots.enumerated() {
            let pointer = cache.cacheBuffer.contents().advanced(
                by: Int(slot) * Q4BoundedExpertCache.expertByteLength)
            #expect(pointer.load(as: UInt8.self) == UInt8(truncatingIfNeeded: experts[index]))
        }
        #expect(cache.misses == 8)
    }

    @Test
    func parallelPlanningReportsOnlyMissPositionsBeforeFetch() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try Q4BoundedExpertCache(device: device, capacity: 8, layerCount: 1)
        let initial = (0..<8).map(UInt32.init)
        let replacement = Array(initial.prefix(7)) + [UInt32(8)]
        let fill: @Sendable (Int, Int, UnsafeMutableRawPointer) -> Void = {
            _, expert, destination in
            destination.storeBytes(of: UInt8(truncatingIfNeeded: expert), as: UInt8.self)
        }
        _ = cache.acquireSlotsParallel(layer: 0, expertIndices: initial, fetchBlock: fill)
        cache.resetMetrics()

        var plannedSlots: [UInt32] = []
        var missPositions: [UInt32] = []
        let result = cache.acquireSlotsParallel(
            layer: 0,
            expertIndices: replacement,
            afterPlanning: {
                plannedSlots = $0
                missPositions = $1
            },
            fetchBlock: fill)

        #expect(plannedSlots == result)
        #expect(missPositions == [7])
        #expect(cache.hits == 7)
        #expect(cache.misses == 1)
    }
}
