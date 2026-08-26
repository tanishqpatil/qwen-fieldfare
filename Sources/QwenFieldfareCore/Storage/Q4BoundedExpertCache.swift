import Foundation
import Metal

public struct Q4ExpertCacheKey: Hashable, Sendable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }
}

public final class Q4BoundedExpertCache: @unchecked Sendable {
    public static let expertByteLength: Int = 1_769_472 // 1.6875 MiB

    public let capacity: Int
    public let probationaryCapacity: Int
    public let protectedCapacity: Int
    public let layerCount: Int
    public let totalByteLength: Int
    public let device: MTLDevice
    public let cacheBuffer: MTLBuffer
    public let recordsAccessTrace: Bool
    private let layerSlotRanges: [Range<Int>]
    private var layerAccessTrace: [[UInt16]]

    // Mapping & LRU State
    private var keyToSlot: [Q4ExpertCacheKey: Int] = [:]
    private var slotToKey: [Int: Q4ExpertCacheKey] = [:]
    private var accessTime: [Int: UInt64] = [:]
    private var currentTick: UInt64 = 0

    // Segmented LRU (SLRU) Partition
    private var probationarySlots: Set<Int> = []
    private var protectedSlots: Set<Int> = []
    private var hitCounts: [Q4ExpertCacheKey: Int] = [:]
    private var prefetchedKeys: Set<Q4ExpertCacheKey> = []

    // Telemetry Metrics
    public private(set) var totalRequests: Int = 0
    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0
    public private(set) var evictions: Int = 0
    public private(set) var bytesFetched: Int = 0
    public private(set) var prefetchesIssued: Int = 0
    public private(set) var usefulPrefetches: Int = 0
    public private(set) var wastedPrefetches: Int = 0
    public private(set) var layerHits: [Int: Int] = [:]
    public private(set) var layerRequests: [Int: Int] = [:]

    public init(
        device: MTLDevice,
        capacity: Int = 1024,
        probationaryRatio: Double = 0.25,
        layerCount requestedLayerCount: Int = 40,
        recordsAccessTrace: Bool = false
    ) throws {
        self.device = device
        self.capacity = capacity
        self.recordsAccessTrace = recordsAccessTrace
        // Each layer needs at least eight simultaneously locked slots for the
        // router's top-8. Small test caches retain the legacy global behavior.
        self.layerCount = capacity >= max(1, requestedLayerCount) * 8
            ? max(1, requestedLayerCount)
            : 1
        if self.layerCount == 40 {
            // Qwen's early and final routers have materially higher expert
            // entropy than the middle stack. These weights were derived from
            // measured per-layer demand misses; every layer retains at least
            // the eight slots required by top-8 routing, and extra capacity is
            // distributed proportionally.
            let weights = [
                86, 82, 78, 72, 64, 58, 54, 48, 35, 42,
                58, 59, 45, 48, 38, 43, 45, 41, 42, 38,
                34, 37, 53, 55, 43, 45, 38, 41, 46, 41,
                43, 40, 48, 54, 55, 42, 45, 47, 51, 66
            ]
            let basePerLayer = 8
            let remaining = capacity - self.layerCount * basePerLayer
            let excessWeights = weights.map { $0 - basePerLayer }
            let excessTotal = excessWeights.reduce(0, +)
            var counts = excessWeights.map {
                basePerLayer + (remaining * $0) / excessTotal
            }
            var slotsLeft = capacity - counts.reduce(0, +)
            let remainderOrder = excessWeights.indices.sorted {
                (remaining * excessWeights[$0]) % excessTotal
                    > (remaining * excessWeights[$1]) % excessTotal
            }
            var remainderIndex = 0
            while slotsLeft > 0 {
                counts[remainderOrder[remainderIndex]] += 1
                slotsLeft -= 1
                remainderIndex += 1
            }
            var start = 0
            var ranges: [Range<Int>] = []
            ranges.reserveCapacity(self.layerCount)
            for count in counts {
                ranges.append(start..<(start + count))
                start += count
            }
            self.layerSlotRanges = ranges
        } else if self.layerCount > 1 {
            var start = 0
            var ranges: [Range<Int>] = []
            ranges.reserveCapacity(self.layerCount)
            for layer in 0..<self.layerCount {
                let count = capacity / self.layerCount
                    + (layer < capacity % self.layerCount ? 1 : 0)
                ranges.append(start..<(start + count))
                start += count
            }
            self.layerSlotRanges = ranges
        } else {
            self.layerSlotRanges = [0..<capacity]
        }
        self.layerAccessTrace = Array(repeating: [], count: self.layerCount)
        let probCap = max(16, Int(Double(capacity) * probationaryRatio))
        self.probationaryCapacity = probCap
        self.protectedCapacity = capacity - probCap
        self.totalByteLength = capacity * Self.expertByteLength
        guard let buf = device.makeBuffer(length: totalByteLength, options: .storageModeShared) else {
            throw NSError(domain: "Q4BoundedExpertCache", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate \(Double(totalByteLength) / 1048576.0) MB cache buffer"])
        }
        self.cacheBuffer = buf
    }

    private func slotRange(for layer: Int) -> Range<Int> {
        let normalizedLayer = min(max(0, layer), layerCount - 1)
        return layerSlotRanges[normalizedLayer]
    }

    public func clear() {
        keyToSlot.removeAll()
        slotToKey.removeAll()
        accessTime.removeAll()
        probationarySlots.removeAll()
        protectedSlots.removeAll()
        hitCounts.removeAll()
        prefetchedKeys.removeAll()
        currentTick = 0
        if recordsAccessTrace {
            for layer in layerAccessTrace.indices {
                layerAccessTrace[layer].removeAll(keepingCapacity: true)
            }
        }
    }

    public func resetMetrics() {
        totalRequests = 0
        hits = 0
        misses = 0
        evictions = 0
        bytesFetched = 0
        prefetchesIssued = 0
        usefulPrefetches = 0
        wastedPrefetches = 0
        layerHits.removeAll()
        layerRequests.removeAll()
    }

    /// Run-level counters count demand lookups only. Prefetch requests are
    /// tracked separately. `bytesFetched` is the logical number of expert-block
    /// bytes copied from the storage-backed mapping into the cache arena; those
    /// bytes are also the staged bytes consumed by the cached MoE kernels.
    public func metricsSnapshot() -> QwenExpertCacheMetrics {
        QwenExpertCacheMetrics(
            capacity: capacity,
            occupancy: keyToSlot.count,
            probationaryOccupancy: probationarySlots.count,
            protectedOccupancy: protectedSlots.count,
            hits: hits,
            misses: misses,
            evictions: evictions,
            fetchedBytes: bytesFetched,
            prefetchRequests: prefetchesIssued,
            usefulPrefetchHits: usefulPrefetches,
            wastedPrefetches: wastedPrefetches
        )
    }

    public func acquireSlots(
        layer: Int,
        expertIndices: [UInt32],
        fetchBlock: (Int, Int, UnsafeMutableRawPointer) -> Void
    ) -> [UInt32] {
        var activeSlots: [UInt32] = []
        var lockedThisLayer = Set<Int>()
        let layerSlots = slotRange(for: layer)
        let layerProtectedCapacity = max(
            8,
            Int(Double(layerSlots.count) * Double(protectedCapacity) / Double(capacity)))

        currentTick += 1
        let tick = currentTick

        for expIdx in expertIndices.prefix(8) {
            let key = Q4ExpertCacheKey(layer: layer, expert: Int(expIdx))
            if recordsAccessTrace {
                layerAccessTrace[min(max(0, layer), layerCount - 1)].append(UInt16(expIdx))
            }
            totalRequests += 1
            layerRequests[layer, default: 0] += 1
            hitCounts[key, default: 0] += 1

            if prefetchedKeys.contains(key) {
                usefulPrefetches += 1
                prefetchedKeys.remove(key)
            }

            if let slot = keyToSlot[key] {
                // Cache HIT
                hits += 1
                layerHits[layer, default: 0] += 1
                accessTime[slot] = tick
                lockedThisLayer.insert(slot)
                activeSlots.append(UInt32(slot))

                // Promote from probationary to protected on recurrent hit
                if probationarySlots.contains(slot) {
                    probationarySlots.remove(slot)
                    protectedSlots.insert(slot)

                    // Demote LRU protected slot if protected partition overflows
                    let protectedInLayer = layerSlots.reduce(into: 0) {
                        if protectedSlots.contains($1) { $0 += 1 }
                    }
                    if protectedInLayer > layerProtectedCapacity {
                        var minProtTick = UInt64.max
                        var demoteCandidate = -1
                        for ps in layerSlots {
                            if protectedSlots.contains(ps) && !lockedThisLayer.contains(ps) {
                                let t = accessTime[ps] ?? 0
                                if t < minProtTick {
                                    minProtTick = t
                                    demoteCandidate = ps
                                }
                            }
                        }
                        if demoteCandidate != -1 {
                            protectedSlots.remove(demoteCandidate)
                            probationarySlots.insert(demoteCandidate)
                        }
                    }
                }
            } else {
                // Cache MISS
                misses += 1
                var victimSlot = -1

                // 1. Unallocated slot
                for s in layerSlots {
                    if slotToKey[s] == nil && !lockedThisLayer.contains(s) {
                        victimSlot = s
                        break
                    }
                }

                // 2. LRU from Probationary segment
                if victimSlot == -1 {
                    var minProbTick = UInt64.max
                    for s in layerSlots {
                        if probationarySlots.contains(s) && !lockedThisLayer.contains(s) {
                            let t = accessTime[s] ?? 0
                            if t < minProbTick {
                                minProbTick = t
                                victimSlot = s
                            }
                        }
                    }
                }

                // 3. Fallback: LRU from Protected segment
                if victimSlot == -1 {
                    var minProtTick = UInt64.max
                    for s in layerSlots {
                        if protectedSlots.contains(s) && !lockedThisLayer.contains(s) {
                            let t = accessTime[s] ?? 0
                            if t < minProtTick {
                                minProtTick = t
                                victimSlot = s
                            }
                        }
                    }
                }

                if victimSlot == -1 {
                    for s in layerSlots {
                        if !lockedThisLayer.contains(s) {
                            victimSlot = s
                            break
                        }
                    }
                }
                if victimSlot == -1 { victimSlot = 0 }

                // Evict old key
                if let oldKey = slotToKey[victimSlot] {
                    if prefetchedKeys.contains(oldKey) {
                        wastedPrefetches += 1
                        prefetchedKeys.remove(oldKey)
                    }
                    keyToSlot.removeValue(forKey: oldKey)
                    probationarySlots.remove(victimSlot)
                    protectedSlots.remove(victimSlot)
                    evictions += 1
                }

                let destPtr = cacheBuffer.contents().advanced(by: victimSlot * Self.expertByteLength)
                fetchBlock(layer, Int(expIdx), destPtr)
                bytesFetched += Self.expertByteLength

                keyToSlot[key] = victimSlot
                slotToKey[victimSlot] = key
                accessTime[victimSlot] = tick
                probationarySlots.insert(victimSlot)
                lockedThisLayer.insert(victimSlot)
                activeSlots.append(UInt32(victimSlot))
            }
        }

        return activeSlots
    }

    /// Plans cache placement serially, then services independent misses in
    /// parallel. Slot ownership is fixed before the reads start, so no two
    /// workers can write the same destination.
    public func acquireSlotsParallel(
        layer: Int,
        expertIndices: [UInt32],
        afterPlanning: (([UInt32], [UInt32]) -> Void)? = nil,
        fetchBlock: @Sendable (Int, Int, UnsafeMutableRawPointer) -> Void
    ) -> [UInt32] {
        struct PendingFetch {
            let layer: Int
            let expert: Int
            let destination: UnsafeMutableRawPointer
        }

        var pending: [PendingFetch] = []
        let slots = acquireSlots(layer: layer, expertIndices: expertIndices) {
            pending.append(PendingFetch(layer: $0, expert: $1, destination: $2))
        }
        let missingExperts = Set(pending.map(\.expert))
        let missPositions = expertIndices.prefix(8).enumerated().compactMap {
            missingExperts.contains(Int($0.element)) ? UInt32($0.offset) : nil
        }
        // Slot ownership is final at this point, but miss destinations have not
        // been overwritten yet. The runtime can safely launch work for cache
        // hits while the independent miss reads proceed below.
        afterPlanning?(slots, missPositions)
        nonisolated(unsafe) let jobs = pending
        if jobs.count == 1 {
            let job = jobs[0]
            fetchBlock(job.layer, job.expert, job.destination)
        } else if jobs.count > 1 {
            DispatchQueue.concurrentPerform(iterations: jobs.count) { index in
                let job = jobs[index]
                fetchBlock(job.layer, job.expert, job.destination)
            }
        }
        return slots
    }

    public func speculativePrefetch(
        layer: Int,
        candidates: [UInt32],
        fetchBlock: (Int, Int, UnsafeMutableRawPointer) -> Void
    ) {
        let layerSlots = slotRange(for: layer)
        guard layerSlots.contains(where: { slotToKey[$0] == nil }) else { return }
        let tick = currentTick

        for cand in candidates {
            let key = Q4ExpertCacheKey(layer: layer, expert: Int(cand))
            if keyToSlot[key] != nil { continue } // Already cached

            var freeSlot = -1
            for s in layerSlots {
                if slotToKey[s] == nil {
                    freeSlot = s
                    break
                }
            }
            guard freeSlot != -1 else { break }

            let destPtr = cacheBuffer.contents().advanced(by: freeSlot * Self.expertByteLength)
            fetchBlock(layer, Int(cand), destPtr)
            bytesFetched += Self.expertByteLength

            keyToSlot[key] = freeSlot
            slotToKey[freeSlot] = key
            accessTime[freeSlot] = tick
            probationarySlots.insert(freeSlot)
            prefetchedKeys.insert(key)
            prefetchesIssued += 1
        }
    }

    public var hitRate: Double {
        guard totalRequests > 0 else { return 0.0 }
        return Double(hits) / Double(totalRequests)
    }

    public func accessTraceSnapshot() -> [[UInt16]] {
        layerAccessTrace
    }

    public var layerCapacities: [Int] {
        layerSlotRanges.map(\.count)
    }

    public var prefetchHitRate: Double {
        guard prefetchesIssued > 0 else { return 0.0 }
        return Double(usefulPrefetches) / Double(prefetchesIssued)
    }
}
