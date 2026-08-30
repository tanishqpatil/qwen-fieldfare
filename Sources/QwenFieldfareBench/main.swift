import Darwin.Mach
import Foundation
import Metal
import QwenFieldfareCore

private let usage = """
Usage:
  QwenFieldfareBench [--model <directory>] [--prompt <text>]
    [--context <tokens>] [--max-new <tokens>] [--cache-slots <count>]
    [--warmup-runs <count>] [--runs <count>]
    [--load-only] [--metrics-only] [--packed-experts] [--analyze-cache]
    [--gpu-profile] [--subcomponent-profile]
"""

private struct Arguments {
    var modelPath = NSHomeDirectory() + "/models/Qwen3.6-35B-A3B-4bit"
    var prompt = "tell me what hasmap is in java with example"
    var contextLength = 8_192
    var maxNewTokens = 512
    var cacheSlots = 1_024
    var warmupRuns = 0
    var measuredRuns = 1
    var loadOnly = false
    var usePackedExperts = false
    var analyzeCache = false
    var metricsOnly = false
    var gpuProfile = false
    var subcomponentProfile = false

    init(_ values: [String]) throws {
        var index = 1
        while index < values.count {
            let option = values[index]
            switch option {
            case "--model", "--prompt", "--context", "--max-new", "--cache-slots",
                 "--warmup-runs", "--runs":
                guard index + 1 < values.count else {
                    throw NSError(domain: "QwenFieldfareBench", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey: "Missing value for \(option)"])
                }
                let value = values[index + 1]
                switch option {
                case "--model": modelPath = value
                case "--prompt": prompt = value
                case "--context":
                    guard let parsed = Int(value), parsed > 0 else {
                        throw NSError(domain: "QwenFieldfareBench", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Invalid --context value: \(value)"])
                    }
                    contextLength = parsed
                case "--max-new":
                    guard let parsed = Int(value), parsed > 0 else {
                        throw NSError(domain: "QwenFieldfareBench", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Invalid --max-new value: \(value)"])
                    }
                    maxNewTokens = parsed
                case "--warmup-runs":
                    guard let parsed = Int(value), parsed >= 0 else {
                        throw NSError(domain: "QwenFieldfareBench", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Invalid --warmup-runs value: \(value)"])
                    }
                    warmupRuns = parsed
                case "--runs":
                    guard let parsed = Int(value), parsed > 0 else {
                        throw NSError(domain: "QwenFieldfareBench", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Invalid --runs value: \(value)"])
                    }
                    measuredRuns = parsed
                default:
                    guard let parsed = Int(value), parsed >= 320 else {
                        throw NSError(domain: "QwenFieldfareBench", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Invalid --cache-slots value: \(value)"])
                    }
                    cacheSlots = parsed
                }
                index += 2
            case "--load-only":
                loadOnly = true
                index += 1
            case "--packed-experts":
                usePackedExperts = true
                index += 1
            case "--analyze-cache":
                analyzeCache = true
                index += 1
            case "--metrics-only":
                metricsOnly = true
                index += 1
            case "--gpu-profile":
                gpuProfile = true
                index += 1
            case "--subcomponent-profile":
                subcomponentProfile = true
                index += 1
            default:
                throw NSError(domain: "QwenFieldfareBench", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Unknown option: \(option)"])
            }
        }
    }
}

private struct TokenSample {
    let tokenCount: Int
    let elapsedSeconds: Double
    let stepMs: Double
    let cacheMissFetchMs: Double
    let commandBufferWaitMs: Double
    let attentionRouterWaitMs: Double
    let moeWaitMs: Double
    let attentionRouterGPUMs: Double
    let sharedHitGPUMs: Double
    let moeTailGPUMs: Double
    let totalGPUMs: Double
    let footprintBytes: UInt64
    let metalBytes: UInt64
    let cacheHits: Int
    let cacheMisses: Int
    let cacheFetchedBytes: Int
}

private final class SampleBox: @unchecked Sendable {
    private let lock = NSLock()
    private var recordingStarted = false
    private var prefillMs = 0.0
    private var samples: [TokenSample] = []
    private var interTokenLatencyMs: [Double] = []
    private var cpuTokenSelectionMs: [Double] = []
    private var firstTokenAt: CFAbsoluteTime?
    private var previousTokenAt: CFAbsoluteTime?
    private var firstTokenCache: QwenExpertCacheMetrics?
    private var tokenCount = 0

    func startDecode(prefillMs: Double) {
        lock.lock()
        self.prefillMs = prefillMs
        recordingStarted = true
        lock.unlock()
    }

    func record(
        profile: QwenQ4ActiveExpertStreamingRunner.StepProfile?,
        device: MTLDevice?,
        cache: QwenExpertCacheMetrics
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard recordingStarted else { return }
        let now = CFAbsoluteTimeGetCurrent()
        tokenCount += 1
        if let previousTokenAt {
            interTokenLatencyMs.append((now - previousTokenAt) * 1_000.0)
        } else {
            firstTokenAt = now
            firstTokenCache = cache
        }
        previousTokenAt = now
        if let profile {
            cpuTokenSelectionMs.append(profile.cpuArgmaxMs)
        }
        guard tokenCount == 1 || tokenCount % 64 == 0 else { return }
        samples.append(makeSample(at: now, profile: profile, device: device, cache: cache))
    }

    func captureFinal(
        profile: QwenQ4ActiveExpertStreamingRunner.StepProfile?,
        device: MTLDevice?,
        cache: QwenExpertCacheMetrics
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard recordingStarted, tokenCount > 0,
              samples.last?.tokenCount != tokenCount,
              let previousTokenAt else { return }
        samples.append(makeSample(
            at: previousTokenAt, profile: profile, device: device, cache: cache))
    }

    private func makeSample(
        at timestamp: CFAbsoluteTime,
        profile: QwenQ4ActiveExpertStreamingRunner.StepProfile?,
        device: MTLDevice?,
        cache: QwenExpertCacheMetrics
    ) -> TokenSample {
        TokenSample(
            tokenCount: tokenCount,
            elapsedSeconds: firstTokenAt.map { timestamp - $0 } ?? 0,
            stepMs: profile?.totalStepMs ?? 0,
            cacheMissFetchMs: profile?.cacheMissFetchMs ?? 0,
            commandBufferWaitMs: profile?.cbCommitWaitMs ?? 0,
            attentionRouterWaitMs: profile?.attentionMs ?? 0,
            moeWaitMs: profile?.stagedMoEMs ?? 0,
            attentionRouterGPUMs: profile?.attentionRouterGPUMs ?? 0,
            sharedHitGPUMs: profile?.sharedHitGPUMs ?? 0,
            moeTailGPUMs: profile?.moeTailGPUMs ?? 0,
            totalGPUMs: profile?.totalGPUMs ?? 0,
            footprintBytes: processFootprintBytes(),
            metalBytes: UInt64(device?.currentAllocatedSize ?? 0),
            cacheHits: cache.hits,
            cacheMisses: cache.misses,
            cacheFetchedBytes: cache.fetchedBytes)
    }

    func snapshot() -> [TokenSample] {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }

    func recordedPrefillMs() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return prefillMs
    }

    func decodeSummary() -> (
        transitionCount: Int,
        elapsedMs: Double,
        latency: QwenLatencySummary,
        cpuTokenSelection: QwenLatencySummary,
        firstTokenCache: QwenExpertCacheMetrics?
    ) {
        lock.lock()
        defer { lock.unlock() }
        let elapsedMs: Double
        if let firstTokenAt, let previousTokenAt {
            elapsedMs = max(0, (previousTokenAt - firstTokenAt) * 1_000.0)
        } else {
            elapsedMs = 0
        }
        return (
            transitionCount: max(0, tokenCount - 1),
            elapsedMs: elapsedMs,
            latency: .summarize(interTokenLatencyMs),
            cpuTokenSelection: .summarize(cpuTokenSelectionMs),
            firstTokenCache: firstTokenCache)
    }
}

private struct RunSummary {
    let label: String
    let tokens: [Int32]
    let finishReason: String
    let prefillMs: Double
    let decodeMs: Double
    let decodeTransitions: Int
    let decodeRate: Double
    let latency: QwenLatencySummary
    let cpuTokenSelection: QwenLatencySummary
    let cache: QwenExpertCacheMetrics
    let decodeMisses: Int
    let decodeFetchedBytes: Int
    let finalFootprintBytes: UInt64
    let finalMetalBytes: UInt64
}

private func processFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

private func gib(_ bytes: UInt64) -> Double {
    Double(bytes) / 1_073_741_824.0
}

private func simulateWindowedFrequency(trace: [UInt16], capacity: Int) -> Int {
    let transientCapacity = min(8, capacity)
    let protectedCapacity = max(0, capacity - transientCapacity)
    var frequencies: [UInt16: Int] = [:]
    var protected: Set<UInt16> = []
    var transient: Set<UInt16> = []
    var lastAccess: [UInt16: Int] = [:]
    var misses = 0
    var tick = 0

    for groupStart in stride(from: 0, to: trace.count, by: 8) {
        let group = trace[groupStart..<min(trace.count, groupStart + 8)]
        var locked: Set<UInt16> = []
        for expert in group {
            tick += 1
            frequencies[expert, default: 0] += 1
            lastAccess[expert] = tick
            if protected.contains(expert) || transient.contains(expert) {
                locked.insert(expert)
                continue
            }
            misses += 1
            if protected.count < protectedCapacity {
                protected.insert(expert)
                locked.insert(expert)
                continue
            }

            let protectedVictim = protected
                .filter { !locked.contains($0) }
                .min {
                    let left = frequencies[$0, default: 0]
                    let right = frequencies[$1, default: 0]
                    return left == right
                        ? lastAccess[$0, default: 0] < lastAccess[$1, default: 0]
                        : left < right
                }
            if let victim = protectedVictim,
               frequencies[expert, default: 0] > frequencies[victim, default: 0] {
                protected.remove(victim)
                transient.remove(expert)
                protected.insert(expert)
            } else {
                if transient.count == transientCapacity,
                   let victim = transient
                    .filter({ !locked.contains($0) })
                    .min(by: { lastAccess[$0, default: 0] < lastAccess[$1, default: 0] }) {
                    transient.remove(victim)
                }
                transient.insert(expert)
            }
            locked.insert(expert)
        }
    }
    return misses
}

private func staticHotSetMisses(trace: [UInt16], capacity: Int) -> Int {
    var frequencies: [UInt16: Int] = [:]
    for expert in trace { frequencies[expert, default: 0] += 1 }
    let reusableHits = frequencies.values.sorted(by: >).prefix(capacity).reduce(0) {
        $0 + max(0, $1 - 1)
    }
    return trace.count - reusableHits
}

private func simulateSegmentedLRU(trace: [UInt16], capacity: Int) -> Int {
    let protectedCapacity = max(8, (capacity * 3) / 4)
    var keyToSlot: [UInt16: Int] = [:]
    var slotToKey: [Int: UInt16] = [:]
    var accessTime: [Int: Int] = [:]
    var probationary: Set<Int> = []
    var protected: Set<Int> = []
    var tick = 0
    var misses = 0

    for groupStart in stride(from: 0, to: trace.count, by: 8) {
        tick += 1
        let group = trace[groupStart..<min(trace.count, groupStart + 8)]
        var locked: Set<Int> = []
        for expert in group {
            if let slot = keyToSlot[expert] {
                accessTime[slot] = tick
                locked.insert(slot)
                if probationary.remove(slot) != nil {
                    protected.insert(slot)
                    if protected.count > protectedCapacity {
                        var candidate = -1
                        var oldest = Int.max
                        for slot in 0..<capacity
                        where protected.contains(slot) && !locked.contains(slot) {
                            let time = accessTime[slot, default: 0]
                            if time < oldest {
                                oldest = time
                                candidate = slot
                            }
                        }
                        if candidate >= 0 {
                            protected.remove(candidate)
                            probationary.insert(candidate)
                        }
                    }
                }
                continue
            }

            misses += 1
            var victim = (0..<capacity).first {
                slotToKey[$0] == nil && !locked.contains($0)
            } ?? -1
            if victim < 0 {
                var oldest = Int.max
                for slot in 0..<capacity
                where probationary.contains(slot) && !locked.contains(slot) {
                    let time = accessTime[slot, default: 0]
                    if time < oldest {
                        oldest = time
                        victim = slot
                    }
                }
            }
            if victim < 0 {
                var oldest = Int.max
                for slot in 0..<capacity
                where protected.contains(slot) && !locked.contains(slot) {
                    let time = accessTime[slot, default: 0]
                    if time < oldest {
                        oldest = time
                        victim = slot
                    }
                }
            }
            if victim < 0 {
                victim = (0..<capacity).first { !locked.contains($0) } ?? 0
            }
            if let old = slotToKey[victim] {
                keyToSlot.removeValue(forKey: old)
                probationary.remove(victim)
                protected.remove(victim)
            }
            keyToSlot[expert] = victim
            slotToKey[victim] = expert
            accessTime[victim] = tick
            probationary.insert(victim)
            locked.insert(victim)
        }
    }
    return misses
}

private func optimizeLayerCapacities(
    traces: [[UInt16]],
    totalCapacity: Int
) -> (misses: Int, capacities: [Int])? {
    let layerCount = traces.count
    let minimumCapacity = 8
    let maximumCapacity = 256
    guard totalCapacity >= layerCount * minimumCapacity else { return nil }

    var curves: [[Int]] = []
    curves.reserveCapacity(layerCount)
    for trace in traces {
        var curve = [Int](repeating: Int.max, count: maximumCapacity + 1)
        for capacity in minimumCapacity...maximumCapacity {
            curve[capacity] = simulateSegmentedLRU(trace: trace, capacity: capacity)
        }
        curves.append(curve)
    }

    let infinity = Int.max / 4
    var previous = [Int](repeating: infinity, count: totalCapacity + 1)
    previous[0] = 0
    var choices = Array(
        repeating: [Int16](repeating: -1, count: totalCapacity + 1),
        count: layerCount)
    for layer in 0..<layerCount {
        var next = [Int](repeating: infinity, count: totalCapacity + 1)
        for alreadyUsed in 0...totalCapacity where previous[alreadyUsed] < infinity {
            let maximumForLayer = min(maximumCapacity, totalCapacity - alreadyUsed)
            guard maximumForLayer >= minimumCapacity else { continue }
            for capacity in minimumCapacity...maximumForLayer {
                let total = alreadyUsed + capacity
                let candidate = previous[alreadyUsed] + curves[layer][capacity]
                if candidate < next[total] {
                    next[total] = candidate
                    choices[layer][total] = Int16(capacity)
                }
            }
        }
        previous = next
    }
    guard previous[totalCapacity] < infinity else { return nil }
    var capacities = [Int](repeating: minimumCapacity, count: layerCount)
    var remaining = totalCapacity
    for layer in stride(from: layerCount - 1, through: 0, by: -1) {
        let capacity = Int(choices[layer][remaining])
        guard capacity >= minimumCapacity else { return nil }
        capacities[layer] = capacity
        remaining -= capacity
    }
    return (previous[totalCapacity], capacities)
}

@main
private enum QwenBenchMain {
    static func main() async {
        do {
            if CommandLine.arguments.dropFirst().contains("--help") ||
                CommandLine.arguments.dropFirst().contains("-h") {
                print(usage)
                return
            }
            let arguments = try Arguments(CommandLine.arguments)
            let totalRequestedRuns = arguments.warmupRuns + arguments.measuredRuns
            if arguments.analyzeCache, totalRequestedRuns != 1 {
                throw NSError(
                    domain: "QwenFieldfareBench",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                        "--analyze-cache requires exactly one run because traces accumulate with retained cache state"])
            }
            let device = MTLCreateSystemDefaultDevice()
            let loadStarted = CFAbsoluteTimeGetCurrent()
            let runner = try QwenQ4ActiveExpertStreamingRunner(
                modelPath: arguments.modelPath,
                maxSeqLen: arguments.contextLength,
                expertCacheCapacity: arguments.cacheSlots,
                usePersistentExpertCache: true,
                keepDenseWeightsResident: true,
                usePackedExpertStore: arguments.usePackedExperts,
                recordExpertAccessTrace: arguments.analyzeCache,
                enableGPUProfiling: arguments.gpuProfile)
            let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStarted

            print(String(format: "Load: %.3f s", loadSeconds))
            print(String(format: "Loaded footprint: %.3f GiB", gib(processFootprintBytes())))
            print(String(format: "Loaded Metal allocation: %.3f GiB", gib(UInt64(device?.currentAllocatedSize ?? 0))))
            print("Context: \(arguments.contextLength)")
            print("Expert cache slots: \(arguments.cacheSlots)")
            print("Expert storage: \(runner.expertStorageMetrics().kind)")
            print("GPU profiling: \(arguments.gpuProfile ? "enabled" : "disabled")")
            print("Warmup runs: \(arguments.warmupRuns)")
            print("Measured runs: \(arguments.measuredRuns)")
            if arguments.subcomponentProfile {
                let profile = runner.profileSubcomponents(iterations: 20)
                print(String(format: "Attention subcomponents: %.3f ms total", profile.totalAttentionMs))
                print(String(format: "  linear: %.3f ms (norm %.3f, in-proj %.3f, conv %.3f, DeltaNet %.3f, out-proj %.3f, residual %.3f)",
                             profile.totalLinAttnMs, profile.linInNormMs, profile.linInProjMs,
                             profile.linConv1dMs, profile.linFusedDeltanetMs,
                             profile.linOutProjMs, profile.linResAddMs))
                print(String(format: "  full:   %.3f ms (norm %.3f, QKV %.3f, split/RoPE/norm %.3f, attention %.3f, out-proj %.3f, residual %.3f)",
                             profile.totalFullAttnMs, profile.fullInNormMs,
                             profile.fullQKVProjMs, profile.fullSplitRopeNormMs,
                             profile.fullCausalAttnMs, profile.fullOutProjMs,
                             profile.fullResAddMs))
            }
            if arguments.loadOnly { return }

            let tokenizer = try await QwenTokenizer.load(
                from: URL(fileURLWithPath: arguments.modelPath))
            let formattedPrompt = QwenChatTemplate.format(
                messages: [], appendingUser: arguments.prompt, enableThinking: false)
            let promptTokens = tokenizer.encode(formattedPrompt)
            var measuredSummaries: [RunSummary] = []
            var referenceTokens: [Int32]?
            var referenceFinishReason: String?
            var deterministic = true

            for runIndex in 0..<totalRequestedRuns {
                let isWarmup = runIndex < arguments.warmupRuns
                let ordinal = isWarmup
                    ? runIndex + 1
                    : runIndex - arguments.warmupRuns + 1
                let label = isWarmup
                    ? "Warmup \(ordinal)/\(arguments.warmupRuns)"
                    : "Measured \(ordinal)/\(arguments.measuredRuns)"
                print("\n=== \(label) ===")

                // Retain cache contents intentionally while making every
                // printed run's counters independent.
                runner.expertCache.resetMetrics()
                let samples = SampleBox()
                let result = try await runner.generateResult(
                    promptTokens: promptTokens,
                    configuration: .greedy(maxNewTokens: arguments.maxNewTokens, seed: 42),
                    onPrefillComplete: {
                        samples.startDecode(prefillMs: $0)
                    },
                    onToken: { _ in
                        samples.record(
                            profile: runner.latestProfile,
                            device: device,
                            cache: runner.expertCache.metricsSnapshot())
                    })
                let cache = runner.expertCache.metricsSnapshot()
                samples.captureFinal(
                    profile: runner.latestProfile, device: device, cache: cache)
                let decode = samples.decodeSummary()
                let prefillMs = samples.recordedPrefillMs()
                let decodeRate = decode.elapsedMs > 0
                    ? Double(decode.transitionCount) * 1_000.0 / decode.elapsedMs
                    : 0
                let finalFootprint = processFootprintBytes()
                let finalMetal = UInt64(device?.currentAllocatedSize ?? 0)
                let decodeHits = max(0, cache.hits - (decode.firstTokenCache?.hits ?? cache.hits))
                let decodeMisses = max(0, cache.misses - (decode.firstTokenCache?.misses ?? cache.misses))
                let decodeRequests = decodeHits + decodeMisses
                let decodeHitRate = decodeRequests > 0
                    ? Double(decodeHits) * 100.0 / Double(decodeRequests)
                    : 0
                let decodeFetchedBytes = max(
                    0, cache.fetchedBytes - (decode.firstTokenCache?.fetchedBytes ?? cache.fetchedBytes))

                if let referenceTokens {
                    deterministic = deterministic
                        && referenceTokens == result.tokens
                        && referenceFinishReason == result.finishReason.label
                } else {
                    referenceTokens = result.tokens
                    referenceFinishReason = result.finishReason.label
                }

                print("Prompt tokens: \(promptTokens.count)")
                print("Generated tokens: \(result.tokens.count)")
                print("Finish reason: \(result.finishReason.label)")
                print(String(format: "Prefill/TTFT: %.3f s", prefillMs / 1_000.0))
                print(String(
                    format: "Decode: %d inter-token transitions in %.3f s (%.2f tok/s)",
                    decode.transitionCount, decode.elapsedMs / 1_000.0, decodeRate))
                print(String(
                    format: "Inter-token latency: mean %.2f ms, p50 %.2f ms, p95 %.2f ms, min %.2f ms, max %.2f ms (%d samples)",
                    decode.latency.meanMs ?? 0, decode.latency.p50Ms ?? 0,
                    decode.latency.p95Ms ?? 0, decode.latency.minimumMs ?? 0,
                    decode.latency.maximumMs ?? 0, decode.latency.sampleCount))
                print(String(
                    format: "CPU token selection: mean %.3f ms, p50 %.3f ms, p95 %.3f ms (%d samples)",
                    decode.cpuTokenSelection.meanMs ?? 0,
                    decode.cpuTokenSelection.p50Ms ?? 0,
                    decode.cpuTokenSelection.p95Ms ?? 0,
                    decode.cpuTokenSelection.sampleCount))
                print(String(
                    format: "Decode-only cache: %d hits, %d misses, %.2f%% hit rate, %.1f misses/token, %.3f GiB fetched",
                    decodeHits, decodeMisses, decodeHitRate,
                    Double(decodeMisses) / Double(max(1, decode.transitionCount)),
                    gib(UInt64(decodeFetchedBytes))))
                print(String(format: "Final footprint: %.3f GiB", gib(finalFootprint)))
                print(String(format: "Final Metal allocation: %.3f GiB", gib(finalMetal)))

                let timingSamples = samples.snapshot()
                for (sampleIndex, sample) in timingSamples.enumerated() {
                    guard sampleIndex > 0 else {
                        print(String(
                            format: "At %4d: first token produced during prefill; footprint %.3f GiB, Metal %.3f GiB",
                            sample.tokenCount, gib(sample.footprintBytes), gib(sample.metalBytes)))
                        continue
                    }
                    let previous = timingSamples[sampleIndex - 1]
                    let bucketTokens = sample.tokenCount - previous.tokenCount
                    let bucketSeconds = sample.elapsedSeconds - previous.elapsedSeconds
                    let bucketRate = bucketSeconds > 0 ? Double(bucketTokens) / bucketSeconds : 0
                    let bucketMisses = sample.cacheMisses - previous.cacheMisses
                    let bucketFetched = sample.cacheFetchedBytes - previous.cacheFetchedBytes
                    let missesPerToken = Double(bucketMisses) / Double(max(1, bucketTokens))
                    print(String(
                        format: "At %4d: %6.2f tok/s, %5.1f misses/token, %.1f MiB/token, step %6.2f ms, miss fetch %6.2f ms, GPU wait %6.2f ms (attn/router %.2f, MoE %.2f), footprint %.3f GiB, Metal %.3f GiB",
                        sample.tokenCount, bucketRate, missesPerToken,
                        Double(bucketFetched) / Double(max(1, bucketTokens)) / 1_048_576.0,
                        sample.stepMs, sample.cacheMissFetchMs,
                        sample.commandBufferWaitMs, sample.attentionRouterWaitMs,
                        sample.moeWaitMs, gib(sample.footprintBytes), gib(sample.metalBytes)))
                    if arguments.gpuProfile {
                        print(String(
                            format: "         exact GPU: %.2f ms (attention/router %.2f, shared/hits %.2f, miss/tail/head %.2f)",
                            sample.totalGPUMs, sample.attentionRouterGPUMs,
                            sample.sharedHitGPUMs, sample.moeTailGPUMs))
                    }
                }

                print(String(format: "Cache including prefill: %d/%d occupied, %d hits, %d misses, %.2f%% hit rate, %d evictions, %.3f GiB fetched",
                             cache.occupancy, cache.capacity, cache.hits, cache.misses,
                             (cache.hitRate ?? 0) * 100, cache.evictions,
                             gib(UInt64(cache.fetchedBytes))))
                let layerRates = (0..<40).map { layer -> String in
                    let requests = runner.expertCache.layerRequests[layer, default: 0]
                    let hits = runner.expertCache.layerHits[layer, default: 0]
                    let rate = requests > 0 ? Double(hits) * 100.0 / Double(requests) : 0
                    return String(format: "%02d:%.1f", layer, rate)
                }
                print("Layer cache hit rates (%): \(layerRates.joined(separator: " "))")

                if arguments.analyzeCache {
                    let traces = runner.expertCache.accessTraceSnapshot()
                    let capacities = runner.expertCache.layerCapacities
                    let simulated = traces.indices.reduce(0) {
                        $0 + simulateWindowedFrequency(trace: traces[$1], capacity: capacities[$1])
                    }
                    let staticHotSet = traces.indices.reduce(0) {
                        $0 + staticHotSetMisses(trace: traces[$1], capacity: capacities[$1])
                    }
                    let replay = traces.indices.reduce(0) {
                        $0 + simulateSegmentedLRU(trace: traces[$1], capacity: capacities[$1])
                    }
                    print("Cache analysis: measured \(cache.misses), SLRU replay \(replay), windowed-frequency \(simulated), static-hot-set \(staticHotSet) misses")
                    print("Layer cache capacities: \(capacities.map(String.init).joined(separator: " "))")
                    if replay == cache.misses,
                       let optimized = optimizeLayerCapacities(
                        traces: traces, totalCapacity: cache.capacity) {
                        print("Trace-optimized SLRU: \(optimized.misses) misses")
                        print("Trace-optimized capacities: \(optimized.capacities.map(String.init).joined(separator: " "))")
                    } else {
                        print("Trace optimization skipped because SLRU replay did not reproduce measured misses")
                    }
                }

                let summary = RunSummary(
                    label: label,
                    tokens: result.tokens,
                    finishReason: result.finishReason.label,
                    prefillMs: prefillMs,
                    decodeMs: decode.elapsedMs,
                    decodeTransitions: decode.transitionCount,
                    decodeRate: decodeRate,
                    latency: decode.latency,
                    cpuTokenSelection: decode.cpuTokenSelection,
                    cache: cache,
                    decodeMisses: decodeMisses,
                    decodeFetchedBytes: decodeFetchedBytes,
                    finalFootprintBytes: finalFootprint,
                    finalMetalBytes: finalMetal)
                if !isWarmup { measuredSummaries.append(summary) }

                if !arguments.metricsOnly {
                    print("TOKEN_IDS=\(result.tokens)")
                    print("--- OUTPUT ---")
                    print(tokenizer.decode(result.tokens))
                    print("--- END OUTPUT ---")
                }
            }

            print("\n=== Measured summary ===")
            let rates = measuredSummaries.map(\.decodeRate)
            let prefills = measuredSummaries.map(\.prefillMs)
            let selectorMeans = measuredSummaries.compactMap(\.cpuTokenSelection.meanMs)
            if let minimumRate = rates.min(), let maximumRate = rates.max() {
                let meanRate = rates.reduce(0, +) / Double(rates.count)
                print(String(
                    format: "Decode rate across %d runs: min %.2f, mean %.2f, max %.2f tok/s, spread %.2f%%",
                    rates.count, minimumRate, meanRate, maximumRate,
                    meanRate > 0 ? (maximumRate - minimumRate) * 100.0 / meanRate : 0))
            }
            if let minimumPrefill = prefills.min(), let maximumPrefill = prefills.max() {
                let meanPrefill = prefills.reduce(0, +) / Double(prefills.count)
                print(String(
                    format: "Prefill/TTFT across %d runs: min %.3f, mean %.3f, max %.3f s",
                    prefills.count, minimumPrefill / 1_000.0,
                    meanPrefill / 1_000.0, maximumPrefill / 1_000.0))
            }
            if let minimumSelector = selectorMeans.min(),
               let maximumSelector = selectorMeans.max() {
                let meanSelector = selectorMeans.reduce(0, +) / Double(selectorMeans.count)
                print(String(
                    format: "CPU token selection across %d runs: min %.3f, mean %.3f, max %.3f ms",
                    selectorMeans.count, minimumSelector, meanSelector, maximumSelector))
            }
            print(totalRequestedRuns > 1
                ? "Repeated greedy outputs identical: \(deterministic ? "yes" : "NO")"
                : "Repeated greedy outputs identical: not applicable (one run)")
            let storage = runner.expertStorageMetrics()
            print("Expert I/O process totals: \(storage.packedReads) packed experts, \(storage.packedPreadOperations) pread operations, \(storage.packedReadFailures) packed read failures")
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n\n\(usage)\n".utf8))
            exit(1)
        }
    }
}
