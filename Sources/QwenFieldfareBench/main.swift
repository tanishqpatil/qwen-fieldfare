import Darwin.Mach
import Foundation
import Metal
import QwenFieldfareCore

private let usage = """
Usage:
  QwenFieldfareBench [--model <directory>] [--prompt <text>]
    [--context <tokens>] [--max-new <tokens>] [--cache-slots <count>]
    [--load-only] [--metrics-only] [--packed-experts] [--analyze-cache]
    [--gpu-profile] [--subcomponent-profile]
"""

private struct Arguments {
    var modelPath = NSHomeDirectory() + "/models/Qwen3.6-35B-A3B-4bit"
    var prompt = "tell me what hasmap is in java with example"
    var contextLength = 8_192
    var maxNewTokens = 512
    var cacheSlots = 1_024
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
            case "--model", "--prompt", "--context", "--max-new", "--cache-slots":
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
}

private final class SampleBox: @unchecked Sendable {
    private let lock = NSLock()
    private var decodeStart: CFAbsoluteTime?
    private var prefillMs = 0.0
    private var samples: [TokenSample] = []
    private var tokenCount = 0

    func startDecode(prefillMs: Double) {
        lock.lock()
        self.prefillMs = prefillMs
        decodeStart = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    func record(
        profile: QwenQ4ActiveExpertStreamingRunner.StepProfile?,
        device: MTLDevice?,
        cache: QwenExpertCacheMetrics
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard let decodeStart else { return }
        tokenCount += 1
        if tokenCount == 1 || tokenCount % 64 == 0 {
            samples.append(TokenSample(
                tokenCount: tokenCount,
                elapsedSeconds: CFAbsoluteTimeGetCurrent() - decodeStart,
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
                cacheMisses: cache.misses))
        }
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
            let samples = SampleBox()
            let runStarted = CFAbsoluteTimeGetCurrent()
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
            let prefillMs = samples.recordedPrefillMs()
            let totalMs = (CFAbsoluteTimeGetCurrent() - runStarted) * 1_000.0
            let decodeMs = max(0, totalMs - prefillMs)
            let rate = decodeMs > 0 ? Double(result.tokens.count) * 1_000.0 / decodeMs : 0

            print("Prompt tokens: \(promptTokens.count)")
            print("Generated tokens: \(result.tokens.count)")
            print("Finish reason: \(result.finishReason.label)")
            print(String(format: "Prefill: %.3f s", prefillMs / 1_000.0))
            print(String(format: "Decode: %.3f s (%.2f tok/s)", decodeMs / 1_000.0, rate))
            print(String(format: "Final footprint: %.3f GiB", gib(processFootprintBytes())))
            print(String(format: "Final Metal allocation: %.3f GiB", gib(UInt64(device?.currentAllocatedSize ?? 0))))

            let timingSamples = samples.snapshot()
            for (index, sample) in timingSamples.enumerated() {
                let previous = index == 0 ? nil : timingSamples[index - 1]
                let bucketTokens = previous.map { sample.tokenCount - $0.tokenCount } ?? sample.tokenCount
                let bucketSeconds = previous.map { sample.elapsedSeconds - $0.elapsedSeconds } ?? sample.elapsedSeconds
                let bucketRate = bucketSeconds > 0 ? Double(bucketTokens) / bucketSeconds : 0
                let bucketMisses = previous.map { sample.cacheMisses - $0.cacheMisses } ?? sample.cacheMisses
                let missesPerToken = Double(bucketMisses) / Double(max(1, bucketTokens))
                print(String(
                    format: "At %4d: %6.2f tok/s, %5.1f misses/token, step %6.2f ms, miss fetch %6.2f ms, GPU wait %6.2f ms (attn/router %.2f, MoE %.2f), footprint %.3f GiB, Metal %.3f GiB",
                    sample.tokenCount, bucketRate, missesPerToken, sample.stepMs, sample.cacheMissFetchMs,
                    sample.commandBufferWaitMs, sample.attentionRouterWaitMs, sample.moeWaitMs,
                    gib(sample.footprintBytes), gib(sample.metalBytes)))
                if arguments.gpuProfile {
                    print(String(
                        format: "         exact GPU: %.2f ms (attention/router %.2f, shared/hits %.2f, miss/tail/head %.2f)",
                        sample.totalGPUMs, sample.attentionRouterGPUMs,
                        sample.sharedHitGPUMs, sample.moeTailGPUMs))
                }
            }
            let cache = runner.expertCache.metricsSnapshot()
            print(String(format: "Cache: %d/%d occupied, %d hits, %d misses, %.2f%% hit rate, %d evictions, %.3f GiB fetched",
                         cache.occupancy, cache.capacity, cache.hits, cache.misses,
                         (cache.hitRate ?? 0) * 100, cache.evictions, gib(UInt64(cache.fetchedBytes))))
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
            let storage = runner.expertStorageMetrics()
            print("Expert I/O: \(storage.packedReads) packed experts, \(storage.packedPreadOperations) pread operations, \(storage.packedReadFailures) packed read failures")
            if !arguments.metricsOnly {
                print("TOKEN_IDS=\(result.tokens)")
                print("--- OUTPUT ---")
                print(tokenizer.decode(result.tokens))
                print("--- END OUTPUT ---")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n\n\(usage)\n".utf8))
            exit(1)
        }
    }
}
