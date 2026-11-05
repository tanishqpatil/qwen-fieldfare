import Darwin
import Darwin.Mach
import Foundation
import Metal
import SwiftUI
import QwenFieldfareCore

private final class GenerationTelemetryBox: @unchecked Sendable {
    struct Update: Sendable {
        let tokens: [Int32]
        let firstTokenMs: Double?
        let prefillMs: Double?
        let latency: QwenLatencySummary
        let rollingTokensPerSecond: Double?
        let runAverageTokensPerSecond: Double?
    }

    private let lock = NSLock()
    private let startedAt = CFAbsoluteTimeGetCurrent()
    private var firstTokenAt: Double?
    private var lastTokenAt: Double?
    private var lastPublishedAt: Double = 0
    private var tokens: [Int32] = []
    private var runAccumulator = QwenBoundedLatencyAccumulator()
    private var rollingAccumulator = QwenBoundedLatencyAccumulator(capacity: 64)
    private(set) var prefillMs: Double?

    func setPrefill(milliseconds: Double) {
        lock.lock()
        prefillMs = milliseconds
        lock.unlock()
    }

    func record(token: Int32) -> Update? {
        let now = CFAbsoluteTimeGetCurrent()
        lock.lock()
        if let previous = lastTokenAt {
            let latencyMs = (now - previous) * 1_000.0
            runAccumulator.record(milliseconds: latencyMs)
            rollingAccumulator.record(milliseconds: latencyMs)
        }
        if firstTokenAt == nil { firstTokenAt = now }
        lastTokenAt = now
        tokens.append(token)
        guard now - lastPublishedAt >= 0.25 else {
            lock.unlock()
            return nil
        }
        lastPublishedAt = now
        let update = makeUpdateLocked()
        lock.unlock()
        return update
    }

    func finalUpdate() -> Update {
        lock.lock()
        let update = makeUpdateLocked()
        lock.unlock()
        return update
    }

    private func makeUpdateLocked() -> Update {
        let runSummary = runAccumulator.summary
        let rollingSummary = rollingAccumulator.summary
        return Update(
            tokens: tokens,
            firstTokenMs: firstTokenAt.map { ($0 - startedAt) * 1_000.0 },
            prefillMs: prefillMs,
            latency: runSummary,
            rollingTokensPerSecond: rollingSummary.meanMs.flatMap { $0 > 0 ? 1_000.0 / $0 : nil },
            runAverageTokensPerSecond: runSummary.meanMs.flatMap { $0 > 0 ? 1_000.0 / $0 : nil }
        )
    }
}

@MainActor
@Observable
public final class ChatService {
    public static let modelPathEnvironmentVariable = "QWENFIELDFARE_MODEL"
    public static let modelPathDefaultsKey = "QwenFieldfare.modelPath"

    public static var defaultModelPath: String {
        if let configured = ProcessInfo.processInfo.environment[modelPathEnvironmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            return (configured as NSString).expandingTildeInPath
        }
        if let stored = UserDefaults.standard.string(forKey: modelPathDefaultsKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !stored.isEmpty {
            return (stored as NSString).expandingTildeInPath
        }
        return NSHomeDirectory() + "/models/Qwen3.6-35B-A3B-4bit"
    }

    public enum ModelState: String, Sendable {
        case unloaded = "Unloaded"
        case loading = "Loading…"
        case ready = "Ready"
        case generating = "Generating"
        case cancelling = "Cancelling…"
        case unloading = "Unloading…"
        case error = "Error"
    }

    public var currentResponse = ""
    public var messages: [ChatMessage] = []
    public var isGenerating = false
    public var errorMessage: String?
    public var memoryWarning: String?
    public var modelState: ModelState = .unloaded
    public var selectedMode: QwenQ4ChatRunner.RuntimeMode = .activeStreaming {
        didSet { metrics.mode = selectedMode.rawValue }
    }
    public private(set) var activeStreamingMemoryBudget: QwenActiveStreamingMemoryBudget = .fiveGiB
    public private(set) var metrics = QwenRuntimeMetricsSnapshot(
        mode: QwenQ4ChatRunner.RuntimeMode.activeStreaming.rawValue,
        modelState: ModelState.unloaded.rawValue
    )
    public var unloadWhenIdle = true {
        didSet { unloadWhenIdle ? resetIdleTimer() : cancelIdleTimer() }
    }
    public var enableThinking = false
    public private(set) var contextLength = 8_192
    public var maxOutputTokens = 4_096
    public var temperature: Float = 1.0
    public var topK = 20
    public var topP: Float = 0.95
    public var presencePenalty: Float = 1.5
    public var repetitionPenalty: Float = 1.0

    public var runtimeMode: String { selectedMode.rawValue }
    public var metalAllocatedGiB: Double { gib(metrics.memory.metalAllocatedBytes) ?? 0 }
    public var systemWiredGiB: Double { gib(metrics.memory.systemWiredBytes) ?? 0 }
    public var memoryPressure: String { metrics.memory.pressure ?? "N/A" }
    public var generationTokPerSec: Double { metrics.steadyTokensPerSecond ?? 0 }
    public var canUnloadModel: Bool {
        !isGenerating
            && modelState != .loading
            && modelState != .unloading
            && (activeRunner != nil || chatRunner != nil)
    }
    public var expertCacheHitRate: Double { (metrics.cache?.hitRate ?? 0) * 100 }
    public var expertCacheMissesPerToken: Double { metrics.missesPerGeneratedToken ?? 0 }
    public var estimatedKVGiB: Double {
        Double(activeStreamingMemoryConfiguration?.kvStateBytes ?? 0) / 1_073_741_824.0
    }
    public var activeStreamingMemoryConfiguration: QwenActiveStreamingMemoryConfiguration? {
        try? QwenActiveStreamingMemoryConfiguration.make(
            budget: activeStreamingMemoryBudget,
            contextLength: contextLength
        )
    }
    public var expertCacheSlots: Int {
        guard selectedMode == .activeStreaming else { return 0 }
        return activeStreamingMemoryConfiguration?.expertCacheSlots ?? 0
    }
    public var estimatedActiveRuntimeGiB: Double {
        Double(activeStreamingMemoryConfiguration?.estimatedTotalBytes ?? 0) / 1_073_741_824.0
    }
    public var supportedContextLengths: [Int] {
        guard selectedMode == .activeStreaming else { return [4_096, 8_192] }
        return [4_096, 8_192, 16_384, 32_768, 65_536].filter {
            (try? QwenActiveStreamingMemoryConfiguration.make(
                budget: activeStreamingMemoryBudget,
                contextLength: $0
            )) != nil
        }
    }

    private var activeRunner: QwenQ4ActiveExpertStreamingRunner?
    private var chatRunner: QwenQ4ChatRunner?
    private var tokenizer: QwenTokenizer?
    private var conversation: [QwenChatTemplate.Message] = []
    private var activeAssistantMessageID: UUID?
    public private(set) var modelPath: String

    @ObservationIgnored nonisolated(unsafe) private var generationTask: Task<Void, Never>?
    @ObservationIgnored nonisolated(unsafe) private var idleTimerTask: Task<Void, Never>?
    @ObservationIgnored nonisolated(unsafe) private var telemetryTimer: Timer?
    private var unloadRequested = false
    private var runSwapBaseline: UInt64?

    public init(modelPath: String? = nil) {
        self.modelPath = modelPath ?? ChatService.defaultModelPath
        updateTelemetry()
        startTelemetryTimer()
        resetIdleTimer()
    }

    deinit {
        telemetryTimer?.invalidate()
        generationTask?.cancel()
        idleTimerTask?.cancel()
    }

    private static func gib(_ bytes: UInt64?) -> Double? {
        bytes.map { Double($0) / 1_073_741_824.0 }
    }

    private func gib(_ bytes: UInt64?) -> Double? { Self.gib(bytes) }

    private func startTelemetryTimer() {
        telemetryTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.updateTelemetry() }
        }
    }

    public func updateTelemetry() {
        let current = Self.captureMemory(device: MTLCreateSystemDefaultDevice())
        let previousPeak = metrics.memory.peakMetalAllocatedBytes ?? 0
        metrics.memory = current
        if let metal = current.metalAllocatedBytes {
            metrics.memory.peakMetalAllocatedBytes = max(previousPeak, metal)
        }
        if let baseline = runSwapBaseline, let swap = current.swapUsedBytes {
            metrics.memory.swapDeltaBytes = Int64(swap) - Int64(baseline)
        }
    }

    public func preflightMemoryCheck() -> (allowed: Bool, warning: String?) {
        updateTelemetry()
        guard memoryPressure != "Critical" else {
            return (false, "System memory pressure is critical.")
        }
        let total = ProcessInfo.processInfo.physicalMemory
        let used = metrics.memory.hostUsedBytes ?? total
        let availableGiB = Double(total > used ? total - used : 0) / 1_073_741_824.0
        let minimumGiB: Double = selectedMode == .fastResident ? 12 : 4
        if selectedMode == .activeStreaming, activeStreamingMemoryConfiguration == nil {
            return (false, "The selected active-streaming context does not fit its memory budget.")
        }
        guard availableGiB >= minimumGiB else {
            return (false, String(format: "Insufficient estimated available memory (%.1f GiB).", availableGiB))
        }
        return (true, nil)
    }

    public func setMode(_ newMode: QwenQ4ChatRunner.RuntimeMode) {
        guard newMode != selectedMode else { return }
        unloadModel()
        guard !isGenerating else { return }
        selectedMode = newMode
        if !supportedContextLengths.contains(contextLength) { contextLength = 8_192 }
        metrics = QwenRuntimeMetricsSnapshot(mode: newMode.rawValue, modelState: modelState.rawValue)
        updateTelemetry()
    }

    public func setContextLength(_ tokens: Int) {
        guard supportedContextLengths.contains(tokens), tokens != contextLength else { return }
        if activeRunner != nil || chatRunner != nil { unloadModel() }
        contextLength = tokens
        maxOutputTokens = min(maxOutputTokens, tokens)
    }

    public func setActiveStreamingMemoryBudget(_ budget: QwenActiveStreamingMemoryBudget) {
        guard budget != activeStreamingMemoryBudget else { return }
        guard !isGenerating else { return }
        if activeRunner != nil || chatRunner != nil { unloadModel() }
        activeStreamingMemoryBudget = budget
        if !supportedContextLengths.contains(contextLength) {
            contextLength = 8_192
            maxOutputTokens = min(maxOutputTokens, contextLength)
        }
    }

    public func setModelPath(_ path: String) throws {
        guard !isGenerating, modelState != .loading, modelState != .unloading else {
            throw NSError(
                domain: "QwenFieldfare.ChatService",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Stop generation before changing the model directory."])
        }
        let validatedPath = try QwenModelDirectory.validate(path: path)
        guard validatedPath != modelPath else { return }
        finishUnload()
        modelPath = validatedPath
        UserDefaults.standard.set(validatedPath, forKey: Self.modelPathDefaultsKey)
        resetConversation()
        errorMessage = nil
    }

    public func loadModelIfNeeded() async throws {
        guard activeRunner == nil && chatRunner == nil else { return }
        let (allowed, warning) = preflightMemoryCheck()
        memoryWarning = warning
        guard allowed else {
            throw NSError(domain: "ChatService", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: warning ?? "Memory preflight failed."])
        }

        setState(.loading)
        let started = CFAbsoluteTimeGetCurrent()
        switch selectedMode {
        case .activeStreaming:
            activeRunner = try QwenQ4ActiveExpertStreamingRunner(
                modelPath: modelPath,
                maxSeqLen: contextLength,
                expertCacheCapacity: expertCacheSlots,
                usePersistentExpertCache: true,
                keepDenseWeightsResident: true
            )
        case .fastResident, .memorySaver:
            chatRunner = try QwenQ4ChatRunner(
                modelPath: modelPath,
                maxSeqLen: contextLength,
                mode: selectedMode
            )
        }
        tokenizer = try await QwenTokenizer.load(from: URL(fileURLWithPath: modelPath))
        metrics.loadDurationMs = (CFAbsoluteTimeGetCurrent() - started) * 1_000.0
        setState(.ready)
        updateTelemetry()
    }

    public func unloadModel() {
        cancelIdleTimer()
        if isGenerating {
            unloadRequested = true
            cancelGeneration()
            return
        }
        finishUnload()
    }

    public func cancelGeneration() {
        guard isGenerating else { return }
        setState(.cancelling)
        generationTask?.cancel()
    }

    private func finishUnload() {
        setState(.unloading)
        activeRunner = nil
        chatRunner = nil
        tokenizer = nil
        activeAssistantMessageID = nil
        currentResponse = ""
        memoryWarning = nil
        metrics.cache = nil
        metrics.steadyTokensPerSecond = nil
        metrics.missesPerGeneratedToken = nil
        metrics.fetchedBytesPerGeneratedToken = nil
        metrics.stagedBytesPerGeneratedToken = nil
        setState(.unloaded)
        updateTelemetry()
        unloadRequested = false
    }

    public func resetConversation() {
        guard !isGenerating else { return }
        conversation.removeAll()
        messages.removeAll()
        activeAssistantMessageID = nil
        currentResponse = ""
    }

    private func cancelIdleTimer() {
        idleTimerTask?.cancel()
        idleTimerTask = nil
    }

    private func resetIdleTimer() {
        cancelIdleTimer()
        guard unloadWhenIdle, !isGenerating else { return }
        idleTimerTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(60))
                guard !Task.isCancelled else { return }
                self?.unloadModel()
            } catch {}
        }
    }

    public func send(message: String) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        cancelIdleTimer()
        isGenerating = true
        errorMessage = nil
        currentResponse = ""

        // Append user turn
        let userMsg = ChatMessage(role: .user, content: trimmed)
        messages.append(userMsg)

        // Create streaming assistant turn placeholder
        let assistantMsgID = UUID()
        activeAssistantMessageID = assistantMsgID
        let assistantMsg = ChatMessage(id: assistantMsgID, role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMsg)

        metrics = QwenRuntimeMetricsSnapshot(mode: selectedMode.rawValue, modelState: modelState.rawValue)
        updateTelemetry()
        runSwapBaseline = metrics.memory.swapUsedBytes

        let prompt = formattedPrompt(appendingUser: trimmed)
        let mode = selectedMode
        let ctxLen = contextLength
        let slots = expertCacheSlots
        let path = modelPath
        let config = QwenGenerationConfiguration(
            maxNewTokens: maxOutputTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            presencePenalty: presencePenalty,
            repetitionPenalty: repetitionPenalty
        )

        let currentRunnerA = activeRunner
        let currentRunnerC = chatRunner
        let currentTokenizer = tokenizer

        let onTokenUpdate: @Sendable (GenerationTelemetryBox.Update, QwenExpertCacheMetrics?, QwenTokenizer) -> Void = { [weak self] update, cache, tok in
            Task { @MainActor [weak self] in
                self?.publish(update: update, cache: cache, tokenizer: tok)
            }
        }

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var runnerA = currentRunnerA
                var runnerC = currentRunnerC
                var tok = currentTokenizer

                if runnerA == nil && runnerC == nil {
                    await MainActor.run {
                        self?.setState(.loading)
                    }
                    let started = CFAbsoluteTimeGetCurrent()
                    switch mode {
                    case .activeStreaming:
                        runnerA = try QwenQ4ActiveExpertStreamingRunner(
                            modelPath: path,
                            maxSeqLen: ctxLen,
                            expertCacheCapacity: slots,
                            usePersistentExpertCache: true,
                            keepDenseWeightsResident: true
                        )
                    case .fastResident, .memorySaver:
                        runnerC = try QwenQ4ChatRunner(
                            modelPath: path,
                            maxSeqLen: ctxLen,
                            mode: mode
                        )
                    }
                    tok = try await QwenTokenizer.load(from: URL(fileURLWithPath: path))
                    let loadMs = (CFAbsoluteTimeGetCurrent() - started) * 1_000.0
                    await MainActor.run {
                        self?.activeRunner = runnerA
                        self?.chatRunner = runnerC
                        self?.tokenizer = tok
                        self?.metrics.loadDurationMs = loadMs
                        self?.setState(.ready)
                        self?.updateTelemetry()
                    }
                }

                try Task.checkCancellation()
                guard let tokenizer = tok else {
                    throw NSError(domain: "ChatService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Tokenizer unavailable."])
                }

                let promptTokens = tokenizer.encode(prompt)

                await MainActor.run {
                    self?.metrics.promptTokenCount = promptTokens.count
                    self?.metrics.classification = (currentRunnerA != nil || currentRunnerC != nil) ? "warm runner; OS page-cache uncontrolled" : "new runner; OS page-cache uncontrolled"
                    self?.metrics.memory.peakMetalAllocatedBytes = self?.metrics.memory.metalAllocatedBytes
                    self?.setState(.generating)
                }

                let telemetryBox = GenerationTelemetryBox()
                runnerA?.expertCache.resetMetrics()

                let generationResult: QwenGenerationResult
                if let runnerA {
                    generationResult = try await runnerA.generateResult(
                        promptTokens: promptTokens,
                        configuration: config,
                        onPrefillComplete: { ms in
                            telemetryBox.setPrefill(milliseconds: ms)
                            runnerA.expertCache.resetMetrics()
                        },
                        onToken: { token in
                            guard let update = telemetryBox.record(token: token) else { return }
                            let cache = runnerA.expertCache.metricsSnapshot()
                            onTokenUpdate(update, cache, tokenizer)
                        }
                    )
                } else if let runnerC {
                    generationResult = try await runnerC.generateResult(
                        promptTokens: promptTokens,
                        configuration: config,
                        onPrefillComplete: { ms in
                            telemetryBox.setPrefill(milliseconds: ms)
                        },
                        onToken: { token in
                            guard let update = telemetryBox.record(token: token) else { return }
                            onTokenUpdate(update, nil, tokenizer)
                        }
                    )
                } else {
                    throw NSError(domain: "ChatService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model runner unavailable."])
                }

                let final = telemetryBox.finalUpdate()
                let decoded = tokenizer.decode(generationResult.tokens)
                let cacheSnapshot = runnerA?.expertCache.metricsSnapshot()

                await MainActor.run {
                    self?.finishGeneration(
                        result: generationResult,
                        decoded: decoded,
                        userPrompt: trimmed,
                        finalUpdate: final,
                        cache: cacheSnapshot,
                        tokenizer: tokenizer,
                        assistantMsgID: assistantMsgID
                    )
                }
            } catch is CancellationError {
                await MainActor.run {
                    self?.handleCancellation(assistantMsgID: assistantMsgID)
                }
            } catch {
                await MainActor.run {
                    self?.handleError(error, assistantMsgID: assistantMsgID)
                }
            }
        }

        generationTask = task
    }

    private func finishGeneration(
        result: QwenGenerationResult,
        decoded: String,
        userPrompt: String,
        finalUpdate: GenerationTelemetryBox.Update,
        cache: QwenExpertCacheMetrics?,
        tokenizer: QwenTokenizer,
        assistantMsgID: UUID
    ) {
        publish(update: finalUpdate, cache: cache, tokenizer: tokenizer)
        metrics.prefillDurationMs = finalUpdate.prefillMs
        metrics.finishReason = result.finishReason.label
        currentResponse = decoded

        let msgMetrics = ChatMessageMetrics(
            steadyTokensPerSecond: metrics.steadyTokensPerSecond,
            runAverageTokensPerSecond: metrics.runAverageTokensPerSecond,
            firstTokenLatencyMs: metrics.firstTokenLatencyMs,
            generatedTokenCount: result.tokens.count,
            prefillDurationMs: metrics.prefillDurationMs,
            cacheHitRate: metrics.cache?.hitRate,
            finishReason: result.finishReason.label
        )
        if let idx = messages.firstIndex(where: { $0.id == assistantMsgID }) {
            messages[idx].content = currentResponse
            messages[idx].isStreaming = false
            messages[idx].metrics = msgMetrics
        }

        conversation.append(QwenChatTemplate.Message(role: .user, content: userPrompt))
        conversation.append(QwenChatTemplate.Message(
            role: .assistant,
            content: QwenChatTemplate.visibleAssistantContent(currentResponse)
        ))
        setState(.ready)
        cleanupAfterSend()
    }

    private func handleCancellation(assistantMsgID: UUID) {
        metrics.wasCancelled = true
        if let idx = messages.firstIndex(where: { $0.id == assistantMsgID }) {
            messages[idx].isStreaming = false
            if messages[idx].content.isEmpty {
                messages[idx].content = "Generation cancelled."
            }
        }
        setState(unloadRequested ? .unloading : .ready)
        cleanupAfterSend()
    }

    private func handleError(_ error: Error, assistantMsgID: UUID) {
        errorMessage = error.localizedDescription
        metrics.error = error.localizedDescription
        if let idx = messages.firstIndex(where: { $0.id == assistantMsgID }) {
            messages[idx].isStreaming = false
            messages[idx].errorMessage = error.localizedDescription
        }
        setState(.error)
        cleanupAfterSend()
    }

    private func cleanupAfterSend() {
        activeAssistantMessageID = nil
        isGenerating = false
        generationTask = nil
        updateTelemetry()
        runSwapBaseline = nil
        if unloadRequested { finishUnload() }
        else if unloadWhenIdle { resetIdleTimer() }
    }

    private func formattedPrompt(appendingUser message: String) -> String {
        QwenChatTemplate.format(
            messages: conversation,
            appendingUser: message,
            enableThinking: enableThinking
        )
    }

    private func publish(
        update: GenerationTelemetryBox.Update,
        cache: QwenExpertCacheMetrics?,
        tokenizer: QwenTokenizer
    ) {
        currentResponse = tokenizer.decode(update.tokens)
        if let id = activeAssistantMessageID, let idx = messages.firstIndex(where: { $0.id == id }) {
            messages[idx].content = currentResponse
        }
        metrics.generatedTokenCount = update.tokens.count
        metrics.firstTokenLatencyMs = update.firstTokenMs
        metrics.prefillDurationMs = update.prefillMs
        metrics.latency = update.latency
        metrics.steadyTokensPerSecond = update.rollingTokensPerSecond
        metrics.runAverageTokensPerSecond = update.runAverageTokensPerSecond
        metrics.cache = cache
        if let cache, !update.tokens.isEmpty {
            let count = Double(update.tokens.count)
            metrics.missesPerGeneratedToken = Double(cache.misses) / count
            metrics.fetchedBytesPerGeneratedToken = Double(cache.fetchedBytes) / count
            metrics.stagedBytesPerGeneratedToken = Double(cache.fetchedBytes) / count
        }
        updateTelemetry()
    }

    private func setState(_ state: ModelState) {
        modelState = state
        metrics.modelState = state.rawValue
    }

    private static func captureMemory(device: MTLDevice?) -> QwenMemoryMetrics {
        var result = QwenMemoryMetrics()
        result.metalAllocatedBytes = device.map { UInt64($0.currentAllocatedSize) }

        var vmStats = vm_statistics64()
        var vmCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let vmResult = withUnsafeMutablePointer(to: &vmStats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmCount)
            }
        }
        if vmResult == KERN_SUCCESS {
            let page = UInt64(getpagesize())
            result.systemWiredBytes = UInt64(vmStats.wire_count) * page
            result.hostUsedBytes = UInt64(vmStats.wire_count + vmStats.active_count + vmStats.compressor_page_count) * page
        }

        var basic = mach_task_basic_info()
        var basicCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let basicResult = withUnsafeMutablePointer(to: &basic) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(basicCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &basicCount)
            }
        }
        if basicResult == KERN_SUCCESS {
            result.processRSSBytes = UInt64(basic.resident_size)
            result.virtualBytes = UInt64(basic.virtual_size)
        }

        var footprint = task_vm_info_data_t()
        var footprintCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let footprintResult = withUnsafeMutablePointer(to: &footprint) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(footprintCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &footprintCount)
            }
        }
        if footprintResult == KERN_SUCCESS { result.physicalFootprintBytes = UInt64(footprint.phys_footprint) }

        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        if sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0) == 0 {
            result.swapUsedBytes = UInt64(swap.xsu_used)
        }

        var pressure: Int32 = 0
        var pressureSize = MemoryLayout<Int32>.size
        if sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &pressureSize, nil, 0) == 0 {
            result.pressure = switch pressure {
            case 1: "Normal"
            case 2: "Warning"
            case 4: "Critical"
            default: "Unknown (\(pressure))"
            }
        }
        return result
    }
}
