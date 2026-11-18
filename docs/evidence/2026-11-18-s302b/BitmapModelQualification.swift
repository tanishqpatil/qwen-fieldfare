import CryptoKit
import Foundation
import Metal
import Testing
@testable import QwenFieldfareCore
@testable import QwenFieldfareServerSupport

private func hash(_ bytes: Data) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
}

private func hash<T>(_ values: [T]) -> String {
    values.withUnsafeBytes { hash(Data($0)) }
}

private final class Capture: @unchecked Sendable {
    private let lock = NSLock()
    private var ticks: [UInt64] = []
    private var selector: [Double] = []
    private var prefill = 0.0
    private var boundaries: [String: [String]] = [:]
    private var footprint = 0
    private let detailed: Bool

    init(detailed: Bool) { self.detailed = detailed }

    func setPrefill(_ time: Double) { lock.withLock { prefill = time } }

    func record(_ runner: QwenQ4ActiveExpertStreamingRunner) {
        let tick = DispatchTime.now().uptimeNanoseconds
        lock.withLock {
            let index = ticks.count
            ticks.append(tick)
            if index > 0 { selector.append(runner.latestProfile?.cpuArgmaxMs ?? 0) }
            if index == 0 || (index + 1).isMultiple(of: 64) {
                footprint = max(footprint, Int(BackendDiagnostics.currentPhysicalFootprint ?? 0))
            }
            if detailed && [0, 15, 127, 255, 465, 511].contains(index) {
                let tap = runner.debugSequenceBoundary()
                boundaries[String(index)] = [hash(tap.hidden), hash(tap.logits)]
            }
        }
    }

    func metrics(start: UInt64, end: UInt64) -> [String: Any] {
        lock.withLock {
            let intervals = zip(ticks.dropFirst(), ticks).map { $0 - $1 }
            return [
                "prefill_ms": prefill,
                "ttft_ns": ticks.first.map { $0 - start } ?? 0,
                "request_ns": end - start,
                "intervals_ns": intervals,
                "selector_cpu_wall_ms": selector,
                "hidden_logits_sha256": boundaries,
                "footprint_bytes": max(footprint, Int(BackendDiagnostics.currentPhysicalFootprint ?? 0)),
            ]
        }
    }
}

@Suite(.serialized)
struct BitmapModelQualification {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWEN_BITMAP_CHECKPOINT"] != nil))
    func runModel() async throws {
        let env = ProcessInfo.processInfo.environment
        let model = try QwenModelDirectory.validate(path: try #require(env["QWEN_BITMAP_CHECKPOINT"]))
        let output = URL(fileURLWithPath: try #require(env["QWEN_BITMAP_OUTPUT"]))
        let phase = try #require(env["QWEN_BITMAP_PHASE"])
        let detailed = phase == "correctness"
        let limit = Int(env["QWEN_BITMAP_LIMIT"] ?? "128") ?? 0
        let text = "Explain HashMap in Java with a short example. Then write a detailed tutorial "
            + "of at least 1500 words, with complete runnable examples for frequency counting, "
            + "grouping, and composite keys, followed by implementation details, failure modes, "
            + "and exercises with full solutions."
        let tokenizer = try await QwenTokenizer.load(from: URL(fileURLWithPath: model))
        func prompt(_ content: String) -> [Int32] {
            tokenizer.encode(QwenChatTemplate.format(messages: [], appendingUser: content, enableThinking: false))
        }
        let plan = try QwenActiveStreamingMemoryConfiguration.make(budget: .fiveGiB, contextLength: 8_192)
        let runner = try QwenQ4ActiveExpertStreamingRunner(
            modelPath: model, maxSeqLen: 8_192, expertCacheCapacity: plan.expertCacheSlots,
            usePersistentExpertCache: true, keepDenseWeightsResident: true, usePackedExpertStore: false,
            recordExpertAccessTrace: detailed, enableGPUProfiling: false)
        let ready = prompt("Reply with exactly: ready")
        let publicPrompt = prompt(text)
        #expect(publicPrompt.count == 65)
        let defaultSample = QwenGenerationConfiguration(maxNewTokens: 512, seed: 42)
        // Fixed one-candidate, nonzero-temperature selector for a known native tool fixture.
        // The headline workload always uses the application's seeded default distribution.
        let toolSample = QwenGenerationConfiguration(maxNewTokens: 128, temperature: 1, topK: 1,
            topP: 1, presencePenalty: 0, repetitionPenalty: 1, seed: 42)
        var records: [[String: Any]] = []
        let info: [String: Any] = [
            "slots": plan.expertCacheSlots, "planned_metal_bytes": plan.estimatedTotalBytes,
            "public_prompt_ids": publicPrompt,
        ]
        func save() throws {
            let result = try JSONSerialization.data(withJSONObject: ["metadata": info, "records": records],
                options: [.prettyPrinted, .sortedKeys])
            try result.write(to: output, options: .atomic)
        }

        func measured(_ label: String, _ ids: [Int32], _ initial: QwenGenerationConfiguration,
                      limit: Int) async throws -> QwenGenerationResult {
            var config = initial
            config.maxNewTokens = limit
            let capture = Capture(detailed: detailed)
            let before = runner.expertCache.metricsSnapshot()
            let traceStarts = detailed ? runner.expertCache.accessTraceSnapshot().map(\.count) : []
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try await runner.generateResult(promptTokens: ids, configuration: config,
                onPrefillComplete: { capture.setPrefill($0) }, onToken: { _ in capture.record(runner) })
            let end = DispatchTime.now().uptimeNanoseconds
            let after = runner.expertCache.metricsSnapshot()
            let state = runner.prefixCacheSnapshot
            let processed = state.processedPromptTokens + state.reusedPrefixTokens
            #expect(processed == ids.count)
            var record = capture.metrics(start: start, end: end)
            record["label"] = label
            record["prompt_sha256"] = hash(ids)
            record["prompt_count"] = ids.count
            record["output_sha256"] = hash(result.tokens)
            record["raw_output_ids"] = result.tokens  // compared locally, stripped before publishing
            record["output_count"] = result.tokens.count
            record["finish"] = String(describing: result.finishReason)
            record["metal_bytes"] = MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0
            record["hits"] = after.hits - before.hits
            record["misses"] = after.misses - before.misses
            record["fetched_bytes"] = after.fetchedBytes - before.fetchedBytes
            record["reused"] = state.reusedPrefixTokens
            record["fresh"] = state.processedPromptTokens
            record["reason"] = state.reason?.rawValue ?? ""
            if detailed {
                let trace = runner.expertCache.accessTraceSnapshot()
                record["expert_sha256_by_layer"] = zip(trace, traceStarts).map {
                    hash(Array($0.dropFirst($1)))
                }
            }
            records.append(record)
            try save()
            return result
        }

        if !detailed {
            #expect(phase == "timing" && (limit == 128 || limit == 512))
            _ = try await runner.generateResult(promptTokens: ready,
                configuration: .init(maxNewTokens: 8, seed: 42))
            let result = try await measured("timing", publicPrompt, defaultSample, limit: limit)
            #expect(result.tokens.count == limit && result.finishReason == .maxNewTokens)
            return
        }

        for limit in [128, 512] {
            runner.expertCache.clear()
            let result = try await measured("sample-\(limit)", publicPrompt, defaultSample, limit: limit)
            #expect(result.tokens.count == limit && result.finishReason == .maxNewTokens)
        }
        runner.expertCache.clear()
        runner.setPrefixCacheMode(.exactPrefix)
        let fixtureURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("docs/evidence/2026-11-18-s300/reconstructed-tool-requests.json")
        guard case .array(let fixtures) = try ToolJSON.parse(String(contentsOf: fixtureURL, encoding: .utf8))
        else { throw APIError(500, "Invalid public tool fixture") }
        let fixture = try #require(fixtures.first { $0["label"]?.string?.hasPrefix("tool-") == true })
        let toolsJSON = try #require(fixture["request"]?["tools"])
        let source = try String(contentsOf: URL(fileURLWithPath: model).appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
        let renderer = try ToolPrompt(source: source)
        let user = "Call audit_chunk with index 1 exactly once. After its result, write a complete Python implementation of a bounded LRU cache with type annotations, detailed docstrings, and at least 200 lines of example-based tests. Do not summarize; provide the complete module."
        let userMessage = ToolJSON.object([.init("role", .string("user")), .init("content", .string(user))])
        func rendered(_ messages: [ToolJSON]) throws -> (ids: [Int32], tools: [FunctionTool]) {
            let conversation = try ToolConversation(.object([
                .init("messages", .array(messages)), .init("tools", toolsJSON),
            ]))
            return (tokenizer.encode(try renderer.render(conversation)), conversation.tools)
        }
        let setup = try rendered([userMessage])
        let call = try await measured("tool-call", setup.ids, toolSample, limit: 128)
        let parser = NativeToolParser(tools: setup.tools)
        try parser.append(tokenizer.decode(call.tokens))
        let parsed = try parser.finish(endedNormally: true)
        let native = try #require(parsed.calls.first)
        #expect(parsed.calls.count == 1 && native.name == "audit_chunk" && native.arguments["index"] == .number("1"))
        guard case .endOfSequence = call.finishReason else { throw APIError(500, "Tool did not finish normally") }
        runner.commitPrefixCache()
        #expect(runner.prefixCacheSnapshot.valid)
        let assistant = ToolJSON.object([
            .init("role", .string("assistant")), .init("content", .string(parsed.content)),
            .init("tool_calls", .array([.object([
                .init("id", .string("sustained-call")), .init("type", .string("function")),
                .init("function", .object([.init("name", .string(native.name)),
                    .init("arguments", .string(native.arguments.encoded))])),
            ])])),
        ])
        let reply = ToolJSON.object([.init("role", .string("tool")),
            .init("tool_call_id", .string("sustained-call")),
            .init("content", .string("Chunk 1 reviewed; proceed with the complete requested Python module."))])
        let suffix = try rendered([userMessage, assistant, reply]).ids
        _ = try await measured("tool-reuse", suffix, toolSample, limit: 32)
        #expect(runner.prefixCacheSnapshot.reusedPrefixTokens > 0)
        records[records.count - 1]["validated_tool"] = native.name

        _ = try await measured("publish-before-mismatch", ready, toolSample, limit: 16)
        runner.commitPrefixCache()
        #expect(runner.prefixCacheSnapshot.valid)
        let mismatch = prompt("Reply with exactly: different")
        _ = try await measured("prompt-mismatch", mismatch, toolSample, limit: 8)
        #expect(runner.prefixCacheSnapshot.reusedPrefixTokens == 0)
        #expect(runner.prefixCacheSnapshot.lastInvalidationReason != nil)
        runner.invalidatePrefixCache(.explicitClear)
        #expect(!runner.prefixCacheSnapshot.valid)

        let cancelled = Task {
            try await runner.generateResult(promptTokens: mismatch,
                configuration: .init(maxNewTokens: 128, seed: 42), onToken: { _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                })
        }
        do {
            _ = try await cancelled.value
            Issue.record("Cancellation returned a completed sequence")
        } catch is CancellationError {}
        #expect(runner.prefixCacheSnapshot.reason == .cancelled && !runner.prefixCacheSnapshot.valid)
        runner.resetState()
        _ = try await measured("post-cancellation-reset", ready, defaultSample, limit: 16)
        #expect(runner.prefixCacheSnapshot.reusedPrefixTokens == 0)
        records[records.count - 1]["cancelled"] = true
        try save()
    }
}
