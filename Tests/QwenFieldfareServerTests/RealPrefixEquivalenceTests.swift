import Foundation
import CryptoKit
import Metal
import Testing
@testable import QwenFieldfareCore
@testable import QwenFieldfareServerSupport

private final class PrefixBoundaryCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var index = 0
    private var boundaries: [Int: [String: String]] = [:]
    static let positions: Set<Int> = [0, 216, 421, 464, 465, 466, 511]

    func record(_ runner: QwenQ4ActiveExpertStreamingRunner, promptCount: Int) {
        lock.withLock {
            defer { index += 1 }
            guard Self.positions.contains(index) else { return }
            let boundary = runner.debugSequenceBoundary()
            boundaries[index] = [
                "absolute_position": String(promptCount - 1 + index),
                "hidden_sha256": Self.digest(boundary.hidden),
                "logits_sha256": Self.digest(boundary.logits)
            ]
        }
    }

    var snapshot: [Int: [String: String]] { lock.withLock { boundaries } }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

@Suite(.serialized) struct RealPrefixEquivalenceTests {
    @Test(.enabled(if:ProcessInfo.processInfo.environment["QWENFIELDFARE_PREFIX_CHECKPOINT"] != nil))
    func realGreedyReplayAndPrefixTokensAgree() async throws {
        let path=try QwenModelDirectory.validate(path:try #require(ProcessInfo.processInfo.environment["QWENFIELDFARE_PREFIX_CHECKPOINT"]))
        let output=URL(fileURLWithPath:ProcessInfo.processInfo.environment["QWENFIELDFARE_PREFIX_OUTPUT"] ?? "/private/tmp/qwen-prefix-equivalence.json")
        let rounds=Int(ProcessInfo.processInfo.environment["QWENFIELDFARE_PREFIX_ROUNDS"] ?? "1") ?? 1
        guard (1...5).contains(rounds) else { throw APIError(500, "Invalid short-cycle round count") }
        let file=URL(fileURLWithPath:FileManager.default.currentDirectoryPath).appendingPathComponent("docs/evidence/2026-11-18-s300/reconstructed-tool-requests.json")
        // Parse ordered JSON directly: schema member ordering participates in the
        // checkpoint template. JSONSerialization/`wire` sorts keys and would
        // silently measure different prompt tokens.
        guard case .array(let corpus)=try ToolJSON.parse(String(contentsOf:file,encoding:.utf8)) else {throw APIError(500,"Invalid baseline corpus")}
        let fixtures=corpus.filter {$0["label"]?.string?.hasPrefix("tool-") == true}
        #expect(fixtures.count == 4)
        let template=try String(contentsOf:URL(fileURLWithPath:path).appendingPathComponent("chat_template.jinja"),encoding:.utf8)
        let renderer=try ToolPrompt(source:template)
        let tokenizer=try await QwenTokenizer.load(from:URL(fileURLWithPath:path))
        var prompts:[[Int32]]=[];var tools:[FunctionTool]=[]
        for (step,fixture) in fixtures.enumerated() {
            // The original Python recorder aliased the growing message list.
            // Preserve original evidence; reconstruct each actual sent turn by
            // its known two-message boundary and verify observed prompt counts.
            guard case .object(var members)=fixture["request"],case .array(let messages)=fixture["request"]?["messages"] else {throw APIError(500,"Invalid recorded tool request")}
            members=members.map {$0.key == "messages" ? .init($0.key,.array(Array(messages.prefix(2+step*2)))):$0}
            let conversation=try ToolConversation(.object(members))
            if tools.isEmpty {tools=conversation.tools}
            prompts.append(tokenizer.encode(try renderer.render(conversation)))
        }
        guard prompts.map(\.count) == [1141,1255,1369,1482] else {throw APIError(500,"Reconstructed baseline token counts differ")}
        let plan=try QwenActiveStreamingMemoryConfiguration.make(budget:.fiveGiB,contextLength:8192)
        let loaded=ProcessInfo.processInfo.systemUptime
        let runner=try QwenQ4ActiveExpertStreamingRunner(modelPath:path,maxSeqLen:8192,expertCacheCapacity:plan.expertCacheSlots)
        var records:[[String:Any]]=[]
        var failure: [String: Any]?
        var sustainedBoundaries: [String: [Int: [String: String]]] = [:]
        let loadSeconds=ProcessInfo.processInfo.systemUptime-loaded
        func save() throws {
            var value:[String:Any]=["kind":"same Swift runner deterministic raw-token equivalence; no HTTP/tool execution", "load_seconds":loadSeconds,"cache_slots":plan.expertCacheSlots,"rounds":rounds,"records":records]
            if let failure { value["first_difference"] = failure }
            try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:output,options:.atomic)
        }
        func measured(_ tokens:[Int32],mode:QwenPrefixCacheMode,round:Int,step:Int,limit:Int=128,workload:String="tool-cycle") async throws -> QwenGenerationResult {
            let timing=RequestTimingRecorder();let before=runner.expertCache.metricsSnapshot()
            let capture = workload == "sustained-512" && limit == 512 ? PrefixBoundaryCapture() : nil
            let result=try await runner.generateResult(
                promptTokens:tokens,configuration:.greedy(maxNewTokens:limit),
                prefixIdentity:"fixed-tool-fixture-v1",onPrefillComplete:{timing.prefill($0)},onToken:{_ in
                    timing.token()
                    capture?.record(runner, promptCount: tokens.count)
                }
            )
            if let capture { sustainedBoundaries[mode.rawValue] = capture.snapshot }
            timing.generatedOutput(count:result.tokens.count)
            let parser=NativeToolParser(tools:tools,emit:{_ in})
            try parser.append(tokenizer.decode(result.tokens))
            let parsed=try parser.finish(endedNormally:result.finishReason == .endOfSequence(tokenID:248046) || result.finishReason == .endOfSequence(tokenID:248044))
            if workload != "tool-mismatch" && step < 3 {
                guard parsed.calls.count == 1, parsed.calls.first?.name == "audit_chunk", parsed.calls.first?.arguments["index"] == .number(String(step+1)) else {throw APIError(500,"Expected native sequential tool fixture output")}
            } else if workload != "tool-mismatch" {
                #expect(parsed.calls.isEmpty)
                if workload == "tool-cycle" { #expect(parsed.content.trimmingCharacters(in: .whitespacesAndNewlines) == "DONE") }
            }
            runner.commitPrefixCache()
            let after=runner.expertCache.metricsSnapshot();let state=runner.prefixCacheSnapshot
            #expect(state.reusedPrefixTokens + state.processedPromptTokens == tokens.count)
            if mode == .exactPrefix && workload == "tool-cycle" {
                #expect(state.reusedPrefixTokens == [0, 1166, 1280, 1394][step])
            }
            if let capture {
                #expect(result.tokens.count == 512 && result.finishReason == .maxNewTokens)
                #expect(Set(capture.snapshot.keys) == PrefixBoundaryCapture.positions)
                #expect(state.reusedPrefixTokens == (mode == .exactPrefix ? 399 : 0))
                #expect(state.pendingOutputToken && !state.valid)
            }
            if workload == "tool-mismatch" { #expect(state.reusedPrefixTokens == 0) }
            let digest=SHA256.hash(data:try JSONEncoder().encode(result.tokens)).map {String(format:"%02x",$0)}.joined()
            let metrics=try JSONSerialization.jsonObject(with:JSONEncoder().encode(timing.finish("completed")))
            let cache=try JSONSerialization.jsonObject(with:JSONEncoder().encode(state))
            records.append(["workload":workload,"round":round,"mode":mode.rawValue,"step":step,"prompt_tokens":tokens.count,"output_tokens":result.tokens.count,"output_sha256":digest,"finish":result.finishReason.label,"timing":metrics,"cache":cache,"expert_hits":after.hits-before.hits,"expert_misses":after.misses-before.misses,"expert_fetched_bytes":after.fetchedBytes-before.fetchedBytes,"metal_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0,"footprint_bytes":BackendDiagnostics.currentPhysicalFootprint ?? 0])
            if let capture { records[records.count - 1]["boundaries"] = capture.snapshot.mapKeysAsStrings() }
            try save()
            return result
        }
        func compare(_ modes: [String: [QwenGenerationResult]], workload: String, round: Int) throws {
            let replay = try #require(modes["off"])
            let reused = try #require(modes["exact-prefix"])
            #expect(replay.count == reused.count)
            for (step, pair) in zip(replay, reused).enumerated() {
                guard pair.0 != pair.1 else { continue }
                let count = min(pair.0.tokens.count, pair.1.tokens.count)
                let first = (0..<count).first { pair.0.tokens[$0] != pair.1.tokens[$0] } ?? count
                var difference: [String: Any] = ["workload": workload, "round": round, "step": step,
                     "output_index": first, "replay_count": pair.0.tokens.count, "reuse_count": pair.1.tokens.count,
                     "replay_finish": pair.0.finishReason.label, "reuse_finish": pair.1.finishReason.label]
                if first < count {
                    difference["replay_token"] = pair.0.tokens[first]
                    difference["reuse_token"] = pair.1.tokens[first]
                }
                failure = difference
                try save()
                throw APIError(500, "Cached and replay outputs differ at \(workload) step \(step), token \(first)")
            }
        }
        let warm=tokenizer.encode(QwenChatTemplate.format(messages:[],appendingUser:"Reply with exactly: ready",enableThinking:false))
        // Start with the sustained gate that rejected the prior candidate. The
        // same real call/result is rendered by the production tool template;
        // reaching 512 is observed, never forced by changing EOS behavior.
        let user="Call audit_chunk with index 1 exactly once. After its result, write a complete Python implementation of a bounded LRU cache with type annotations, detailed docstrings, and at least 200 lines of example-based tests. Do not summarize; provide the complete module."
        let firstRoot=ToolJSON.object([.init("messages",.array([.object([.init("role",.string("user")),.init("content",.string(user))])])),.init("tools",.array(tools.map(\.json)))])
        let initial=tokenizer.encode(try renderer.render(ToolConversation(firstRoot)))
        guard initial.count == 355 else { throw APIError(500, "Sustained setup prompt count changed") }
        do {
            let round = 0
            var results:[String:[QwenGenerationResult]]=[:]
            var previousPrompt:[Int32]?
            let order:[QwenPrefixCacheMode]=[.off,.exactPrefix]
            for mode in order {
                runner.setPrefixCacheMode(.off);runner.resetState();runner.expertCache.clear()
                _=try await runner.generateResult(promptTokens:warm,configuration:.greedy(maxNewTokens:8))
                runner.setPrefixCacheMode(mode)
                let call=try await measured(initial,mode:mode,round:round,step:0,workload:"sustained-512")
                guard call.tokens.count == 44 else { throw APIError(500, "Sustained setup output count changed") }
                let parser=NativeToolParser(tools:tools);try parser.append(tokenizer.decode(call.tokens))
                let parsed=try parser.finish(endedNormally:true)
                let native=try #require(parsed.calls.first)
                let assistant=ToolJSON.object([.init("role",.string("assistant")),.init("content",.string(parsed.content)),.init("tool_calls",.array([.object([.init("id",.string("sustained-call")),.init("type",.string("function")),.init("function",.object([.init("name",.string(native.name)),.init("arguments",.string(native.arguments.encoded))]))])]))])
                let result=ToolJSON.object([.init("role",.string("tool")),.init("tool_call_id",.string("sustained-call")),.init("content",.string("Chunk 1 reviewed; proceed with the complete requested Python module."))])
                let root=ToolJSON.object([.init("messages",.array([firstRoot["messages"]!.array![0],assistant,result])),.init("tools",.array(tools.map(\.json)))])
                let prompt=tokenizer.encode(try renderer.render(ToolConversation(root)))
                guard prompt.count == 430 else { throw APIError(500, "Sustained continuation prompt count changed") }
                if let previousPrompt {guard previousPrompt == prompt else {throw APIError(500,"Sustained inputs differ")}}
                previousPrompt=prompt
                let continuation=try await measured(prompt,mode:mode,round:round,step:3,limit:512,workload:"sustained-512")
                results[mode.rawValue]=[call,continuation]
            }
            try compare(results, workload: "sustained-512", round: round)
            for position in PrefixBoundaryCapture.positions.sorted() {
                let off = sustainedBoundaries["off"]?[position]
                let reused = sustainedBoundaries["exact-prefix"]?[position]
                if off != reused || off == nil {
                    failure = ["workload": "sustained-512", "output_index": position,
                               "replay": off ?? [:], "reuse": reused ?? [:]]
                    try save()
                    throw APIError(500, "Sustained hidden or logit bytes diverged at output \(position)")
                }
            }
        }
        for round in 0..<rounds {
            var results:[String:[QwenGenerationResult]]=[:]
            let order:[QwenPrefixCacheMode]=round.isMultiple(of:2) ? [.off,.exactPrefix]:[.exactPrefix,.off]
            for mode in order {
                // Identical empty expert cache plus fixed retained-runner warmup
                // at each sequence start; never change the SLRU policy itself.
                runner.setPrefixCacheMode(.off);runner.resetState();runner.expertCache.clear()
                _=try await runner.generateResult(promptTokens:warm,configuration:.greedy(maxNewTokens:8))
                runner.setPrefixCacheMode(mode)
                var values:[QwenGenerationResult]=[]
                for step in prompts.indices {values.append(try await measured(prompts[step],mode:mode,round:round,step:step))}
                results[mode.rawValue]=values
            }
            try compare(results, workload: "tool-cycle", round: round)
        }
        runner.invalidatePrefixCache(.explicitClear)
        #expect(!runner.prefixCacheSnapshot.valid)
        try save()
    }
}


private extension Dictionary where Key == Int {
    func mapKeysAsStrings() -> [String: Value] {
        Dictionary<String, Value>(uniqueKeysWithValues: map { (String($0.key), $0.value) })
    }
}
