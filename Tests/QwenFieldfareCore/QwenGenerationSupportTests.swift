import Testing
@testable import QwenFieldfareCore

@Suite
struct QwenGenerationSupportTests {
    @Test
    func denseArenaEndsAtFinalSharedExpertTensor() {
        #expect(DenseLayerArena.totalByteLength == DenseLayerArena.shDownBOff + 32_768)
    }

    @Test
    func disabledThinkingTemplateMatchesCheckpointContract() {
        let prompt = QwenChatTemplate.format(
            messages: [],
            appendingUser: "Explain HashMap.",
            enableThinking: false
        )

        #expect(prompt == "<|im_start|>user\nExplain HashMap.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test
    func retainedAssistantThinkingIsNotReplayed() {
        let prompt = QwenChatTemplate.format(
            messages: [
                .init(role: .user, content: "First"),
                .init(role: .assistant, content: "<think>\nprivate\n</think>\n\nVisible answer")
            ],
            appendingUser: "Second",
            enableThinking: false
        )

        #expect(!prompt.contains("private"))
        #expect(prompt.contains("<|im_start|>assistant\nVisible answer<|im_end|>"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    @Test
    func unfinishedThinkingIsNotStoredAsAssistantContent() {
        #expect(QwenChatTemplate.visibleAssistantContent("<think>\nunfinished") == "")
        #expect(QwenChatTemplate.visibleAssistantContent("Visible answer") == "Visible answer")
    }

    @Test
    func longGenerationDoesNotHaveAnImplicit512TokenLimit() throws {
        let configuration = QwenGenerationConfiguration.greedy(maxNewTokens: 1_024)
        var forwarded = 0
        let result = try QwenGenerationLoop.run(
            initialToken: 1,
            promptTokens: [100, 101],
            maxSequenceLength: 2_048,
            configuration: configuration,
            checkCancellation: {},
            onToken: { _ in },
            nextToken: { token, _, _, _ in
                forwarded += 1
                return token + 1
            }
        )

        #expect(result.tokens.count == 1_024)
        #expect(result.finishReason == .maxNewTokens)
        #expect(forwarded == 1_023)
    }

    @Test
    func contextLimitHasASeparateFinishReason() throws {
        let result = try QwenGenerationLoop.run(
            initialToken: 7,
            promptTokens: [1, 2, 3],
            maxSequenceLength: 5,
            configuration: .greedy(maxNewTokens: 10),
            checkCancellation: {},
            onToken: { _ in },
            nextToken: { token, _, _, _ in token + 1 }
        )

        #expect(result.tokens == [7, 8])
        #expect(result.finishReason == .contextLength)
    }

    @Test
    func checkpointStopTokensEndWithoutBeingEmitted() throws {
        let result = try QwenGenerationLoop.run(
            initialToken: 248_044,
            promptTokens: [1],
            maxSequenceLength: 32,
            configuration: .greedy(maxNewTokens: 10),
            checkCancellation: {},
            onToken: { _ in Issue.record("EOS must not be published") },
            nextToken: { _, _, _, _ in
                Issue.record("EOS must not run another forward step")
                return 0
            }
        )

        #expect(result.tokens.isEmpty)
        #expect(result.finishReason == .endOfSequence(tokenID: 248_044))
    }

    @Test
    func presencePenaltyChangesGreedySelection() {
        let logits: [Float] = [1, 5, 4]
        let unpenalized = QwenLogitSelector.select(
            logits: logits,
            configuration: .greedy(maxNewTokens: 1)
        )
        var penalizedConfiguration = QwenGenerationConfiguration.greedy(maxNewTokens: 1)
        penalizedConfiguration.presencePenalty = 2
        let penalized = QwenLogitSelector.select(
            logits: logits,
            seenTokenIDs: [1],
            configuration: penalizedConfiguration
        )

        #expect(unpenalized == 1)
        #expect(penalized == 2)
    }

    @Test
    func exactGreedySelectionPreservesArgmaxEdgeCases() {
        let cases: [(logits: [Float], expected: Int32)] = [
            ([-5, -1, -3], 1),
            ([2, 7, 7, 4], 1),
            ([-Float.infinity, -2, -3], 1),
            ([Float.infinity, 9, Float.infinity], 0),
            ([Float.nan, 2, 1], 0),
            ([1, Float.nan, 2], 2),
            ([Float.nan, Float.nan], 0),
        ]

        for testCase in cases {
            let selected = QwenLogitSelector.select(
                logits: testCase.logits,
                seenTokenIDs: [0, 1, 2],
                configuration: .greedy(maxNewTokens: 1)
            )
            #expect(selected == testCase.expected)
        }
    }

    @Test
    func seededSamplingIsDeterministicAndBoundedByTopK() {
        let logits: [Float] = [0, 1, 2, 3, 4]
        let configuration = QwenGenerationConfiguration(
            maxNewTokens: 1,
            temperature: 1,
            topK: 2,
            topP: 1,
            presencePenalty: 0,
            repetitionPenalty: 1,
            seed: 123
        )
        let first = QwenLogitSelector.select(
            logits: logits,
            configuration: configuration,
            samplePosition: 9
        )
        let second = QwenLogitSelector.select(
            logits: logits,
            configuration: configuration,
            samplePosition: 9
        )

        #expect(first == second)
        #expect(first == 3 || first == 4)
    }
}

extension QwenGenerationSupportTests {
    @Test func consumedSequenceAndSamplingPositionsAtEachFinishBoundary() throws {
        for end in ["eos","output","context"] {
            var consumed:[Int32]=[];var positions:[Int]=[];var draws:[Int]=[]
            let result=try QwenGenerationLoop.run(initialToken:10,promptTokens:[7,8],maxSequenceLength:end == "context" ? 3:32,configuration:.greedy(maxNewTokens:end == "output" ? 2:8),checkCancellation:{},onToken:{_ in},nextToken:{token,position,seen,draw in
                consumed.append(token);positions.append(position);draws.append(draw)
                #expect(seen == Set([Int32(7),8]+consumed))
                return token == 10 ? 11:248046
            })
            if end == "eos" {
                #expect(result.tokens == [10,11] && consumed == [10,11])
                #expect(positions == [2,3] && draws == [1,2])
                #expect(result.finishReason == .endOfSequence(tokenID:248046))
            } else if end == "output" {
                #expect(result.tokens == [10,11] && consumed == [10])
                #expect(positions == [2] && draws == [1])
                #expect(result.finishReason == .maxNewTokens)
            } else {
                #expect(result.tokens == [10] && consumed.isEmpty && positions.isEmpty)
                #expect(result.finishReason == .contextLength)
            }
        }
    }
}

extension QwenGenerationSupportTests {
    @Test
    func setSelectorReferencePreservesPenaltyTieAndNonfiniteBehavior() {
        let penalty = QwenGenerationConfiguration(
            maxNewTokens: 1, temperature: 0, topK: 1, topP: 1,
            presencePenalty: 1, repetitionPenalty: 2, seed: 42)
        let cases: [(logits: [Float], seen: Set<Int32>, expected: Int32)] = [
            ([8, 4, 2], [0], 1),
            ([-1, -2, -4], [0], 1),
            ([4, 1, 1], [0], 0),
            ([-0.0, +0.0], [-1, 2], 0),
            ([.nan, .infinity, 2], [], 0),
            ([-.infinity, .nan, .infinity], [], 2),
        ]
        for test in cases {
            #expect(QwenLogitSelector.select(logits: test.logits, seenTokenIDs: test.seen,
                configuration: penalty) == test.expected)
        }
    }
}

@Suite
struct QwenSeenTokenBitmapTests {
    private func select(
        _ logits: [Float], bitmap: QwenSeenTokenBitmap,
        configuration: QwenGenerationConfiguration, position: Int
    ) -> Int32 {
        logits.withUnsafeBufferPointer {
            QwenLogitSelector.select(
                logits: $0.baseAddress!, seenBitmap: bitmap,
                configuration: configuration, samplePosition: position)
        }
    }

    @Test
    func bitmapMatchesSetForSamplingAndBoundaryValues() {
        let edge: [Float] = [-.infinity, .infinity, .nan, 0.0, -0.0, -3, 4, 4, -3]
        let logitsCases: [[Float]] = [
            edge, [.nan, .infinity, -.infinity, 0.0],
            (0..<1_027).map { Float(($0 * 59) % 29 - 14) / 4 },
        ]
        for logits in logitsCases {
            let bound = Int32(logits.count)
            let histories: [[Int32]] = [
                [], [0], [bound - 1], [-1, .min, bound, .max, 0, 0, bound - 1],
                (0..<bound).filter { $0.isMultiple(of: 3) },
                Array(0..<bound), Array(1..<bound),
            ]
            for history in histories {
                let bitmap = QwenSeenTokenBitmap(tokenCount: logits.count, promptTokens: history)
                let set = Set(history)
                for seed: UInt64 in [0, 42, .max] {
                    let configurations: [QwenGenerationConfiguration] = [
                        .greedy(maxNewTokens: 1, seed: seed),
                        .init(maxNewTokens: 1, temperature: 0, topK: 1,
                              topP: 1, presencePenalty: 1.5, repetitionPenalty: 2, seed: seed),
                        .init(maxNewTokens: 1, temperature: 1, topK: 20,
                              topP: 0.95, presencePenalty: 1.5, repetitionPenalty: 1, seed: seed),
                        .init(maxNewTokens: 1, temperature: 0.5, topK: 20,
                              topP: .leastNormalMagnitude, presencePenalty: -1.5, repetitionPenalty: 1, seed: seed),
                        .init(maxNewTokens: 1, temperature: 2, topK: 256,
                              topP: 1, presencePenalty: 0, repetitionPenalty: 1.25, seed: seed),
                    ]
                    for configuration in configurations {
                        for position in [0, 1, 511] {
                            let oracle = QwenLogitSelector.select(
                                logits: logits, seenTokenIDs: set, configuration: configuration,
                                samplePosition: position)
                            #expect(select(logits, bitmap: bitmap, configuration: configuration, position: position) == oracle)
                        }
                    }
                }
            }
        }
    }

    @Test
    func bitmapMatchesSetAtFullCheckpointVocabulary() {
        let count = 248_320
        var logits = [Float](repeating: -4, count: count)
        for index in [0, 63, 64, 65, count / 2, count - 1] { logits[index] = 4 }
        let history = Array(Int32(0)..<Int32(count))
        let configuration = QwenGenerationConfiguration(maxNewTokens: 2, seed: 42)
        let bitmap = QwenSeenTokenBitmap(tokenCount: count, promptTokens: Array(history.dropLast()))
        var set = Set(history.dropLast())
        for addFinal in [false, true] {
            if addFinal {
                set.insert(Int32(count - 1))
                bitmap.insert(Int32(count - 1))
            }
            #expect(select(logits, bitmap: bitmap, configuration: configuration, position: 511)
                    == QwenLogitSelector.select(logits: logits, seenTokenIDs: set,
                        configuration: configuration, samplePosition: 511))
        }
    }

    @Test
    func generationPreservesFirstPredictionUpdatesAndCancellationBoundaries() throws {
        let prompt: [Int32] = [7, 2, 7]
        let logits: [Float] = [0, 0, 4, 0, 0, 0, 0, 5, 3.75]
        let policy = QwenGenerationConfiguration(maxNewTokens: 5, temperature: 0, topK: 1,
            topP: 1, presencePenalty: 2, repetitionPenalty: 1, seed: 42)
        let cases: [(maxNew: Int, context: Int, cancel: Bool)] = [
            (5, 32, false), (1, 32, false), (5, 4, false),
            (5, 32, true), (5, 32, false),
        ]
        for test in cases {
            var configuration = policy
            configuration.maxNewTokens = test.maxNew
            let bitmap = QwenSeenTokenBitmap(tokenCount: logits.count, promptTokens: prompt)
            let first = select(logits, bitmap: bitmap, configuration: configuration, position: 0)
            #expect(first == QwenLogitSelector.select(logits: logits, seenTokenIDs: Set(prompt),
                configuration: configuration, samplePosition: 0))
            #expect(first == 8)
            var bitmapEvents: [String] = []
            var setEvents: [String] = []
            func checkCancellation(_ events: [String]) throws {
                if test.cancel && !events.isEmpty { throw CancellationError() }
            }
            let candidate = Result {
                try QwenGenerationLoop.run(
                    initialToken: first, promptTokenCount: prompt.count, initialSeen: bitmap,
                    maxSequenceLength: test.context, configuration: configuration,
                    checkCancellation: { try checkCancellation(bitmapEvents) },
                    onToken: { bitmapEvents.append("callback \($0)") },
                    markSeen: { $0.insert($1) },
                    nextToken: { token, position, seen, sample in
                        let prediction = select(logits, bitmap: seen, configuration: configuration, position: sample)
                        bitmapEvents.append("forward \(token) \(position) \(sample) \(prediction)")
                        return sample == 2 ? 248_046 : prediction
                    })
            }
            let oracle = Result {
                try QwenGenerationLoop.run(
                    initialToken: first, promptTokens: prompt,
                    maxSequenceLength: test.context, configuration: configuration,
                    checkCancellation: { try checkCancellation(setEvents) },
                    onToken: { setEvents.append("callback \($0)") },
                    nextToken: { token, position, seen, sample in
                        let prediction = QwenLogitSelector.select(
                            logits: logits, seenTokenIDs: seen, configuration: configuration, samplePosition: sample)
                        setEvents.append("forward \(token) \(position) \(sample) \(prediction)")
                        return sample == 2 ? 248_046 : prediction
                    })
            }
            #expect(bitmapEvents == setEvents)
            if test.cancel {
                #expect(throws: CancellationError.self) { try candidate.get() }
                #expect(throws: CancellationError.self) { try oracle.get() }
            } else {
                #expect(try candidate.get() == oracle.get())
            }
        }
    }
}
