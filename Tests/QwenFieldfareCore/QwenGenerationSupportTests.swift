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
