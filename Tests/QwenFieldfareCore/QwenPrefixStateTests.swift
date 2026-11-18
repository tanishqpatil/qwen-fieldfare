import Testing
@testable import QwenFieldfareCore

struct QwenPrefixStateTests {
    func committed(_ state: QwenPrefixState) throws {
        state.setMode(.exactPrefix)
        #expect(try state.begin(prompt:[1,2],identity:"tools-a") == 0)
        try state.consume(1,position:0,prompt:true);try state.consume(2,position:1,prompt:true)
        try state.consume(3,position:2,prompt:false)
        try state.finish(promptCount:2,result:.init(tokens:[3],finishReason:.endOfSequence(tokenID:248046)))
        #expect(!state.snapshot.valid)
        state.commit();#expect(state.snapshot.valid && state.snapshot.consumedTokens == 3)
    }
    @Test func strictPrefixInPlaceAndNoUnsupportedRewind() throws {
        let state=QwenPrefixState(maximum:16);try committed(state)
        #expect(try state.begin(prompt:[1,2,3,4],identity:"tools-a") == 3)
        try state.consume(4,position:3,prompt:true)
        try state.finish(promptCount:4,result:.init(tokens:[],finishReason:.endOfSequence(tokenID:248046)))
        state.commit();#expect(state.snapshot.consumedTokens == 4 && state.snapshot.processedPromptTokens == 1)
        #expect(try state.begin(prompt:[1,2,8,9],identity:"tools-a") == 0)
        #expect(state.snapshot.reason == .prefixMismatch)
        state.abort(.cancelled)
    }
    @Test func emptySuffixAheadAndSchemaChangesReplay() throws {
        for (prompt,identity,reason) in [([Int32(1),2,3],"tools-a",QwenPrefixInvalidation.emptySuffix),([1,2],"tools-a",.stateAhead),([1,2,3,4],"tools-b",.incompatibleIdentity)] {
            let state=QwenPrefixState(maximum:16);try committed(state)
            #expect(try state.begin(prompt:prompt,identity:identity) == 0)
            #expect(state.snapshot.reason == reason)
            state.abort(.inferenceError)
        }
    }
    @Test func disabledLimitAndClearCannotPublishPendingState() throws {
        let off=QwenPrefixState(maximum:8)
        #expect(try off.begin(prompt:[1],identity:"a") == 0)
        try off.consume(1,position:0,prompt:true)
        try off.finish(promptCount:1,result:.init(tokens:[7],finishReason:.maxNewTokens))
        off.commit();#expect(!off.snapshot.valid && off.snapshot.tokenStorageBytes == 0)
        let state=QwenPrefixState(maximum:16);try committed(state)
        #expect(try state.begin(prompt:[1,2,3,4],identity:"tools-a") == 3)
        state.clear(.explicitClear)
        try state.consume(4,position:3,prompt:true)
        try state.finish(promptCount:4,result:.init(tokens:[],finishReason:.endOfSequence(tokenID:248046)))
        state.commit();#expect(!state.snapshot.valid && state.snapshot.reason == .explicitClear)
        #expect(try state.begin(prompt:[1,2,3,4,5],identity:"tools-a") == 0)
        state.abort(.cancelled)
    }
    @Test func incorrectPositionsAndOversizedInputsAreRecoverable() throws {
        let state=QwenPrefixState(maximum:4)
        #expect(throws:QwenPrefixStateError.self) {try state.begin(prompt:[1,2,3,4],identity:"x")}
        _=try state.begin(prompt:[1],identity:"x")
        #expect(throws:QwenPrefixStateError.self) {try state.consume(1,position:1,prompt:true)}
        state.abort(.invalidBoundary);#expect(!state.snapshot.valid)
    }
    @Test func failedOrInvalidatedTransactionsCannotBeCommitted() throws {
        for reason in [QwenPrefixInvalidation.cancelled,.inferenceError,.outputValidationFailed,.shutdown,.rejectedRequest] {
            let state=QwenPrefixState(maximum:16);try committed(state)
            _=try state.begin(prompt:[1,2,3,4],identity:"tools-a")
            try state.consume(4,position:3,prompt:true)
            state.abort(reason);state.commit()
            #expect(!state.snapshot.valid && state.snapshot.reason == reason && state.snapshot.tokenStorageBytes == 0)
        }
        let state=QwenPrefixState(maximum:16);try committed(state)
        state.clear(.explicitClear);state.commit()
        #expect(!state.snapshot.valid && state.snapshot.tokenStorageBytes == 0)
        // A new runner/session never inherits another runner's state.
        let reloaded=QwenPrefixState(maximum:16)
        #expect(reloaded.snapshot.session != state.snapshot.session && !reloaded.snapshot.valid)
    }
    @Test func pendingFinalTokensAndModeChangesNeverPublish() throws {
        for finish in [QwenGenerationFinishReason.maxNewTokens,.contextLength] {
            let state=QwenPrefixState(maximum:16);state.setMode(.exactPrefix)
            _=try state.begin(prompt:[1,2],identity:"a")
            try state.consume(1,position:0,prompt:true);try state.consume(2,position:1,prompt:true)
            try state.consume(3,position:2,prompt:false)
            try state.finish(promptCount:2,result:.init(tokens:[3,4],finishReason:finish));state.commit()
            #expect(!state.snapshot.valid && state.snapshot.reason == .outputIncomplete && state.snapshot.pendingOutputToken)
        }
        let state=QwenPrefixState(maximum:16);try committed(state)
        _=try state.begin(prompt:[1,2,3,4],identity:"tools-a")
        state.setMode(.off);state.setMode(.exactPrefix)
        try state.consume(4,position:3,prompt:true)
        try state.finish(promptCount:4,result:.init(tokens:[],finishReason:.endOfSequence(tokenID:248046)))
        state.commit();#expect(!state.snapshot.valid)
    }
    @Test func clearBeforePrefillRejectsStaleRequestEpoch() throws {
        let state = QwenPrefixState(maximum: 4)
        try committed(state)
        let staleEpoch = state.currentEpoch
        state.clear(.explicitClear)
        #expect(try state.begin(prompt: [1,2,3], identity: "tools-a", expectedEpoch: staleEpoch) == 0)
        for (position, token) in [Int32(1), 2, 3].enumerated() {
            try state.consume(token, position: position, prompt: true)
        }
        try state.finish(promptCount: 3, result: .init(tokens: [], finishReason: .endOfSequence(tokenID: 248046)))
        state.commit()
        #expect(!state.snapshot.valid && state.snapshot.fullReplays == 2 && state.snapshot.maximumTokens == 4)
        #expect(state.snapshot.reason == .explicitClear && !state.snapshot.pendingOutputToken)
    }
    @Test func actualSamplingConfigurationMustMatchButOutputLimitMayChange() throws {
        let state = QwenPrefixState(maximum: 8)
        state.setMode(.exactPrefix)
        _ = try state.begin(prompt: [1], identity: "tools", configuration: .greedy(maxNewTokens: 2))
        try state.consume(1, position: 0, prompt: true)
        try state.finish(promptCount: 1, result: .init(tokens: [], finishReason: .endOfSequence(tokenID: 248046)))
        state.commit()
        #expect(try state.begin(prompt: [1,2], identity: "tools", configuration: .greedy(maxNewTokens: 3)) == 1)
        try state.consume(2, position: 1, prompt: true)
        try state.finish(promptCount: 2, result: .init(tokens: [], finishReason: .endOfSequence(tokenID: 248046)))
        state.commit()
        var changed = QwenGenerationConfiguration.greedy(maxNewTokens: 3)
        changed.temperature = 0.5
        #expect(try state.begin(prompt: [1,2,3], identity: "tools", configuration: changed) == 0)
        #expect(state.snapshot.reason == .incompatibleIdentity && state.snapshot.fullReplays == 2)
        state.abort(.cancelled)
    }
}
