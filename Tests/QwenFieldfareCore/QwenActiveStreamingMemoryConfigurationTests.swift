import Testing
@testable import QwenFieldfareCore

struct QwenActiveStreamingMemoryConfigurationTests {
    @Test
    func eightKProfilesRetainExpectedSlotBudgets() throws {
        #expect(try QwenActiveStreamingMemoryConfiguration.make(budget: .threeGiB, contextLength: 8_192).expertCacheSlots == 969)
        #expect(try QwenActiveStreamingMemoryConfiguration.make(budget: .fourGiB, contextLength: 8_192).expertCacheSlots == 1_576)
        #expect(try QwenActiveStreamingMemoryConfiguration.make(budget: .fiveGiB, contextLength: 8_192).expertCacheSlots == 2_183)
    }

    @Test
    func everySupportedPlanStaysWithinItsTargetAndKeepsRouterLocks() throws {
        for budget in QwenActiveStreamingMemoryBudget.allCases {
            for context in [4_096, 8_192, 16_384, 32_768, 65_536] {
                guard let plan = try? QwenActiveStreamingMemoryConfiguration.make(
                    budget: budget,
                    contextLength: context
                ) else {
                    continue
                }
                #expect(plan.expertCacheSlots >= QwenActiveStreamingMemoryConfiguration.minimumExpertCacheSlots)
                #expect(plan.estimatedTotalBytes <= plan.targetBytes)
            }
        }
    }

    @Test
    func rejectsBudgetContextPairThatCannotKeepAllLayerLocks() throws {
        #expect(throws: QwenActiveStreamingMemoryConfigurationError.self) {
            _ = try QwenActiveStreamingMemoryConfiguration.make(budget: .threeGiB, contextLength: 65_536)
        }
    }

    @Test
    func moreContextNeverCreatesMoreSlotsAtTheSameBudget() throws {
        let short = try QwenActiveStreamingMemoryConfiguration.make(budget: .fiveGiB, contextLength: 8_192)
        let long = try QwenActiveStreamingMemoryConfiguration.make(budget: .fiveGiB, contextLength: 65_536)
        #expect(long.expertCacheSlots < short.expertCacheSlots)
        #expect(long.kvStateBytes > short.kvStateBytes)
    }
}
