import Foundation

/// Host CPU reference implementation for Flint MoE Fixed Router-Order Expert Reduction.
public enum FlintReductionReference {

    /// Reduces 6 router contributions strictly in fixed order 0..5, plus optional shared expert.
    ///
    /// - Parameters:
    ///   - contributions: Flat array of shape `[batch, numTokens, 6, 2048]`.
    ///   - weights: Flat array of shape `[batch, numTokens, 6]`.
    ///   - sharedExpert: Optional flat array of shape `[batch, numTokens, 2048]`.
    ///   - batch: Number of sequences in batch.
    ///   - numTokens: Number of tokens per sequence.
    /// - Returns: Reduced output of shape `[batch, numTokens, 2048]`.
    public static func evaluateReduction(
        contributions: [Float],
        weights: [Float],
        sharedExpert: [Float]? = nil,
        batch: Int = 1,
        numTokens: Int
    ) -> [Float] {
        precondition(batch > 0 && numTokens > 0, "batch and numTokens must be positive")
        let hiddenDim = 2048
        let topK = 6

        let expectedContribCount = batch * numTokens * topK * hiddenDim
        let expectedWeightsCount = batch * numTokens * topK
        let expectedOutCount = batch * numTokens * hiddenDim

        precondition(contributions.count == expectedContribCount, "Contributions count mismatch")
        precondition(weights.count == expectedWeightsCount, "Weights count mismatch")
        if let s = sharedExpert {
            precondition(s.count == expectedOutCount, "Shared expert count mismatch")
        }

        var output = [Float](repeating: 0, count: expectedOutCount)

        for b in 0..<batch {
            for t in 0..<numTokens {
                let tokenWOffset = (b * numTokens + t) * topK
                let w0 = weights[tokenWOffset + 0]
                let w1 = weights[tokenWOffset + 1]
                let w2 = weights[tokenWOffset + 2]
                let w3 = weights[tokenWOffset + 3]
                let w4 = weights[tokenWOffset + 4]
                let w5 = weights[tokenWOffset + 5]

                let tokenCOffset = (b * numTokens + t) * (topK * hiddenDim)
                let tokenOutOffset = (b * numTokens + t) * hiddenDim

                for d in 0..<hiddenDim {
                    let c0 = contributions[tokenCOffset + 0 * hiddenDim + d]
                    let c1 = contributions[tokenCOffset + 1 * hiddenDim + d]
                    let c2 = contributions[tokenCOffset + 2 * hiddenDim + d]
                    let c3 = contributions[tokenCOffset + 3 * hiddenDim + d]
                    let c4 = contributions[tokenCOffset + 4 * hiddenDim + d]
                    let c5 = contributions[tokenCOffset + 5 * hiddenDim + d]

                    // Strict sequential FP32 accumulation in fixed router order 0, 1, 2, 3, 4, 5
                    var acc = c0 * w0
                    acc += c1 * w1
                    acc += c2 * w2
                    acc += c3 * w3
                    acc += c4 * w4
                    acc += c5 * w5

                    if let s = sharedExpert {
                        acc += s[tokenOutOffset + d]
                    }

                    output[tokenOutOffset + d] = acc
                }
            }
        }

        return output
    }
}
