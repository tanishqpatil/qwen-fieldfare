import Foundation

/// Host CPU reference for Flint MoE Sigmoid Top-6 Router.
public enum FlintRouterReference {
    /// Computes CPU reference forward pass for Flint Sigmoid Top-6 Router in Float32.
    ///
    /// - Parameters:
    ///   - input: Token activation vectors [batch * numTokens * 2048]
    ///   - gateWeight: Expert gate weights [256 * 2048]
    ///   - batch: Number of batch elements
    ///   - numTokens: Number of tokens per batch
    /// - Returns: Tuple of selected expert indices [batch * numTokens * 6] (UInt32)
    ///   and normalized routing weights [batch * numTokens * 6] (Float).
    public static func evaluateForwardFP32(
        input: [Float],
        gateWeight: [Float],
        batch: Int = 1,
        numTokens: Int
    ) -> (experts: [UInt32], weights: [Float]) {
        let hiddenDim = 2048
        let numExperts = 256
        let topK = 6

        var outExperts = [UInt32](repeating: 0, count: batch * numTokens * topK)
        var outWeights = [Float](repeating: 0.0, count: batch * numTokens * topK)

        for b in 0..<batch {
            for t in 0..<numTokens {
                let tokenOffset = (b * numTokens + t) * hiddenDim
                var scores = [Float](repeating: 0.0, count: numExperts)

                // Compute sigmoid(x @ W_e^T) for each expert e
                for e in 0..<numExperts {
                    let weightOffset = e * hiddenDim
                    var acc: Float = 0.0
                    for i in 0..<hiddenDim {
                        acc += input[tokenOffset + i] * gateWeight[weightOffset + i]
                    }
                    scores[e] = 1.0 / (1.0 + exp(-acc))
                }

                // Top-6 selection with deterministic tie-breaking (lower expert ID preferred)
                var topScores = [Float](repeating: 0.0, count: topK)
                var topIndices = [UInt32](repeating: 0, count: topK)

                for k in 0..<topK {
                    var maxVal: Float = -1e30
                    var bestIdx: Int = 0
                    for e in 0..<numExperts {
                        let val = scores[e]
                        if val > maxVal {
                            maxVal = val
                            bestIdx = e
                        }
                    }
                    topScores[k] = maxVal
                    topIndices[k] = UInt32(bestIdx)
                    scores[bestIdx] = -1e30 // Exclude for next rounds
                }

                // Normalize top-6 scores
                var sum: Float = 1e-20
                for k in 0..<topK {
                    sum += topScores[k]
                }

                let outTokenOffset = (b * numTokens + t) * topK
                for k in 0..<topK {
                    outExperts[outTokenOffset + k] = topIndices[k]
                    outWeights[outTokenOffset + k] = topScores[k] / sum
                }
            }
        }

        return (outExperts, outWeights)
    }
}
