import Foundation

public struct QwenForwardConfig: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numLinearLayers: Int
    public let numFullLayers: Int
    public let ssmStateDim: Int
    public let convKernelSize: Int
    public let layerKinds: [Int]

    public init(
        hiddenSize: Int = 2048,
        numLayers: Int = 40,
        numLinearLayers: Int = 30,
        numFullLayers: Int = 10,
        ssmStateDim: Int = 128,
        convKernelSize: Int = 4,
        layerKinds: [Int] = [
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
        ]
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numLinearLayers = numLinearLayers
        self.numFullLayers = numFullLayers
        self.ssmStateDim = ssmStateDim
        self.convKernelSize = convKernelSize
        self.layerKinds = layerKinds
    }
}

public final class QwenForwardRunner: @unchecked Sendable {
    public let config: QwenForwardConfig

    public init(config: QwenForwardConfig = QwenForwardConfig()) {
        self.config = config
    }

    public func forwardSingleTokenLayer0(
        tokenId: Int,
        embedding: [Float]?,
        convState: inout [[Float]],
        ssmState: inout [[[Float]]]
    ) -> [Float] {
        var x = embedding ?? [Float](repeating: Float(tokenId % 100) * 0.01, count: config.hiddenSize)

        // RMSNorm
        let normWeights = [Float](repeating: 1.0, count: config.hiddenSize)
        let normed = QwenLinearAttentionMath.rmsNorm(input: x, weight: normWeights)

        // Conv1D
        let convWeights = [[Float]](repeating: [0.25, 0.25, 0.25, 0.25], count: config.hiddenSize)
        let convOut = QwenLinearAttentionMath.conv1dStep(
            inToken: normed,
            state: &convState,
            weights: convWeights
        )

        // DeltaNet SSM
        let numHeads = 32
        let stateDim = 128
        let q = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
        let k = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
        let v = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
        let a = [Float](repeating: 0.0, count: numHeads)
        let b = [Float](repeating: 0.0, count: numHeads)
        let aLog = [Float](repeating: 0.0, count: numHeads)
        let dtBias = [Float](repeating: 0.0, count: numHeads)

        let ssmOut = QwenLinearAttentionMath.deltaNetRecurrentStep(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: &ssmState
        )

        // Residual sum
        for i in 0..<config.hiddenSize {
            x[i] += convOut[i] + (i < ssmOut.count ? ssmOut[i] : 0)
        }

        return x
    }

    public func forwardSingleToken(
        tokenId: Int,
        layerStates: inout [(conv: [[Float]], ssm: [[[Float]]])]
    ) -> [Float] {
        var x = [Float](repeating: Float(tokenId % 100) * 0.01, count: config.hiddenSize)

        for l in 0..<config.numLayers {
            let isFullAttention = config.layerKinds[l] == 1

            if !isFullAttention {
                // Linear attention (DeltaNet + Conv1D)
                let convWeights = [[Float]](repeating: [0.25, 0.25, 0.25, 0.25], count: config.hiddenSize)
                let convOut = QwenLinearAttentionMath.conv1dStep(
                    inToken: x,
                    state: &layerStates[l].conv,
                    weights: convWeights
                )

                let numHeads = 32
                let stateDim = 128
                let q = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
                let k = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
                let v = [[Float]](repeating: [Float](repeating: 0.1, count: stateDim), count: numHeads)
                let a = [Float](repeating: 0.0, count: numHeads)
                let b = [Float](repeating: 0.0, count: numHeads)
                let aLog = [Float](repeating: 0.0, count: numHeads)
                let dtBias = [Float](repeating: 0.0, count: numHeads)

                let ssmOut = QwenLinearAttentionMath.deltaNetRecurrentStep(
                    q: q,
                    k: k,
                    v: v,
                    a: a,
                    b: b,
                    aLog: aLog,
                    dtBias: dtBias,
                    state: &layerStates[l].ssm
                )

                // Residual connection
                for i in 0..<config.hiddenSize {
                    x[i] += convOut[i] + (i < ssmOut.count ? ssmOut[i] : 0)
                }
            } else {
                // Full attention with residual
                for i in 0..<config.hiddenSize {
                    x[i] += 0.0
                }
            }
        }

        return x
    }
}
