import Foundation

/// Host CPU reference implementation for Flint Dense and Shared Expert SwiGLU MLP.
public enum FlintMLPReference {

    public struct IntermediateOutputs: Sendable {
        public let gateProj: [Float]
        public let upProj: [Float]
        public let siluGate: [Float]
        public let gatedProduct: [Float]
        public let downProj: [Float]
    }

    /// Dequantizes 4-bit MLX affine quantized weights into Float32.
    /// Packed weights: [rows, cols / 8] UInt32.
    /// Scales: [rows, cols / 64] Float32.
    /// Biases: [rows, cols / 64] Float32.
    public static func dequantize4Bit(
        packedWeights: [UInt32],
        scales: [Float],
        biases: [Float],
        rows: Int,
        cols: Int,
        groupSize: Int = 64
    ) -> [Float] {
        precondition(cols % groupSize == 0, "cols must be multiple of groupSize")
        let numGroupsPerRow = cols / groupSize
        precondition(scales.count == rows * numGroupsPerRow, "scales count mismatch")
        precondition(biases.count == rows * numGroupsPerRow, "biases count mismatch")
        precondition(packedWeights.count == rows * (cols / 8), "packedWeights count mismatch")

        var output = [Float](repeating: 0, count: rows * cols)

        for r in 0..<rows {
            let rowPackedOffset = r * (cols / 8)
            let rowMetaOffset = r * numGroupsPerRow
            let rowOutOffset = r * cols

            for c in 0..<cols {
                let g = c / groupSize
                let scale = scales[rowMetaOffset + g]
                let bias = biases[rowMetaOffset + g]

                let wordIdx = rowPackedOffset + (c / 8)
                let bitShift = (c % 8) * 4
                let intVal = Float((packedWeights[wordIdx] >> bitShift) & 0xF)

                output[rowOutOffset + c] = scale * intVal + bias
            }
        }

        return output
    }

    /// GEMM: Y = X * W^T
    /// X: [batch, numTokens, K]
    /// W: [M, K]
    /// Returns: [batch, numTokens, M]
    public static func gemm(
        x: [Float],
        w: [Float],
        batch: Int,
        numTokens: Int,
        K: Int,
        M: Int
    ) -> [Float] {
        precondition(x.count == batch * numTokens * K, "x count mismatch")
        precondition(w.count == M * K, "w count mismatch")

        var y = [Float](repeating: 0, count: batch * numTokens * M)

        for b in 0..<batch {
            for t in 0..<numTokens {
                let xOffset = (b * numTokens + t) * K
                let yOffset = (b * numTokens + t) * M

                for m in 0..<M {
                    let wOffset = m * K
                    var acc: Float = 0.0
                    for k in 0..<K {
                        acc += x[xOffset + k] * w[wOffset + k]
                    }
                    y[yOffset + m] = acc
                }
            }
        }

        return y
    }

    /// SiLU: y = x / (1 + exp(-x))
    public static func silu(_ x: [Float]) -> [Float] {
        x.map { val in
            val / (1.0 + exp(-val))
        }
    }

    /// Elementwise multiplication: y = a * b
    public static func mul(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Array lengths must match for multiplication")
        return zip(a, b).map { $0 * $1 }
    }

    /// Full SwiGLU forward pass capturing all intermediate stages.
    public static func forwardWithIntermediates(
        x: [Float],
        wGate: [Float],
        wUp: [Float],
        wDown: [Float],
        batch: Int,
        numTokens: Int,
        hiddenDim: Int = 2048,
        intermediateDim: Int
    ) -> IntermediateOutputs {
        let gateProj = gemm(x: x, w: wGate, batch: batch, numTokens: numTokens, K: hiddenDim, M: intermediateDim)
        let upProj = gemm(x: x, w: wUp, batch: batch, numTokens: numTokens, K: hiddenDim, M: intermediateDim)
        let siluGate = silu(gateProj)
        let gatedProduct = mul(siluGate, upProj)
        let downProj = gemm(x: gatedProduct, w: wDown, batch: batch, numTokens: numTokens, K: intermediateDim, M: hiddenDim)

        return IntermediateOutputs(
            gateProj: gateProj,
            upProj: upProj,
            siluGate: siluGate,
            gatedProduct: gatedProduct,
            downProj: downProj
        )
    }
}
