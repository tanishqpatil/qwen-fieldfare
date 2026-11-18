import Foundation

/// Host-side CPU mathematical reference for Flint Split-Half RoPE.
/// Used for independent verification and cross-checking Metal GPU execution.
public enum FlintRoPEReference {

    /// Applies split-half RoPE on FP32 input with layout `[batch, numHeads, seqLen, headDim]`.
    /// Head dimension must be 128.
    public static func evaluateFP32(
        input: [Float],
        batch: Int = 1,
        numHeads: Int,
        seqLen: Int,
        offset: Int,
        base: Float = 10000.0
    ) -> [Float] {
        let headDim = 128
        let halfDim = 64
        let totalCount = batch * numHeads * seqLen * headDim
        precondition(input.count == totalCount, "Input count (\(input.count)) != expected (\(totalCount))")

        var output = [Float](repeating: 0, count: totalCount)

        // Precompute angular frequencies for d in 0..<64: base ** (-d / 64.0)
        var freqs = [Float](repeating: 0, count: halfDim)
        for d in 0..<halfDim {
            freqs[d] = pow(base, -Float(d) / Float(halfDim))
        }

        for b in 0..<batch {
            for h in 0..<numHeads {
                for t in 0..<seqLen {
                    let rowOffset = b * (numHeads * seqLen * headDim) + h * (seqLen * headDim) + t * headDim
                    let position = Float(offset + t)

                    for d in 0..<halfDim {
                        let theta = position * freqs[d]
                        let cosT = cos(theta)
                        let sinT = sin(theta)

                        let x1 = input[rowOffset + d]
                        let x2 = input[rowOffset + d + halfDim]

                        output[rowOffset + d] = x1 * cosT - x2 * sinT
                        output[rowOffset + d + halfDim] = x2 * cosT + x1 * sinT
                    }
                }
            }
        }
        return output
    }

    /// Evaluates RoPE on raw BF16 UInt16 array.
    public static func evaluateBF16(
        input: [UInt16],
        batch: Int = 1,
        numHeads: Int,
        seqLen: Int,
        offset: Int,
        base: Float = 10000.0
    ) -> [UInt16] {
        let inFloats = FlintRMSNormReference.bf16ArrayToFloat(input)
        let outFloats = evaluateFP32(
            input: inFloats,
            batch: batch,
            numHeads: numHeads,
            seqLen: seqLen,
            offset: offset,
            base: base
        )
        return FlintRMSNormReference.floatArrayToBF16(outFloats)
    }
}
