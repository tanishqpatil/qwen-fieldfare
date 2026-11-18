import Foundation

/// Host-side CPU mathematical reference for Flint RMSNorm.
/// Used for independent verification, unit tests, and cross-checking Metal GPU execution.
public enum FlintRMSNormReference {

    /// Converts a 16-bit bfloat16 raw value to a 32-bit Float.
    @inline(__always)
    public static func bf16ToFloat(_ u16: UInt16) -> Float {
        let u32 = UInt32(u16) << 16
        return Float(bitPattern: u32)
    }

    /// Converts a 32-bit Float to a 16-bit bfloat16 raw value using round-to-nearest-even (RNE).
    @inline(__always)
    public static func floatToBF16(_ f: Float) -> UInt16 {
        let u = f.bitPattern
        // Preserve NaNs
        if (u & 0x7F80_0000) == 0x7F80_0000 {
            let mantissa = u & 0x007F_FFFF
            if mantissa != 0 {
                return UInt16((u >> 16) | 0x0040)
            }
        }
        // Round to nearest even: add rounding bias
        let lsb = (u >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF + lsb
        let rounded = u &+ roundingBias
        return UInt16(rounded >> 16)
    }

    /// Converts an array of raw BF16 UInt16 values to Float array.
    public static func bf16ArrayToFloat(_ raw: [UInt16]) -> [Float] {
        raw.map { bf16ToFloat($0) }
    }

    /// Converts an array of Float values to raw BF16 UInt16 array.
    public static func floatArrayToBF16(_ floats: [Float]) -> [UInt16] {
        floats.map { floatToBF16($0) }
    }

    /// Evaluates RMSNorm on host CPU using full FP32 arithmetic.
    ///
    /// Mathematical definition:
    ///   y_i = x_i * rsqrt(mean(x^2) + eps) * w_i
    ///
    /// - Parameters:
    ///   - input: 1D or flattened 2D input array of size (rows * dimension).
    ///   - weights: 1D weight array of size dimension.
    ///   - rows: Number of independent rows (tokens or heads).
    ///   - dimension: Feature dimension (e.g. 2048 or 128).
    ///   - epsilon: Small additive constant for numerical stability (e.g. 1e-6).
    /// - Returns: Normalized output array of size (rows * dimension).
    public static func evaluateFP32(
        input: [Float],
        weights: [Float],
        rows: Int,
        dimension: Int,
        epsilon: Float
    ) -> [Float] {
        precondition(input.count == rows * dimension, "Input count (\(input.count)) != rows (\(rows)) * dimension (\(dimension))")
        precondition(weights.count == dimension, "Weight count (\(weights.count)) != dimension (\(dimension))")

        var output = [Float](repeating: 0, count: rows * dimension)
        let invD = 1.0 / Float(dimension)

        for r in 0..<rows {
            let offset = r * dimension
            var sumSq: Float = 0.0
            for i in 0..<dimension {
                let v = input[offset + i]
                sumSq += v * v
            }
            let meanSq = sumSq * invD
            let scale = 1.0 / sqrt(meanSq + epsilon)

            for i in 0..<dimension {
                output[offset + i] = input[offset + i] * scale * weights[i]
            }
        }
        return output
    }

    /// Evaluates RMSNorm on host CPU on raw BF16 inputs and weights, returning raw BF16 output.
    public static func evaluateBF16(
        input: [UInt16],
        weights: [UInt16],
        rows: Int,
        dimension: Int,
        epsilon: Float
    ) -> [UInt16] {
        let inFloats = bf16ArrayToFloat(input)
        let wFloats = bf16ArrayToFloat(weights)
        let outFloats = evaluateFP32(
            input: inFloats,
            weights: wFloats,
            rows: rows,
            dimension: dimension,
            epsilon: epsilon
        )
        return floatArrayToBF16(outFloats)
    }
}
