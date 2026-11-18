import Foundation

/// Host CPU reference implementation for Flint 3-bit and 4-bit affine dequantization.
public enum FlintDequantizeReference {

    /// Dequantizes a 4-bit packed matrix (group size 64) with BF16 scales and biases to FP32.
    public static func evaluate4Bit(
        packedWeights: [UInt32],
        scales: [Float],
        biases: [Float],
        rows: Int,
        cols: Int
    ) -> [Float] {
        precondition(rows > 0 && cols > 0, "rows and cols must be positive")
        precondition(cols % 64 == 0, "cols must be a multiple of group size (64)")

        let numGroups = cols / 64
        let wordsPerRow = numGroups * 8
        precondition(packedWeights.count == rows * wordsPerRow, "Packed weights count mismatch")
        precondition(scales.count == rows * numGroups, "Scales count mismatch")
        precondition(biases.count == rows * numGroups, "Biases count mismatch")

        var output = [Float](repeating: 0, count: rows * cols)

        for r in 0..<rows {
            for g in 0..<numGroups {
                let s = scales[r * numGroups + g]
                let b = biases[r * numGroups + g]

                for j in 0..<64 {
                    let wordIdx = j / 8
                    let bitOffset = (j % 8) * 4
                    let word = packedWeights[r * wordsPerRow + g * 8 + wordIdx]
                    let intVal = (word >> bitOffset) & 0xF
                    let val = s * Float(intVal) + b
                    output[r * cols + g * 64 + j] = val
                }
            }
        }

        return output
    }

    /// Dequantizes a 3-bit packed matrix (group size 64) with continuous bit-packing across word boundaries.
    public static func evaluate3Bit(
        packedWeights: [UInt32],
        scales: [Float],
        biases: [Float],
        rows: Int,
        cols: Int
    ) -> [Float] {
        precondition(rows > 0 && cols > 0, "rows and cols must be positive")
        precondition(cols % 64 == 0, "cols must be a multiple of group size (64)")

        let numGroups = cols / 64
        let wordsPerRow = numGroups * 6
        precondition(packedWeights.count == rows * wordsPerRow, "Packed weights count mismatch")
        precondition(scales.count == rows * numGroups, "Scales count mismatch")
        precondition(biases.count == rows * numGroups, "Biases count mismatch")

        var output = [Float](repeating: 0, count: rows * cols)

        for r in 0..<rows {
            for g in 0..<numGroups {
                let s = scales[r * numGroups + g]
                let b = biases[r * numGroups + g]

                let groupWords = Array(packedWeights[(r * wordsPerRow + g * 6)..<(r * wordsPerRow + g * 6 + 6)])

                for j in 0..<64 {
                    let bitIdx = j * 3
                    let w0 = bitIdx / 32
                    let b0 = bitIdx % 32

                    let wLo = UInt64(groupWords[w0])
                    let wHi = (w0 + 1 < 6) ? UInt64(groupWords[w0 + 1]) : 0
                    let combined = (wHi << 32) | wLo
                    let intVal = (combined >> b0) & 0x7
                    let val = s * Float(intVal) + b
                    output[r * cols + g * 64 + j] = val
                }
            }
        }

        return output
    }
}
