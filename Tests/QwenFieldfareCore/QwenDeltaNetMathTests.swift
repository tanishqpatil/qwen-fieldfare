import Testing
@testable import QwenFieldfareCore

@Suite struct QwenDeltaNetMathTests {
    @Test func matrixVectorMultiplyUsesRowMajorWeights() {
        let result = QwenLinearAttentionMath.matVecMul(
            weight: [1, 2, 3, 4, 5, 6],
            vec: [2, -1, 3],
            rows: 2,
            cols: 3)
        #expect(result == [9, 21])
    }

    @Test func rmsNormProducesFiniteScaledValues() {
        let result = QwenLinearAttentionMath.rmsNorm(
            input: [3, 4],
            weight: [1, 2],
            eps: 0)
        #expect(abs(result[0] - 1.6970563) < 0.00001)
        #expect(abs(result[1] - 3.3941126) < 0.00001)
    }

    @Test func zeroInputRemainsZero() {
        let result = QwenLinearAttentionMath.rmsNorm(
            input: [0, 0, 0],
            weight: [1, 1, 1])
        #expect(result == [0, 0, 0])
    }
}
