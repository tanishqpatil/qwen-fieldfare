import Foundation

/// Host-side CPU mathematical reference for Flint Attention (GQA, Attention Sinks, RoPE, and SDPA).
/// Used for independent verification and cross-checking Metal GPU execution.
public enum FlintAttentionReference {

    /// Evaluates SDPA with Grouped-Query Attention and Learned Attention Sinks on host CPU in FP32.
    ///
    /// - Parameters:
    ///   - queries: Shape `[batch, 32, T_q, 128]`
    ///   - keys: Shape `[batch, 4, T_kv, 128]`
    ///   - values: Shape `[batch, 4, T_kv, 128]`
    ///   - sinks: Shape `[32]` (per-head learned scalar sink)
    ///   - batch: Batch size (typically 1)
    ///   - T_q: Number of queries
    ///   - T_kv: Number of physically retained keys/values
    ///   - cacheOffset: Absolute starting position of queries
    ///   - retainedKVStart: Absolute starting position of physical KV index 0
    ///   - isSliding: True if sliding window (513 tokens), false if full causal attention
    ///   - scale: Attention scaling factor (1.0 / sqrt(128.0))
    /// - Returns: Output array of shape `[batch, 32, T_q, 128]`
    public static func evaluatePrefillFP32(
        queries: [Float],
        keys: [Float],
        values: [Float],
        sinks: [Float],
        batch: Int = 1,
        T_q: Int,
        T_kv: Int,
        cacheOffset: Int = 0,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        scale: Float = 1.0 / sqrt(128.0)
    ) -> [Float] {
        let headDim = 128
        let numQHeads = 32
        let numKVHeads = 4
        let groupSize = numQHeads / numKVHeads // 8

        let expectedQCount = batch * numQHeads * T_q * headDim
        let expectedKVCount = batch * numKVHeads * T_kv * headDim
        precondition(queries.count == expectedQCount, "Query count mismatch")
        precondition(keys.count == expectedKVCount, "Key count mismatch")
        precondition(values.count == expectedKVCount, "Value count mismatch")
        precondition(sinks.count == numQHeads, "Sinks count mismatch")

        var output = [Float](repeating: 0, count: expectedQCount)

        for b in 0..<batch {
            for h in 0..<numQHeads {
                let h_kv = h / groupSize
                let s_h = sinks[h]

                let qHeadOffset = b * (numQHeads * T_q * headDim) + h * (T_q * headDim)
                let kvHeadOffset = b * (numKVHeads * T_kv * headDim) + h_kv * (T_kv * headDim)

                for t in 0..<T_q {
                    let qOffset = qHeadOffset + t * headDim
                    let p = cacheOffset + t

                    let kStartAbs = (isSliding && p >= 512) ? (p - 512) : 0
                    let kEndAbs = p
                    let kvEndAbs = retainedKVStart + T_kv - 1

                    let effStart = max(kStartAbs, retainedKVStart)
                    let effEnd = min(kEndAbs, kvEndAbs)

                    var physStart = 0
                    var physEnd = 0
                    if effStart <= effEnd && effStart >= retainedKVStart {
                        physStart = effStart - retainedKVStart
                        physEnd = effEnd - retainedKVStart + 1
                    }

                    // Calculate dot product scores for all attended physical keys
                    var scores = [Float]()
                    var validKeys = [Int]()

                    if physStart < physEnd {
                        scores.reserveCapacity(physEnd - physStart)
                        validKeys.reserveCapacity(physEnd - physStart)

                        for j in physStart..<physEnd {
                            let kOffset = kvHeadOffset + j * headDim
                            var dot: Float = 0.0
                            for d in 0..<headDim {
                                dot += queries[qOffset + d] * keys[kOffset + d]
                            }
                            scores.append(dot * scale)
                            validKeys.append(j)
                        }
                    }

                    // Softmax with learned sink s_h
                    var maxScore = s_h
                    for s in scores {
                        if s > maxScore { maxScore = s }
                    }

                    var expSum: Float = exp(s_h - maxScore) // Sink contributes exp(s_h - m)
                    var expScores = [Float](repeating: 0, count: scores.count)
                    for (idx, s) in scores.enumerated() {
                        let e = exp(s - maxScore)
                        expScores[idx] = e
                        expSum += e
                    }

                    // Accumulate weighted values
                    let outOffset = qOffset
                    for d in 0..<headDim {
                        var acc: Float = 0.0
                        for (idx, j) in validKeys.enumerated() {
                            let vOffset = kvHeadOffset + j * headDim
                            let prob = expScores[idx] / expSum
                            acc += prob * values[vOffset + d]
                        }
                        output[outOffset + d] = acc
                    }
                }
            }
        }
        return output
    }

    /// Evaluates single-token decode attention on host CPU in FP32 with bounded-KV indexing.
    public static func evaluateDecodeFP32(
        queries: [Float],
        keys: [Float],
        values: [Float],
        sinks: [Float],
        batch: Int = 1,
        T_kv: Int,
        queryAbsPos: Int? = nil,
        retainedKVStart: Int = 0,
        isSliding: Bool,
        scale: Float = 1.0 / sqrt(128.0)
    ) -> [Float] {
        let absPos = queryAbsPos ?? (retainedKVStart + T_kv - 1)
        return evaluatePrefillFP32(
            queries: queries,
            keys: keys,
            values: values,
            sinks: sinks,
            batch: batch,
            T_q: 1,
            T_kv: T_kv,
            cacheOffset: absPos,
            retainedKVStart: retainedKVStart,
            isSliding: isSliding,
            scale: scale
        )
    }
}
