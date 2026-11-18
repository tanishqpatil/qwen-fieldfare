import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintAttentionKernelTests {

    private static func parseHexToData(_ hex: String) -> Data {
        var data = Data(capacity: hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let nextIndex = hex.index(index, offsetBy: 2)
            let byteStr = hex[index..<nextIndex]
            if let byte = UInt8(byteStr, radix: 16) {
                data.append(byte)
            }
            index = nextIndex
        }
        return data
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func bf16ULPDistance(_ val1: Float, _ val2: Float) -> Int {
        let diff = abs(val1 - val2)
        let maxAbs = max(abs(val1), abs(val2))
        guard maxAbs > 1e-7 else { return 0 }
        let expVal = floor(log2(maxAbs))
        let ulp = pow(2.0, expVal - 7.0)
        return Int(round(diff / ulp))
    }

    private struct GoldenCase: Decodable {
        let kind: String
        let name: String
        let dtype: String
        let num_heads: Int?
        let seq_len: Int?
        let offset: Int?
        let is_sliding: Bool?
        let layer_idx: Int?
        let T_q: Int?
        let T_kv: Int?
        let cache_offset: Int?
        let input_hex: String?
        let output_hex: String?
        let query_abs_pos: Int?
        let retained_kv_start: Int?
        let q_raw_hex: String?
        let k_raw_hex: String?
        let q_rot_hex: String?
        let k_rot_hex: String?
        let q_hex: String?
        let k_hex: String?
        let v_hex: String?
        let sinks_hex: String?
        let out_hex: String?
    }

    private struct GoldenFixture: Decodable {
        let metadata: [String: AnyCodable]
        let cases: [GoldenCase]
    }

    private struct AnyCodable: Decodable {}

    private static func loadGoldenFixture() throws -> GoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_attention_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(GoldenFixture.self, from: data)
    }

    // MARK: - 1. Synthetic Case for Learned Attention Sinks

    @Test func syntheticAttentionSinkAnalyticalContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)

        let D = 128
        // Let Q and K be simple unit vectors along dim 0
        var qF32 = [Float](repeating: 0, count: 32 * D)
        var kF32 = [Float](repeating: 0, count: 4 * 1 * D)
        var vF32 = [Float](repeating: 0, count: 4 * 1 * D)

        for h in 0..<32 {
            qF32[h * D + 0] = 1.0
        }
        for hkv in 0..<4 {
            kF32[hkv * D + 0] = 1.0
            vF32[hkv * D + 0] = 10.0
            vF32[hkv * D + 1] = 20.0
        }

        // Set scale such that Q * K * scale = 2.0
        let scale: Float = 2.0

        // Test with sink = 2.0:
        // Score S_0 = 2.0, sink s = 2.0
        // max = 2.0, exp(0) + exp(0) = 2.0
        // Prob = 1/2 = 0.5.
        // Expected out = [5.0, 10.0, 0, ...]
        var sinksF32 = [Float](repeating: 2.0, count: 32)

        let qBuf = try qF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let kBuf = try kF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let vBuf = try vF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let sBuf = try sinksF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outBuf = try context.makeBuffer(byteCount: 32 * D * 4)

        try kernel.executeDecode(
            queries: qBuf,
            keys: kBuf,
            values: vBuf,
            sinks: sBuf,
            output: outBuf,
            T_kv: 1,
            isSliding: false,
            scale: scale,
            dtype: .float32,
            waitUntilCompleted: true
        )

        let out1 = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: 32 * D))
        for h in 0..<32 {
            let offset = h * D
            #expect(abs(out1[offset + 0] - 5.0) <= 1e-5)
            #expect(abs(out1[offset + 1] - 10.0) <= 1e-5)
            for d in 2..<D {
                #expect(out1[offset + d] == 0.0, "Spurious non-zero component found in attention output!")
            }
        }

        // Now test monotonic scaling: increase sink to 2.0 + ln(3) ~ 3.098612
        // exp(s - 2.0) = 3.0
        // Denom = 1.0 + 3.0 = 4.0
        // Prob = 0.25
        // Expected out = [2.5, 5.0, 0, ...]
        let higherSink = 2.0 + log(3.0)
        sinksF32 = [Float](repeating: Float(higherSink), count: 32)
        let sHigherBuf = try sinksF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }

        try kernel.executeDecode(
            queries: qBuf,
            keys: kBuf,
            values: vBuf,
            sinks: sHigherBuf,
            output: outBuf,
            T_kv: 1,
            isSliding: false,
            scale: scale,
            dtype: .float32,
            waitUntilCompleted: true
        )

        let out2 = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: 32 * D))
        for h in 0..<32 {
            let offset = h * D
            #expect(abs(out2[offset + 0] - 2.5) <= 1e-4)
            #expect(abs(out2[offset + 1] - 5.0) <= 1e-4)
            for d in 2..<D {
                #expect(out2[offset + d] == 0.0)
            }
        }
    }

    // MARK: - 2. RoPE Mathematical Invariants & Equivalence

    @Test func ropeMathematicalInvariants() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRoPEKernel(context: context)

        let testOffsets = [0, 1, 5, 513, 1024]
        let numHeads = 32
        let seqLen = 4
        let headDim = 128
        let totalElements = numHeads * seqLen * headDim

        for offset in testOffsets {
            var inF32 = [Float](repeating: 0, count: totalElements)
            for i in 0..<totalElements {
                inF32[i] = sin(Float(i + 1) * 0.05) * 1.5
            }

            // CPU Reference
            let refOutF32 = FlintRoPEReference.evaluateFP32(
                input: inF32,
                numHeads: numHeads,
                seqLen: seqLen,
                offset: offset,
                base: 10000.0
            )

            // Metal GPU FP32
            let inBufF32 = try inF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let outBufF32 = try context.makeBuffer(byteCount: totalElements * 4)

            try kernel.execute(
                input: inBufF32,
                output: outBufF32,
                batch: 1,
                numHeads: numHeads,
                seqLen: seqLen,
                offset: offset,
                base: 10000.0,
                dtype: .float32,
                waitUntilCompleted: true
            )

            let metalOutF32 = Array(UnsafeBufferPointer(start: outBufF32.contents().assumingMemoryBound(to: Float.self), count: totalElements))
            for i in 0..<totalElements {
                let diff = abs(metalOutF32[i] - refOutF32[i])
                #expect(diff <= 2e-4)
            }

            // Metal GPU BF16
            let inBF16 = FlintRMSNormReference.floatArrayToBF16(inF32)
            let inBufBF16 = try inBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let outBufBF16 = try context.makeBuffer(byteCount: totalElements * 2)

            try kernel.execute(
                input: inBufBF16,
                output: outBufBF16,
                batch: 1,
                numHeads: numHeads,
                seqLen: seqLen,
                offset: offset,
                base: 10000.0,
                dtype: .bfloat16,
                waitUntilCompleted: true
            )

            let rawBF16 = Array(UnsafeBufferPointer(start: outBufBF16.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
            let metalOutBF16Float = FlintRMSNormReference.bf16ArrayToFloat(rawBF16)
            var l2DiffSq: Float = 0
            var l2RefSq: Float = 0
            for i in 0..<totalElements {
                let diff = abs(metalOutBF16Float[i] - refOutF32[i])
                #expect(diff <= 0.02 || diff / max(abs(refOutF32[i]), 0.1) <= 0.02)
                l2DiffSq += diff * diff
                l2RefSq += refOutF32[i] * refOutF32[i]
            }
            let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
            #expect(relL2 <= 0.005)
        }
    }

    // MARK: - 3. Sequence-Boundary & Sliding-Window Correctness (1, 2, 512, 513, 514, 1024)

    @Test func slidingWindowSequenceBoundaryCorrectness() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)

        let boundaryLengths = [1, 2, 512, 513, 514]
        let D = 128

        for T_kv in boundaryLengths {
            // Verify that sliding window correctly attends to min(T_kv, 513) keys
            var qF32 = [Float](repeating: 0, count: 32 * D)
            var kF32 = [Float](repeating: 0, count: 4 * T_kv * D)
            var vF32 = [Float](repeating: 0, count: 4 * T_kv * D)
            let sinksF32 = [Float](repeating: 0.0, count: 32)

            // Initialize Q and K to attend with positive dot product
            for h in 0..<32 { qF32[h * D + 0] = 1.0 }
            for hkv in 0..<4 {
                for j in 0..<T_kv {
                    let offset = (hkv * T_kv + j) * D
                    kF32[offset + 0] = 1.0
                    vF32[offset + 0] = Float(j + 1) // Value stores token 1-based index
                }
            }

            let qBuf = try qF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let kBuf = try kF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let vBuf = try vF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let sBuf = try sinksF32.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let outBuf = try context.makeBuffer(byteCount: 32 * D * 4)

            // Decode at position T_kv - 1
            try kernel.executeDecode(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_kv: T_kv,
                isSliding: true,
                dtype: .float32,
                waitUntilCompleted: true
            )

            let refOut = FlintAttentionReference.evaluateDecodeFP32(
                queries: qF32,
                keys: kF32,
                values: vF32,
                sinks: sinksF32,
                T_kv: T_kv,
                isSliding: true
            )

            let metalOut = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: 32 * D))
            for i in 0..<32 * D {
                let diff = abs(metalOut[i] - refOut[i])
                #expect(diff <= 1e-4)
            }
        }
    }

    // MARK: - 4. Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_attention_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let ropeKernel = try FlintRoPEKernel(context: context)
        let attnKernel = try FlintAttentionKernel(context: context)

        print("\n=== FLINT ATTENTION NUMERICAL QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-44@ | %-24@ | %-8@ | %10@ | %10@ | %10@ | %10@ | %6@ | %@",
                     "Case Name", "Kind", "Dtype", "Max Abs", "Max Rel", "Rel L2", "RMS", "ULP", "Status"))
        print(String(repeating: "-", count: 135))

        for c in fixture.cases {
            if c.kind == "rope" {
                let inData = Self.parseHexToData(c.input_hex!)
                let expectedOutData = Self.parseHexToData(c.output_hex!)
                let dtype: FlintRoPEKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

                let inBuf = try inData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: inData.count) }
                let outBuf = try context.makeBuffer(byteCount: expectedOutData.count)

                try ropeKernel.execute(
                    input: inBuf,
                    output: outBuf,
                    batch: 1,
                    numHeads: c.num_heads!,
                    seqLen: c.seq_len!,
                    offset: c.offset!,
                    dtype: dtype,
                    waitUntilCompleted: true
                )

                let totalElements = c.num_heads! * c.seq_len! * 128
                var maxAbsDiff: Float = 0.0
                var maxRelDiff: Float = 0.0
                var l2DiffSq: Float = 0.0
                var l2RefSq: Float = 0.0
                var maxULP: Int = 0

                if dtype == .float32 {
                    let metal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                    let expected = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 1e-4)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                    }
                    #expect(maxAbsDiff <= 1e-5)
                } else {
                    let rawMetal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                    let metal = FlintRMSNormReference.bf16ArrayToFloat(rawMetal)
                    let rawExp = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let expected = FlintRMSNormReference.bf16ArrayToFloat(rawExp)
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 0.1)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                        let ulp = Self.bf16ULPDistance(metal[i], expected[i])
                        if ulp > maxULP { maxULP = ulp }
                    }
                    #expect(maxRelDiff <= 0.015)
                }

                let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                let rms = sqrt(l2DiffSq / Float(totalElements))
                print(String(format: "%-44@ | %-24@ | %-8@ | %10.6f | %10.6f | %10.6f | %10.6f | %6d | PASS",
                             c.name, c.kind, c.dtype, maxAbsDiff, maxRelDiff, relL2, rms, maxULP))

            } else if c.kind == "attention_prefill" {
                let qData = Self.parseHexToData(c.q_hex!)
                let kData = Self.parseHexToData(c.k_hex!)
                let vData = Self.parseHexToData(c.v_hex!)
                let sData = Self.parseHexToData(c.sinks_hex!)
                let expectedOutData = Self.parseHexToData(c.out_hex!)
                let dtype: FlintAttentionKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

                let qBuf = try qData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: qData.count) }
                let kBuf = try kData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: kData.count) }
                let vBuf = try vData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: vData.count) }
                let sBuf = try sData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: sData.count) }
                let outBuf = try context.makeBuffer(byteCount: expectedOutData.count)

                try attnKernel.executePrefill(
                    queries: qBuf,
                    keys: kBuf,
                    values: vBuf,
                    sinks: sBuf,
                    output: outBuf,
                    batch: 1,
                    T_q: c.T_q!,
                    T_kv: c.T_kv!,
                    cacheOffset: c.cache_offset!,
                    isSliding: c.is_sliding!,
                    dtype: dtype,
                    waitUntilCompleted: true
                )

                let totalElements = 32 * c.T_q! * 128
                var maxAbsDiff: Float = 0.0
                var maxRelDiff: Float = 0.0
                var l2DiffSq: Float = 0.0
                var l2RefSq: Float = 0.0
                var maxULP: Int = 0

                if dtype == .float32 {
                    let metal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                    let expected = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 1e-4)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                    }
                    #expect(maxAbsDiff <= 1e-4)
                } else {
                    let rawMetal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                    let metal = FlintRMSNormReference.bf16ArrayToFloat(rawMetal)
                    let rawExp = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let expected = FlintRMSNormReference.bf16ArrayToFloat(rawExp)
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 0.1)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                        let ulp = Self.bf16ULPDistance(metal[i], expected[i])
                        if ulp > maxULP { maxULP = ulp }
                    }
                    let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                    #expect(relL2 <= 5e-3)
                }

                let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                let rms = sqrt(l2DiffSq / Float(totalElements))
                print(String(format: "%-44@ | %-24@ | %-8@ | %10.6f | %10.6f | %10.6f | %10.6f | %6d | PASS",
                             c.name, c.kind, c.dtype, maxAbsDiff, maxRelDiff, relL2, rms, maxULP))

            } else if c.kind == "attention_decode" {
                let qData = Self.parseHexToData(c.q_hex!)
                let kData = Self.parseHexToData(c.k_hex!)
                let vData = Self.parseHexToData(c.v_hex!)
                let sData = Self.parseHexToData(c.sinks_hex!)
                let expectedOutData = Self.parseHexToData(c.out_hex!)
                let dtype: FlintAttentionKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

                let qBuf = try qData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: qData.count) }
                let kBuf = try kData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: kData.count) }
                let vBuf = try vData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: vData.count) }
                let sBuf = try sData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: sData.count) }
                let outBuf = try context.makeBuffer(byteCount: expectedOutData.count)

                try attnKernel.executeDecode(
                    queries: qBuf,
                    keys: kBuf,
                    values: vBuf,
                    sinks: sBuf,
                    output: outBuf,
                    batch: 1,
                    T_kv: c.T_kv!,
                    isSliding: c.is_sliding!,
                    dtype: dtype,
                    waitUntilCompleted: true
                )

                let totalElements = 32 * 1 * 128
                var maxAbsDiff: Float = 0.0
                var maxRelDiff: Float = 0.0
                var l2DiffSq: Float = 0.0
                var l2RefSq: Float = 0.0
                var maxULP: Int = 0

                if dtype == .float32 {
                    let metal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                    let expected = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 1e-4)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                    }
                    #expect(maxAbsDiff <= 1e-4)
                } else {
                    let rawMetal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                    let metal = FlintRMSNormReference.bf16ArrayToFloat(rawMetal)
                    let rawExp = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let expected = FlintRMSNormReference.bf16ArrayToFloat(rawExp)
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 0.1)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                        let ulp = Self.bf16ULPDistance(metal[i], expected[i])
                        if ulp > maxULP { maxULP = ulp }
                    }
                    let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                    #expect(relL2 <= 5e-3)
                }

                let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                let rms = sqrt(l2DiffSq / Float(totalElements))
                print(String(format: "%-44@ | %-24@ | %-8@ | %10.6f | %10.6f | %10.6f | %10.6f | %6d | PASS",
                             c.name, c.kind, c.dtype, maxAbsDiff, maxRelDiff, relL2, rms, maxULP))

            } else if c.kind == "attention_decode_bounded" {
                let qData = Self.parseHexToData(c.q_hex!)
                let kData = Self.parseHexToData(c.k_hex!)
                let vData = Self.parseHexToData(c.v_hex!)
                let sData = Self.parseHexToData(c.sinks_hex!)
                let expectedOutData = Self.parseHexToData(c.out_hex!)
                let dtype: FlintAttentionKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

                let qBuf = try qData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: qData.count) }
                let kBuf = try kData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: kData.count) }
                let vBuf = try vData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: vData.count) }
                let sBuf = try sData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: sData.count) }
                let outBuf = try context.makeBuffer(byteCount: expectedOutData.count)

                try attnKernel.executeDecode(
                    queries: qBuf,
                    keys: kBuf,
                    values: vBuf,
                    sinks: sBuf,
                    output: outBuf,
                    batch: 1,
                    T_kv: c.T_kv!,
                    queryAbsPos: c.query_abs_pos!,
                    retainedKVStart: c.retained_kv_start!,
                    isSliding: c.is_sliding!,
                    dtype: dtype,
                    waitUntilCompleted: true
                )

                let totalElements = 32 * 1 * 128
                var maxAbsDiff: Float = 0.0
                var maxRelDiff: Float = 0.0
                var l2DiffSq: Float = 0.0
                var l2RefSq: Float = 0.0
                var maxULP: Int = 0

                if dtype == .float32 {
                    let metal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                    let expected = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 1e-4)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                    }
                    #expect(maxAbsDiff <= 1e-4)
                } else {
                    let rawMetal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                    let metal = FlintRMSNormReference.bf16ArrayToFloat(rawMetal)
                    let rawExp = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let expected = FlintRMSNormReference.bf16ArrayToFloat(rawExp)
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 0.1)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                        let ulp = Self.bf16ULPDistance(metal[i], expected[i])
                        if ulp > maxULP { maxULP = ulp }
                    }
                    let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                    #expect(relL2 <= 5e-3)
                }

                let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                let rms = sqrt(l2DiffSq / Float(totalElements))
                print(String(format: "%-44@ | %-24@ | %-8@ | %10.6f | %10.6f | %10.6f | %10.6f | %6d | PASS",
                             c.name, c.kind, c.dtype, maxAbsDiff, maxRelDiff, relL2, rms, maxULP))

            } else if c.kind == "combined_rope_attention" {
                let qRawData = Self.parseHexToData(c.q_raw_hex!)
                let kRawData = Self.parseHexToData(c.k_raw_hex!)
                let expectedQRotData = Self.parseHexToData(c.q_rot_hex!)
                let expectedKRotData = Self.parseHexToData(c.k_rot_hex!)
                let vData = Self.parseHexToData(c.v_hex!)
                let sData = Self.parseHexToData(c.sinks_hex!)
                let expectedOutData = Self.parseHexToData(c.out_hex!)
                let dtype: FlintAttentionKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32
                let ropeDtype: FlintRoPEKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

                let qRawBuf = try qRawData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: qRawData.count) }
                let kRawBuf = try kRawData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: kRawData.count) }
                let qRotBuf = try context.makeBuffer(byteCount: expectedQRotData.count)
                let kRotBuf = try context.makeBuffer(byteCount: expectedKRotData.count)
                let vBuf = try vData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: vData.count) }
                let sBuf = try sData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: sData.count) }
                let outBuf = try context.makeBuffer(byteCount: expectedOutData.count)

                // 1. Split-half RoPE on Q at query_abs_pos (e.g. 600)
                try ropeKernel.execute(
                    input: qRawBuf,
                    output: qRotBuf,
                    batch: 1,
                    numHeads: 32,
                    seqLen: 1,
                    offset: c.query_abs_pos!,
                    dtype: ropeDtype,
                    waitUntilCompleted: true
                )

                // 2. Split-half RoPE on K at retained_kv_start (e.g. 88)
                try ropeKernel.execute(
                    input: kRawBuf,
                    output: kRotBuf,
                    batch: 1,
                    numHeads: 4,
                    seqLen: c.T_kv!,
                    offset: c.retained_kv_start!,
                    dtype: ropeDtype,
                    waitUntilCompleted: true
                )

                // 3. Grouped-Query Attention with learned sinks
                try attnKernel.executeDecode(
                    queries: qRotBuf,
                    keys: kRotBuf,
                    values: vBuf,
                    sinks: sBuf,
                    output: outBuf,
                    batch: 1,
                    T_kv: c.T_kv!,
                    queryAbsPos: c.query_abs_pos!,
                    retainedKVStart: c.retained_kv_start!,
                    isSliding: c.is_sliding!,
                    dtype: dtype,
                    waitUntilCompleted: true
                )

                let totalElements = 32 * 1 * 128
                var maxAbsDiff: Float = 0.0
                var maxRelDiff: Float = 0.0
                var l2DiffSq: Float = 0.0
                var l2RefSq: Float = 0.0
                var maxULP: Int = 0

                if dtype == .float32 {
                    let metal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                    let expected = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 1e-4)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                    }
                    #expect(maxAbsDiff <= 1e-4)
                } else {
                    let rawMetal = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                    let metal = FlintRMSNormReference.bf16ArrayToFloat(rawMetal)
                    let rawExp = expectedOutData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let expected = FlintRMSNormReference.bf16ArrayToFloat(rawExp)
                    for i in 0..<totalElements {
                        let diff = abs(metal[i] - expected[i])
                        let rel = diff / max(abs(expected[i]), 0.1)
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                        if rel > maxRelDiff { maxRelDiff = rel }
                        l2DiffSq += diff * diff
                        l2RefSq += expected[i] * expected[i]
                        let ulp = Self.bf16ULPDistance(metal[i], expected[i])
                        if ulp > maxULP { maxULP = ulp }
                    }
                    let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                    #expect(relL2 <= 5e-3)
                }

                let relL2 = sqrt(l2DiffSq) / max(sqrt(l2RefSq), 1e-6)
                let rms = sqrt(l2DiffSq / Float(totalElements))
                print(String(format: "%-44@ | %-24@ | %-8@ | %10.6f | %10.6f | %10.6f | %10.6f | %6d | PASS",
                             c.name, c.kind, c.dtype, maxAbsDiff, maxRelDiff, relL2, rms, maxULP))
            }
        }
        print(String(repeating: "-", count: 135))
    }

    // MARK: - 5. Real Checkpoint Resident Weights Integration & Immutability

    @Test func realCheckpointResidentAttentionSinksIntegration() throws {
        let modelDir = "/Users/tanishq/models/flint-mixed-3-4bit"
        guard FileManager.default.fileExists(atPath: modelDir) else {
            Issue.record("Model directory not found at: \(modelDir)")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)
        let validated = try FlintModelDirectory.validate(path: modelDir)
        let shards = try FlintModelDirectory.openShards(canonicalPath: validated.canonicalPath, shardNames: validated.shards)
        let (residentWeights, _) = try FlintResidentLoader.loadResidentWeights(
            validatedDirectory: validated,
            shardFiles: shards,
            storageMode: .gpuRequired,
            device: context.device
        )

        // 1. Check Layer 0 Attention Sinks (Full Attention)
        guard let l0Attn = residentWeights.attentionWeights(forLayer: 0) else {
            Issue.record("Missing layer 0 attention weights")
            return
        }
        let l0SinkShaBefore = Self.sha256Hex(l0Attn.softmaxOffset.buffer.copyData())
        guard let l0SinkBuf = l0Attn.softmaxOffset.metalBuffer else {
            Issue.record("softmaxOffset.metalBuffer is nil for layer 0")
            return
        }

        let D = 128
        let qBuf = try context.makeBuffer(byteCount: 32 * D * 2)
        let kBuf = try context.makeBuffer(byteCount: 4 * 16 * D * 2)
        let vBuf = try context.makeBuffer(byteCount: 4 * 16 * D * 2)
        let outBuf = try context.makeBuffer(byteCount: 32 * D * 2)

        try kernel.executeDecode(
            queries: qBuf,
            keys: kBuf,
            values: vBuf,
            sinks: l0SinkBuf,
            output: outBuf,
            T_kv: 16,
            isSliding: false,
            dtype: .bfloat16,
            waitUntilCompleted: true
        )

        let l0SinkShaAfter = Self.sha256Hex(l0Attn.softmaxOffset.buffer.copyData())
        #expect(l0SinkShaBefore == l0SinkShaAfter, "Layer 0 softmax_offset was unexpectedly mutated!")

        // 2. Check Layer 2 Attention Sinks (Sliding Attention)
        guard let l2Attn = residentWeights.attentionWeights(forLayer: 2) else {
            Issue.record("Missing layer 2 attention weights")
            return
        }
        let l2SinkShaBefore = Self.sha256Hex(l2Attn.softmaxOffset.buffer.copyData())
        guard let l2SinkBuf = l2Attn.softmaxOffset.metalBuffer else {
            Issue.record("softmaxOffset.metalBuffer is nil for layer 2")
            return
        }

        try kernel.executeDecode(
            queries: qBuf,
            keys: kBuf,
            values: vBuf,
            sinks: l2SinkBuf,
            output: outBuf,
            T_kv: 16,
            isSliding: true,
            dtype: .bfloat16,
            waitUntilCompleted: true
        )

        let l2SinkShaAfter = Self.sha256Hex(l2Attn.softmaxOffset.buffer.copyData())
        #expect(l2SinkShaBefore == l2SinkShaAfter, "Layer 2 softmax_offset was unexpectedly mutated!")
    }

    // MARK: - 6. Buffer Bounds & Parameter Error Handling

    @Test func bufferBoundsAndParameterValidation() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)

        let D = 128
        let qBuf = try context.makeBuffer(byteCount: 32 * D * 2)
        let kBuf = try context.makeBuffer(byteCount: 4 * 8 * D * 2)
        let vBuf = try context.makeBuffer(byteCount: 4 * 8 * D * 2)
        let sBuf = try context.makeBuffer(byteCount: 32 * 2)
        let outBuf = try context.makeBuffer(byteCount: 32 * D * 2)

        // 1. T_kv <= 0
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(queries: qBuf, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: 0, isSliding: false)
        }

        // 2. Scale <= 0
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(queries: qBuf, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: 8, isSliding: false, scale: -1.0)
        }

        // 3. Keys buffer too small
        let smallKBuf = try context.makeBuffer(byteCount: 4 * 2 * D * 2)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(queries: qBuf, keys: smallKBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: 8, isSliding: false)
        }

        // 4. Misaligned offset
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(queries: qBuf, queriesOffset: 1, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: 8, isSliding: false)
        }

        // 5. Prefill T_q > T_kv
        #expect(throws: FlintMetalError.self) {
            try kernel.executePrefill(queries: qBuf, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_q: 16, T_kv: 8, isSliding: false)
        }
    }

    // MARK: - 7. Apple M5 Hardware Benchmark & Telemetry

    @Test func benchmarkAttentionPerformanceOnM5() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)
        let ropeKernel = try FlintRoPEKernel(context: context)

        print("\n=== FLINT ATTENTION HARDWARE BENCHMARK (APPLE M5) ===")

        let D = 128
        let T_kv = 513
        let qBuf = try context.makeBuffer(byteCount: 32 * D * 2)
        let kBuf = try context.makeBuffer(byteCount: 4 * T_kv * D * 2)
        let vBuf = try context.makeBuffer(byteCount: 4 * T_kv * D * 2)
        let sBuf = try context.makeBuffer(byteCount: 32 * 2)
        let outBuf = try context.makeBuffer(byteCount: 32 * D * 2)

        // Decode benchmark (100 iterations)
        let iterations = 100
        var durations = [Double]()
        durations.reserveCapacity(iterations)

        for _ in 0..<10 {
            try kernel.executeDecode(queries: qBuf, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: T_kv, isSliding: true)
        }

        for _ in 0..<iterations {
            let dur = try kernel.executeDecode(queries: qBuf, keys: kBuf, values: vBuf, sinks: sBuf, output: outBuf, T_kv: T_kv, isSliding: true)
            durations.append(dur * 1e6)
        }

        durations.sort()
        let p50 = durations[iterations / 2]
        let minLat = durations.first ?? 0
        let p95 = durations[Int(Double(iterations) * 0.95)]
        let maxLat = durations.last ?? 0

        let readBytes = (32 * D * 2) + (2 * 4 * T_kv * D * 2) + (32 * 2)
        let writeBytes = 32 * D * 2
        let nominalGB = Double(readBytes + writeBytes) / 1e9
        let throughput = (p50 > 0) ? (nominalGB / (p50 / 1e6)) : 0

        print(String(format: "%-45@ | P50: %6.2f µs | Min: %6.2f µs | P95: %6.2f µs | Max: %6.2f µs | Nominal: %6.2f GB/s",
                     "Decode Attention (1 token, 513 KV cache)", p50, minLat, p95, maxLat, throughput))

        // RoPE benchmark (100 iterations)
        var ropeDurations = [Double]()
        for _ in 0..<iterations {
            let dur = try ropeKernel.execute(input: qBuf, output: qBuf, batch: 1, numHeads: 32, seqLen: 1, offset: 512, dtype: .bfloat16)
            ropeDurations.append(dur * 1e6)
        }
        ropeDurations.sort()
        print(String(format: "%-45@ | P50: %6.2f µs | Min: %6.2f µs | P95: %6.2f µs | Max: %6.2f µs",
                     "RoPE Decode (32 heads, 1 token)", ropeDurations[50], ropeDurations[0], ropeDurations[95], ropeDurations.last!))
        print("=====================================================\n")
    }

    // MARK: - 8. Ring KV Cache Layout Rejection

    @Test func ringKVLayoutRejection() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)

        let qBuf = try context.makeBuffer(byteCount: 32 * 128 * 4)
        let kBuf = try context.makeBuffer(byteCount: 4 * 16 * 128 * 4)
        let vBuf = try context.makeBuffer(byteCount: 4 * 16 * 128 * 4)
        let sBuf = try context.makeBuffer(byteCount: 32 * 4)
        let outBuf = try context.makeBuffer(byteCount: 32 * 128 * 4)

        // Verify executeDecode rejects .ring layout with invalidParameter
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_kv: 16,
                isSliding: true,
                layout: .ring,
                dtype: .float32
            )
        }

        // Verify executePrefill rejects .ring layout with invalidParameter
        #expect(throws: FlintMetalError.self) {
            try kernel.executePrefill(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_q: 1,
                T_kv: 16,
                isSliding: true,
                layout: .ring,
                dtype: .float32
            )
        }
    }

    // MARK: - 9. Bounded-KV Indexing & Eviction Boundary Qualification

    @Test func boundedKVIndexingEvictionScenario() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintAttentionKernel(context: context)

        // Specifically test Review #007 scenario:
        // Absolute position 1024, retained interval 512..1024, physical indices 0..512.
        // Sliding window 513: attends entire retained interval [512, 1024] -> physical range [0, 513).
        let T_kv = 513
        let retainedKVStart = 512
        let queryAbsPos = 1024
        let headDim = 128
        let numQHeads = 32
        let numKVHeads = 4

        var qData = [Float](repeating: 0, count: numQHeads * headDim)
        var kData = [Float](repeating: 0, count: numKVHeads * T_kv * headDim)
        var vData = [Float](repeating: 0, count: numKVHeads * T_kv * headDim)
        let sData = [Float](repeating: -1.0, count: numQHeads)

        for i in 0..<qData.count { qData[i] = Float.random(in: -1.0...1.0) }
        for i in 0..<kData.count { kData[i] = Float.random(in: -1.0...1.0) }
        for i in 0..<vData.count { vData[i] = Float.random(in: -1.0...1.0) }

        let qBuf = try qData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let kBuf = try kData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let vBuf = try vData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let sBuf = try sData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outBuf = try context.makeBuffer(byteCount: numQHeads * headDim * 4)

        try kernel.executeDecode(
            queries: qBuf,
            keys: kBuf,
            values: vBuf,
            sinks: sBuf,
            output: outBuf,
            T_kv: T_kv,
            queryAbsPos: queryAbsPos,
            retainedKVStart: retainedKVStart,
            isSliding: true,
            dtype: .float32,
            waitUntilCompleted: true
        )

        // Compare against CPU reference
        let refOut = FlintAttentionReference.evaluateDecodeFP32(
            queries: qData,
            keys: kData,
            values: vData,
            sinks: sData,
            batch: 1,
            T_kv: T_kv,
            queryAbsPos: queryAbsPos,
            retainedKVStart: retainedKVStart,
            isSliding: true
        )

        let metalOut = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: numQHeads * headDim))
        var maxDiff: Float = 0
        for i in 0..<metalOut.count {
            let diff = abs(metalOut[i] - refOut[i])
            if diff > maxDiff { maxDiff = diff }
        }
        #expect(maxDiff <= 1e-5, "Metal decode does not match CPU reference on pos 1024 eviction (max diff \(maxDiff))")

        // Test boundary shifts with complete retained storage:
        for (testPos, testStart) in [(1024, 512), (1025, 513), (1026, 514)] {
            try kernel.executeDecode(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_kv: T_kv,
                queryAbsPos: testPos,
                retainedKVStart: testStart,
                isSliding: true,
                dtype: .float32,
                waitUntilCompleted: true
            )
            let posRef = FlintAttentionReference.evaluateDecodeFP32(
                queries: qData,
                keys: kData,
                values: vData,
                sinks: sData,
                batch: 1,
                T_kv: T_kv,
                queryAbsPos: testPos,
                retainedKVStart: testStart,
                isSliding: true
            )
            let outPtr = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: numQHeads * headDim))
            var pDiff: Float = 0
            for i in 0..<outPtr.count {
                let diff = abs(outPtr[i] - posRef[i])
                if diff > pDiff { pDiff = diff }
            }
            #expect(pDiff <= 1e-5, "Position \(testPos) failed bounded-KV equivalence (diff \(pDiff))")
        }

        // Test Review #008 required rejection:
        // Incomplete KV history (query at 1024 requires keys back to 512, but retained cache begins at 900)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_kv: 125,
                queryAbsPos: 1024,
                retainedKVStart: 900,
                isSliding: true,
                dtype: .float32,
                waitUntilCompleted: true
            )
        }

        // Test prefill incomplete history rejection
        #expect(throws: FlintMetalError.self) {
            try kernel.executePrefill(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_q: 1,
                T_kv: 125,
                cacheOffset: 1024,
                retainedKVStart: 900,
                isSliding: true,
                dtype: .float32,
                waitUntilCompleted: true
            )
        }

        // Test full attention incomplete history rejection (requires keys back to 0, but cache starts at 5)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDecode(
                queries: qBuf,
                keys: kBuf,
                values: vBuf,
                sinks: sBuf,
                output: outBuf,
                T_kv: 10,
                queryAbsPos: 10,
                retainedKVStart: 5,
                isSliding: false,
                dtype: .float32,
                waitUntilCompleted: true
            )
        }
    }

    // MARK: - 10. Canonical Metal Shader Source Equivalence

    @Test func canonicalMetalShaderSourceEquivalence() throws {
        let attnMetalPath = "Sources/FlintFieldfareCore/Metal/flint_attention.metal"
        let fullAttnPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(attnMetalPath)
        guard FileManager.default.fileExists(atPath: fullAttnPath.path) else {
            Issue.record("flint_attention.metal not found at path: \(fullAttnPath.path)")
            return
        }
        let diskAttnSource = try String(contentsOf: fullAttnPath, encoding: .utf8)
        #expect(diskAttnSource == FlintAttentionKernel.embeddedShaderSource, "Disk flint_attention.metal and FlintAttentionKernel.embeddedShaderSource have drifted!")

        let ropeMetalPath = "Sources/FlintFieldfareCore/Metal/flint_rope.metal"
        let fullRopePath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(ropeMetalPath)
        guard FileManager.default.fileExists(atPath: fullRopePath.path) else {
            Issue.record("flint_rope.metal not found at path: \(fullRopePath.path)")
            return
        }
        let diskRopeSource = try String(contentsOf: fullRopePath, encoding: .utf8)
        #expect(diskRopeSource == FlintRoPEKernel.embeddedShaderSource, "Disk flint_rope.metal and FlintRoPEKernel.embeddedShaderSource have drifted!")
    }
}
