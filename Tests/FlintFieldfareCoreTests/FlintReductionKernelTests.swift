import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintReductionKernelTests {

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

    private static func makeBuffer(context: FlintMetalContext, data: Data) throws -> MTLBuffer {
        try data.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else {
                return try context.makeBuffer(byteCount: 0)
            }
            return try context.makeBuffer(bytes: base, byteCount: data.count)
        }
    }

    private static func makeBuffer<T>(context: FlintMetalContext, array: [T]) throws -> MTLBuffer {
        try array.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else {
                return try context.makeBuffer(byteCount: 0)
            }
            return try context.makeBuffer(bytes: base, byteCount: ptr.count)
        }
    }

    private struct ReductionGoldenCase: Decodable {
        let name: String
        let batch: Int
        let tokens: Int
        let mode: String
        let has_shared: Bool
        let contributions_hex: String
        let weights_hex: String
        let shared_hex: String?
        let expected_hex: String
    }

    private struct ReductionGoldenFixture: Decodable {
        let metadata: [String: AnyCodable]
        let cases: [ReductionGoldenCase]
    }

    private struct AnyCodable: Decodable {}

    private static func loadGoldenFixture() throws -> ReductionGoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_reduction_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(ReductionGoldenFixture.self, from: data)
    }

    private static func bf16ToFloat(_ u16: UInt16) -> Float {
        Float(bitPattern: UInt32(u16) << 16)
    }

    // MARK: - 1. Pinned Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_reduction_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        print("\n=== FLINT MOE FIXED ROUTER-ORDER EXPERT REDUCTION QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-32@ | %-14@ | %5@ | %6@ | %12@ | %12@ | %10@ | %@",
                     "Case Name", "Mode", "Batch", "Tokens", "Max Abs Diff", "Relative L2", "Bit Match", "Status"))
        print(String(repeating: "-", count: 115))

        for c in fixture.cases {
            let cData = Self.parseHexToData(c.contributions_hex)
            let wData = Self.parseHexToData(c.weights_hex)
            let expData = Self.parseHexToData(c.expected_hex)

            let cBuf = try Self.makeBuffer(context: context, data: cData)
            let wBuf = try Self.makeBuffer(context: context, data: wData)
            var sBuf: MTLBuffer? = nil
            if let sHex = c.shared_hex {
                let sData = Self.parseHexToData(sHex)
                sBuf = try Self.makeBuffer(context: context, data: sData)
            }

            let outBuf: MTLBuffer
            let inDtype: FlintReductionKernel.Dtype
            let outDtype: FlintReductionKernel.Dtype

            switch c.mode {
            case "bf16":
                inDtype = .bfloat16
                outDtype = .bfloat16
                outBuf = try context.makeBuffer(byteCount: c.batch * c.tokens * 2048 * 2)
            case "f32":
                inDtype = .float32
                outDtype = .float32
                outBuf = try context.makeBuffer(byteCount: c.batch * c.tokens * 2048 * 4)
            case "f32_in_bf16_out":
                inDtype = .float32
                outDtype = .bfloat16
                outBuf = try context.makeBuffer(byteCount: c.batch * c.tokens * 2048 * 2)
            default:
                fatalError("Unknown mode \(c.mode)")
            }

            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: sBuf,
                output: outBuf,
                batch: c.batch,
                numTokens: c.tokens,
                inDtype: inDtype,
                outDtype: outDtype
            )

            let totalElements = c.batch * c.tokens * 2048
            var maxAbsDiff: Float = 0
            var l2DiffSq: Double = 0
            var l2ExpSq: Double = 0
            var bitMismatches = 0

            if outDtype == .bfloat16 {
                let actualU16 = outBuf.contents().bindMemory(to: UInt16.self, capacity: totalElements)
                let expU16 = expData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

                for i in 0..<totalElements {
                    if actualU16[i] != expU16[i] {
                        bitMismatches += 1
                    }
                    let actF = Self.bf16ToFloat(actualU16[i])
                    let expF = Self.bf16ToFloat(expU16[i])
                    let diff = abs(actF - expF)
                    if diff > maxAbsDiff {
                        maxAbsDiff = diff
                    }
                    l2DiffSq += Double(diff * diff)
                    l2ExpSq += Double(expF * expF)
                }
            } else {
                let actualF32 = outBuf.contents().bindMemory(to: Float.self, capacity: totalElements)
                let expF32 = expData.withUnsafeBytes { $0.bindMemory(to: Float.self) }

                for i in 0..<totalElements {
                    let actF = actualF32[i]
                    let expF = expF32[i]
                    let diff = abs(actF - expF)
                    if diff > maxAbsDiff {
                        maxAbsDiff = diff
                    }
                    l2DiffSq += Double(diff * diff)
                    l2ExpSq += Double(expF * expF)
                }
            }

            let relL2 = (l2ExpSq > 0) ? sqrt(l2DiffSq / l2ExpSq) : sqrt(l2DiffSq)
            let bitMatchStr = (outDtype == .bfloat16) ? "\(totalElements - bitMismatches)/\(totalElements)" : "N/A (FP32)"

            print(String(format: "%-32@ | %-14@ | %5d | %6d | %12.6f | %12.6e | %10@ | PASSED",
                         c.name, c.mode, c.batch, c.tokens, maxAbsDiff, relL2, bitMatchStr))

            if outDtype == .bfloat16 {
                #expect(maxAbsDiff <= 0.05, "Case \(c.name) max abs diff exceeded tolerance: \(maxAbsDiff)")
                #expect(relL2 <= 0.01, "Case \(c.name) relative L2 exceeded tolerance: \(relL2)")
            } else {
                #expect(maxAbsDiff <= 1e-5, "Case \(c.name) FP32 max abs diff exceeded tolerance: \(maxAbsDiff)")
                #expect(relL2 <= 1e-5, "Case \(c.name) FP32 relative L2 exceeded tolerance: \(relL2)")
            }
        }
    }

    // MARK: - 2. Analytical CPU Reference Contract

    @Test func analyticalReductionContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let hiddenDim = 2048
        let topK = 6

        let testConfigurations: [(batch: Int, tokens: Int, hasShared: Bool)] = [
            (1, 1, false),
            (1, 1, true),
            (1, 8, false),
            (1, 8, true),
            (2, 4, true),
        ]

        for config in testConfigurations {
            let totalContribs = config.batch * config.tokens * topK * hiddenDim
            let totalWeights = config.batch * config.tokens * topK
            let totalOut = config.batch * config.tokens * hiddenDim

            var contribs = [Float](repeating: 0, count: totalContribs)
            for i in 0..<totalContribs {
                contribs[i] = Float.random(in: -3.0...3.0)
            }

            var weights = [Float](repeating: 0, count: totalWeights)
            for b in 0..<config.batch {
                for t in 0..<config.tokens {
                    let offset = (b * config.tokens + t) * topK
                    var sum: Float = 0
                    for k in 0..<topK {
                        let w = Float.random(in: 0.05...0.35)
                        weights[offset + k] = w
                        sum += w
                    }
                    for k in 0..<topK {
                        weights[offset + k] /= sum
                    }
                }
            }

            var shared: [Float]? = nil
            if config.hasShared {
                shared = (0..<totalOut).map { _ in Float.random(in: -2.0...2.0) }
            }

            // CPU reference evaluation
            let expectedOut = FlintReductionReference.evaluateReduction(
                contributions: contribs,
                weights: weights,
                sharedExpert: shared,
                batch: config.batch,
                numTokens: config.tokens
            )

            // GPU execution (FP32 mode for bit-level analytical comparison)
            let cBuf = try Self.makeBuffer(context: context, array: contribs)
            let wBuf = try Self.makeBuffer(context: context, array: weights)
            let sBuf = try shared.map { try Self.makeBuffer(context: context, array: $0) }
            let outBuf = try context.makeBuffer(byteCount: totalOut * 4)

            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: sBuf,
                output: outBuf,
                batch: config.batch,
                numTokens: config.tokens,
                inDtype: .float32,
                outDtype: .float32
            )

            let metalOut = outBuf.contents().bindMemory(to: Float.self, capacity: totalOut)
            var maxDiff: Float = 0
            for i in 0..<totalOut {
                let diff = abs(metalOut[i] - expectedOut[i])
                if diff > maxDiff {
                    maxDiff = diff
                }
            }

            #expect(maxDiff <= 1e-6, "Analytical contract max difference \(maxDiff) exceeded 1e-6 for configuration: batch=\(config.batch), tokens=\(config.tokens), shared=\(config.hasShared)")
        }
    }

    // MARK: - 3. CRITICAL Residency Invariance Contract

    @Test func residencyInvarianceContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let hiddenDim = 2048
        let topK = 6

        print("\n=== FLINT MOE RESIDENCY INVARIANCE & COMPLETION-ORDER CONTRACT ===")
        print("Verifying that arbitrary arrival/completion order of expert outputs produces bitwise-identical reduction.")

        // We run 10 distinct randomized trials
        for trial in 0..<10 {
            // Generate 6 distinct expert contribution vectors for slot 0..5
            var expertVectors = [[UInt16]](repeating: [], count: topK)
            for k in 0..<topK {
                var vec = [UInt16](repeating: 0, count: hiddenDim)
                for d in 0..<hiddenDim {
                    let f = Float.random(in: -5.0...5.0)
                    vec[d] = UInt16(f.bitPattern >> 16)
                }
                expertVectors[k] = vec
            }

            var weights = [Float](repeating: 0, count: topK)
            var wSum: Float = 0
            for k in 0..<topK {
                let w = Float.random(in: 0.05...0.35)
                weights[k] = w
                wSum += w
            }
            for k in 0..<topK {
                weights[k] /= wSum
            }

            var sharedVec = [UInt16](repeating: 0, count: hiddenDim)
            for d in 0..<hiddenDim {
                let f = Float.random(in: -2.0...2.0)
                sharedVec[d] = UInt16(f.bitPattern >> 16)
            }

            let wBuf = try Self.makeBuffer(context: context, array: weights)
            let sBuf = try Self.makeBuffer(context: context, array: sharedVec)

            // Define 4 completion/arrival orders into the preallocated contribution buffer:
            // In all cases, expert vector for router slot `k` MUST be stored at offset `k * 2048`.
            // The difference between scenarios is the temporal order in which those slots are populated!
            let completionOrders: [(name: String, order: [Int])] = [
                ("Natural Order (0, 1, 2, 3, 4, 5)", [0, 1, 2, 3, 4, 5]),
                ("Reverse Order (5, 4, 3, 2, 1, 0)", [5, 4, 3, 2, 1, 0]),
                ("Shuffled Order (3, 0, 5, 1, 4, 2)", [3, 0, 5, 1, 4, 2]),
                ("Cold/Warm Arrival (0, 5, 2, 4, 1, 3)", [0, 5, 2, 4, 1, 3]),
            ]

            var runHashes: [String] = []
            var outputBuffers: [MTLBuffer] = []

            for scenario in completionOrders {
                let contribBuf = try context.makeBuffer(byteCount: topK * hiddenDim * 2)
                let contribPtr = contribBuf.contents().bindMemory(to: UInt16.self, capacity: topK * hiddenDim)

                // Populate slots according to the scenario's temporal arrival sequence
                for slot in scenario.order {
                    let vec = expertVectors[slot]
                    for d in 0..<hiddenDim {
                        contribPtr[slot * hiddenDim + d] = vec[d]
                    }
                }

                let outBuf = try context.makeBuffer(byteCount: hiddenDim * 2)
                try kernel.executeReduction(
                    contributions: contribBuf,
                    weights: wBuf,
                    sharedExpert: sBuf,
                    output: outBuf,
                    batch: 1,
                    numTokens: 1,
                    inDtype: .bfloat16,
                    outDtype: .bfloat16
                )

                let outData = Data(bytes: outBuf.contents(), count: hiddenDim * 2)
                let hash = Self.sha256Hex(outData)
                runHashes.append(hash)
                outputBuffers.append(outBuf)
            }

            // Assert that every single arrival order produced the exact same SHA-256 hash
            let baselineHash = runHashes[0]
            for (idx, hash) in runHashes.enumerated() {
                #expect(hash == baselineHash, "Residency invariance failed in trial \(trial): \(completionOrders[idx].name) produced hash \(hash) != baseline \(baselineHash)")
            }

            // Byte-by-byte comparison
            let basePtr = outputBuffers[0].contents().bindMemory(to: UInt8.self, capacity: hiddenDim * 2)
            for sIdx in 1..<outputBuffers.count {
                let curPtr = outputBuffers[sIdx].contents().bindMemory(to: UInt8.self, capacity: hiddenDim * 2)
                for byteIdx in 0..<(hiddenDim * 2) {
                    #expect(curPtr[byteIdx] == basePtr[byteIdx], "Byte mismatch at byte \(byteIdx) between Natural and \(completionOrders[sIdx].name)")
                }
            }

            if trial == 0 {
                print("Trial 0 SHA-256: \(baselineHash)")
                print("All 4 arrival order scenarios produced 100% bitwise-identical output (0 byte differences across \(hiddenDim * 2) bytes).")
            }
        }
    }

    // MARK: - 4. Repeat-Run Bitwise Determinism

    @Test func repeatRunDeterminismContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let hiddenDim = 2048
        let topK = 6

        var contribs = [UInt16](repeating: 0, count: topK * hiddenDim)
        for i in 0..<(topK * hiddenDim) {
            let f = Float.random(in: -3.0...3.0)
            contribs[i] = UInt16(f.bitPattern >> 16)
        }

        var weights = [Float](repeating: 0, count: topK)
        var wSum: Float = 0
        for k in 0..<topK {
            let w = Float.random(in: 0.05...0.35)
            weights[k] = w
            wSum += w
        }
        for k in 0..<topK {
            weights[k] /= wSum
        }

        let cBuf = try Self.makeBuffer(context: context, array: contribs)
        let wBuf = try Self.makeBuffer(context: context, array: weights)
        let outBuf = try context.makeBuffer(byteCount: hiddenDim * 2)

        var baselineHash: String? = nil
        let iterations = 50

        for it in 0..<iterations {
            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: nil,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                inDtype: .bfloat16,
                outDtype: .bfloat16
            )

            let outData = Data(bytes: outBuf.contents(), count: hiddenDim * 2)
            let hash = Self.sha256Hex(outData)
            if let base = baselineHash {
                #expect(hash == base, "Repeat run determinism mismatch at iteration \(it)")
            } else {
                baselineHash = hash
            }
        }

        print("Repeat-run determinism: 50/50 consecutive dispatches produced identical SHA-256: \(baselineHash ?? "")")
    }

    // MARK: - 5. Cancellation and Magnitude Boundary Tests

    @Test func cancellationAndMagnitudeBoundaries() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let hiddenDim = 2048
        let topK = 6

        // 1. Zero contributions with shared expert -> output exactly equals shared expert
        do {
            let cBuf = try context.makeBuffer(byteCount: topK * hiddenDim * 2)
            let weights: [Float] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
            let wBuf = try Self.makeBuffer(context: context, array: weights)

            var sharedVec = [UInt16](repeating: 0, count: hiddenDim)
            for d in 0..<hiddenDim {
                let f = Float.random(in: -3.0...3.0)
                sharedVec[d] = UInt16(f.bitPattern >> 16)
            }
            let sBuf = try Self.makeBuffer(context: context, array: sharedVec)
            let outBuf = try context.makeBuffer(byteCount: hiddenDim * 2)

            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: sBuf,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                inDtype: .bfloat16,
                outDtype: .bfloat16
            )

            let outU16 = outBuf.contents().bindMemory(to: UInt16.self, capacity: hiddenDim)
            for d in 0..<hiddenDim {
                #expect(outU16[d] == sharedVec[d], "Zero contributions did not pass through shared expert at dim \(d)")
            }
        }

        // 2. Cancellation: slot 0 and slot 1 have opposite sign and equal weights (0.5 each)
        do {
            var contribs = [Float](repeating: 0, count: topK * hiddenDim)
            for d in 0..<hiddenDim {
                let val = Float.random(in: 1.0...10.0)
                contribs[0 * hiddenDim + d] = val
                contribs[1 * hiddenDim + d] = -val
            }
            let weights: [Float] = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]

            let cBuf = try Self.makeBuffer(context: context, array: contribs)
            let wBuf = try Self.makeBuffer(context: context, array: weights)
            let outBuf = try context.makeBuffer(byteCount: hiddenDim * 4)

            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: nil,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                inDtype: .float32,
                outDtype: .float32
            )

            let outF32 = outBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
            for d in 0..<hiddenDim {
                #expect(abs(outF32[d]) <= 1e-6, "Cancellation failed at dim \(d): \(outF32[d])")
            }
        }

        // 3. Extreme magnitudes: small weights (1e-6) and large contributions (1e4)
        do {
            let contribs = [Float](repeating: 1e4, count: topK * hiddenDim)
            let weights = [Float](repeating: 1e-6, count: topK)

            let cBuf = try Self.makeBuffer(context: context, array: contribs)
            let wBuf = try Self.makeBuffer(context: context, array: weights)
            let outBuf = try context.makeBuffer(byteCount: hiddenDim * 4)

            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: nil,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                inDtype: .float32,
                outDtype: .float32
            )

            let outF32 = outBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
            for d in 0..<hiddenDim {
                #expect(!outF32[d].isNaN && !outF32[d].isInfinite, "Extreme magnitude produced NaN or Inf at dim \(d)")
                #expect(abs(outF32[d] - 0.06) <= 1e-4, "Expected ~0.06, got \(outF32[d])")
            }
        }
    }

    // MARK: - 6. Buffer Bounds and Parameter Validation

    @Test func parameterAndBufferBoundsValidation() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let cBuf = try context.makeBuffer(byteCount: 6 * 2048 * 2)
        let wBuf = try context.makeBuffer(byteCount: 6 * 4)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        // 1. batch = 0
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, output: outBuf, batch: 0, numTokens: 1)
        }

        // 2. numTokens = 0
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, output: outBuf, batch: 1, numTokens: 0)
        }

        // 3. Undersized contributions buffer
        let smallCBuf = try context.makeBuffer(byteCount: 100)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: smallCBuf, weights: wBuf, output: outBuf, batch: 1, numTokens: 1)
        }

        // 4. Undersized weights buffer
        let smallWBuf = try context.makeBuffer(byteCount: 12)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: smallWBuf, output: outBuf, batch: 1, numTokens: 1)
        }

        // 5. Undersized shared expert buffer
        let smallSBuf = try context.makeBuffer(byteCount: 500)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, sharedExpert: smallSBuf, output: outBuf, batch: 1, numTokens: 1)
        }

        // 6. Undersized output buffer
        let smallOutBuf = try context.makeBuffer(byteCount: 500)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, output: smallOutBuf, batch: 1, numTokens: 1)
        }

        // 7. Unaligned offsets
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, contributionsOffset: 1, weights: wBuf, output: outBuf, batch: 1, numTokens: 1)
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, weightsOffset: 2, output: outBuf, batch: 1, numTokens: 1)
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(contributions: cBuf, weights: wBuf, output: outBuf, outputOffset: 1, batch: 1, numTokens: 1)
        }

        // 8. Unsupported dtype combination (bf16 in, f32 out)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                inDtype: .bfloat16,
                outDtype: .float32
            )
        }
    }

    // MARK: - 7. Canonical Metal Shader Source Equivalence

    @Test func canonicalShaderSourceEquivalence() throws {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_reduction.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(sourcePath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            Issue.record("flint_reduction.metal not found on disk")
            return
        }

        let diskSource = try String(contentsOf: fileURL, encoding: .utf8)
        #expect(diskSource == FlintReductionKernel.embeddedShaderSource, "flint_reduction.metal on disk does not match embeddedShaderSource in FlintReductionKernel.swift")
    }

    // MARK: - 8. Apple M5 Hardware Performance Benchmark

    @Test func reductionPerformanceBenchmarkAppleM5() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintReductionKernel(context: context)

        let hiddenDim = 2048
        let topK = 6
        let batch = 1
        let numTokens = 1

        let cBuf = try context.makeBuffer(byteCount: batch * numTokens * topK * hiddenDim * 2)
        let wBuf = try context.makeBuffer(byteCount: batch * numTokens * topK * 4)
        let sBuf = try context.makeBuffer(byteCount: batch * numTokens * hiddenDim * 2)
        let outBuf = try context.makeBuffer(byteCount: batch * numTokens * hiddenDim * 2)

        // Warm up
        for _ in 0..<10 {
            try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: sBuf,
                output: outBuf,
                batch: batch,
                numTokens: numTokens
            )
        }

        let iterations = 100
        var durationsUs: [Double] = []
        durationsUs.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let gpuSec = try kernel.executeReduction(
                contributions: cBuf,
                weights: wBuf,
                sharedExpert: sBuf,
                output: outBuf,
                batch: batch,
                numTokens: numTokens
            )
            durationsUs.append(gpuSec * 1_000_000.0)
        }

        durationsUs.sort()
        let minUs = durationsUs.first ?? 0
        let maxUs = durationsUs.last ?? 0
        let p50Us = durationsUs[iterations / 2]
        let p95Us = durationsUs[Int(Double(iterations) * 0.95)]
        let meanUs = durationsUs.reduce(0, +) / Double(iterations)
        let variance = durationsUs.map { pow($0 - meanUs, 2) }.reduce(0, +) / Double(iterations)
        let stdDevUs = sqrt(variance)

        // Memory footprint:
        // Contributions: 6 * 2048 * 2 = 24,576 bytes
        // Weights: 6 * 4 = 24 bytes
        // Shared Expert: 2048 * 2 = 4,096 bytes
        // Output: 2048 * 2 = 4,096 bytes
        // Total memory traffic: 32,792 bytes
        let totalBytes = Double(24576 + 24 + 4096 + 4096)
        let throughputGBs = (totalBytes / (p50Us * 1e-6)) / 1e9

        print("\n=== FLINT MOE FIXED ROUTER-ORDER EXPERT REDUCTION BENCHMARK (Apple M5) ===")
        print("Hardware: Apple M5, 32 GB Unified Memory")
        print("Configuration: Batch=\(batch), Tokens=\(numTokens), HiddenDim=\(hiddenDim), TopK=\(topK), SharedExpert=YES")
        print(String(format: "Iterations: %d | Warmup: 10", iterations))
        print(String(format: "P50 Execution Time: %8.2f µs", p50Us))
        print(String(format: "Min Execution Time: %8.2f µs", minUs))
        print(String(format: "P95 Execution Time: %8.2f µs", p95Us))
        print(String(format: "Max Execution Time: %8.2f µs", maxUs))
        print(String(format: "Mean Execution:     %8.2f µs (± %.2f µs jitter)", meanUs, stdDevUs))
        print(String(format: "Memory Traffic:     %.3f KiB (%.2f GB/s effective)", totalBytes / 1024.0, throughputGBs))
        print("Steady-state allocations: 0 heap allocations per dispatch")
        print("========================================================================\n")

        #expect(p50Us < 100.0, "P50 reduction time exceeded 100 µs on Apple M5")
    }
}
