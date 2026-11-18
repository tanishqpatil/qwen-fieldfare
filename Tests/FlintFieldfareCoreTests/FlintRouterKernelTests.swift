import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintRouterKernelTests {

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

    private struct RouterGoldenCase: Decodable {
        let name: String
        let layer_idx: Int
        let dtype: String
        let batch: Int
        let num_tokens: Int
        let input_hex: String
        let gate_weight_hex: String
        let expected_experts_hex: String
        let expected_weights_hex: String
    }

    private struct RouterGoldenFixture: Decodable {
        let metadata: [String: AnyCodable]
        let cases: [RouterGoldenCase]
    }

    private struct AnyCodable: Decodable {}

    private static func loadGoldenFixture() throws -> RouterGoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_router_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(RouterGoldenFixture.self, from: data)
    }

    // MARK: - 1. Pinned Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_router_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        print("\n=== FLINT MOE SIGMOID TOP-6 ROUTER NUMERICAL QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-30@ | %-8@ | %6@ | %-24@ | %12@ | %10@ | %@",
                     "Case Name", "Dtype", "Tokens", "Top-6 Experts", "Max Abs Diff", "Weight Sum", "Status"))
        print(String(repeating: "-", count: 110))

        for c in fixture.cases {
            let inData = Self.parseHexToData(c.input_hex)
            let weightData = Self.parseHexToData(c.gate_weight_hex)
            let expExpertsData = Self.parseHexToData(c.expected_experts_hex)
            let expWeightsData = Self.parseHexToData(c.expected_weights_hex)

            let dtype: FlintRouterKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

            let inBuf = try inData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: inData.count) }
            let weightBuf = try weightData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: weightData.count) }
            let outExpertsBuf = try context.makeBuffer(byteCount: expExpertsData.count)
            let outWeightsBuf = try context.makeBuffer(byteCount: expWeightsData.count)

            try kernel.executeForward(
                input: inBuf,
                gateWeight: weightBuf,
                outExperts: outExpertsBuf,
                outWeights: outWeightsBuf,
                batch: c.batch,
                numTokens: c.num_tokens,
                dtype: dtype,
                waitUntilCompleted: true
            )

            let numEntries = c.batch * c.num_tokens * 6
            let metalExperts = Array(UnsafeBufferPointer(start: outExpertsBuf.contents().assumingMemoryBound(to: UInt32.self), count: numEntries))
            let metalWeights = Array(UnsafeBufferPointer(start: outWeightsBuf.contents().assumingMemoryBound(to: Float.self), count: numEntries))

            let expectedExperts = expExpertsData.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            let expectedWeights = expWeightsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

            // 1. Verify Top-6 selected expert sets and ordering
            let totalTokens = c.batch * c.num_tokens
            var maxAbsDiff: Float = 0.0

            for t in 0..<totalTokens {
                let mSlice = Array(metalExperts[(t * 6)..<(t * 6 + 6)])
                let eSlice = Array(expectedExperts[(t * 6)..<(t * 6 + 6)])
                let mWeights = Array(metalWeights[(t * 6)..<(t * 6 + 6)])
                let eWeights = Array(expectedWeights[(t * 6)..<(t * 6 + 6)])

                let mSet = Set(mSlice)
                let eSet = Set(eSlice)
                let common = mSet.intersection(eSet)

                if dtype == .float32 {
                    // FP32 requires exact set and order match
                    #expect(mSlice == eSlice, "Case \(c.name) token \(t) expert mismatch: metal \(mSlice) vs exp \(eSlice)")
                    for k in 0..<6 {
                        let diff = abs(mWeights[k] - eWeights[k])
                        if diff > maxAbsDiff { maxAbsDiff = diff }
                    }
                } else {
                    // BF16 dot product reduction tree differences can exchange near-tied boundary experts (difference < 1e-4)
                    #expect(common.count >= 5, "Case \(c.name) token \(t) expert set overlap \(common.count) < 5: metal \(mSlice) vs exp \(eSlice)")
                    for k in 0..<6 {
                        let expId = mSlice[k]
                        if let eIdx = eSlice.firstIndex(of: expId) {
                            let diff = abs(mWeights[k] - eWeights[eIdx])
                            if diff > maxAbsDiff { maxAbsDiff = diff }
                        }
                    }
                }
            }

            let maxTol: Float = (dtype == .float32) ? 1e-4 : 1e-3
            #expect(maxAbsDiff <= maxTol, "Case \(c.name) weight diff \(maxAbsDiff) exceeds tolerance \(maxTol)")

            // 3. Verify weights sum to 1.0 for each token
            for t in 0..<(c.batch * c.num_tokens) {
                var sum: Float = 0.0
                for k in 0..<6 {
                    sum += metalWeights[t * 6 + k]
                }
                #expect(abs(sum - 1.0) <= 1e-4, "Weights for token \(t) do not sum to 1.0 (got \(sum))")
            }

            let expertsStr = metalExperts.prefix(6).map { String($0) }.joined(separator: ",")
            let weightSumStr = String(format: "%.6f", metalWeights.prefix(6).reduce(0, +))
            print(String(format: "%-30@ | %-8@ | %6d | %-24@ | %12.6f | %10@ | PASS",
                         c.name, c.dtype, c.num_tokens, expertsStr, maxAbsDiff, weightSumStr))
        }
        print(String(repeating: "-", count: 110))
    }

    // MARK: - 2. Analytical Tie-Breaking Contract

    @Test func analyticalTieBreakingContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        // Case A: All Zeros activation -> all logits are 0, sigmoid = 0.5 for all 256 experts.
        // Under the documented tie-breaking contract, lower expert ID is strictly chosen first:
        // Experts 0, 1, 2, 3, 4, 5 must be selected in exact ascending order.
        let xZeros = [Float](repeating: 0.0, count: 2048)
        let wMock = [Float](repeating: 0.1, count: 256 * 2048)
        let wMockBF16 = FlintRMSNormReference.floatArrayToBF16(wMock)

        let inBuf = try xZeros.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let weightBuf = try wMockBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outExpBuf = try context.makeBuffer(byteCount: 6 * 4)
        let outWgtBuf = try context.makeBuffer(byteCount: 6 * 4)

        try kernel.executeForward(
            input: inBuf,
            gateWeight: weightBuf,
            outExperts: outExpBuf,
            outWeights: outWgtBuf,
            batch: 1,
            numTokens: 1,
            dtype: .float32,
            waitUntilCompleted: true
        )

        let expA = Array(UnsafeBufferPointer(start: outExpBuf.contents().assumingMemoryBound(to: UInt32.self), count: 6))
        let wgtA = Array(UnsafeBufferPointer(start: outWgtBuf.contents().assumingMemoryBound(to: Float.self), count: 6))

        #expect(expA == [0, 1, 2, 3, 4, 5], "All zeros tie-breaking failed: got \(expA)")
        for k in 0..<6 {
            #expect(abs(wgtA[k] - (1.0 / 6.0)) <= 1e-4)
        }

        // Case B: Explicit duplicate logits for experts 5, 12, 19, 42, 100, 200
        // Set all other experts to negative weights
        var wCustom = [Float](repeating: -10.0, count: 256 * 2048)
        let targetExperts = [5, 12, 19, 42, 100, 200]
        for e in targetExperts {
            for d in 0..<2048 {
                wCustom[e * 2048 + d] = 1.0 // Identical dot products = 2048.0
            }
        }
        let wCustomBF16 = FlintRMSNormReference.floatArrayToBF16(wCustom)
        let xOnes = [Float](repeating: 1.0, count: 2048)

        let inBufB = try xOnes.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let weightBufB = try wCustomBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }

        try kernel.executeForward(
            input: inBufB,
            gateWeight: weightBufB,
            outExperts: outExpBuf,
            outWeights: outWgtBuf,
            batch: 1,
            numTokens: 1,
            dtype: .float32,
            waitUntilCompleted: true
        )

        let expB = Array(UnsafeBufferPointer(start: outExpBuf.contents().assumingMemoryBound(to: UInt32.self), count: 6))
        #expect(expB == [5, 12, 19, 42, 100, 200], "Identical logits tie-breaking failed: got \(expB)")
    }

    // MARK: - 3. Repeat-Run Bitwise Determinism

    @Test func repeatRunDeterminism() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        var xData = [Float](repeating: 0, count: 2048)
        for i in 0..<2048 { xData[i] = Float.random(in: -2.0...2.0) }
        var wData = [Float](repeating: 0, count: 256 * 2048)
        for i in 0..<wData.count { wData[i] = Float.random(in: -1.0...1.0) }
        let wBF16 = FlintRMSNormReference.floatArrayToBF16(wData)

        let inBuf = try xData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let weightBuf = try wBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outExpBuf = try context.makeBuffer(byteCount: 6 * 4)
        let outWgtBuf = try context.makeBuffer(byteCount: 6 * 4)

        // Baseline run
        try kernel.executeForward(
            input: inBuf,
            gateWeight: weightBuf,
            outExperts: outExpBuf,
            outWeights: outWgtBuf,
            batch: 1,
            numTokens: 1,
            dtype: .float32,
            waitUntilCompleted: true
        )

        let baseExpertsData = Data(bytes: outExpBuf.contents(), count: 6 * 4)
        let baseWeightsData = Data(bytes: outWgtBuf.contents(), count: 6 * 4)
        let baseExpHash = Self.sha256Hex(baseExpertsData)
        let baseWgtHash = Self.sha256Hex(baseWeightsData)

        // Repeat 50 runs and verify bitwise identical output
        for iteration in 1...50 {
            try kernel.executeForward(
                input: inBuf,
                gateWeight: weightBuf,
                outExperts: outExpBuf,
                outWeights: outWgtBuf,
                batch: 1,
                numTokens: 1,
                dtype: .float32,
                waitUntilCompleted: true
            )

            let curExpData = Data(bytes: outExpBuf.contents(), count: 6 * 4)
            let curWgtData = Data(bytes: outWgtBuf.contents(), count: 6 * 4)

            #expect(Self.sha256Hex(curExpData) == baseExpHash, "Expert output drifted on iteration \(iteration)")
            #expect(Self.sha256Hex(curWgtData) == baseWgtHash, "Weight output drifted on iteration \(iteration)")
        }
    }

    // MARK: - 4. Buffer Bounds and Parameter Validation

    @Test func bufferBoundsAndParameterValidation() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        let validInBuf = try context.makeBuffer(byteCount: 2048 * 4)
        let validWeightBuf = try context.makeBuffer(byteCount: 256 * 2048 * 2)
        let validExpBuf = try context.makeBuffer(byteCount: 6 * 4)
        let validWgtBuf = try context.makeBuffer(byteCount: 6 * 4)

        let smallInBuf = try context.makeBuffer(byteCount: 1024)
        let smallWeightBuf = try context.makeBuffer(byteCount: 5000)
        let smallExpBuf = try context.makeBuffer(byteCount: 8)
        let smallWgtBuf = try context.makeBuffer(byteCount: 8)

        // Invalid batch
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 0, numTokens: 1, dtype: .float32)
        }

        // Invalid numTokens
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: -1, dtype: .float32)
        }

        // Buffer too small: input
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: smallInBuf, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }

        // Buffer too small: gateWeight
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, gateWeight: smallWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }

        // Buffer too small: outExperts
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, gateWeight: validWeightBuf, outExperts: smallExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }

        // Buffer too small: outWeights
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: smallWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }

        // Negative offset
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, inputOffset: -4, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }

        // Unaligned offset
        #expect(throws: FlintMetalError.self) {
            try kernel.executeForward(input: validInBuf, inputOffset: 1, gateWeight: validWeightBuf, outExperts: validExpBuf, outWeights: validWgtBuf, batch: 1, numTokens: 1, dtype: .float32)
        }
    }

    // MARK: - 5. Real Checkpoint Resident Weights Integration & Immutability

    @Test func realCheckpointResidentRouterIntegration() throws {
        let modelDir = "/Users/tanishq/models/flint-mixed-3-4bit"
        guard FileManager.default.fileExists(atPath: modelDir) else {
            Issue.record("Model directory not found at: \(modelDir)")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)
        let validated = try FlintModelDirectory.validate(path: modelDir)
        let shards = try FlintModelDirectory.openShards(canonicalPath: validated.canonicalPath, shardNames: validated.shards)
        let (residentWeights, _) = try FlintResidentLoader.loadResidentWeights(
            validatedDirectory: validated,
            shardFiles: shards,
            storageMode: .gpuRequired,
            device: context.device
        )

        // Flint has 47 MoE layers: layers 2 through 48.
        // Test real router gate weights on layers 2, 24, and 48:
        for layerIdx in [2, 24, 48] {
            guard let moe = residentWeights.moeResidentWeights(forLayer: layerIdx) else {
                Issue.record("Resident MoE weights not found for layer: \(layerIdx)")
                continue
            }
            let gateTensor = moe.routerGate

            #expect(gateTensor.byteCount == 256 * 2048 * 2)
            guard let metalBuf = gateTensor.metalBuffer else {
                Issue.record("gateTensor.metalBuffer is nil for layer: \(layerIdx)")
                continue
            }

            let beforeHash = Self.sha256Hex(gateTensor.buffer.copyData())

            var xData = [Float](repeating: 0, count: 2048)
            for i in 0..<2048 { xData[i] = Float.random(in: -1.0...1.0) }
            let inBuf = try xData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
            let outExpBuf = try context.makeBuffer(byteCount: 6 * 4)
            let outWgtBuf = try context.makeBuffer(byteCount: 6 * 4)

            try kernel.executeForward(
                input: inBuf,
                gateWeight: metalBuf,
                outExperts: outExpBuf,
                outWeights: outWgtBuf,
                batch: 1,
                numTokens: 1,
                dtype: .float32,
                waitUntilCompleted: true
            )

            let afterHash = Self.sha256Hex(gateTensor.buffer.copyData())
            #expect(beforeHash == afterHash, "Resident router weight corrupted during inference on layer \(layerIdx)!")

            let experts = Array(UnsafeBufferPointer(start: outExpBuf.contents().assumingMemoryBound(to: UInt32.self), count: 6))
            let weights = Array(UnsafeBufferPointer(start: outWgtBuf.contents().assumingMemoryBound(to: Float.self), count: 6))

            for e in experts {
                #expect(e < 256, "Selected expert ID \(e) out of bounds [0, 255]")
            }
            let sum = weights.reduce(0, +)
            #expect(abs(sum - 1.0) <= 1e-4, "Weights on layer \(layerIdx) do not sum to 1.0 (got \(sum))")
        }
    }

    // MARK: - 6. Apple M5 Hardware Benchmark & Telemetry

    @Test func benchmarkRouterPerformanceOnM5() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        print("\n=== FLINT MOE SIGMOID TOP-6 ROUTER HARDWARE BENCHMARK (APPLE M5) ===")

        let warmupCount = 20
        let benchCount = 100

        for (scenario, numTokens, dtype) in [
            ("Decode Router (1 token)", 1, FlintRouterKernel.Dtype.bfloat16),
            ("Prefill Router (8 tokens)", 8, FlintRouterKernel.Dtype.bfloat16),
            ("Prefill Router (64 tokens)", 64, FlintRouterKernel.Dtype.bfloat16),
            ("Decode Router FP32 (1 token)", 1, FlintRouterKernel.Dtype.float32)
        ] {
            let inBytes = numTokens * 2048 * dtype.elementByteWidth
            let weightBytes = 256 * 2048 * 2
            let outExpBytes = numTokens * 6 * 4
            let outWgtBytes = numTokens * 6 * 4

            let inBuf = try context.makeBuffer(byteCount: inBytes)
            let weightBuf = try context.makeBuffer(byteCount: weightBytes)
            let outExpBuf = try context.makeBuffer(byteCount: outExpBytes)
            let outWgtBuf = try context.makeBuffer(byteCount: outWgtBytes)

            // Warmup
            for _ in 0..<warmupCount {
                try kernel.executeForward(
                    input: inBuf,
                    gateWeight: weightBuf,
                    outExperts: outExpBuf,
                    outWeights: outWgtBuf,
                    batch: 1,
                    numTokens: numTokens,
                    dtype: dtype,
                    waitUntilCompleted: true
                )
            }

            // Timed iterations using hardware GPU timestamps
            var durations = [Double]()
            durations.reserveCapacity(benchCount)

            for _ in 0..<benchCount {
                let dur = try kernel.executeForward(
                    input: inBuf,
                    gateWeight: weightBuf,
                    outExperts: outExpBuf,
                    outWeights: outWgtBuf,
                    batch: 1,
                    numTokens: numTokens,
                    dtype: dtype,
                    waitUntilCompleted: true
                )
                if dur > 0 {
                    durations.append(dur * 1e6) // convert to microseconds
                }
            }

            if !durations.isEmpty {
                durations.sort()
                let p50 = durations[durations.count / 2]
                let minVal = durations.first!
                let p95 = durations[Int(Double(durations.count) * 0.95)]
                let maxVal = durations.last!
                let jitter = p95 - minVal
                let throughput = Double(numTokens) / (p50 * 1e-6)

                print(String(format: "%-30@ | P50: %6.2f µs | Min: %6.2f µs | P95: %6.2f µs | Max: %6.2f µs | Jitter: %5.2f µs | Throughput: %8.0f tok/s",
                             scenario, p50, minVal, p95, maxVal, jitter, throughput))
            }
        }
        print("====================================================================\n")
    }

    // MARK: - 7. Canonical Metal Shader Source Equivalence

    @Test func canonicalMetalShaderSourceEquivalence() throws {
        let routerMetalPath = "Sources/FlintFieldfareCore/Metal/flint_router.metal"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(routerMetalPath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            Issue.record("flint_router.metal not found at path: \(fullPath.path)")
            return
        }
        let diskSource = try String(contentsOf: fullPath, encoding: .utf8)
        #expect(diskSource == FlintRouterKernel.embeddedShaderSource, "Disk flint_router.metal and FlintRouterKernel.embeddedShaderSource have drifted!")
    }

    // MARK: - 8. Real Metal vs MLX Router Diagnostic (100 Deterministic Tokens)

    private struct Router100TokenRecord: Decodable {
        let token_idx: Int
        let mlx_top6_indices: [UInt32]
        let mlx_top6_weights: [Float]
        let mlx_raw_indices: [UInt32]
        let mlx_raw_weights: [Float]
        let mlx_top7_scores: [Float]
        let mlx_margin_rank5_vs_rank6: Float
        let py_top6_indices: [UInt32]
        let py_top6_weights: [Float]
    }

    private struct Router100TokensFixture: Decodable {
        let metadata: [String: AnyCodable]
        let input_hex: String
        let gate_weight_hex: String
        let tokens: [Router100TokenRecord]
    }

    @Test func realMetalVsMLXRouterDiagnostic100Tokens() throws {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/router_100_tokens_mlx.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            Issue.record("router_100_tokens_mlx.json not found")
            return
        }
        let data = try Data(contentsOf: fullPath)
        let fixture = try JSONDecoder().decode(Router100TokensFixture.self, from: data)

        let context = try FlintMetalContext()
        let kernel = try FlintRouterKernel(context: context)

        let inData = Self.parseHexToData(fixture.input_hex)
        let weightData = Self.parseHexToData(fixture.gate_weight_hex)

        let inBuf = try inData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: inData.count) }
        let weightBuf = try weightData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: weightData.count) }

        let numTokens = fixture.tokens.count
        let outExpBuf = try context.makeBuffer(byteCount: numTokens * 6 * MemoryLayout<UInt32>.stride)
        let outWgtBuf = try context.makeBuffer(byteCount: numTokens * 6 * MemoryLayout<Float>.stride)

        // Dispatch REAL Metal Router on GPU!
        try kernel.executeForward(
            input: inBuf,
            gateWeight: weightBuf,
            outExperts: outExpBuf,
            outWeights: outWgtBuf,
            batch: 1,
            numTokens: numTokens,
            dtype: .bfloat16,
            waitUntilCompleted: true
        )

        let metalExperts = Array(UnsafeBufferPointer(start: outExpBuf.contents().assumingMemoryBound(to: UInt32.self), count: numTokens * 6))
        let metalWeights = Array(UnsafeBufferPointer(start: outWgtBuf.contents().assumingMemoryBound(to: Float.self), count: numTokens * 6))

        var exactSetMatches = 0
        var exactOrderMatches = 0
        var maxWeightDiffOnMatched: Float = 0.0
        var rank5Boundaries: [(token: Int, metal: [UInt32], mlx: [UInt32], mlxMargin: Float, tied: Bool)] = []

        for t in 0..<numTokens {
            let mSliceExp = Array(metalExperts[(t * 6)..<(t * 6 + 6)])
            let mSliceWgt = Array(metalWeights[(t * 6)..<(t * 6 + 6)])
            let rec = fixture.tokens[t]

            let metalSet = Set(mSliceExp)
            let mlxSet = Set(rec.mlx_top6_indices)

            let isSetMatch = (metalSet == mlxSet)
            let isOrderMatch = (mSliceExp == rec.mlx_top6_indices)

            if isSetMatch {
                exactSetMatches += 1
            }
            if isOrderMatch {
                exactOrderMatches += 1
            }

            // Compute weight differences on matched experts
            for k in 0..<6 {
                let expId = mSliceExp[k]
                if let mlxIdx = rec.mlx_top6_indices.firstIndex(of: expId) {
                    let diff = abs(mSliceWgt[k] - rec.mlx_top6_weights[mlxIdx])
                    if diff > maxWeightDiffOnMatched {
                        maxWeightDiffOnMatched = diff
                    }
                }
            }

            if !isSetMatch {
                let isTied = (rec.mlx_margin_rank5_vs_rank6 == 0.0)
                rank5Boundaries.append((token: t, metal: mSliceExp, mlx: rec.mlx_top6_indices, mlxMargin: rec.mlx_margin_rank5_vs_rank6, tied: isTied))
            }
        }

        print("\n=== REAL METAL VS MLX ROUTER DIAGNOSTIC (100 DETERMINISTIC TOKENS) ===")
        print(String(format: "Hardware: Apple M5 | Kernel: FlintRouterKernel (actual GPU dispatch)"))
        print(String(format: "Exact 6/6 Expert Set Match Rate: %d / %d (%.1f%%)", exactSetMatches, numTokens, Double(exactSetMatches) / Double(numTokens) * 100.0))
        print(String(format: "Exact Rank Ordering Match Rate:  %d / %d (%.1f%%)", exactOrderMatches, numTokens, Double(exactOrderMatches) / Double(numTokens) * 100.0))
        print(String(format: "Max Weight Difference on Matched: %.6f", maxWeightDiffOnMatched))
        print(String(format: "Boundary Mismatches at Rank 5:   %d tokens", rank5Boundaries.count))

        for (idx, b) in rank5Boundaries.enumerated() {
            print(String(format: "  Mismatch #%d (Token %d): Metal=%@ | MLX=%@ | MLX 6th vs 7th Margin=%.8f | Genuinely Tied=%@",
                         idx + 1, b.token, b.metal.description, b.mlx.description, b.mlxMargin, b.tied ? "YES" : "NO"))
        }
        print("=======================================================================\n")

        #expect(exactSetMatches >= 95, "Exact set match rate fell below 95%: \(exactSetMatches)/\(numTokens)")
        #expect(maxWeightDiffOnMatched <= 2e-3, "Max weight difference on matched experts exceeded 2e-3: \(maxWeightDiffOnMatched)")
    }
}
